"""Read a local container's cgroup v2 counters from the host, without exec."""

from __future__ import annotations

import errno
import subprocess
import time
from pathlib import Path


def read_counters(directory: Path, *, cpu: bool = True, ram: bool = True) -> dict:
    result = {"sampled_wall_ns": time.time_ns(), "sampled_monotonic_ns": time.monotonic_ns()}
    if cpu:
        counters = dict(line.split() for line in (directory / "cpu.stat").read_text().splitlines())
        quota, period = (directory / "cpu.max").read_text().split()
        result.update(
            cpu_usage_usec=int(counters["usage_usec"]),
            cpu_user_usec=int(counters["user_usec"]),
            cpu_system_usec=int(counters["system_usec"]),
            cpu_throttled_usec=int(counters.get("throttled_usec", 0)),
            cpu_limit_cores=None if quota == "max" else int(quota) / int(period),
        )
    if ram:
        memory = dict(line.split() for line in (directory / "memory.stat").read_text().splitlines())
        limit = (directory / "memory.max").read_text().strip()
        try:
            peak = int((directory / "memory.peak").read_text())
        except FileNotFoundError:
            peak = None
        result.update(
            memory_current_bytes=int((directory / "memory.current").read_text()),
            memory_peak_bytes=peak,
            memory_limit_bytes=None if limit == "max" else int(limit),
            memory_anon_bytes=int(memory.get("anon", 0)),
            memory_file_bytes=int(memory.get("file", 0)),
        )
    return result


def cpu_rate(current: dict, previous: dict | None) -> dict:
    """CPU is an interval rate: 100% means one fully occupied CPU core."""
    result = {"cpu_percent": None, "cpu_interval_s": None, "cpu_interval_started_monotonic_ns": None}
    if previous is None or current["container_id"] != previous["container_id"]:
        return result
    elapsed = current["sampled_monotonic_ns"] - previous["sampled_monotonic_ns"]
    usage = current["cpu_usage_usec"] - previous["cpu_usage_usec"]
    if elapsed > 0 and usage >= 0:
        result.update(
            cpu_percent=100 * usage * 1000 / elapsed,
            cpu_interval_s=elapsed / 1e9,
            cpu_interval_started_monotonic_ns=previous["sampled_monotonic_ns"],
        )
    return result


class ContainerResources:
    """Resolve each ephemeral container once, then read its kernel counters.

    The harness reuses its container name for preflight, model, and postcheck.
    Each replacement has a new ID and CPU baseline. No host-process fallback:
    unsupported or inaccessible accounting is reported as missing telemetry.
    """

    def __init__(self, engine: str, container_name: str, *, cpu: bool = True, ram: bool = True):
        self.cpu, self.ram = cpu, ram
        self.engine, self.container_name = engine, container_name
        self.directory = None
        self.container_id = None
        self.previous = None

    def sample(self) -> dict:
        started = time.monotonic_ns()
        if self.directory is None:
            try:
                inspected = subprocess.run(
                    [self.engine, "inspect", "--format", "{{.Id}} {{.State.Pid}}", self.container_name],
                    capture_output=True, text=True, timeout=3, check=False,
                )
            except subprocess.TimeoutExpired:
                # Busy runtimes can miss a polling deadline while the build is
                # healthy. Record the gap and resolve the container next poll.
                return {"status": "retrying", "reason": "container_inspect_timeout"}
            if inspected.returncode:
                detail = inspected.stderr.strip()
                if any(text in detail.lower() for text in ("no such", "does not exist", "not found")):
                    return {"status": "not_running"}
                raise RuntimeError(f"Cannot inspect resource counters: {detail}")
            identity, pid = inspected.stdout.strip().split()
            if int(pid) <= 0:
                return {"status": "not_running"}
            try:
                groups = Path(f"/proc/{int(pid)}/cgroup").read_text().splitlines()
            except FileNotFoundError:
                return {"status": "not_running"}
            relative = next((line[3:] for line in groups if line.startswith("0::")), None)
            if not relative or relative == "/" or ".." in Path(relative).parts or identity not in relative:
                raise RuntimeError("Container-specific local cgroup v2 accounting is unavailable")
            directory = Path("/sys/fs/cgroup") / relative.lstrip("/")
            # A disabled container cgroup can point at the caller's own group.
            if Path("/proc/self/cgroup").read_text().splitlines() == groups:
                raise RuntimeError("Container shares host cgroup; refusing to report host usage as container usage")
            self.directory, self.container_id = directory, identity
            self.previous = None
        try:
            result = (read_counters(self.directory) if self.cpu and self.ram
                      else read_counters(self.directory, cpu=self.cpu, ram=self.ram))
        except OSError as exc:
            # An offlined cgroup can return ENODEV during teardown, even
            # before its directory disappears. Re-resolve the next container.
            if exc.errno != errno.ENODEV and not (
                exc.errno == errno.ENOENT and not self.directory.exists()
            ):
                raise  # Missing accounting on a live group is unsupported.
            self.directory, self.container_id, self.previous = None, None, None
            return {"status": "not_running"}
        result.update(status="sample", container_id=self.container_id, cgroup=str(self.directory))
        if self.cpu:
            result.update(cpu_rate(result, self.previous))
        result["collection_duration_s"] = (time.monotonic_ns() - started) / 1e9
        self.previous = result
        return result
