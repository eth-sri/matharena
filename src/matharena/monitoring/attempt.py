"""Host-owned source checkpoints and execution traces for one harness attempt.

Checkpoints are observations, not atomic filesystem snapshots. Stable file
reads and capture errors are recorded; writes reverted between observations
cannot be recovered. Command clocks come from the container wrappers, whereas
harness timestamps are explicitly labeled as host observations.
"""

from __future__ import annotations

import difflib
import gzip
import hashlib
import io
import json
import os
import shutil
import socket
import stat
import tarfile
import tempfile
import threading
import time
import uuid
from pathlib import Path

from matharena.monitoring.actions import action_event, summarize_actions
from matharena.monitoring.host_commands import HostCommands
from matharena.monitoring.source_files import is_build_artifact


def write_json(path: Path, value: dict) -> None:
    """Publish complete JSON documents atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


EXCLUDED = {".lake", ".git", ".lean-strip", "__pycache__"}


def read_tree(root: Path, *, native_traces: bool = False, exclude_scratch: bool = False) -> dict:
    """Use directory descriptors and no-follow reads, even during file replacement."""
    result = {}
    excluded = EXCLUDED | {".scratch"} if exclude_scratch else EXCLUDED

    def fail(error):
        raise error

    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for directory, dirs, names, directory_fd in os.fwalk(".", dir_fd=root_fd, follow_symlinks=False, onerror=fail):
            dirs[:] = sorted(
                name for name in dirs if name not in excluded and not (native_traces and name == "repository")
                and not (not native_traces and name == ".harness-home")
            )
            names += [
                name for name in dirs if stat.S_ISLNK(os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode)
            ]
            for name in sorted(names):
                if not native_traces and is_build_artifact(name):
                    continue
                if exclude_scratch and name == ".scratch":
                    continue
                if native_traces and not name.endswith(".jsonl"):
                    continue
                relative = (Path(directory) / name).as_posix()
                info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                if stat.S_ISLNK(info.st_mode):
                    if not native_traces:
                        result[relative] = ("symlink", os.readlink(name, dir_fd=directory_fd), 0o777)
                    continue
                if not stat.S_ISREG(info.st_mode):
                    continue
                for attempt in range(4):
                    fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
                    with os.fdopen(fd, "rb") as stream:
                        before = os.fstat(stream.fileno())
                        if not stat.S_ISREG(before.st_mode):
                            raise ValueError(f"File changed type during capture: {relative}")
                        data = stream.read()
                        after = os.fstat(stream.fileno())
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
                        after.st_size,
                        after.st_mtime_ns,
                        after.st_ctime_ns,
                    ):
                        result[relative] = ("file", data, stat.S_IMODE(after.st_mode))
                        break
                else:
                    raise RuntimeError(f"File kept changing during capture: {relative}")
    finally:
        os.close(root_fd)
    return result


def archive_bytes(state: dict) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, (kind, value, mode) in sorted(state.items()):
            info = tarfile.TarInfo(name)
            info.mode = mode
            if kind == "symlink":
                info.type, info.linkname = tarfile.SYMTYPE, value
                archive.addfile(info)
            else:
                info.size = len(value)
                archive.addfile(info, io.BytesIO(value))
    return buffer.getvalue()


def source_diff(before: dict, after: dict) -> str:
    result = []
    for name in sorted(before.keys() | after.keys()):
        old, new = before.get(name), after.get(name)
        if old == new:
            continue
        result.append(f"diff --leanlean {json.dumps(name)}\n")
        result.append(f"metadata {repr(old[::2]) if old else None} -> {repr(new[::2]) if new else None}\n")

        def lines(entry):
            if entry is None:
                return []
            data = entry[1] if entry[0] == "file" else entry[1].encode()
            return data.decode("utf-8").splitlines(keepends=True)

        try:
            for line in difflib.unified_diff(lines(old), lines(new), fromfile=f"a/{name}", tofile=f"b/{name}"):
                result.append(line if line.endswith("\n") else line + "\n\\ No newline at end of file\n")
        except UnicodeDecodeError:
            result.append("Binary content changed; exact bytes are in the source archive.\n")
        result.append("\n")
    return "".join(result)


def lake_subcommand(arguments: list[str]) -> str | None:
    values = {"--dir", "-d", "--file", "-f", "-K", "--packages", "--log-level", "--fail-level"}
    args = iter(arguments)
    for arg in args:
        if arg in {"--help", "-h", "--version"}:
            return None
        if arg in values:
            next(args, None)
        elif not arg.startswith(("-", "+")):
            return arg
    return None


def summarize_commands(events: list[dict]) -> dict:
    starts, ends, duplicates = {}, {}, []
    for event in events:
        target = starts if event["event"] == "start" else ends
        if event["span_id"] in target:
            duplicates.append(event["span_id"])
        target[event["span_id"]] = event
    spans, intervals = [], {}
    for key in sorted(starts.keys() | ends.keys()):
        start, end = starts.get(key), ends.get(key)
        row = {"span_id": key, "start": start, "end": end, "status": "complete"}
        if start is None or end is None:
            row["status"] = "missing_start" if start is None else "missing_end"
        elif end["monotonic_ns"] < start["monotonic_ns"]:
            row["status"] = "invalid_clock_order"
        else:
            execution_start = end.get("execution_started_monotonic_ns") or start["monotonic_ns"]
            row["execution_started_wall_ns"] = end.get("execution_started_wall_ns") or start.get("wall_time_ns")
            row["duration_s"] = (end["monotonic_ns"] - execution_start) / 1e9
            row["start_collection_s"] = (execution_start - start["monotonic_ns"]) / 1e9
            subcommand = lake_subcommand(start["argv"][1:]) if start["kind"] == "lake" else None
            row["lake_subcommand"] = subcommand
            kind = "lake_build" if subcommand == "build" else start["kind"]
            intervals.setdefault((start["phase"], kind), []).append((execution_start, end["monotonic_ns"]))
        spans.append(row)
    totals = {}
    for (phase, kind), values in intervals.items():
        merged = []
        for start, end in sorted(values):
            if merged and start <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], end)
            else:
                merged.append([start, end])
        totals.setdefault(phase, {})[kind] = {
            "commands": len(values),
            "process_seconds": sum(end - start for start, end in values) / 1e9,
            "busy_wall_seconds": sum(end - start for start, end in merged) / 1e9,
        }
    return {
        "spans": spans,
        "totals": totals,
        "duplicate_span_events": duplicates,
        "complete": not duplicates and all(row["status"] == "complete" for row in spans),
    }


class AttemptMonitor:
    def __init__(
        self, workspace: Path, *, interval_s: float = 30.0, metadata: dict | None = None,
        resources=None, resource_interval_s: float = 1.0,
        source_root: Path | None = None, diff: bool = True, commands: bool = True,
        cpu: bool = True, ram: bool = True, require_lake_build: bool = False,
        command_backend: str = "host",
    ):
        if interval_s <= 0 or resource_interval_s <= 0:
            raise ValueError("Monitoring intervals must be positive")
        self.workspace = workspace.resolve()
        self.source_root = source_root if source_root is not None else self.workspace
        self.diff_enabled, self.commands_enabled = diff, commands
        self.cpu_enabled, self.ram_enabled = cpu, ram
        self.require_lake_build = require_lake_build
        if command_backend not in {"host", "wrapper"}:
            raise ValueError("command_backend must be host or wrapper")
        self.command_backend = command_backend
        self.host_commands = HostCommands(self.append) if commands and command_backend == "host" else None
        self.output = self.workspace.parent / (self.workspace.name + ".telemetry") / uuid.uuid4().hex
        self.output.mkdir(parents=True)
        if self.diff_enabled:
            (self.output / "sources").mkdir()
        self.interval_s = interval_s
        self.phase = "preflight"
        self.metadata = metadata or {}
        self.commands, self.errors, self.warnings = [], [], []
        self.actions, self.resource_samples = [], []
        self.resources, self.resource_interval_s = resources, resource_interval_s
        self._session_id = None
        self.observation_count, self.snapshot_count = 0, 0
        self.summary = None
        self._state, self._digest = {}, None
        self._lock, self._capture_lock = threading.Lock(), threading.Lock()
        self._stop = threading.Event()
        # Keep AF_UNIX paths short even when the experiment's output path is long.
        self._socket_directory = None
        self.socket_path = None
        self._server = None
        if self.commands_enabled and self.command_backend == "wrapper":
            self._socket_directory = Path(tempfile.mkdtemp(prefix="matharena-trace-"))
            self.socket_path = self._socket_directory / "events.sock"
            self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self._server.bind(str(self.socket_path))
            self._server.listen(128)
            self._server.settimeout(0.2)
        self._threads = []

    def append(self, name: str, value: dict) -> None:
        with self._lock, (self.output / name).open("a") as stream:
            stream.write(json.dumps(value, ensure_ascii=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def capture(self, reason: str, *, tool_call_id: str | None = None, span_id: str | None = None) -> None:
        if not self.diff_enabled:
            return
        with self._capture_lock:
            record = {
                "reason": reason,
                "phase": self.phase,
                "tool_call_id": tool_call_id,
                "span_id": span_id,
                "started_wall_ns": time.time_ns(),
                "started_monotonic_ns": time.monotonic_ns(),
            }
            try:
                state = read_tree(self.source_root, exclude_scratch=True)
                changed = self._digest is None or state != self._state
                data = archive_bytes(state) if changed else None
                digest = hashlib.sha256(data).hexdigest() if changed else self._digest
                record.update(source_sha256=digest, parent_sha256=self._digest, changed=digest != self._digest)
                if digest != self._digest:
                    archive = self.output / "sources" / f"{digest}.tar.gz"
                    if not archive.exists():
                        archive.write_bytes(gzip.compress(data, compresslevel=1, mtime=0))
                    diff = f"{self._digest or 'empty'}--{digest}.diff"
                    (self.output / "sources" / diff).write_text(source_diff(self._state, state))
                    record.update(archive=f"sources/{digest}.tar.gz", diff=f"sources/{diff}")
                    self._state, self._digest = state, digest
            except (OSError, ValueError, RuntimeError) as exc:
                record["error"] = repr(exc)
                self.errors.append({"source_capture": repr(exc)})
            record["finished_monotonic_ns"] = time.monotonic_ns()
            self.observation_count += 1
            record["observation_index"] = self.observation_count
            if record.get("changed") and "error" not in record:
                self.snapshot_count += 1
                record["snapshot_index"] = self.snapshot_count
                self.append("checkpoints.jsonl", record)
            self.append("observations.jsonl", record)

    def on_event(self, event) -> None:
        observed = {
            "observed_wall_ns": time.time_ns(),
            "observed_monotonic_ns": time.monotonic_ns(),
            "phase": self.phase,
        }
        self.append("harness-events.jsonl", {**observed, "event": event.to_dict()})
        if self.host_commands is not None:
            self.host_commands.on_event(event, observed)
        if event.session_id:
            self._session_id = event.session_id
        boundary = action_event(event, observed, self._session_id)
        if boundary:
            self.actions.append(boundary)
            self.append("action-events.jsonl", boundary)
        # Read at completed-action observations, without pausing the container.
        # Codex file changes also have started/updated events; only completion
        # is an action boundary. Keep the native item ID when no tool ID exists.
        completed = event.type in {"tool_result", "result"} or (
            event.type == "file_change" and event.raw.get("type") not in {"item.started", "item.updated"}
        )
        if completed:
            item = event.raw.get("item")
            call_id = event.tool_call_id or (item.get("id") if isinstance(item, dict) else None)
            self.capture("harness_" + event.type, tool_call_id=call_id)

    def _collect(self):
        # The container is stopped before exit. Drain queued events before
        # finalizing, including those already sent when shutdown started.
        while True:
            try:
                connection, _ = self._server.accept()
            except TimeoutError:
                if self._stop.is_set():
                    return
                continue
            try:
                with connection:
                    connection.settimeout(2)
                    with connection.makefile("rb") as stream:
                        line = stream.readline(1024 * 1024 + 1)
                    if len(line) > 1024 * 1024 or not line.endswith(b"\n"):
                        raise ValueError("Malformed command event")
                    event = json.loads(line)
                    if (
                        event.get("event") not in {"start", "end"}
                        or not isinstance(event.get("span_id"), str)
                        or not isinstance(event.get("monotonic_ns"), int)
                        or not isinstance(event.get("wall_time_ns"), int)
                        or not isinstance(event.get("argv"), list)
                        or not all(isinstance(arg, str) for arg in event["argv"])
                        or event.get("kind") not in {"bash", "sh", "dash", "lake", "lake_proxy"}
                    ):
                        raise ValueError("Invalid command span")
                    event.update(
                        phase=self.phase, received_wall_ns=time.time_ns(), received_monotonic_ns=time.monotonic_ns()
                    )
                    self.append("commands.jsonl", event)
                    self.commands.append(event)
                    # Commands wait only for metadata persistence, never a
                    # source scan, diff, or archive. Snapshots use harness events.
                    try:
                        connection.sendall(b"1")
                    except OSError as exc:
                        # A caller can hit its short deadline after persistence.
                        # An ACK failure is not evidence of a missing event.
                        self.warnings.append({
                            "command_acknowledgement": repr(exc),
                            "span_id": event["span_id"], "event": event["event"],
                        })
            except Exception as exc:  # noqa: BLE001 -- keep unexpected worker failures in the durable summary
                self.errors.append({"command_collector": repr(exc)})

    def _observe(self):
        while not self._stop.wait(self.interval_s):
            try:
                self.capture("periodic")
            except Exception as exc:  # noqa: BLE001 -- background failures must not disappear with the thread
                self.errors.append({"periodic_collector": repr(exc)})
                return

    def _observe_resources(self):
        while not self._stop.is_set():
            started = time.monotonic()
            record = {
                "observed_wall_ns": time.time_ns(),
                "observed_monotonic_ns": time.monotonic_ns(),
                "phase": self.phase,
            }
            try:
                record.update(self.resources.sample())
                if record["status"] == "sample":
                    self.resource_samples.append(record)
            except Exception as exc:  # noqa: BLE001 -- preserve failures without interrupting the agent
                record.update(status="error", error=repr(exc))
                self.errors.append({"resource_collector": repr(exc)})
            try:
                self.append("resources.jsonl", record)
            except Exception as exc:  # noqa: BLE001
                self.errors.append({"resource_output": repr(exc)})
                return
            if self._stop.wait(max(0, self.resource_interval_s - (time.monotonic() - started))):
                return

    def assert_ready(self, *, require_commands=False, require_resources=False):
        summary = summarize_commands(self.commands)
        if (
            self.errors or not summary["complete"]
            or (require_commands and not self.commands)
            or (self.require_lake_build and not summary["totals"].get("preflight", {}).get("lake_build"))
            or (require_resources and self.resources is not None and not self.resource_samples)
        ):
            raise RuntimeError(f"Harness telemetry preflight failed; see {self.output}")

    def __enter__(self):
        write_json(
            self.output / "attempt.json",
            {
                "schema_version": 3,
                "command_backend": self.command_backend if self.commands_enabled else None,
                "workspace": str(self.workspace),
                "started_wall_ns": time.time_ns(),
                "interval_s": self.interval_s,
                "resource_interval_s": self.resource_interval_s if self.resources else None,
                "resource_source": "host_cgroup_v2" if self.resources else None,
                "source_root": str(self.source_root),
                "monitoring": {"diff": self.diff_enabled, "cpu": self.cpu_enabled,
                               "ram": self.ram_enabled, "commands": self.commands_enabled},
                "snapshot_triggers": ["completed_action", "periodic", "lifecycle"] if self.diff_enabled else [],
                **self.metadata,
            },
        )
        for name in ("checkpoints.jsonl", "observations.jsonl", "action-events.jsonl"):
            (self.output / name).touch()
        self.capture("baseline")
        targets = []
        if self.commands_enabled and self.command_backend == "wrapper":
            targets.append(self._collect)
        if self.diff_enabled:
            targets.append(self._observe)
        if self.resources is not None:
            targets.append(self._observe_resources)
        for target in targets:
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop.set()
        for thread in self._threads:
            thread.join()
        if self._server is not None:
            self._server.close()
            shutil.rmtree(self._socket_directory)
        self.capture("final")
        try:
            for name, (_, data, _) in read_tree(self.workspace, native_traces=True).items():
                target = self.output / "native" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        except (OSError, ValueError, RuntimeError) as error:
            self.errors.append({"native_trace_capture": repr(error)})
        summary = summarize_commands(self.commands)
        summary.update(
            finished_wall_ns=time.time_ns(), errors=self.errors, warnings=self.warnings,
            attempt_error=repr(exc) if exc else None,
        )
        actions = summarize_actions(self.actions)
        write_json(self.output / "actions.json", actions)
        summary["actions"] = {"count": len(actions["spans"]), "complete": actions["complete"], "path": "actions.json"}
        summary["source_history"] = {"snapshots": self.snapshot_count, "observations": self.observation_count}
        summary["resources"] = {
            "enabled": self.resources is not None,
            "cpu": self.resources is not None and self.cpu_enabled,
            "ram": self.resources is not None and self.ram_enabled,
            "samples": len(self.resource_samples),
            "interval_s": self.resource_interval_s,
            "peak_sampled_memory_bytes": max(
                (row["memory_current_bytes"] for row in self.resource_samples if "memory_current_bytes" in row), default=None,
            ),
            "peak_sampled_cpu_percent": max(
                (row["cpu_percent"] for row in self.resource_samples if row.get("cpu_percent") is not None), default=None,
            ),
        }
        if self.host_commands is not None:
            host_summary = self.host_commands.summary()
            write_json(self.output / "host-command-summary.json", host_summary)
            summary["host_commands"] = {"path": "host-command-summary.json", "complete": host_summary["complete"],
                                        "count": len(host_summary["spans"]), "nested_process_timings": False}
        summary["command_backend"] = self.command_backend if self.commands_enabled else None
        summary["complete"] = (
            (not self.commands_enabled or self.command_backend == "host" or bool(self.commands)) and summary["complete"] and actions["complete"] and not self.errors
            and (self.resources is None or bool(self.resource_samples))
            and (self.host_commands is None or summary["host_commands"]["complete"])
        )
        write_json(self.output / "summary.json", summary)
        self.summary = summary
