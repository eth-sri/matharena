"""Execution timestamps from inside the evaluation container (stdlib only).

Installed over the image's shells and Lake binaries. Events go to a host-owned
Unix socket; the model never receives a writable mount of the trace archive.
This measures process lifetimes, including compound shell commands, not each
shell builtin. It is observability instrumentation, not a security boundary.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

SOCKET_ENV = "LEANLEAN_TRACE_SOCKET"
PARENT_ENV = "LEANLEAN_TRACE_PARENT"
CONTAINER_SOCKET = "/run/leanlean-trace.sock"
TRACE_VERSION = 2


def emit(address: str, event: dict) -> str | None:
    """Bound the metadata handoff; collection failures must not break commands."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(0.25)
            connection.connect(address)
            connection.sendall(json.dumps(event).encode() + b"\n")
            if connection.recv(1) != b"1":
                raise RuntimeError("Command collector did not acknowledge the event")
    except (OSError, RuntimeError) as exc:
        return repr(exc)
    return None


def run(kind: str, real: str, arguments: list[str]) -> int:
    command = [sys.argv[0], *arguments]
    address = os.environ.get(SOCKET_ENV)
    if not address:
        os.execv(real, command)
    span = uuid.uuid4().hex
    common = {
        "schema_version": 1,
        "span_id": span,
        "parent_span_id": os.environ.get(PARENT_ENV),
        "kind": kind,
        "argv": command,
        "cwd": os.getcwd(),
        "pid": os.getpid(),
        "ppid": os.getppid(),
    }
    started = time.monotonic_ns()
    environment = {**os.environ, PARENT_ENV: span}
    child = None
    pending_signals = []

    def forward(signum, frame):
        if child is None:
            pending_signals.append(signum)
        else:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    previous = {
        sig: signal.signal(sig, forward) for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)
    }
    code, error = None, None
    execution_started, execution_wall = None, None
    delivery_error = None
    try:
        delivery_error = emit(address, {**common, "event": "start", "wall_time_ns": time.time_ns(), "monotonic_ns": started})
        # Exclude the short metadata handoff from execution timing.
        execution_started, execution_wall = time.monotonic_ns(), time.time_ns()
        child = subprocess.Popen(command, executable=real, env=environment)
        for signum in pending_signals:
            child.send_signal(signum)
        code = child.wait()
    except BaseException as exc:
        error = repr(exc)
        raise
    finally:
        ended = time.monotonic_ns()
        emit(
            address,
            {
                **common,
                "event": "end",
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": ended,
                "duration_ns": ended - started,
                "returncode": code,
                "error": error,
                "start_delivery_error": delivery_error,
                "execution_started_monotonic_ns": execution_started,
                "execution_started_wall_ns": execution_wall,
            },
        )
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    if code < 0:
        signal.signal(-code, signal.SIG_DFL)
        os.kill(os.getpid(), -code)
    return code


def install() -> None:
    """Run once while building the image, before its root becomes read-only."""
    marker = Path("/opt/leanlean/command-trace-version")
    if marker.exists():
        raise RuntimeError("Command tracing is already installed")
    binaries = Path("/opt/leanlean/real-bin")
    binaries.mkdir(parents=True, exist_ok=True)

    def replace(path: Path, kind: str, real: Path, *, copy: bool = False):
        if copy:
            shutil.copy2(path, real, follow_symlinks=True)
        else:
            path.rename(real)
        path.unlink(missing_ok=True)
        path.write_text(
            "#!/usr/local/bin/python3 -I\nimport sys\nsys.path.insert(0, '/pipeline')\n"
            "from matharena.monitoring.commands import run\n"
            f"sys.exit(run({kind!r}, {str(real)!r}, sys.argv[1:]))\n"
        )
        path.chmod(0o755)

    for name in ("bash", "sh", "dash"):
        # /bin and /usr/bin are merged in the Debian runtime. Preserve dash
        # before replacing sh, which is normally a symlink to it.
        path = Path("/usr/bin") / name
        replace(path, name, binaries / name, copy=True)
    for path in sorted(Path("/opt/elan/toolchains").glob("*/bin/lake")):
        replace(path, "lake", path.with_name(".lake-real"))
    # Preserve Elan's own +toolchain selection and error handling. The proxy
    # then invokes the instrumented toolchain binary as a nested Lake span.
    proxy = binaries / "elan/lake"
    proxy.parent.mkdir()
    replace(Path("/opt/elan/bin/lake"), "lake_proxy", proxy, copy=True)
    marker.write_text(f"{TRACE_VERSION}\n")


if __name__ == "__main__":
    if sys.argv[1:] != ["install"]:
        raise SystemExit("Usage: python -m matharena.monitoring.commands install")
    install()
