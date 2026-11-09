import json
import os
import signal
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from harness_wrapper import AgentEvent

from matharena.monitoring.commands import SOCKET_ENV
from matharena.monitoring.attempt import AttemptMonitor as SharedAttemptMonitor, read_tree, summarize_commands


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class AttemptMonitor(SharedAttemptMonitor):
    """LeanLean workspace layout with wrapper command tracing."""

    def __init__(self, workspace, **kwargs):
        kwargs.setdefault("require_lake_build", True)
        kwargs.setdefault("command_backend", "wrapper")
        kwargs.setdefault("source_root", Path(workspace) / "repository")
        super().__init__(workspace, **kwargs)

    def assert_ready(self, **kwargs):
        kwargs.setdefault("require_resources", True)
        super().assert_ready(**kwargs)


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "attempt"
    (root / "repository").mkdir(parents=True)
    (root / "repository/Solution.lean").write_text("theorem t : True := by trivial\n")
    return root


@pytest.fixture
def wrapper(tmp_path):
    path = tmp_path / "traced-bash"
    path.write_text(
        f"#!{sys.executable}\nimport sys\n"
        f"sys.path.insert(0, {str(Path(__file__).resolve().parents[1] / 'src')!r})\n"
        "from matharena.monitoring.commands import run\n"
        "sys.exit(run('bash', '/bin/bash', sys.argv[1:]))\n"
    )
    path.chmod(0o755)
    return path


def test_snapshots_retain_edits_deletions_and_restoration(workspace):
    source = workspace / "repository/Solution.lean"
    original = source.read_bytes()
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        source.write_text("theorem t : True := True.intro\n")
        monitor.capture("edit")
        source.unlink()
        monitor.capture("delete")
        source.write_bytes(original)
        monitor.capture("restore")
        monitor.capture("unchanged")
        output = monitor.output
    checkpoints = records(output / "observations.jsonl")
    assert [record["changed"] for record in checkpoints] == [True, True, True, True, False, False]
    assert checkpoints[0]["source_sha256"] == checkpoints[3]["source_sha256"]
    assert len(list((output / "sources").glob("*.tar.gz"))) == 3
    snapshots = records(output / "checkpoints.jsonl")
    assert len(snapshots) == 4
    assert all(row["changed"] and "diff" in row for row in snapshots)
    assert monitor.summary["source_history"] == {"snapshots": 4, "observations": 6}
    edit = checkpoints[1]
    assert "+theorem t : True := True.intro" in (output / edit["diff"]).read_text()
    with tarfile.open(output / edit["archive"]) as archive:
        assert b"True.intro" in archive.extractfile("Solution.lean").read()


def test_symlinks_cache_exclusion_and_replaced_root(workspace, tmp_path):
    repository = workspace / "repository"
    (repository / "external").symlink_to(tmp_path)
    (repository / ".lake").mkdir()
    (repository / ".lake/large-cache").write_text("excluded")
    state = read_tree(repository)
    assert state["external"] == ("symlink", str(tmp_path), 0o777)
    assert set(state) == {"external", "Solution.lean"}
    repository.rename(workspace / "old")
    repository.symlink_to(tmp_path)
    with pytest.raises(OSError):
        read_tree(repository)


def test_scratch_changes_and_nested_scratch_links_are_not_checkpointed(workspace, tmp_path):
    repository = workspace / "repository"
    (repository / ".scratch").mkdir()
    (repository / ".scratch/test.lean").write_text("scratch source")
    (repository / ".scratch/test.olean").write_bytes(b"compiled")
    assert ".scratch/test.lean" in read_tree(repository)
    assert ".scratch/test.olean" not in read_tree(repository)
    (repository / "nested").mkdir()
    (repository / "nested/.scratch").symlink_to(tmp_path)
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        (repository / ".scratch/test.olean").write_bytes(b"changed compiled output")
        monitor.capture("scratch changed")
    frames = records(monitor.output / "checkpoints.jsonl")
    assert len(frames) == 1
    with tarfile.open(monitor.output / frames[0]["archive"]) as archive:
        assert archive.getnames() == ["Solution.lean"]


def test_native_trace_and_failed_attempt_survive_workspace_removal(workspace):
    native = workspace / ".harness-home/.codex/sessions/rollout.jsonl"
    native.parent.mkdir(parents=True)
    native.write_bytes(b'{"native": "exact bytes"}\n')
    monitor = AttemptMonitor(workspace, interval_s=60)
    with pytest.raises(RuntimeError, match="model failed"), monitor:
        raise RuntimeError("model failed")
    import shutil

    shutil.rmtree(workspace)
    assert (
        monitor.output / "native/.harness-home/.codex/sessions/rollout.jsonl"
    ).read_bytes() == b'{"native": "exact bytes"}\n'
    assert "model failed" in json.loads((monitor.output / "summary.json").read_text())["attempt_error"]
    assert list((monitor.output / "sources").glob("*.tar.gz"))


def test_parallel_nested_commands_status_and_parent_pairing(workspace, wrapper):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        env = {**os.environ, SOCKET_ENV: str(monitor.socket_path), "INNER": str(wrapper)}

        def command(number):
            return subprocess.run(
                [str(wrapper), "-c", '"$INNER" -c "sleep 0.05; exit 7"'], env=env, capture_output=True, check=False
            )

        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(command, range(3)))
        assert [result.returncode for result in results] == [7, 7, 7]
    summary = json.loads((monitor.output / "summary.json").read_text())
    assert summary["complete"]
    assert len(summary["spans"]) == 6
    parents = {row["span_id"] for row in summary["spans"] if row["start"]["parent_span_id"] is None}
    assert len(parents) == 3
    assert all(row["start"]["parent_span_id"] in parents for row in summary["spans"] if row["span_id"] not in parents)
    assert all(row["duration_s"] >= 0.05 and row["end"]["returncode"] == 7 for row in summary["spans"])


def test_signal_is_forwarded_and_recorded(workspace, wrapper):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        process = subprocess.Popen(
            [str(wrapper), "-c", "exec sleep 30"], env={**os.environ, SOCKET_ENV: str(monitor.socket_path)}
        )
        deadline = time.monotonic() + 5
        while not monitor.commands and time.monotonic() < deadline:
            time.sleep(0.01)
        assert monitor.commands
        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == -signal.SIGTERM
    summary = json.loads((monitor.output / "summary.json").read_text())
    assert summary["complete"]
    assert summary["spans"][0]["end"]["returncode"] == -signal.SIGTERM


def test_incomplete_spans_and_overlapping_builds_are_not_invented():
    def event(span, action, second):
        return {
            "span_id": span,
            "event": action,
            "monotonic_ns": second * 10**9,
            "argv": ["lake", "build"],
            "kind": "lake",
            "phase": "model",
        }

    summary = summarize_commands(
        [
            event("a", "start", 1),
            event("b", "start", 3),
            event("a", "end", 5),
            event("b", "end", 6),
            event("c", "start", 7),
            event("d", "end", 8),
        ]
    )
    assert not summary["complete"]
    assert summary["totals"]["model"]["lake_build"] == {"commands": 2, "process_seconds": 7, "busy_wall_seconds": 5}
    assert [row["status"] for row in summary["spans"]] == ["complete", "complete", "missing_end", "missing_start"]
    assert all("duration_s" not in row for row in summary["spans"][2:])


def test_uninstrumented_preflight_fails_before_model(workspace):
    with (
        AttemptMonitor(workspace, interval_s=60) as monitor,
        pytest.raises(RuntimeError, match="telemetry preflight failed"),
    ):
        monitor.assert_ready()


def test_periodic_capture_during_long_command(workspace, wrapper):
    with AttemptMonitor(workspace, interval_s=0.05) as monitor:
        process = subprocess.Popen(
            [str(wrapper), "-c", "sleep .8"], env={**os.environ, SOCKET_ENV: str(monitor.socket_path)}
        )
        time.sleep(0.15)
        (workspace / "repository/Solution.lean").write_text("changed mid-command\n")
        process.wait(timeout=5)
    checkpoints = records(monitor.output / "observations.jsonl")
    assert any(row["reason"] == "periodic" and row["changed"] for row in checkpoints)


def test_every_completed_action_observes_edits_and_restoration(workspace):
    source = workspace / "repository/Solution.lean"
    original = source.read_bytes()
    with AttemptMonitor(workspace) as monitor:
        assert monitor.interval_s == 30
        for index, text in enumerate(("intermediate\n", original.decode(), original.decode())):
            monitor.on_event(AgentEvent(type="tool_call", tool_call_id=str(index)))
            source.write_text(text)
            monitor.on_event(AgentEvent(type="tool_result", tool_call_id=str(index)))
    checkpoints = records(monitor.output / "observations.jsonl")
    actions = [row for row in checkpoints if row["reason"] == "harness_tool_result"]
    assert [row["tool_call_id"] for row in actions] == ["0", "1", "2"]
    assert [row["changed"] for row in actions] == [True, True, False]
    assert len(checkpoints) == 5  # baseline, every completed action, final
    assert checkpoints[0]["source_sha256"] == checkpoints[-1]["source_sha256"]


def test_file_change_observed_only_on_completion(workspace):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        for phase in ("started", "updated", "completed"):
            monitor.on_event(AgentEvent(
                type="file_change", raw={"type": f"item.{phase}", "item": {"id": "edit-1"}}
            ))
    actions = [row for row in records(monitor.output / "observations.jsonl")
               if row["reason"] == "harness_file_change"]
    assert len(actions) == 1
    assert actions[0]["tool_call_id"] == "edit-1"


def test_commands_do_not_wait_for_source_capture(workspace, wrapper):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        # A slow or blocked capture must not hold up command execution.
        with monitor._capture_lock:
            result = subprocess.run(
                [str(wrapper), "-c", "printf out; printf err >&2; exit 7"],
                env={**os.environ, SOCKET_ENV: str(monitor.socket_path)},
                capture_output=True, timeout=5, check=False,
            )
    assert (result.returncode, result.stdout, result.stderr) == (7, b"out", b"err")
    assert monitor.summary["complete"]
    assert len(monitor.summary["spans"]) == 1
    assert [row["reason"] for row in records(monitor.output / "observations.jsonl")] == ["baseline", "final"]


def test_collector_failure_does_not_change_command_result(tmp_path, wrapper):
    result = subprocess.run(
        [str(wrapper), "-c", "printf out; printf err >&2; exit 7"],
        env={**os.environ, SOCKET_ENV: str(tmp_path / "missing.sock")},
        capture_output=True, timeout=5, check=False,
    )
    assert (result.returncode, result.stdout, result.stderr) == (7, b"out", b"err")


@pytest.mark.parametrize("start_delivered", [True, False])
def test_delivery_error_uses_persisted_events_to_determine_completeness(start_delivered):
    common = {"span_id": "a", "argv": ["bash"], "kind": "bash", "phase": "model"}
    start = {**common, "event": "start", "monotonic_ns": 1}
    end = {**common, "event": "end", "monotonic_ns": 2, "start_delivery_error": "timeout"}
    summary = summarize_commands(([start] if start_delivered else []) + [end])
    assert summary["complete"] is start_delivered
    assert summary["spans"][0]["status"] == ("complete" if start_delivered else "missing_start")
    assert summary["spans"][0]["end"]["start_delivery_error"] == "timeout"


def test_unchanged_observations_do_not_rebuild_archives(workspace, monkeypatch):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        def unexpected_archive(state):
            pytest.fail("Unchanged source must reuse its archive")

        monkeypatch.setattr("matharena.monitoring.attempt.archive_bytes", unexpected_archive)
        monitor.on_event(AgentEvent(type="tool_result", tool_call_id="read-only-action"))
    assert not monitor.errors


@pytest.mark.parametrize(
    "arguments, expected",
    [
        (["build"], "build"),
        (["-q", "--dir", "project", "build"], "build"),
        (["+v4.35.0", "-K", "key=build", "--dir=project", "build"], "build"),
        (["env", "lake", "build"], "env"),
        (["--help", "build"], None),
    ],
)
def test_lake_build_detection_with_options(arguments, expected):
    from matharena.monitoring.attempt import lake_subcommand

    assert lake_subcommand(arguments) == expected


def test_lost_acknowledgements_do_not_discard_persisted_spans(workspace, wrapper, monkeypatch):
    import socket

    send = socket.socket.sendall

    def lose_ack(connection, data, *args, **kwargs):
        if data == b"1":
            raise BrokenPipeError("client disconnected before acknowledgement")
        return send(connection, data, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "sendall", lose_ack)
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        result = subprocess.run(
            [str(wrapper), "-c", "exit 0"],
            env={**os.environ, SOCKET_ENV: str(monitor.socket_path)}, check=False,
        )
        assert result.returncode == 0
    assert monitor.summary["complete"]
    assert monitor.summary["errors"] == []
    assert len(monitor.summary["warnings"]) == 2
    assert monitor.summary["spans"][0]["end"]["start_delivery_error"]
    assert monitor.summary["spans"][0]["duration_s"] >= 0
    assert len(records(monitor.output / "commands.jsonl")) == 2


def test_failed_command_persistence_remains_incomplete(workspace, wrapper, monkeypatch):
    with AttemptMonitor(workspace, interval_s=60) as monitor:
        append = monitor.append

        def fail_commands(name, value):
            if name == "commands.jsonl":
                raise OSError("test disk full")
            return append(name, value)

        monkeypatch.setattr(monitor, "append", fail_commands)
        result = subprocess.run(
            [str(wrapper), "-c", "exit 0"],
            env={**os.environ, SOCKET_ENV: str(monitor.socket_path)}, check=False,
        )
        assert result.returncode == 0
    assert not monitor.summary["complete"]
    assert len(monitor.summary["errors"]) == 2
    assert not monitor.summary["spans"]


def test_shutdown_drains_already_queued_events(workspace):
    import socket

    monitor = AttemptMonitor(workspace, interval_s=60)
    for kind, second in [("start", 1), ("end", 2)]:
        event = {
            "event": kind, "span_id": "queued", "monotonic_ns": second * 10**9,
            "wall_time_ns": second * 10**9, "kind": "bash", "argv": ["bash", "-c", "true"],
        }
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.connect(str(monitor.socket_path))
            client.sendall(json.dumps(event).encode() + b"\n")
    # Exercise collection after stop has already been requested, deterministically.
    monitor._stop.set()
    monitor._collect()
    monitor.__exit__(None, None, None)
    assert monitor.summary["complete"]
    assert monitor.summary["spans"][0]["duration_s"] == 1
