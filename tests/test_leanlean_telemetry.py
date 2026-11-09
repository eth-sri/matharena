import errno
import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_wrapper import AgentEvent

from matharena.monitoring.actions import action_event, summarize_actions
from test_leanlean_monitoring import AttemptMonitor
from matharena.monitoring.resources import ContainerResources, cpu_rate, read_counters


def action(kind, call_id, second, *, session="s", raw=None):
    return action_event(
        AgentEvent(type=kind, tool_call_id=call_id, raw=raw or {}),
        {"observed_wall_ns": second * 10**9, "observed_monotonic_ns": second * 10**9, "phase": "model"},
        session,
    )


def test_actions_pair_by_id_with_parallel_execution_and_missing_boundaries():
    result = summarize_actions([
        action("tool_call", "a", 1), action("tool_call", "b", 2),
        action("tool_result", "b", 4), action("tool_result", "a", 5),
        action("tool_call", "c", 6), action("tool_result", "d", 7),
    ])
    assert not result["complete"]
    assert [row["duration_s"] for row in result["spans"]] == [4, 2, None, None]
    assert [row["status"] for row in result["spans"]] == ["complete", "complete", "missing_end", "missing_start"]
    assert result["spans"][0]["timestamp_source"] == "host_observation"


def test_actions_do_not_cross_sessions_or_guess_missing_ids():
    result = summarize_actions([
        action("tool_call", "a", 1), action("tool_result", "a", 2, session="different"),
        action("tool_call", None, 3), action("tool_result", None, 4),
    ])
    assert len(result["spans"]) == 4
    assert all(row["duration_s"] is None for row in result["spans"])


def test_repeated_call_ids_get_distinct_actions():
    result = summarize_actions([
        action("tool_call", "a", 1), action("tool_result", "a", 2),
        action("tool_call", "a", 3), action("tool_result", "a", 5),
    ])
    assert result["complete"]
    assert [row["duration_s"] for row in result["spans"]] == [1, 2]


def test_native_times_used_for_batched_completion_without_fake_zero_duration():
    raw = {"type": "tool_use", "part": {"state": {"status": "completed", "time": {"start": 1000, "end": 4000}}}}
    result = summarize_actions([action("tool_call", "a", 5, raw=raw), action("tool_result", "a", 5, raw=raw)])
    assert result["spans"][0]["duration_s"] == 3
    assert result["spans"][0]["timestamp_source"] == "native_execution"
    raw["part"]["state"].pop("time")
    assert action("tool_call", "a", 5, raw=raw) is None
    result = summarize_actions([action("tool_result", "a", 5, raw=raw)])
    assert result["spans"][0]["status"] == "missing_start"


def test_codex_empty_completion_is_retained():
    from harness_wrapper.harnesses.codex_cli.adapter import CodexCLIAgent

    # normalize_event only needs its turn-text field for tool events.
    agent = object.__new__(CodexCLIAgent)
    events = agent.normalize_event({"type": "item.completed", "item": {"id": "empty", "type": "mcp_tool_call"}})
    assert len(events) == 1
    assert events[0].type == "tool_result" and events[0].tool_call_id == "empty"


def test_cpu_rate_uses_actual_interval_and_resets_at_container_replacement():
    previous = {"container_id": "a", "sampled_monotonic_ns": 1_000_000_000, "cpu_usage_usec": 100_000}
    current = {"container_id": "a", "sampled_monotonic_ns": 3_000_000_000, "cpu_usage_usec": 3_100_000}
    assert cpu_rate(current, previous)["cpu_percent"] == 150
    assert cpu_rate(current, previous)["cpu_interval_s"] == 2
    assert cpu_rate(current, None)["cpu_percent"] is None
    assert cpu_rate({**current, "container_id": "b"}, previous)["cpu_percent"] is None
    assert cpu_rate({**current, "cpu_usage_usec": 0}, previous)["cpu_percent"] is None


def test_cgroup_memory_and_limits(tmp_path):
    values = {
        "cpu.stat": "usage_usec 300\nuser_usec 200\nsystem_usec 100\nthrottled_usec 7\n",
        "cpu.max": "200000 100000", "memory.current": "1048576", "memory.peak": "2097152",
        "memory.max": "max", "memory.stat": "anon 524288\nfile 262144\n",
    }
    for name, value in values.items():
        (tmp_path / name).write_text(value)
    sample = read_counters(tmp_path)
    assert sample["memory_current_bytes"] == 1048576
    assert sample["memory_peak_bytes"] == 2097152
    assert sample["memory_limit_bytes"] is None
    assert sample["memory_anon_bytes"] == 524288
    assert sample["cpu_limit_cores"] == 2
    assert sample["cpu_throttled_usec"] == 7
    (tmp_path / "memory.peak").unlink()
    assert read_counters(tmp_path)["memory_peak_bytes"] is None


@pytest.mark.parametrize("error", [errno.ENODEV, errno.ENOENT])
def test_resource_sampling_recovers_after_container_teardown(tmp_path, monkeypatch, error):
    resources = ContainerResources("podman", "attempt")
    resources.directory = tmp_path / "old"
    if error == errno.ENODEV:
        resources.directory.mkdir()  # Offline cgroups can still have a directory.
    resources.container_id = "old-id"
    resources.previous = {"container_id": "old-id"}

    def offline(directory):
        raise OSError(error, "container exited")

    monkeypatch.setattr("matharena.monitoring.resources.read_counters", offline)
    assert resources.sample() == {"status": "not_running"}
    assert resources.directory is resources.container_id is resources.previous is None

    monkeypatch.setattr(
        "matharena.monitoring.resources.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="new-id 123", stderr=""),
    )
    monkeypatch.setattr(
        Path, "read_text",
        lambda path: "0::/user.slice/new-id\n" if str(path) == "/proc/123/cgroup" else "0::/host\n",
    )
    monkeypatch.setattr(
        "matharena.monitoring.resources.read_counters",
        lambda directory: {"sampled_monotonic_ns": 100, "cpu_usage_usec": 10},
    )
    sample = resources.sample()
    assert sample["status"] == "sample" and sample["container_id"] == "new-id"
    assert sample["cpu_percent"] is None  # The replacement needs a new baseline.


@pytest.mark.parametrize("error", [errno.ENOENT, errno.EACCES, errno.EIO])
def test_resource_sampling_does_not_hide_live_accounting_failures(tmp_path, monkeypatch, error):
    resources = ContainerResources("podman", "attempt")
    resources.directory, resources.container_id = tmp_path, "live-id"

    def fail(directory):
        raise OSError(error, "accounting failed")

    monkeypatch.setattr("matharena.monitoring.resources.read_counters", fail)
    with pytest.raises(OSError) as caught:
        resources.sample()
    assert caught.value.errno == error
    assert resources.container_id == "live-id"


def test_resource_sampling_runs_independently_of_source_capture(tmp_path):
    (tmp_path / "repository").mkdir()
    sampled = threading.Event()

    class Resources:
        def sample(self):
            sampled.set()
            return {"status": "sample", "memory_current_bytes": 42, "cpu_percent": 5}

    with AttemptMonitor(tmp_path, resources=Resources(), resource_interval_s=0.01) as monitor:
        with monitor._capture_lock:
            sampled.clear()
            assert sampled.wait(2)
    assert monitor.summary["resources"]["samples"] >= 1
    assert monitor.summary["resources"]["peak_sampled_memory_bytes"] == 42
    assert not monitor.errors


def test_resource_failures_are_recorded(tmp_path):
    (tmp_path / "repository").mkdir()
    failed = threading.Event()

    class Resources:
        def sample(self):
            failed.set()
            raise PermissionError("cgroup denied")

    with AttemptMonitor(tmp_path, resources=Resources(), resource_interval_s=0.01) as monitor:
        assert failed.wait(2)
    rows = [json.loads(line) for line in (monitor.output / "resources.jsonl").read_text().splitlines()]
    assert rows[0]["status"] == "error" and "cgroup denied" in rows[0]["error"]
    assert not monitor.summary["complete"]


def test_refuses_engine_host_cgroup_instead_of_container_usage(monkeypatch):
    monkeypatch.setattr(
        "matharena.monitoring.resources.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="container-id 123", stderr=""),
    )
    monkeypatch.setattr(Path, "read_text", lambda self: "0::/system.slice/docker.service\n")
    with pytest.raises(RuntimeError, match="Container-specific"):
        ContainerResources("docker", "attempt").sample()
