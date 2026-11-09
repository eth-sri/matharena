import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_wrapper import AgentEvent

from matharena.monitoring.host_commands import HostCommands
from test_harness_monitoring import solver


def observed(n):
    return {"phase": "model", "observed_wall_ns": n * 10**9, "observed_monotonic_ns": n * 10**9}


@pytest.mark.parametrize("name,content", [
    ("Bash", {"command": "lake build && lake comparator"}),
    ("command_execution", "/bin/bash -c 'lake build && lake comparator'"),
    ("run_shell_command", {"CommandLine": "lake build && lake comparator"}),
    ("exec_command", {"cmd": "lake build && lake comparator"}),
])
def test_whole_tool_call_preserves_command_and_does_not_invent_nested_timings(name, content):
    records = []
    recorder = HostCommands(lambda path, event: records.append(event))
    recorder.on_event(AgentEvent(type="tool_call", tool_name=name, tool_call_id="one", content=content), observed(1))
    recorder.on_event(AgentEvent(type="tool_result", tool_name=name, tool_call_id="one", content="done"), observed(4))
    result = recorder.summary()
    assert result["complete"]
    assert result["nested_process_timings"] is False
    assert len(result["spans"]) == 1
    assert result["spans"][0]["duration_s"] == 3
    assert result["spans"][0]["timestamp_source"] == "host_observation"
    assert result["spans"][0]["coverage"] == "whole_tool_call"
    assert "lake comparator" in result["spans"][0]["command"]
    assert [r["event"] for r in records] == ["start", "end"]


def test_missing_boundaries_and_ambiguous_ids_remain_incomplete():
    recorder = HostCommands(lambda *args: None)
    recorder.on_event(AgentEvent(type="tool_result", tool_name="bash", tool_call_id="missing"), observed(1))
    for n in (2, 3):
        recorder.on_event(AgentEvent(type="tool_call", tool_name="bash", tool_call_id="duplicate", content="lake build"), observed(n))
    recorder.on_event(AgentEvent(type="tool_result", tool_name="bash", tool_call_id="duplicate"), observed(4))
    result = recorder.summary()
    assert not result["complete"]
    assert [r["status"] for r in result["spans"]] == ["missing_start", "missing_end", "missing_end"]


def test_preflight_timeout_is_recorded_with_a_host_duration():
    recorder = HostCommands(lambda *args: None)
    with pytest.raises(subprocess.TimeoutExpired):
        with recorder.lifecycle(["sh", "-c", "build"], "preflight"):
            raise subprocess.TimeoutExpired("build", 1)
    result = recorder.summary()
    assert result["complete"]
    assert result["spans"][0]["end"]["error_type"] == "TimeoutExpired"
    assert result["spans"][0]["end"]["returncode"] is None
    assert result["spans"][0]["duration_s"] >= 0


def test_host_backend_adds_no_container_mounts_environment_or_command_wrappers(tmp_path, monkeypatch):
    from matharena.solvers import SolverResponse

    subject = solver(tmp_path, monkeypatch, monitoring={"commands": True}, preflight="lake build")
    sandbox = SimpleNamespace(image="clean", mounts=(), stop=lambda: None)

    def preflight(command, **kwargs):
        assert command == ["sh", "-c", "lake build"]
        assert not any(key.startswith("LEANLEAN_TRACE") for key in kwargs["env"])
        assert sandbox.mounts == ()
        return SimpleNamespace(returncode=0, stdout="built", stderr="")

    def run_model(*args, **kwargs):
        assert sandbox.mounts == ()
        assert not any(key.startswith("LEANLEAN_TRACE") for key in kwargs["environment"])
        for event in [AgentEvent(type="tool_call", tool_name="bash", tool_call_id="one", content={"command": "lake comparator"}),
                      AgentEvent(type="tool_result", tool_name="bash", tool_call_id="one", content="ok")]:
            kwargs["on_event"](event)
        kwargs["after_run"]()
        return SolverResponse(0, [], {}, history=[{}])

    sandbox.run = preflight
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: sandbox)
    monkeypatch.setattr(subject, "_run_prepared", run_model)
    response = subject._solve_one(0, "Problem", None, 1, 0)
    directory = Path(response.history[0]["monitoring"]["directory"])
    summary = json.loads((directory / "host-command-summary.json").read_text())
    assert response.history[0]["monitoring"]["complete"]
    assert summary["complete"] and len(summary["spans"]) == 2
    assert {s["phase"] for s in summary["spans"]} == {"preflight", "model"}
    assert not (directory / "commands.jsonl").exists()
    assert json.loads((directory / "attempt.json").read_text())["command_backend"] == "host"


def test_native_execution_times_are_preserved_when_both_boundaries_have_them():
    recorder = HostCommands(lambda *args: None)
    raw = {"type": "tool_use", "part": {"state": {
        "status": "completed", "time": {"start": 1000, "end": 4000},
    }}}
    recorder.on_event(AgentEvent(type="tool_call", tool_name="bash", tool_call_id="one",
                                content={"command": "lake build"}, raw=raw), observed(10))
    recorder.on_event(AgentEvent(type="tool_result", tool_name="bash", tool_call_id="one", raw=raw), observed(10))
    span = recorder.summary()["spans"][0]
    assert span["timestamp_source"] == "native_execution"
    assert span["duration_s"] == 3


@pytest.mark.parametrize("analysis", ["lake", "time"])
def test_process_analysis_does_not_treat_host_tool_durations_as_process_times(tmp_path, analysis):
    (tmp_path / "attempt.json").write_text(json.dumps({"command_backend": "host"}))
    if analysis == "lake":
        from leanlean.lake_analysis import analyze_attempt
        invoke = lambda: analyze_attempt(tmp_path, {})
    else:
        from leanlean.time_analysis import analyze
        invoke = lambda: analyze(tmp_path)
    with pytest.raises(ValueError, match="host tool durations"):
        invoke()


@pytest.mark.parametrize("duration,expected_status", [(12.5, "complete"), (None, "duration_unavailable"), (-1, "duration_unavailable"), (True, "duration_unavailable")])
def test_gemini_completion_callbacks_use_native_duration_without_fabricated_start_time(duration, expected_status):
    recorder = HostCommands(lambda *args: None)
    raw = {"event": "step_update", "step_update": {
        "step_type": "tool", "state": "DONE", "duration_seconds": duration,
    }}
    for kind in ("tool_call", "tool_result"):
        recorder.on_event(AgentEvent(type=kind, tool_name="run_command", tool_call_id="gemini:1",
                                    content={"CommandLine": "lake build"} if kind == "tool_call" else "done",
                                    raw=raw), observed(20))
    summary = recorder.summary()
    row = summary["spans"][0]
    assert row["status"] == expected_status
    assert row["duration_s"] == (12.5 if duration == 12.5 else None)
    assert row["started_wall_ns"] is None and row["ended_wall_ns"] is None
    assert row["observed_completed_wall_ns"] == 20 * 10**9
