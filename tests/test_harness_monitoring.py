"""Shared monitoring and preflight work without a LeanLean dataset or image."""

import json
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from harness_wrapper import AgentEvent, TokenUsage

from matharena.monitoring import AttemptMonitor, monitoring_flags
from matharena.monitoring.resources import read_counters
from matharena.solvers import harness_solver


@pytest.mark.parametrize("value", [None, {}])
def test_monitoring_defaults_off(value):
    assert not any(monitoring_flags(value).values())


@pytest.mark.parametrize("value", [True, False, [], "cpu", {"cpu": "true"}, {"build": True}])
def test_invalid_monitoring_is_rejected(value):
    with pytest.raises((TypeError, ValueError), match="monitoring"):
        monitoring_flags(value)


def solver(tmp_path, monkeypatch, **settings):
    config = {
        "model": "test", "harness": "codex", "read_cost": 1, "write_cost": 1,
        "harness_config": {"managed_cli": False, "workspace_root": str(tmp_path / "attempt"), **settings},
    }
    result = harness_solver.HarnessSolver({}, "{problem}", config, "Try again")
    monkeypatch.setattr(result, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(result, "_build_model", lambda: None)
    return result


def test_preflight_runs_before_model_without_monitoring(tmp_path, monkeypatch):
    subject = solver(tmp_path, monkeypatch, preflight="lake build && lake comparator", max_time_seconds=300)
    order = []

    def preflight(command, **kwargs):
        order.append("preflight")
        assert command == ["sh", "-c", "lake build && lake comparator"]
        assert kwargs["timeout"] == 1800
        return SimpleNamespace(returncode=0, stdout="built", stderr="checked")

    def run_model(*args, **kwargs):
        order.append("model")
        from matharena.solvers import SolverResponse
        return SolverResponse(0, [], {}, history=[{}])

    monkeypatch.setattr(subject, "_run_prepared", run_model)
    sandbox = SimpleNamespace(run=preflight, stop=lambda: order.append("stop"))
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: sandbox)
    response = subject._solve_one(0, "Problem", None, 1, 0)
    assert order == ["preflight", "model", "stop"]
    assert "monitoring" not in response.history[0]
    assert (tmp_path / "attempt/preflight_stdout.log").read_text() == "built"
    assert not (tmp_path / "attempt.telemetry").exists()


@pytest.mark.parametrize("failure", ["exit", "timeout"])
def test_failed_preflight_never_builds_model_and_retains_logs(tmp_path, monkeypatch, failure):
    subject = solver(tmp_path, monkeypatch, preflight="build")
    stopped = []

    def preflight(*args, **kwargs):
        if failure == "timeout":
            raise subprocess.TimeoutExpired("build", 1, output=b"partial", stderr=b"failed")
        return SimpleNamespace(returncode=1, stdout="partial", stderr="failed")

    monkeypatch.setattr(subject, "_build_model", lambda: pytest.fail("model accessed"))
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: SimpleNamespace(
        run=preflight, stop=lambda: stopped.append(True),
    ))
    with pytest.raises((RuntimeError, subprocess.TimeoutExpired)):
        subject._solve_one(0, "Problem", None, 1, 0)
    assert stopped
    assert (tmp_path / "attempt/preflight_stdout.log").read_text() == "partial"
    assert (tmp_path / "attempt/preflight_stderr.log").read_text() == "failed"


def test_diff_monitor_tracks_normal_workspace_and_resume(tmp_path, monkeypatch):
    subject = solver(tmp_path, monkeypatch, monitoring={"diff": True})
    sandbox = SimpleNamespace(image="test", stop=lambda: None)
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: sandbox)

    class Agent:
        session_id = "session"

        def __init__(self, **kwargs):
            self.observe = kwargs["on_event"]
            self.workspace = kwargs["dir"]

        def execute(self, text):
            (self.workspace / "answer.txt").write_text(text)
            events = [AgentEvent(type="result", content=text)]
            for event in events:
                self.observe(event)
            return events

        def run(self, prompt):
            hidden = self.workspace / ".harness-home"
            hidden.mkdir(exist_ok=True)
            (hidden / "private").write_text("omit from source archive")
            return self.execute("first")

        def resume(self, *args, **kwargs):
            return self.execute("second")

        def get_tokens(self):
            return TokenUsage()

    monkeypatch.setattr(harness_solver, "Agent", Agent)
    response = subject._solve_one(0, "Problem", None, 1, 0)
    first = Path(response.history[0]["monitoring"]["directory"])
    assert response.history[0]["monitoring"]["complete"]
    assert not (first / "commands.jsonl").exists()
    response = subject.last_chance(response)
    second = Path(response.history[-1]["monitoring"]["directory"])
    assert first != second
    assert response.history[-1]["monitoring"]["complete"]
    assert json.loads((second / "summary.json").read_text())["source_history"]["snapshots"] == 2
    import tarfile
    for path in first.glob("sources/*.tar.gz"):
        with tarfile.open(path) as archive:
            assert all(".harness-home" not in name for name in archive.getnames())


@pytest.mark.parametrize("flag,expected", [("cpu", "cpu_percent"), ("ram", "memory_current_bytes")])
def test_resource_monitoring_without_preflight_or_source_scan(tmp_path, monkeypatch, flag, expected):
    subject = solver(tmp_path, monkeypatch, monitoring={flag: True})
    sampled = threading.Event()

    class Resources:
        def __init__(self, engine, name, *, cpu, ram):
            assert cpu == (flag == "cpu") and ram == (flag == "ram")

        def sample(self):
            sampled.set()
            return {"status": "sample", expected: 10}

    def run_model(*args, **kwargs):
        assert sampled.wait(2)
        from matharena.solvers import SolverResponse
        return SolverResponse(0, [], {}, history=[{}])

    monkeypatch.setattr(harness_solver, "ContainerResources", Resources)
    monkeypatch.setattr(subject, "_run_prepared", run_model)
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: SimpleNamespace(
        image="test", _relay_executable="engine", container_name="attempt", stop=lambda: None,
    ))
    response = subject._solve_one(0, "Problem", None, 1, 0)
    output = Path(response.history[0]["monitoring"]["directory"])
    assert response.history[0]["monitoring"]["complete"]
    assert not (output / "sources").exists()
    assert json.loads((output / "resources.jsonl").read_text().splitlines()[0])[expected] == 10


@pytest.mark.parametrize("flag", ["cpu", "ram"])
def test_counters_only_read_enabled_controller(tmp_path, flag):
    if flag == "cpu":
        (tmp_path / "cpu.stat").write_text("usage_usec 1\nuser_usec 1\nsystem_usec 0\n")
        (tmp_path / "cpu.max").write_text("max 100000")
    else:
        (tmp_path / "memory.stat").write_text("anon 1\nfile 2\n")
        (tmp_path / "memory.max").write_text("max")
        (tmp_path / "memory.current").write_text("3")
    sample = read_counters(tmp_path, cpu=flag == "cpu", ram=flag == "ram")
    assert ("cpu_usage_usec" in sample) == (flag == "cpu")
    assert ("memory_current_bytes" in sample) == (flag == "ram")


def test_command_readiness_requires_actual_events(tmp_path):
    with AttemptMonitor(tmp_path, diff=False, commands=False) as monitor:
        with pytest.raises(RuntimeError, match="telemetry preflight"):
            monitor.assert_ready(require_commands=True)


def test_shared_command_monitor_records_preflight_and_model_builds(tmp_path, monkeypatch):
    import time
    from matharena.monitoring.commands import CONTAINER_SOCKET, SOCKET_ENV, emit
    from matharena.solvers import SolverResponse

    subject = solver(tmp_path, monkeypatch, monitoring={"commands": True}, preflight="lake build", command_monitoring_backend="wrapper")
    sandbox = SimpleNamespace(image="traced", mounts=(), stop=lambda: None)

    def build(phase):
        address = str(sandbox.mounts[-1].source)
        common = {"span_id": phase, "kind": "lake", "argv": ["lake", "build"]}
        for boundary in ("start", "end"):
            event = {**common, "event": boundary, "monotonic_ns": time.monotonic_ns(),
                     "wall_time_ns": time.time_ns()}
            assert emit(address, event) is None

    def preflight(command, **kwargs):
        assert "command-trace-version" in command[2]
        assert kwargs["env"][SOCKET_ENV] == CONTAINER_SOCKET
        build("preflight")
        return SimpleNamespace(returncode=0, stdout="ready", stderr="")

    def run_model(*args, **kwargs):
        build("model")
        kwargs["after_run"]()
        return SolverResponse(0, [], {}, history=[{}])

    sandbox.run = preflight
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: sandbox)
    monkeypatch.setattr(subject, "_run_prepared", run_model)
    response = subject._solve_one(0, "Problem", None, 1, 0)
    output = Path(response.history[0]["monitoring"]["directory"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"]
    assert summary["totals"]["preflight"]["lake_build"]["commands"] == 1
    assert summary["totals"]["model"]["lake_build"]["commands"] == 1
    assert sandbox.mounts == ()
    assert not (output / "sources").exists()


def test_repo_eval_snapshots_sources_and_installs_dataset_helpers(tmp_path, monkeypatch):
    import tarfile

    source = tmp_path / "input-repository"
    source.mkdir()
    (source / "main.py").write_text("baseline")
    helpers = tmp_path / "dataset/helpers"
    helpers.mkdir(parents=True)
    helper = helpers / "check.sh"
    helper.write_text("#!/bin/sh\nexit 0\n")
    helper.chmod(0o755)
    subject = solver(tmp_path, monkeypatch, repo_eval=True, helper_files=str(helpers), keep_workspaces=False)
    from matharena.solvers import SolverResponse

    def run_model(*args, **kwargs):
        workspace = tmp_path / "attempt"
        assert (workspace / "check.sh").stat().st_mode & 0o111
        assert (workspace / "check.sh").read_text() == helper.read_text()
        (workspace / "repository/main.py").write_text("submitted")
        (workspace / "repository/.lake").mkdir()
        (workspace / "repository/.lake/cache").write_text("omit")
        return SolverResponse(0, [], {}, history=[{}])

    def stop():
        (tmp_path / "attempt/repository/main.py").write_text("settled")

    monkeypatch.setattr(subject, "_run_prepared", run_model)
    monkeypatch.setattr(subject, "_build_sandbox", lambda workspace: SimpleNamespace(stop=stop))
    response = subject._solve_one(0, {"repository_path": str(source), "problem": "Fix it"}, None, 1, 0)
    assert response.detailed_cost["deferred_grading"]
    assert response.history[0]["eval_type"] == "repo"
    assert not (tmp_path / "attempt").exists()
    snapshot = response.history[0]["repo_submission"]
    with tarfile.open(snapshot["archive"]) as archive:
        assert archive.getnames() == ["main.py"]
        assert archive.extractfile("main.py").read() == b"settled"
    assert (source / "main.py").read_text() == "baseline"


@pytest.mark.parametrize("legacy", [False, True])
def test_repo_eval_flag_selects_leanlean_adapter_by_dataset(legacy):
    from matharena.runner import Runner
    from leanlean.solver import LeanLeanSolver

    config = {"model": "test", "harness": "codex", "dataset_format": "leanlean",
              "harness_config": {"managed_cli": False, "leanlean_projects" if legacy else "repo_eval": True}}
    subject = Runner._initialize_solver(SimpleNamespace(container_runtime="docker"),
                                        {"harness": "codex"}, "{problem}", config, "")
    assert isinstance(subject, LeanLeanSolver)
    assert subject.repo_eval
