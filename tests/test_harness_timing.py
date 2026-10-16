"""Deterministic duration accounting without running CLIs or model requests."""

from types import SimpleNamespace

import pytest

from harness_wrapper import AgentEvent, TokenUsage
from matharena.json_zst import load_json_zst
from matharena.runs import Runs
from matharena.solvers import harness_solver


class Clock:
    now = 100.0

    def monotonic(self):
        return self.now


def make_solver(tmp_path, monkeypatch, harness="codex", tools_enabled=True):
    clock = Clock()
    monkeypatch.setattr(harness_solver, "time", clock)
    config = {
        "model": "test-model",
        "api": "openai",
        "harness": harness,
        "allow_harness": tools_enabled,
        "read_cost": 10,
        "cache_read_cost": 1,
        "write_cost": 50,
        "harness_config": {
            "managed_cli": False,
            "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
        },
    }
    solver = harness_solver.HarnessSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None},
        "{problem}",
        config,
        "Format the answer",
    )

    def run(prompt):
        # The timer includes everything inside the harness: tools and retries too.
        clock.now += 7.25
        return [AgentEvent(type="result", content="42")]

    def resume(prompt, *, session_id):
        assert session_id == "timed-session"
        clock.now += 2.5
        return [AgentEvent(type="result", content="42")]

    agent = SimpleNamespace(
        run=run,
        resume=resume,
        session_id="timed-session",
        get_tokens=lambda: TokenUsage(
            input_tokens=100, cache_read_tokens=40, output_tokens=20
        ),
    )

    def build_model():
        clock.now += 500  # Setup must not inflate the duration.
        return None

    def check(*args, **kwargs):
        clock.now += 300  # Post-run validation must not inflate it either.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(harness_solver, "Agent", lambda **kwargs: agent)
    monkeypatch.setattr(solver, "_build_model", build_model)
    monkeypatch.setattr(
        solver, "_build_sandbox", lambda workspace: SimpleNamespace(run=check)
    )
    return solver, clock


@pytest.mark.parametrize(
    "harness", ["codex", "claude", "kimi", "qwen", "gravity", "opencode"]
)
def test_harnesses_record_elapsed_seconds(tmp_path, monkeypatch, harness):
    solver, _ = make_solver(tmp_path, monkeypatch, harness)
    response = solver._solve_one(0, "Problem", None, 1, 0)
    assert response.detailed_cost["time"] == 7.25
    assert response.detailed_cost["request_time"] == 7.25
    assert response.history[0]["elapsed_s"] == 7.25


def test_tool_free_codex_is_also_timed(tmp_path, monkeypatch):
    solver, _ = make_solver(tmp_path, monkeypatch, tools_enabled=False)
    response = solver._solve_one(0, "Problem", None, 1, 0)
    assert response.detailed_cost["time"] == 7.25


def test_post_run_validation_is_not_harness_time(tmp_path, monkeypatch):
    solver, _ = make_solver(tmp_path, monkeypatch)
    solver.run_check_after_harness = True
    response = solver._solve_one(0, "Problem", None, 1, 0)
    assert response.detailed_cost["time"] == 7.25
    assert response.detailed_cost["request_time"] == 7.25
    assert response.history[0]["check_returncode"] == 0


def test_repeated_continuations_accumulate_without_gaps_or_double_counting(
    tmp_path, monkeypatch
):
    solver, clock = make_solver(tmp_path, monkeypatch)
    response = solver._solve_one(0, "Problem", None, 1, 0)
    cost = response.detailed_cost["cost"]
    for expected in (9.75, 12.25):
        clock.now += 1000  # Time between invocations is not model runtime.
        assert solver.last_chance(response) is response
        assert response.detailed_cost["time"] == expected
        assert response.detailed_cost["request_time"] == expected
        assert response.detailed_cost["cost"] == cost
        assert response.detailed_cost["input_tokens"] == 100
    assert [step["elapsed_s"] for step in response.history] == [7.25, 2.5, 2.5]


def test_resume_does_not_make_up_unknown_original_time(tmp_path, monkeypatch):
    solver, _ = make_solver(tmp_path, monkeypatch)
    response = solver._solve_one(0, "Problem", None, 1, 0)
    response.detailed_cost.pop("time")
    response.detailed_cost["request_time"] = None
    solver.last_chance(response)
    assert response.detailed_cost["time"] is None
    assert response.detailed_cost["request_time"] is None
    assert response.history[-1]["elapsed_s"] == 2.5


def test_independent_attempts_serialize_and_aggregate_time(tmp_path, monkeypatch):
    solver, clock = make_solver(tmp_path, monkeypatch)
    first = solver._solve_one(0, "Problem", None, 1, 0)
    solver.last_chance(first)
    clock.now += 2000
    second = solver._solve_one(1, "Problem", None, 1, 1)
    assert second.detailed_cost["time"] == 7.25
    runs = Runs(
        "test",
        True,
        "test-model",
        "agent",
        {
            "problem_idx": 1,
            "problem": "Problem",
            "answer": "42",
        },
        str(tmp_path),
    )
    for response in (first, second):
        runs.add_run(response, ("42", True, 0))
    runs.save_to_file()
    saved = load_json_zst(runs.path)
    assert saved["cost"]["time"] == 17.0
    assert saved["cost"]["request_time"] == 17.0
    assert [cost["time"] for cost in saved["detailed_costs"]] == [9.75, 7.25]
    runs.load_from_file()
    assert runs.N == 2
    assert runs.cost["time"] == 17.0


def test_recovery_counts_accumulate_across_continuations(tmp_path, monkeypatch):
    solver, _ = make_solver(tmp_path, monkeypatch)
    agent = harness_solver.Agent()
    notices = [
        AgentEvent(type="recovery", content={"reason": "provider_stream_error"}),
        AgentEvent(type="recovery", content={"reason": "error_retry", "attempt": 1}),
        AgentEvent(type="session", session_id="timed-session"),
        AgentEvent(type="recovery", content={"reason": "error_retry", "attempt": 2}),
        AgentEvent(type="result", content="42"),
    ]
    agent.run = lambda prompt: notices
    agent.resume = lambda prompt, **kwargs: [
        AgentEvent(type="recovery", content={"reason": "error_retry", "attempt": 1}),
        AgentEvent(type="result", content="42"),
    ]
    response = solver._solve_one(0, "Problem", None, 1, 0)
    assert response.detailed_cost["n_retries"] == 2
    initial_cost = response.detailed_cost["cost"]
    for expected in (3, 4):
        solver.last_chance(response)
        assert response.detailed_cost["n_retries"] == expected
        assert response.detailed_cost["cost"] == initial_cost
    agent.run = lambda prompt: [AgentEvent(type="result", content="42")]
    second = solver._solve_one(1, "Problem", None, 1, 1)
    assert second.detailed_cost["n_retries"] == 0
