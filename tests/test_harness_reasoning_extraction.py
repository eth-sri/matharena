"""Reasoning extraction refusals are paid, completed incorrect attempts (offline)."""

from types import SimpleNamespace

import pytest

from harness_wrapper import AgentEvent
from matharena.json_zst import load_json_zst
from matharena.solvers.harness_solver import REASONING_EXTRACTION_MESSAGE
from matharena.runner import Runner
from matharena.runs import Runs
from matharena.solvers import SolverResponse
from test_harness_completion import make_solver


def refusal_events():
    return [
        AgentEvent(type="reasoning", content="Work before the cutoff"),
        AgentEvent(type="result", content="", raw={"refusal_category": "reasoning_extraction"}),
    ]


def forbidden(*args, **kwargs):
    raise AssertionError("A refused attempt must not be continued or graded")


@pytest.mark.parametrize("last_chance", [False, True])
def test_harness_saves_explanation_and_usage_without_continuing(tmp_path, monkeypatch, last_chance):
    solver = make_solver(tmp_path, monkeypatch, [] if last_chance else refusal_events(), harness="claude")
    monkeypatch.setattr(solver, "_detailed_cost", lambda agent: {"output_tokens": 128025, "cost": 2.56})
    if last_chance:
        response = SolverResponse(0, [{"role": "assistant", "content": "partial"}],
                                  {"time": 1, "request_time": 1}, [])
        solver._response_agents[id(response)] = SimpleNamespace(
            session_id="session", resume=lambda *a, **k: refusal_events(),
        )
        response = solver.last_chance(response)
    else:
        response = solver._solve_one(0, "Problem", None, 4, 0)
    assert response.detailed_cost["refusal_category"] == "reasoning_extraction"
    assert response.conversation[-1]["content"] == REASONING_EXTRACTION_MESSAGE
    assert response.history[-1]["messages"][-1]["content"] == REASONING_EXTRACTION_MESSAGE
    assert response.detailed_cost["output_tokens"] == 128025
    assert response.detailed_cost["cost"] == 2.56
    assert not response.history[-1].get("empty_completion")
    assert solver.last_chance(response) is response


def test_refusal_does_not_start_cost_limit_grace(tmp_path, monkeypatch):
    solver = make_solver(tmp_path, monkeypatch, [], harness="claude")
    monkeypatch.setattr(solver, "_stream_with_budget", lambda *a, **k: refusal_events())
    budget = SimpleNamespace(exceeded=lambda: True, begin_cost_grace=forbidden)
    events = solver._run_with_budget(SimpleNamespace(session_id="session"), None, budget, "Problem")
    assert events[-1].raw["refusal_category"] == "reasoning_extraction"


@pytest.mark.parametrize("grading", ["fa", "answer_judge", "manual"])
@pytest.mark.parametrize("last_chance", [False, True])
def test_runner_persists_incorrect_skips_judge_and_continues_batch(tmp_path, monkeypatch, grading, last_chance):
    if last_chance and grading != "fa":
        pytest.skip("Only final-answer parsing has a last-chance call")
    runner = Runner.__new__(Runner)
    runner.comp_name = "test"; runner.runs_per_problem = 1
    runner.is_fa_comp = grading != "manual"
    runner.is_auto_graded_comp = grading == "fa"
    runner.options = None
    runner.competition_config = {"grading": "answer_judge"} if grading == "answer_judge" else {}
    runner.problems = [{"problem_idx": i, "problem": "Problem", "answer": "42"} for i in (1, 2)]
    all_runs = {p["problem_idx"]: Runs("test", True, "opus", "pure_model", p, str(tmp_path))
                for p in runner.problems}
    refused = SolverResponse(0, [{"role": "user", "content": "Problem"},
                                     {"role": "assistant", "type": "response", "content": REASONING_EXTRACTION_MESSAGE}],
                             {"refusal_category": "reasoning_extraction", "output_tokens": 128025,
                              "input_tokens": 100, "cost": 2.56}, None)
    healthy = SolverResponse(1, [{"role": "assistant", "content": r"\boxed{42}"}],
                             {"output_tokens": 5, "input_tokens": 100, "cost": .01}, None)
    initial = SolverResponse(0, [{"role": "assistant", "content": "partial"}], {}, None)
    continuations = []
    def reprompt(response):
        assert last_chance and response is initial
        continuations.append(response)
        return refused
    def grade(conversation, *args, **kwargs):
        assert conversation[-1]["content"] == r"\boxed{42}"
        return "42", True, 0
    monkeypatch.setattr("matharena.runner.extract_and_grade", grade)
    monkeypatch.setattr(runner, "_update_status", lambda *args: None)
    runner.process_solver_responses("opus", SimpleNamespace(last_chance=reprompt), all_runs,
                                    {0: 1, 1: 2}, "unused", [initial if last_chance else refused, healthy],
                                    print_final_status=False)
    saved = load_json_zst(all_runs[1].path)
    assert saved["N"] == 1 and saved["correct"] == [False]
    assert saved["messages"][0][-1]["content"] == REASONING_EXTRACTION_MESSAGE
    assert saved["detailed_costs"][0]["output_tokens"] == 128025
    assert all_runs[2].N == 1
    assert len(continuations) == int(last_chance)
    # Subsequent regrading cannot overwrite this terminal scoring policy.
    all_runs[1].update_run_grading(0, ("42", True, 0))
    assert all_runs[1].correct == [False]


def test_answer_judge_recomputation_keeps_refusal_incorrect():
    from test_answer_judging import JUDGE
    data = {"messages": [[{"role": "assistant", "content": REASONING_EXTRACTION_MESSAGE}]],
            "detailed_costs": [{"refusal_category": "reasoning_extraction"}],
            "judgment": [[{"points": 1, "max_points": 1}]]}
    JUDGE.recompute_correct_and_pass_at_1(data)
    assert data["correct"] == [0] and data["pass_at_1"] == 0
