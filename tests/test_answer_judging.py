import csv
import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from matharena.grader import extract_and_grade
from matharena.json_zst import load_json_zst
from matharena.runner import Runner
from matharena.solvers import SolverResponse
from matharena.solvers.harness_solver import HarnessSolver
from matharena.solvers.judges.simple_judge import SimpleJudge
from matharena.utils import normalize_conversation


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("answer_judge_cli", ROOT / "scripts/judge/judge.py")
JUDGE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGE)


def judge_config():
    reference = yaml.safe_load((ROOT / "configs/judges/answer_judge.yaml").read_text())
    config = yaml.safe_load((ROOT / "configs" / f"{reference['scaffold_config']}.yaml").read_text())
    config["model_config"] = yaml.safe_load(
        (ROOT / "configs/models" / f"{reference['model_config']}.yaml").read_text()
    )
    config.update(reference["override"])
    return config


def write_answer_fixture(directory, n=1):
    (directory / "problems").mkdir(parents=True)
    with (directory / "answers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"])
        writer.writeheader()
        for index in range(1, n + 1):
            writer.writerow({"id": index, "answer": rf"\frac{{{index}}}{{2}}"})
            (directory / "problems" / f"{index}.tex").write_text(f"What is {index} divided by two?")
    return directory


def make_runner(tmp_path, monkeypatch):
    config = yaml.safe_load((ROOT / "configs/competitions/arxiv/august.yaml").read_text())
    # Keep this lifecycle test offline and independent of published benchmark data.
    config["dataset_path"] = str(write_answer_fixture(tmp_path / "data"))
    comp_dir = tmp_path / "competitions/arxiv"
    comp_dir.mkdir(parents=True)
    (comp_dir / "august.yaml").write_text(yaml.safe_dump(config))
    monkeypatch.chdir(tmp_path)
    return Runner(
        "arxiv/august", 1, [1], str(comp_dir.parent), str(ROOT / "configs/models"),
        str(tmp_path / "outputs"), False,
    )


class FakeJudgeAPI:
    calls = []
    verdict = "<points>1</points><assessment>Equivalent final answers.</assessment>"

    def __init__(self, **kwargs):
        assert kwargs["model"] == "gemini-3.8-flash"
        assert kwargs["api"] == "google"
        assert kwargs["tools"] == [] and kwargs["max_tool_calls"] == 0
        assert not {"harness", "harness_version", "harness_config", "tool_choice"} & kwargs.keys()

    def run_queries(self, queries, **kwargs):
        self.calls.extend(queries)
        for idx, query in enumerate(queries):
            yield idx, query + [{"role": "assistant", "content": self.verdict}], {
                "cost": 0.001, "input_tokens": 10, "output_tokens": 5,
            }


@pytest.mark.parametrize("points", ["0", "1", "2", "7", "-1", "0.5", "one"])
def test_answer_judge_accepts_only_binary_scores(monkeypatch, points):
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", FakeJudgeAPI)
    monkeypatch.setattr(FakeJudgeAPI, "verdict", f"<points>{points}</points><assessment>Test.</assessment>")
    config = judge_config()
    before = deepcopy(config)
    response = SimpleJudge(0, 1, 0, config).solve("Question?", "", ["reference"], "candidate")
    assert response.points == (int(points) if points in {"0", "1"} else None)
    assert response.history[0]["step"] == "judge"
    assert config == before


def test_answer_judge_rejects_multiple_scores(monkeypatch):
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", FakeJudgeAPI)
    monkeypatch.setattr(FakeJudgeAPI, "verdict", "<points>1</points><points>0</points>")
    assert SimpleJudge(0, 1, 0, judge_config()).solve("Q", "", ["A"], "B").points is None


def test_answer_judge_prompt_preserves_literal_latex_braces(monkeypatch):
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", FakeJudgeAPI)
    monkeypatch.setattr(FakeJudgeAPI, "calls", [])
    monkeypatch.setattr(FakeJudgeAPI, "verdict", "<points>1</points><assessment>Equivalent.</assessment>")
    reference = r"\frac{1}{2}"
    candidate = r"\boxed{\frac{1}{2}}"
    result = SimpleJudge(0, 1, 0, judge_config()).solve("Q", "", [reference], candidate)
    assert result.points == 1
    prompt = FakeJudgeAPI.calls[0][0]["content"]
    assert r"\boxed{}" in prompt
    assert reference in prompt and candidate in prompt
    assert "{student_answer}" not in prompt


def test_answer_references_are_loaded_as_text(tmp_path):
    dataset = write_answer_fixture(tmp_path / "data", n=3)
    grading_map = JUDGE.load_grading_map(str(dataset), answer_judging=True)
    with (dataset / "answers.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(grading_map) == 3
    for row in rows:
        assert grading_map[int(row["id"])]["ground_truth_solutions"] == [row["answer"]]
        assert grading_map[int(row["id"])]["points"] == 1


def test_parser_cannot_grade_answer_judged_competition():
    with pytest.raises(ValueError, match="scripts/judge/judge.py"):
        extract_and_grade(
            [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "1/2"}],
            3, "1/2", {"grading": "answer_judge"},
        )


@pytest.mark.parametrize("invalid_first_verdict", [False, True])
def test_august_generation_and_judge_cli_end_to_end(tmp_path, monkeypatch, invalid_first_verdict):
    runner = make_runner(tmp_path, monkeypatch)
    assert runner.is_fa_comp and runner.uses_answer_judge
    assert not runner.is_auto_graded_comp
    prepared = runner.prepare_run("openai/gpt-6-astra", set_request_metadata=False)
    assert isinstance(prepared["solver"], HarnessSolver)
    assert prepared["solver"].tools_enabled
    assert prepared["solver"]._harness_cli is None
    assert prepared["batch"][0][0] == {"problem": runner.problems[0]["problem"]}
    assert "sage -python" in prepared["solver"].build_prompt(prepared["batch"][0][0])

    def forbidden(*args, **kwargs):
        raise AssertionError("No parser or solver reformatting call is allowed")

    monkeypatch.setattr("matharena.runner.extract_answer", forbidden)
    monkeypatch.setattr("matharena.runner.extract_and_grade", forbidden)
    monkeypatch.setattr(prepared["solver"], "last_chance", forbidden)
    conversation = [
        {"role": "user", "content": runner.problems[0]["problem"]},
        {"role": "assistant", "content": "My final answer is a non-boxed mathematical expression."},
    ]
    response = SolverResponse(
        0, conversation, {"cost": 0.01, "input_tokens": 100, "output_tokens": 20, "time": 1},
        [{"step": "harness", "timestep": 0, "messages": deepcopy(conversation)}],
    )
    runner.process_solver_responses(
        prepared["solver_name"], prepared["solver"], prepared["all_runs"],
        prepared["batch_idx_to_problem_idx"], prepared["status_path"], [response],
        print_final_status=False,
    )
    result_path = tmp_path / "outputs/arxiv/august/openai/gpt-6-astra/1.json.zst"
    pending = load_json_zst(result_path)
    assert pending["N"] == 1 and pending["correct"] == ["TODO Grading"]
    assert pending["gold_answer"] == runner.problems[0]["answer"]
    assert pending["messages"][0] == normalize_conversation(conversation)

    monkeypatch.setattr("matharena.solvers.judges.simple_judge.APIClient", FakeJudgeAPI)
    monkeypatch.setattr(FakeJudgeAPI, "calls", [])
    monkeypatch.setattr("sys.argv", [
        "judge.py", "--comp", "arxiv/august", "--models", "openai/gpt-6-astra",
        "--output-dir", str(tmp_path / "outputs"), "--comp-configs-dir", runner.comp_configs_dir,
        "--configs-dir", str(ROOT / "configs"), "--model-configs-dir", str(ROOT / "configs/models"),
    ])
    expected_calls = 1
    if invalid_first_verdict:
        monkeypatch.setattr(FakeJudgeAPI, "verdict", "<points>2</points><assessment>Invalid score.</assessment>")
        JUDGE.main()
        assert load_json_zst(result_path) == pending
        monkeypatch.setattr(FakeJudgeAPI, "verdict", "<points>1</points><assessment>Equivalent final answers.</assessment>")
        assert len(FakeJudgeAPI.calls) == 3  # Exhaust bounded automatic retries.
        expected_calls += 3
    JUDGE.main()
    judged = load_json_zst(result_path)
    assert judged["correct"] == [1.0] and judged["pass_at_1"] == 1.0
    assert judged["answers"] == [conversation[-1]["content"]]
    assert judged["messages"] == pending["messages"]
    assert judged["cost"] == pending["cost"]  # Judge costs are accounted separately.
    entry = judged["judgment"][0][0]
    assert entry["judge_id"] == "judges/answer_judge"
    assert entry["max_points"] == 1 and entry["cost"]["cost"] == 0.001
    assert len(entry["history"]) == 1
    prompt = FakeJudgeAPI.calls[0][0]["content"]
    assert runner.problems[0]["answer"] in prompt and conversation[-1]["content"] in prompt
    JUDGE.main()
    assert len(FakeJudgeAPI.calls) == expected_calls  # Only pending judgments are retried.


def test_invalid_judgment_is_not_recorded():
    from matharena.solvers.judges.judge_response import JudgeResponse
    for points in (None, -1, 2):
        response = JudgeResponse(0, points, "invalid", {}, [])
        assert JUDGE.build_judgment_entry(response, 1, "reference", "judges/answer_judge", 1) is None


def test_budget_failure_is_saved_incorrect_and_skipped_by_judge(tmp_path, monkeypatch):
    runner = make_runner(tmp_path, monkeypatch)
    prepared = runner.prepare_run("openai/gpt-6-astra", set_request_metadata=False)
    conversation = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "type": "response", "content": "42"},
    ]
    response = SolverResponse(
        0, conversation,
        {"cost": 101, "output_tokens": 2000000, "run_limits": {"exceeded": "cost_limit"}},
        [{"step": "harness", "timestep": 0, "messages": deepcopy(conversation)}],
    )
    runner.process_solver_responses(
        prepared["solver_name"], prepared["solver"], prepared["all_runs"],
        prepared["batch_idx_to_problem_idx"], prepared["status_path"], [response],
        print_final_status=False,
    )
    path = tmp_path / "outputs/arxiv/august/openai/gpt-6-astra/1.json.zst"
    assert load_json_zst(path)["correct"] == [False]

    def forbidden(*args, **kwargs):
        raise AssertionError("Budget failures must not generate judge requests")

    monkeypatch.setattr(JUDGE.JudgePool, "solve_batch", forbidden)
    monkeypatch.setattr("sys.argv", [
        "judge.py", "--comp", "arxiv/august", "--models", "openai/gpt-6-astra", "--redo",
        "--output-dir", str(tmp_path / "outputs"), "--comp-configs-dir", runner.comp_configs_dir,
        "--configs-dir", str(ROOT / "configs"), "--model-configs-dir", str(ROOT / "configs/models"),
    ])
    JUDGE.main()
    saved = load_json_zst(path)
    assert saved["correct"] == [0] and saved["pass_at_1"] == 0
    assert saved["cost"]["cost"] == 101
    assert saved["messages"][0][-1]["content"] == "42"
