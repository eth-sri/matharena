"""Judge comparison replays archived inputs and leaves canonical grades untouched."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from harness_wrapper.models.oauth import openai_bridge

from matharena.json_zst import dump_json_zst, load_json_zst
from matharena.solvers.judges.judge_response import JudgeResponse
from matharena.solvers.judges.simple_judge import SimpleJudge

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("judge_comparison", ROOT / "scripts/judge/compare_judges.py")
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


@pytest.mark.parametrize("judge_ref, tools, scale, threads", [
    ("judges/answer_judge", False, 1, 16),
    ("judges/arxiv_judge_gemini_38_flash", True, 3, 64),
])
@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_comparison_keeps_codex_and_original_tool_policy(monkeypatch, judge_ref, tools, scale, threads, runtime):
    captured = []
    def capture(*args, **kwargs):
        assert kwargs == {"container_runtime": runtime}
        captured.append(args)
        return object()
    monkeypatch.setattr("matharena.solvers.judges.simple_judge.HarnessSolver", capture)
    config = comparison.comparison_config(judge_ref)
    judge = SimpleJudge(0, 1, 0, config, container_runtime=runtime)
    assert judge.harness_solver is not None
    model = captured[0][2]
    assert model["harness"] == "codex"
    assert model["reasoning_effort"] == "low"
    assert model["harness_config"]["tools_enabled"] is tools
    assert config["judge_points_max"] == scale
    assert config["n_threads"] == threads
    prompt = "Archived rubric, not today's rubric. Math: \\frac{a}{b}. {student_answer}"
    assert config["judge_prompt"].format(student_answer=prompt) == prompt


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_comparison_preserves_sources_replays_prompts_and_resumes(tmp_path, monkeypatch, runtime):
    (tmp_path / "configs").symlink_to(ROOT / "configs", target_is_directory=True)
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    originals = {}
    archived = {}
    for comp, baseline_points in (("arxiv/august", [0, 1, 1]), ("arxiv_false/august", [0, 1, 3])):
        judge_ref, = comparison.read_config(f"competitions/{comp}")["judge_configs"]
        maximum = comparison.read_config(judge_ref)["judge_points_max"]
        path = tmp_path / "outputs" / comp / "model/test/1.json.zst"
        path.parent.mkdir(parents=True)
        prompts = [f"Archived {comp} prompt {i}, with {{literal}} braces" for i in range(3)]
        archived[maximum] = prompts
        entries = [{
            "points": score, "max_points": maximum, "judge_id": judge_ref,
            "history": [{"messages": [{"role": "user", "content": prompt}]}],
            "details": [{"desc": "BASELINE_ASSESSMENT_MUST_NOT_BE_SENT"}],
        } for score, prompt in zip(baseline_points, prompts)]
        entries[1] = {"judge_id": judge_ref, "points": 0, "max_points": maximum,
                      "original_judgment": entries[1]}
        dump_json_zst({"idx": 1, "messages": [[], [], [], []], "judgment": [entries + [None]],
                       "correct": baseline_points + ["TODO Grading"]}, path)
        originals[path] = path.read_bytes()

    calls, pacers = [], []
    class FakePool:
        def __init__(self, solver_config, *, container_runtime):
            self.config = solver_config
            assert container_runtime == runtime

        def solve_batch(self, queries, problem_indices, run_indices):
            assert openai_bridge._wait_for_connection is not None
            pacers.append(openai_bridge._wait_for_connection)
            scale = self.config["judge_points_max"]
            calls.append(scale)
            assert [payload["student_answer"] for _, payload in queries] == archived[scale]
            assert list(problem_indices.values()) == [1, 1, 1]
            assert list(run_indices.values()) == [0, 1, 2]
            scores = [0, 1, 0] if scale == 1 else [0, 2, 3]
            for i, score in enumerate(scores):
                yield JudgeResponse(i, score, "Astra assessment", {"cost": 0.1}, [], {})

    monkeypatch.setattr(comparison, "JudgePool", FakePool)
    output = tmp_path / "logs/comparison"
    monkeypatch.setattr(sys, "argv", [
        "compare_judges.py", "--run", "--output-dir", str(output), "--container-runtime", runtime,
    ])
    comparison.main()
    comparison.main()
    assert calls == [1, 3]  # The resumed run makes no further judge requests.
    assert pacers[0] is pacers[1]  # One shared limit across both benchmarks.
    assert openai_bridge._wait_for_connection is None
    assert all(path.read_bytes() == raw for path, raw in originals.items())
    summary = json.loads((output / "summary.json").read_text())
    assert summary["arxiv/august"]["agreement_percent"] == pytest.approx(200 / 3)
    assert summary["arxiv_false/august"]["agreement_percent"] == pytest.approx(200 / 3)
    assert summary["arxiv_false/august"]["confusion_matrix"][1][2] == 1
    assert summary["arxiv_false/august"]["mean_score_change_pp"] == pytest.approx(100 / 9)
    saved = load_json_zst(output / "arxiv_false/august/model/test/1_r2.json.zst")
    assert saved["baseline"]["points"] == 1 and saved["candidate"]["points"] == 2

    path = next(iter(originals))
    changed = load_json_zst(path)
    changed["judgment"][0][0]["history"][0]["messages"][0]["content"] += " Changed"
    dump_json_zst(changed, path)
    with pytest.raises(ValueError, match="Inputs/config changed"):
        comparison.main()
    assert calls == [1, 3]


def test_comparison_refuses_canonical_output_destination(tmp_path, monkeypatch):
    monkeypatch.setattr(comparison, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare_judges.py", "--run", "--output-dir", str(tmp_path / "outputs")])
    with pytest.raises(SystemExit):
        comparison.main()
