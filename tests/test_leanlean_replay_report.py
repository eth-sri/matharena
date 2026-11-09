import pytest

from leanlean.common import sha256, write_json
from leanlean.replay_report import graph_availability, owners, report, timing


def plan(root, checkpoints="capture"):
    state = dict(source_sha256="source", archive_sha256="archive")
    record = dict(
        job=dict(problem_idx=1, key="repo", metrics={"before": {"lean_tokens": 100}}),
        states=[state],
        frames=[dict(source_sha256="source"), dict(source_sha256="source")],
        checkpoints_sha256=checkpoints,
        baseline_sources={},
    )
    p = {
        name: "same"
        for name in (
            "postprocess_sha256",
            "runtime",
            "extractor_sha256",
            "extractor_commit",
            "tokens_sha256",
            "pipeline_sha256",
        )
    }
    write_json(root / "plan.json", {**p, "records": [record]})
    return state


def test_later_owner_deduplicates_states_but_preserves_checkpoint_timeline(tmp_path):
    first, second, output = (tmp_path / name for name in ("first", "second", "output"))
    plan(first)
    state = plan(second)
    write_json(
        second / "repositories/repo/states/source/result.json",
        {
            **state,
            "status": "complete",
            "verified": True,
            "metrics": {"compression_pct": 25},
            "stages": {"build": {"started_wall_ns": 1_000_000_000, "ended_wall_ns": 4_000_000_000}},
        },
    )
    result = report([first, second], output)
    assert result["total_states"] == result["completed"] == result["verified_states"] == 1
    assert result["total_checkpoints"] == 2
    assert result["projects"][0]["run"] == str(second)
    assert result["timings"]["build"]["total_seconds"] == 3


def test_rejects_different_captures_for_overlapping_project(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    plan(first)
    plan(second, "other")
    with pytest.raises(ValueError, match="Different captured inputs"):
        owners([first, second])


def test_pending_states_are_not_counted_as_completed(tmp_path):
    root = tmp_path / "run"
    plan(root)
    result = report([root], tmp_path / "out")
    assert result["status"] == "running"
    assert result["completed"] == 0
    assert timing([])["mean_seconds"] is None


def test_graph_failure_reason_requires_matching_compiler_evidence(tmp_path):
    attempt = tmp_path / "attempts/0001"
    attempt.mkdir(parents=True)
    write_json(attempt / "report.json", {"stage": "repository_build"})
    log = attempt / "comparator.stdout.log"
    log.write_text("error: Solution.lean:9:22: unexpected token ':='; expected ':'\n")
    value = dict(
        status="graph_failed",
        stages={"comparator": {"returncode": 1}},
        artifacts={str(p.relative_to(tmp_path)): sha256(p) for p in attempt.iterdir()},
    )
    reason = graph_availability(value, tmp_path / "result.json")
    assert reason["status"] == "solution_build_failed"
    assert len(reason["compiler_errors"]) == 1
    value["stages"]["comparator"]["returncode"] = 124
    assert graph_availability(value, tmp_path / "result.json")["status"] == "extraction_failed"
    log.write_text("different error")
    with pytest.raises(ValueError, match="evidence changed"):
        graph_availability(value, tmp_path / "result.json")
