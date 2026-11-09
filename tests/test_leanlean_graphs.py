import json
import shutil

import pytest

from leanlean.common import sha256, write_json
from leanlean.compression_origins import REFERENCE, check_reference, graph_position_audit, summarize
from leanlean.graphs import extract
from leanlean.graphs_container import source_hashes


def test_source_guard_ignores_generated_state_but_detects_source_changes(tmp_path):
    source = tmp_path / "Solution.lean"
    source.write_text("theorem x : True := by trivial")
    initial = source_hashes(tmp_path)
    for name in (".lake/build/X.olean", ".lean-strip/work/Solution.lean"):
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_text("generated")
    assert source_hashes(tmp_path) == initial
    source.write_text("theorem x : True := by sorry")
    assert source_hashes(tmp_path) != initial


def test_resume_rejects_graph_tampering(tmp_path):
    workspace = tmp_path / "repo" / "before"
    workspace.mkdir(parents=True)
    graph = workspace / "dependency-graph.json"
    graph.write_text("original")
    provenance = {"version": "1.1.2"}
    write_json(
        tmp_path / "P1-before.json",
        {
            "status": "passed",
            "provenance": provenance,
            "graph_sha256": sha256(graph),
        },
    )
    job = {"problem_idx": 1, "directory": str(workspace.parent)}
    graph.write_text("changed")
    with pytest.raises(ValueError, match="Graph or provenance changed"):
        extract(job, "before", {}, tmp_path, tmp_path, provenance)


def test_origin_summary_keeps_failed_projects_out_of_partial_means(tmp_path):
    rows = [
        {
            "status": "passed",
            "problem_idx": 1,
            "key": "one",
            "baseline_tokens": 100,
            "final_tokens": 95,
            "tokens_saved": 5,
            "categories": {"automation": 8, "added": -3},
        },
        {"status": "failed", "problem_idx": 2, "key": "two", "error": "missing graph"},
    ]
    summarize(rows, 2, tmp_path)
    result = json.loads((tmp_path / "summary.json").read_text())
    assert result["passed"] == 1
    assert result["completed"] == result["total"] == 2
    assert result["mean_percentage_points"] == {"automation": 8, "added": -3}
    assert len((tmp_path / "origins.csv").read_text().splitlines()) == 2


def test_origin_analysis_rejects_line_shifting_normalization(tmp_path):
    report = {
        "engine_report": {
            "grind_normalization": {
                "files": [
                    {
                        "path": "Solution.lean",
                        "calls": [
                            {
                                "persisted_after_file_decision": True,
                                "normalization": {
                                    "original_call": "grind +suggestions",
                                    "replacement": "grind [a,\nb]",
                                },
                            }
                        ],
                    }
                ]
            }
        }
    }
    write_json(tmp_path / "before/repository/.lean-strip/report.json", report)
    with pytest.raises(ValueError, match="shifts source lines"):
        graph_position_audit(tmp_path, {"project_path": ""})


def test_vendored_classifier_matches_its_manifest(tmp_path):
    assert check_reference(REFERENCE)["head"]
    copy = tmp_path / "lean-lean"
    shutil.copytree(REFERENCE, copy)
    (copy / "scripts/analysis/ledger.py").write_text("# edited\n")
    with pytest.raises(ValueError, match="differ"):
        check_reference(copy)
    (copy / "scripts/analysis/ledger.py").unlink()
    with pytest.raises(ValueError, match="differ"):
        check_reference(copy)
