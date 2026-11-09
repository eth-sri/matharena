import importlib
import json
from pathlib import Path

import pytest


@pytest.mark.parametrize("module", ["Solution", "Solution.R05DerivativeExtension"])
def test_default_build_success_does_not_hide_broken_solution(tmp_path, monkeypatch, module):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "src/leanlean"))
    runner = importlib.import_module("leanlean.replay_container")
    work, output = tmp_path / "work", tmp_path / "output"
    repo = work / "repository"
    repo.mkdir(parents=True)
    output.mkdir()
    (repo / "comparator.json").write_text(json.dumps({"solution_module": module}))
    (work / "leanlean-project.json").write_text(json.dumps({"project_path": "", "comparator_path": "comparator.json"}))
    calls = []

    def stage(out, name, command, project, timeout):
        calls.append(command)
        return {"returncode": 0 if command == ["lake", "build"] else 1}

    monkeypatch.setattr(runner, "stage", stage)
    result = runner.check_checkpoint(work, output)
    assert calls == [["lake", "build", f"+{module}"]]
    assert result["status"] == "build_failed"
    assert result["graph_unavailable_reason"] == "solution_build_failed"
    assert result["verified"] is False
    assert result["sources_unchanged"] is True


def test_extractor_failure_after_successful_verification_stays_a_graph_failure(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "src/leanlean"))
    runner = importlib.import_module("leanlean.replay_container")
    work, output = tmp_path / "work", tmp_path / "output"
    repo = work / "repository"
    repo.mkdir(parents=True)
    output.mkdir()
    (repo / "comparator.json").write_text('{"solution_module": "Solution"}')
    (work / "leanlean-project.json").write_text(json.dumps({"project_path": "", "comparator_path": "comparator.json"}))
    calls = []

    def stage(out, name, command, project, timeout):
        calls.append(name)
        return {"returncode": 1 if name == "graph" else 0}

    monkeypatch.setattr(runner, "stage", stage)
    result = runner.check_checkpoint(work, output)
    assert calls == ["build", "comparator", "graph"]
    assert result["status"] == "graph_failed"
    assert result["verified"] is True
    assert "graph_unavailable_reason" not in result
