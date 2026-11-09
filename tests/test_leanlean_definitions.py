import importlib
import json
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).parents[1] / "src/leanlean"


@pytest.fixture
def definitions(monkeypatch):
    monkeypatch.syspath_prepend(str(CODE))
    return importlib.import_module("definitions_container")


def test_challenge_copy_moves_only_registered_roots(definitions):
    source = "\n".join(
        [
            "namespace GoldbachLean.Wu2008",
            "def em (k : ℝ) : ℝ := 1 / 2 - 3 * k",
            "def x := GoldbachLean.Wu2008.em 0 + _root_.GoldbachLean.Wu2008.em 1",
            "def y := Foo.GoldbachLean + GoldbachLean' + GoldbachLeanExtra + Mathlib.Nat",
            "end GoldbachLean.Wu2008",
        ]
    )
    copy = definitions.reference_source(source, ["GoldbachLean.Wu2008.em"])
    assert copy.splitlines() == [
        "namespace LeanLeanReference.GoldbachLean.Wu2008",
        "def em (k : ℝ) : ℝ := 1 / 2 - 3 * k",
        "def x := LeanLeanReference.GoldbachLean.Wu2008.em 0 + _root_.LeanLeanReference.GoldbachLean.Wu2008.em 1",
        "def y := Foo.GoldbachLean + GoldbachLean' + GoldbachLeanExtra + Mathlib.Nat",
        "end LeanLeanReference.GoldbachLean.Wu2008",
    ]


def test_equivalence_tries_rfl_then_reviewed_proofs_and_falls_back_to_sorry(definitions):
    source = definitions.check_source(["A.f", "A.g"], "Solution", {"A.g": ["funext x\nsimp [A.g]"]})
    assert source.splitlines()[:2] == ["import LeanLeanReference", "import Solution"]
    assert "theorem LeanLeanDefCheck.«A.f» : @_root_.A.f = @_root_.LeanLeanReference.A.f := by" in source
    g = source[source.index("«A.g»") :]
    assert [line.strip() for line in g.splitlines()[2:]] == [
        "| (", "rfl", ")",
        "| (", "with_unfolding_all exact rfl", ")",
        "| (", "funext x", "simp [A.g]", ")",
        "| (", "sorry", ")",
    ]  # fmt: skip


@pytest.mark.parametrize(
    ("names", "comparator", "definitions_status", "expected", "checked"),
    [
        ([], 0, None, 0, False),
        (["A.f"], 1, None, 1, False),
        (["A.f"], 0, "passed", 0, True),
        (["A.f"], 0, "failed", 1, True),
    ],
)
def test_unproved_definitions_fail_final_verification(
    tmp_path, monkeypatch, definitions, names, comparator, definitions_status, expected, checked
):
    runner = importlib.import_module("leanlean.postprocess_container")
    repository = tmp_path / "repository"
    repository.mkdir()
    (tmp_path / "leanlean-project.json").write_text(json.dumps({"project_path": "", "comparator_path": "c.json"}))
    (repository / "c.json").write_text(json.dumps({"solution_module": "Solution", "definition_names": names}))
    (tmp_path / "definition-proofs.json").write_text('{"A.f": ["simp"]}')
    calls = []

    def path(value):
        original = Path(value)
        return tmp_path / original.relative_to("/work") if original.is_relative_to("/work") else original

    def check(entry, config, repo, project, proofs):
        calls.append(proofs)
        return {"status": definitions_status}

    monkeypatch.setattr(runner, "Path", path)
    monkeypatch.setattr(runner, "run_stage", lambda name, *_: {"returncode": comparator if name == "comparator" else 0})
    monkeypatch.setattr(definitions, "check_definitions", check)
    monkeypatch.setattr(sys, "argv", ["postprocess", "--phase", "after"])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == expected
    assert calls == ([{"A.f": ["simp"]}] if checked else [])
    status = json.loads((tmp_path / "verification.json").read_text())
    assert ("definitions" in status) == checked
