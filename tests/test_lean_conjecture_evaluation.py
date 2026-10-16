"""Evaluation-only regressions, independent of the generation pipeline."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest
import yaml

from matharena import grader
from matharena.lean_conjecture import (
    ENVIRONMENT, LEAN_VERSION, TOOLCHAIN, check_submission, lean_tokens,
    statement_code, submission_verdict, target_statement,
)
from matharena.tools import lean_execution

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_VALUE = {"toolchain": TOOLCHAIN, "mathlib_revision": "a" * 40,
                 "manifest_sha256": "m", "lakefile_sha256": "l"}


def test_registered_config_matches_evaluated_snapshot():
    config = yaml.safe_load((ROOT / "configs/competitions/arxivlean/september.yaml").read_text())
    setup = runpy.run_path(str(ROOT / "arxivmath/scripts/lean/setup_conjecture_runtime.py"))
    assert config["dataset_path"] == "MathArena/arxivlean-0926"
    assert config["lean_task"] == "prove_or_disprove" and config["n_problems"] == 53
    assert config["lean_environment"] == ENVIRONMENT
    assert config["lean_project_identity"]["toolchain"] == TOOLCHAIN == setup["TOOLCHAIN"]
    assert config["lean_project_identity"]["mathlib_revision"] == setup["MATHLIB_REV"]
    assert config["harness_config"]["lean_toolchain_dir"] == f"leanprover--lean4---v{LEAN_VERSION}"
    assert "chmod 0644 /etc/codex/requirements.toml" in (
        ROOT / "docker/harness-arxivlean/Conjecture.Dockerfile").read_text()


def test_isolated_runtime_override(tmp_path):
    code = "from matharena.tools.lean_execution import COMPARATOR_PROJECT_DIR; print(COMPARATOR_PROJECT_DIR)"
    env = {**os.environ, "MATHARENA_LEAN_RUNTIME_ROOT": str(tmp_path)}
    result = subprocess.run([sys.executable, "-c", code], env=env, check=True, text=True, capture_output=True)
    assert result.stdout.strip() == str(tmp_path / "external/comparator_project")


def test_runtime_loader_fix_is_narrow_and_idempotent(tmp_path):
    setup = runpy.run_path(str(ROOT / "arxivmath/scripts/lean/setup_conjecture_runtime.py"))
    source = tmp_path / "Main.lean"
    source.write_text('let args := #["--best-effort", "--ro", "/", "--rw", "/dev", "-ldd", "-add-exec"]')
    setup["allow_elf_interpreter"](source, Path("/lib64/ld-linux-x86-64.so.2"))
    patched = source.read_text()
    setup["allow_elf_interpreter"](source, Path("/lib64/ld-linux-x86-64.so.2"))
    assert source.read_text() == patched
    assert '"--rox", "/lib64/ld-linux-x86-64.so.2"' in patched
    source.write_text("unexpected upstream")
    with pytest.raises(RuntimeError, match="Unexpected upstream"):
        setup["allow_elf_interpreter"](source, Path("/lib64/ld-linux-x86-64.so.2"))


def test_registered_competition_loads_without_generation_code(tmp_path, monkeypatch):
    from matharena.runner import Runner
    config = yaml.safe_load((ROOT / "configs/competitions/arxivlean/september.yaml").read_text())
    monkeypatch.setattr("matharena.lean_conjecture.runtime_identity", lambda: config["lean_project_identity"])
    class Dataset:
        def to_list(self):
            return [{"problem_idx": 1, "problem": "Prove or disprove True.",
                     "formal_statement": statement_code("", "True"),
                     "answer": statement_code("", "True")} ]
    monkeypatch.setattr("matharena.runner.load_dataset", lambda *a, **kw: Dataset())
    runner = Runner("arxivlean/september", 1, None, "configs/competitions", "configs/models",
                    str(tmp_path / "outputs"), False)
    assert len(runner.problems) == 1
    assert all("resolution" not in p for p in runner.problems)
    model = runner.load_solver_config("configs/models/openai/gpt-61-sol.yaml")
    assert model["model_config"]["harness"] == "codex"
    assert runner._prepare_default_api_client_args(model["model_config"]) is not None



@pytest.mark.parametrize("context,prop", [
    ("axiom fact : False", "False"), ("", "by sorry"), ("variable (n : Nat)", "n = n"),
    ("namespace X", "True"), ("import Challenge", "True"),
    ("theorem arxivlean_proof : True := by trivial", "True"),
])
def test_unsafe_or_open_statement_rejected(context, prop):
    with pytest.raises(ValueError):
        statement_code(context, prop)


def test_negation_is_of_the_whole_closed_conjecture():
    code = statement_code("", "forall n : Nat, n = 0")
    negative = target_statement(code, "disprove")
    assert "theorem arxivlean_disproof : Not arxivleanConjecture" in negative
    assert code in negative
    assert "forall n : Nat, Not" not in negative


def test_comment_and_string_masking():
    assert "sorry" not in lean_tokens('/- nested /- sorry -/ x -/\n-- sorry\n#check "sorry"')
    with pytest.raises(ValueError):
        lean_tokens("/- unclosed")


@pytest.mark.parametrize("code", [
    "theorem t : P ∨ Not P := Classical.em P",
    "-- theorem arxivlean_proof : True := by trivial",
    "theorem arxivlean_proof : True := by trivial\ntheorem arxivlean_disproof : False := by sorry",
])
def test_requires_exactly_one_explicit_branch(code):
    with pytest.raises(ValueError):
        submission_verdict(code)


def test_comparator_receives_exact_submission_and_expected_branch(monkeypatch):
    captured = {}
    def comparator(code, challenge, **kwargs):
        captured.update(code=code, challenge=challenge, **kwargs)
        return None
    monkeypatch.setattr(lean_execution, "_run_comparator_check", comparator)
    formal = statement_code("", "False")
    code = formal + "\ntheorem arxivlean_disproof : Not arxivleanConjecture := by intro h; exact h"
    result = check_submission(f"```lean\n{code}\n```", formal)
    assert result["okay"] and result["verdict"] == "disprove"
    assert captured["exact_submission"] is True
    assert captured["code"].strip() == code
    assert captured["theorem_names"] == ["arxivlean_disproof"]


def test_wrong_theorem_type_is_not_replaced(monkeypatch):
    formal = statement_code("", "False")
    def comparator(code, challenge, **kwargs):
        assert "theorem arxivlean_proof : True" in code
        assert "theorem arxivlean_proof : arxivleanConjecture" in challenge
        return "Type mismatch"
    monkeypatch.setattr(lean_execution, "_run_comparator_check", comparator)
    feedback = check_submission("theorem arxivlean_proof : True := by trivial", formal)
    assert not feedback["okay"] and feedback["status"] == "rejected"


@pytest.mark.parametrize("error", list(lean_execution.COMPARATOR_FALLBACK_WARNINGS))
def test_comparator_failures_are_unverified_not_accepted(monkeypatch, error):
    monkeypatch.setattr("matharena.lean_conjecture.runtime_identity", lambda: RUNTIME_VALUE)
    monkeypatch.setattr(lean_execution, "_run_comparator_check", lambda *a, **k: error)
    formal = statement_code("", "True")
    submission = "theorem arxivlean_proof : arxivleanConjecture := by trivial"
    assert check_submission(submission, formal)["status"] == "unverified"
    with pytest.raises(RuntimeError, match="unverified"):
        grader.grade_lean_submission([{"role": "assistant", "content": submission}],
                                    {"lean_task": "prove_or_disprove", "lean_environment": "lean-4.34.1",
                                     "lean_project_identity": RUNTIME_VALUE},
                                    {"formal_statement": formal})


@pytest.mark.parametrize("body", ["by sorry", "by admit", "by native_decide"])
def test_placeholders_rejected_before_comparator(monkeypatch, body):
    monkeypatch.setattr(lean_execution, "_run_comparator_check", lambda *a, **k: pytest.fail("must not run"))
    result = check_submission(f"theorem arxivlean_proof : arxivleanConjecture := {body}", statement_code("", "True"))
    assert result["status"] == "rejected"



def test_harness_seeds_neutral_conjecture_without_gold(tmp_path):
    from matharena.solvers.harness_solver import HarnessSolver
    solver = object.__new__(HarnessSolver)
    solver.harness_config = {}
    solver.is_lean_comp = True
    solver.config = {"lean_task": "prove_or_disprove"}
    solver.toolchain_dir = "leanprover--lean4---v4.34.1"
    workspace = tmp_path / "workspace"
    formal = statement_code("", "False")
    solver._prepare_workspace(workspace, {"formal_statement": formal})
    solution = (workspace / "Solution.lean").read_text()
    assert formal in solution
    assert "theorem arxivlean_proof" in solution
    assert not (workspace / "Challenge.lean").exists()
    assert "lean Solution.lean" in (workspace / "check.sh").read_text()


def test_grader_does_not_require_matching_private_outcome(monkeypatch):
    monkeypatch.setattr("matharena.lean_conjecture.runtime_identity", lambda: RUNTIME_VALUE)
    monkeypatch.setattr(lean_execution, "_run_comparator_check", lambda *a, **k: None)
    verdict, correct, _ = grader.grade_lean_submission(
        [{"role": "assistant", "content": "theorem arxivlean_proof : arxivleanConjecture := by trivial"}],
        {"lean_task": "prove_or_disprove", "lean_environment": "lean-4.34.1", "lean_project_identity": RUNTIME_VALUE},
        {"formal_statement": statement_code("", "True"), "resolution": "disproved"},
    )
    assert verdict == "prove" and correct


def test_edited_runtime_is_unverified(monkeypatch):
    monkeypatch.setattr("matharena.lean_conjecture.runtime_identity", lambda: {**RUNTIME_VALUE, "mathlib_revision": "b" * 40})
    result = check_submission("theorem arxivlean_proof : arxivleanConjecture := by trivial",
                              statement_code("", "True"), expected_runtime=RUNTIME_VALUE)
    assert result["status"] == "unverified"
