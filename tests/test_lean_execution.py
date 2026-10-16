import json
from types import SimpleNamespace

from matharena import grader
from matharena.parser import WarningType
from matharena.tools import lean_execution


def test_strip_lean_imports_removes_imports_from_any_position():
    source = """import Mathlib

lemma first : True := by trivial

  import Mathlib.Algebra.Group.Basic

lemma second : True := by trivial
"""

    normalized = lean_execution._strip_lean_imports(source)

    assert "import " not in normalized
    assert "lemma first" in normalized
    assert "lemma second" in normalized


def test_run_comparator_check_reports_missing_assets(monkeypatch, tmp_path):
    monkeypatch.setattr(lean_execution, "COMPARATOR_BIN", tmp_path / "missing-comparator")
    monkeypatch.setattr(lean_execution, "LEAN4EXPORT_BIN", tmp_path / "missing-lean4export")
    monkeypatch.setattr(lean_execution, "LANDRUN_BIN", tmp_path / "missing-landrun")
    monkeypatch.setattr(lean_execution, "COMPARATOR_PROJECT_DIR", tmp_path / "missing-project")

    result = lean_execution._run_comparator_check(
        "theorem target : True := by trivial",
        "theorem target : True := by sorry",
    )

    assert result == lean_execution.COMPARATOR_UNAVAILABLE_WARNING


def test_run_comparator_check_normalizes_persistent_imports(monkeypatch, tmp_path):
    comparator_bin = tmp_path / "comparator" / "comparator"
    lean4export_bin = tmp_path / "lean4export" / "lean4export"
    landrun_bin = tmp_path / "landrun" / "landrun"
    comparator_project = tmp_path / "comparator-project"
    for path in [comparator_bin, lean4export_bin, landrun_bin]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    comparator_project.mkdir()
    (comparator_project / "lakefile.lean").write_text("import Lake\n", encoding="utf-8")
    (comparator_project / "lean-toolchain").write_text("leanprover/lean4:v4.31.0\n", encoding="utf-8")

    monkeypatch.setattr(lean_execution, "COMPARATOR_BIN", comparator_bin)
    monkeypatch.setattr(lean_execution, "LEAN4EXPORT_BIN", lean4export_bin)
    monkeypatch.setattr(lean_execution, "LANDRUN_BIN", landrun_bin)
    monkeypatch.setattr(lean_execution, "COMPARATOR_PROJECT_DIR", comparator_project)

    captured = {}
    lean_prefix = tmp_path / "lean-toolchain"
    (lean_prefix / "bin").mkdir(parents=True)
    (lean_prefix / "bin" / "lake").touch()

    def fake_run(*args, **kwargs):
        if args[0] == ["lean", "--print-prefix"]:
            return SimpleNamespace(returncode=0, stdout=str(lean_prefix), stderr="")
        workdir = kwargs["cwd"]
        captured["challenge"] = (workdir / "Challenge.lean").read_text(encoding="utf-8")
        captured["solution"] = (workdir / "Solution.lean").read_text(encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(lean_execution.subprocess, "run", fake_run)
    messages = [
        {
            "role": "assistant",
            "type": "tool_call",
            "tool_name": "add_to_file",
            "tool_call_id": "call-1",
            "arguments": json.dumps({"code": "import Mathlib\n\nlemma helper : True := by trivial"}),
            "content": "",
        },
        {
            "role": "tool_response",
            "tool_name": "add_to_file",
            "tool_call_id": "call-1",
            "content": "### Added To File ###\nTrue\n\nAdded to file.",
        },
    ]
    model_output = """```lean
import Mathlib

theorem target : True := by
  exact helper
```"""

    result = lean_execution._run_comparator_check(
        model_output,
        "import Mathlib\n\ntheorem target : True := by sorry",
        messages=messages,
    )

    assert result is None
    assert captured["challenge"].count("import ") == 1
    assert captured["challenge"].startswith("import Mathlib\n")
    assert captured["solution"].count("import ") == 1
    assert captured["solution"].startswith("import Mathlib\n")
    assert "lemma helper" in captured["solution"]
    assert "theorem target" in captured["solution"]


def test_unavailable_comparator_is_reported_as_possible_warning(monkeypatch):
    monkeypatch.setattr(
        grader,
        "get_lean_feedback_dict_with_formal_statement",
        lambda *args, **kwargs: {
            "okay": True,
            "errors": [],
            "warnings": [lean_execution.COMPARATOR_UNAVAILABLE_WARNING],
            "infos": [],
        },
    )

    _, is_correct, warning = grader.grade_lean_submission(
        [{"role": "assistant", "content": "theorem target : True := by trivial"}],
        {"lean_environment": "lean-4.31.0"},
        {"formal_statement": "theorem target : True := by sorry"},
    )

    assert is_correct is True
    assert warning == WarningType.POSSIBLE.value


def test_landrun_permission_error_is_reported_as_unavailable(monkeypatch, tmp_path):
    comparator_bin = tmp_path / "comparator" / "comparator"
    lean4export_bin = tmp_path / "lean4export" / "lean4export"
    landrun_bin = tmp_path / "landrun" / "landrun"
    comparator_project = tmp_path / "comparator-project"
    for path in [comparator_bin, lean4export_bin, landrun_bin]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    comparator_project.mkdir()
    (comparator_project / "lakefile.lean").write_text("import Lake\n", encoding="utf-8")
    (comparator_project / "lean-toolchain").write_text("leanprover/lean4:v4.31.0\n", encoding="utf-8")

    monkeypatch.setattr(lean_execution, "COMPARATOR_BIN", comparator_bin)
    monkeypatch.setattr(lean_execution, "LEAN4EXPORT_BIN", lean4export_bin)
    monkeypatch.setattr(lean_execution, "LANDRUN_BIN", landrun_bin)
    monkeypatch.setattr(lean_execution, "COMPARATOR_PROJECT_DIR", comparator_project)
    monkeypatch.setattr(
        lean_execution.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=1,
            stdout="Building Challenge\n",
            stderr="[landrun:error] permission denied\nuncaught exception: Child exited with 1\n",
        ),
    )

    result = lean_execution._run_comparator_check(
        "theorem target : True := by trivial",
        "theorem target : True := by sorry",
    )

    assert result == lean_execution.COMPARATOR_UNAVAILABLE_WARNING
