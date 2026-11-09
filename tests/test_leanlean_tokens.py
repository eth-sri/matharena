import importlib.util
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[1] / "third_party" / "lean-lean"

spec = importlib.util.spec_from_file_location("vendored_tokens", REFERENCE / "src/leanlean/metrics/tokens.py")
tokens = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tokens)


def test_code_after_imports_on_the_same_line_is_counted():
    statement = "theorem protected_result : ∀ n : Nat, n + 0 = n := by intro n; simp\n"
    separate = tokens.count_lean_tokens_in_source("import Lean\n" + statement)
    assert separate > 0
    assert tokens.count_lean_tokens_in_source("import Lean " + statement) == separate
    assert tokens.count_lean_tokens_in_source("public import Lean Foo.Bar " + statement) == separate


def test_import_commands_stay_unscored():
    body = "theorem t : True := trivial\n"
    expected = tokens.count_lean_tokens_in_source(body)
    source = "import Mathlib\npublic import Foo.Bar Baz -- theorem fake : True := trivial\n\n" + body
    assert tokens.count_lean_tokens_in_source(source) == expected
    assert tokens.count_lean_tokens_in_source('def s := "import Lean theorem"\n') > 0


def test_split_import_command_returns_modules_and_trailing_code():
    line = "import A B.C @[simp] theorem t : True := trivial"
    assert tokens.split_import_command(line, line) == (["A", "B.C"], "@[simp] theorem t : True := trivial")
    assert tokens.split_import_command("import A", "import A") == (["A"], "")
    assert tokens.split_import_command("def x := 1", "def x := 1") == ([], "def x := 1")
