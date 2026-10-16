"""Opt-in prove/disprove contract; the comparator is the acceptance boundary."""

from __future__ import annotations

import hashlib
import json
import re

MODE = "prove_or_disprove"
LEAN_VERSION = "4.34.1"
TOOLCHAIN = f"leanprover/lean4:v{LEAN_VERSION}"
ENVIRONMENT = f"lean-{LEAN_VERSION}"
CONJECTURE = "arxivleanConjecture"
TARGETS = {"prove": "arxivlean_proof", "disprove": "arxivlean_disproof"}


def lean_tokens(code: str) -> str:
    """Mask comments and strings for structural checks (not a Lean verifier)."""
    result = []
    i = depth = 0
    while i < len(code):
        if code.startswith("/-", i):
            depth += 1
            result.append("  ")
            i += 2
        elif depth and code.startswith("-/", i):
            depth -= 1
            result.append("  ")
            i += 2
        elif depth:
            result.append("\n" if code[i] == "\n" else " ")
            i += 1
        elif code.startswith("--", i):
            end = code.find("\n", i)
            end = len(code) if end < 0 else end
            result.append(" " * (end - i))
            i = end
        elif code[i] == '"':
            result.append(" ")
            i += 1
            while i < len(code):
                char = code[i]
                result.append("\n" if char == "\n" else " ")
                i += 1
                if char == "\\" and i < len(code):
                    result.append(" ")
                    i += 1
                elif char == '"':
                    break
        else:
            result.append(code[i])
            i += 1
    if depth:
        raise ValueError("Unterminated Lean comment")
    return "".join(result)


def validate_statement(code: str) -> None:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Empty conjecture statement")
    tokens = lean_tokens(code)
    if len(re.findall(r"\bdef\s+arxivleanConjecture\s*:\s*Prop\s*:=", tokens)) != 1:
        raise ValueError("Define exactly one closed def arxivleanConjecture : Prop := ...")
    if re.search(r"\b(sorry|admit|axiom|constant|opaque|unsafe|native_decide|implemented_by)\b", tokens):
        raise ValueError("Statement contains a placeholder or unsupported trusted declaration")
    if re.search(r"\b(arxivlean_proof|arxivlean_disproof|arxivlean_well_typed)\b", tokens):
        raise ValueError("Statement contains a reserved solution declaration")
    # No ambient parameters: the entire original conjecture must be closed in P.
    if re.search(r"(?m)^\s*(variable|variables|parameter|parameters|namespace|section)\b", tokens):
        raise ValueError("Use closed declarations, without ambient variables or namespaces")
    if re.search(r"(?m)^\s*import\s+(?!Mathlib(?:\.|\s|$))", tokens):
        raise ValueError("Only Mathlib imports are allowed in the frozen statement")


def statement_code(context: str, proposition: str) -> str:
    code = f"import Mathlib\n\n{context.strip()}\n\ndef {CONJECTURE} : Prop :=\n  {proposition.strip()}\n"
    validate_statement(code)
    return code


def target_statement(formal: str, verdict: str) -> str:
    validate_statement(formal)
    name = TARGETS[verdict]
    target = CONJECTURE if verdict == "prove" else f"Not {CONJECTURE}"
    return f"{formal.rstrip()}\n\ntheorem {name} : {target} := by\n  sorry\n"


def submission_code(text: str) -> str:
    blocks = re.findall(r"```lean\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if "```" in text and len(blocks) != 1:
        raise ValueError("Return exactly one complete Lean code block")
    return blocks[0] if blocks else text


def submission_verdict(code: str) -> str:
    tokens = lean_tokens(code)
    choices = [verdict for verdict, name in TARGETS.items()
               for _ in re.finditer(rf"(?m)^\s*theorem\s+{name}\s*:", tokens)]
    if len(choices) != 1:
        raise ValueError("Declare exactly one theorem arxivlean_proof or arxivlean_disproof")
    return choices[0]


def runtime_identity() -> dict:
    """Pin the actual local mathlib revision, not merely a movable tag."""
    from matharena.tools import lean_execution as execution

    for path in (execution.COMPARATOR_BIN, execution.LEAN4EXPORT_BIN, execution.LANDRUN_BIN):
        if not path.is_file():
            raise ValueError(f"Missing comparator runtime asset: {path}")
    project = execution.COMPARATOR_PROJECT_DIR
    toolchain = (project / "lean-toolchain").read_text().strip()
    manifest = json.loads((project / "lake-manifest.json").read_text())
    mathlib = [p for p in manifest["packages"] if p.get("name") == "mathlib"]
    if toolchain != TOOLCHAIN or len(mathlib) != 1:
        raise ValueError(f"Comparator project must pin Lean {LEAN_VERSION} and mathlib")
    rev = mathlib[0].get("rev", "")
    if not re.fullmatch(r"[0-9a-f]{40}", rev):
        raise ValueError("mathlib must have a locked Git revision")
    return {
        "toolchain": toolchain, "mathlib_revision": rev,
        "manifest_sha256": hashlib.sha256((project / "lake-manifest.json").read_bytes()).hexdigest(),
        "lakefile_sha256": hashlib.sha256((project / "lakefile.lean").read_bytes()).hexdigest(),
    }


def _check(code: str, challenge: str, name: str) -> dict:
    from matharena.tools import lean_execution as execution

    error = execution._run_comparator_check(
        code, challenge, theorem_names=[name], exact_submission=True, environment=ENVIRONMENT,
    )
    unavailable = error in execution.COMPARATOR_FALLBACK_WARNINGS or (
        error is not None and "environment does not match" in error
    )
    if unavailable:
        error = "Comparator unavailable, timed out, or mismatched; submission is unverified."
    return {"okay": error is None, "errors": [error] if error else [], "warnings": [],
            "infos": [], "status": "unverified" if unavailable else "rejected" if error else "verified"}


def check_statement(formal: str) -> dict:
    validate_statement(formal)
    name = "arxivlean_well_typed"
    code = f"{formal}\ntheorem {name} : True := by trivial\n"
    return _check(code, code, name)


def check_submission(text: str, formal: str, *, environment=ENVIRONMENT, expected_runtime=None) -> dict:
    try:
        if environment != ENVIRONMENT:
            raise ValueError(f"Conjecture grading requires the pinned Lean {LEAN_VERSION} environment")
        if expected_runtime is not None and runtime_identity() != expected_runtime:
            return {"okay": False, "errors": ["Pinned Lean/mathlib project changed"],
                    "warnings": [], "infos": [], "status": "unverified"}
        validate_statement(formal)
    except (ValueError, OSError, KeyError) as exc:
        return {"okay": False, "errors": [str(exc)], "warnings": [], "infos": [], "status": "unverified"}
    try:
        code = submission_code(text)
        verdict = submission_verdict(code)
        if re.search(r"\b(sorry|admit|axiom|constant|unsafe|native_decide|implemented_by)\b", lean_tokens(code)):
            raise ValueError("Submission contains a placeholder or forbidden trusted declaration")
    except ValueError as exc:
        return {"okay": False, "errors": [str(exc)], "warnings": [], "infos": [], "status": "rejected"}
    challenge = target_statement(formal, verdict)
    feedback = _check(code, challenge, TARGETS[verdict])
    feedback["verdict"] = verdict
    return feedback
