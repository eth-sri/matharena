"""Prove protected definitions equal to the registered challenge's (runs offline in the container).

`lake comparator` treats `definition_names` as holes: it compares their types but
never their values. This stage elaborates a renamed copy of the challenge without
the submission, proves each final definition equal to its copy (kernel `rfl`, or a
reviewed proof from the definition-proofs registry), and accepts a proof only after
`definition_audit.lean` replays it in a fresh kernel with the standard axioms.
Challenge definitions whose value is `sorry` are intended holes and are not compared.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from postprocess_container import run_stage

REFERENCE = "LeanLeanReference"
CHECK = "LeanLeanDefCheck"
TRUSTED_PREFIXES = ("/opt/leanlean/", "/opt/elan/")
# Lean only elaborates modules under the project root; `.lake` is outside the source digest.
WORK = Path(".lake/leanlean-definitions")


def reference_source(challenge: str, names: list[str]) -> str:
    """Move every namespace root used by `names` under `LeanLeanReference`."""
    roots = sorted({name.split(".")[0] for name in names}, key=len, reverse=True)
    alternatives = "|".join(map(re.escape, roots))
    # `_root_.Root` is redirected as well; otherwise the copy could name a final constant.
    pattern = re.compile(rf"(?:(?<=_root_\.)|(?<![\w.'!?]))({alternatives})(?![\w'!?])")
    return pattern.sub(rf"{REFERENCE}.\1", challenge)


def check_source(names: list[str], solution_module: str, proofs: dict[str, list[str]]) -> str:
    lines = [f"import {REFERENCE}", f"import {solution_module}", ""]
    lines += ["set_option maxHeartbeats 4000000", "set_option linter.all false", ""]
    for name in names:
        # Unproved definitions elaborate to `sorry`; the audit reports them per name.
        alternatives = ["rfl", "with_unfolding_all exact rfl", *proofs.get(name, []), "sorry"]
        lines.append(f"theorem {CHECK}.«{name}» : @_root_.{name} = @_root_.{REFERENCE}.{name} := by")
        lines.append("  first")
        for tactic in alternatives:
            lines += ["  | (", *(f"    {line}" for line in tactic.strip().splitlines()), "    )"]
        lines.append("")
    return "\n".join(lines)


class Runner:
    def __init__(self, project: Path, stages: dict):
        self.project, self.stages, self.work = project, stages, project / WORK
        (self.work / "lib").mkdir(parents=True, exist_ok=True)

    def lean(self, name: str, *args: str) -> bool:
        # `lake env` appends an inherited LEAN_PATH, which exposes the generated modules.
        env = {**os.environ, "LEAN_PATH": str(self.work / "lib")}
        self.stages[name] = run_stage(name, ["lake", "env", "lean", *args], self.project, 7200, env)
        return self.stages[name]["returncode"] == 0

    def compile(self, name: str, module: str, source: str) -> bool:
        (self.work / f"{module}.lean").write_text(source)
        return self.lean(name, str(self.work / f"{module}.lean"), "-o", str(self.work / "lib" / f"{module}.olean"))

    def audit(self, name: str, mode: str, module: str, names: list[str]) -> dict | None:
        spec = self.work / f"{mode}-names.json"
        spec.write_text(json.dumps(names))
        script = str(Path(__file__).with_name("definition_audit.lean"))
        if not self.lean(name, "--run", script, mode, module, str(spec), *TRUSTED_PREFIXES):
            return None
        return json.loads((Path("/work") / f"{name}.stdout.log").read_text().strip().splitlines()[-1])


def check_definitions(entry: dict, config: dict, repository: Path, project: Path, proofs: dict) -> dict:
    """Compare `config["definition_names"]`; `status` is `passed` or `failed`."""
    names = config["definition_names"]
    result = {"status": "failed", "stages": {}, "definitions": {}}
    run = Runner(project, result["stages"])
    challenge = (repository / entry["challenge_path"]).read_text(encoding="utf-8")
    # The copy sees only the challenge's own imports, never the submission.
    reference = None
    if run.compile("definitions_reference", REFERENCE, reference_source(challenge, names)):
        reference = run.audit("definitions_classify", "reference", REFERENCE, names)
    if reference is None:
        result["error"] = "challenge copy did not elaborate or classify"
        return result
    result["definitions"] = reference
    if any(reference[name]["status"] not in {"definition", "intended_hole"} for name in names):
        result["error"] = "challenge copy lacks a registered definition"
        return result
    compared = [name for name in names if reference[name]["status"] == "definition"]
    if compared:
        checked = None
        if run.compile("definitions_check", CHECK, check_source(compared, config["solution_module"], proofs)):
            checked = run.audit("definitions_audit", "check", CHECK, compared)
        if checked is None:
            result["error"] = "equivalence file did not elaborate or replay"
            return result
        result["replayed_constants"] = checked["replayed_constants"]
        result["definitions"].update(checked["definitions"])
    if all(row["status"] in {"equal", "intended_hole"} for row in result["definitions"].values()):
        result["status"] = "passed"
    return result
