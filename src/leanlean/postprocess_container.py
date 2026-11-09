"""Trusted offline verification entry point; artifacts remain on the host mount."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path


def write(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_stage(name: str, command: list[str], project: Path, timeout: int, env: dict | None = None) -> dict:
    started = time.time_ns()
    with (
        (Path("/work") / f"{name}.stdout.log").open("w") as out,
        (Path("/work") / f"{name}.stderr.log").open("w") as err,
    ):
        result = subprocess.run(
            ["timeout", "--kill-after=10s", str(timeout), *command], cwd=project, stdout=out, stderr=err, env=env
        )
    return {
        "command": command,
        "returncode": result.returncode,
        "started_wall_ns": started,
        "ended_wall_ns": time.time_ns(),
        "timeout_s": timeout,
    }


def digest_sources(project: Path) -> dict:
    return {
        str(p.relative_to(project)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(project.rglob("*"))
        if p.is_file() and ".lake" not in p.relative_to(project).parts
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["before", "after"], required=True)
    args = parser.parse_args()
    entry = json.loads(Path("/work/leanlean-project.json").read_text())
    project = Path("/work/repository") / entry["project_path"]
    initial = digest_sources(project)
    status = {"status": "running", "phase": args.phase, "stages": {}}
    path = Path("/work/verification.json")
    write(path, status)
    # These workspaces start without .lake/build or cached Lake configuration.
    # Keep every generated artifact afterward for the graph extractor.
    config = json.loads((Path("/work/repository") / entry["comparator_path"]).read_text())
    solution_module = config["solution_module"]
    status["stages"]["build"] = run_stage("build", ["lake", "build", f"+{solution_module}"], project, 21600)
    write(path, status)
    if args.phase == "after":
        status["stages"]["comparator"] = run_stage(
            "comparator",
            ["lake", "comparator", "--config", str(Path("/work/repository") / entry["comparator_path"])],
            project,
            3600,
        )
        # Comparator never compares the values of `definition_names`.
        if config.get("definition_names") and all(row["returncode"] == 0 for row in status["stages"].values()):
            write(path, status)
            from definitions_container import check_definitions

            proofs = Path("/work/definition-proofs.json")
            proofs = json.loads(proofs.read_text()) if proofs.is_file() else {}
            status["definitions"] = check_definitions(entry, config, Path("/work/repository"), project, proofs)
    status["sources_unchanged"] = initial == digest_sources(project)
    status["status"] = (
        "passed"
        if status["sources_unchanged"]
        and all(row["returncode"] == 0 for row in status["stages"].values())
        and status.get("definitions", {}).get("status", "passed") == "passed"
        else "failed"
    )
    status["retained_build_directory"] = str(project / ".lake")
    write(path, status)
    raise SystemExit(0 if status["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
