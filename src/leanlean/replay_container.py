"""Offline checks for one checkpoint; logs and graphs outlive the container."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

from graphs_container import source_hashes
from postprocess_container import write


def stage(output: Path, name: str, command: list[str], project: Path, seconds: int) -> dict:
    started = time.time_ns()
    with (output / f"{name}.stdout.log").open("w") as stdout, (output / f"{name}.stderr.log").open("w") as stderr:
        result = subprocess.run(
            ["timeout", "--kill-after=10s", str(seconds), *command],
            cwd=project,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return dict(
        command=command,
        returncode=result.returncode,
        started_wall_ns=started,
        ended_wall_ns=time.time_ns(),
        timeout_s=seconds,
    )


def check_checkpoint(work: Path, output: Path) -> dict:
    repository = work / "repository"
    entry = json.loads((work / "leanlean-project.json").read_text())
    project = repository / entry["project_path"]
    comparator = str(repository / entry["comparator_path"])
    before = source_hashes(repository)
    result = dict(status="running", stages={})
    status_path = output / "checks.json"
    write(status_path, result)
    try:
        # Match lean-strip/LeanLean: defaultTargets may omit the solution.
        module = json.loads(Path(comparator).read_text())["solution_module"]
        if not isinstance(module, str) or not module.strip():
            raise ValueError("Comparator has no solution module")
        result["stages"]["build"] = stage(output, "build", ["lake", "build", f"+{module}"], project, 21600)
        write(status_path, result)
        if result["stages"]["build"]["returncode"]:
            result["status"] = "build_failed"
            result["graph_unavailable_reason"] = "solution_build_failed"
        else:
            result["stages"]["comparator"] = stage(
                output,
                "comparator",
                ["lake", "comparator", "--config", comparator],
                project,
                3600,
            )
            write(status_path, result)
            # A buildable but invalid proof is still useful in the edit timeline.
            # Preserve its graph without treating its compression as valid.
            result["stages"]["graph"] = stage(
                output,
                "graph",
                [
                    sys.executable,
                    "-m",
                    "lean_strip.cli",
                    str(project),
                    "--graph-only",
                    "--comparator",
                    comparator,
                    "--jobs",
                    "8",
                    "--timeout",
                    "21600",
                ],
                project,
                43200,
            )
            generated = project / ".lean-strip"
            if result["stages"]["graph"]["returncode"] == 0:
                shutil.copyfile(generated / "dependency-graph.json", output / "dependency-graph.json")
                result["status"] = "complete"
            else:
                result["status"] = "graph_failed"
            for name in ("report.json", "diagnostics.json"):
                if (generated / name).is_file():
                    shutil.copyfile(generated / name, output / name)
        result["verified"] = all(
            result["stages"].get(name, {}).get("returncode") == 0 for name in ("build", "comparator")
        )
        result["sources_unchanged"] = before == source_hashes(repository)
        if not result["sources_unchanged"]:
            result.update(status="source_changed", verified=False)
    except Exception as error:
        result.update(status="error", verified=False, error=f"{type(error).__name__}: {error}")
    finally:
        write(status_path, result)
    return result


def main() -> None:
    result = check_checkpoint(Path("/work"), Path("/output"))
    # A failed proof/build is a completed observation, not an orchestration error.
    raise SystemExit(1 if result["status"] in {"error", "source_changed"} else 0)


if __name__ == "__main__":
    main()
