"""Extract a graph from a verified workspace without changing its sources."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


def source_hashes(project: Path) -> dict[str, str]:
    hashes = {}
    for directory, dirs, names in os.walk(project):
        dirs[:] = sorted(d for d in dirs if d not in {".lake", ".git", ".lean-strip"})
        for name in sorted(names):
            path = Path(directory) / name
            hashes[str(path.relative_to(project))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def main() -> None:
    work = Path("/work")
    entry = json.loads((work / "leanlean-project.json").read_text())
    project = work / "repository" / entry["project_path"]
    before = source_hashes(work / "repository")
    command = [
        sys.executable,
        "-m",
        "lean_strip.cli",
        str(project),
        "--graph-only",
        "--comparator",
        str(work / "repository" / entry["comparator_path"]),
        "--jobs",
        "8",
        "--timeout",
        "21600",
    ]
    status = {"command": command, "started_wall_ns": time.time_ns(), "status": "running"}
    destination = work / "graph-status.json"
    destination.write_text(json.dumps(status, indent=2) + "\n")
    try:
        # Older retained workspaces may have had their build caches pruned.
        # Restore the same immutable dependency tree used by verification.
        packages = project / ".lake" / "packages"
        pinned = Path("/opt/leanlean/packages") / entry["key"]
        if not pinned.is_dir():
            raise FileNotFoundError(f"Missing pinned dependencies: {pinned}")
        if packages.is_symlink():
            if packages.resolve() != pinned.resolve():
                raise ValueError(f"Unexpected dependency link: {packages}")
        else:
            packages.parent.mkdir(parents=True, exist_ok=True)
            if packages.exists():
                packages.rename(packages.with_name(f"packages-before-recovery-{time.time_ns()}"))
            packages.symlink_to(pinned, target_is_directory=True)
            status["dependencies_restored"] = str(pinned)
            # Build against the offline dependencies before the extractor's
            # cache-fetch fallback can run.
            config = json.loads((work / "repository" / entry["comparator_path"]).read_text())
            subprocess.run(["lake", "build", f'+{config["solution_module"]}'], cwd=project,
                           check=True, timeout=21600)
        result = subprocess.run(command, check=False)
        status["returncode"] = result.returncode
        status["sources_unchanged"] = before == source_hashes(work / "repository")
        if result.returncode != 0 or not status["sources_unchanged"]:
            raise RuntimeError("Graph extraction failed or changed repository sources")
        graph = project / ".lean-strip" / "dependency-graph.json"
        shutil.copyfile(graph, work / "dependency-graph.json")
        status.update(status="passed", graph_sha256=hashlib.sha256(graph.read_bytes()).hexdigest())
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        status["ended_wall_ns"] = time.time_ns()
        destination.write_text(json.dumps(status, indent=2) + "\n")
    raise SystemExit(0 if status["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
