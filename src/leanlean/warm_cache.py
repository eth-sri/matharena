"""Complete dependency build metadata before freezing an evaluation image."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from leanlean.common import read_json
from leanlean.evaluation import IMAGE_ROOT


def warm_project(entry: dict) -> None:
    # Build at the final dependency paths: Lake may write missing artifact hashes
    # even when its downloaded oleans are otherwise complete. Only the scratch
    # project is removed; dependency metadata stays in the image layer.
    with tempfile.TemporaryDirectory(prefix="leanlean-cache-") as temporary:
        repository = Path(temporary) / "repository"
        shutil.copytree(IMAGE_ROOT / "sources" / entry["key"], repository, symlinks=True)
        project = repository / entry["project_path"]
        lake = project / ".lake"
        lake.mkdir(exist_ok=True)
        (lake / "packages").symlink_to(IMAGE_ROOT / "packages" / entry["key"], target_is_directory=True)
        environment = {**os.environ, "LEAN_NUM_THREADS": os.environ.get("LEAN_NUM_THREADS", "8")}
        print(f"Warming dependencies: {entry['key']}", flush=True)
        solution_module = read_json(repository / entry["comparator_path"])["solution_module"]
        subprocess.run(
            ["lake", "build", f"+{solution_module}"], cwd=project, env=environment, check=True, timeout=21600
        )
        print(f"Dependencies ready: {entry['key']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--project", action="append", help="Repair selected project keys; default: all projects")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    projects = read_json(IMAGE_ROOT / "manifest.json")["projects"]
    if args.project:
        requested = set(args.project)
        missing = requested - {entry["key"] for entry in projects}
        if missing:
            parser.error(f"Unknown project keys: {sorted(missing)}")
        projects = [entry for entry in projects if entry["key"] in requested]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(warm_project, projects))


if __name__ == "__main__":
    main()
