"""Adapt a verified LeanLean bundle to MathArena's ordinary harness runner."""

from __future__ import annotations

import gzip
import os
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path

from leanlean.common import read_json, sha256, tree_sha256, within, write_json

IMAGE_ROOT = Path("/opt/leanlean")


def load_problems(bundle: str, problem_ids: list[int] | None = None) -> list[dict]:
    root = Path(bundle).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No LeanLean bundle at {bundle}. See readmes/README_leanlean.md for setup.")
    manifest = read_json(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Evaluation requires a complete verified bundle")
    runtime = root / "eval-runtime.json"
    if runtime.exists() and read_json(runtime)["bundle_sha256"] != sha256(manifest_path):
        raise ValueError("Prepared runtime does not match the bundle")
    problems = []
    for index, entry in enumerate(manifest["projects"], 1):
        if problem_ids is not None and index not in problem_ids:
            continue
        source = within(root, entry["project_directory"])
        if entry.get("status") != "verified" or tree_sha256(source) != entry["project_tree_sha256"]:
            raise ValueError(f"Bundle source changed or is unverified: {entry['key']}")
        project = Path("/work/repository") / entry["project_path"]
        problems.append(
            {
                "problem_idx": index,
                "problem": (
                    f"{entry['title']}\n{entry.get('abstract', '')}\n"
                    f"Source: {entry['github_url']} at {entry['commit']}\n"
                    f"Project directory: {project}\n"
                    f"Solution: /work/repository/{entry['solution_path']}\n"
                    f"Challenge: /work/repository/{entry['challenge_path']}\n"
                    f"Comparator configuration: /work/repository/{entry['comparator_path']}\n"
                    f"Lean toolchain: {entry['lean_toolchain']}"
                ),
                "answer": None,
                "leanlean_project": {**entry, "source_directory": str(source)},
            }
        )
    return problems


def prepare_workspace(workspace: Path, entry: dict) -> None:
    """Copy only source files; dependencies remain immutable inside the image."""
    source = Path(entry["source_directory"])
    if tree_sha256(source) != entry["project_tree_sha256"]:
        raise ValueError("LeanLean bundle changed after loading")
    repository = workspace / "repository"
    shutil.copytree(source, repository, symlinks=True)
    project = repository / entry["project_path"]
    project.resolve().relative_to(repository.resolve())
    lake = project / ".lake"
    lake.mkdir(exist_ok=True)
    (lake / "packages").symlink_to(IMAGE_ROOT / "packages" / entry["key"], target_is_directory=True)
    write_json(workspace / "leanlean-project.json", entry)
    command = f"cd {shlex.quote(str(Path('/work/repository') / entry['project_path']))}\n"
    for name, tool in [("check.sh", "leanlean-check"), ("proof_length.py", "proof-length")]:
        path = workspace / name
        if name.endswith(".py"):
            path.write_text('#!/usr/bin/env python3\nimport os\nos.execvp("proof-length", ["proof-length"])\n')
        else:
            path.write_text("#!/bin/sh\nset -eu\n" + command + f'exec {tool} "$@"\n')
        path.chmod(0o755)


def write_submission_archive(root: Path, archive: Path) -> dict:
    """Write repository sources using stable headers and keep file bytes and modes."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Submission repository must remain a directory, not a symlink")
    excluded = {".lake", ".git", ".lean-strip", "__pycache__"}

    def normalized(member: tarfile.TarInfo) -> tarfile.TarInfo:
        member.uid = member.gid = member.mtime = 0
        member.uname = member.gname = ""
        return member

    files = {}
    with (
        archive.open("xb") as output,
        gzip.GzipFile(filename="", fileobj=output, mode="wb", mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w", dereference=False) as stream,
    ):
        for directory, dirs, names in os.walk(root, followlinks=False):
            dirs[:] = sorted(name for name in dirs if name not in excluded)
            names.extend(name for name in dirs if (Path(directory) / name).is_symlink())
            for name in sorted(name for name in names if name not in excluded):
                path = Path(directory) / name
                relative = path.relative_to(root).as_posix()
                # Never follow model-created symlinks into host files.
                if path.is_symlink():
                    files[relative] = {"symlink": os.readlink(path)}
                elif path.is_file():
                    files[relative] = {"sha256": sha256(path)}
                else:
                    continue
                stream.add(path, arcname=relative, recursive=False, filter=normalized)
    return files


def capture_submission(workspace: Path, entry: dict, *, output: Path | None = None) -> dict:
    """Retain an independent source archive for deferred grading, without caches."""
    root = workspace / "repository"
    destination = output if output is not None else workspace
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "leanlean-submission.tar.gz"
    # Publish only a complete archive. Replacing an existing symlink does not follow it.
    with tempfile.TemporaryDirectory(prefix=".submission-", dir=destination) as scratch:
        staged = Path(scratch) / archive.name
        files = write_submission_archive(root, staged)
        staged.replace(archive)
    return {
        "key": entry["key"],
        "baseline_sha256": entry["project_tree_sha256"],
        "archive": str(archive.resolve()),
        "sha256": sha256(archive),
        "files": files,
    }
