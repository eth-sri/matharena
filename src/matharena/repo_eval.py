"""Repository submissions and dataset-owned helper files for harness evaluations."""

import gzip
import hashlib
import shutil
import uuid
from pathlib import Path

from matharena.monitoring.attempt import archive_bytes, read_tree


def copy_helper_files(source: Path, workspace: Path) -> None:
    """Copy trusted dataset helpers without following links out of either tree."""
    if not source.is_dir() or source.is_symlink():
        raise ValueError("helper_files must name a directory of dataset files")
    root = workspace.resolve()
    if root.is_relative_to(source.resolve()):
        raise ValueError("Helper files must be outside the workspace")
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Helper files must not contain symlinks: {path}")
        relative = path.relative_to(source)
        target = workspace / relative
        if target.is_symlink() or not target.resolve().is_relative_to(root):
            raise ValueError(f"Helper destination escapes the workspace: {relative}")
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def snapshot_repository(workspace: Path, *, output: Path | None = None) -> dict:
    """Archive repository sources independently of optional monitoring."""
    data = archive_bytes(read_tree(workspace / "repository"))
    destination = output if output is not None else (
        workspace.parent / (workspace.name + ".submissions") / uuid.uuid4().hex
    )
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "repository.tar.gz"
    archive.write_bytes(gzip.compress(data, compresslevel=1, mtime=0))
    return {"type": "repository", "archive": str(archive.resolve()),
            "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
