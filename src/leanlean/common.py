"""Filesystem helpers for LeanLean source snapshots."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    """Readers see either the previous complete document or the new one."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, ensure_ascii=False, sort_keys=True)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def python_package_sha256(root: Path) -> str:
    """Fingerprint installed Python sources without mutable bytecode caches."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    """Fingerprint source names, bytes, modes and links, without following links."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            content = {"link": os.readlink(path)}
        elif path.is_file():
            content = {"sha256": sha256(path), "executable": bool(path.stat().st_mode & 0o111)}
        elif path.is_dir():
            content = {"directory": True}
        else:
            raise ValueError(f"Unsupported source file: {path}")
        digest.update(json.dumps([path.relative_to(root).as_posix(), content], sort_keys=True).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def relative_path(value: str) -> str:
    """Accept only normalized repository-relative POSIX paths."""
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Invalid relative path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", "..", ".git"} for part in value.split("/")):
        raise ValueError(f"Unsafe relative path: {value!r}")
    return path.as_posix()


def within(root: Path, relative: str) -> Path:
    path = root / relative_path(relative)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes source tree: {relative}")
    return path


@contextmanager
def exclusive_lock(path: Path, *, blocking: bool = False) -> Iterator[None]:
    """Prevent concurrent commands from modifying the same output directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as error:
            raise RuntimeError(f"Another command holds {path}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def write_index(root: Path, entries: list[dict]) -> None:
    """A compact review surface; complete metadata remains in entries/*.json."""
    columns = [
        "id",
        "version",
        "title",
        "github_url",
        "project_url",
        "challenge_url",
        "solution_url",
        "comparator_url",
        "project_path",
        "challenge_path",
        "solution_path",
        "comparator_path",
        "lean_toolchain",
        "theorem_names",
        "fetch_status",
        "error",
    ]
    temporary = root / ".index.csv.tmp"
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow({**entry, "theorem_names": "; ".join(entry.get("theorem_names", []))})
    os.replace(temporary, root / "index.csv")
