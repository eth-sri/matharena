"""Bundle and source-archive format shared by the release tooling.

Bundles are split zstd tar files with a hashed member index. Source trees can
leave out scratch files and build caches; checkpoint containers are replaced by
hashed file objects and explicit trees. Website requests never open these bundles.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import contextmanager
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import tarfile
import tempfile

import zstandard

from matharena.monitoring.source_files import is_build_artifact

from leanlean.common import sha256


MODELS = {"gpt-61-sol-max": "sol", "opus-55-high": "opus",
          "gemini-4-argon": "gemini", "deepseek-v41-flash-max": "deepseek",
          "kimi-k31-pro": "kimi", "deepseek-v41-flash-dsh": "dsh"}
EXCLUDED = {".lake", ".git", ".lean-strip", "__pycache__"}
PART_SIZE = 32 * 1024**2


def encoded(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def packed(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(encoded(value), mtime=0))


def safe_name(name: str) -> str:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or str(path) != name:
        raise ValueError(f"Unsafe relative path: {name}")
    return name


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class HashedReader:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def read(self, size=-1):
        data = self.stream.read(size)
        self.digest.update(data)
        return data


def read_archive(path: Path, *, remove_scratch=False, exclude_dirs=(), exclude_build=False):
    """Read a monitor checkpoint archive, hashing both its gzip and tar bytes."""
    excluded = set(exclude_dirs) | ({".scratch"} if remove_scratch else set())
    state, removed, removed_bytes = {}, 0, 0
    with path.open("rb") as raw:
        compressed = HashedReader(raw)
        with gzip.GzipFile(fileobj=compressed) as uncompressed:
            source = HashedReader(uncompressed)
            with tarfile.open(fileobj=source, mode="r|", bufsize=1024 * 1024) as archive:
                seen = set()
                for member in archive:
                    name = member.name
                    parts = PurePosixPath(name)
                    if parts.is_absolute() or ".." in parts.parts or name in seen:
                        raise ValueError(f"Unsafe/duplicate checkpoint path: {name}")
                    seen.add(name)
                    if any(part in excluded for part in parts.parts) or (exclude_build and is_build_artifact(name)):
                        removed += 1
                        removed_bytes += member.size
                        continue
                    if member.isfile():
                        state[name] = ("file", archive.extractfile(member).read(), member.mode)
                    elif member.issym():
                        state[name] = ("symlink", member.linkname, member.mode)
                    else:
                        raise ValueError(f"Unsupported checkpoint entry: {name}")
            while source.read(8 * 1024**2):
                pass
    return state, dict(source_sha256=source.digest.hexdigest(), archive_sha256=compressed.digest.hexdigest(),
                       removed_files=removed, removed_bytes=removed_bytes)


def bundle_parts(path: Path) -> list[Path]:
    if path.exists():
        return [path]
    parts = sorted(path.parent.glob(path.name + ".part-*"))
    if not parts or any(p.name != f"{path.name}.part-{i:04d}" or p.is_symlink() for i, p in enumerate(parts)):
        raise ValueError(f"Missing or unordered bundle parts: {path}")
    return parts


class PartReader(io.RawIOBase):
    def __init__(self, paths):
        self.paths = iter(paths)
        self.current = None

    def readable(self):
        return True

    def readinto(self, buffer):
        while True:
            if self.current is None:
                path = next(self.paths, None)
                if path is None:
                    return 0
                self.current = path.open("rb")
            count = self.current.readinto(buffer)
            if count:
                return count
            self.current.close()
            self.current = None

    def close(self):
        if self.current:
            self.current.close()
        super().close()


def bundle_checksum(path: Path) -> str:
    with io.BufferedReader(PartReader(bundle_parts(path))) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def split_bundle(path: Path) -> list[dict]:
    if path.stat().st_size <= PART_SIZE:
        return []
    parts = []
    with path.open("rb") as stream:
        while data := stream.read(PART_SIZE):
            name = f"{path.name}.part-{len(parts):04d}"
            (path.parent / name).write_bytes(data)
            parts.append(dict(path=name, bytes=len(data), sha256=digest(data)))
    path.unlink()
    return parts


class Bundle:
    """Deterministic streaming tar.zst with per-member content receipts."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.raw = path.open("xb")
        self.compressed = zstandard.ZstdCompressor(level=9).stream_writer(self.raw, closefd=False)
        self.tar = tarfile.open(mode="w|", fileobj=self.compressed, format=tarfile.PAX_FORMAT)
        self.members = {}

    def add(self, name: str, data: bytes, mode: int = 0o644):
        safe_name(name)
        if name in self.members:
            raise ValueError(f"Duplicate bundle member: {name}")
        info = tarfile.TarInfo(name)
        info.size, info.mode = len(data), mode
        self.tar.addfile(info, io.BytesIO(data))
        self.members[name] = dict(sha256=digest(data), bytes=len(data))

    def file(self, name: str, path: Path):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Expected regular artifact: {path}")
        safe_name(name)
        if name in self.members:
            raise ValueError(f"Duplicate bundle member: {name}")
        before = path.stat()
        info = tarfile.TarInfo(name)
        info.size = before.st_size
        with path.open("rb") as stream:
            hashed = HashedReader(stream)
            self.tar.addfile(info, hashed)
        after = path.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError(f"Artifact changed while packaging: {path}")
        self.members[name] = dict(sha256=hashed.digest.hexdigest(), bytes=before.st_size)

    def stream(self, name: str, source, size: int, *, mode=0o644):
        """Copy an archive member without loading its whole body into memory."""
        safe_name(name)
        if name in self.members:
            raise ValueError(f"Duplicate bundle member: {name}")
        info = tarfile.TarInfo(name)
        info.size, info.mode = size, mode
        hashed = HashedReader(source)
        self.tar.addfile(info, hashed)
        self.members[name] = dict(sha256=hashed.digest.hexdigest(), bytes=size)

    def finish(self) -> dict:
        self.add("bundle-members.json", encoded(self.members))
        self.tar.close()
        self.compressed.close()
        self.raw.close()
        result = dict(sha256=sha256(self.path), bytes=self.path.stat().st_size,
                      files=len(self.members), raw_bytes=sum(x["bytes"] for x in self.members.values()))
        result["parts"] = split_bundle(self.path)
        return result


class Sources:
    def __init__(self, path: Path, *, exclude_scratch=False, external_blobs=None, exclude_dirs=(), exclude_build=False):
        self.bundle = Bundle(path)
        self.blobs, self.trees = set(), set()
        self.states, self.timeline = {}, []
        self.text_states = {}
        self.archive_bytes = 0
        self.exclude_scratch = exclude_scratch
        self.exclude_dirs = set(exclude_dirs)
        self.exclude_build = exclude_build
        self.input_archives = []
        self.external_blobs = external_blobs or {}
        self.used_external = {}

    def archive(self, label: str, path: Path, *, archive_sha256=None, source_sha256=None) -> str:
        state, identity = read_archive(path, remove_scratch=self.exclude_scratch, exclude_dirs=self.exclude_dirs, exclude_build=self.exclude_build)
        return self.prepared_archive(label, path, state, identity,
                                     archive_sha256=archive_sha256, source_sha256=source_sha256)

    def prepared_archive(self, label: str, path: Path, state: dict, identity: dict, *,
                         file_hashes=None, archive_sha256=None, source_sha256=None) -> str:
        """Store an already-read archive in the same deterministic order."""
        if archive_sha256 and identity["archive_sha256"] != archive_sha256:
            raise ValueError(f"Source archive changed: {path}")
        if source_sha256 and identity["source_sha256"] != source_sha256:
            raise ValueError(f"Checkpoint archive changed: {path}")
        self.input_archives.append(dict(state=label, **identity))
        tree, texts = {}, {}
        for name, (kind, data, mode) in state.items():
            safe_name(name)
            if self.exclude_dirs.intersection(PurePosixPath(name).parts) or (self.exclude_build and is_build_artifact(name)):
                continue
            if any(part in EXCLUDED for part in PurePosixPath(name).parts):
                raise ValueError(f"Unexpected source archive member: {name}")
            if kind == "file":
                key = file_hashes[name] if file_hashes is not None else digest(data)
                if key in self.external_blobs:
                    self.used_external[key] = self.external_blobs[key]
                elif key not in self.blobs:
                    self.bundle.add(f"blobs/{key}", data)
                    self.blobs.add(key)
                tree[name] = ["file", mode, key]
                if label in {"before", "submitted", "after"} and name.endswith(".lean") and not any(part.startswith(".") for part in PurePosixPath(name).parts):
                    texts[name] = data.decode("utf-8")
            elif kind == "symlink":
                tree[name] = ["symlink", mode, data]
        tree_bytes = encoded(tree)
        key = digest(tree_bytes)
        if key not in self.trees:
            self.bundle.add(f"trees/{key}.json", tree_bytes)
            self.trees.add(key)
        self.states[label] = key
        if label in {"before", "submitted", "after"}:
            self.text_states[label] = texts
        self.archive_bytes += path.stat().st_size
        return key

    def finish(self) -> dict:
        self.bundle.add("sources.json", encoded(dict(schema_version=1, states=self.states, timeline=self.timeline,
                                                    excluded=sorted(self.exclude_dirs | ({".scratch"} if self.exclude_scratch else set())),
                                                    input_archives=self.input_archives,
                                                    external_blobs=self.used_external,
                                                    excluded_build_artifacts=self.exclude_build)))
        result = self.bundle.finish()
        return dict(**result, states=self.states, checkpoints=len(self.timeline),
                    unique_trees=len(self.trees), unique_files=len(self.blobs),
                    input_archive_bytes=self.archive_bytes)


def prepare_archive(args):
    path, remove_scratch, exclude_dirs, exclude_build = args
    state, identity = read_archive(path, remove_scratch=remove_scratch, exclude_dirs=exclude_dirs, exclude_build=exclude_build)
    hashes = {name: digest(data) for name, (kind, data, _mode) in state.items() if kind == "file"}
    return path, state, identity, hashes


def prepared_archives(paths, *, workers=8, remove_scratch=False, exclude_dirs=(), exclude_build=False):
    """Read/hash large checkpoints concurrently with bounded memory and ordered output."""
    source = iter(paths)
    executor = ProcessPoolExecutor if exclude_dirs or exclude_build else ThreadPoolExecutor
    with executor(max_workers=workers) as pool:
        pending = deque()
        for _ in range(workers):
            path = next(source, None)
            if path is None:
                break
            pending.append(pool.submit(prepare_archive, (path, remove_scratch, exclude_dirs, exclude_build)))
        while pending:
            result = pending.popleft().result()
            yield result
            # Only submit after the caller has consumed the previous state.
            path = next(source, None)
            if path is not None:
                pending.append(pool.submit(prepare_archive, (path, remove_scratch, exclude_dirs, exclude_build)))


@contextmanager
def open_bundle(path: Path):
    with io.BufferedReader(PartReader(bundle_parts(path))) as raw, zstandard.ZstdDecompressor().stream_reader(raw) as stream:
        with tarfile.open(mode="r|", fileobj=stream) as archive:
            yield archive


def bundle_json(path: Path, name: str):
    with open_bundle(path) as archive:
        for member in archive:
            if member.name == name and member.isfile():
                return json.load(archive.extractfile(member))
    raise ValueError(f"Missing bundle member: {name}")


def restore_sources(path: Path, state: str, output: Path, expected_sha256: str, *, baseline: Path | None = None) -> dict:
    """Restore one tree to a new directory; never follow stored symlinks."""
    if bundle_checksum(path) != expected_sha256:
        raise ValueError("Bundle checksum mismatch")
    index = bundle_json(path, "sources.json")
    key = index["states"][state]
    tree = bundle_json(path, f"trees/{key}.json")
    if digest(encoded(tree)) != key:
        raise ValueError("Tree checksum mismatch")
    for name, (kind, mode, value) in tree.items():
        safe_name(name)
        if kind not in {"file", "symlink"} or not isinstance(mode, int) or not 0 <= mode <= 0o7777:
            raise ValueError("Invalid source tree entry")
        for parent in PurePosixPath(name).parents:
            if str(parent) in tree:
                raise ValueError("Source entry used as a parent directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=output.parent) as temporary:
        root = Path(temporary) / "tree"
        root.mkdir()
        needed = {}
        for name, (kind, mode, value) in tree.items():
            if kind == "file":
                needed.setdefault(value, []).append((name, mode))
        for key, relative in index.get("external_blobs", {}).items():
            if key not in needed:
                continue
            if baseline is None:
                raise ValueError("This checkpoint requires the pinned baseline dataset")
            source = baseline / safe_name(relative)
            if source.is_symlink() or not source.resolve().is_relative_to(baseline.resolve()):
                raise ValueError("Baseline file escapes its project")
            data = source.read_bytes()
            if digest(data) != key:
                raise ValueError("Pinned baseline file checksum mismatch")
            for name, mode in needed.pop(key):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
                target.chmod(mode)
        with open_bundle(path) as archive:
            for member in archive:
                if not member.name.startswith("blobs/") or member.name[6:] not in needed:
                    continue
                data = archive.extractfile(member).read()
                key = member.name[6:]
                if digest(data) != key:
                    raise ValueError("Blob checksum mismatch")
                for name, mode in needed.pop(key):
                    target = root / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                    target.chmod(mode)
        if needed:
            raise ValueError("Missing source blobs")
        for name, (kind, mode, value) in tree.items():
            if kind == "symlink":
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.symlink_to(value)
        root.rename(output)
    return dict(files=len(tree), tree=index["states"][state])


def origin_summary(model: dict, root: Path) -> dict:
    samples = []
    for row in model["results"]:
        if row.get("status", "verified") != "verified":
            continue
        with gzip.open(root / model["id"] / str(row["id"]) / "detail.json.gz", "rt") as stream:
            detail = json.load(stream)
        if detail.get("origins") is not None and detail.get("baseline", 0) > 0 and detail.get("status", "verified") == "verified":
            samples.append({key: 100 * value / detail["baseline"] for key, value in detail["origins"].items()})
    keys = {key for sample in samples for key in sample}
    return dict(count=len(samples), mean_percentage_points={
        key: sum(sample.get(key, 0) for sample in samples) / len(samples) for key in sorted(keys)
    })


def output_tokens_summary(model: dict, root: Path) -> dict:
    samples = []
    for row in model["results"]:
        with gzip.open(root / model["id"] / str(row["id"]) / "detail.json.gz", "rt") as stream:
            tokens = (json.load(stream).get("cost_breakdown") or {}).get("tokens", {})
        output, reasoning = tokens.get("output"), tokens.get("reasoning")
        if output is not None or reasoning is not None:
            samples.append((output or 0) + (reasoning or 0))
    return dict(samples=len(samples), mean=sum(samples) / len(samples) if samples else None)


def artifact_files(root: Path):
    """Walk analysis artifacts, pruning workspaces before descending into them."""
    if not root.is_dir():
        return
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in EXCLUDED | {"repository", "native", "src"}
                         and not (Path(directory) / d).is_symlink())
        for name in sorted(files):
            path = Path(directory) / name
            if path.is_file() and not path.is_symlink() and name != "original-sources.tar.gz":
                yield path
