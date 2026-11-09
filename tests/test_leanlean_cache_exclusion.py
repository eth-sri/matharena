import gzip
import hashlib

import pytest

from leanlean.publication import read_archive
from matharena.monitoring.attempt import AttemptMonitor, archive_bytes, read_tree
from matharena.monitoring.source_files import is_build_artifact


def test_archive_filter_keeps_sources_and_original_hashes(tmp_path):
    state = {"A.lean": ("file", b"theorem x := 1\r\n", 0o644),
             "run": ("file", bytes(range(256)) * 8192, 0o755),
             "link": ("symlink", "A.lean", 0o777),
             ".compress/B.olean": ("file", b"cache" * 1_000_000, 0o644),
             ".compress/helper.lean": ("file", b"source", 0o644),
             ".compress/params.json": ("file", b"{}", 0o644),
             "arbitrary/Build.olean.private": ("file", b"cache", 0o644),
             "nested/.scratch/C.lean": ("file", b"scratch", 0o644)}
    raw = archive_bytes(state)
    packed = gzip.compress(raw, mtime=0)
    path = tmp_path / "input.tar.gz"
    path.write_bytes(packed)
    standard, original = read_archive(path)
    actual, identity = read_archive(path, remove_scratch=True, exclude_build=True)
    assert actual == {k: v for k, v in standard.items() if ".scratch" not in k.split("/") and not is_build_artifact(k)}
    assert identity["source_sha256"] == original["source_sha256"] == hashlib.sha256(raw).hexdigest()
    assert identity["archive_sha256"] == original["archive_sha256"] == hashlib.sha256(packed).hexdigest()
    assert identity["removed_files"] == 3
    path.write_bytes(packed[:-20])
    with pytest.raises((EOFError, ValueError, OSError)):
        read_archive(path, exclude_build=True)


def test_capture_ignores_build_artifacts(tmp_path):
    capture = tmp_path / "capture"
    (capture / ".compress").mkdir(parents=True)
    (capture / ".compress/cache.olean").write_bytes(b"ignored")
    (capture / "A.lean").write_bytes(b"proof")
    (capture / ".compress/helper.lean").write_bytes(b"source")
    (capture / "Elsewhere.olean.private").write_bytes(b"ignored")
    assert read_tree(capture) == {"A.lean": ("file", b"proof", 0o644), ".compress/helper.lean": ("file", b"source", 0o644)}


def test_compiler_activity_does_not_create_source_checkpoints(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / ".compress").mkdir(parents=True)
    (workspace / ".compress/helper.lean").write_text("source")
    with AttemptMonitor(workspace, commands=False, cpu=False, ram=False, interval_s=60) as monitor:
        original_count = monitor.snapshot_count
        (workspace / "Foo.olean").write_bytes(b"compiled")
        (workspace / ".compress/Foo.olean.private").write_bytes(b"private compiled")
        monitor.capture("compiled_only")
        assert monitor.snapshot_count == original_count
        (workspace / ".compress/helper.lean").write_text("changed source")
        monitor.capture("source_change")
        assert monitor.snapshot_count == original_count + 1
