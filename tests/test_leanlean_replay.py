import gzip
import hashlib

import pytest

from leanlean.common import sha256, write_json
from leanlean.replay import cached_result, sync_sources, verify_archive


def put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_deletion_invalidates_local_builds_but_preserves_external_packages(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    put(old, "Removed.lean", "theorem old : True := by trivial")
    put(old, "Kept.lean", "def kept := 1")
    kept = put(new, "Kept.lean", "def kept := 1")
    put(old, ".lake/build/Removed.olean", "stale")
    put(old, "vendor/lib/.lake/build/Removed.olean", "stale")
    package = put(tmp_path, "external/build/Library.olean", "pinned")
    (old / ".lake/packages").symlink_to(package.parent.parent, target_is_directory=True)
    before_mtime = (old / "Kept.lean").stat().st_mtime_ns
    result = sync_sources(new, old)
    assert result == {"deleted": ["Removed.lean"], "changed": [], "build_cache_reset": True}
    assert not (old / ".lake/build").exists()
    assert not (old / "vendor/lib/.lake/build").exists()
    assert package.read_text() == "pinned"
    assert (old / "Kept.lean").stat().st_mtime_ns == before_mtime
    assert (old / "Kept.lean").read_bytes() == kept.read_bytes()


def test_edit_preserves_build_cache_and_reverts_and_path_type_changes(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    put(old, "Solution.lean", "def x := 1")
    artifact = put(old, ".lake/build/Solution.olean", "warm")
    put(new, "Solution.lean", "def x := 2")
    assert not sync_sources(new, old)["build_cache_reset"]
    assert artifact.exists()
    assert (old / "Solution.lean").read_text() == "def x := 2"
    (new / "Solution.lean").write_text("def x := 1")
    sync_sources(new, old)
    assert (old / "Solution.lean").read_text() == "def x := 1"
    (new / "Solution.lean").unlink()
    put(new, "Solution.lean/Child.lean", "def child := 1")
    assert sync_sources(new, old)["build_cache_reset"]
    (new / "Solution.lean/Child.lean").unlink()
    (new / "Solution.lean").rmdir()
    put(new, "Solution.lean", "def x := 3")
    sync_sources(new, old)
    assert (old / "Solution.lean").read_text() == "def x := 3"


def test_unexpected_symlink_cannot_redirect_source_sync(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    target = put(tmp_path, "external/Solution.lean", "original")
    old.mkdir()
    (old / "Solution.lean").symlink_to(target)
    put(new, "Solution.lean", "new")
    with pytest.raises(ValueError, match="symlink"):
        sync_sources(new, old)
    assert target.read_text() == "original"


def test_archive_identity_is_uncompressed_tar_not_gzip_bytes(tmp_path):
    archive = tmp_path / "state.tar.gz"
    content = b"tar payload"
    archive.write_bytes(gzip.compress(content))
    state = dict(
        archive=str(archive), archive_sha256=sha256(archive), source_sha256=hashlib.sha256(content).hexdigest()
    )
    assert verify_archive(state) == archive
    state["source_sha256"] = state["archive_sha256"]
    with pytest.raises(ValueError, match="Uncompressed"):
        verify_archive(state)
    archive.write_bytes(gzip.compress(b"different"))
    with pytest.raises(ValueError, match="Archive changed"):
        verify_archive(state)


def test_resume_requires_untampered_artifacts_and_does_not_skip_infrastructure_errors(tmp_path):
    graph = put(tmp_path, "attempts/0001/dependency-graph.json", "{}")
    state = dict(source_sha256="source", archive_sha256="archive")
    result = dict(**state, status="complete", artifacts={str(graph.relative_to(tmp_path)): sha256(graph)})
    path = tmp_path / "result.json"
    write_json(path, result)
    assert cached_result(path, state) == result
    graph.write_text("changed")
    with pytest.raises(ValueError, match="Cached artifact changed"):
        cached_result(path, state)
    write_json(path, {**result, "status": "error"})
    assert cached_result(path, state) is None
