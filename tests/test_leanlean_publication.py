"""A publication must restore bytes, preserve identities, and serve summaries cheaply."""

import gzip
import io
import tarfile

import pytest

from leanlean.publication import Bundle, Sources, digest, encoded, restore_sources
from leanlean.website_export import packed


def snapshot(path, files):
    with tarfile.open(path, "w:gz") as archive:
        for name, value in files.items():
            member = tarfile.TarInfo(name)
            if isinstance(value, tuple):
                member.type, member.linkname = tarfile.SYMTYPE, value[0]
                archive.addfile(member)
            else:
                member.size, member.mode = len(value), 0o755 if name == "run.sh" else 0o644
                archive.addfile(member, io.BytesIO(value))


def test_restore_complete_states_with_binary_files_modes_links_and_reverts(tmp_path):
    first, second = tmp_path / "a.tar.gz", tmp_path / "b.tar.gz"
    snapshot(first, {"A.lean": b"original\r\n", "binary": b"\x00\xff", "run.sh": b"#!/bin/sh\n", "link": ("A.lean",)})
    snapshot(second, {"A.lean": b"modified", "binary": b"\x00\xfe", "new": b"new"})
    bundle = tmp_path / "sources.tar.zst"
    sources = Sources(bundle)
    before = sources.archive("before", first)
    after = sources.archive("after", second)
    assert sources.archive("revert", first) == before != after
    receipt = sources.finish()
    assert receipt["unique_trees"] == 2
    for label, expected in [("before", b"original\r\n"), ("after", b"modified"), ("revert", b"original\r\n")]:
        output = tmp_path / label
        restore_sources(bundle, label, output, receipt["sha256"])
        assert (output / "A.lean").read_bytes() == expected
    assert (tmp_path / "before/binary").read_bytes() == b"\x00\xff"
    assert (tmp_path / "before/run.sh").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "before/link").is_symlink()
    assert not (tmp_path / "after/run.sh").exists()
    with pytest.raises(FileExistsError):
        restore_sources(bundle, "before", tmp_path / "before", receipt["sha256"])
    with pytest.raises(ValueError, match="checksum"):
        restore_sources(bundle, "before", tmp_path / "tampered", "0" * 64)


def test_restore_reads_split_compressed_bundle(tmp_path, monkeypatch):
    import leanlean.publication as publication

    monkeypatch.setattr(publication, "PART_SIZE", 50)
    original = tmp_path / "original.tar.gz"
    snapshot(original, {"A.lean": b"theorem a : True := by trivial"})
    source = Sources(tmp_path / "sources.tar.zst")
    source.archive("before", original)
    receipt = source.finish()
    assert receipt["parts"] and not (tmp_path / "sources.tar.zst").exists()
    restore_sources(tmp_path / "sources.tar.zst", "before", tmp_path / "restored", receipt["sha256"])
    assert (tmp_path / "restored/A.lean").read_bytes() == b"theorem a : True := by trivial"


def test_restorer_rejects_writes_through_symlink_ancestors(tmp_path):
    tree = {"outside": ["symlink", 0o777, str(tmp_path)], "outside/escape": ["file", 0o644, "fake"]}
    key = digest(encoded(tree))
    bundle = Bundle(tmp_path / "bad.tar.zst")
    bundle.add(f"trees/{key}.json", encoded(tree))
    bundle.add("sources.json", encoded(dict(states={"before": key})))
    receipt = bundle.finish()
    with pytest.raises(ValueError, match="parent"):
        restore_sources(tmp_path / "bad.tar.zst", "before", tmp_path / "restored", receipt["sha256"])
    assert not (tmp_path / "escape").exists()
