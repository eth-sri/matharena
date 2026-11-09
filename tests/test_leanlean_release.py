import gzip
import json
import tarfile

import pytest

from leanlean.publication import Sources, bundle_json, digest, restore_sources
from leanlean.release import graph_receipt, post_archive, read_post, website_archive, write_tree
from leanlean.website_export import packed


def test_post_is_plain_extractable_source_tree(tmp_path):
    state = {"A.lean": ("file", b"exact\r\n", 0o644), "run.sh": ("file", b"run", 0o755),
             "link": ("symlink", "A.lean", 0o777)}
    path = tmp_path / "post.tar.zst"
    receipt = post_archive(state, path)
    assert receipt["files"] == 3
    assert read_post(path) == state
    write_tree(read_post(path), tmp_path / "restored")
    assert (tmp_path / "restored/A.lean").read_bytes() == b"exact\r\n"
    assert (tmp_path / "restored/run.sh").stat().st_mode & 0o777 == 0o755
    assert (tmp_path / "restored/link").is_symlink()
    with pytest.raises(FileExistsError):
        write_tree(state, tmp_path / "restored")


def test_checkpoints_reuse_baseline_and_exclude_scratch(tmp_path):
    import io

    baseline = tmp_path / "pre"
    baseline.mkdir()
    (baseline / "unchanged.lean").write_bytes(b"shared source")
    original = tmp_path / "snapshot.tar.gz"
    with tarfile.open(original, "w:gz") as archive:
        for name, data in {"unchanged.lean": b"shared source", "changed.lean": b"new source", ".scratch/cache": b"discard"}.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    path = tmp_path / "checkpoints.tar.zst"
    sources = Sources(path, exclude_scratch=True, external_blobs={digest(b"shared source"): "unchanged.lean"})
    sources.archive("checkpoint-0", original)
    receipt = sources.finish()
    index = bundle_json(path, "sources.json")
    assert index["external_blobs"] == {digest(b"shared source"): "unchanged.lean"}
    assert receipt["unique_files"] == 1
    restore_sources(path, "checkpoint-0", tmp_path / "restored", receipt["sha256"], baseline=baseline)
    assert (tmp_path / "restored/unchanged.lean").read_bytes() == b"shared source"
    assert (tmp_path / "restored/changed.lean").read_bytes() == b"new source"
    assert not (tmp_path / "restored/.scratch").exists()
    (baseline / "unchanged.lean").write_bytes(b"wrong source")
    with pytest.raises(ValueError, match="baseline file checksum"):
        restore_sources(path, "checkpoint-0", tmp_path / "wrong", receipt["sha256"], baseline=baseline)


def test_invalid_utf8_lean_checkpoint_is_preserved_as_bytes(tmp_path):
    import io

    raw = b"interrupted edit\n\xa0\xff"
    snapshot = tmp_path / "original.tar.gz"
    with tarfile.open(snapshot, "w:gz") as archive:
        member = tarfile.TarInfo("A.lean")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    path = tmp_path / "checkpoints.tar.zst"
    sources = Sources(path, exclude_scratch=True)
    sources.archive("checkpoint-0", snapshot)
    receipt = sources.finish()
    restore_sources(path, "checkpoint-0", tmp_path / "restored", receipt["sha256"])
    assert (tmp_path / "restored/A.lean").read_bytes() == raw


def test_graph_provenance_uses_matching_completed_extraction_over_newer_failure(tmp_path):
    from leanlean.common import sha256, write_json
    graph = tmp_path / 'verification/after/dependency-graph.json'
    graph.parent.mkdir(parents=True)
    graph.write_text('{"actual":true}')
    completed = tmp_path / 'graphs-v1.1.5/model/P1-after.json'
    write_json(completed, dict(status='passed', graph_path=str(graph), graph_sha256=sha256(graph)))
    write_json(tmp_path / 'analysis-new/model/graphs/P1-after.json', dict(status='failed'))
    write_json(tmp_path / 'analysis-invalid/model/graphs/P1-after.json',
               dict(status='passed', graph_path=str(graph), graph_sha256='wrong'))
    assert graph_receipt(tmp_path, 'model', 1, 'after', graph) == completed


def test_parallel_checkpoint_preparation_preserves_exact_bundle_bytes(tmp_path):
    import io
    from leanlean.publication import prepared_archives
    paths = []
    for n in range(3):
        path = tmp_path / f'source-{n}.tar.gz'
        with tarfile.open(path, 'w:gz') as archive:
            for name, data in {'A.lean': f'def value := {n}\n'.encode(),
                               'binary': b'\xff\x00' * 10000,
                               '.scratch/temp': b'excluded'}.items():
                member = tarfile.TarInfo(name)
                member.size = len(data)
                member.mode = 0o755 if name == 'binary' else 0o644
                archive.addfile(member, io.BytesIO(data))
        paths.append(path)
    serial = Sources(tmp_path / 'serial.tar.zst', exclude_scratch=True)
    for n, path in enumerate(paths):
        serial.archive(f'checkpoint-{n}', path)
    expected = serial.finish()
    parallel = Sources(tmp_path / 'parallel.tar.zst', exclude_scratch=True)
    for n, (path, state, identity, hashes) in enumerate(prepared_archives(paths, workers=2, remove_scratch=True)):
        parallel.prepared_archive(f'checkpoint-{n}', path, state, identity, file_hashes=hashes,
                                  source_sha256=identity['source_sha256'])
    actual = parallel.finish()
    assert actual['sha256'] == expected['sha256']
    assert actual['states'] == expected['states']
