import pytest

from leanlean.common import write_json
from leanlean.local_telemetry import split_telemetry
from leanlean.publication import Bundle, open_bundle
from leanlean.release import refresh_files, verify


def members(path):
    with open_bundle(path) as archive:
        return {member.name: (archive.extractfile(member).read(), member.mode)
                for member in archive if member.name != "bundle-members.json"}


def test_local_telemetry_split_preserves_exact_members_and_keeps_graphs(tmp_path):
    folder = tmp_path / "model/artifacts/1"
    folder.mkdir(parents=True)
    archive = Bundle(folder / "analysis.tar.zst")
    originals = {
        "telemetry/resources.jsonl": (b'{"cpu":12,"ram":1024}\n' * 1000, 0o644),
        "telemetry/summary.json": (b'{"commands":[]}', 0o644),
        "telemetry/sources/before--after.diff": (b"+exact source diff\n", 0o644),
        "analysis/timing.json": (b'{"build":12}', 0o644),
        "verification/after/dependency-graph.json": (b'{"nodes":[],"edges":[]}', 0o644),
        "verification/check.sh": (b"#!/bin/sh\n", 0o755),
    }
    for name, (data, mode) in originals.items():
        archive.add(name, data, mode)
    write_json(folder / "manifest.json", dict(model="sol", id=1, analysis=archive.finish()))
    row = split_telemetry(folder)
    kept = members(folder / "analysis.tar.zst")
    local = members(folder / "telemetry.tar.zst")
    assert {**kept, **local} == originals
    assert set(local) == {name for name in originals if name.startswith("telemetry/")}
    assert row["telemetry"]["local_only"]
    assert split_telemetry(folder) == row
    assert not list(folder.parent.glob(".telemetry-*"))
    row["checkpoints"] = dict(checkpoints=1)
    write_json(folder / "manifest.json", row)
    manifest = dict(results=[row], scope=dict(telemetry_local_only=True))
    refresh_files(tmp_path, manifest)
    relative = "model/artifacts/1/telemetry.tar.zst"
    assert relative in manifest["local_files"] and relative not in manifest["files"]
    assert verify(tmp_path)["results"] == 1
    (folder / "telemetry.tar.zst").unlink()
    assert verify(tmp_path)["results"] == 1


def test_modified_analysis_is_rejected_before_originals_are_replaced(tmp_path):
    folder = tmp_path / "result"
    folder.mkdir()
    archive = Bundle(folder / "analysis.tar.zst")
    archive.add("telemetry/resources.jsonl", b"original")
    write_json(folder / "manifest.json", dict(analysis=archive.finish()))
    (folder / "analysis.tar.zst").write_bytes(b"damaged")
    with pytest.raises(ValueError, match="changed before telemetry split"):
        split_telemetry(folder)
    assert (folder / "analysis.tar.zst").read_bytes() == b"damaged"
