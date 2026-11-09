import gzip

import pytest

from leanlean.common import read_json, sha256, write_json
from leanlean.output_layout import prepare_outputs, publish_outputs
from leanlean.publication import Sources, digest, encoded
from leanlean.release import post_archive, refresh_files, restore, verify
from leanlean.result_trace import split_file
from matharena.json_zst import dump_json_zst, load_json_zst
from matharena.monitoring.attempt import archive_bytes, read_tree


def release(tmp_path):
    root = tmp_path / "release"
    folder = root / "results/gpt-61-sol-max/1"
    folder.mkdir(parents=True)
    original = dict(messages=[[dict(role="assistant", content="original transcript")]],
                    problem="problem", detailed_costs=[dict(cost=1.25)], correct=[True])
    dump_json_zst(original, folder / "result.json.zst")
    record = split_file(folder / "result.json.zst")
    summary = load_json_zst(folder / "result.json.zst")
    summary["leanlean"] = dict(verified=True, compression_percent=25)
    dump_json_zst(summary, folder / "result.json.zst")
    record["result"].update(sha256=sha256(folder / "result.json.zst"), bytes=(folder / "result.json.zst").stat().st_size)
    state = {"Proof.lean": ("file", b"theorem test : True := by trivial\n", 0o644)}
    record.update(model="gpt-61-sol-max", id=1, key="project", post=post_archive(state, folder / "post.tar.zst"))
    raw = tmp_path / "source.tar.gz"
    raw.write_bytes(gzip.compress(archive_bytes(state), mtime=0))
    sources = Sources(folder / "checkpoints.tar.zst")
    sources.archive("checkpoint-0", raw)
    record["checkpoints"] = sources.finish()
    write_json(folder / "manifest.json", record)
    write_json(root / "benchmark-manifest.json", {})
    write_json(root / "validation.json", dict(results=1, checkpoint_positions=1))
    write_json(root / "READY.json", dict(results=1))
    refresh_files(root, dict(results=[record], scope=dict(normalized_traces_separate=True)))
    return root, original, state


def test_normal_outputs_keep_transcripts_and_restore_source_artifacts(tmp_path):
    source, original, state = release(tmp_path)
    destination = tmp_path / "outputs/leanlean/october-2026"
    destination.mkdir(parents=True)
    (destination / "previous.json").write_text("previous output")
    backup = tmp_path / "local-backup"
    checked = publish_outputs(source, destination, backup, workers=1)
    assert checked["results"] == 1
    result = load_json_zst(destination / "openai/gpt-61-sol/1.json.zst")
    assert result == dict(original, leanlean=dict(verified=True, compression_percent=25))
    assert (backup / "previous.json").read_text() == "previous output"
    manifest = read_json(destination / "manifest.json")
    row = manifest["results"][0]
    assert row["artifact_directory"] == "openai/gpt-61-sol/artifacts/1"
    assert row["original_canonical_json_sha256"] == digest(encoded(original))
    assert "agent_trace" not in row
    assert not list(destination.rglob("agent-trace.json.zst"))
    assert verify(destination)["results"] == 1
    restore(destination, "gpt-61-sol-max", 1, "post", tmp_path / "restored", tmp_path / "dataset")
    assert read_tree(tmp_path / "restored") == state
    assert verify(source)["results"] == 1
    checkpoints = destination / row["artifact_directory"] / "checkpoints.tar.zst"
    relative = checkpoints.relative_to(destination).as_posix()
    assert relative in manifest["local_files"]
    assert relative not in manifest["files"]
    checkpoints.write_bytes(b"damaged local checkpoint")
    with pytest.raises(ValueError, match="Release file changed"):
        verify(destination)
    checkpoints.unlink()
    assert verify(destination)["results"] == 1


def test_output_publication_requires_completed_source_audit(tmp_path):
    source, _, _ = release(tmp_path)
    (source / "READY.json").unlink()
    destination = tmp_path / "outputs"
    with pytest.raises(ValueError, match="audit must finish"):
        prepare_outputs(source, destination)
    assert not destination.exists()
