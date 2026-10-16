"""Performance regressions should be tested as file reads, not timing thresholds."""

import os
from pathlib import Path

import pytest
import yaml

import matharena.configs as configs
import matharena.viewer_cache as cache_module
from matharena.json_zst import dump_json_zst
from matharena.viewer_cache import ViewerResultCache


def test_summaries_survive_restart_and_detect_same_size_rewrites(tmp_path, monkeypatch):
    path = tmp_path / "1.json.zst"
    record = {"correct": [True], "cost": {"cost": 1}, "messages": [["large trace"]], "history": [["history"]]}
    dump_json_zst(record, path)
    cache = ViewerResultCache(tmp_path / "cache")
    expected = {str(path): {"correct": [True], "cost": {"cost": 1}}}
    assert cache.summaries([path]) == expected
    assert cache.detail(path) == record
    original_load = cache_module.load_json_zst
    loads = []

    def tracked_load(path):
        loads.append(path)
        return original_load(path)

    monkeypatch.setattr(cache_module, "load_json_zst", tracked_load)
    restarted = ViewerResultCache(tmp_path / "cache")
    assert restarted.summaries([path]) == expected
    assert loads == []
    assert cache.detail(path) == record
    assert loads == []

    # Preserve mtime and size: ctime still detects a rewritten result.
    stat = path.stat()
    record["cost"]["cost"] = 2
    dump_json_zst(record, path)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert restarted.summaries([path])[str(path)]["cost"] == {"cost": 2}
    assert cache.detail(path)["cost"] == {"cost": 2}
    assert loads == [path, path]


@pytest.mark.parametrize("broken_database", [False, True])
def test_unavailable_cache_falls_back_to_source_files(tmp_path, broken_database):
    path = tmp_path / "1.json.zst"
    dump_json_zst({"correct": [True]}, path)
    directory = tmp_path / "cache"
    if broken_database:
        directory.mkdir()
        (directory / "results.sqlite3").write_text("not a sqlite database")
    else:
        directory.write_text("not a directory")
    assert ViewerResultCache(directory).summaries([path]) == {str(path): {"correct": [True]}}


@pytest.mark.parametrize("allow_unjudged", [False, True])
def test_config_discovery_only_reads_proofs_when_judgments_are_required(tmp_path, monkeypatch, allow_unjudged):
    model_config = tmp_path / "models/provider/model.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text(yaml.safe_dump({"human_readable_id": "Test Model"}))
    competition_config = tmp_path / "competitions/test/proof.yaml"
    competition_config.parent.mkdir(parents=True)
    competition_config.write_text(yaml.safe_dump({"final_answer": False}))
    output = tmp_path / "outputs/test/proof/provider/model/1.json.zst"
    output.parent.mkdir(parents=True)
    dump_json_zst({"correct": ["TODO Grading"]}, output)
    loads = []
    original_load = configs.load_json_zst

    def tracked_load(path):
        loads.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(configs, "load_json_zst", tracked_load)
    _, ids = configs.extract_existing_configs(
        "test/proof", tmp_path / "outputs", tmp_path / "models", tmp_path / "competitions",
        allow_non_existing_judgment=allow_unjudged,
    )
    assert ids == ({"provider/model": "Test Model"} if allow_unjudged else {})
    assert loads == ([] if allow_unjudged else [output])

    if not allow_unjudged:
        dump_json_zst({"correct": [1], "judgment": []}, output)
        _, ids = configs.extract_existing_configs(
            "test/proof", tmp_path / "outputs", tmp_path / "models", tmp_path / "competitions",
        )
        assert ids == {"provider/model": "Test Model"}
