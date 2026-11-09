"""Contract tests for explicit selections and verification-gated demo exports."""

import subprocess
import threading
from types import SimpleNamespace

import pytest

from leanlean import palomar_source, strip
from leanlean.common import read_json, sha256, tree_sha256, write_json

KEY = "PALOMAR-2026-08-29-000003-v1"


def demo_entry():
    return {
        "key": KEY,
        "id": "PALOMAR-2026-08-29-000003",
        "version": 1,
        "project_path": "lean",
        "solution_path": "lean/Solution.lean",
        "challenge_path": "lean/Challenge.lean",
        "comparator_path": "lean/comparator.json",
        "metadata_path": "lean/formalization.yaml",
        "github_url": "https://github.com/example/proof",
        "source": {"license": {"path": "LICENSE"}},
    }


def test_explicit_selection_is_frozen_and_cannot_be_replaced(tmp_path):
    selection = tmp_path / "selection.json"
    row = {
        "id": "PALOMAR-2026-08-29-000003",
        "version": 1,
        "path": f"entries/{KEY}.json",
        "source": {"repository": "example/proof", "commit": "a" * 40},
    }
    write_json(selection, {"schema_version": 1, "entries": [row]})
    snapshot = tmp_path / "snapshot"
    first = palomar_source.snapshot_selection(snapshot, selection)
    assert palomar_source.snapshot_selection(snapshot, selection) == first
    write_json(selection, {"schema_version": 1, "entries": [{**row, "source": {**row["source"], "commit": "b" * 40}}]})
    with pytest.raises(ValueError, match="another selection"):
        palomar_source.snapshot_selection(snapshot, selection)
    assert read_json(snapshot / "snapshot.json") == first


def test_october_snapshot_preserves_published_problem_order(tmp_path):
    from pathlib import Path

    selection = Path("configs/leanlean/october-2026.json")
    snapshot = palomar_source.snapshot_selection(tmp_path, selection)
    keys = [palomar_source.entry_key(entry) for entry in snapshot["entries"]]
    assert len(keys) == 32
    assert keys == read_json(selection)["problem_order"]
    assert keys[16] == "PALOMAR-2026-10-04-000005-v1"


def test_bundle_keeps_sources_and_attribution_but_excludes_build_caches(tmp_path):
    entry = demo_entry()
    snapshot, output = tmp_path / "snapshot", tmp_path / "run"
    source = snapshot / "sources" / KEY
    (source / "lean").mkdir(parents=True)
    (source / "LICENSE").write_text("attribution")
    (source / "lean/formalization.yaml").write_text("metadata")
    write_json(snapshot / "records" / f"{KEY}.json", {"id": entry["id"]})
    work = output / "work" / KEY
    (work / "lean/.lake").mkdir(parents=True)
    (work / "lean/.lean-strip").mkdir()
    (work / "lean/.lake/build.bin").write_bytes(b"cache")
    (work / "lean/.lean-strip/original.lean").write_text("unstripped")
    (work / "lean/Solution.lean").write_text("stripped")
    for name in ("status.json", "strip-report.json"):
        write_json(output / "results" / KEY / name, {"status": "verified"})
    result = output / "results" / KEY
    diagnostics = work / "lean/.lean-strip/diagnostics"
    write_json(diagnostics / "dependency-graph.json", {"nodes": ["demo"], "edges": []})
    write_json(result / "verification.json", {"status": "accepted"})
    (result / "run.log").write_text("Comparator accepted\n")
    descriptor = strip.export_project(snapshot, output, entry)
    exported = output / "bundle" / descriptor["project_directory"]
    assert (exported / "lean/Solution.lean").read_text() == "stripped"
    assert not (exported / "lean/.lake").exists()
    assert not (exported / "lean/.lean-strip").exists()
    assert (output / "bundle/provenance" / KEY / "LICENSE").read_text() == "attribution"
    provenance = output / "bundle/provenance" / KEY
    assert (provenance / "diagnostics/dependency-graph.json").read_bytes() == (
        diagnostics / "dependency-graph.json"
    ).read_bytes()
    assert read_json(provenance / "verification.json") == {"status": "accepted"}
    assert (provenance / "run.log").read_text() == "Comparator accepted\n"
    assert descriptor["project_tree_sha256"] == tree_sha256(exported)
    graph = descriptor["dependency_graph"]
    assert graph["path"] == f"provenance/{KEY}/diagnostics/dependency-graph.json"
    assert graph["sha256"] == sha256(diagnostics / "dependency-graph.json")
    assert (graph["nodes"], graph["edges"]) == (1, 0)
    manifest = strip.write_bundle_manifest(output, [entry], {KEY: {"status": "verified"}})
    index = read_json(output / "bundle" / manifest["dependency_graph_index"])
    assert index == {"schema_version": 1, "graphs": [graph], "missing": []}


@pytest.fixture
def prepared(tmp_path):
    entry = demo_entry()
    entry["toolchain_path"] = "lean/lean-toolchain"
    snapshot, output = tmp_path / "snapshot", tmp_path / "run"
    source = snapshot / "sources" / KEY
    for name, content in {
        "lean/Challenge.lean": "challenge",
        "lean/Solution.lean": "solution",
        "lean/comparator.json": "{}",
        "lean/lean-toolchain": "leanprover/lean4:v4.35.0",
        "lean/formalization.yaml": "metadata",
        "LICENSE": "license",
    }.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    entry["file_hashes"] = {
        entry[field]: sha256(source / entry[field]) for field in ("challenge_path", "comparator_path", "toolchain_path")
    }
    write_json(snapshot / "records" / f"{KEY}.json", {"id": entry["id"]})
    settings = {
        "image": "native-image",
        "cpus": 8,
        "memory": "64g",
        "timeout": 60,
        "podman": ["podman", "--root", "/test/store"],
        "elan_home": "/test/elan",
        "lean_strip_package": "/test/package",
    }
    return entry, snapshot, output, settings


def fake_run(monkeypatch, prepared, report, *, reject_final=False, change_challenge=False):
    _, _, output, _ = prepared
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if "lean_strip" in command:
            write_json(output / "work" / KEY / "lean/.lean-strip/report.json", report)
            if change_challenge:
                (output / "work" / KEY / "lean/Challenge.lean").write_text("weakened")
        else:
            if reject_final:
                raise subprocess.CalledProcessError(1, command)
            write_json(
                output / "results" / KEY / "verification.json", {"status": "accepted", "judge": "lake comparator"}
            )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(strip.subprocess, "run", run)
    return calls


@pytest.mark.parametrize("status", ["stripped", "repaired"])
def test_native_cli_then_applied_tree_validation_and_export(prepared, monkeypatch, status):
    entry, snapshot, output, settings = prepared
    calls = fake_run(monkeypatch, prepared, {"status": status, "comparator": {"status": "accepted"}})
    result = strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())
    assert result["status"] == "verified"
    assert result["strip_status"] == status
    command, options = calls[0]
    assert command[1:3] == ["-m", "lean_strip"]
    assert command[command.index("--container") + 1] == "podman"
    assert command[command.index("--image") + 1] == "native-image"
    assert command[command.index("--cpus") + 1] == "8"
    assert "--skip-comparator" not in command
    assert options["env"]["XDG_CACHE_HOME"] == str(output / "cache" / KEY)
    assert "leanlean.verify_project" in calls[1][0]
    assert (output / "bundle/projects" / KEY / "lean/Solution.lean").exists()
    assert read_json(output / "results" / KEY / "strip-report.json")["status"] == status
    # Verified exports resume without calling the tool again.
    assert strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())["reused"]
    assert len(calls) == 2


@pytest.mark.parametrize(
    "report",
    [
        {"status": "discarded"},
        {"status": "stripped", "comparator": {"status": "unavailable"}},
        {"status": "stripped", "comparator": {"status": "rejected"}},
        {"status": "stripped", "dry_run": True, "comparator": {"status": "accepted"}},
    ],
)
def test_zero_exit_without_accepted_output_is_not_exported(prepared, monkeypatch, report):
    entry, snapshot, output, settings = prepared
    calls = fake_run(monkeypatch, prepared, report)
    result = strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())
    assert result["status"] == "failed"
    assert not (output / "bundle/projects" / KEY).exists()
    assert len(calls) == 1


def test_applied_tree_failure_overrules_successful_scratch_report(prepared, monkeypatch):
    entry, snapshot, output, settings = prepared
    report = {"status": "stripped", "lakefile": {"lake_build": {"ok": True}}, "comparator": {"status": "accepted"}}
    fake_run(monkeypatch, prepared, report, reject_final=True)
    result = strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())
    assert result["status"] == "failed" and result["stage"] == "verify"
    assert not (output / "bundle/projects" / KEY).exists()
    assert read_json(output / "results" / KEY / "strip-report.json") == report


def test_protected_inputs_cannot_change(prepared, monkeypatch):
    entry, snapshot, output, settings = prepared
    calls = fake_run(
        monkeypatch, prepared, {"status": "stripped", "comparator": {"status": "accepted"}}, change_challenge=True
    )
    result = strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())
    assert result["status"] == "failed" and "protected input" in result["error"]
    assert len(calls) == 1


def test_retry_retains_failed_source_and_reports(prepared, monkeypatch):
    entry, snapshot, output, settings = prepared
    fake_run(monkeypatch, prepared, {"status": "discarded"})
    strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())
    (output / "work" / KEY / "lean/Solution.lean").write_text("failed attempt retained")
    fake_run(monkeypatch, prepared, {"status": "stripped", "comparator": {"status": "accepted"}})
    assert strip.process_entry(snapshot, output, entry, settings, {}, threading.Event())["status"] == "verified"
    saved = next((output / "attempts" / KEY).iterdir())
    assert (saved / "work/lean/Solution.lean").read_text() == "failed attempt retained"
    assert read_json(saved / "strip-report.json")["status"] == "discarded"


def test_final_check_uses_selected_store_and_native_package(prepared):
    entry, _, output, settings = prepared
    command = strip.verification_command(output, entry, settings)
    assert command[: len(settings["podman"])] == settings["podman"]
    assert f"{settings['lean_strip_package']}:/pipeline/lean_strip:ro" in command
    assert f"{output / 'work' / KEY}:{output / 'work' / KEY}:rw" in command


def test_partial_bundle_is_explicitly_incomplete(tmp_path):
    result = strip.write_bundle_manifest(tmp_path, [demo_entry()], {KEY: {"status": "failed", "error": "build failed"}})
    assert result["status"] == "incomplete"
    assert result["projects"] == []
    assert "build failed" in (tmp_path / "bundle/index.csv").read_text()


def test_graph_index_keeps_reused_source_revision_explicit(tmp_path):
    key = "example-v2"
    provenance = tmp_path / "provenance" / key
    write_json(provenance / "diagnostics/dependency-graph.json", {"nodes": [], "edges": []})
    write_json(provenance / "strip-report.json", {"status": "stripped"})
    write_json(provenance / "status.json", {
        "upstream_source_commit": "a" * 40,
        "verification_origin": "upstream_saved_reports_for_identical_build_inputs",
    })
    entry = {"key": key, "repository": "example/repo", "commit": "b" * 40}
    graph = strip.dependency_graph_record(tmp_path, entry)
    assert graph["source_commit"] == "b" * 40
    assert graph["graph_source_commit"] == "a" * 40
    strip.write_dependency_graph_index(tmp_path, [{**entry, "dependency_graph": graph}, {"key": "missing"}])
    assert read_json(tmp_path / "dependency-graphs.json")["missing"] == ["missing"]
