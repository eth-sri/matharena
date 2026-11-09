"""Cache-layout and provenance checks for offline image preparation."""

import shutil
from pathlib import Path

import pytest

from leanlean import prepare_eval
from leanlean.common import python_package_sha256, read_json, write_json
from leanlean.eval_tools import COMPARATOR_COMMIT, NANODA_COMMIT


@pytest.fixture(params=[False, True], ids=["legacy", "native"])
def prepared_run(tmp_path, monkeypatch, request):
    native = request.param
    run, bundle, output = (tmp_path / name for name in ("run", "bundle", "eval"))
    source = bundle / "projects/example"
    source.mkdir(parents=True)
    write_json(source / "lake-manifest.json", {"packages": []})
    write_json(source / "comparator.json", {"enable_nanoda": True})
    entry = {
        "key": "example",
        "source_directory": str(source),
        "project_path": "",
        "project_tree_sha256": "source-hash",
        "lean_toolchain": "leanprover/lean4:v4.33.0",
        "comparator_path": "comparator.json",
    }
    write_json(bundle / "manifest.json", {"projects": [entry]})
    write_json(run / "bundle/provenance/example/entry.json", entry)
    write_json(
        run / "results/example/status.json",
        {
            "status": "verified",
            "verification": {
                "judge": "comparator 57567492" if native else "standalone comparator",
                "comparator_commit": COMPARATOR_COMMIT,
            },
        },
    )
    package = tmp_path / "lean_strip"
    package.mkdir()
    (package / "__init__.py").write_text('VERSION = "recorded"\n')
    metadata = {
        "schema_version": 2 if native else 1,
        "image": "sha256:runtime",
        "storage_root": str(tmp_path / "podman"),
        "elan_home": str(tmp_path / "elan"),
        "lean_strip_package": str(package),
        "lean_strip_sha256": python_package_sha256(package),
        "comparator_commit": COMPARATOR_COMMIT,
        "nanoda_commit": NANODA_COMMIT,
    }
    write_json(run / "run.json", metadata)
    elan = Path(metadata["elan_home"]) if native else run / "cache/elan"
    toolchain = elan / "toolchains/leanprover--lean4---v4.33.0/bin"
    toolchain.mkdir(parents=True)
    (toolchain / "lean").write_text("cached toolchain")
    cache = run / "cache/example/lean-strip/comparator-tools" if native else run / "cache/tools"
    tools = [
        f"comparator-{COMPARATOR_COMMIT}/.lake/build/bin/comparator",
        f"comparator-{COMPARATOR_COMMIT}/scripts/fake-landrun.sh",
        "lean4export-v4.33.0/.lake/build/bin/lean4export",
        f"nanoda-{NANODA_COMMIT}/target/release/nanoda_bin",
    ]
    for relative in tools:
        cached = (
            relative.replace(COMPARATOR_COMMIT, COMPARATOR_COMMIT[:12]).replace(NANODA_COMMIT, NANODA_COMMIT[:12])
            if native
            else relative
        )
        path = cache / cached
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    commands = []

    def check_output(command, **kwargs):
        commands.append(command)
        return "true" if "info" in command else '[{"Id": "sha256:eval"}]'

    monkeypatch.setattr(prepare_eval, "load_problems", lambda _: [{"leanlean_project": entry}])
    monkeypatch.setattr(prepare_eval, "copy_directory", lambda src, dst: shutil.copytree(src, dst, dirs_exist_ok=True))
    monkeypatch.setattr(prepare_eval.subprocess, "check_output", check_output)
    monkeypatch.setattr(prepare_eval.subprocess, "run", lambda command, **kwargs: commands.append(command))
    return run, bundle, output, native, tools, commands


def test_preparation_materializes_correct_tool_cache(prepared_run):
    run, bundle, output, native, tools, commands = prepared_run
    runtime = prepare_eval.prepare(run, bundle, output, None)
    context = output / "context"
    for relative in tools:
        assert (context / "tools" / relative).read_text() == relative
    assert (context / "packages/example").is_dir()  # no external dependencies
    assert (context / "toolchains/leanprover--lean4---v4.33.0/bin/lean").read_text() == "cached toolchain"
    assert (context / "pipeline/lean_strip/__init__.py").exists() == native
    assert (context / "pipeline/matharena/monitoring/commands.py").exists()
    assert runtime["docker_image"] == "sha256:eval"
    assert commands[0][1:3] == ["--root", str(run.parent / "podman/storage")]
    assert ("lean_strip_sha256" in read_json(output / "inputs.json")) == native


def test_preparation_rejects_missing_dependencies(prepared_run):
    run, bundle, output, _, _, _ = prepared_run
    write_json(bundle / "projects/example/lake-manifest.json", {"packages": [{"name": "mathlib"}]})
    with pytest.raises(ValueError, match="Missing prepared dependencies"):
        prepare_eval.prepare(run, bundle, output, None)


def test_preparation_rejects_changed_native_tool(prepared_run):
    run, bundle, output, native, _, commands = prepared_run
    if not native:
        pytest.skip("Legacy image already contains its stripping tool")
    metadata = read_json(run / "run.json")
    (Path(metadata["lean_strip_package"]) / "__init__.py").write_text('VERSION = "changed"\n')
    with pytest.raises(ValueError, match="lean-strip sources changed"):
        prepare_eval.prepare(run, bundle, output, None)
    assert commands == []


@pytest.mark.parametrize("backend", ["host", "wrapper"])
def test_preparation_records_command_backend_and_build_selection(prepared_run, backend):
    run, bundle, output, _, _, commands = prepared_run
    runtime = prepare_eval.prepare(run, bundle, output, None, command_monitoring=backend)
    assert runtime["command_monitoring_backend"] == backend
    assert runtime["command_trace_version"] == (2 if backend == "wrapper" else None)
    build = next(command for command in commands if "build" in command)
    assert f"COMMAND_MONITORING={backend}" in build
