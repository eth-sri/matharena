"""Opt-in native lean-strip/Podman test; requires the LeanLean extra and cached Lean.

LEANLEAN_STRIP_TEST_ROOT chooses a persistent, fresh workspace. Optionally set
LEANLEAN_STRIP_TEST_STORAGE to reuse an existing Podman store.
"""

import os
import subprocess
from pathlib import Path

import pytest

from matharena.monitoring.commands import CONTAINER_SOCKET, SOCKET_ENV
from leanlean.common import read_json, sha256, tree_sha256, write_json
from leanlean.evaluation import load_problems, prepare_workspace
from test_leanlean_monitoring import AttemptMonitor
from leanlean.prepare_eval import prepare
from leanlean.strip import strip_snapshot


@pytest.mark.skipif(not os.environ.get("LEANLEAN_STRIP_TEST_ROOT"), reason="requires native lean-strip and Podman")
def test_native_container_strip_verify_bundle():
    root = Path(os.environ["LEANLEAN_STRIP_TEST_ROOT"]).resolve()
    snapshot = root / "snapshot"
    key = "PALOMAR-2026-10-18-999999-v1"
    source = snapshot / "sources" / key
    source.mkdir(parents=True, exist_ok=True)
    files = {
        "Solution.lean": "def unused : Nat := 37\ntheorem demo : True := True.intro\n",
        "Challenge.lean": "theorem demo : True := by sorry\n",
        "lean-toolchain": "leanprover/lean4:v4.35.0-rc3\n",
        "lakefile.toml": 'name = "NativeSmoke"\nversion = "0.1.0"\ndefaultTargets = ["Solution"]\n'
        '[[lean_lib]]\nname = "Solution"\n[[lean_lib]]\nname = "Challenge"\n',
        "formalization.yaml": "title: Synthetic native-container integration test\n",
        "LICENSE": "Synthetic MathArena test fixture\n",
    }
    for name, content in files.items():
        (source / name).write_text(content)
    write_json(
        source / "comparator.json",
        {
            "challenge_module": "Challenge",
            "solution_module": "Solution",
            "theorem_names": ["demo"],
            "definition_names": [],
            "permitted_axioms": ["propext", "Quot.sound", "Classical.choice"],
            "enable_nanoda": True,
        },
    )
    entry = {
        "key": key,
        "id": "PALOMAR-2026-10-18-999999",
        "version": 1,
        "fetch_status": "ready",
        "title": "Synthetic native-container integration test (not a registry entry)",
        "github_url": "https://github.com/example/synthetic-test",
        "commit": "a" * 40,
        "source": {"repository": "example/synthetic-test", "commit": "a" * 40, "license": {"path": "LICENSE"}},
        "project_path": "",
        "solution_path": "Solution.lean",
        "challenge_path": "Challenge.lean",
        "comparator_path": "comparator.json",
        "metadata_path": "formalization.yaml",
        "toolchain_path": "lean-toolchain",
        "lakefile_path": "lakefile.toml",
        "lean_toolchain": "leanprover/lean4:v4.35.0-rc3",
        "file_hashes": {name: sha256(source / name) for name in (*files, "comparator.json")},
        "source_tree_sha256": tree_sha256(source),
    }
    record = snapshot / "records" / f"{key}.json"
    write_json(record, {"synthetic_test_fixture": True})
    entry["record_sha256"] = sha256(record)
    write_json(snapshot / "entries" / f"{key}.json", entry)
    write_json(snapshot / "snapshot.json", {"schema_version": 1, "entries": [entry]})
    storage = os.environ.get("LEANLEAN_STRIP_TEST_STORAGE")
    result = strip_snapshot(
        snapshot,
        root / "run",
        workers=1,
        cpus=2,
        memory="8g",
        timeout=300,
        storage_root=Path(storage).resolve() if storage else None,
    )
    assert result["status"] == "complete", read_json(root / "run/summary.json")
    project = root / "run/bundle/projects" / key
    assert "unused" not in (project / "Solution.lean").read_text()
    assert (project / "Challenge.lean").read_text() == files["Challenge.lean"]
    status = read_json(root / "run/results" / key / "status.json")
    assert status["verification"]["status"] == "accepted"
    assert load_problems(str(root / "run/bundle"))[0]["leanlean_project"]["key"] == key
    print(f"Native container bundle: {root / 'run/bundle'}")

    # Package the exact native runtime and check it with no network, using
    # the same tools and command collector as an evaluation preflight.
    runtime = prepare(root / "run", root / "run/bundle", root / "eval", None, command_monitoring="wrapper")
    workspace = root / "preflight"
    workspace.mkdir()
    prepare_workspace(workspace, load_problems(str(root / "run/bundle"))[0]["leanlean_project"])
    with AttemptMonitor(workspace, command_backend="wrapper") as monitor:
        result = subprocess.run(
            [
                runtime["container_engine"],
                "run",
                "--rm",
                "--init",
                "--network=none",
                "--read-only",
                "--user=0:0",
                "--cpus=2",
                "--memory=8g",
                "--pids-limit=-1",
                *runtime["docker_extra_args"],
                "--tmpfs=/tmp:rw",
                "--volume",
                f"{workspace}:/work:rw",
                "--volume",
                f"{monitor.socket_path}:{CONTAINER_SOCKET}:ro",
                "--env",
                f"{SOCKET_ENV}={CONTAINER_SOCKET}",
                "--env",
                "LEAN_NUM_THREADS=2",
                runtime["docker_image"],
                "sh",
                "-c",
                "python proof_length.py && ./check.sh",
            ],
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        (workspace / "stdout.log").write_text(result.stdout)
        (workspace / "stderr.log").write_text(result.stderr)
        assert result.returncode == 0, result.stdout + result.stderr
        monitor.assert_ready()
    assert monitor.summary["complete"], monitor.summary
    assert monitor.summary["totals"]["preflight"]["lake_build"]["commands"] >= 1
    print(f"Offline preflight traces: {monitor.output}")
