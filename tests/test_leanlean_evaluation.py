import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from leanlean.common import tree_sha256, write_json
from leanlean.evaluation import capture_submission, load_problems, prepare_workspace
from matharena.runner import Runner
from matharena.eval_run import prepare_run_directory
from leanlean.solver import LeanLeanSolver


def test_runner_shuffle_preserves_project_identity_and_is_reproducible(tmp_path, monkeypatch):
    projects = [{"problem_idx": i, "leanlean_project": {"key": f"project-{i}"}} for i in range(1, 33)]
    monkeypatch.setattr(Runner, "_load_problems", lambda self, ids: [p for p in projects if ids is None or p["problem_idx"] in ids])
    config = tmp_path / "example.yaml"

    def load(seed=None, ids=None):
        values = {"final_answer": False}
        if seed is not None:
            values["shuffle_seed"] = seed
        config.write_text(yaml.safe_dump(values))
        return Runner("example", 1, ids, str(tmp_path), str(tmp_path), str(tmp_path), False).problems

    shuffled = load(20261029)
    assert shuffled != projects
    assert shuffled == load(20261029)
    assert shuffled != load(20261030)
    assert {p["problem_idx"] for p in shuffled} == set(range(1, 33))
    assert all(p is projects[p["problem_idx"] - 1] for p in shuffled)
    assert load() == projects
    assert {p["problem_idx"] for p in load(20261029, [1, 9, 17, 25])} == {1, 9, 17, 25}


@pytest.fixture
def bundle(tmp_path):
    source = tmp_path / "bundle/projects/example"
    project = source / "nested"
    project.mkdir(parents=True)
    (project / "Solution.lean").write_text("theorem t : True := by trivial\n")
    (project / "Challenge.lean").write_text("theorem t : True := by sorry\n")
    entry = {
        "key": "example",
        "status": "verified",
        "title": "Example",
        "github_url": "https://github.com/a/b",
        "commit": "a" * 40,
        "project_directory": "projects/example",
        "project_path": "nested",
        "solution_path": "nested/Solution.lean",
        "challenge_path": "nested/Challenge.lean",
        "comparator_path": "nested/comparator.json",
        "lean_toolchain": "leanprover/lean4:v4.35.0",
        "project_tree_sha256": tree_sha256(source),
    }
    root = tmp_path / "bundle"
    write_json(root / "manifest.json", {"status": "complete", "projects": [entry]})
    return root


def test_loader_selection_and_runner_dispatch(bundle):
    runner = SimpleNamespace(competition_config={"dataset_format": "leanlean", "dataset_path": str(bundle)})
    problem = Runner._load_problems(runner, [1])[0]
    assert problem["problem_idx"] == 1
    assert "/work/repository/nested" in problem["problem"]
    assert load_problems(str(bundle), [2]) == []
    (bundle / "projects/example/nested/Solution.lean").write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        load_problems(str(bundle))


def test_frozen_run_keeps_leanlean_runtime_and_source_identity(bundle, tmp_path):
    from leanlean.common import sha256

    write_json(bundle / "eval-runtime.json", {"bundle_sha256": sha256(bundle / "manifest.json")})
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "example.yaml").write_text(yaml.safe_dump({
        "dataset_format": "leanlean", "dataset_path": str(bundle),
        "final_answer": False, "allow_harness": True,
        "harness_config": {"runtime_manifest": str(bundle / "eval-runtime.json")},
    }))
    root = prepare_run_directory(
        tmp_path / "frozen", competitions=["example"], models=[],
        competition_configs=config_dir, model_configs=config_dir,
    )
    runner = Runner("example", 1, None, str(root / "configs/competitions"),
                    str(root / "configs/models"), str(root / "outputs"), False)
    entry = runner.problems[0]["leanlean_project"]
    original = load_problems(str(bundle))[0]["leanlean_project"]
    assert entry["project_tree_sha256"] == original["project_tree_sha256"]
    assert entry["key"] == original["key"]
    assert Path(entry["source_directory"]).is_relative_to(root / "datasets")
    assert Path(runner.competition_config["harness_config"]["runtime_manifest"]).exists()


def test_workspace_sources_dependencies_and_submission(bundle, tmp_path):
    entry = load_problems(str(bundle))[0]["leanlean_project"]
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    prepare_workspace(workspace, entry)
    project = workspace / "repository/nested"
    assert (project / ".lake/packages").readlink() == Path("/opt/leanlean/packages/example")
    assert "/work/repository/nested" in (workspace / "check.sh").read_text()
    (project / "Solution.lean").write_text("theorem t : True := True.intro\n")
    (project / "external.lean").symlink_to("/etc/passwd")
    submission = capture_submission(workspace, entry)
    with tarfile.open(submission["archive"]) as archive:
        assert "nested/.lake/packages" not in archive.getnames()
        assert archive.getmember("nested/external.lean").issym()
        assert b"True.intro" in archive.extractfile("nested/Solution.lean").read()
    assert "trivial" in (bundle / "projects/example/nested/Solution.lean").read_text()
    assert submission["baseline_sha256"] == entry["project_tree_sha256"]


@pytest.mark.parametrize("tmpfs_size", [None, "4g"])
@pytest.mark.parametrize("container_runtime", ["docker", "podman"])
def test_harness_uses_prepared_image_and_rootless_engine(bundle, tmp_path, tmpfs_size, container_runtime):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        json.dumps(
            {
                "docker_image": "sha256:example", "container_engine": "/tmp/podman-engine", "container_user": "0:0",
                "docker_extra_args": ["--cap-add", "SETFCAP", "--security-opt", "unmask=/proc/*"],
            }
        )
    )
    config = {
        "model": "test",
        "harness": "codex",
        "harness_config": {"leanlean_projects": True, "runtime_manifest": str(runtime), "managed_cli": False},
    }
    if tmpfs_size is not None:
        config["harness_config"]["tmpfs_size"] = tmpfs_size
    solver_config = {"type": "pure_model", "model_config": config, "scaffold_config": None}
    solver = LeanLeanSolver(solver_config, "{problem}", config, "", container_runtime=container_runtime)
    workspace = tmp_path / "workspace"
    problem = load_problems(str(bundle))[0]
    solver._prepare_workspace(workspace, problem)
    sandbox = solver._build_sandbox(workspace)
    assert f"size={tmpfs_size or '16g'}," in sandbox.extra_args[sandbox.extra_args.index("--tmpfs") + 1]
    assert sandbox.image == "sha256:example"
    engine = sandbox.podman_executable if container_runtime == "podman" else sandbox.docker_executable
    assert engine == "/tmp/podman-engine"
    assert sandbox._relay_executable == "/tmp/podman-engine"
    assert sandbox.extra_args[sandbox.extra_args.index("--user") + 1] == "0:0"
    assert sandbox.extra_args[sandbox.extra_args.index("--cap-add") + 1] == "SETFCAP"
    assert "unmask=/proc/*" in sandbox.extra_args
    assert "no-new-privileges:true" in sandbox.extra_args
    assert "--read-only" in sandbox.extra_args
    assert not (workspace / "Solution.lean").exists()
    assert (workspace / "repository/nested/Solution.lean").exists()
    config["harness_config"]["keep_workspaces"] = False
    with pytest.raises(ValueError, match="keep_workspaces"):
        LeanLeanSolver(solver_config, "{problem}", config, "")


def test_runtime_must_match_bundle(bundle):
    write_json(bundle / "eval-runtime.json", {"bundle_sha256": "wrong"})
    with pytest.raises(ValueError, match="does not match"):
        load_problems(str(bundle))


def test_failed_preflight_stops_before_model_access(bundle, tmp_path, monkeypatch):
    config = {
        "model": "test",
        "harness": "codex",
        "harness_config": {"leanlean_projects": True, "workspace_root": str(tmp_path / "attempt")},
    }
    solver = LeanLeanSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None}, "{problem}", config, ""
    )
    monkeypatch.setattr(solver, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(
        solver,
        "_build_sandbox",
        lambda workspace: SimpleNamespace(
            image="test-image",
            _relay_executable="/bin/false",
            container_name="failed-preflight-test",
            mounts=(),
            stop=lambda: None,
            run=lambda *args, **kwargs: SimpleNamespace(returncode=1, stdout="", stderr="bad proof"),
        ),
    )
    monkeypatch.setattr(solver, "_build_model", lambda: pytest.fail("model accessed before preflight"))
    with pytest.raises(RuntimeError, match="preflight failed"):
        solver._solve_one(0, load_problems(str(bundle))[0], None, 1, 0)
    assert (tmp_path / "attempt/preflight_stderr.log").read_text() == "bad proof"


@pytest.mark.parametrize("native", [True, False])
def test_feedback_uses_native_or_pinned_comparator(monkeypatch, tmp_path, native):
    from leanlean import eval_tools

    entry = {
        "key": "example",
        "project_path": "",
        "lean_toolchain": "leanprover/lean4:v4.35.0",
        "challenge_path": "Challenge.lean",
        "comparator_path": "comparator.json",
        "toolchain_path": "lean-toolchain",
        "lakefile_path": "lakefile.toml",
    }
    monkeypatch.setattr(eval_tools, "context", lambda: (entry, Path("/work/repository")))
    monkeypatch.setattr(eval_tools, "sha256", lambda path: "same")
    monkeypatch.setattr(eval_tools, "read_json", lambda path: {"solution_module": "Solution.Main"})
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout="Judge a solution" if native else "", stderr="")

    monkeypatch.setattr(eval_tools.subprocess, "run", run)
    toolchain = tmp_path / "toolchain/bin"
    toolchain.mkdir(parents=True)
    (toolchain / "lean").touch()
    alias = tmp_path / "elan-alias"
    alias.symlink_to(toolchain.parent, target_is_directory=True)
    monkeypatch.setattr(
        eval_tools.subprocess, "check_output", lambda *args, **kwargs: f"{alias}/bin/lean\n"
    )
    eval_tools.check()
    assert calls[0][0] == ["lake", "build", "+Solution.Main"]
    final, options = calls[-1]
    assert final[:3] == ["timeout", "--kill-after=10s", "3600s"]
    final = final[3:]
    if native:
        assert options["env"]["PATH"].startswith(f"{toolchain}:")
        assert final == [
            "lake",
            "comparator",
            "--config",
            "/work/repository/comparator.json",
        ]
    else:
        assert final[:2] == ["lake", "env"]
        assert eval_tools.COMPARATOR_COMMIT in final[2]
        assert "lean4export-v4.35.0" in options["env"]["COMPARATOR_LEAN4EXPORT"]


def test_submission_cannot_follow_workspace_links(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    secret = private / "secret"
    secret.write_text("keep unchanged")
    (workspace / "repository").symlink_to(private, target_is_directory=True)
    with pytest.raises(ValueError, match="not a symlink"):
        capture_submission(workspace, {})
    (workspace / "repository").unlink()
    (workspace / "repository").mkdir()
    (workspace / "leanlean-submission.tar.gz").symlink_to(secret)
    capture_submission(workspace, {"key": "example", "project_tree_sha256": "baseline"})
    assert secret.read_text() == "keep unchanged"
    assert not (workspace / "leanlean-submission.tar.gz").is_symlink()


def test_submission_survives_workspace_replacement(bundle, tmp_path):
    import shutil

    entry = load_problems(str(bundle))[0]["leanlean_project"]
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    prepare_workspace(workspace, entry)
    submission = capture_submission(workspace, entry, output=tmp_path / "telemetry/submission")
    digest = submission["sha256"]
    shutil.rmtree(workspace)
    workspace.mkdir()
    prepare_workspace(workspace, entry)
    from leanlean.common import sha256
    assert sha256(Path(submission["archive"])) == digest


@pytest.mark.parametrize("binary", [False, True])
def test_preflight_timeout_preserves_partial_output(tmp_path, binary):
    import subprocess

    config = {"model": "test", "harness": "codex", "harness_config": {"leanlean_projects": True}}
    solver = LeanLeanSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None}, "{problem}", config, ""
    )

    def timeout(*args, **kwargs):
        stdout, stderr = "Exporting solution\n", "No space left on device\n"
        if binary:
            stdout, stderr = stdout.encode(), stderr.encode()
        raise subprocess.TimeoutExpired("preflight", 3600, output=stdout, stderr=stderr)

    sandbox = SimpleNamespace(run=timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        solver._preflight(tmp_path, sandbox, solver._agent_environment())
    assert (tmp_path / "preflight_stdout.log").read_text() == "Exporting solution\n"
    assert (tmp_path / "preflight_stderr.log").read_text() == "No space left on device\n"


@pytest.mark.parametrize("leanlean_projects", [False, True])
def test_runner_selects_project_solver_only_when_enabled(leanlean_projects):
    config = {
        "model": "test", "harness": "codex",
        "harness_config": {"leanlean_projects": leanlean_projects, "managed_cli": False},
    }
    solver = Runner._initialize_solver(
        SimpleNamespace(container_runtime="docker"), {"harness": "codex"}, "{problem}", config, ""
    )
    from matharena.solvers.harness_solver import HarnessSolver
    assert type(solver) is (LeanLeanSolver if leanlean_projects else HarnessSolver)


@pytest.mark.parametrize("failure", [None, "preflight", "model"])
def test_project_attempt_stops_writers_before_submission_and_closes_on_failure(bundle, tmp_path, monkeypatch, failure):
    from matharena.solvers import SolverResponse
    from matharena.solvers.harness_solver import HarnessSolver

    config = {"model": "test", "harness": "codex", "harness_config": {"managed_cli": False, "monitoring": {"diff": True}}}
    solver = LeanLeanSolver({}, "{problem}", config, "")
    workspace = tmp_path / "attempt"
    workspace.mkdir()
    payload = load_problems(str(bundle))[0]
    prepare_workspace(workspace, payload["leanlean_project"])
    solution = workspace / "repository/nested/Solution.lean"
    lifecycle = []

    class Monitor:
        def __init__(self, *args, **kwargs):
            self.output = tmp_path / "telemetry"
            self.socket_path = tmp_path / "socket"
            self.socket_path.touch()
            self.summary = {"complete": True}

        def __enter__(self):
            return self

        def __exit__(self, *args):
            lifecycle.append("closed")

        def assert_ready(self, **kwargs):
            pass

        def capture(self, reason):
            lifecycle.append(reason)

        def on_event(self, event):
            pass

    def preflight(*args, **kwargs):
        lifecycle.append("preflight")
        if failure == "preflight":
            raise RuntimeError("preflight failed")
        return SimpleNamespace(returncode=0, stdout="ready", stderr="")

    def run_model(self, *args, **kwargs):
        lifecycle.append("model")
        if failure == "model":
            raise RuntimeError("model failed")
        kwargs["after_run"]()
        response = SolverResponse(0, [], {}, history=[{}])
        self._response_agents[id(response)] = object()
        self._response_budgets[id(response)] = object()
        return response

    def stop():
        lifecycle.append("stopped")
        solution.write_text("settled after background writers stopped\n")

    sandbox = SimpleNamespace(
        _relay_executable="unused", container_name="attempt", image="test-image",
        mounts=(), run=preflight, stop=stop,
    )
    monkeypatch.setattr("matharena.solvers.harness_solver.AttemptMonitor", Monitor)
    monkeypatch.setattr(HarnessSolver, "_run_prepared", run_model)
    if failure:
        with pytest.raises(RuntimeError, match=f"{failure} failed"):
            solver._solve_prepared(0, payload, 1, 0, workspace, sandbox)
        if failure == "preflight":
            assert "model" not in lifecycle
    else:
        response = solver._solve_prepared(0, payload, 1, 0, workspace, sandbox)
        assert response.detailed_cost["deferred_grading"]
        assert response.history[0]["monitoring"]["complete"]
        with tarfile.open(response.history[0]["leanlean_submission"]["archive"]) as archive:
            assert archive.extractfile("nested/Solution.lean").read().startswith(b"settled after")
    assert lifecycle[-2:] == ["stopped", "closed"]
    assert not solver._response_agents and not solver._response_budgets


@pytest.mark.parametrize("backend, trace_version", [("host", None), ("wrapper", 2)])
def test_command_monitoring_follows_prepared_image(tmp_path, backend, trace_version):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({
        "command_monitoring_backend": backend, "command_trace_version": trace_version,
        "docker_image": "test-image", "container_engine": "podman", "container_user": "0:0",
    }))
    harness = {"managed_cli": False, "monitoring": {"commands": True}, "runtime_manifest": str(manifest)}
    config = {"harness": "codex", "model": "test", "harness_config": harness}
    solver = LeanLeanSolver({}, "{problem}", config, "", container_runtime="podman")
    assert solver.command_backend == backend
    harness["command_monitoring_backend"] = "wrapper" if backend == "host" else "host"
    with pytest.raises(ValueError, match=f"Prepared eval image uses {backend}"):
        LeanLeanSolver({}, "{problem}", config, "", container_runtime="podman")


def test_host_monitoring_rejects_historical_wrapper_image(tmp_path):
    manifest = tmp_path / "runtime.json"
    manifest.write_text(json.dumps({"command_trace_version": 2}))
    config = {"harness": "codex", "model": "test", "harness_config": {
        "managed_cli": False, "monitoring": {"commands": True}, "runtime_manifest": str(manifest),
    }}
    with pytest.raises(ValueError, match="rebuilding the eval image"):
        LeanLeanSolver({}, "{problem}", config, "")


@pytest.mark.parametrize("monitoring", ["missing", None, {"diff": True}])
def test_frozen_legacy_launches_preserve_full_telemetry_unless_explicitly_overridden(monitoring):
    harness = {"leanlean_projects": True, "managed_cli": False}
    if monitoring != "missing":
        harness["monitoring"] = monitoring
    solver = LeanLeanSolver({}, "{problem}", {"model": "test", "harness": "codex", "harness_config": harness}, "")
    if monitoring == "missing":
        assert solver.monitoring == {"diff": True, "cpu": True, "ram": True, "commands": True}
        assert solver.command_backend == "wrapper"
    elif monitoring is None:
        assert not any(solver.monitoring.values())
    else:
        assert solver.monitoring == {"diff": True, "cpu": False, "ram": False, "commands": False}
