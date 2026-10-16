"""Container selection and cancellation without a container engine or model API."""

import json
import os
import subprocess
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_wrapper import TokenUsage
from matharena.solvers.harness_solver import (
    DockerHarnessSandbox,
    HarnessSolver,
    PodmanHarnessSandbox,
)
from matharena.solvers.run_budget import RunBudget


def _solver(*, container_runtime="docker", **settings):
    return HarnessSolver(
        {}, "{problem}",
        {
            "model": "gpt-test", "api": "openai", "harness": "codex",
            "harness_config": {"managed_cli": False, **settings},
        },
        "",
        container_runtime=container_runtime,
    )


@pytest.mark.parametrize(
    ("runtime", "sandbox_type"),
    [(None, DockerHarnessSandbox), ("docker", DockerHarnessSandbox),
     ("podman", PodmanHarnessSandbox), (" Podman ", PodmanHarnessSandbox)],
)
def test_runtime_selection_preserves_limits_and_helper_commands(
    tmp_path, monkeypatch, runtime, sandbox_type
):
    monkeypatch.delenv("MATHARENA_CONTAINER_USERNS", raising=False)
    monkeypatch.setattr("os.getuid", lambda: 1000)
    monkeypatch.setattr("os.getgid", lambda: 1000)
    settings = {} if runtime is None else {"container_runtime": runtime}
    sandbox = _solver(**settings)._build_sandbox(tmp_path)
    assert isinstance(sandbox, sandbox_type)
    argv, _, env = sandbox.prepare_command(
        ["/usr/bin/true", "exec"], cwd=tmp_path,
        env={"API_TOKEN": "secret", "HOME": "/work/.harness-home"},
    )
    assert argv[1] == "run"
    assert argv[-2:] == ["codex", "exec"]
    assert argv[argv.index("--name") + 1] == sandbox.container_name
    assert argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--memory") + 1] == "12g"
    assert argv[argv.index("--cpus") + 1] == "4"
    assert argv[argv.index("--pids-limit") + 1] == "768"
    assert argv[argv.index("--cap-drop") + 1] == "ALL"
    assert argv[argv.index("--user") + 1] == "1000:1000"
    assert "--read-only" in argv
    assert "secret" not in " ".join(argv)
    assert env["API_TOKEN"] == "secret"
    if sandbox_type is PodmanHarnessSandbox:
        assert argv[0] == "podman"
        assert argv[argv.index("--userns") + 1] == "keep-id"
        assert "--http-proxy=false" in argv
    else:
        assert "--userns" not in argv
    helper, _, _ = sandbox.prepare_command(["./check.sh"], cwd=tmp_path)
    assert helper[-1] == "./check.sh"


@pytest.mark.parametrize("runtime", ["", "containerd", None, True, 1])
def test_invalid_runtime_fails_before_starting_a_process(runtime):
    with pytest.raises((TypeError, ValueError), match="container_runtime"):
        _solver(container_runtime=runtime)


def test_model_config_cannot_select_the_execution_runtime(tmp_path):
    config = {
        "harness": "codex",
        "harness_config": {"container_runtime": "podman", "managed_cli": False},
    }
    solver = HarnessSolver({}, "{problem}", config, "")
    assert isinstance(solver._build_sandbox(tmp_path), DockerHarnessSandbox)
    solver = HarnessSolver({}, "{problem}", config, "", container_runtime="podman")
    assert isinstance(solver._build_sandbox(tmp_path), PodmanHarnessSandbox)
    assert "container_runtime" not in solver.harness_config


@pytest.mark.parametrize(
    ("uid", "configured", "ambient", "expected"),
    [(1000, None, None, "keep-id"), (0, None, None, None),
     (1000, None, "host", "host"), (1000, "auto", "host", "auto")],
)
def test_podman_user_namespace_defaults_and_overrides(
    tmp_path, monkeypatch, uid, configured, ambient, expected
):
    monkeypatch.setattr("os.getuid", lambda: uid)
    monkeypatch.delenv("MATHARENA_CONTAINER_USERNS", raising=False)
    if ambient is not None:
        monkeypatch.setenv("MATHARENA_CONTAINER_USERNS", ambient)
    sandbox = _solver(container_runtime="podman", userns=configured)._build_sandbox(tmp_path)
    if expected is None:
        assert "--userns" not in sandbox.extra_args
    else:
        assert sandbox.extra_args[sandbox.extra_args.index("--userns") + 1] == expected


@pytest.mark.parametrize("runtime", ["docker", "podman"])
@pytest.mark.parametrize("generic_keys", [False, True])
def test_image_and_launch_arguments_support_existing_configs(tmp_path, runtime, generic_keys):
    settings = {
        "container_runtime": runtime, "docker_image": "legacy-image",
        "docker_extra_args": ["--env", "LEGACY_FLAG=1"],
    }
    if generic_keys:
        settings.update(container_image="new-image", container_extra_args=["--env", "NEW_FLAG=1"])
    sandbox = _solver(**settings)._build_sandbox(tmp_path)
    assert sandbox.image == ("new-image" if generic_keys else "legacy-image")
    assert ("NEW_FLAG=1" if generic_keys else "LEGACY_FLAG=1") in sandbox.extra_args
    assert ("LEGACY_FLAG=1" if generic_keys else "NEW_FLAG=1") not in sandbox.extra_args
    assert "no-new-privileges:true" in sandbox.extra_args


@pytest.mark.parametrize("rootless", [False, True])
def test_podman_proxy_network_and_relay_cleanup_on_failure(tmp_path, monkeypatch, rootless):
    calls, relays = [], []

    def run(argv, **kwargs):
        calls.append(argv)
        output = str(rootless).lower() if argv[1] == "info" else ""
        return subprocess.CompletedProcess(argv, 0, output, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(PodmanHarnessSandbox, "_start_rootless_relay", lambda self, path: ("relay", 12345))
    monkeypatch.setattr(PodmanHarnessSandbox, "_stop_rootless_relay", staticmethod(relays.append))
    sandbox = _solver(container_runtime="podman")._build_sandbox(tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        with sandbox.expose_host_service() as route:
            network = calls[0][-1]
            argv, _, _ = sandbox.prepare_command(["codex"], cwd=tmp_path)
            assert argv[argv.index("--network") + 1] == network
            assert route.client_host == calls[0][calls[0].index("--gateway") + 1]
            assert bool(route.unix_socket) == rootless
            assert route.client_port == (12345 if rootless else None)
            raise RuntimeError("simulated agent failure")
    assert "--internal" in calls[0] and "--disable-dns" in calls[0]
    assert calls[-1] == ["podman", "network", "rm", "--force", network]
    assert relays == (["relay"] if rootless else [])
    argv, _, _ = sandbox.prepare_command(["codex"], cwd=tmp_path)
    assert argv[argv.index("--network") + 1] == "none"


def test_networked_podman_routes_proxy_through_host_gateway(tmp_path):
    sandbox = _solver(container_runtime="podman", direct_network=True)._build_sandbox(tmp_path)
    with sandbox.expose_host_service() as route:
        assert route.client_host == "host.containers.internal"
        argv, _, _ = sandbox.prepare_command(["codex"], cwd=tmp_path)
        assert argv[argv.index("--network") + 1] == "bridge"
        assert "host.docker.internal:host-gateway" not in argv


def test_podman_uses_host_configuration_paths_and_container_environment(tmp_path, monkeypatch):
    host_paths = {
        "HOME": str(tmp_path / "host-home"),
        "XDG_CONFIG_HOME": str(tmp_path / "host-config"),
        "XDG_CACHE_HOME": str(tmp_path / "host-cache"),
        "PATH": "/host/bin:/usr/bin",
    }
    for key, value in host_paths.items():
        monkeypatch.setenv(key, value)
    container_paths = {key: f"/work/{key.lower()}" for key in host_paths}
    sandbox = _solver(container_runtime="podman")._build_sandbox(tmp_path)
    argv, _, process_env = sandbox.prepare_command(
        ["codex"], cwd=tmp_path, env={**container_paths, "API_TOKEN": "model-secret"},
    )
    passed = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
    for key, value in container_paths.items():
        assert f"{key}={value}" in passed
        assert process_env[key] == host_paths[key]
    assert "API_TOKEN" in passed
    assert "model-secret" not in " ".join(argv)
    assert process_env["API_TOKEN"] == "model-secret"


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_budget_force_removes_the_named_container(tmp_path, monkeypatch, runtime):
    solver = _solver(container_runtime=runtime)
    sandbox = solver._build_sandbox(tmp_path)
    argv, _, _ = sandbox.prepare_command(["codex"], cwd=tmp_path)
    removed, terminated = [], threading.Event()

    def run(command, **kwargs):
        removed.append(command)
        return subprocess.CompletedProcess(command, 0)

    def stream(_prompt):
        assert terminated.wait(5), "watchdog did not terminate the harness"
        return iter(())

    monkeypatch.setattr(subprocess, "run", run)
    agent = SimpleNamespace(
        get_tokens=lambda: TokenUsage(), interrupt=lambda: None,
        terminate=terminated.set, stream=stream,
    )
    budget = RunBudget({"max_time_seconds": 0.01}, lambda usage: 0, priced=False)
    assert solver._stream_with_budget(agent, sandbox, budget, "solve") == []
    removal = [argv[0], "rm", "--force"]
    if runtime == "podman":
        removal.extend(("--time", "0"))
    assert removed == [[*removal, sandbox.container_name]]
    assert budget.reason == "time_limit"


@pytest.mark.parametrize("runtime", [None, "docker", "podman"])
def test_build_helper_uses_selected_engine_and_workspace_mapping(tmp_path, monkeypatch, runtime):
    engine = tmp_path / (runtime or "docker")
    log = tmp_path / "runtime.jsonl"
    engine.write_text("""#!/usr/bin/env python3
import json, os, sys
with open(os.environ['MATHARENA_TEST_RUNTIME_LOG'], 'a') as log:
    log.write(json.dumps(sys.argv[1:]) + '\\n')
if sys.argv[1:3] == ['volume', 'inspect']:
    sys.exit(1)
""")
    engine.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MATHARENA_TEST_RUNTIME_LOG", str(log))
    monkeypatch.delenv("MATHARENA_CONTAINER_RUNTIME", raising=False)
    monkeypatch.delenv("MATHARENA_CONTAINER_USERNS", raising=False)
    if runtime is not None:
        monkeypatch.setenv("MATHARENA_CONTAINER_RUNTIME", runtime)
    script = Path(__file__).resolve().parents[1] / "scripts/build_harness_arxivlean_image.sh"
    subprocess.run(["bash", str(script)], capture_output=True, text=True, check=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0][0] == "build"
    assert calls[1][:2] == ["volume", "inspect"]
    assert calls[2][:2] == ["volume", "create"]
    smoke = calls[-1]
    assert smoke[0] == "run"
    assert "--read-only" in smoke
    assert smoke[smoke.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
    if runtime == "podman" and os.getuid() != 0:
        assert smoke[smoke.index("--userns") + 1] == "keep-id"
    else:
        assert "--userns" not in smoke
