from __future__ import annotations

import subprocess
import sys
from pathlib import Path, PurePosixPath

import pytest

from harness_wrapper.sandbox import (
    HostServiceRoute,
    LocalSandbox,
    PodmanSandbox,
    SandboxMount,
)


def test_local_run_and_environment(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    result = sandbox.run(
        [sys.executable, "-c", "import os; print(os.getcwd()); print(os.environ['TRACE_TEST'])"],
        env={"TRACE_TEST": "yes"},
    )
    assert result.ok
    assert result.stdout.splitlines() == [str(tmp_path), "yes"]
    assert sandbox.enabled is False


def test_local_check_raises_with_captured_output(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(subprocess.CalledProcessError) as caught:
        sandbox.run([sys.executable, "-c", "import sys; print('bad'); sys.exit(7)"], check=True)
    assert caught.value.returncode == 7
    assert caught.value.stdout == "bad\n"


def test_local_host_service_uses_loopback(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    with sandbox.expose_host_service() as route:
        assert route == HostServiceRoute()


def test_cwd_cannot_escape_root(tmp_path: Path) -> None:
    sandbox = LocalSandbox(tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        sandbox.run(["true"], cwd=tmp_path.parent)


def test_copy_and_delete_are_scoped(tmp_path: Path) -> None:
    source = tmp_path.parent / "source.txt"
    source.write_text("payload", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    sandbox = LocalSandbox(root)

    copied = sandbox.copy_in(source, "nested/copied.txt")
    assert copied.read_text(encoding="utf-8") == "payload"
    output = tmp_path / "output.txt"
    assert sandbox.copy_out("nested/copied.txt", output) == output
    assert output.read_text(encoding="utf-8") == "payload"
    sandbox.delete("nested")
    assert not copied.exists()
    with pytest.raises(ValueError):
        sandbox.delete(".")
    with pytest.raises(ValueError):
        sandbox.delete("../output.txt")


def test_podman_builds_precise_isolated_command(tmp_path: Path) -> None:
    readonly = tmp_path / "inputs"
    readonly.mkdir()
    writable = tmp_path / "outputs"
    writable.mkdir()
    root = tmp_path / "repo"
    work = root / "src"
    work.mkdir(parents=True)
    sandbox = PodmanSandbox(
        root=root,
        image="example/image:pinned",
        network_enabled=False,
        mounts=(
            SandboxMount(readonly, PurePosixPath("/inputs")),
            SandboxMount(writable, PurePosixPath("/outputs"), writable=True),
        ),
    )

    argv, host_cwd, _ = sandbox.prepare_command(
        ["python", "main.py"], cwd=work.resolve(), env={"MODEL": "tiny"}
    )
    assert sandbox.enabled is True
    assert host_cwd == root.resolve()
    assert argv == [
        "podman",
        "run",
        "--rm",
        "--interactive",
        "--http-proxy=false",
        "--network",
        "none",
        "--workdir",
        "/workspace/src",
        "--volume",
        f"{root.resolve()}:/workspace:rw",
        "--volume",
        f"{readonly.resolve()}:/inputs:ro",
        "--volume",
        f"{writable.resolve()}:/outputs:rw",
        "--env",
        "MODEL",
        "example/image:pinned",
        "python",
        "main.py",
    ]


def test_podman_exposes_proxy_through_temporary_internal_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        stdout = "false\n" if command[1] == "info" else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", run)
    sandbox = PodmanSandbox(tmp_path, network_enabled=False)

    with sandbox.expose_host_service() as route:
        assert route.listen_host == "0.0.0.0"
        assert route.client_host.endswith(".1")
        assert route.allow_remote_clients is True
        argv, _, _ = sandbox.prepare_command(["agent"], cwd=tmp_path.resolve(), env={})
        network = calls[0][-1]
        assert argv[argv.index("--network") + 1] == network

    argv, _, _ = sandbox.prepare_command(["agent"], cwd=tmp_path.resolve(), env={})
    assert argv[argv.index("--network") + 1] == "none"
    assert calls[0][1:5] == ["network", "create", "--internal", "--disable-dns"]
    assert "--subnet" in calls[0]
    assert "--gateway" in calls[0]
    assert calls[1][1:3] == ["info", "--format"]
    assert calls[2][1:4] == ["network", "rm", "--force"]


def test_rootless_podman_uses_namespace_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []
    relay = object()
    stopped: list[object] = []

    def run(command, **kwargs):
        calls.append(command)
        stdout = "true\n" if command[1] == "info" else ""
        return subprocess.CompletedProcess(command, 0, stdout, "")

    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(
        PodmanSandbox,
        "_start_rootless_relay",
        lambda self, socket_path: (relay, 32123),
    )
    monkeypatch.setattr(
        PodmanSandbox,
        "_stop_rootless_relay",
        staticmethod(lambda process: stopped.append(process)),
    )
    sandbox = PodmanSandbox(tmp_path, network_enabled=False)

    with sandbox.expose_host_service() as route:
        assert route.client_host.endswith(".1")
        assert route.client_port == 32123
        assert route.unix_socket is not None
        assert route.unix_socket.parent.exists()

    assert stopped == [relay]
    assert route.unix_socket is not None
    assert not route.unix_socket.parent.exists()


def test_podman_exposes_proxy_through_host_gateway_when_networked(tmp_path: Path) -> None:
    sandbox = PodmanSandbox(tmp_path, network_enabled=True)
    with sandbox.expose_host_service() as route:
        assert route == HostServiceRoute("0.0.0.0", "host.containers.internal", True)


def test_mount_validation(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        SandboxMount(tmp_path / "missing", PurePosixPath("/data"))
    with pytest.raises(ValueError):
        SandboxMount(tmp_path, PurePosixPath("relative"))


def test_copy_in_rejects_nested_destination_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    source = tmp_path / "source"
    outside = tmp_path / "outside"
    (source / "nested").mkdir(parents=True)
    (source / "nested" / "payload").write_text("secret", encoding="utf-8")
    (root / "destination").mkdir(parents=True)
    outside.mkdir()
    (root / "destination" / "nested").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        LocalSandbox(root).copy_in(source, "destination")
    assert not (outside / "payload").exists()


def test_copy_in_does_not_truncate_destination_hardlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    source = tmp_path / "source"
    source.write_text("new", encoding="utf-8")
    victim = tmp_path / "victim"
    victim.write_text("keep", encoding="utf-8")
    (root / "result").hardlink_to(victim)

    LocalSandbox(root).copy_in(source, "result")
    assert victim.read_text(encoding="utf-8") == "keep"
    assert (root / "result").read_text(encoding="utf-8") == "new"


def test_copy_out_rejects_nested_source_symlink(tmp_path: Path) -> None:
    root = tmp_path / "root"
    source = root / "results"
    source.mkdir(parents=True)
    private = tmp_path / "private"
    private.write_text("do not export", encoding="utf-8")
    (source / "leak").symlink_to(private)

    with pytest.raises(ValueError, match="symlink"):
        LocalSandbox(root).copy_out("results", tmp_path / "export")
    assert not (tmp_path / "export" / "leak").exists()


def test_podman_env_values_stay_out_of_argv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOST_ONLY_SECRET", "ambient")
    sandbox = PodmanSandbox(tmp_path)
    argv, _, process_env = sandbox.prepare_command(
        ["agent"], cwd=tmp_path.resolve(), env={"API_TOKEN": "very-secret"}
    )

    passed_names = [argv[index + 1] for index, value in enumerate(argv) if value == "--env"]
    assert passed_names == ["API_TOKEN"]
    assert all("very-secret" not in argument for argument in argv)
    assert "HOST_ONLY_SECRET" not in passed_names
    assert process_env["API_TOKEN"] == "very-secret"


def test_podman_maps_absolute_executables(tmp_path: Path) -> None:
    root = tmp_path / "root"
    executable = root / "bin" / "agent"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    sandbox = PodmanSandbox(root, container_executable="/opt/agent/bin/agent")
    argv, _, _ = sandbox.prepare_command([str(executable), "run"], cwd=root.resolve(), env={})
    assert argv[-2:] == ["/opt/agent/bin/agent", "run"]


@pytest.mark.parametrize("executable", ["agent", "/opt/agent/bin/agent"])
def test_podman_preserves_helper_commands_with_configured_executable(
    tmp_path: Path, executable: str
) -> None:
    sandbox = PodmanSandbox(tmp_path, container_executable=executable)
    argv, _, _ = sandbox.prepare_command(["/host/bin/agent", "run"], cwd=tmp_path)
    assert argv[-2:] == [executable, "run"]
    argv, _, _ = sandbox.prepare_command(["./check.sh"], cwd=tmp_path)
    assert argv[-1] == "./check.sh"


def test_podman_translates_executable_in_mounted_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    executable = root / "bin" / "agent"
    executable.parent.mkdir(parents=True)
    executable.write_text("", encoding="utf-8")
    sandbox = PodmanSandbox(root)
    argv, _, _ = sandbox.prepare_command([str(executable)], cwd=root.resolve(), env={})
    assert argv[-1] == "/workspace/bin/agent"


def test_podman_rejects_unmapped_absolute_executable(tmp_path: Path) -> None:
    sandbox = PodmanSandbox(tmp_path)
    with pytest.raises(ValueError, match="not mounted"):
        sandbox.prepare_command(["/host/bin/agent"], cwd=tmp_path.resolve(), env={})
