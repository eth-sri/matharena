"""Small, extensible execution environments for agent harnesses.

``Sandbox`` deliberately means "no isolation" unless explicitly enabled.  This
makes the library safe to adopt without silently changing how a CLI behaves,
while ``PodmanSandbox`` provides process and network isolation when requested.
"""

from __future__ import annotations

import os
import secrets
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

Command = str | Sequence[str]


@dataclass(frozen=True, slots=True)
class HostServiceRoute:
    """Addresses used to expose a host service to one sandbox invocation."""

    listen_host: str = "127.0.0.1"
    client_host: str = "127.0.0.1"
    allow_remote_clients: bool = False
    unix_socket: Path | None = None
    client_port: int | None = None


@dataclass(frozen=True, slots=True)
class CommandResult:
    """The stable, CLI-agnostic result of a command execution."""

    command: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


@dataclass(frozen=True, slots=True)
class SandboxMount:
    """A single Podman bind mount.

    ``source`` is resolved on the host and must already exist. ``target`` is an
    absolute container path. Mounts are read-only unless explicitly writable.
    """

    source: Path
    target: PurePosixPath
    writable: bool = False

    def __post_init__(self) -> None:
        source = Path(self.source).expanduser().resolve()
        target = PurePosixPath(self.target)
        if not source.exists():
            raise FileNotFoundError(f"sandbox mount does not exist: {source}")
        if not target.is_absolute() or ".." in target.parts:
            raise ValueError(f"sandbox mount target must be absolute: {target}")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "target", target)


@dataclass(slots=True)
class Sandbox:
    """Local command runner and filesystem boundary.

    This class is intentionally concrete and easy to subclass. It does not
    provide OS-level isolation: ``enabled`` is always false and commands run on
    the host. Filesystem helper destinations are constrained to ``root`` to
    prevent accidental broad deletes or writes.
    """

    root: Path = field(default_factory=Path.cwd)
    network_enabled: bool = True

    def __post_init__(self) -> None:
        self.root = Path(self.root).expanduser().resolve()
        if not self.root.is_dir():
            raise NotADirectoryError(f"sandbox root is not a directory: {self.root}")

    @property
    def enabled(self) -> bool:
        """Whether commands receive OS-level sandbox isolation."""

        return False

    def _inside_root(self, path: str | os.PathLike[str], *, allow_root: bool = True) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.root / candidate
        candidate = candidate.resolve(strict=False)
        if candidate != self.root and self.root not in candidate.parents:
            raise ValueError(f"path escapes sandbox root: {path}")
        if not allow_root and candidate == self.root:
            raise ValueError("operation on the sandbox root itself is not allowed")
        return candidate

    @staticmethod
    def _argv(command: Command) -> list[str]:
        if isinstance(command, str):
            return shlex.split(command)
        argv = [os.fspath(part) for part in command]
        if not argv:
            raise ValueError("command cannot be empty")
        return argv

    def prepare_command(
        self,
        command: Command,
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> tuple[list[str], Path, dict[str, str]]:
        """Prepare execution; isolated subclasses normally override this method."""

        merged_env = os.environ.copy()
        if env:
            merged_env.update({str(key): str(value) for key, value in env.items()})
        return self._argv(command), cwd, merged_env

    @contextmanager
    def expose_host_service(self) -> Iterator[HostServiceRoute]:
        """Yield addresses through which this environment reaches a host service.

        Isolated subclasses must override this method. The default is valid only
        for local execution, where the child shares the host loopback interface.
        """

        if self.enabled:
            raise NotImplementedError(f"{type(self).__name__} must implement expose_host_service()")
        yield HostServiceRoute()

    def run(
        self,
        command: Command,
        *,
        cwd: str | os.PathLike[str] | None = None,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        check: bool = False,
        input: str | None = None,
    ) -> CommandResult:
        """Run a command and capture text stdout/stderr."""

        workdir = self._inside_root(cwd or self.root)
        if not workdir.is_dir():
            raise NotADirectoryError(f"working directory does not exist: {workdir}")
        argv, host_cwd, process_env = self.prepare_command(command, cwd=workdir, env=env)
        completed = subprocess.run(
            argv,
            cwd=host_cwd,
            env=process_env,
            input=input,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        result = CommandResult(
            tuple(argv), completed.returncode, completed.stdout, completed.stderr
        )
        if check and not result.ok:
            raise subprocess.CalledProcessError(
                result.returncode,
                list(result.command),
                output=result.stdout,
                stderr=result.stderr,
            )
        return result

    def copy_in(self, source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> Path:
        """Copy a host file or directory beneath the sandbox root."""

        source_path = Path(source).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination_path = self._inside_root(destination)
        if source_path.is_dir():
            destination_fd = self._open_root_directory(
                destination_path.relative_to(self.root).parts, create=True
            )
            try:
                self._copy_tree_to_directory_fd(source_path, destination_fd)
            finally:
                os.close(destination_fd)
        else:
            if destination_path.is_dir():
                destination_path = destination_path / source_path.name
            parts = destination_path.relative_to(self.root).parts
            if not parts:
                raise ValueError("cannot replace the sandbox root with a file")
            destination_fd = self._open_root_directory(parts[:-1], create=True)
            try:
                self._copy_file_to_directory_fd(source_path, destination_fd, parts[-1])
            finally:
                os.close(destination_fd)
        return destination_path

    def copy_out(self, source: str | os.PathLike[str], destination: str | os.PathLike[str]) -> Path:
        """Copy a file or directory beneath the root to a host destination."""

        source_path = self._inside_root(source)
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        destination_path = Path(destination).expanduser().resolve(strict=False)
        relative = source_path.relative_to(self.root)
        parent_fd = self._open_root_directory(relative.parts[:-1])
        try:
            name = relative.parts[-1] if relative.parts else "."
            source_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            if stat.S_ISLNK(source_stat.st_mode):
                raise ValueError(f"copy source contains a symlink: {source_path}")
            if stat.S_ISDIR(source_stat.st_mode):
                source_fd = self._open_child_directory(parent_fd, name)
                try:
                    self._copy_tree_from_directory_fd(source_fd, destination_path)
                finally:
                    os.close(source_fd)
            elif stat.S_ISREG(source_stat.st_mode):
                if destination_path.is_dir():
                    destination_path = destination_path / source_path.name
                source_fd = self._open_child_file(parent_fd, name)
                try:
                    self._copy_open_file_to_path(source_fd, source_stat, destination_path)
                finally:
                    os.close(source_fd)
            else:
                raise ValueError(f"copy source is not a regular file: {source_path}")
        finally:
            os.close(parent_fd)
        return destination_path

    @staticmethod
    def _directory_flags() -> int:
        return (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )

    def _open_root_directory(self, parts: Sequence[str], *, create: bool = False) -> int:
        """Traverse beneath root using directory descriptors and no-follow opens."""

        current_fd = os.open(self.root, self._directory_flags())
        try:
            for part in parts:
                try:
                    next_fd = os.open(part, self._directory_flags(), dir_fd=current_fd)
                except FileNotFoundError:
                    if not create:
                        raise
                    with suppress(FileExistsError):
                        os.mkdir(part, dir_fd=current_fd)
                    next_fd = os.open(part, self._directory_flags(), dir_fd=current_fd)
                except OSError as error:
                    raise ValueError(f"sandbox path contains a symlink: {part}") from error
                os.close(current_fd)
                current_fd = next_fd
            return current_fd
        except BaseException:
            os.close(current_fd)
            raise

    @classmethod
    def _open_child_directory(cls, parent_fd: int, name: str, *, create: bool = False) -> int:
        try:
            return os.open(name, cls._directory_flags(), dir_fd=parent_fd)
        except FileNotFoundError:
            if not create:
                raise
            with suppress(FileExistsError):
                os.mkdir(name, dir_fd=parent_fd)
            return os.open(name, cls._directory_flags(), dir_fd=parent_fd)
        except OSError as error:
            raise ValueError(f"copy destination contains a symlink: {name}") from error

    @staticmethod
    def _open_child_file(parent_fd: int, name: str) -> int:
        try:
            return os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except OSError as error:
            raise ValueError(f"copy source contains a symlink: {name}") from error

    @classmethod
    def _copy_file_to_directory_fd(cls, source: Path, destination_fd: int, name: str) -> None:
        if source.is_symlink():
            raise ValueError(f"copy source contains a symlink: {source}")
        source_stat = source.stat(follow_symlinks=False)
        if not stat.S_ISREG(source_stat.st_mode):
            raise ValueError(f"copy source is not a regular file: {source}")
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        source_fd = os.open(source, os.O_RDONLY | no_follow)
        try:
            with suppress(FileNotFoundError):
                os.unlink(name, dir_fd=destination_fd)
            try:
                destination_file_fd = os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | no_follow,
                    stat.S_IMODE(source_stat.st_mode),
                    dir_fd=destination_fd,
                )
            except OSError as error:
                raise ValueError(f"copy destination is a symlink or directory: {name}") from error
            try:
                cls._copy_file_descriptors(source_fd, destination_file_fd)
                os.fchmod(destination_file_fd, stat.S_IMODE(source_stat.st_mode))
            finally:
                os.close(destination_file_fd)
        finally:
            os.close(source_fd)

    @classmethod
    def _copy_tree_to_directory_fd(cls, source: Path, destination_fd: int) -> None:
        if source.is_symlink():
            raise ValueError(f"copy source contains a symlink: {source}")
        with os.scandir(source) as entries:
            for entry in entries:
                source_child = source / entry.name
                if entry.is_symlink():
                    raise ValueError(f"copy source contains a symlink: {source_child}")
                if entry.is_dir(follow_symlinks=False):
                    child_fd = cls._open_child_directory(destination_fd, entry.name, create=True)
                    try:
                        cls._copy_tree_to_directory_fd(source_child, child_fd)
                    finally:
                        os.close(child_fd)
                elif entry.is_file(follow_symlinks=False):
                    cls._copy_file_to_directory_fd(source_child, destination_fd, entry.name)
                else:
                    raise ValueError(f"copy source is not a regular file: {source_child}")

    @staticmethod
    def _copy_file_descriptors(source_fd: int, destination_fd: int) -> None:
        while chunk := os.read(source_fd, 1024 * 1024):
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written == 0:
                    raise OSError("short write while copying file")
                view = view[written:]

    @classmethod
    def _copy_tree_from_directory_fd(cls, source_fd: int, destination: Path) -> None:
        cls._assert_destination(destination, boundary=None, directory=True)
        destination.mkdir(parents=True, exist_ok=True)
        for name in os.listdir(source_fd):
            source_stat = os.stat(name, dir_fd=source_fd, follow_symlinks=False)
            destination_child = destination / name
            if stat.S_ISLNK(source_stat.st_mode):
                raise ValueError(f"copy source contains a symlink: {name}")
            if stat.S_ISDIR(source_stat.st_mode):
                child_fd = cls._open_child_directory(source_fd, name)
                try:
                    cls._copy_tree_from_directory_fd(child_fd, destination_child)
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(source_stat.st_mode):
                child_fd = cls._open_child_file(source_fd, name)
                try:
                    cls._copy_open_file_to_path(child_fd, source_stat, destination_child)
                finally:
                    os.close(child_fd)
            else:
                raise ValueError(f"copy source is not a regular file: {name}")

    @classmethod
    def _copy_open_file_to_path(
        cls, source_fd: int, source_stat: os.stat_result, path: Path
    ) -> None:
        cls._assert_destination(path, boundary=None, directory=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        no_follow = getattr(os, "O_NOFOLLOW", 0)
        try:
            destination_fd = os.open(
                path,
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC | no_follow,
                stat.S_IMODE(source_stat.st_mode),
            )
        except OSError as error:
            raise ValueError(f"copy destination is a symlink or directory: {path}") from error
        try:
            cls._copy_file_descriptors(source_fd, destination_fd)
            os.fchmod(destination_fd, stat.S_IMODE(source_stat.st_mode))
        finally:
            os.close(destination_fd)

    @staticmethod
    def _assert_destination(
        destination: Path,
        *,
        boundary: Path | None,
        directory: bool,
    ) -> None:
        """Reject link redirection at every existing destination component."""

        if boundary is not None:
            boundary = boundary.resolve()
            try:
                relative = destination.absolute().relative_to(boundary)
            except ValueError as error:
                raise ValueError(f"copy destination escapes sandbox root: {destination}") from error
            current = boundary
            for part in relative.parts:
                current = current / part
                if (current.exists() or current.is_symlink()) and current.is_symlink():
                    raise ValueError(f"copy destination contains a symlink: {current}")
        if destination.is_symlink():
            raise ValueError(f"copy destination is a symlink: {destination}")
        if destination.exists() and directory != destination.is_dir():
            expected = "directory" if directory else "file"
            raise ValueError(f"copy destination must be a {expected}: {destination}")

    def delete(self, path: str | os.PathLike[str], *, missing_ok: bool = True) -> None:
        """Delete a path beneath the root, never the root itself."""

        target = self._inside_root(path, allow_root=False)
        parts = target.relative_to(self.root).parts
        parent_fd = self._open_root_directory(parts[:-1])
        try:
            self._delete_at(parent_fd, parts[-1], missing_ok=missing_ok)
        finally:
            os.close(parent_fd)

    @classmethod
    def _delete_at(cls, parent_fd: int, name: str, *, missing_ok: bool) -> None:
        try:
            target_stat = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            if missing_ok:
                return
            raise
        if not stat.S_ISDIR(target_stat.st_mode) or stat.S_ISLNK(target_stat.st_mode):
            os.unlink(name, dir_fd=parent_fd)
            return
        directory_fd = cls._open_child_directory(parent_fd, name)
        try:
            for child in os.listdir(directory_fd):
                cls._delete_at(directory_fd, child, missing_ok=False)
        finally:
            os.close(directory_fd)
        os.rmdir(name, dir_fd=parent_fd)

    @property
    def _relay_executable(self) -> str:
        """Container CLI able to enter the rootless network namespace."""

        return "podman"

    def _start_rootless_relay(self, socket_path: Path) -> tuple[subprocess.Popen[str], int]:
        """Publish a host Unix socket inside the rootless network namespace.

        A rootless bridge lives in a network namespace of its own, so its
        gateway address is absent from the host and a host-bound listener is
        unreachable from the container. Relaying from inside that namespace to a
        Unix socket on the host puts the service at the address the container
        already routes to, without granting it any external connectivity.
        """

        relay_script = Path(__file__).with_name("_tcp_unix_relay.py")
        relay = subprocess.Popen(
            [
                self._relay_executable,
                "unshare",
                "--rootless-netns",
                sys.executable,
                str(relay_script),
                str(socket_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        assert relay.stdout is not None
        ready = selectors.DefaultSelector()
        try:
            ready.register(relay.stdout, selectors.EVENT_READ)
            if not ready.select(timeout=5):
                self._stop_rootless_relay(relay)
                raise RuntimeError("timed out starting isolated Podman proxy relay")
            line = relay.stdout.readline().strip()
            try:
                port = int(line)
            except ValueError as error:
                self._stop_rootless_relay(relay)
                raise RuntimeError("cannot start isolated Podman proxy relay") from error
            if not 0 < port < 65536:
                self._stop_rootless_relay(relay)
                raise RuntimeError("isolated Podman proxy relay returned an invalid port")
            return relay, port
        finally:
            ready.close()

    @staticmethod
    def _stop_rootless_relay(relay: subprocess.Popen[str]) -> None:
        if relay.poll() is None:
            with suppress(ProcessLookupError):
                os.killpg(relay.pid, signal.SIGTERM)
            try:
                relay.wait(timeout=2)
            except subprocess.TimeoutExpired:
                with suppress(ProcessLookupError):
                    os.killpg(relay.pid, signal.SIGKILL)
                relay.wait(timeout=2)
        for stream in (relay.stdout, relay.stderr):
            if stream is not None:
                stream.close()

    @contextmanager
    def gateway_host_service_route(
        self, gateway: str, probe_executable: str
    ) -> Iterator[HostServiceRoute]:
        """Route a host service to a container on an internal bridge network.

        A rootful runtime puts the bridge gateway on a host interface, so a
        host-bound listener already answers there. A rootless one keeps the
        bridge in a network namespace of its own, where nothing bound on the
        host is visible, so the service is relayed in from that namespace
        instead. Either way the container reaches it at ``gateway`` and gains no
        route off the internal network.
        """

        probe = subprocess.run(
            [probe_executable, "info", "--format", "{{.Host.Security.Rootless}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        # Only Podman answers this probe, and only it can host the relay, so any
        # other runtime is treated as mapping identities directly.
        if probe.returncode != 0 or probe.stdout.strip().lower() != "true":
            yield HostServiceRoute("0.0.0.0", gateway, True)
            return
        with tempfile.TemporaryDirectory(prefix="harness-wrapper-proxy-") as directory:
            socket_path = Path(directory) / "proxy.sock"
            relay, port = self._start_rootless_relay(socket_path)
            try:
                yield HostServiceRoute(
                    client_host=gateway,
                    allow_remote_clients=True,
                    unix_socket=socket_path,
                    client_port=port,
                )
            finally:
                self._stop_rootless_relay(relay)


class LocalSandbox(Sandbox):
    """Explicit name for the default, non-isolating sandbox."""


@dataclass(slots=True)
class PodmanSandbox(Sandbox):
    """Ephemeral Podman container with explicit bind mounts.

    The root is mounted read/write at ``container_root``. Additional mounts are
    read-only by default and must opt in to writes. The container is removed
    after each command; persistent changes therefore belong in writable mounts.
    """

    image: str = "docker.io/library/python:3.12-slim"
    container_root: PurePosixPath = field(default_factory=lambda: PurePosixPath("/workspace"))
    mounts: Sequence[SandboxMount] = field(default_factory=tuple)
    podman_executable: str = "podman"
    container_executable: str | PurePosixPath | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    _host_service_network: str | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        super(PodmanSandbox, self).__post_init__()
        self.container_root = PurePosixPath(self.container_root)
        if not self.container_root.is_absolute() or ".." in self.container_root.parts:
            raise ValueError("container_root must be an absolute container path")
        self.mounts = tuple(self.mounts)
        if self.container_executable is not None:
            container_executable = PurePosixPath(self.container_executable)
            if ".." in container_executable.parts:
                raise ValueError("container_executable cannot contain '..'")
            self.container_executable = container_executable
        targets = [self.container_root, *(mount.target for mount in self.mounts)]
        if len(targets) != len(set(targets)):
            raise ValueError("sandbox mount targets must be unique")

    @property
    def enabled(self) -> bool:
        return True

    def _container_path(self, host_path: Path) -> PurePosixPath:
        relative = host_path.relative_to(self.root)
        return self.container_root.joinpath(*relative.parts)

    def _container_command(self, command: Command) -> list[str]:
        command_argv = self._argv(command)
        executable = Path(command_argv[0]).expanduser()
        if self.container_executable is not None and executable.is_absolute():
            command_argv[0] = str(self.container_executable)
        elif executable.is_absolute():
            resolved = executable.resolve(strict=False)
            if resolved == self.root or self.root in resolved.parents:
                command_argv[0] = str(self._container_path(resolved))
            else:
                for mount in self.mounts:
                    if resolved == mount.source or mount.source in resolved.parents:
                        relative = resolved.relative_to(mount.source)
                        command_argv[0] = str(mount.target.joinpath(*relative.parts))
                        break
                else:
                    raise ValueError(
                        "absolute command executable is not mounted in the container; "
                        "set container_executable or add a SandboxMount"
                    )
        return command_argv

    def prepare_command(
        self,
        command: Command,
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> tuple[list[str], Path, dict[str, str]]:
        argv = [self.podman_executable, "run", "--rm", "--interactive", "--http-proxy=false"]
        network = self._host_service_network or ("bridge" if self.network_enabled else "none")
        argv.extend(["--network", network])
        argv.extend(["--workdir", str(self._container_path(cwd))])
        argv.extend(["--volume", f"{self.root}:{self.container_root}:rw"])
        for mount in self.mounts:
            mode = "rw" if mount.writable else "ro"
            argv.extend(["--volume", f"{mount.source}:{mount.target}:{mode}"])
        explicit_env = {str(key): str(value) for key, value in (env or {}).items()}
        for key in explicit_env:
            argv.extend(["--env", key])
        argv.extend(self.extra_args)
        argv.append(self.image)
        argv.extend(self._container_command(command))
        # Podman resolves name-only --env entries from its process environment;
        # values never appear in argv or CommandResult/CalledProcessError objects.
        process_env = os.environ.copy()
        process_env.update(explicit_env)
        return argv, self.root, process_env

    @contextmanager
    def expose_host_service(self) -> Iterator[HostServiceRoute]:
        """Expose one host proxy without granting direct external network access."""

        if self._host_service_network is not None:
            raise RuntimeError("this PodmanSandbox already has an active host service")
        if self.network_enabled:
            yield HostServiceRoute("0.0.0.0", "host.containers.internal", True)
            return

        network = f"harness-wrapper-{os.getpid()}-{secrets.token_hex(6)}"
        create: subprocess.CompletedProcess[str] | None = None
        gateway = ""
        for _ in range(5):
            second_octet = 192 + secrets.randbelow(64)
            third_octet = secrets.randbelow(256)
            subnet = f"10.{second_octet}.{third_octet}.0/24"
            gateway = f"10.{second_octet}.{third_octet}.1"
            create = subprocess.run(
                [
                    self.podman_executable,
                    "network",
                    "create",
                    "--internal",
                    "--disable-dns",
                    "--subnet",
                    subnet,
                    "--gateway",
                    gateway,
                    network,
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if create.returncode == 0:
                break
        if create is None or create.returncode != 0:
            detail = create.stderr.strip() if create is not None else "unknown error"
            raise RuntimeError(f"cannot create isolated Podman proxy network: {detail}")
        try:
            info = subprocess.run(
                [
                    self.podman_executable,
                    "info",
                    "--format",
                    "{{.Host.Security.Rootless}}",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self._host_service_network = network
            if info.returncode != 0:
                detail = info.stderr.strip() or "rootless status is unavailable"
                raise RuntimeError(f"cannot inspect Podman runtime: {detail}")
            rootless = info.stdout.strip().lower()
            if rootless == "false":
                yield HostServiceRoute("0.0.0.0", gateway, True)
            elif rootless == "true":
                with tempfile.TemporaryDirectory(prefix="harness-wrapper-proxy-") as directory:
                    socket_path = Path(directory) / "proxy.sock"
                    relay, port = self._start_rootless_relay(socket_path)
                    try:
                        yield HostServiceRoute(
                            client_host=gateway,
                            allow_remote_clients=True,
                            unix_socket=socket_path,
                            client_port=port,
                        )
                    finally:
                        self._stop_rootless_relay(relay)
            else:
                raise RuntimeError(f"cannot inspect Podman runtime: unexpected value {rootless!r}")
        finally:
            self._host_service_network = None
            subprocess.run(
                [self.podman_executable, "network", "rm", "--force", network],
                capture_output=True,
                text=True,
                check=False,
            )

    @property
    def _relay_executable(self) -> str:
        return self.podman_executable
