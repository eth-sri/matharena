"""Small, dependency-free utilities shared by CLI harness adapters."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import uuid
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HarnessError(RuntimeError):
    """Base exception raised by harness-wrapper."""


class ExecutableNotFoundError(HarnessError):
    """The requested agent CLI could not be found."""


class UnsupportedCLIVersionError(HarnessError):
    """An installed CLI differs from the version tested by this package."""


class CLIProcessError(HarnessError):
    """A CLI process exited unsuccessfully."""

    def __init__(
        self,
        command: Sequence[str],
        returncode: int,
        stderr: str = "",
    ) -> None:
        self.command = tuple(command)
        self.returncode = returncode
        self.stderr = stderr
        detail = stderr.strip() or "no error output"
        super().__init__(f"CLI exited with status {returncode}: {detail}")


class ModelCompatibilityError(HarnessError, ValueError):
    """A model cannot be used through the selected CLI protocol."""


class SessionCache:
    """Repo-local index of native harness sessions used for deterministic resume."""

    def __init__(self, root: str | os.PathLike[str]) -> None:
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / ".harness_wrapper" / "sessions.json"

    def _read(self) -> dict[str, Any]:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"schema_version": 1, "harnesses": {}}
        except (OSError, json.JSONDecodeError) as exc:
            raise HarnessError(f"cannot read session cache {self.path}: {exc}") from exc
        if not isinstance(value, dict) or not isinstance(value.get("harnesses"), dict):
            raise HarnessError(f"invalid session cache: {self.path}")
        return value

    def record(self, harness: str, session_id: str, timestamp: str) -> None:
        data = self._read()
        harnesses = data.setdefault("harnesses", {})
        records = harnesses.setdefault(harness, [])
        if not isinstance(records, list):
            raise HarnessError(f"invalid session cache entry for {harness!r}")
        records[:] = [
            item
            for item in records
            if not isinstance(item, dict) or item.get("session_id") != session_id
        ]
        records.insert(0, {"session_id": session_id, "last_activity_at": timestamp})
        del records[100:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_text(
                json.dumps(data, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)

    def last(self, harness: str) -> str | None:
        records = self._read().get("harnesses", {}).get(harness, [])
        if not isinstance(records, list):
            return None
        for item in records:
            if isinstance(item, dict):
                session_id = item.get("session_id")
                if isinstance(session_id, str):
                    return session_id
        return None


def default_cli_prefix() -> Path:
    """Return the user-local prefix reserved for pinned agent CLIs."""

    return Path.home() / ".harness-wrapper" / "clis" / "native"


@dataclass(frozen=True, slots=True)
class CLIInstallation:
    """CLI package metadata and its requested install version."""

    executable: str
    package: str
    version: str = "latest"
    version_args: tuple[str, ...] = ("--version",)

    @property
    def package_spec(self) -> str:
        return f"{self.package}@{self.version}"

    def cli_bin_dir(self, prefix: Path | None = None) -> Path:
        return (prefix or default_cli_prefix()) / "bin"

    def executable_path(self, prefix: Path | None = None) -> Path:
        return self.cli_bin_dir(prefix) / self.executable

    def install_command(self, prefix: Path | None = None) -> list[str]:
        """Return, but do not execute, the native installation command."""

        target = prefix or default_cli_prefix()
        release_names = {
            "claude": "claude-code",
            "codex": "codex-cli",
            "kimi": "kimi-code",
            "agy": "antigravity-cli",
            "qwen": "qwen-code",
            "muse": "muse-code",
            "opencode": "opencode",
            "vibe": "mistral-vibe",
        }
        release_name = release_names.get(self.executable, self.executable)
        command = [
            "harness-wrapper",
            "install-clis",
            release_name,
            "--prefix",
            str(target),
        ]
        if self.version != "latest":
            command.extend(("--version", f"{release_name}={self.version}"))
        return command


def install_cli(
    installation: CLIInstallation,
    *,
    prefix: Path | None = None,
    runner: Callable[..., Any] = subprocess.run,
) -> Path:
    """Install one native CLI into the harness-wrapper user directory."""

    target = prefix or default_cli_prefix()
    target.mkdir(parents=True, exist_ok=True)
    runner(installation.install_command(target), check=True)
    return installation.executable_path(target)


def find_executable(
    executable: str,
    *,
    explicit: str | os.PathLike[str] | None = None,
    installation: CLIInstallation | None = None,
    path: str | None = None,
) -> str:
    """Resolve explicit, wrapper-local, then PATH executables in that order."""

    if explicit is not None:
        candidate = Path(explicit).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        # Bare executable names are useful with hermetic test PATHs.
        if len(candidate.parts) == 1:
            found = shutil.which(str(candidate), path=path)
            if found:
                return found
        raise ExecutableNotFoundError(f"CLI executable is not executable: {candidate}")

    if installation is not None:
        local = installation.executable_path()
        if local.is_file() and os.access(local, os.X_OK):
            return str(local.resolve())

    found = shutil.which(executable, path=path)
    if found:
        return found
    hint = installation.install_command() if installation else None
    suffix = f"; install it with: {' '.join(hint)}" if hint else ""
    raise ExecutableNotFoundError(f"Could not find {executable!r}{suffix}")


_VERSION_RE = re.compile(r"(?<![\d.])(\d+(?:\.\d+){1,3}(?:[-+][0-9A-Za-z.-]+)?)")


def extract_version(output: str) -> str:
    match = _VERSION_RE.search(output)
    if not match:
        raise UnsupportedCLIVersionError(f"Could not parse CLI version from {output!r}")
    return match.group(1)


def validate_cli_version(
    executable: str,
    installation: CLIInstallation,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    """Check an exact pin, or only parse the installed version for latest."""

    completed = runner(
        [executable, *installation.version_args],
        check=True,
        capture_output=True,
        text=True,
    )
    output = f"{getattr(completed, 'stdout', '')}\n{getattr(completed, 'stderr', '')}"
    actual = extract_version(output)
    if installation.version != "latest" and actual != installation.version:
        raise UnsupportedCLIVersionError(
            f"{installation.executable} {actual} is installed, but adapter support is "
            f"pinned to {installation.version}"
        )
    return actual


class JSONLineParser:
    """Incrementally parse JSONL without assuming a particular CLI schema.

    In non-strict mode, human-readable/banner lines are retained as ``stdout``
    events instead of being silently discarded.
    """

    def __init__(self, *, strict: bool = False) -> None:
        self.strict = strict
        self._buffer = ""

    def feed(self, chunk: str | bytes) -> list[dict[str, Any]]:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        self._buffer += chunk
        lines = self._buffer.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._buffer = lines.pop()
        else:
            self._buffer = ""
        return [event for line in lines if (event := self._parse(line)) is not None]

    def finish(self) -> list[dict[str, Any]]:
        if not self._buffer:
            return []
        line, self._buffer = self._buffer, ""
        event = self._parse(line)
        return [] if event is None else [event]

    def _parse(self, line: str) -> dict[str, Any] | None:
        text = line.strip()
        if not text:
            return None
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            if self.strict:
                raise
            return {"type": "stdout", "text": text}
        if isinstance(value, Mapping):
            return dict(value)
        return {"type": "json", "value": value}


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def merge_environment(*environments: Mapping[str, Any] | None) -> dict[str, str]:
    """Merge environment overlays while rejecting implicit ``None`` strings."""

    merged: dict[str, str] = dict(os.environ)
    for environment in environments:
        if environment:
            merged.update(
                {str(key): str(value) for key, value in environment.items() if value is not None}
            )
    return merged


def as_argv(command: Iterable[str | os.PathLike[str]]) -> list[str]:
    return [os.fspath(part) for part in command]
