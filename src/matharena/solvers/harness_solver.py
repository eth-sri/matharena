"""Run MathArena models through a coding-agent harness.

The generic harness lifecycle comes from the vendored ``harness-wrapper``
package.  This module supplies the MathArena-specific pieces: API credential
resolution, per-problem workspaces, cost conversion, and a container sandbox that
lets the harness reach only a host-side model proxy.

For harnesses, ``time`` and ``request_time`` both measure cumulative wall-clock
seconds inside agent.run/resume, including tool use, retries, and quota waits.
They exclude batch queueing, CLI installation, workspace setup, post-run Lean
validation, and grading; they are not provider-only inference durations.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, override

from harness_wrapper import (
    Agent,
    AgentEvent,
    HostServiceRoute,
    InstalledCLI,
    Model,
    PodmanSandbox,
    Sandbox,
    SandboxMount,
    ensure_cli,
)
from harness_wrapper.models.oauth import (
    anthropic_oauth_config,
    kimi_oauth_config,
    openai_oauth_config,
)
from loguru import logger

from matharena.api_client import APIClient
from matharena.container_runtime import validate_container_runtime
from matharena.solvers import BaseSolver, SolverResponse
from matharena.solvers.run_budget import RunBudget
from matharena.solvers.native_usage import NativeUsageReader
from matharena.token_pricing import TokenPricing
from matharena.solvers.codex_cli_solver import (
    CONTAINER_BASE_PATH,
    CONTAINER_ELAN_HOME,
    CONTAINER_LEAN_CACHE,
    CONTAINER_TMPDIR,
    DEFAULT_PACKAGE_ORDER,
    _PromptFields,
    _display_workspace_path,
    _format_template,
    _package_names,
    _read_text,
    _toolchain_dir,
    _write_text,
)


REASONING_EXTRACTION_MESSAGE = (
    'The model hit a "reasoning extraction" refusal. We made many attempts to avoid '
    'such refusals, but could not and are therefore marking the problem as incorrect.'
)


def _append_refusal_message(conversation):
    conversation.append({
        "role": "assistant", "type": "response", "content": REASONING_EXTRACTION_MESSAGE,
    })


def _has_reasoning_extraction_refusal(events):
    return any(event.raw.get("refusal_category") == "reasoning_extraction" for event in events)


def _count_harness_retries(events):
    # Provider-error notifications can precede a retry event. Count only
    # numbered retry attempts, not both the notification and the retry.
    return sum(
        event.type == "recovery"
        and isinstance(event.content, Mapping)
        and isinstance(event.content.get("attempt"), int)
        and event.content["attempt"] > 0
        for event in events
    )


DEFAULT_WORKSPACE_ROOT = (
    "logs/harness_workspaces/{competition}/{solver_name}/p{problem_idx}_r{run_idx}"
)
DEFAULT_CONTAINER_ROOT = PurePosixPath("/work")
CONTAINER_HARNESS_VENV = "/opt/harness-venv"
CONTAINER_HARNESS_CLI_ROOT = PurePosixPath("/opt/harness-cli")
CONTAINER_HARNESS_PATH = f"{CONTAINER_HARNESS_CLI_ROOT}/bin:{CONTAINER_HARNESS_VENV}/bin:{CONTAINER_BASE_PATH}"
DEFAULT_HARNESS_DOCKER_IMAGE = "matharena-harness-arxivlean:lean4.31"
HARNESS_EXECUTABLES = {
    "grok": "grok",
    "grok-build": "grok",
    "grok-cli": "grok",
    "muse": "muse",
    "muse-code": "muse",
    "muse-cli": "muse",
    "anthropic": "claude",
    "claude": "claude",
    "claude-code": "claude",
    "codex": "codex",
    "codex-cli": "codex",
    "openai-codex": "codex",
    "kimi": "kimi",
    "kimi-code": "kimi",
    "moonshot": "kimi",
    "agy": "agy",
    "antigravity": "agy",
    "antigravity-cli": "agy",
    "google-antigravity": "agy",
    "gravity": "agy",
    "gravity-cli": "agy",
    "qwen": "qwen",
    "qwen-cli": "qwen",
    "qwen-code": "qwen",
    "open-code": "opencode",
    "opencode": "opencode",
    "deepcode": "deepcode",
    "deepcode-cli": "deepcode",
    "deep-code": "deepcode",
    "vibe": "vibe",
    "vibe-cli": "vibe",
    "mistral-vibe": "vibe",
    "mistral": "vibe",
}
HARNESS_RELEASES = {
    "grok": "grok-build",
    "grok-build": "grok-build",
    "grok-cli": "grok-build",
    "muse": "muse-code",
    "muse-code": "muse-code",
    "muse-cli": "muse-code",
    "anthropic": "claude-code",
    "claude": "claude-code",
    "claude-code": "claude-code",
    "codex": "codex-cli",
    "codex-cli": "codex-cli",
    "openai-codex": "codex-cli",
    "kimi": "kimi-code",
    "kimi-code": "kimi-code",
    "moonshot": "kimi-code",
    "agy": "antigravity-cli",
    "antigravity": "antigravity-cli",
    "antigravity-cli": "antigravity-cli",
    "google-antigravity": "antigravity-cli",
    "gravity": "antigravity-cli",
    "gravity-cli": "antigravity-cli",
    "qwen": "qwen-code",
    "qwen-cli": "qwen-code",
    "qwen-code": "qwen-code",
    "open-code": "opencode",
    "opencode": "opencode",
    "deepcode": "deepcode",
    "deepcode-cli": "deepcode",
    "deep-code": "deepcode",
    "vibe": "mistral-vibe",
    "vibe-cli": "mistral-vibe",
    "mistral-vibe": "mistral-vibe",
    "mistral": "mistral-vibe",
}
INLINE_CONTAINER_ENV = frozenset(
    {
        "ELAN_HOME",
        "HOME",
        "KIMI_CODE_HOME",
        "LD_LIBRARY_PATH",
        "LEAN_PATH",
        "LEAN_SRC_PATH",
        "PATH",
        "TMPDIR",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
    }
)


@dataclass(slots=True)
class DockerHarnessSandbox(Sandbox):
    """Ephemeral Docker environment with a private model-proxy network.

    The problem workspace is the only writable bind mount.  When
    ``network_enabled`` is false, ``expose_host_service`` creates an internal
    bridge so the harness can reach the wrapper's model proxy but cannot reach
    the public network directly.
    """

    network_enabled: bool = False
    image: str = DEFAULT_HARNESS_DOCKER_IMAGE
    container_root: PurePosixPath = field(
        default_factory=lambda: DEFAULT_CONTAINER_ROOT
    )
    mounts: Sequence[SandboxMount] = field(default_factory=tuple)
    docker_executable: str = "docker"
    container_executable: str | PurePosixPath | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)
    _host_service_network: str | None = field(default=None, init=False, repr=False)
    container_name: str = field(
        default_factory=lambda: f"matharena-{os.getpid()}-{secrets.token_hex(8)}"
    )

    def __post_init__(self) -> None:
        super(DockerHarnessSandbox, self).__post_init__()
        self.container_root = PurePosixPath(self.container_root)
        if not self.container_root.is_absolute() or ".." in self.container_root.parts:
            raise ValueError("container_root must be an absolute container path")
        self.mounts = tuple(self.mounts)
        self.extra_args = tuple(str(arg) for arg in self.extra_args)
        if self.container_executable is not None:
            executable = PurePosixPath(self.container_executable)
            if executable.is_absolute() and ".." in executable.parts:
                raise ValueError("container_executable cannot contain '..'")
            self.container_executable = executable
        targets = [self.container_root, *(mount.target for mount in self.mounts)]
        if len(targets) != len(set(targets)):
            raise ValueError("sandbox mount targets must be unique")

    @property
    def enabled(self) -> bool:
        return True

    def _container_path(self, host_path: Path) -> PurePosixPath:
        return self.container_root.joinpath(*host_path.relative_to(self.root).parts)

    def _container_command(self, command: str | Sequence[str]) -> list[str]:
        argv = self._argv(command)
        executable = Path(argv[0]).expanduser()
        if self.container_executable is not None and executable.is_absolute():
            argv[0] = str(self.container_executable)
        elif executable.is_absolute():
            resolved = executable.resolve(strict=False)
            if resolved == self.root or self.root in resolved.parents:
                argv[0] = str(self._container_path(resolved))
            else:
                for mount in self.mounts:
                    if resolved == mount.source or mount.source in resolved.parents:
                        argv[0] = str(
                            mount.target.joinpath(
                                *resolved.relative_to(mount.source).parts
                            )
                        )
                        break
                else:
                    raise ValueError(
                        "absolute harness executable is not mounted in Docker; "
                        "set container_executable or add a SandboxMount"
                    )
        return argv

    def prepare_command(
        self,
        command: str | Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> tuple[list[str], Path, dict[str, str]]:
        network = self._host_service_network or (
            "bridge" if self.network_enabled else "none"
        )
        argv = [
            shutil.which(self.docker_executable) or self.docker_executable,
            "run",
            "--rm",
            "--interactive",
            "--name",
            self.container_name,
            "--network",
            network,
            "--workdir",
            str(self._container_path(cwd)),
            "--volume",
            f"{self.root}:{self.container_root}:rw",
        ]
        if self.network_enabled:
            argv.extend(("--add-host", "host.docker.internal:host-gateway"))
        for mount in self.mounts:
            mode = "rw" if mount.writable else "ro"
            argv.extend(("--volume", f"{mount.source}:{mount.target}:{mode}"))
        explicit_env = {str(key): str(value) for key, value in (env or {}).items()}
        for key, value in explicit_env.items():
            # Snap rewrites HOME/XDG/PATH before its Docker client resolves a
            # name-only --env. These path-only values are safe to pass inline.
            docker_value = f"{key}={value}" if key in INLINE_CONTAINER_ENV else key
            argv.extend(("--env", docker_value))
        argv.extend(self.extra_args)
        argv.append(self.image)
        argv.extend(self._container_command(command))
        process_env = os.environ.copy()
        process_env.update(
            {k: v for k, v in explicit_env.items() if k not in INLINE_CONTAINER_ENV}
        )
        return argv, self.root, process_env

    def stop(self) -> None:
        """Remove this attempt's container, including any running child tools."""
        subprocess.run(
            [
                shutil.which(self.docker_executable) or self.docker_executable,
                "rm",
                "--force",
                self.container_name,
            ],
            capture_output=True,
            timeout=10,
            check=False,
        )

    @contextmanager
    def expose_host_service(self) -> Iterator[HostServiceRoute]:
        if self._host_service_network is not None:
            raise RuntimeError("this Docker sandbox already has an active host service")
        if self.network_enabled:
            yield HostServiceRoute("0.0.0.0", "host.docker.internal", True)
            return

        network = f"matharena-harness-{os.getpid()}-{secrets.token_hex(6)}"
        created: subprocess.CompletedProcess[str] | None = None
        gateway = ""
        for _ in range(5):
            second_octet = 192 + secrets.randbelow(32)
            third_octet = secrets.randbelow(256)
            subnet = f"10.{second_octet}.{third_octet}.0/24"
            gateway = f"10.{second_octet}.{third_octet}.1"
            created = subprocess.run(
                [
                    self.docker_executable,
                    "network",
                    "create",
                    "--internal",
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
            if created.returncode == 0:
                break
        if created is None or created.returncode != 0:
            detail = created.stderr.strip() if created is not None else "unknown error"
            raise RuntimeError(f"cannot create isolated Docker proxy network: {detail}")
        self._host_service_network = network
        try:
            with self.gateway_host_service_route(gateway, self.docker_executable) as route:
                yield route
        finally:
            self._host_service_network = None
            subprocess.run(
                [self.docker_executable, "network", "rm", network],
                capture_output=True,
                text=True,
                check=False,
            )


@dataclass(slots=True)
class PodmanHarnessSandbox(PodmanSandbox):
    """Podman proxy isolation with MathArena's named-container cancellation."""

    network_enabled: bool = False
    image: str = DEFAULT_HARNESS_DOCKER_IMAGE
    container_root: PurePosixPath = field(
        default_factory=lambda: DEFAULT_CONTAINER_ROOT
    )
    container_name: str = field(
        default_factory=lambda: f"matharena-{os.getpid()}-{secrets.token_hex(8)}"
    )

    def prepare_command(
        self,
        command: str | Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
    ) -> tuple[list[str], Path, dict[str, str]]:
        explicit_env = {str(key): str(value) for key, value in (env or {}).items()}
        # Keep Podman's own HOME/XDG/PATH intact so it uses the host's image
        # store and configuration. Container paths are safe to pass inline;
        # credentials still use name-only --env entries from the base sandbox.
        process_values = {
            key: value for key, value in explicit_env.items()
            if key not in INLINE_CONTAINER_ENV
        }
        argv, host_cwd, process_env = super(PodmanHarnessSandbox, self).prepare_command(
            command, cwd=cwd, env=process_values
        )
        options = ["--name", self.container_name]
        for key, value in explicit_env.items():
            if key in INLINE_CONTAINER_ENV:
                options.extend(("--env", f"{key}={value}"))
        argv[2:2] = options
        return argv, host_cwd, process_env

    def stop(self) -> None:
        """Remove this attempt's container, including any running child tools."""
        subprocess.run(
            # Podman otherwise waits its default ten-second stop grace period,
            # colliding with this cleanup timeout and delaying budget enforcement.
            [self.podman_executable, "rm", "--force", "--time", "0", self.container_name],
            capture_output=True,
            timeout=10,
            check=False,
        )


class HarnessSolver(BaseSolver):
    """Run one isolated harness session for each problem/run pair."""

    def __init__(
        self,
        solver_config,
        default_prompt_template,
        default_api_client_args,
        last_chance_prompt,
        *,
        container_runtime: str = "docker",
    ):
        super().__init__(
            solver_config,
            default_prompt_template,
            default_api_client_args,
            last_chance_prompt,
        )
        self.config = default_api_client_args.copy()
        self.pricing = TokenPricing.from_config(self.config)
        self.harness = str(self.config["harness"])
        self.harness_key = self.harness.lower().replace("_", "-")
        raw_harness_version = self.config.get("harness_version", "latest")
        if not isinstance(raw_harness_version, str) or not raw_harness_version.strip():
            raise TypeError("harness_version must be a non-empty string")
        self.harness_version = raw_harness_version.strip()
        raw_harness_config = self.config.get("harness_config") or {}
        if not isinstance(raw_harness_config, dict):
            raise TypeError("harness_config must be a mapping")
        self.harness_config = raw_harness_config.copy()
        self.harness_config.pop("container_runtime", None)
        self.container_runtime = validate_container_runtime(container_runtime)
        if not isinstance(self.harness_config.get("direct_network", False), bool):
            raise TypeError("harness_config.direct_network must be a boolean")
        allow_harness = self.config.get("allow_harness", True)
        requested_tools = self.harness_config.get("tools_enabled", True)
        if not isinstance(allow_harness, bool) or not isinstance(requested_tools, bool):
            raise TypeError(
                "allow_harness and harness_config.tools_enabled must be booleans"
            )
        self.tools_enabled = allow_harness and requested_tools
        if not self.tools_enabled and self.harness_key not in {
            "codex",
            "codex-cli",
            "openai-codex",
        }:
            raise ValueError("Tool-free harness mode is supported only by Codex CLI")
        self.competition = str(self.config.get("competition") or "unknown_competition")
        self.solver_name = str(self.config.get("solver_name") or "harness")
        self.is_lean_comp = bool(self.config.get("is_lean_comp", False))
        if self.is_lean_comp and not self.tools_enabled:
            raise ValueError(
                "Lean harness runs require allow_harness: true and tools_enabled: true"
            )
        self.lean_environment = str(self.config.get("lean_environment") or "")
        self.concurrent_requests = int(self.config.get("concurrent_requests") or 1)
        self.keep_workspaces = bool(self.harness_config.get("keep_workspaces", True))
        self.minimal_context = bool(self.harness_config.get("minimal_context", True))
        if not self.tools_enabled:
            self.minimal_context = True
        self.run_check_after_harness = bool(
            self.harness_config.get("run_check_after_harness", self.is_lean_comp)
        )
        if self.harness_key not in HARNESS_RELEASES and not (
            self.harness_config.get("container_image")
            or self.harness_config.get("docker_image")
        ):
            raise ValueError(
                "Unrecognized harnesses must set harness_config.container_image and provide "
                "their executable in that image."
            )
        self.manage_harness_cli = bool(self.harness_config.get("managed_cli", True))
        self._harness_cli: InstalledCLI | None = None
        self._harness_cli_lock = threading.Lock()
        self.workspace_root_template = str(
            self.harness_config.get("workspace_root") or DEFAULT_WORKSPACE_ROOT
        )
        self.host_lean_project = Path(
            str(
                self.harness_config.get("lean_project_root")
                or "external/comparator_project"
            )
        ).expanduser()
        if not self.host_lean_project.is_absolute():
            self.host_lean_project = Path.cwd() / self.host_lean_project
        self.host_lean_project = self.host_lean_project.resolve()
        self.host_lake_cache = Path(
            str(
                self.harness_config.get("lake_cache_root")
                or self.host_lean_project / ".lake"
            )
        ).expanduser()
        if not self.host_lake_cache.is_absolute():
            self.host_lake_cache = Path.cwd() / self.host_lake_cache
        self.host_lake_cache = self.host_lake_cache.resolve()
        self.host_elan_home = (
            Path(str(self.harness_config.get("elan_home") or Path.home() / ".elan"))
            .expanduser()
            .resolve()
        )
        self.lean_runtime = str(
            self.harness_config.get("lean_runtime") or "image"
        ).lower()
        if self.lean_runtime not in {"host", "image"}:
            raise ValueError("harness_config.lean_runtime must be 'host' or 'image'")
        self.lean_cache_volume = self.harness_config.get("lean_cache_volume")
        self.cache_read_only = bool(self.harness_config.get("cache_read_only", True))
        self.container_userns = self.harness_config.get("userns") or os.environ.get(
            "MATHARENA_CONTAINER_USERNS"
        )
        if (
            not self.container_userns
            and self.container_runtime == "podman"
            and os.getuid() != 0
        ):
            # Native rootless Podman otherwise maps the host user to container
            # root, making --user UID:GID unable to write the host workspace.
            self.container_userns = "keep-id"
        self.package_names = _package_names(self.host_lean_project)
        self.toolchain_dir = str(
            self.harness_config.get("lean_toolchain_dir")
            or _toolchain_dir(self.host_lean_project)
        )
        expected_lean = self.lean_environment.removeprefix("lean-")
        if (
            self.is_lean_comp
            and expected_lean
            and f"v{expected_lean}" not in self.toolchain_dir
        ):
            raise ValueError(
                f"Competition requires Lean {expected_lean}, but the harness toolchain is "
                f"{self.toolchain_dir!r}. Use a version-matched image/toolchain."
            )
        self._response_agents: dict[int, Agent] = {}
        self._response_budgets: dict[
            int, tuple[RunBudget, DockerHarnessSandbox | PodmanHarnessSandbox]
        ] = {}

    def _prepare_harness_cli(self) -> InstalledCLI | None:
        """Resolve one CLI version before workers start and reuse its read-only cache."""

        if not self.manage_harness_cli:
            return None
        release_name = HARNESS_RELEASES.get(self.harness_key)
        if release_name is None:
            return None
        with self._harness_cli_lock:
            if self._harness_cli is None:
                self._harness_cli = ensure_cli(
                    release_name,
                    self.harness_version,
                    cache_root=self.harness_config.get("cli_cache_root"),
                )
                logger.info(
                    "Using {} {} for {} harness (requested {})",
                    release_name,
                    self._harness_cli.release.version,
                    self.harness,
                    self.harness_version,
                )
            return self._harness_cli

    def build_prompt(self, text: str | dict[str, Any] | None) -> str:
        if isinstance(text, dict):
            fields = {key: value for key, value in text.items() if value is not None}
            fields.setdefault("problem", "")
        else:
            fields = {"problem": "" if text is None else text}
        return self.default_prompt_template.format_map(_PromptFields(fields))

    @override
    def solve_batch(
        self,
        stmt_batch: list[tuple[str, Any]],
        batch_idx_to_problem_idx: dict[int, int],
        batch_idx_to_run_idx: dict[int, int],
    ):
        self._prepare_harness_cli()
        max_workers = max(1, min(self.concurrent_requests, len(stmt_batch)))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(
                    self._solve_one,
                    batch_idx,
                    payload,
                    image_b64,
                    batch_idx_to_problem_idx[batch_idx],
                    batch_idx_to_run_idx[batch_idx],
                ): batch_idx
                for batch_idx, (payload, image_b64) in enumerate(stmt_batch)
            }
            for future in as_completed(futures):
                batch_idx = futures[future]
                try:
                    yield future.result()
                except Exception as exc:
                    # Do not unwind the generator: that waits for remaining
                    # workers but discards their successful responses before
                    # Runner can persist them. Leave only this run pending.
                    returncode = getattr(exc, "returncode", None)
                    exit_detail = (
                        f" (harness CLI exit {returncode})"
                        if returncode is not None
                        else ""
                    )
                    # Exception text may contain commands or credentials; report
                    # only the exception type and the OS's standard errno message.
                    error_detail = type(exc).__name__
                    if isinstance(exc, OSError) and isinstance(exc.errno, int):
                        error_detail += (
                            f": [errno {exc.errno}] {os.strerror(exc.errno)}"
                        )
                    logger.error(
                        "Harness run failed for problem {} run {}{}: {}; leaving it pending",
                        batch_idx_to_problem_idx[batch_idx],
                        batch_idx_to_run_idx[batch_idx],
                        exit_detail,
                        error_detail,
                    )

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        if previous_response.detailed_cost.get("refusal_category") == "reasoning_extraction":
            return previous_response
        agent = self._response_agents.get(id(previous_response))
        if agent is None:
            return previous_response
        budget, sandbox = self._response_budgets.get(
            id(previous_response), (None, None)
        )
        if budget and (budget.exceeded() or budget.cost_grace is not None):
            previous_response.detailed_cost["run_limits"] = budget.metadata()
            return previous_response
        started = time.monotonic()
        events = (
            self._run_with_budget(
                agent, sandbox, budget, self.last_chance_prompt, resume=True
            )
            if budget and budget.has_limits
            else agent.resume(self.last_chance_prompt, session_id=agent.session_id)
        )
        elapsed_s = time.monotonic() - started
        additions = [{"role": "user", "content": self.last_chance_prompt}]
        additions.extend(_events_to_messages(events, harness=self.harness_key))
        refusal = _has_reasoning_extraction_refusal(events)
        if refusal:
            _append_refusal_message(additions)
        previous_response.conversation.extend(additions)
        detailed_cost = (
            self._detailed_cost(agent, usage=budget.usage(agent.get_tokens()), budget=budget)
            if budget
            else self._detailed_cost(agent)
        )
        detailed_cost["n_retries"] = (
            previous_response.detailed_cost.get("n_retries", 0)
            + _count_harness_retries(events)
        )
        if budget and budget.has_limits:
            detailed_cost["run_limits"] = budget.metadata()
            if budget.exceeded() and not _ends_with_response(
                previous_response.conversation,
                previous_response.conversation[-1].get("content"),
            ):
                previous_response.conversation.append(
                    {"role": "assistant", "type": "response", "content": ""}
                )
        for key in ("time", "request_time"):
            previous = previous_response.detailed_cost.get(key)
            # Do not misreport a partial duration as the total of an untimed run.
            detailed_cost[key] = previous + elapsed_s if previous is not None else None
        if refusal:
            detailed_cost["refusal_category"] = "reasoning_extraction"
        previous_response.detailed_cost = detailed_cost
        previous_response.history.append(
            {
                "step": "harness_last_chance",
                "timestep": len(previous_response.history),
                "elapsed_s": elapsed_s,
                "messages": additions,
                "harness": self.harness,
                "tools_enabled": self.tools_enabled,
                "harness_version": (
                    self._harness_cli.release.version
                    if self._harness_cli is not None
                    else self.harness_version
                ),
                "harness_version_requested": self.harness_version,
                "session_id": agent.session_id,
                "model_requests": self._model_request_history_path(agent),
                "events": [event.to_dict() for event in events],
            }
        )
        return previous_response

    def _solve_one(
        self,
        batch_idx: int,
        payload: str | dict[str, Any] | None,
        image_b64: str | None,
        problem_idx: int,
        run_idx: int,
    ) -> SolverResponse:
        self._prepare_harness_cli()
        if image_b64 is not None:
            raise ValueError("HarnessSolver currently supports text problems only")
        workspace = self._workspace_for(problem_idx, run_idx)
        self._prepare_workspace(workspace, payload)
        model = self._build_model()
        sandbox = self._build_sandbox(workspace)
        budget = None
        if any(
            self.harness_config.get(key) is not None
            for key in ("max_time_seconds", "max_cost_usd")
        ) or self.pricing.threshold is not None:
            budget = RunBudget(
                self.harness_config,
                lambda usage: self._detailed_cost(None, usage=usage)["cost"],
                priced=all(
                    self.config.get(key) is not None
                    for key in ("read_cost", "write_cost")
                ),
                request_level_cost=self.pricing.threshold is not None,
                base_cost_for_usage=lambda usage: self._detailed_cost(None, usage=usage, base_only=True)["cost"],
                native_usage_reader=NativeUsageReader(workspace, self.harness_key),
            )
            fields = (
                dict(payload)
                if isinstance(payload, dict)
                else {"problem": payload or ""}
            )
            fields.update(budget.prompt_fields())
            prompt = self.build_prompt(fields)
            time_prompt = self.harness_config.get("time_limit_prompt", "")
            if time_prompt:
                prompt += "\n\n" + time_prompt.format_map(
                    _PromptFields(budget.prompt_fields())
                )
        else:
            prompt = self.build_prompt(payload)
        agent = Agent(
            type=self.harness,
            model=model,
            env=sandbox,
            dir=workspace,
            subagents=self.harness_config.get("subagents", {}),
            executable=self._agent_executable(),
            # Managed binaries have already been version-checked before they
            # are mounted. Agent sees a harmless host placeholder which the
            # sandbox remaps to that mounted executable.
            validate_version=(
                bool(self.harness_config.get("validate_version", False))
                and self._harness_cli is None
            ),
            environment=self._agent_environment(),
            minimal_context=self.minimal_context,
            tools_enabled=self.tools_enabled,
            model_context_window=self.harness_config.get("model_context_window"),
            on_model_response=budget.capture_response if budget else None,
            on_event=(
                lambda event: budget.usage(agent.get_tokens())
                if event.type == "usage"
                else None
            )
            if budget
            else None,
            should_stop=budget.exceeded if budget and budget.has_limits else None,
            auto_wait=bool(self.harness_config.get("auto_wait", True)),
            auto_fallback=bool(self.harness_config.get("auto_fallback", True)),
            rate_limit_timeout=self.harness_config.get("rate_limit_timeout"),
            rate_limit_poll_interval=float(
                self.harness_config.get("rate_limit_poll_interval", 30)
            ),
            max_recovery_attempts=int(
                self.harness_config.get("max_recovery_attempts", 10)
            ),
        )
        logger.info(
            f"Starting {self.harness} harness run for P{problem_idx} r{run_idx} "
            f"(container_runtime={self.container_runtime}, tools_enabled={self.tools_enabled}) in {workspace}"
        )
        started = time.monotonic()
        events = (
            self._run_with_budget(agent, sandbox, budget, prompt)
            if budget and budget.has_limits
            else agent.run(prompt)
        )
        elapsed_s = time.monotonic() - started
        final_result = next((event for event in reversed(events) if event.type == "result"), None)
        # An explicit empty completion must not trigger a final-answer reprompt.
        # Lean runs still submit Solution.lean, regardless of the final chat text.
        refusal = _has_reasoning_extraction_refusal(events)
        empty_completion = (
            not refusal and final_result is not None
            and not str(final_result.content or "").strip()
        )
        check_result = None
        if self.run_check_after_harness and not refusal and not (budget and budget.exceeded()):
            check_result = sandbox.run(
                ["./check.sh"],
                env=self._agent_environment(),
                timeout=float(self.harness_config.get("check_timeout_s", 600)),
            )
            _write_text(workspace / "check_stdout.log", check_result.stdout)
            _write_text(workspace / "check_stderr.log", check_result.stderr)

        conversation = [
            {"role": "user", "content": prompt},
            *_events_to_messages(events, harness=self.harness_key),
        ]
        if empty_completion and not _ends_with_response(conversation, ""):
            conversation.append({"role": "assistant", "type": "response", "content": ""})
        solution_text = (
            _read_text(workspace / "Solution.lean") if self.is_lean_comp else ""
        )
        if solution_text.strip() and not refusal:
            final_content = f"```lean\n{solution_text.rstrip()}\n```"
            if not _ends_with_response(conversation, final_content):
                conversation.append(
                    {"role": "assistant", "type": "response", "content": final_content}
                )
        if refusal:
            _append_refusal_message(conversation)
        last_message = conversation[-1]
        if not (
            last_message.get("role") == "assistant"
            and last_message.get("type", "response") == "response"
            and isinstance(last_message.get("content"), str)
            and (last_message["content"].strip() or final_result is not None)
        ):
            if budget and budget.exceeded():
                last_message = {"role": "assistant", "type": "response", "content": ""}
                conversation.append(last_message)
            else:
                raise RuntimeError(
                    f"{self.harness} P{problem_idx} r{run_idx} ended without a final response; "
                    "refusing to save this attempt as a completed run"
                )
        if not last_message["content"].strip():
            logger.warning(
                f"{self.harness} P{problem_idx} r{run_idx} completed with an empty "
                "final response; saving the attempt and its token usage"
            )

        history = [
            {
                "step": "harness",
                "timestep": 0,
                **({"empty_completion": True} if empty_completion else {}),
                "elapsed_s": elapsed_s,
                "messages": conversation,
                "harness": self.harness,
                "container_runtime": self.container_runtime,
                "tools_enabled": self.tools_enabled,
                "harness_version": (
                    self._harness_cli.release.version
                    if self._harness_cli is not None
                    else self.harness_version
                ),
                "harness_version_requested": self.harness_version,
                "session_id": agent.session_id,
                "model_requests": self._model_request_history_path(agent),
                "workspace": _display_workspace_path(workspace),
                "events": [event.to_dict() for event in events],
                "check_returncode": check_result.returncode
                if check_result is not None
                else None,
            }
        ]
        detailed_cost = (
            self._detailed_cost(agent, usage=budget.usage(agent.get_tokens()), budget=budget)
            if budget
            else self._detailed_cost(agent)
        )
        detailed_cost["n_retries"] = _count_harness_retries(events)
        if budget and budget.has_limits:
            detailed_cost["run_limits"] = budget.metadata()
        detailed_cost.update(time=elapsed_s, request_time=elapsed_s)
        if refusal:
            detailed_cost["refusal_category"] = "reasoning_extraction"
        response = SolverResponse(
            batch_idx, conversation, detailed_cost, history=history
        )
        if not empty_completion and not refusal:
            self._response_agents[id(response)] = agent
            if budget:
                self._response_budgets[id(response)] = (budget, sandbox)
        if not self.keep_workspaces:
            shutil.rmtree(workspace)
        return response

    def _run_with_budget(self, agent, sandbox, budget, prompt, *, resume=False):
        events = self._stream_with_budget(agent, sandbox, budget, prompt, resume=resume)
        # A CLI can report its usage only as it exits, before the watcher sees it.
        budget.exceeded()
        if _has_reasoning_extraction_refusal(events):
            return events
        grace_prompt = (
            budget.begin_cost_grace() if agent.session_id is not None else None
        )
        if grace_prompt is not None:
            logger.warning(
                "Cost limit reached; resuming session {} for a final answer until {}",
                agent.session_id,
                budget.cost_grace["deadline_at"],
            )
            events.append(AgentEvent(type="message", role="user", content=grace_prompt))
            final_events = self._stream_with_budget(
                agent, sandbox, budget, grace_prompt, resume=True
            )
            events.extend(final_events)
            messages = _events_to_messages(final_events, harness=self.harness_key)
            final_text = messages[-1].get("content", "") if messages else ""
            budget.finish_cost_grace(
                not _has_reasoning_extraction_refusal(final_events)
                and isinstance(final_text, str)
                and bool(final_text.strip())
                and _ends_with_response(messages, final_text)
            )
        return events

    def _stream_with_budget(self, agent, sandbox, budget, prompt, *, resume=False):
        done = threading.Event()

        def watch():
            while not done.wait(0.25):
                budget.usage(agent.get_tokens())
                if budget.exceeded():
                    logger.warning("Harness stopped: {}", budget.reason)
                    try:
                        agent.interrupt()
                        if not done.wait(2):
                            try:
                                sandbox.stop()
                            finally:
                                agent.terminate()
                    except (OSError, subprocess.TimeoutExpired):
                        logger.warning(
                            "Harness process cleanup failed after reaching its limit"
                        )
                    return

        watcher = threading.Thread(target=watch, daemon=True, name="harness-budget")
        watcher.start()
        events = []
        try:
            stream = (
                agent.stream_resume(prompt, session_id=agent.session_id)
                if resume
                else agent.stream(prompt)
            )
            events.extend(stream)
        except Exception:
            budget.usage(agent.get_tokens())
            if not budget.exceeded():
                if budget.cost_grace is None:
                    raise
                budget.finish_cost_grace(False)
                logger.warning(
                    "Final budget response failed; saving the attempt as incorrect"
                )
        finally:
            done.set()
            watcher.join(timeout=12)
            budget.usage(agent.get_tokens())
        return events

    @staticmethod
    def _model_request_history_path(agent: Agent) -> str | None:
        """Return a safe display path for this run's exact proxied requests."""

        path = getattr(agent, "model_request_log_path", None)
        if path is None or not Path(path).is_file():
            return None
        return _display_workspace_path(Path(path))

    def _workspace_for(self, problem_idx: int, run_idx: int) -> Path:
        raw = _format_template(
            self.workspace_root_template,
            {
                "competition": self.competition,
                "solver_name": self.solver_name,
                "problem_idx": problem_idx,
                "run_idx": run_idx,
            },
        )
        path = Path(raw).expanduser()
        return (path if path.is_absolute() else Path.cwd() / path).resolve()

    def _prepare_workspace(
        self, workspace: Path, payload: str | dict[str, Any] | None
    ) -> None:
        self._reset_workspace(workspace)
        workspace.mkdir(parents=True, exist_ok=True)
        # Seed files into the container HOME. Some CLIs expose settings only
        # through their config file, with no environment variable to set them.
        for name, content in (self.harness_config.get("home_files") or {}).items():
            target = workspace / ".harness-home" / str(name)
            if ".." in Path(name).parts or Path(name).is_absolute():
                raise ValueError(f"harness_config.home_files path escapes HOME: {name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_text(target, str(content))
        if not self.is_lean_comp:
            return
        formal = (
            str(payload.get("formal_statement") or "")
            if isinstance(payload, dict)
            else ""
        )
        initial_solution = (
            f"import Mathlib\n\n{formal.rstrip()}\n"
            if formal.strip()
            else "import Mathlib\n\n"
        )
        if self.config.get("lean_task") == "prove_or_disprove":
            from matharena.lean_conjecture import target_statement

            # A neutral template, not a gold label; the solver may choose either branch.
            initial_solution = target_statement(formal, "prove")
        solution_path = workspace / "Solution.lean"
        if not solution_path.exists() or self.harness_config.get(
            "overwrite_solution", True
        ):
            _write_text(solution_path, initial_solution)
        _write_text(workspace / "check.sh", self._lean_check_script())
        (workspace / "check.sh").chmod(0o755)

    @staticmethod
    def _reset_workspace(workspace: Path) -> None:
        """Start every independent run from a newly created filesystem state."""

        resolved = workspace.resolve(strict=False)
        protected_roots = (
            Path.cwd().resolve(),
            Path.home().resolve(),
            Path(tempfile.gettempdir()).resolve(),
        )
        protected = {
            candidate for root in protected_roots for candidate in (root, *root.parents)
        }
        if resolved in protected:
            raise ValueError(f"refusing to reset unsafe harness workspace: {resolved}")
        if resolved.exists():
            if not resolved.is_dir():
                raise ValueError(f"harness workspace is not a directory: {resolved}")
            logger.debug("Resetting existing harness workspace {}", resolved)
            shutil.rmtree(resolved)

    def _lean_check_script(self) -> str:
        return (
            "#!/usr/bin/env sh\n"
            "set -eu\n"
            f'export PATH="{CONTAINER_ELAN_HOME}/toolchains/{self.toolchain_dir}/bin:'
            f'{CONTAINER_ELAN_HOME}/bin:$PATH"\n'
            'check_log="${TMPDIR:-/tmp}/lean-check-$$.log"\n'
            'if ! lean Solution.lean > "$check_log" 2>&1; then\n'
            '  cat "$check_log"\n'
            "  exit 1\n"
            "fi\n"
            'cat "$check_log"\n'
            'if grep -q "warning:" "$check_log"; then exit 1; fi\n'
        )

    def _agent_executable(self) -> str | None:
        configured = self.harness_config.get("executable")
        if configured:
            return str(configured)
        cli_name = HARNESS_EXECUTABLES.get(self.harness_key)
        if cli_name is None:
            return None
        host_cli = shutil.which(cli_name)
        if host_cli is not None:
            return host_cli
        # The command is always remapped to container_executable before
        # spawning. Agent still validates an explicit host path, so use a
        # harmless existing executable as the container-only placeholder.
        placeholder = shutil.which("true")
        if placeholder is None:
            raise RuntimeError(
                f"cannot locate host placeholder for container CLI {cli_name!r}"
            )
        return placeholder

    def _build_model(self):
        reasoning = _reasoning_effort(self.config)
        auth = str(self.harness_config.get("auth") or "api").lower()
        if auth != "api" and self.harness_key in {
            "muse",
            "muse-code",
            "muse-cli",
            "agy",
            "antigravity",
            "antigravity-cli",
            "google-antigravity",
            "gravity",
            "gravity-cli",
            "qwen",
            "qwen-cli",
            "qwen-code",
            "open-code",
            "opencode",
            "grok",
            "grok-build",
            "grok-cli",
            "deepcode",
            "deepcode-cli",
            "deep-code",
            "vibe",
            "vibe-cli",
            "mistral-vibe",
            "mistral",
        }:
            raise ValueError(f"{self.harness} currently supports API-credit auth only")
        if auth in {"chatgpt", "oauth", "subscription"}:
            harness_name = self.harness.lower().replace("_", "-")
            default_provider = {
                "anthropic": "anthropic",
                "claude": "anthropic",
                "claude-code": "anthropic",
                "kimi": "kimi",
                "kimi-code": "kimi",
                "moonshot": "kimi",
            }.get(harness_name, "openai")
            oauth_provider = str(
                self.harness_config.get("oauth_provider")
                or ("openai" if auth == "chatgpt" else default_provider)
            ).lower()
            common_oauth_args = {
                # Benchmark workers must never stop for an unexpected browser
                # flow. Authenticate on the host before starting the run.
                "auto_login": bool(self.harness_config.get("oauth_auto_login", False)),
                "auto_relogin": bool(
                    self.harness_config.get("oauth_auto_relogin", False)
                ),
                "expiry_leeway": float(
                    self.harness_config.get("oauth_expiry_leeway_s", 30)
                ),
            }
            if oauth_provider == "openai":
                oauth_config = openai_oauth_config(
                    device_auth=bool(
                        self.harness_config.get("oauth_device_auth", True)
                    ),
                    **common_oauth_args,
                )
            elif oauth_provider == "anthropic":
                oauth_config = anthropic_oauth_config(**common_oauth_args)
            elif oauth_provider == "kimi":
                oauth_config = kimi_oauth_config(**common_oauth_args)
            else:
                raise ValueError(
                    "harness_config.oauth_provider must be 'openai', 'anthropic', or 'kimi'"
                )
            return Model(
                str(self.config["model"]),
                oauth=oauth_provider,
                oauth_config=oauth_config,
                reasoning=reasoning,
            )
        if auth != "api":
            raise ValueError(
                "harness_config.auth must be 'api', 'subscription', or 'oauth'"
            )

        api_args = self.config.copy()
        for key in (
            "competition",
            "harness",
            "harness_version",
            "harness_config",
            "allow_harness",
            "is_lean_comp",
            "lean_task",
            "lean_environment",
            "max_tool_calls",
            "lean_environment_override",
            "solver_name",
            "tools",
        ):
            api_args.pop(key, None)
        # Harnesses execute interactive calls; batch settings belong to the API solver.
        api_args["batch_processing"] = False
        api_args.pop("batch_max_tokens", None)
        api_args.pop("anthropic_betas", None)
        requested_api = str(api_args.get("api") or "openai").lower()
        client = APIClient(**api_args)
        antigravity_harness = self.harness_key in {
            "agy",
            "antigravity",
            "antigravity-cli",
            "google-antigravity",
            "gravity",
            "gravity-cli",
        }
        native_google_harness = self.harness_key in {
            "open-code",
            "opencode",
            "qwen",
            "qwen-cli",
            "qwen-code",
        }
        if antigravity_harness or (native_google_harness and requested_api == "google"):
            if antigravity_harness and requested_api != "google":
                raise ValueError(
                    "Antigravity CLI requires a model config with api: google"
                )
            if not client.api_key:
                raise ValueError("Native Gemini harness auth requires GOOGLE_API_KEY")
            generation_config: dict[str, object] = {}
            for source, target in (
                ("max_tokens", "maxOutputTokens"),
                ("max_completion_tokens", "maxOutputTokens"),
                ("max_output_tokens", "maxOutputTokens"),
                ("temperature", "temperature"),
                ("top_p", "topP"),
                ("top_k", "topK"),
                ("presence_penalty", "presencePenalty"),
                ("frequency_penalty", "frequencyPenalty"),
                ("stop", "stopSequences"),
            ):
                if source in client.kwargs:
                    generation_config[target] = client.kwargs[source]
            configured_thinking = _gemini_thinking_config(client.kwargs)
            if configured_thinking is not None:
                generation_config["thinkingConfig"] = configured_thinking
            if reasoning is not None:
                generation_config["thinkingConfig"] = {
                    "includeThoughts": True,
                    "thinkingLevel": reasoning,
                }
            request_overrides: dict[str, object] = {}
            if generation_config:
                request_overrides["generationConfig"] = generation_config
            explicit_overrides = self.harness_config.get("request_overrides", {})
            if not isinstance(explicit_overrides, Mapping):
                raise TypeError("harness_config.request_overrides must be a mapping")
            request_overrides.update(explicit_overrides)
            return Model(
                client.model,
                api_url=str(
                    self.harness_config.get("gemini_base_url")
                    or "https://generativelanguage.googleapis.com"
                ),
                api_key=secrets.token_urlsafe(32),
                api_type="gemini",
                headers={"x-goog-api-key": str(client.api_key)},
                request_overrides=request_overrides,
                # Antigravity initializes thinkingBudget=-1 itself, then its
                # --effort flag adds thinkingLevel. Gemini rejects requests
                # containing both mutually exclusive controls.
                request_drop_fields=("generationConfig.thinkingConfig.thinkingBudget",)
                if reasoning is not None
                else (),
                reasoning=reasoning,
            )
        protocol = {
            "openai": "openai",
            "anthropic": "anthropic",
            "openrouter": "openai",
            "together": "openai",
        }.get(client.api)
        if protocol is None:
            raise ValueError(
                f"Harnesses require an OpenAI- or Anthropic-compatible endpoint, got {client.api!r}"
            )
        openrouter_claude = requested_api == "openrouter" and self.harness_key in {
            "claude", "claude-code", "anthropic"
        }
        base_url = client.base_url or (
            "https://api.openai.com/v1"
            if protocol == "openai"
            else "https://api.anthropic.com"
        )
        if openrouter_claude:
            # Claude Code speaks Messages, not Chat Completions, through this gateway.
            protocol = "anthropic"
            base_url = "https://openrouter.ai/api"
        upstream_headers = dict(client.default_headers or {})
        bearer_auth = protocol == "openai" or openrouter_claude
        auth_header = "Authorization" if bearer_auth else "x-api-key"
        if not any(name.lower() == auth_header.lower() for name in upstream_headers):
            upstream_headers[auth_header] = (
                f"Bearer {client.api_key}"
                if bearer_auth
                else str(client.api_key)
            )
        request_overrides: dict[str, object] = {}
        if openrouter_claude:
            # Preserve routing/caching; the CLI supplies native thinking and effort.
            extra_body = client.kwargs.get("extra_body", {})
            if not isinstance(extra_body, Mapping):
                raise TypeError("API extra_body must be a mapping")
            request_overrides.update(extra_body)
        if self.harness.lower().replace("_", "-") in {
            "kimi",
            "kimi-code",
            "moonshot",
            "qwen",
            "qwen-cli",
            "qwen-code",
            "open-code",
            "opencode",
            "deepcode",
            "deepcode-cli",
            "deep-code",
            "vibe",
            "vibe-cli",
            "mistral-vibe",
            "mistral",
        }:
            # These CLIs own the Chat Completions request body. Preserve the
            # parameters that the normal MathArena API client would send.
            request_overrides.update(client.kwargs)
            if "tool_choice" in api_args:
                request_overrides["tool_choice"] = api_args["tool_choice"]
            extra_body = request_overrides.pop("extra_body", None)
            if extra_body is not None:
                if not isinstance(extra_body, Mapping):
                    raise TypeError("API extra_body must be a mapping")
                request_overrides.update(extra_body)
        if self.harness_key in {"muse", "muse-code", "muse-cli", "grok", "grok-build", "grok-cli"}:
            # These native backends use Responses even when an API config uses Chat Completions.
            request_overrides.update(client.kwargs)
            for field in ("max_tokens", "max_completion_tokens"):
                if field in request_overrides:
                    request_overrides["max_output_tokens"] = request_overrides.pop(
                        field
                    )
            effort = request_overrides.pop("reasoning_effort", reasoning)
            if effort is not None:
                request_overrides["reasoning"] = {"effort": effort}
        if self.harness_key in {"grok", "grok-build", "grok-cli"}:
            extra_body = request_overrides.pop("extra_body", None)
            if extra_body is not None:
                if not isinstance(extra_body, Mapping):
                    raise TypeError("API extra_body must be a mapping")
                request_overrides.update(extra_body)
        explicit_overrides = self.harness_config.get("request_overrides", {})
        if not isinstance(explicit_overrides, Mapping):
            raise TypeError("harness_config.request_overrides must be a mapping")
        request_overrides.update(explicit_overrides)
        configured_drop_fields = self.harness_config.get("request_drop_fields", ())
        if isinstance(configured_drop_fields, str) or not isinstance(
            configured_drop_fields, Sequence
        ):
            raise TypeError("harness_config.request_drop_fields must be a sequence")
        request_drop_fields = {str(field) for field in configured_drop_fields}
        if requested_api == "google":
            # Kimi Code emits this OpenAI-specific cache hint. Gemini's
            # OpenAI-compatible endpoint rejects it as an unknown JSON field.
            request_drop_fields.add("prompt_cache_key")
        return Model(
            client.model,
            api_url=base_url,
            # Keep the real key in host-side proxy headers. The CLI sees only a placeholder.
            api_key=secrets.token_urlsafe(32),
            api_type=protocol,
            headers=upstream_headers,
            request_overrides=request_overrides,
            request_drop_fields=request_drop_fields,
            reasoning=reasoning,
        )

    def _build_sandbox(
        self, workspace: Path
    ) -> DockerHarnessSandbox | PodmanHarnessSandbox:
        image = str(
            self.harness_config.get("container_image")
            or self.harness_config.get("docker_image")
            or DEFAULT_HARNESS_DOCKER_IMAGE
        )
        mounts: list[SandboxMount] = []
        if self._harness_cli is not None:
            mounts.append(
                SandboxMount(
                    self._harness_cli.prefix,
                    CONTAINER_HARNESS_CLI_ROOT,
                )
            )
        extra_args = [
            "--memory",
            f"{self.harness_config.get('memory_gb', 12)}g",
            "--cpus",
            str(self.harness_config.get("cpu_limit", 4)),
            "--pids-limit",
            str(self.harness_config.get("pids_limit", 768)),
            "--cap-drop",
            "ALL",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--tmpfs",
            f"{CONTAINER_TMPDIR}:rw,nosuid,nodev,size={self.harness_config.get('tmpfs_size', '1g')},mode=1777",
        ]
        if self.container_userns:
            extra_args.extend(("--userns", str(self.container_userns)))
        image_enforces_no_new_privileges = bool(
            self.harness_config.get(
                "image_enforces_no_new_privileges",
                image == DEFAULT_HARNESS_DOCKER_IMAGE,
            )
        )
        if not image_enforces_no_new_privileges:
            extra_args.extend(("--security-opt", "no-new-privileges:true"))
        if self.harness_config.get("read_only_rootfs", True):
            extra_args.append("--read-only")
        if self.is_lean_comp and self.lean_runtime == "host":
            mounts.extend(
                (
                    SandboxMount(
                        self.host_elan_home, PurePosixPath(CONTAINER_ELAN_HOME)
                    ),
                    SandboxMount(
                        self.host_lake_cache,
                        PurePosixPath(f"{CONTAINER_LEAN_CACHE}/.lake"),
                        writable=not self.cache_read_only,
                    ),
                )
            )
        elif self.is_lean_comp and self.lean_cache_volume:
            suffix = ",readonly" if self.cache_read_only else ""
            extra_args.extend(
                (
                    "--mount",
                    f"type=volume,src={self.lean_cache_volume},dst={CONTAINER_LEAN_CACHE}{suffix}",
                )
            )
        extra_args.extend(
            str(value)
            for value in self.harness_config.get(
                "container_extra_args", self.harness_config.get("docker_extra_args", ())
            )
        )
        if self._harness_cli is not None:
            executable: str | PurePosixPath | None = (
                CONTAINER_HARNESS_CLI_ROOT
                / "bin"
                / self._harness_cli.release.executable
            )
        else:
            executable = self.harness_config.get(
                "container_executable"
            ) or HARNESS_EXECUTABLES.get(self.harness_key)
        sandbox_class = (
            PodmanHarnessSandbox
            if self.container_runtime == "podman"
            else DockerHarnessSandbox
        )
        return sandbox_class(
            root=workspace,
            network_enabled=bool(self.harness_config.get("direct_network", False)),
            image=image,
            container_root=DEFAULT_CONTAINER_ROOT,
            mounts=mounts,
            container_executable=executable,
            extra_args=extra_args,
        )

    def _agent_environment(self) -> dict[str, str]:
        # Persist native CLI sessions between the initial turn and last-chance
        # resume. The container itself is disposable, while /work is the one
        # writable bind mount shared by both invocations.
        harness_home = str(DEFAULT_CONTAINER_ROOT / ".harness-home")
        environment = {
            "ELAN_HOME": CONTAINER_ELAN_HOME,
            "HOME": harness_home,
            "KIMI_CODE_HOME": f"{harness_home}/.kimi-code",
            "VIRTUAL_ENV": CONTAINER_HARNESS_VENV,
            "XDG_CONFIG_HOME": f"{harness_home}/.config",
            "XDG_CACHE_HOME": f"{harness_home}/.cache",
            "XDG_DATA_HOME": f"{harness_home}/.local/share",
            "XDG_STATE_HOME": f"{harness_home}/.local/state",
            "TMPDIR": CONTAINER_TMPDIR,
            "PATH": CONTAINER_HARNESS_PATH,
        }
        if self.is_lean_comp:
            environment.update(
                _lean_environment(self.package_names, self.toolchain_dir)
            )
        environment.update(
            {
                str(key): str(value)
                for key, value in self.harness_config.get("environment", {}).items()
            }
        )
        return environment

    def _detailed_cost(self, agent: Agent | None, *, usage=None, budget=None, base_only=False) -> dict[str, Any]:
        if usage is None:
            usage = agent.get_tokens()
        cached = usage.cache_read_tokens
        if self.harness.lower() in {"claude", "claude-code", "anthropic"}:
            total_input = usage.input_tokens + cached
        else:
            total_input = usage.input_tokens
        pricing_input = total_input
        if self.pricing.threshold is not None and self.harness_key in {"codex", "codex-cli", "openai-codex"}:
            # Codex/OpenAI include cache writes in input_tokens; Anthropic
            # reports them separately. The calculator accepts disjoint counts.
            pricing_input = max(0, pricing_input - usage.cache_write_tokens)
        if budget and budget.request_level_cost:
            cost = budget.cost(usage)
        else:
            cost = self.pricing.cost(
                pricing_input, usage.output_tokens, cached, usage.cache_write_tokens, base_only=base_only,
            )
        result = {
            "cost": cost,
            "input_tokens": total_input,
            "cached_input_tokens": cached,
            "cached_write_tokens": usage.cache_write_tokens,
            "output_tokens": usage.output_tokens,
            "read_cost": self.config.get("read_cost"),
            "cache_read_cost": self.config.get("cache_read_cost"),
            "cache_write_cost": self.config.get("cache_write_cost"),
            "write_cost": self.config.get("write_cost"),
            "model": self.config.get("model"),
            "n_retries": 0,
        }
        if budget and budget.request_level_cost and budget.cost_estimated:
            result["long_context_cost_estimated"] = True
        return result


def _gemini_thinking_config(config: Mapping[str, Any]) -> dict[str, object] | None:
    """Translate the Google OpenAI-extension shape used by model YAML files."""

    current: object = config.get("extra_body")
    thinking: Mapping[str, object] | None = None
    while isinstance(current, Mapping):
        google = current.get("google")
        candidates = [
            google.get("thinking_config") if isinstance(google, Mapping) else None,
            google.get("thinkingConfig") if isinstance(google, Mapping) else None,
            current.get("thinking_config"),
            current.get("thinkingConfig"),
        ]
        thinking = next(
            (item for item in candidates if isinstance(item, Mapping)), None
        )
        if thinking is not None:
            break
        current = current.get("extra_body")
    if thinking is None:
        return None
    aliases = {
        "include_thoughts": "includeThoughts",
        "thinking_level": "thinkingLevel",
        "thinking_budget": "thinkingBudget",
    }
    return {aliases.get(str(key), str(key)): value for key, value in thinking.items()}


def _reasoning_effort(config: Mapping[str, Any]) -> str | None:
    model = str(config.get("model") or "")
    if "--" in model:
        return model.split("--", 1)[1]
    if config.get("reasoning_effort"):
        return str(config["reasoning_effort"])
    if config.get("model_reasoning_effort"):
        return str(config["model_reasoning_effort"])
    output_config = config.get("output_config")
    if isinstance(output_config, Mapping) and output_config.get("effort"):
        return str(output_config["effort"])
    return None


def _coalesce_grok_events(events: Sequence[AgentEvent]) -> Iterator[AgentEvent]:
    """Join native Grok deltas without changing the raw event history.

    Reasoning and visible text remain separate blocks. Tool interactions, user
    messages, and recovery/session boundaries end a block; usage and tool-status
    notifications do not split an otherwise continuous stream.
    """
    first: AgentEvent | None = None
    parts: list[str] = []

    def block() -> AgentEvent:
        assert first is not None
        return AgentEvent(
            type="message" if first.type == "text" else "reasoning",
            content="".join(parts),
            role="assistant",
            session_id=first.session_id,
            timestamp=first.timestamp,
        )

    for event in events:
        if event.type in {"usage", "event"}:
            # These events were never conversation messages. Keep them in the
            # original history, rather than splitting an assistant's text.
            continue
        if event.type in {"text", "reasoning"}:
            content = str(event.content or "")
            if not content:
                continue
            if first is not None and first.type != event.type:
                yield block()
                first, parts = None, []
            if first is None:
                first = event
            parts.append(content)
            continue
        if first is not None:
            yield block()
            first, parts = None, []
        yield event
    if first is not None:
        yield block()


def _events_to_messages(
    events: Sequence[AgentEvent], *, harness: str | None = None
) -> list[dict[str, Any]]:
    # Other harnesses can emit both text deltas and complete message snapshots.
    # Only Grok's deltas are authoritative here; treating all harnesses alike
    # would duplicate their assistant responses.
    normalized_events = (
        _coalesce_grok_events(events)
        if (harness or "").lower().replace("_", "-") in {"grok", "grok-build", "grok-cli"}
        else iter(events)
    )
    messages: list[dict[str, Any]] = []
    for event in normalized_events:
        if event.type == "message" and event.role == "user":
            messages.append({"role": "user", "content": str(event.content or "")})
        elif event.type == "message" and event.role in {None, "assistant"}:
            messages.append(
                {
                    "role": "assistant",
                    "type": "response",
                    "content": str(event.content or ""),
                }
            )
        elif event.type == "reasoning":
            messages.append(
                {
                    "role": "assistant",
                    "type": "cot",
                    "content": str(event.content or ""),
                }
            )
        elif event.type == "tool_call":
            arguments = (
                event.content
                if isinstance(event.content, Mapping)
                else {"command": str(event.content or "")}
            )
            messages.append(
                {
                    "role": "assistant",
                    "type": "tool_call",
                    "tool_name": event.tool_name or "harness_tool",
                    "tool_call_id": event.tool_call_id,
                    "arguments": arguments,
                }
            )
        elif event.type == "tool_result":
            messages.append(
                {
                    "role": "tool_response",
                    "tool_name": event.tool_name or "harness_tool",
                    "tool_call_id": event.tool_call_id,
                    "content": (
                        event.content
                        if isinstance(event.content, str)
                        else json.dumps(event.content, default=str)
                    ),
                }
            )
        elif event.type == "result" and event.content is not None:
            content = str(event.content)
            if not _ends_with_response(messages, content):
                messages.append(
                    {"role": "assistant", "type": "response", "content": content}
                )
    return messages


def _ends_with_response(messages: Sequence[Mapping[str, Any]], content: str) -> bool:
    return bool(
        messages
        and messages[-1].get("role") == "assistant"
        and messages[-1].get("type", "response") == "response"
        and messages[-1].get("content") == content
    )


def _lean_environment(
    package_names: Sequence[str], toolchain_dir: str
) -> dict[str, str]:
    names = package_names or DEFAULT_PACKAGE_ORDER
    lean_paths = [
        f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}/.lake/build/lib/lean"
        for name in names
    ]
    lean_paths.extend(
        (
            f"{CONTAINER_LEAN_CACHE}/.lake/build/lib/lean",
            f"{CONTAINER_ELAN_HOME}/toolchains/{toolchain_dir}/lib/lean",
        )
    )
    native_paths = [
        f"{CONTAINER_ELAN_HOME}/toolchains/{toolchain_dir}/lib/lean",
        f"{CONTAINER_ELAN_HOME}/toolchains/{toolchain_dir}/lib",
        f"{CONTAINER_LEAN_CACHE}/.lake/build/lib",
    ]
    native_paths.extend(
        f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}/.lake/build/lib"
        for name in reversed(names)
    )
    source_paths = [
        *(f"{CONTAINER_LEAN_CACHE}/.lake/packages/{name}" for name in names),
        f"{CONTAINER_ELAN_HOME}/toolchains/{toolchain_dir}/src/lean/lake",
    ]
    return {
        "LEAN_PATH": ":".join(lean_paths),
        "LEAN_SRC_PATH": ":".join(source_paths),
        "LD_LIBRARY_PATH": ":".join(native_paths),
    }


__all__ = ["DockerHarnessSandbox", "PodmanHarnessSandbox", "HarnessSolver"]
