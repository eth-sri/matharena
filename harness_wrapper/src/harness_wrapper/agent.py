"""Harness-neutral agent abstraction and factory."""

from __future__ import annotations

import importlib
import os
import re
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from loguru import logger

from .sandbox import HostServiceRoute
from .tools import (
    CLIInstallation,
    CLIProcessError,
    JSONLineParser,
    ModelCompatibilityError,
    SessionCache,
    compact_json,
    find_executable,
    merge_environment,
    validate_cli_version,
)
from .traces import TokenUsage


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True, slots=True)
class AgentEvent:
    """A compact universal view over a harness-specific JSON event."""

    type: str
    content: Any = None
    role: str | None = None
    tool_name: str | None = None
    tool_call_id: str | None = None
    session_id: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        value = {
            "type": self.type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "raw": dict(self.raw),
        }
        for name in ("role", "tool_name", "tool_call_id", "session_id"):
            item = getattr(self, name)
            if item is not None:
                value[name] = item
        return value


ProcessFactory = Callable[..., Any]
_SUBAGENT_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]{0,63}")
_SIGHUP = int(getattr(signal, "SIGHUP", 1))
_SIGHUP_RETURN_CODES = frozenset((-_SIGHUP, 128 + _SIGHUP))
_CAPACITY_FAILURE = re.compile(
    r"(?:\bmodel (?:is )?at capacity\b|"
    r"\bserver(?:[_ -]is)?[_ -]overloaded\b|"
    r"\bmodel (?:is )?(?:temporarily )?overloaded\b)",
    re.IGNORECASE,
)
_SENSITIVE_ERROR_VALUE = re.compile(
    r"(?i)\b(api[_ -]?key|authorization|access[_ -]?token|refresh[_ -]?token|"
    r"password|secret)\b(\s*[:=]\s*)[\"']?[^,\s\"']+"
)
_BEARER_VALUE = re.compile(r"(?i)\bbearer\s+[^,\s\"']+")
# Docker and Podman reserve these for faults they raise themselves, before the
# harness CLI is reached. Their documented meanings are not reliable enough to
# repeat -- Podman answers 127 for a network-setup failure, which has nothing to
# do with a missing command -- so point at the output rather than guess a cause.
_RUNTIME_LAUNCH_STATUS = frozenset((125, 126, 127))


def _exit_status_hint(returncode: int, command: Sequence[str]) -> str:
    """Attribute a launch-time exit status to the runtime that produced it."""

    if returncode not in _RUNTIME_LAUNCH_STATUS or not command:
        return ""
    launcher = Path(str(command[0])).name
    return f" (raised by {launcher} itself, not by the harness CLI; see output below)"


def _safe_provider_error(value: Any) -> dict[str, Any]:
    """Reduce a provider error to useful, credential-safe diagnostic fields."""

    if isinstance(value, Mapping):
        result = {
            name: item
            for name in ("code", "status", "type", "message")
            if (item := value.get(name)) is not None and isinstance(item, bool | int | float | str)
        }
    else:
        result = {"message": str(value)}
    if not result:
        result = {"type": "provider_error"}
    for name, item in tuple(result.items()):
        if not isinstance(item, str):
            continue
        redacted = _SENSITIVE_ERROR_VALUE.sub(r"\1\2[REDACTED]", item)
        redacted = _BEARER_VALUE.sub("Bearer [REDACTED]", redacted)
        result[name] = redacted if len(redacted) <= 2000 else f"{redacted[:1997]}..."
    return result


def _model_request_succeeded(payload: Mapping[str, Any]) -> bool:
    """Recognize provider completions, excluding partial and token-limited output."""

    if payload.get("error") is not None:
        return False
    kind = payload.get("type")
    if kind in ("error", "response.failed", "response.incomplete"):
        return False
    if kind == "response.completed" or payload.get("object") == "response":
        response = payload.get("response", payload)
        default_status = "completed" if kind == "response.completed" else None
        return (
            isinstance(response, Mapping)
            and response.get("error") is None
            and response.get("status", default_status) == "completed"
        )
    choices = payload.get("choices")
    if isinstance(choices, list) and any(
        isinstance(choice, Mapping)
        and choice.get("finish_reason") in ("stop", "tool_calls", "function_call")
        for choice in choices
    ):
        return True
    if kind in ("message", "message_delta"):
        message = payload.get("delta", {}) if kind == "message_delta" else payload
        return isinstance(message, Mapping) and message.get("stop_reason") in (
            "end_turn", "tool_use", "stop_sequence",
        )
    candidates = payload.get("candidates")
    return isinstance(candidates, list) and any(
        isinstance(candidate, Mapping) and candidate.get("finishReason") == "STOP"
        for candidate in candidates
    )


class OutputTokenLimitError(CLIProcessError):
    """A native turn exhausted its generation allowance, not its connection retries."""


class Agent(ABC):
    """Base class and public factory for all supported agent CLIs.

    Calling ``Agent(type="codex", ...)`` returns the registered concrete
    adapter. Concrete classes can also be instantiated directly, which makes
    custom extensions and unit tests straightforward.
    """

    _registry: ClassVar[dict[str, type[Agent]]] = {}
    aliases: ClassVar[tuple[str, ...]] = ()
    harness_name: ClassVar[str]
    prompt_via_stdin: ClassVar[bool] = False
    installation: ClassVar[CLIInstallation]
    accepted_api_types: ClassVar[tuple[str, ...]] = ()

    def __new__(cls, *args: Any, **kwargs: Any) -> Agent:
        if cls is Agent:
            harness_type = kwargs.get("type") or kwargs.get("harness")
            if harness_type is None:
                raise TypeError("Agent requires a harness type")
            cls._load_builtin_adapters()
            key = cls._canonical_key(str(harness_type))
            try:
                concrete = cls._registry[key]
            except KeyError as exc:
                supported = ", ".join(sorted(set(cls._registry)))
                raise ValueError(
                    f"unknown agent type {harness_type!r}; supported: {supported}"
                ) from exc
            return object.__new__(concrete)
        return object.__new__(cls)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "harness_name", None)
        if name:
            for alias in (name, *getattr(cls, "aliases", ())):
                Agent._registry[Agent._canonical_key(alias)] = cls

    @staticmethod
    def _canonical_key(value: str) -> str:
        return value.strip().lower().replace("_", "-").replace(" ", "-")

    @classmethod
    def _load_builtin_adapters(cls) -> None:
        for module in (
            "harness_wrapper.harnesses.claude_code",
            "harness_wrapper.harnesses.codex_cli",
            "harness_wrapper.harnesses.kimi_code",
            "harness_wrapper.harnesses.antigravity_cli",
            "harness_wrapper.harnesses.qwen_code",
            "harness_wrapper.harnesses.opencode",
            "harness_wrapper.harnesses.deepcode",
            "harness_wrapper.harnesses.grok_build",
            "harness_wrapper.harnesses.muse_code",
            "harness_wrapper.harnesses.mistral_vibe",
        ):
            importlib.import_module(module)

    @classmethod
    def create(cls, type: str, **kwargs: Any) -> Agent:
        return cls(type=type, **kwargs)

    @classmethod
    def available(cls) -> tuple[str, ...]:
        cls._load_builtin_adapters()
        return tuple(sorted(cls._registry))

    def __init__(
        self,
        model: Any,
        *,
        env: Any | None = None,
        subagents: Mapping[str, Any] | None = None,
        dir: str | Path | None = None,
        root_dir: str | Path | None = None,
        trace: Any | None = None,
        executable: str | Path | None = None,
        process_factory: ProcessFactory = subprocess.Popen,
        version_runner: Callable[..., Any] = subprocess.run,
        validate_version: bool = False,
        environment: Mapping[str, str] | None = None,
        on_activity: Callable[[datetime], Any] | None = None,
        on_event: Callable[[AgentEvent], Any] | None = None,
        on_model_response: Callable[[str, Mapping[str, Any]], Any] | None = None,
        should_stop: Callable[[], bool] | None = None,
        session_cache: Any | None = None,
        minimal_context: bool = False,
        auto_wait: bool = True,
        auto_fallback: bool = True,
        rate_limit_timeout: float | None = None,
        rate_limit_poll_interval: float = 30.0,
        max_recovery_attempts: int = 10,
        type: str | None = None,
        harness: str | None = None,
        **_: Any,
    ) -> None:
        del type, harness
        if rate_limit_timeout is not None and rate_limit_timeout <= 0:
            raise ValueError("rate_limit_timeout must be positive or None")
        if rate_limit_poll_interval <= 0:
            raise ValueError("rate_limit_poll_interval must be positive")
        if max_recovery_attempts < 0:
            raise ValueError("max_recovery_attempts cannot be negative")
        self.auto_wait = bool(auto_wait)
        self.auto_fallback = bool(auto_fallback)
        self.minimal_context = bool(minimal_context)
        self.rate_limit_timeout = rate_limit_timeout
        self.rate_limit_poll_interval = rate_limit_poll_interval
        self.max_recovery_attempts = max_recovery_attempts
        self._requested_model = model
        self.model = self._select_model(model, allow_fallback=self.auto_fallback)
        self.env = env
        requested_subagents = None if subagents is None else dict(subagents)
        requested_root = root_dir if root_dir is not None else dir
        if self.env is not None and requested_root is None:
            requested_root = getattr(self.env, "root", Path.cwd())
        root_path = Path(requested_root or Path.cwd()).expanduser()
        if self.env is not None and not root_path.is_absolute():
            root_path = Path(getattr(self.env, "root", Path.cwd())) / root_path
        self.root = root_path.resolve()
        if self.env is None and not self.root.is_dir():
            raise NotADirectoryError(f"agent root is not a directory: {self.root}")
        self.executable = find_executable(
            self.installation.executable,
            explicit=executable,
            installation=self.installation,
        )
        self.subagents = (
            self._default_oauth_subagents() if requested_subagents is None else requested_subagents
        )
        if validate_version:
            validate_cli_version(self.executable, self.installation, runner=version_runner)
        self._process_factory = process_factory
        self._environment_override = dict(environment or {})
        self._on_activity = on_activity
        self._on_event = on_event
        self._on_model_response = on_model_response
        self._should_stop = should_stop
        self._process: Any | None = None
        self._process_lock = threading.RLock()
        self._activity_lock = threading.Lock()
        self._tokens_lock = threading.Lock()
        self._token_usage = TokenUsage()
        self._token_session_id: str | None = None
        self._captured_prompt_lock = threading.Lock()
        self._captured_prompts: set[str] = set()
        self._captured_reasoning: set[str] = set()
        self._successful_model_requests = 0  # Guarded by _activity_lock.
        self._last_activity_at: datetime | None = None
        self._session_id: str | None = None
        self._interrupted_process: Any | None = None
        self._turn_cancelled = False
        self.session_cache = session_cache if session_cache is not None else SessionCache(self.root)
        self.trace = trace if trace is not None else self._new_trace()
        self.model_request_trace = self._new_model_request_trace()
        logger.bind(
            component="agent",
            harness=self.harness_name,
            model=self.model_name,
        ).info(
            "Agent initialized: minimal_context={minimal_context}, auto_wait={auto_wait}, "
            "auto_fallback={auto_fallback}, max_recovery_attempts={max_recovery_attempts}",
            minimal_context=self.minimal_context,
            auto_wait=self.auto_wait,
            auto_fallback=self.auto_fallback,
            max_recovery_attempts=self.max_recovery_attempts,
        )

    def _new_trace(self) -> Any | None:
        try:
            from .traces import Trace

            return Trace(root=self.root, harness=self.harness_name, model=self.model_name)
        except (ImportError, TypeError):
            return None

    def _new_model_request_trace(self) -> Any | None:
        """Create a dedicated durable log for final proxied request bodies."""

        try:
            from .traces import Trace

            trace_session_id = getattr(self.trace, "session_id", None)
            metadata = (
                {"agent_trace_session_id": str(trace_session_id)} if trace_session_id else None
            )
            return Trace(
                session_id="model_requests",
                root=self.root,
                storage_dir=self.root / ".harness_wrapper",
                harness=self.harness_name,
                model=self.model_name,
                metadata=metadata,
            )
        except (ImportError, TypeError):
            return None

    @property
    def model_request_log_path(self) -> Path | None:
        """Path containing exact post-override JSON request payloads, if available."""

        path = getattr(self.model_request_trace, "path", None)
        return Path(path) if path is not None else None

    def _select_model(self, model: Any, *, allow_fallback: bool = True) -> Any:
        if not self.accepted_api_types:
            return model
        selected = model
        choose = getattr(model, "model_for", None)
        try:
            if callable(choose) and allow_fallback:
                selected = choose(self.accepted_api_types)
            validate = getattr(selected, "assert_compatible", None)
            if callable(validate):
                validate(self.accepted_api_types)
            else:
                self._validate_model_protocols(selected)
            validate_harness = getattr(selected, "assert_harness_compatible", None)
            if callable(validate_harness):
                validate_harness(self.harness_name)
        except (TypeError, ValueError) as exc:
            raise ModelCompatibilityError(
                f"{self.harness_name} accepts {', '.join(self.accepted_api_types)} APIs: {exc}"
            ) from exc
        return selected

    def _compatible_fallbacks(self) -> tuple[Any, ...]:
        provider = getattr(self._requested_model, "compatible_fallbacks", None)
        if callable(provider):
            return tuple(provider(self.accepted_api_types))
        result: list[Any] = []
        for fallback in getattr(self._requested_model, "fallbacks", ()):
            check = getattr(fallback, "assert_compatible", None)
            if not callable(check):
                continue
            try:
                check(self.accepted_api_types)
            except ValueError:
                continue
            result.append(fallback)
        return tuple(result)

    def _activate_model(self, model: Any) -> None:
        previous = self.model_name
        self.model = self._select_model(model, allow_fallback=False)
        logger.bind(
            component="agent",
            harness=self.harness_name,
            model=self.model_name,
        ).warning(
            "Activated recovery model: previous_model={previous_model}",
            previous_model=previous,
        )

    def _prepare_runtime_model(self) -> None:
        """Re-evaluate transient OAuth state before every model turn."""

        previous = self.model
        selected = self._select_model(
            self._requested_model,
            allow_fallback=self.auto_fallback,
        )
        self.model = selected
        bound_logger = logger.bind(
            component="agent",
            harness=self.harness_name,
            model=self.model_name,
        )
        if selected is not previous:
            bound_logger.warning(
                "Runtime model selection changed: fallback_active={fallback_active}",
                fallback_active=selected is not self._requested_model,
            )
        if selected is not self._requested_model:
            return
        if getattr(selected, "auth_mode", None) != "oauth":
            return
        bound_logger.debug("Preparing OAuth model for a new turn")
        ensure = getattr(selected, "ensure_authenticated", None)
        if callable(ensure):
            ensure()
        # model_for() already polled when a compatible fallback exists. If it
        # kept OAuth active, capacity was available and a second network poll
        # would only add latency.
        if self.auto_fallback and self._compatible_fallbacks():
            return
        if not self.auto_wait:
            return
        poll = getattr(selected, "poll_rate_limits", None)
        wait = getattr(selected, "auto_wait", None)
        if not callable(poll) or not callable(wait):
            return
        snapshot = poll(timeout=self.rate_limit_timeout)
        if snapshot.limited:
            bound_logger.warning("OAuth model has no capacity; entering automatic wait")
            wait(
                timeout=self.rate_limit_timeout,
                poll_interval=self.rate_limit_poll_interval,
            )
            if callable(ensure):
                ensure()

    def _validate_model_protocols(self, model: Any) -> None:
        supported = getattr(model, "supported_endpoints", None)
        if callable(supported):
            protocols = {str(value).lower() for value in supported()}
        else:
            provider = getattr(model, "provider", None) or getattr(model, "api_type", None)
            if provider is None:
                return  # Deliberately allow minimal test/custom model objects.
            protocols = {str(getattr(provider, "value", provider)).lower()}
        if protocols.isdisjoint(self.accepted_api_types):
            raise ValueError(f"model provides {sorted(protocols)}")

    @property
    def model_name(self) -> str:
        if isinstance(self.model, str):
            return self.model
        for name in ("model", "model_id", "name"):
            value = getattr(self.model, name, None)
            if value:
                return str(value)
        return str(self.model)

    def _model_protocol(self) -> str | None:
        supported = getattr(self.model, "supported_endpoints", None)
        values = set(supported()) if callable(supported) else set()
        for accepted in self.accepted_api_types:
            if accepted in {str(getattr(value, "value", value)) for value in values}:
                return accepted
        return self.accepted_api_types[0] if self.accepted_api_types else None

    @property
    def internet_enabled(self) -> bool:
        """Enable web tools only for an explicitly networked environment."""
        return bool(self.env is not None and getattr(self.env, "network_enabled", False))

    def network_instructions(self, offline_prompt: str) -> str:
        if not self.internet_enabled:
            return offline_prompt
        return "Use the available file, shell, and web tools. Internet access is enabled."

    @property
    def _offline_models(self) -> frozenset[str]:
        if self.env is not None and not self.env.network_enabled:
            return frozenset({self.model_name})
        return frozenset()

    @contextmanager
    def _host_service_route(self) -> Iterator[HostServiceRoute]:
        if self.env is None:
            yield HostServiceRoute()
            return
        expose = getattr(self.env, "expose_host_service", None)
        if not callable(expose):
            raise TypeError("sandbox must provide expose_host_service()")
        with expose() as route:
            if not isinstance(route, HostServiceRoute):
                raise TypeError("expose_host_service() must yield HostServiceRoute")
            yield route

    def model_environment(self) -> dict[str, str]:
        provider = getattr(self.model, "cli_environment", None)
        if not callable(provider):
            return self.subagent_environment()
        protocol = self._model_protocol()
        try:
            value = provider(protocol) if protocol is not None else provider()
        except TypeError:
            value = provider()
        result = {str(key): str(item) for key, item in dict(value or {}).items()}
        result.update(self.subagent_environment())
        return result

    def model_cli_args(self) -> list[str]:
        provider = getattr(self.model, "cli_args", None)
        if not callable(provider):
            return []
        try:
            value = provider(self.harness_name)
        except (TypeError, ValueError):
            protocol = self._model_protocol()
            value = provider(protocol) if protocol is not None else provider()
        return [str(item) for item in (value or ())]

    @property
    def session_id(self) -> str | None:
        return self._session_id

    @property
    def last_activity_at(self) -> datetime | None:
        with self._activity_lock:
            return self._last_activity_at

    def seconds_since_activity(self, *, now: datetime | None = None) -> float | None:
        last = self.last_activity_at
        if last is None:
            return None
        current = now or _now()
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        return max(0.0, (current.astimezone(timezone.utc) - last).total_seconds())

    def idle_for(self) -> float | None:
        return self.seconds_since_activity()

    def get_tokens(self) -> TokenUsage:
        """Return cumulative token usage for the current native session.

        A fresh ``run()`` resets the counters, while ``resume()`` for the same
        session continues accumulating them. Providers that do not report a
        cache category leave the corresponding value at zero.
        """

        with self._tokens_lock:
            return self._token_usage

    @staticmethod
    def _usage_count(usage: Mapping[str, Any], *names: str) -> int:
        for name in names:
            value = usage.get(name)
            if isinstance(value, bool):
                continue
            if isinstance(value, int) and value >= 0:
                return value
            if isinstance(value, str) and value.isdecimal():
                return int(value)
        return 0

    @classmethod
    def _normalize_token_usage(cls, value: Any) -> TokenUsage | None:
        if not isinstance(value, Mapping):
            return None
        cache_read = cls._usage_count(
            value,
            "cache_read_tokens",
            "cache_read_input_tokens",
            "cached_input_tokens",
            "cache_read",
            "cached",
        )
        if cache_read == 0:
            for details_name in ("input_tokens_details", "prompt_tokens_details"):
                details = value.get(details_name)
                if isinstance(details, Mapping):
                    cache_read = cls._usage_count(details, "cached_tokens", "cache_read_tokens")
                    if cache_read:
                        break
        input_fields = ("input_tokens", "prompt_tokens", "read_tokens", "tokens_read")
        input_tokens = cls._usage_count(value, *input_fields)
        output_tokens = cls._usage_count(
            value, "output_tokens", "completion_tokens", "written_tokens", "tokens_written"
        )
        if "total_tokens" in value and any(name in value for name in input_fields):
            # Gemini's OpenAI-compatible usage can omit thinking from the output
            # field while including it in total_tokens. Other providers already
            # include reasoning in output; taking the maximum avoids counting it twice.
            output_tokens = max(
                output_tokens, cls._usage_count(value, "total_tokens") - input_tokens
            )
        return TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cls._usage_count(
                value,
                "cache_write_tokens",
                "cache_creation_input_tokens",
                "cache_write",
            ),
        )

    def _begin_token_session(self, *, resume: bool, session_id: str | None) -> None:
        with self._tokens_lock:
            if not resume or (session_id is not None and session_id != self._token_session_id):
                self._token_usage = TokenUsage()
            if session_id is not None or not resume:
                self._token_session_id = session_id

    def _observe_token_usage(self, event: AgentEvent) -> None:
        usage = self._normalize_token_usage(event.content) if event.type == "usage" else None
        with self._tokens_lock:
            if event.session_id is not None:
                if (
                    self._token_session_id is not None
                    and event.session_id != self._token_session_id
                ):
                    self._token_usage = TokenUsage()
                self._token_session_id = event.session_id
            if usage is not None:
                current = self._token_usage
                self._token_usage = TokenUsage(
                    input_tokens=current.input_tokens + usage.input_tokens,
                    output_tokens=current.output_tokens + usage.output_tokens,
                    cache_read_tokens=current.cache_read_tokens + usage.cache_read_tokens,
                    cache_write_tokens=current.cache_write_tokens + usage.cache_write_tokens,
                )

    def _touch(self) -> None:
        timestamp = _now()
        with self._activity_lock:
            self._last_activity_at = timestamp
        if self._on_activity is not None:
            self._on_activity(timestamp)

    @abstractmethod
    def build_command(
        self,
        message: str | None,
        *,
        resume: bool = False,
        session_id: str | None = None,
        last: bool = False,
    ) -> list[str]:
        """Build the native CLI invocation."""

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        event_type = str(event.get("type", event.get("role", "event")))
        session_id = event.get("session_id") or event.get("thread_id")
        content = event.get("content", event.get("text", event))
        return [
            AgentEvent(
                type=event_type,
                content=content,
                role=event.get("role"),
                session_id=str(session_id) if session_id else None,
                raw=event,
            )
        ]

    def stream(self, message: str) -> Iterator[AgentEvent]:
        return self._stream(message, resume=False)

    def stream_resume(
        self,
        message: str | None = None,
        *,
        session_id: str | None = None,
        last: bool | None = None,
    ) -> Iterator[AgentEvent]:
        if session_id is not None and last:
            raise ValueError("session_id and last=True are mutually exclusive")
        resume_last = session_id is None if last is None else last
        if session_id is None and resume_last:
            cached = self.session_cache.last(self.harness_name)
            if cached is not None:
                session_id = cached
                resume_last = False
        if message is None:
            return self._resume_selection(session_id=session_id, last=resume_last)
        return self._stream(
            message,
            resume=True,
            session_id=session_id,
            last=resume_last,
        )

    def _resume_selection(self, *, session_id: str | None, last: bool) -> Iterator[AgentEvent]:
        """Select a resumable session without starting an input-less CLI turn."""

        self._begin_token_session(resume=True, session_id=session_id)
        selection = "last" if last else "id"
        raw: dict[str, Any] = {"type": "session.selection", "selection": selection}
        if session_id is not None:
            raw["session_id"] = session_id
        event = AgentEvent(
            type="session",
            content=raw,
            role="meta",
            session_id=session_id,
            raw=raw,
        )
        self._observe(event)
        yield event

    def run(self, message: str) -> list[AgentEvent]:
        return list(self.stream(message))

    def resume(
        self,
        message: str | None = None,
        *,
        session_id: str | None = None,
        last: bool | None = None,
    ) -> list[AgentEvent]:
        return list(self.stream_resume(message, session_id=session_id, last=last))

    query = run

    def _stream(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        try:
            yield from self._stream_with_recovery(
                message, resume=resume, session_id=session_id, last=last
            )
        except OutputTokenLimitError:
            # The shared recovery loop has finished. Keep this spent attempt,
            # including usage and partial work, instead of leaving it pending.
            result = AgentEvent(
                type="result", content="", session_id=self.session_id,
                raw={"reason": "max_output_tokens"},
            )
            self._observe(result)
            yield result

    def _stream_with_recovery(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        self._turn_cancelled = False
        self._session_id = session_id
        self._begin_token_session(resume=resume, session_id=session_id)
        attempts = 0
        original_message = message
        if original_message is not None:
            self._record_trace_event(
                AgentEvent(
                    type="message",
                    content=original_message,
                    role="user",
                    raw={"source": "harness_wrapper", "kind": "original_message"},
                )
            )
        retry_resume = resume
        retry_session_id = session_id
        retry_last = last
        tried_models: set[int] = {id(self.model)}
        runtime_prepared = False
        bound_logger = logger.bind(component="agent", harness=self.harness_name)
        while True:
            if self._turn_cancelled or (self._should_stop is not None and self._should_stop()):
                return
            with self._activity_lock:
                successful_requests_before_turn = self._successful_model_requests
            turn_emitted_event = False
            try:
                if not runtime_prepared:
                    self._prepare_runtime_model()
                    runtime_prepared = True
                    tried_models.add(id(self.model))
                for event in self._stream_once(
                    message,
                    resume=retry_resume,
                    session_id=retry_session_id,
                    last=retry_last,
                ):
                    turn_emitted_event |= event.session_id is not None or event.type in {
                        "session", "message", "reasoning", "tool_call", "tool_result", "result"
                    }
                    yield event
                return
            except Exception as exception:
                if self._turn_cancelled or (self._should_stop is not None and self._should_stop()):
                    return
                with self._activity_lock:
                    made_progress = self._successful_model_requests > successful_requests_before_turn
                if made_progress:
                    # A resumed CLI can make many successful requests before
                    # failing again. Only fresh provider completions reset the
                    # allowance; replayed native messages/usage never do.
                    if attempts:
                        bound_logger.info(
                            "Resetting harness recovery attempts after a successful model request"
                        )
                    attempts = 0
                error = (
                    exception if isinstance(exception, CLIProcessError) else CLIProcessError(
                        [self.executable], 1, f"{type(exception).__name__}: {exception}"
                    )
                )
                bound_logger.warning(
                    "Harness CLI turn failed: model={model}, returncode={returncode}, "
                    "recovery_attempt={recovery_attempt}\n"
                    "  cause: {cause}",
                    model=self.model_name,
                    returncode=error.returncode,
                    recovery_attempt=attempts,
                    cause=(error.stderr or "").strip() or "<no output on stderr or stdout>",
                )
                # Backend overload is transient for both API and OAuth models.
                # Do not treat it as account quota exhaustion or change models.
                capacity_failure = bool(_CAPACITY_FAILURE.search(error.stderr or str(error)))
                if attempts >= self.max_recovery_attempts:
                    bound_logger.error("Harness CLI recovery attempts exhausted")
                    yield self._terminal_error_event(error)
                    raise
                # Classification selects an optional recovery action, never
                # decides whether an error is eligible for a bounded retry.
                recovery = "capacity_retry" if capacity_failure else None
                if recovery is None:
                    try:
                        recovery = self._recover_cli_failure(error, tried_models=tried_models)
                    except Exception as recovery_error:
                        bound_logger.warning("Recovery action failed: {}", recovery_error)
                recovery = recovery or "error_retry"
                attempts += 1
                retry_session_id = self._session_id or retry_session_id
                if retry_session_id is not None:
                    retry_resume = True
                    retry_last = False
                elif turn_emitted_event:
                    # Some CLIs persist a native session without including its
                    # ID in non-interactive output. If the failed turn already
                    # produced events, resume that working directory's latest
                    # session instead of replaying the original prompt in a
                    # fresh context.
                    retry_resume = True
                    retry_last = True
                retry_delay = 60.0
                bound_logger.warning(
                    "Retrying failed harness turn in {delay:g}s "
                    "(attempt {attempt}/{limit})",
                    delay=retry_delay,
                    attempt=attempts,
                    limit=self.max_recovery_attempts,
                )
                bound_logger.info(
                    "Retrying harness CLI turn after recovery: action={action}, "
                    "attempt={attempt}, model={model}, resume={resume}",
                    action=recovery,
                    attempt=attempts,
                    model=self.model_name,
                    resume=self._session_id is not None or retry_resume,
                )
                recovery_event = AgentEvent(
                    type="recovery",
                    content={
                        "reason": recovery,
                        "attempt": attempts,
                        "model": self.model_name,
                        "retry_delay_seconds": retry_delay,
                    },
                    role="meta",
                    session_id=self._session_id,
                    raw={"stderr": error.stderr, "returncode": error.returncode},
                )
                self._observe(recovery_event)
                yield recovery_event
                time.sleep(retry_delay)
                message = (
                    self._recovery_message(recovery)
                    if retry_resume
                    else original_message
                )

    def _recovery_message(self, recovery: str) -> str:
        return "Continue the interrupted task from where it stopped."

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        stdin_message = message if self.prompt_via_stdin else None
        command_message = "-" if stdin_message is not None else message
        command = self.build_command(
            command_message, resume=resume, session_id=session_id, last=last
        )
        explicit_env = self.model_environment()
        explicit_env.update(self._environment_override)
        host_cwd: Path | None = self.root
        if self.env is not None:
            prepare = getattr(self.env, "prepare_command", None)
            if not callable(prepare):
                raise TypeError("sandbox must provide prepare_command(command, cwd, env)")
            command, host_cwd, process_env = prepare(command, cwd=self.root, env=explicit_env)
        else:
            process_env = merge_environment(explicit_env)

        bound_logger = logger.bind(
            component="agent",
            harness=self.harness_name,
            model=self.model_name,
            session_id=session_id,
        )
        bound_logger.info(
            "Starting harness CLI turn: resume={resume}, resume_last={resume_last}",
            resume=resume,
            resume_last=last,
        )
        with self._process_lock:
            if self._turn_cancelled or (self._should_stop is not None and self._should_stop()):
                return
            if self._process is not None and self._process.poll() is None:
                raise RuntimeError("this Agent already has a running process")
            process_options = {
                "cwd": host_cwd,
                "env": process_env,
                "stdin": subprocess.DEVNULL,
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "bufsize": 1,
            }
            if os.name == "posix":
                # Keep Docker and its container independent of the launch terminal.
                process_options["start_new_session"] = True
            # A file-backed stdin avoids both OS argument limits and pipe
            # deadlocks while the CLI is starting. Popen duplicates the descriptor.
            with (
                tempfile.TemporaryFile() if stdin_message is not None else nullcontext()
            ) as prompt_input:
                if prompt_input is not None and stdin_message is not None:
                    prompt_input.write(stdin_message.encode("utf-8"))
                    prompt_input.seek(0)
                    process_options["stdin"] = prompt_input
                process = self._process_factory(command, **process_options)
            self._process = process
            self._interrupted_process = None

        stderr_lines: list[str] = []

        def drain_stderr() -> None:
            stream = getattr(process, "stderr", None)
            if stream is None:
                return
            for line in stream:
                self._touch()
                stderr_lines.append(line)

        stderr_thread = threading.Thread(target=drain_stderr, name="harness-stderr", daemon=True)
        stderr_thread.start()
        parser = JSONLineParser()
        pending_errors: list[AgentEvent] = []
        reaped = False
        try:
            stdout = getattr(process, "stdout", None)
            if stdout is None:
                raise RuntimeError("process factory did not provide stdout")
            for line in stdout:
                self._touch()
                for raw in parser.feed(line):
                    for event in self.normalize_event(raw):
                        if event.type == "error":
                            pending_errors.append(event)
                            continue
                        if event.type == "result" and event.content:
                            pending_errors.clear()
                        self._observe(event)
                        yield event
            for raw in parser.finish():
                for event in self.normalize_event(raw):
                    if event.type == "error":
                        pending_errors.append(event)
                        continue
                    if event.type == "result" and event.content:
                        pending_errors.clear()
                    self._observe(event)
                    yield event
            returncode = process.wait()
            reaped = True
            stderr_thread.join(timeout=2)
            if returncode and self._interrupted_process is not process:
                native_errors = "\n".join(str(event.content) for event in pending_errors)
                detail = "\n".join(
                    item for item in ("".join(stderr_lines).strip(), native_errors) if item
                )
                # The exit status alone rarely identifies the fault: a sandboxed
                # harness fails long before the CLI runs when the container
                # runtime cannot start, and that reason only exists on stderr.
                bound_logger.warning(
                    "Harness CLI process exited unsuccessfully: returncode={returncode}{hint}\n"
                    "  command: {command}\n"
                    "  output: {detail}",
                    returncode=returncode,
                    hint=_exit_status_hint(returncode, command),
                    command=shlex.join(str(part) for part in command),
                    detail=detail or "<no output on stderr or stdout>",
                )
                raise CLIProcessError(command, returncode, detail)
            if pending_errors and self._interrupted_process is not process:
                detail = "\n".join(str(event.content) for event in pending_errors)
                raise CLIProcessError(command, returncode or 1, detail)
            bound_logger.info(
                "Harness CLI turn completed: returncode={returncode}",
                returncode=returncode,
            )
        finally:
            if not reaped and process.poll() is None:
                bound_logger.warning("Terminating unfinished harness CLI process")
                process.terminate()
                try:
                    process.wait(timeout=2)
                except TypeError:  # Minimal/injected process implementations.
                    process.wait()
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)
            stderr_thread.join(timeout=2)
            with self._process_lock:
                if self._process is process:
                    self._process = None
                if self._interrupted_process is process:
                    self._interrupted_process = None

    def _terminal_error_event(self, error: CLIProcessError) -> AgentEvent:
        event = AgentEvent(
            type="error",
            content=error.stderr.strip() or f"CLI exited with status {error.returncode}",
            raw={"returncode": error.returncode, "stderr": error.stderr},
        )
        self._observe(event)
        return event

    def _recover_cli_failure(
        self,
        error: CLIProcessError,
        *,
        tried_models: set[int],
    ) -> str | None:
        # A direct subprocess reports -SIGHUP, while `docker run` commonly
        # reports the shell-style 128 + SIGHUP status after proxying it. This is
        # an infrastructure interruption, not a model/API failure, and can be
        # resumed within the same recovery budget for both API and OAuth models.
        if error.returncode in _SIGHUP_RETURN_CODES:
            return "process_interrupted"

        classifier = getattr(self._requested_model, "classify_cli_failure", None)
        if not callable(classifier):
            return None
        failure = classifier(error.stderr or str(error))
        if failure is None:
            return None
        bound_logger = logger.bind(
            component="agent",
            harness=self.harness_name,
            model=self.model_name,
            failure=failure,
        )
        bound_logger.warning("Classified harness CLI failure")

        requested_is_oauth = getattr(self._requested_model, "auth_mode", None) == "oauth"
        active_is_oauth = getattr(self.model, "auth_mode", None) == "oauth"
        if failure == "authentication" and requested_is_oauth and active_is_oauth:
            relogin = getattr(self._requested_model, "force_relogin", None)
            if not callable(relogin):
                return None
            try:
                bound_logger.info("Running interactive OAuth re-login recovery")
                relogin()
                refresh_transport = getattr(self, "_refresh_oauth_transport", None)
                if callable(refresh_transport):
                    refresh_transport()
                return "authentication_relogin"
            except Exception as exc:
                bound_logger.warning(
                    "OAuth re-login recovery failed: error_type={error_type}",
                    error_type=type(exc).__name__,
                )
                if not self.auto_fallback:
                    raise

        if failure == "rate_limit" and requested_is_oauth and self.auto_wait:
            self._activate_model(self._requested_model)
            wait = getattr(self._requested_model, "auto_wait", None)
            if not callable(wait):
                return None
            bound_logger.warning("Waiting for OAuth capacity before resuming the task")
            wait(
                timeout=self.rate_limit_timeout,
                poll_interval=self.rate_limit_poll_interval,
            )
            ensure = getattr(self._requested_model, "ensure_authenticated", None)
            if callable(ensure):
                ensure()
            return "rate_limit_wait"

        if self.auto_fallback:
            for fallback in self._compatible_fallbacks():
                if id(fallback) in tried_models:
                    continue
                bound_logger.warning(
                    "Switching from failed model to API fallback: fallback_model={fallback_model}",
                    fallback_model=getattr(fallback, "model", type(fallback).__name__),
                )
                self._activate_model(fallback)
                tried_models.add(id(fallback))
                return f"{failure}_fallback"
        return None

    def _observe(self, event: AgentEvent) -> None:
        self._touch()
        self._observe_token_usage(event)
        if self._on_event is not None:
            self._on_event(event)
        if event.session_id:
            self._session_id = event.session_id
            self.session_cache.record(
                self.harness_name,
                event.session_id,
                event.timestamp.isoformat(),
            )
        self._record_trace_event(event)
        if event.type != "reasoning_encrypted":
            self._capture_reasoning_payload(
                event.raw,
                source="native_event",
                include_plaintext=False,
            )

    def _record_trace_event(self, event: AgentEvent) -> None:
        """Persist an event without adding it to the public output stream."""

        if self.trace is None:
            return
        append = getattr(self.trace, "append", None)
        if callable(append):
            metadata = {"harness": self.harness_name, "raw": dict(event.raw)}
            if event.session_id:
                metadata["session_id"] = event.session_id
            append(
                event.type,
                event.content,
                role=event.role,
                tool_name=event.tool_name,
                tool_call_id=event.tool_call_id,
                metadata=metadata,
            )
            return
        record = getattr(self.trace, "record_event", None)
        if callable(record):
            record(event.to_dict())

    def _capture_model_request(self, path: str, payload: Mapping[str, Any]) -> bool | None:
        """Reject stopped turns; otherwise persist and inspect the final request body."""

        if self._turn_cancelled or (self._should_stop is not None and self._should_stop()):
            return False
        self._record_model_request(path, payload)

        prompts: list[tuple[str, Any, str]] = []
        for field_name in ("instructions", "system"):
            content = payload.get(field_name)
            if content is not None:
                prompts.append(("system", content, field_name))
        for field_name in ("input", "messages"):
            messages = payload.get(field_name)
            if not isinstance(messages, list):
                continue
            for item in messages:
                if not isinstance(item, Mapping):
                    continue
                role = item.get("role")
                if role not in {"system", "developer"}:
                    continue
                prompts.append((str(role), item.get("content", item), field_name))

        for role, content, field_name in prompts:
            key = compact_json({"role": role, "content": content})
            with self._captured_prompt_lock:
                if key in self._captured_prompts:
                    continue
                self._captured_prompts.add(key)
            self._record_trace_event(
                AgentEvent(
                    type="message",
                    content=content,
                    role=role,
                    raw={
                        "source": "model_request",
                        "field": field_name,
                        "path": path,
                    },
                )
            )
        self._capture_reasoning_payload(
            payload,
            source="model_request",
            path=path,
            include_plaintext=True,
        )

    def _record_model_request(self, path: str, payload: Mapping[str, Any]) -> None:
        """Append one exact post-override JSON body without recording HTTP headers."""

        append = getattr(self.model_request_trace, "append", None)
        if not callable(append):
            return
        append(
            "model.request",
            dict(payload),
            metadata={
                "path": path,
                "source": "model_proxy",
                "http_headers_included": False,
            },
        )

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        """Record plaintext or recoverable reasoning returned by a provider."""

        if _model_request_succeeded(payload):
            with self._activity_lock:
                self._successful_model_requests += 1
        if self._on_model_response is not None:
            self._on_model_response(path, payload)
        provider_error = payload.get("error")
        if provider_error is not None:
            safe_error = _safe_provider_error(provider_error)
            self._record_trace_event(
                AgentEvent(
                    type="error",
                    content=safe_error,
                    role="meta",
                    raw={"source": "model_response", "path": path},
                )
            )
            logger.bind(
                component="model_provider",
                harness=self.harness_name,
                model=self.model_name,
            ).error(
                "Model-provider request failed: path={path}, error={error}",
                path=path,
                error=compact_json(safe_error),
            )

        self._capture_reasoning_payload(
            payload,
            source="model_response",
            path=path,
            include_plaintext=True,
        )

    def _capture_reasoning_payload(
        self,
        payload: Any,
        *,
        source: str,
        path: str | None = None,
        include_plaintext: bool,
    ) -> None:
        """Extract provider reasoning blocks without changing their replay payload."""

        def record(event_type: str, content: Any, field: str) -> None:
            key = compact_json({"type": event_type, "content": content})
            with self._captured_prompt_lock:
                if key in self._captured_reasoning:
                    return
                self._captured_reasoning.add(key)
            raw: dict[str, Any] = {"source": source, "field": field}
            if path is not None:
                raw["path"] = path
            self._record_trace_event(
                AgentEvent(type=event_type, content=content, role="assistant", raw=raw)
            )

        def record_text(content: Any, field: str) -> None:
            if include_plaintext and isinstance(content, str) and content:
                record("reasoning", content, field)

        def walk(value: Any, field: str) -> None:
            if isinstance(value, list):
                for index, item in enumerate(value):
                    walk(item, f"{field}[{index}]")
                return
            if not isinstance(value, Mapping):
                return

            block_type = str(value.get("type", ""))
            encrypted_fields = (
                "encrypted_content",
                "thought_signature",
                "signature",
            )
            has_recoverable_state = any(
                isinstance(value.get(name), str) and value.get(name) for name in encrypted_fields
            ) or (
                block_type == "redacted_thinking"
                and isinstance(value.get("data"), str)
                and bool(value.get("data"))
            )
            if has_recoverable_state and block_type in {
                "compaction",
                "reasoning",
                "redacted_thinking",
                "signature_delta",
                "thinking",
            }:
                # Preserve the complete block: provider replay can require its
                # id/type and plaintext alongside the opaque state.
                record("reasoning_encrypted", dict(value), field)

            if block_type == "thinking":
                record_text(value.get("thinking"), f"{field}.thinking")
            elif block_type in {
                "reasoning_text",
                "response.reasoning_summary_text.done",
                "response.reasoning_text.done",
                "summary_text",
            }:
                record_text(value.get("text"), f"{field}.text")

            for name in ("reasoning_content", "thinking_content"):
                record_text(value.get(name), f"{field}.{name}")

            for name, item in value.items():
                if isinstance(item, Mapping | list):
                    walk(item, f"{field}.{name}")

        walk(payload, "payload")

    def interrupt(self) -> bool:
        """Interrupt the active turn, returning whether a signal was sent."""

        with self._process_lock:
            self._turn_cancelled = True
            process = self._process
            if process is None or process.poll() is not None:
                return False
            self._interrupted_process = process
            logger.bind(
                component="agent", harness=self.harness_name, model=self.model_name
            ).warning("Interrupting active harness CLI turn")
            process.send_signal(signal.SIGINT)
            return True

    def terminate(self) -> bool:
        with self._process_lock:
            self._turn_cancelled = True
            process = self._process
            if process is None or process.poll() is not None:
                return False
            logger.bind(
                component="agent", harness=self.harness_name, model=self.model_name
            ).warning("Terminating active harness CLI turn")
            self._interrupted_process = process
            process.terminate()
            return True

    def __enter__(self) -> Agent:
        return self

    def __exit__(self, *_: Any) -> None:
        self.terminate()

    def subagent_payload(self) -> dict[str, dict[str, str]]:
        """Return the portable subagent description adapters can inject."""

        payload: dict[str, dict[str, str]] = {}
        for name, model in self.subagents.items():
            if not _SUBAGENT_NAME.fullmatch(name):
                raise ValueError(
                    f"invalid subagent name {name!r}; use 1-64 letters, digits, '_' or '-'"
                )
            model_name = model if isinstance(model, str) else getattr(model, "model", str(model))
            payload[name] = {
                "description": (
                    f"Delegate focused work to {name}; this subagent runs model {model_name}."
                ),
                "prompt": f"You are the {name} subagent. Complete the delegated task.",
                "model": str(model_name),
            }
        return payload

    def available_subagent_models(self) -> tuple[str, ...]:
        """Models currently advertised to this harness for delegation."""

        return tuple(
            dict.fromkeys(
                str(model if isinstance(model, str) else getattr(model, "model", model))
                for model in self.subagents.values()
            )
        )

    def _default_oauth_subagents(self) -> dict[str, str]:
        if getattr(self.model, "auth_mode", None) != "oauth":
            return {}
        discover = getattr(self.model, "available_models", None)
        try:
            models = tuple(discover()) if callable(discover) else (self.model_name,)
        except (OSError, RuntimeError, TypeError, ValueError):
            # Model discovery is an enhancement, not a reason to prevent the
            # authenticated main model from running on older provider CLIs.
            models = (self.model_name,)

        result: dict[str, str] = {}
        for model_name in dict.fromkeys(str(item) for item in models if item):
            stem = re.sub(r"[^A-Za-z0-9_-]+", "-", model_name).strip("-_").lower()
            base = f"model-{stem or 'oauth'}"[:64].rstrip("-_")
            name = base
            suffix = 2
            while name in result:
                marker = f"-{suffix}"
                name = f"{base[: 64 - len(marker)].rstrip('-_')}{marker}"
                suffix += 1
            result[name] = model_name
        return result

    def subagent_environment(self) -> dict[str, str]:
        if not self.subagents:
            return {}
        return {"HARNESS_WRAPPER_SUBAGENTS": compact_json(self.subagent_payload())}
