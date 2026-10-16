"""Kimi Code CLI adapter."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from ...agent import Agent, AgentEvent
from ...models.api_header_bridge import APIHeaderProxy
from ...models.oauth.openai_bridge import CodexOAuthResponsesProxy
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation
from ...traces import TokenUsage

_KIMI_CODE_BASE_URL = "https://api.kimi.com/coding/v1"
_KIMI_API_FAILURE = re.compile(
    r"\bllm request failed\s+.*?\bturnStep=(?P<turn_step>\S+)\s+"
    r".*?\berrorName=(?P<error_name>\S+)\s+errorMessage=(?P<message>.*)$"
)
_MINIMAL_AGENT_FILE = """---
name: harness
description: Minimal reproducible benchmark agent
tools:
{tools}
subagents: []
---
Use the available tools to solve the user's task.
"""


def _session_directories(root: Path, session_id: str | None = None) -> tuple[Path, ...]:
    sessions_root = root / ".harness-home" / ".kimi-code" / "sessions"
    if not sessions_root.is_dir():
        return ()
    try:
        directories = tuple(
            session
            for workspace in sessions_root.iterdir()
            if workspace.is_dir()
            for session in workspace.iterdir()
            if session.is_dir() and (session_id is None or session.name == session_id)
        )
    except OSError:
        return ()
    return tuple(sorted(directories))


def _usage_value(usage: Mapping[str, Any], name: str) -> int:
    value = usage.get(name)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int) and value >= 0:
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return 0


def _kimi_usage_record(value: Any) -> TokenUsage | None:
    if not isinstance(value, Mapping) or value.get("type") != "usage.record":
        return None
    usage = value.get("usage")
    if not isinstance(usage, Mapping):
        return None
    cache_read = _usage_value(usage, "inputCacheRead")
    return TokenUsage(
        # Kimi splits prompt tokens into uncached and cache-read categories.
        # MathArena's non-Claude convention expects input_tokens to include both.
        input_tokens=_usage_value(usage, "inputOther") + cache_read,
        output_tokens=_usage_value(usage, "output"),
        cache_read_tokens=cache_read,
        cache_write_tokens=_usage_value(usage, "inputCacheCreation"),
    )


def _add_usage(left: TokenUsage, right: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=left.input_tokens + right.input_tokens,
        output_tokens=left.output_tokens + right.output_tokens,
        cache_read_tokens=left.cache_read_tokens + right.cache_read_tokens,
        cache_write_tokens=left.cache_write_tokens + right.cache_write_tokens,
    )


def _max_usage(*values: TokenUsage) -> TokenUsage:
    return TokenUsage(
        input_tokens=max(value.input_tokens for value in values),
        output_tokens=max(value.output_tokens for value in values),
        cache_read_tokens=max(value.cache_read_tokens for value in values),
        cache_write_tokens=max(value.cache_write_tokens for value in values),
    )


def read_kimi_token_usage(root: str | Path, session_id: str | None = None) -> TokenUsage | None:
    """Read cumulative usage from Kimi Code's native session wires.

    Kimi's public ``stream-json`` output currently omits usage, while its native
    wire records one ``usage.record`` per model turn. Turn-scoped records are
    summed. If a wire contains only session-scoped records, its final cumulative
    record is used instead.
    """

    total = TokenUsage()
    found = False
    for session_dir in _session_directories(Path(root), session_id):
        for wire_path in sorted((session_dir / "agents").glob("*/wire.jsonl")):
            turn_total = TokenUsage()
            found_turn = False
            last_session_usage: TokenUsage | None = None
            try:
                with wire_path.open(encoding="utf-8") as wire:
                    for line in wire:
                        try:
                            value = json.loads(line)
                        except (json.JSONDecodeError, UnicodeDecodeError):
                            # An actively written final line may be incomplete.
                            continue
                        usage = _kimi_usage_record(value)
                        if usage is None:
                            continue
                        if value.get("usageScope") == "turn":
                            turn_total = _add_usage(turn_total, usage)
                            found_turn = True
                        elif value.get("usageScope") == "session":
                            last_session_usage = usage
            except OSError:
                continue
            wire_usage = turn_total if found_turn else last_session_usage
            if wire_usage is not None:
                total = _add_usage(total, wire_usage)
                found = True
    return total if found else None


class KimiCodeAgent(Agent):
    harness_name = "kimi-code"
    aliases = ("kimi", "moonshot")
    # Kimi Code natively supports OpenAI-compatible and Anthropic providers.
    accepted_api_types = ("openai", "anthropic")
    installation = CLIInstallation(
        executable="kimi",
        package="@moonshot-ai/kimi-code",
    )

    def _begin_token_session(self, *, resume: bool, session_id: str | None) -> None:
        previous_session_id = self._token_session_id
        super()._begin_token_session(resume=resume, session_id=session_id)
        if not resume or (session_id is not None and session_id != previous_session_id):
            with self._tokens_lock:
                self._provider_token_usage = TokenUsage()

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        super()._capture_model_response(path, payload)
        usage_payload = payload.get("usage")
        response = payload.get("response")
        if usage_payload is None and isinstance(response, Mapping):
            usage_payload = response.get("usage")
        usage = self._normalize_token_usage(usage_payload)
        if usage is None or usage.total_tokens == 0:
            return
        with self._tokens_lock:
            current = getattr(self, "_provider_token_usage", TokenUsage())
            self._provider_token_usage = _add_usage(current, usage)
        self._record_trace_event(AgentEvent(
            type="usage", content=usage.to_dict(), session_id=self.session_id,
            raw={"source": "model_response", "path": path, "usage": dict(usage_payload)},
        ))

    def _uses_codex_oauth_bridge(self) -> bool:
        return (
            getattr(self.model, "auth_mode", None) == "oauth"
            and getattr(self.model, "provider", None) == "openai"
        )

    def _api_endpoint_with_headers(self) -> Any | None:
        if getattr(self.model, "auth_mode", None) != "api":
            return None
        endpoint_for = getattr(self.model, "endpoint_for", None)
        if not callable(endpoint_for):
            return None
        endpoint = endpoint_for(self._model_protocol() or "openai")
        return endpoint if getattr(endpoint, "headers", None) else None

    def _native_session_log_paths(self) -> tuple[Path, ...]:
        return tuple(
            session_dir / "logs" / "kimi-code.log"
            for session_dir in _session_directories(self.root)
            if (session_dir / "logs" / "kimi-code.log").is_file()
        )

    def _native_log_offsets(self) -> dict[Path, int]:
        offsets: dict[Path, int] = {}
        for path in self._native_session_log_paths():
            try:
                offsets[path] = path.stat().st_size
            except OSError:
                continue
        return offsets

    @staticmethod
    def _decode_log_value(value: str) -> str:
        value = value.strip()
        if value.startswith('"'):
            try:
                decoded = json.loads(value)
            except json.JSONDecodeError:
                pass
            else:
                if isinstance(decoded, str):
                    return decoded
        return value

    def _report_native_api_failures(self, offsets: dict[Path, int]) -> None:
        """Expose Kimi's internally retried provider failures in the run log.

        Deliberately inspect only ``llm request failed`` records. Tool results,
        including failed tool results, remain exclusively in the transcript.
        """

        bound_logger = logger.bind(
            component="model_provider",
            harness=self.harness_name,
            model=self.model_name,
        )
        for path in self._native_session_log_paths():
            offset = offsets.get(path, 0)
            try:
                size = path.stat().st_size
                if size < offset:
                    offset = 0
                with path.open("rb") as native_log:
                    native_log.seek(offset)
                    appended = native_log.read()
            except OSError:
                continue
            complete_end = appended.rfind(b"\n")
            if complete_end < 0:
                continue
            offsets[path] = offset + complete_end + 1
            lines = appended[: complete_end + 1].decode("utf-8", errors="replace").splitlines()
            for line in lines:
                match = _KIMI_API_FAILURE.search(line)
                if match is None:
                    continue
                detail = self._decode_log_value(match.group("message"))
                if len(detail) > 1000:
                    detail = f"{detail[:997]}..."
                bound_logger.warning(
                    "Kimi Code model-provider request failed: error_type={error_type}, "
                    "turn_step={turn_step}, detail={detail}",
                    error_type=match.group("error_name"),
                    turn_step=match.group("turn_step"),
                    detail=detail,
                )

    def _watch_native_api_failures(self, offsets: dict[Path, int], stop: threading.Event) -> None:
        while not stop.wait(0.5):
            self._report_native_api_failures(offsets)

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        offsets = self._native_log_offsets()
        stop_watcher = threading.Event()
        watcher = threading.Thread(
            target=self._watch_native_api_failures,
            args=(offsets, stop_watcher),
            name="kimi-provider-log",
            daemon=True,
        )
        watcher.start()
        try:
            yield from self._stream_once_with_transport(
                message,
                resume=resume,
                session_id=session_id,
                last=last,
            )
        finally:
            stop_watcher.set()
            watcher.join(timeout=2)
            self._report_native_api_failures(offsets)

    def _stream_once_with_transport(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        auth_mode = getattr(self.model, "auth_mode", None)
        endpoint = self._api_endpoint_with_headers()
        if self._uses_codex_oauth_bridge():
            yield from super()._stream_once(
                message,
                resume=resume,
                session_id=session_id,
                last=last,
            )
            return
        override_provider = getattr(self.model, "request_overrides", None)
        drop_provider = getattr(self.model, "request_drop_fields", None)
        if endpoint is not None:
            logger.bind(component="api_header_bridge", harness=self.harness_name).info(
                "Preparing Kimi Code to use an API endpoint with protected custom headers"
            )
            with (
                self._host_service_route() as route,
                APIHeaderProxy(
                    endpoint.url,
                    endpoint.headers,
                    on_request=self._capture_model_request,
                    on_response=self._capture_model_response,
                    request_overrides=override_provider() if callable(override_provider) else {},
                    request_drop_fields=drop_provider() if callable(drop_provider) else (),
                    listen_host=route.listen_host,
                    client_host=route.client_host,
                    allow_remote_clients=route.allow_remote_clients,
                    offline_models=self._offline_models,
                    unix_socket=route.unix_socket,
                    client_port=route.client_port,
                ) as bridge,
            ):
                self._api_header_bridge_url = bridge.base_url
                self._api_header_bridge_api_key = bridge.client_api_key
                try:
                    yield from super()._stream_once(
                        message,
                        resume=resume,
                        session_id=session_id,
                        last=last,
                    )
                finally:
                    del self._api_header_bridge_url
                    del self._api_header_bridge_api_key
            return
        if auth_mode == "api":
            endpoint_for = getattr(self.model, "endpoint_for", None)
            if not callable(endpoint_for):
                yield from super()._stream_once(
                    message, resume=resume, session_id=session_id, last=last
                )
                return
            capture_endpoint = endpoint_for(self._model_protocol() or "openai")
            upstream_url = str(capture_endpoint.url)
            environment_name = "_request_capture_url"
        elif auth_mode == "oauth" and getattr(self.model, "provider", None) == "kimi":
            upstream_url = _KIMI_CODE_BASE_URL
            environment_name = "_native_oauth_capture_url"
        else:
            yield from super()._stream_once(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        override_provider = getattr(self.model, "request_overrides", None)
        drop_provider = getattr(self.model, "request_drop_fields", None)
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                upstream_url,
                self._capture_model_request,
                on_response=self._capture_model_response,
                request_overrides=override_provider() if callable(override_provider) else {},
                request_drop_fields=drop_provider() if callable(drop_provider) else (),
                listen_host=route.listen_host,
                client_host=route.client_host,
                allow_remote_clients=route.allow_remote_clients,
                offline_models=self._offline_models,
                unix_socket=route.unix_socket,
                client_port=route.client_port,
            ) as proxy,
        ):
            setattr(self, environment_name, proxy.base_url)
            try:
                yield from super()._stream_once(
                    message, resume=resume, session_id=session_id, last=last
                )
            finally:
                delattr(self, environment_name)

    def get_tokens(self) -> TokenUsage:
        native_usage = read_kimi_token_usage(self.root, self.session_id)
        stream_usage = super().get_tokens()
        with self._tokens_lock:
            provider_usage = getattr(self, "_provider_token_usage", TokenUsage())
        if native_usage is None:
            return _max_usage(stream_usage, provider_usage)
        return _max_usage(native_usage, stream_usage, provider_usage)

    def _stream(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if not self._uses_codex_oauth_bridge():
            yield from super()._stream(
                message,
                resume=resume,
                session_id=session_id,
                last=last,
            )
            return
        ensure_authenticated = getattr(self.model, "ensure_authenticated", None)
        if callable(ensure_authenticated):
            ensure_authenticated()
        logger.bind(component="oauth_bridge", harness=self.harness_name, provider="openai").info(
            "Preparing Kimi Code to use OpenAI OAuth through the loopback bridge"
        )
        with (
            self._host_service_route() as route,
            CodexOAuthResponsesProxy(
                on_request=self._capture_model_request,
                on_response=self._capture_model_response,
                listen_host=route.listen_host,
                client_host=route.client_host,
                allow_remote_clients=route.allow_remote_clients,
                offline_models=self._offline_models,
                unix_socket=route.unix_socket,
                client_port=route.client_port,
            ) as bridge,
        ):
            self._codex_oauth_bridge = bridge
            self._codex_oauth_bridge_url = bridge.base_url
            self._codex_oauth_bridge_api_key = bridge.client_api_key
            try:
                yield from super()._stream(
                    message,
                    resume=resume,
                    session_id=session_id,
                    last=last,
                )
            finally:
                del self._codex_oauth_bridge
                del self._codex_oauth_bridge_url
                del self._codex_oauth_bridge_api_key

    def _refresh_oauth_transport(self) -> None:
        bridge = getattr(self, "_codex_oauth_bridge", None)
        reload_credentials = getattr(bridge, "reload_credentials", None)
        if callable(reload_credentials):
            logger.bind(
                component="oauth_bridge", harness=self.harness_name, provider="openai"
            ).info("Refreshing Kimi Code OAuth bridge transport")
            reload_credentials()

    def model_environment(self) -> dict[str, str]:
        """Translate the generic model contract to Kimi's explicit env channel.

        Kimi Code intentionally ignores conventional ``OPENAI_*`` and
        ``ANTHROPIC_*`` process variables. ``KIMI_MODEL_*`` creates an
        invocation-local provider without modifying the user's Kimi config.
        Native Kimi OAuth remains CLI-owned; OpenAI OAuth uses an invocation-
        local bridge without persisting credentials in Kimi's configuration.
        """

        base = super().model_environment()
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            base["KIMI_MODEL_THINKING_EFFORT"] = str(reasoning)
        base["KIMI_CODE_NO_AUTO_UPDATE"] = "1"
        if self.subagents:
            # Print mode needs the v2 engine for the Agent/AgentSwarm tools.
            # Kimi currently supports one invocation-local secondary model;
            # subagents without an override continue to use the primary.
            base["KIMI_CODE_EXPERIMENTAL_FLAG"] = "1"
            secondary = None
            if (
                getattr(self.model, "auth_mode", None) != "api"
                and not self._uses_codex_oauth_bridge()
            ):
                secondary = next(
                    (item for item in self.available_subagent_models() if item != self.model_name),
                    None,
                )
            if secondary is not None:
                base["KIMI_SECONDARY_MODEL"] = secondary
        if self._uses_codex_oauth_bridge():
            bridge_url = getattr(self, "_codex_oauth_bridge_url", None)
            if bridge_url is None:
                return base
            base.update(
                {
                    "KIMI_MODEL_NAME": self.model_name,
                    "KIMI_MODEL_PROVIDER_TYPE": "openai_responses",
                    "KIMI_MODEL_BASE_URL": str(bridge_url),
                    # A random invocation-local capability protects the
                    # loopback listener; the real token never enters Kimi.
                    "KIMI_MODEL_API_KEY": str(self._codex_oauth_bridge_api_key),
                    "KIMI_MODEL_MAX_CONTEXT_SIZE": "32768",
                    "KIMI_MODEL_CAPABILITIES": "tool_use",
                }
            )
            return base
        native_oauth_capture_url = getattr(self, "_native_oauth_capture_url", None)
        if native_oauth_capture_url is not None:
            base["KIMI_CODE_BASE_URL"] = str(native_oauth_capture_url)
        if getattr(self.model, "auth_mode", None) != "api":
            return base
        protocol = self._model_protocol() or "openai"
        key_name = "OPENAI_API_KEY" if protocol == "openai" else "ANTHROPIC_API_KEY"
        url_name = "OPENAI_BASE_URL" if protocol == "openai" else "ANTHROPIC_BASE_URL"
        api_key = base.get(key_name) or getattr(self.model, "api_key", None)
        base_url = base.get(url_name) or getattr(self.model, "api_url", None)
        bridge_url = getattr(self, "_api_header_bridge_url", None)
        if bridge_url is not None:
            base.pop(key_name, None)
            base.pop(url_name, None)
            api_key = self._api_header_bridge_api_key
            base_url = bridge_url
        capture_url = getattr(self, "_request_capture_url", None)
        if capture_url is not None:
            base_url = capture_url
        if api_key:
            base["KIMI_MODEL_API_KEY"] = str(api_key)
        if base_url:
            base["KIMI_MODEL_BASE_URL"] = str(base_url)
        base.update(
            {
                "KIMI_MODEL_NAME": self.model_name,
                "KIMI_MODEL_PROVIDER_TYPE": protocol,
                # Kimi otherwise assumes a 256K window and can request an
                # equally large completion from smaller compatible servers.
                # Callers can override this through Agent(environment=...).
                "KIMI_MODEL_MAX_CONTEXT_SIZE": "32768",
                "KIMI_MODEL_CAPABILITIES": "tool_use",
            }
        )
        return base

    def _minimal_context_paths(self) -> tuple[str, str]:
        state_dir = self.root / ".harness_wrapper"
        skills_dir = state_dir / "empty-skills"
        agent_file = state_dir / "minimal-kimi-agent.md"
        skills_dir.mkdir(parents=True, exist_ok=True)
        tools = ["Bash", "WebSearch", "FetchURL"] if self.internet_enabled else ["Bash"]
        content = _MINIMAL_AGENT_FILE.format(tools="\n".join(f"  - {name}" for name in tools))
        if not agent_file.exists() or agent_file.read_text(encoding="utf-8") != content:
            agent_file.write_text(content, encoding="utf-8")
        return (
            Path(".harness_wrapper/minimal-kimi-agent.md").as_posix(),
            Path(".harness_wrapper/empty-skills").as_posix(),
        )

    def build_command(
        self,
        message: str | None,
        *,
        resume: bool = False,
        session_id: str | None = None,
        last: bool = False,
    ) -> list[str]:
        if not resume and (session_id is not None or last):
            raise ValueError("session selection requires resume=True")
        # Kimi's non-interactive --prompt mode implicitly uses its auto
        # permission policy and rejects explicit --yolo/--auto flags.
        command = [self.executable]
        if self.minimal_context:
            agent_file, skills_dir = self._minimal_context_paths()
            command.extend(("--skills-dir", skills_dir))
            if not resume:
                command.extend(("--agent-file", agent_file))
        if resume:
            if session_id is not None:
                command.extend(("--session", session_id))
            else:
                command.append("--continue")
        if message is not None:
            command.extend(("-p", message, "--output-format", "stream-json"))
            # KIMI_MODEL_NAME enables a temporary in-memory provider. Passing
            # -m would override it with a persisted config.toml model alias.
            if (
                self.model_name
                and getattr(self.model, "auth_mode", None) != "api"
                and not self._uses_codex_oauth_bridge()
            ):
                command.extend(("--model", self.model_name))
            command.extend(self.model_cli_args())
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        role = event.get("role")
        native_type = event.get("type")
        if role == "meta" and native_type == "session.resume_hint":
            session_id = event.get("session_id")
            return [
                AgentEvent(
                    type="session",
                    content=event.get("content", event),
                    role="meta",
                    session_id=str(session_id) if session_id else None,
                    raw=event,
                )
            ]
        if role == "meta":
            return [
                AgentEvent(type=str(native_type or "meta"), content=event, role="meta", raw=event)
            ]
        if role == "tool":
            call_id = event.get("tool_call_id")
            return [
                AgentEvent(
                    type="tool_result",
                    content=event.get("content"),
                    role="tool",
                    tool_call_id=str(call_id) if call_id else None,
                    raw=event,
                )
            ]
        if role == "assistant":
            normalized: list[AgentEvent] = []
            reasoning = event.get("reasoning_content", event.get("thinking_content"))
            if reasoning:
                normalized.append(
                    AgentEvent(type="reasoning", content=reasoning, role="assistant", raw=event)
                )
            if event.get("content"):
                normalized.append(
                    AgentEvent(
                        type="message", content=event["content"], role="assistant", raw=event
                    )
                )
            for call in event.get("tool_calls", ()):
                function = call.get("function", {}) if isinstance(call, Mapping) else {}
                normalized.append(
                    AgentEvent(
                        type="tool_call",
                        content=function.get("arguments"),
                        role="assistant",
                        tool_name=function.get("name"),
                        tool_call_id=call.get("id"),
                        raw=event,
                    )
                )
            if event.get("usage") is not None:
                normalized.append(AgentEvent(type="usage", content=event["usage"], raw=event))
            return normalized or [
                AgentEvent(type="message", content="", role="assistant", raw=event)
            ]
        return super().normalize_event(event)


__all__ = ["KimiCodeAgent", "read_kimi_token_usage"]
