"""DeepSeek Harness (dsh) headless adapter, with proxied Messages requests."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation
from ...traces import TokenUsage

# dsh's own defaults (llm-deepseek maxTokens and the official context window).
_DEFAULT_MAX_TOKENS = 256_000
_DEFAULT_CONTEXT_WINDOW = 1_048_576
# Rows that send data to DeepSeek or make model calls unrelated to the task:
# the per-request session-log and plugin-inventory fields, OTel uploads, and
# the first-prompt session title request.
_DISABLED_ROWS = (
    "session-log-deepseek",
    "plugin-package-inventory-deepseek",
    "session-telemetry-otel",
    "session-title-llm",
    # Subagents, workflow orchestration, skills, MCP, goal continuation rounds,
    # plan mode, and slash commands, which the other harnesses also disable.
    "subagent",
    "subagent-spawn-in-process",
    "subagent-fork-in-process",
    "tool-subagent-control",
    "tool-subagent-list-agents",
    "tool-subagent",
    "tool-subagent-fork",
    "workflow-ptc",
    "tool-workflow",
    "skill",
    "skill-filesystem",
    "tool-skill",
    "mcp-resources",
    "goal",
    "goal-round-driver",
    "command-goal",
    "tool-goal",
    "plan-mode",
    "commands",
    "command-feedback",
    "command-compact",
)
_LOCAL_TOOLS = frozenset(
    {
        "bash",
        "read",
        "read_image",
        "write",
        "edit",
        "glob",
        "grep",
        "working_directory",
        "job_list",
        "job_output",
        "job_kill",
        "todo_write",
    }
)
_WEB_TOOLS = frozenset({"web_search", "web_fetch"})
# An upstream request can hang while OpenRouter keeps sending SSE keepalive
# comments, which dsh and the proxy timeout both accept indefinitely. Closing
# such a stream makes dsh fail the turn, so the wrapper resumes the session.
_STREAM_STALL_SECONDS = 600
_MINIMAL_INSTRUCTIONS = "Use the available local file and shell tools to solve the user's task."


_USAGE_COUNTERS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


def _drop_null_usage(line: bytes) -> bytes:
    """Remove null Messages usage counters, which dsh rejects as malformed.

    OpenRouter streams e.g. ``"cache_read_input_tokens": null`` where Anthropic
    omits the field; dsh accepts only an absent field or a non-negative integer.
    """

    if not line.startswith(b"data:") or b"null" not in line:
        return line
    try:
        event = json.loads(line[5:])
    except ValueError:
        return line
    if not isinstance(event, dict):
        return line
    holder = event.get("message") if event.get("type") == "message_start" else event
    usage = holder.get("usage") if isinstance(holder, dict) else None
    if not isinstance(usage, dict) or not any(
        name in usage and usage[name] is None for name in _USAGE_COUNTERS
    ):
        return line
    for name in _USAGE_COUNTERS:
        if name in usage and usage[name] is None:
            del usage[name]
    ending = line[len(line.rstrip(b"\r\n")) :]
    return b"data: " + json.dumps(event, separators=(",", ":")).encode() + ending


class DeepSeekHarnessAgent(Agent):
    harness_name = "deepseek-harness"
    aliases = ("dsh",)
    accepted_api_types = ("anthropic",)
    prompt_via_stdin = True
    installation = CLIInstallation(executable="dsh", package="@deepseek-ai/dsh")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._message_usage: dict[str, TokenUsage] = {}
        self._response_context = threading.local()

    @property
    def _private_home(self) -> Path:
        return self.root / ".harness-home"

    def _limit(self, name: str, default: int) -> int:
        value = self._environment_override.get(name)
        return default if value is None else int(value)

    def _patch(self) -> list[dict[str, Any]]:
        """Point dsh's DeepSeek route at the capture proxy; keep other rows at defaults."""

        llm: dict[str, Any] = {
            "baseURL": getattr(self, "_request_capture_url", None)
            or self.model.endpoint_for("anthropic").url,
            "apiKeyEnv": "DEEPSEEK_API_KEY",
            "maxTokens": self._limit("DSH_MAX_OUTPUT_TOKENS", _DEFAULT_MAX_TOKENS),
            # An unlisted id would get dsh's 1,000,000 fallback window. Omit the
            # official entry's in-history/addition-only update modes, which
            # rely on DeepSeek-specific Messages extensions.
            "models": [
                {
                    "id": self.model_name,
                    "contextWindow": self._limit(
                        "DSH_CONTEXT_WINDOW", _DEFAULT_CONTEXT_WINDOW
                    ),
                }
            ],
        }
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            llm["reasoningEffort"] = "off" if reasoning == "none" else str(reasoning)
        patch: list[dict[str, Any]] = [
            {"id": "llm-deepseek", "config": llm},
            {
                "id": "system-prompt",
                "config": {
                    "includeHarnessIdentity": False,
                    "personaPrefix": self.network_instructions(_MINIMAL_INSTRUCTIONS),
                },
            },
            {
                "id": "agent-default-model",
                "config": {"provider": "deepseek-official", "model": self.model_name},
            },
        ]
        patch.extend({"id": row, "disabled": True} for row in _DISABLED_ROWS)
        if not self.internet_enabled:
            patch.append({"id": "tool-web", "disabled": True})
        return patch

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        endpoint = self.model.endpoint_for("anthropic")
        # JSON is a YAML subset, so the patch-list overlay needs no YAML writer.
        patch_path = self._private_home / "dsh-patch.yml"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        patch_path.write_text(json.dumps(self._patch(), indent=2), encoding="utf-8")
        runtime_root = getattr(self.env, "container_root", self.root)
        environment.update(
            {
                "HOME": str(runtime_root / ".harness-home"),
                "DSH_HOME": str(runtime_root / ".harness-home/.dsh"),
                "DEEPSEEK_API_KEY": endpoint.api_key,
                "DSH_TELEMETRY_DISABLED": "1",
                # The container is the sandbox; this preset also sets approval to never.
                "DSH_PERMISSION_MODE": "danger-full-access",
            }
        )
        return environment

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
        if message is None and not resume:
            raise ValueError("dsh requires a prompt")
        runtime_root = getattr(self.env, "container_root", self.root)
        command = [
            self.executable,
            "--profile",
            "headless",
            "--patch",
            str(runtime_root / ".harness-home/dsh-patch.yml"),
            "--json",
        ]
        if resume:
            selected = session_id or self.session_id
            if selected is None:
                raise ValueError("dsh can only resume a known session id")
            command.extend(("--session-id", selected))
        command.extend(("--", message or "Continue the interrupted task from where it stopped."))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        native_type = event.get("type")
        session_id = event.get("sessionId") or self.session_id

        def make(event_type: str, content: Any = None, **fields: Any) -> AgentEvent:
            return AgentEvent(
                type=event_type,
                content=content,
                session_id=str(session_id) if session_id else None,
                raw=event,
                **fields,
            )

        if native_type == "session":
            return [make("session", event)]
        if native_type == "thinking":
            return [make("reasoning", event.get("text", ""), role="assistant")]
        if native_type == "text":
            return [make("message", event.get("text", ""), role="assistant")]
        if native_type == "tool_call":
            return [
                make(
                    "tool_call",
                    event.get("input"),
                    role="assistant",
                    tool_name=event.get("tool"),
                    tool_call_id=event.get("callId"),
                )
            ]
        if native_type == "tool_result":
            return [
                make(
                    "tool_result",
                    event.get("result"),
                    role="tool",
                    tool_call_id=event.get("callId"),
                )
            ]
        if native_type == "final":
            # dsh 0.2.1 always reports `reason: null` here and signals failure
            # only through exit status 1, which the wrapper already treats as
            # a failed turn; later versions carry the turn's terminal reason.
            reason = event.get("reason")
            if isinstance(reason, Mapping) and reason.get("kind") != "completed":
                return [make("error", reason)]
            return [make("result", event.get("text", ""), role="assistant")]
        if native_type == "error":
            return [make("error", event.get("message"))]
        return [make(str(native_type or "event"), event)]

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if not path.rstrip("/").endswith("/v1/messages"):
            raise ValueError("dsh may only access Messages through this proxy")
        # dsh sends DeepSeek's budget-free `thinking: enabled` with
        # output_config.effort; OpenRouter's Messages schema requires a budget
        # for `enabled` and accepts the same effort under `adaptive`.
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled" and (
            "budget_tokens" not in thinking
        ):
            thinking["type"] = "adaptive"
        # Backstop for rows a newer dsh adds outside the disabled list.
        tools = payload.get("tools")
        if isinstance(tools, list):
            allowed = _LOCAL_TOOLS | (_WEB_TOOLS if self.internet_enabled else frozenset())
            payload["tools"] = [
                tool for tool in tools if isinstance(tool, dict) and tool.get("name") in allowed
            ]

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        self._touch()
        super()._capture_model_response(path, payload)
        # Messages streams report usage in message_start and cumulatively in
        # message_delta; keep each field's maximum per message.
        kind = payload.get("type")
        if kind == "message_start":
            message = payload.get("message")
            if not isinstance(message, Mapping):
                return
            key = message.get("id")
            self._response_context.message_id = key
            raw_usage = message.get("usage")
        elif kind == "message_delta":
            key = getattr(self._response_context, "message_id", None)
            raw_usage = payload.get("usage")
        else:
            return
        usage = self._normalize_token_usage(raw_usage)
        if key is None or usage is None:
            return
        with self._tokens_lock:
            old = self._message_usage.get(str(key), TokenUsage())
            self._message_usage[str(key)] = TokenUsage(
                input_tokens=max(old.input_tokens, usage.input_tokens),
                output_tokens=max(old.output_tokens, usage.output_tokens),
                cache_read_tokens=max(old.cache_read_tokens, usage.cache_read_tokens),
                cache_write_tokens=max(old.cache_write_tokens, usage.cache_write_tokens),
            )

    def get_tokens(self) -> TokenUsage:
        with self._tokens_lock:
            usages = list(self._message_usage.values())
        return TokenUsage(
            input_tokens=sum(item.input_tokens for item in usages),
            output_tokens=sum(item.output_tokens for item in usages),
            cache_read_tokens=sum(item.cache_read_tokens for item in usages),
            cache_write_tokens=sum(item.cache_write_tokens for item in usages),
        )

    def _begin_token_session(self, *, resume: bool, session_id: str | None) -> None:
        previous = self._token_session_id
        super()._begin_token_session(resume=resume, session_id=session_id)
        if not resume or (session_id is not None and session_id != previous):
            with self._tokens_lock:
                self._message_usage = {}

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if getattr(self.model, "auth_mode", None) != "api":
            raise ValueError("dsh currently supports API-credit auth only")
        endpoint = self.model.endpoint_for("anthropic")
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                request_overrides=self.model.request_overrides(),
                request_drop_fields=self.model.request_drop_fields(),
                request_transform=self._filter_request,
                stream_line_transform=_drop_null_usage,
                stream_stall_timeout=_STREAM_STALL_SECONDS,
                # dsh always sends its placeholder key as x-api-key. Beside a
                # Bearer token, OpenRouter's Messages endpoint treats it as an
                # Anthropic key and restricts routing to provider anthropic.
                drop_headers=(
                    ("x-api-key",)
                    if any(name.lower() == "authorization" for name in endpoint.headers)
                    else ()
                ),
                listen_host=route.listen_host,
                client_host=route.client_host,
                allow_remote_clients=route.allow_remote_clients,
                offline_models=self._offline_models,
                unix_socket=route.unix_socket,
                client_port=route.client_port,
            ) as proxy,
        ):
            self._request_capture_url = proxy.base_url
            try:
                yield from super()._stream_once(
                    message, resume=resume, session_id=session_id, last=last
                )
            finally:
                del self._request_capture_url


__all__ = ["DeepSeekHarnessAgent"]
