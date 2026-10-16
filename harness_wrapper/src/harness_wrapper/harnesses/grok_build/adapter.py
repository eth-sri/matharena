"""Grok Build headless sessions through an isolated Responses proxy."""

from __future__ import annotations

import io
import json
import re
import urllib.error
import urllib.request
import uuid
from collections.abc import Iterator, Mapping
from pathlib import Path
from threading import Lock
from typing import Any
from urllib.parse import urlsplit

from loguru import logger

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...models.request_policy import open_offline_upstream
from ...tools import CLIInstallation, CLIProcessError
from ...traces import TokenUsage

_LOCAL_TOOLS = frozenset(
    {
        "run_terminal_command",
        "read_file",
        "search_replace",
        "list_dir",
        "grep",
        "write",
        "todo_write",
        "get_command_or_subagent_output",
        "kill_command_or_subagent",
    }
)
_WEB_TOOLS = frozenset({"web_fetch", "web_search", "x_search"})
_MINIMAL_PROMPT = (
    "Solve the user's task using mathematical reasoning and the available local file "
    "and terminal tools. The writable working directory is /work. "
    "There is no public internet access. Do not use web search, remote APIs, skills, "
    "MCP, or subagents. Read installed libraries outside /work when needed."
)
_MODEL_ALIAS = "harness-wrapper"
_CONTEXT_LIMIT = re.compile(r"maximum context length is ([\d,]+) tokens", re.IGNORECASE)
_CONTEXT_INPUT = re.compile(
    r"\(([\d,]+) of text input(?:, ([\d,]+) of tool input)?, ([\d,]+) in the output\)"
)
_CONTEXT_HEADROOM = 8192


def _reduced_output_budget(message: str, budget: Any) -> int | None:
    """Use OpenRouter's own token count; never estimate encrypted reasoning size."""
    if type(budget) is not int or budget <= 0:
        return None
    limit = _CONTEXT_LIMIT.search(message)
    counts = _CONTEXT_INPUT.search(message)
    if limit is None or counts is None:
        return None
    def number(value: str | None) -> int:
        return int((value or "0").replace(",", ""))

    context = number(limit[1])
    input_tokens = number(counts[1]) + number(counts[2])
    if number(counts[3]) != budget or input_tokens + budget <= context:
        return None
    reduced = context - input_tokens - _CONTEXT_HEADROOM
    return reduced if 0 < reduced < budget else None


class GrokBuildAgent(Agent):
    harness_name = "grok-build"
    aliases = ("grok", "grok-cli")
    accepted_api_types = ("openai",)
    installation = CLIInstallation(executable="grok", package="xai/grok-build")

    def __init__(self, *args: Any, model_context_window: int | None = None, **kwargs: Any) -> None:
        if model_context_window is not None and (
            type(model_context_window) is not int or model_context_window <= 0
        ):
            raise ValueError("model_context_window must be a positive integer")
        self.model_context_window = model_context_window or 500000
        self._output_token_ceiling: int | None = None
        self._context_budget_lock = Lock()
        self._provider_usage = TokenUsage()
        self._final_text = ""
        self._ended = False
        self._tool_names: dict[str, str] = {}
        super().__init__(*args, **kwargs)

    @property
    def _private_home(self) -> Path:
        return self.root / ".harness-home"

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        endpoint = self.model.endpoint_for("openai")
        runtime_root = getattr(self.env, "container_root", self.root)
        runtime_home = runtime_root / ".harness-home"
        config = [
            f"[model.{_MODEL_ALIAS}]",
            f"model = {json.dumps(self.model_name)}",
            f"base_url = {json.dumps(getattr(self, '_request_capture_url', endpoint.url))}",
            'env_key = "GROK_HARNESS_API_KEY"',
            'api_backend = "responses"',
            f"context_window = {self.model_context_window}",
            f"supports_backend_search = {str(self.internet_enabled).lower()}",
            "supports_reasoning_effort = true",
            "inference_idle_timeout_secs = 28800",
            "[models]",
            f'default = "{_MODEL_ALIAS}"',
            f'session_summary = "{_MODEL_ALIAS}"',
            f'web_search = "{_MODEL_ALIAS}"',
            f'allowed_models = ["{_MODEL_ALIAS}"]',
            "[cli]",
            "auto_update = false",
            "[session]",
            "load_envrc = false",
        ]
        if self.minimal_context:
            config.extend(
                [
                    "[features]",
                    f"web_fetch = {str(self.internet_enabled).lower()}",
                    "lsp_tools = false",
                    "tool_search = false",
                    "[memory]",
                    "enabled = false",
                    "[subagents]",
                    "enabled = false",
                    "[skills]",
                    "paths = []",
                    "[plugins]",
                    "paths = []",
                    "enabled = []",
                ]
            )
            for compat in ("cursor", "claude"):
                config.append(f"[compat.{compat}]")
                config.extend(
                    f"{key} = false" for key in ("skills", "rules", "agents", "mcps", "hooks")
                )
        config_path = self._private_home / ".grok/config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("\n".join(config) + "\n", encoding="utf-8")
        environment.update(
            {
                "HOME": str(runtime_home),
                "GROK_HOME": str(runtime_home / ".grok"),
                "GROK_HARNESS_API_KEY": endpoint.api_key,
                "GROK_DISABLE_AUTOUPDATER": "1",
                "GROK_BACKEND_SEARCH": "1" if self.internet_enabled else "0",
            }
        )
        if self.minimal_context:
            environment.update(
                {
                    "GROK_MEMORY": "0",
                    "GROK_SUBAGENTS": "0",
                    "GROK_WEB_FETCH": "1" if self.internet_enabled else "0",
                    "GROK_TOOL_SEARCH": "0",
                    "GROK_LSP_TOOLS": "0",
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
            raise ValueError("Grok Build requires a prompt")
        # A file avoids the OS argument limit for long benchmark statements.
        prompt_path = self.root / ".harness_wrapper/grok-prompt.txt"
        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        prompt_path.write_text(
            message or "Continue the interrupted task from where it stopped.", encoding="utf-8"
        )
        runtime_root = getattr(self.env, "container_root", self.root)
        command = [
            self.executable,
            "--prompt-file",
            str(runtime_root / prompt_path.relative_to(self.root)),
            "--model",
            _MODEL_ALIAS,
            "--output-format",
            "streaming-json",
            "--always-approve",
            # The outer container supplies the filesystem/network boundary.
            "--sandbox",
            "off",
        ]
        if self.minimal_context:
            command.extend(
                (
                    "--no-plan",
                    "--no-subagents",
                    "--system-prompt-override",
                    self.network_instructions(_MINIMAL_PROMPT),
                )
            )
        if not self.internet_enabled:
            command.append("--disable-web-search")
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            command.extend(("--reasoning-effort", str(reasoning)))
        if resume:
            command.extend(("--resume", session_id) if session_id else ("--continue",))
        else:
            command.extend(
                ("--session-id", getattr(self, "_new_session_id", None) or str(uuid.uuid4()))
            )
        return command

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if not path.rstrip("/").endswith("/responses"):
            raise ValueError("Grok Build may only access Responses through this proxy")
        # OpenRouter's Responses endpoint is stateless. Grok Build 1.0.40
        # replays complete history natively; fail rather than drop any future
        # stateful continuation whose history may be incomplete.
        if payload.get("previous_response_id") is not None:
            raise ValueError(
                "Grok Build must send full Responses history without previous_response_id"
            )
        payload.pop("previous_response_id", None)
        payload["store"] = False
        with self._context_budget_lock:
            ceiling = self._output_token_ceiling
        if ceiling is not None and type(payload.get("max_output_tokens")) is int:
            payload["max_output_tokens"] = min(payload["max_output_tokens"], ceiling)
        if not self.minimal_context:
            return
        tools = payload.get("tools")
        if isinstance(tools, list):
            # Preserve Grok's auxiliary title schema, a local structured output.
            title_only = (
                len(tools) == 1
                and isinstance(tools[0], dict)
                and tools[0].get("name") == "session_title"
            )
            if not title_only:
                payload["tools"] = [
                    tool
                    for tool in tools
                    if isinstance(tool, dict)
                    and (
                        tool.get("type") == "function" and (
                            tool.get("name") in _LOCAL_TOOLS
                            or self.internet_enabled and tool.get("name") in _WEB_TOOLS
                        )
                        or self.internet_enabled and tool.get("type") in {"web_search", "x_search"}
                    )
                ]

    def _open_budgeted_request(self, request: urllib.request.Request, *, timeout: float) -> Any:
        """Correct a rejected output reservation once, keeping the native turn intact.

        Only an explicit context-overflow HTTP 400 is retried here. Streaming
        failures and reasoning-only responses retain Grok Build's native retries.
        """
        opener = open_offline_upstream if self._offline_models else urllib.request.urlopen
        try:
            return opener(request, timeout=timeout)
        except urllib.error.HTTPError as error:
            if error.code != 400:
                raise
            body = error.read()
            error.close()
            try:
                details = json.loads(body).get("error", {})
                payload = json.loads(request.data or b"{}")
                budget = payload.get("max_output_tokens")
                reduced = _reduced_output_budget(details.get("message", ""), budget)
            except (ValueError, AttributeError, TypeError):
                reduced = None
            if reduced is None:
                raise urllib.error.HTTPError(
                    error.url, error.code, error.msg, error.headers, io.BytesIO(body)
                ) from error

        with self._context_budget_lock:
            self._output_token_ceiling = min(self._output_token_ceiling or reduced, reduced)
            payload["max_output_tokens"] = self._output_token_ceiling
        path = urlsplit(request.full_url).path
        self._record_trace_event(AgentEvent(
            type="recovery", role="meta", content={
                "reason": "context_output_budget_reduced",
                "previous_max_output_tokens": budget,
                "max_output_tokens": payload["max_output_tokens"],
                "headroom_tokens": _CONTEXT_HEADROOM,
                "provider_error": details,
            },
        ))
        logger.warning(
            "Grok context limit: retrying the same request with max_output_tokens {} -> {}",
            budget, payload["max_output_tokens"],
        )
        request.data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request.remove_header("Content-length")
        if self._capture_model_request(path, payload) is False:
            raise urllib.error.HTTPError(
                request.full_url, 409, "harness run limit reached", {}, io.BytesIO(b"")
            )
        return opener(request, timeout=timeout)

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        self._touch()
        super()._capture_model_response(path, payload)
        kind = payload.get("type")
        if kind in {"response.completed", "response.incomplete", "response.failed"}:
            response = payload.get("response", {})
        elif payload.get("object") == "response":
            response = payload
        else:
            return  # Ignore created/in-progress snapshots: their usage is cumulative.
        usage = self._normalize_token_usage(response.get("usage"))
        if usage is not None:
            with self._tokens_lock:
                old = self._provider_usage
                self._provider_usage = TokenUsage(
                    input_tokens=old.input_tokens + usage.input_tokens,
                    output_tokens=old.output_tokens + usage.output_tokens,
                    cache_read_tokens=old.cache_read_tokens + usage.cache_read_tokens,
                    cache_write_tokens=old.cache_write_tokens + usage.cache_write_tokens,
                )

    def get_tokens(self) -> TokenUsage:
        with self._tokens_lock:
            return self._provider_usage

    def _begin_token_session(self, *, resume: bool, session_id: str | None) -> None:
        previous = self._token_session_id
        super()._begin_token_session(resume=resume, session_id=session_id)
        if not resume or (session_id is not None and session_id != previous):
            with self._tokens_lock:
                self._provider_usage = TokenUsage()

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        kind = event.get("type")
        session_id = event.get("sessionId")
        common = {"raw": dict(event), "session_id": str(session_id) if session_id else None}
        if kind == "text":
            text = str(event.get("data") or "")
            self._final_text += text
            return [AgentEvent(type="text", content=text, role="assistant", **common)]
        if kind in {"thought", "thinking", "reasoning"}:
            return [
                AgentEvent(
                    type="reasoning", content=event.get("data", ""), role="assistant", **common
                )
            ]
        if kind == "tool_call":
            self._final_text = ""
            call_id = str(event.get("toolCallId") or "")
            name = str(event.get("toolName") or event.get("title") or "")
            self._tool_names[call_id] = name
            return [
                AgentEvent(
                    type="tool_call",
                    content=event.get("rawInput"),
                    tool_name=name,
                    tool_call_id=call_id,
                    role="assistant",
                    **common,
                )
            ]
        if kind == "tool_call_update" and event.get("status") in {"completed", "failed"}:
            call_id = str(event.get("toolCallId") or "")
            return [
                AgentEvent(
                    type="tool_result",
                    content=event.get("rawOutput") or event.get("content"),
                    tool_name=self._tool_names.get(call_id),
                    tool_call_id=call_id,
                    role="tool",
                    **common,
                )
            ]
        if kind == "usage":
            # Native usage omits title/auxiliary requests and reports cached input
            # separately; get_tokens uses the complete upstream totals instead.
            return [AgentEvent(type="usage", content=event.get("usage"), **common)]
        if kind == "end":
            self._ended = True
            if event.get("stopReason") != "end_turn":
                return [
                    AgentEvent(
                        type="error",
                        content=f"Grok Build stopped: {event.get('stopReason')}",
                        **common,
                    )
                ]
            return [AgentEvent(type="result", content=self._final_text, role="assistant", **common)]
        if kind == "error":
            return [
                AgentEvent(
                    type="error", content=event.get("message", event.get("error", event)), **common
                )
            ]
        return [AgentEvent(type="event", content=dict(event), **common)]

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if getattr(self.model, "auth_mode", None) != "api":
            raise ValueError("Grok Build currently supports API-credit auth only")
        self._final_text = ""
        self._ended = False
        self._tool_names = {}
        if not resume:
            self._new_session_id = str(uuid.uuid4())
            event = AgentEvent(type="session", session_id=self._new_session_id)
            self._observe(event)
            yield event
        endpoint = self.model.endpoint_for("openai")
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                open_upstream=self._open_budgeted_request,
                request_overrides=self.model.request_overrides(),
                request_drop_fields=self.model.request_drop_fields(),
                request_transform=self._filter_request,
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
                if not self._ended and not self._turn_cancelled:
                    raise CLIProcessError(
                        [self.executable], 1, "Grok Build exited without a terminal event"
                    )
            finally:
                del self._request_capture_url
