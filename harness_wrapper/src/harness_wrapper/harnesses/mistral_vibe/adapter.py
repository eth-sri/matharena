"""Mistral Vibe programmatic sessions through an isolated Chat Completions proxy."""

from __future__ import annotations

import json
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

from loguru import logger

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...models.request_policy import open_offline_upstream
from ...tools import CLIInstallation, CLIProcessError
from ...traces import TokenUsage

# Vibe's Unified Harness also offers search_tool_functions/run_typescript (an
# embedded TypeScript runtime that can spawn subagents and processes), cron, and
# a scratchpad. They cannot be disabled natively, so the proxy removes them.
_LOCAL_TOOLS = frozenset({"bash", "read_file", "write_file", "edit", "todo"})
_MINIMAL_PROMPT = (
    "Solve the user's task using mathematical reasoning and the available local file "
    "and terminal tools. The writable working directory is /work. "
    "There is no public internet access. Do not use web search, remote APIs, skills, "
    "MCP, or subagents. Read installed libraries outside /work when needed."
)
# First sentence of the system prompt assembled by Vibe's harness core.
_NATIVE_PROMPT_PREFIX = "You help users get complex tasks done."
_MODEL_ALIAS = "harness-wrapper"
_CONTEXT_HEADROOM = 8192
# Vibe's bash tool starts a login shell, so /etc/profile replaces the sandbox
# PATH, and the PyInstaller bundle's LD_LIBRARY_PATH makes system libraries
# load its older libstdc++. Restore the PATH the shell inherited (its initial
# environ; Vibe's own is overwritten by its process title) and the original
# library path.
_SHELL_PROFILE = """\
_vibe_path=$(tr '\\0' '\\n' < /proc/$$/environ 2>/dev/null | sed -n 's/^PATH=//p')
[ -n "$_vibe_path" ] && export PATH="$_vibe_path"
unset _vibe_path
if [ -n "${LD_LIBRARY_PATH_ORIG+set}" ]; then
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH_ORIG"
    unset LD_LIBRARY_PATH_ORIG
else
    unset LD_LIBRARY_PATH
fi
"""
_THINKING_LEVELS = {
    "none": "off",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "max",
    "max": "max",
}


def _stream_error(line: bytes) -> Any:
    """Return a provider error carried by one SSE line, if any."""

    line = line.strip()
    if not line.startswith(b"data:"):
        return None
    try:
        chunk = json.loads(line[5:].strip())
    except ValueError:
        return None
    if not isinstance(chunk, dict):
        return None
    if chunk.get("error") is not None:
        return chunk["error"]
    for choice in chunk.get("choices") or ():
        if isinstance(choice, dict) and choice.get("finish_reason") == "error":
            return choice.get("error") or {"message": "finish_reason=error"}
    return None


class _ProviderErrorCutoff:
    """End an SSE stream before a mid-stream provider error reaches Vibe.

    Vibe 2.26 maps finish_reason="error" to an ordinary finish, which would
    save a truncated answer as complete. Without a finish reason, Vibe instead
    reports an incomplete stream and retries or fails the turn.
    """

    def __init__(self, response: Any, on_error: Callable[[Any], None]) -> None:
        self._response = response
        self._on_error = on_error
        self._buffer = b""
        self._stopped = False
        self.status = response.status
        self.headers = response.headers

    def read1(self, size: int = -1) -> bytes:
        while not self._stopped:
            data = self._response.read1(size)
            if not data:
                self._stopped = True
                rest, self._buffer = self._buffer, b""
                return rest
            lines = (self._buffer + data).split(b"\n")
            self._buffer = lines.pop()
            forwarded = []
            for line in lines:
                error = _stream_error(line)
                if error is not None:
                    self._stopped = True
                    self._on_error(error)
                    break
                forwarded.append(line + b"\n")
            if forwarded:
                return b"".join(forwarded)
        return b""

    read = read1

    def close(self) -> None:
        self._response.close()

    def __enter__(self) -> _ProviderErrorCutoff:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


class MistralVibeAgent(Agent):
    harness_name = "mistral-vibe"
    aliases = ("vibe", "vibe-cli", "mistral")
    accepted_api_types = ("openai",)
    # `-p` without text reads the prompt from stdin, avoiding argument limits.
    prompt_via_stdin = True
    installation = CLIInstallation(executable="vibe", package="mistralai/mistral-vibe")

    def __init__(self, *args: Any, model_context_window: int | None = None, **kwargs: Any) -> None:
        if model_context_window is not None and (
            type(model_context_window) is not int or model_context_window <= 0
        ):
            raise ValueError("model_context_window must be a positive integer")
        self.model_context_window = model_context_window
        self._provider_usage = TokenUsage()
        self._seen_entries: set[str] = set()
        self._final_text = ""
        super().__init__(*args, **kwargs)

    @property
    def _private_home(self) -> Path:
        return self.root / ".harness-home"

    def _model_settings(self) -> list[str]:
        settings = [
            f"name = {json.dumps(self.model_name)}",
            f'provider = "{_MODEL_ALIAS}"',
            f'alias = "{_MODEL_ALIAS}"',
        ]
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            thinking = _THINKING_LEVELS.get(str(reasoning).lower(), "high")
            settings.append(f'thinking = "{thinking}"')
        if self.model_context_window is not None:
            output_limit = self.model.request_overrides().get("max_tokens")
            threshold = (
                self.model_context_window - output_limit - _CONTEXT_HEADROOM
                if type(output_limit) is int and output_limit > 0
                else 0
            )
            if threshold <= 0:
                raise ValueError(
                    "Mistral Vibe model_context_window requires a positive max_tokens that "
                    f"leaves room for history and {_CONTEXT_HEADROOM} tokens of headroom"
                )
            # Vibe sends no output limit itself; compact before the proxy's
            # max_tokens allowance no longer fits beside the history.
            settings.extend(
                (
                    f"max_context_length = {self.model_context_window}",
                    f"auto_compact_threshold = {threshold}",
                )
            )
        return settings

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        endpoint = self.model.endpoint_for("openai")
        runtime_root = getattr(self.env, "container_root", self.root)
        runtime_home = runtime_root / ".harness-home"
        config = [
            f'active_model = "{_MODEL_ALIAS}"',
            "enable_telemetry = false",
            "enable_update_checks = false",
            "enable_auto_update = false",
            "enable_notifications = false",
            "enable_connectors = false",
            "enable_subagents = false",
            # Read timeout is per streamed chunk; retries resend the whole request.
            "api_timeout = 28800.0",
            "api_retry_max_elapsed_time = 3600.0",
        ]
        if self.minimal_context:
            config.append("include_project_context = false")
        config.extend(
            [
                "[experiments]",
                "enable = false",
                "[session_logging]",
                "enabled = true",
                "generate_titles = false",
                "[[providers]]",
                f'name = "{_MODEL_ALIAS}"',
                f"api_base = {json.dumps(getattr(self, '_request_capture_url', endpoint.url))}",
                'api_key_env_var = "VIBE_HARNESS_API_KEY"',
                'backend = "generic"',
                'api_style = "openai"',
                # OpenRouter's Chat Completions field; Vibe replays it on later turns.
                'reasoning_field_name = "reasoning"',
                "[[models]]",
                *self._model_settings(),
            ]
        )
        config_path = self._private_home / ".vibe/config.toml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("\n".join(config) + "\n", encoding="utf-8")
        (self._private_home / ".bash_profile").write_text(_SHELL_PROFILE, encoding="utf-8")
        environment.update(
            {
                "HOME": str(runtime_home),
                "VIBE_HOME": str(runtime_home / ".vibe"),
                # The Rust CLI rejects --resume together with --prompt.
                "VIBE_CLI": "python",
                "VIBE_TEST_DISABLE_KEYRING": "1",
                "VIBE_HARNESS_API_KEY": endpoint.api_key,
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
            raise ValueError("Mistral Vibe requires a prompt")
        command = [self.executable, "-p"]
        if message not in (None, "-"):
            command.append(str(message))
        command.extend(
            (
                "--output",
                "streaming",
                "--auto-approve",
                # The outer container supplies the filesystem/network boundary.
                "--trust",
            )
        )
        if self.minimal_context:
            for tool in sorted(_LOCAL_TOOLS):
                command.extend(("--enabled-tools", tool))
        if resume:
            command.extend(("--resume", session_id) if session_id else ("--continue",))
        return command

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if not path.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Mistral Vibe may only access Chat Completions through this proxy")
        # Vibe always sends its model temperature (0.2 unless configured). Leave
        # sampling to the provider default unless the model config sets one.
        if "temperature" not in self.model.request_overrides():
            payload.pop("temperature", None)
        if not self.minimal_context:
            return
        tools = payload.get("tools")
        if isinstance(tools, list):
            payload["tools"] = [
                tool
                for tool in tools
                if isinstance(tool, dict)
                and tool.get("type") == "function"
                and tool.get("function", {}).get("name") in _LOCAL_TOOLS
            ]
        for message in payload.get("messages", []):
            if (
                isinstance(message, dict)
                and message.get("role") == "system"
                and str(message.get("content", "")).startswith(_NATIVE_PROMPT_PREFIX)
            ):
                message["content"] = _MINIMAL_PROMPT

    def _record_stream_error(self, error: Any) -> None:
        self._record_trace_event(
            AgentEvent(
                type="recovery",
                role="meta",
                content={"reason": "provider_stream_error", "provider_error": error},
            )
        )
        logger.bind(component="model_provider", harness=self.harness_name).warning(
            "Provider ended the stream with an error; withholding it so Vibe retries"
        )

    def _open_upstream(self, request: urllib.request.Request, *, timeout: float) -> Any:
        opener = open_offline_upstream if self._offline_models else urllib.request.urlopen
        response = opener(request, timeout=timeout)
        if "text/event-stream" not in str(response.headers.get("content-type", "")):
            return response
        return _ProviderErrorCutoff(response, self._record_stream_error)

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        self._touch()
        super()._capture_model_response(path, payload)
        usage = self._normalize_token_usage(payload.get("usage"))
        if usage is None:
            return
        # Vibe's programmatic output reports no usage; count every proxied response.
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
        entry_id = event.get("id")
        if isinstance(entry_id, str):
            # A resumed session first re-prints its completed history.
            if entry_id in self._seen_entries:
                return []
            self._seen_entries.add(entry_id)
        kind = event.get("type")
        session_id = event.get("sessionId")
        common: dict[str, Any] = {
            "raw": dict(event),
            "session_id": str(session_id) if session_id else None,
        }
        if kind == "message":
            if event.get("role") != "assistant":
                # The wrapper records submitted prompts; Vibe's echo carries the session.
                return [AgentEvent(type="session", role="meta", **common)]
            text = "".join(
                str(part.get("text") or "")
                for part in event.get("content") or ()
                if isinstance(part, Mapping) and part.get("type") == "text"
            )
            self._final_text = text
            return [AgentEvent(type="message", content=text, role="assistant", **common)]
        if kind == "reasoning":
            return [
                AgentEvent(
                    type="reasoning", content=event.get("text") or "", role="assistant", **common
                )
            ]
        if kind == "effect":
            self._final_text = ""
            detail = event.get("detail") or {}
            state = event.get("state") or {}
            name = str(detail.get("toolName") or event.get("title") or "")
            call_id = str(entry_id or "")
            result = state.get("outputText")
            if result is None:
                result = state.get("output", state.get("error"))
            return [
                AgentEvent(
                    type="tool_call",
                    content=detail.get("input"),
                    tool_name=name,
                    tool_call_id=call_id,
                    role="assistant",
                    **common,
                ),
                AgentEvent(
                    type="tool_result",
                    content=result,
                    tool_name=name,
                    tool_call_id=call_id,
                    role="tool",
                    **common,
                ),
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
            raise ValueError("Mistral Vibe currently supports API-credit auth only")
        self._final_text = ""
        endpoint = self.model.endpoint_for("openai")
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                open_upstream=self._open_upstream,
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
            finally:
                del self._request_capture_url
        if self._turn_cancelled:
            return
        if self.session_id is None:
            raise CLIProcessError([self.executable], 1, "Mistral Vibe reported no session")
        # Vibe exits 0 without a terminal event; its answer is the turn's last text.
        result = AgentEvent(
            type="result", content=self._final_text, role="assistant", session_id=self.session_id
        )
        self._observe(result)
        yield result
