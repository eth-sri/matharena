"""Qwen Code CLI adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, CLIProcessError
from ...traces import TokenUsage

_MINIMAL_INSTRUCTIONS = """Use the available local file and shell tools to solve the user's task.
Do not use internet or remote-search tools.
"""


class QwenCodeAgent(Agent):
    harness_name = "qwen-code"
    aliases = ("qwen", "qwen-cli")
    accepted_api_types = ("openai", "anthropic", "gemini")
    installation = CLIInstallation(
        executable="qwen",
        package="@qwen-code/qwen-code",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._cumulative_usage = TokenUsage()
        self._cumulative_usage_session_id: str | None = None

    def _begin_token_session(self, *, resume: bool, session_id: str | None) -> None:
        previous_session_id = self._token_session_id
        super()._begin_token_session(resume=resume, session_id=session_id)
        if not resume or (session_id is not None and session_id != previous_session_id):
            with self._tokens_lock:
                self._cumulative_usage = TokenUsage()
                self._cumulative_usage_session_id = session_id

    def _incremental_usage(self, value: Any, session_id: str | None) -> Any:
        """Qwen results report session totals, even after failed turns and resumes."""
        usage = self._normalize_token_usage(value)
        if usage is None:
            return value
        fields = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
        with self._tokens_lock:
            current_session_id = session_id or self._token_session_id
            if (
                current_session_id is not None
                and self._cumulative_usage_session_id is not None
                and current_session_id != self._cumulative_usage_session_id
            ):
                self._cumulative_usage = TokenUsage()
            self._cumulative_usage_session_id = current_session_id
            previous = self._cumulative_usage
            # Duplicate, partial, or older snapshots must not lower the baseline.
            self._cumulative_usage = TokenUsage(**{
                key: max(getattr(previous, key), getattr(usage, key)) for key in fields
            })
            return {
                key: getattr(self._cumulative_usage, key) - getattr(previous, key)
                for key in fields
            }

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        environment["QWEN_STREAM_IDLE_TIMEOUT_MS"] = "28800000"
        environment["QWEN_STREAM_MAX_LIFETIME_MS"] = "28800000"
        capture_url = getattr(self, "_request_capture_url", None)
        protocol = self._model_protocol() or "openai"
        if capture_url is not None:
            base_name = {
                "openai": "OPENAI_BASE_URL",
                "anthropic": "ANTHROPIC_BASE_URL",
                "gemini": "GOOGLE_GEMINI_BASE_URL",
            }[protocol]
            environment[base_name] = str(capture_url)
        environment["QWEN_CODE_SUPPRESS_YOLO_WARNING"] = "1"
        environment["QWEN_CODE_EMIT_TOOL_USE_SUMMARIES"] = "0"
        environment["QWEN_CODE_SAFE_MODE"] = "true" if self.minimal_context else "false"
        return environment

    def _stream_once(
        self, message: str | None, *, resume: bool,
        session_id: str | None = None, last: bool = False,
    ) -> Iterator[AgentEvent]:
        completed = False
        for event in self._stream_with_capture(
            message, resume=resume, session_id=session_id, last=last
        ):
            completed |= event.type == "result"
            yield event
        if not completed:
            raise CLIProcessError(
                [self.executable], 1, "Qwen Code ended without a completed result"
            )

    def _stream_with_capture(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if getattr(self.model, "auth_mode", None) != "api":
            yield from super()._stream_once(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        endpoint_for = getattr(self.model, "endpoint_for", None)
        if not callable(endpoint_for):
            yield from super()._stream_once(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        endpoint = endpoint_for(self._model_protocol() or "openai")
        override_provider = getattr(self.model, "request_overrides", None)
        drop_provider = getattr(self.model, "request_drop_fields", None)
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                request_overrides=(override_provider() if callable(override_provider) else {}),
                request_drop_fields=(drop_provider() if callable(drop_provider) else ()),
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

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if (
            not self.internet_enabled
            or self._model_protocol() != "openai"
            or not path.rstrip("/").endswith("/chat/completions")
            or urlsplit(self.model.endpoint_for("openai").url).hostname
            not in {"openrouter.ai", "us.openrouter.ai"}
        ):
            return
        tools = payload.get("tools")
        # Keep auxiliary/tool-free requests intact; OpenRouter executes searches.
        if isinstance(tools, list) and tools and not any(
            tool.get("type") == "openrouter:web_search" for tool in tools
        ):
            tools.append({"type": "openrouter:web_search"})

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
        command = [
            self.executable,
            "--output-format",
            "stream-json",
            "--model",
            self.model_name,
            "--auth-type",
            self._model_protocol() or "openai",
            "--approval-mode",
            "yolo",
        ]
        if self.minimal_context:
            # Bare mode removes web_fetch even when it is not excluded.
            if not self.internet_enabled:
                command.append("--bare")
            command.extend(
                (
                    "--safe-mode",
                    "--exclude-tools",
                    "agent" if self.internet_enabled else "agent,web_fetch,web_search",
                    "--system-prompt",
                    self.network_instructions(_MINIMAL_INSTRUCTIONS),
                )
            )
        if resume:
            if session_id is not None:
                command.extend(("--resume", session_id))
            else:
                command.append("--continue")
        if message is not None:
            command.extend(("--prompt", message))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        native_type = str(event.get("type", "event"))
        session_id = event.get("session_id")
        normalized_session_id = str(session_id) if session_id else None

        def make(
            event_type: str,
            content: Any = None,
            *,
            role: str | None = None,
            tool_name: str | None = None,
            tool_call_id: str | None = None,
        ) -> AgentEvent:
            return AgentEvent(
                type=event_type,
                content=content,
                role=role,
                tool_name=tool_name,
                tool_call_id=tool_call_id,
                session_id=normalized_session_id,
                raw=event,
            )

        if native_type == "system" and event.get("subtype") == "session_start":
            return [make("session", event)]
        if native_type in {"assistant", "user"}:
            message = event.get("message", event)
            role = message.get("role", native_type) if isinstance(message, Mapping) else native_type
            blocks = message.get("content", []) if isinstance(message, Mapping) else message
            if isinstance(blocks, str):
                blocks = [{"type": "text", "text": blocks}]
            normalized: list[AgentEvent] = []
            for block in blocks if isinstance(blocks, list) else []:
                if not isinstance(block, Mapping):
                    normalized.append(make("message", block, role=role))
                    continue
                block_type = block.get("type")
                if block_type == "text":
                    normalized.append(make("message", block.get("text", ""), role=role))
                elif block_type == "thinking":
                    normalized.append(make("reasoning", block.get("thinking", ""), role=role))
                elif block_type == "tool_use":
                    normalized.append(
                        make(
                            "tool_call",
                            block.get("input"),
                            role=role,
                            tool_name=block.get("name"),
                            tool_call_id=block.get("id"),
                        )
                    )
                elif block_type == "tool_result":
                    normalized.append(
                        make(
                            "tool_result",
                            block.get("content"),
                            role="tool",
                            tool_call_id=block.get("tool_use_id"),
                        )
                    )
                else:
                    normalized.append(make(str(block_type or "message"), block, role=role))
            return normalized
        if native_type == "result":
            content = event.get("result")
            # Qwen can report API failures as successful terminal answers.
            api_error = isinstance(content, str) and content.strip().startswith("[API Error:")
            if (
                event.get("is_error")
                or str(event.get("subtype", "")).startswith("error")
                or api_error
            ):
                error = event.get("error")
                detail = error.get("message") if isinstance(error, Mapping) else error
                result = [make("error", detail or content or event)]
            else:
                result = [make("result", content)]
            if event.get("usage") is not None:
                result.append(make(
                    "usage", self._incremental_usage(event["usage"], normalized_session_id)
                ))
            return result
        return super().normalize_event(event)


__all__ = ["QwenCodeAgent"]
