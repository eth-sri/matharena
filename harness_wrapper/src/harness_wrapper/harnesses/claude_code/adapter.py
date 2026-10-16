"""Claude Code CLI adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ...agent import Agent, AgentEvent, OutputTokenLimitError
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, CLIProcessError, compact_json

_MINIMAL_INSTRUCTIONS = "Use the available shell tools to solve the user's task."


def _is_reasoning_extraction_refusal(payload) -> bool:
    """Check Claude's refusal metadata, never generated or tool text."""
    if not isinstance(payload, Mapping):
        return False
    details = payload.get("stop_details")
    if isinstance(details, Mapping) and details.get("category") == "reasoning_extraction":
        return True
    if payload.get("type") == "system" and payload.get("apiRefusalCategory") == "reasoning_extraction":
        return True
    return any(_is_reasoning_extraction_refusal(payload.get(key)) for key in ("message", "delta"))


class ClaudeCodeAgent(Agent):
    harness_name = "claude-code"
    aliases = ("claude", "anthropic")
    accepted_api_types = ("anthropic",)
    installation = CLIInstallation(
        executable="claude",
        package="@anthropic-ai/claude-code",
    )

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        if getattr(self.model, "auth_mode", None) == "api":
            endpoint = self.model.endpoint_for("anthropic")
            if any(name.lower() == "authorization" for name in endpoint.headers):
                # The CLI gets a placeholder; the host proxy injects the real bearer key.
                environment["ANTHROPIC_AUTH_TOKEN"] = endpoint.api_key
                environment["ANTHROPIC_API_KEY"] = ""
        environment["API_TIMEOUT_MS"] = "28800000"
        # Bun's separate socket-idle timer otherwise aborts non-streaming
        # fallback requests after about six minutes, ignoring API_TIMEOUT_MS.
        environment["BUN_CONFIG_HTTP_IDLE_TIMEOUT"] = "0"
        capture_url = getattr(self, "_request_capture_url", None)
        if capture_url is not None:
            environment["ANTHROPIC_BASE_URL"] = str(capture_url)
        return environment

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        output_limited = False
        self._reasoning_extraction_refusal = False
        try:
            for event in self._stream_with_capture(
                message, resume=resume, session_id=session_id, last=last
            ):
                if (
                    event.raw.get("type") == "result"
                    and event.raw.get("stop_reason") == "max_tokens"
                ):
                    output_limited = True
                    if event.type in {"result", "error"}:
                        continue
                yield event
        except CLIProcessError as error:
            legacy_output_limit = any(
                marker in error.stderr
                for marker in ("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "max_output_tokens")
            ) and any(word in error.stderr.lower() for word in ("exceeded", "exhausted", "reached"))
            if not self._reasoning_extraction_refusal and not output_limited and not legacy_output_limit:
                raise
            output_limited = True
        # Drain the terminal usage and reap the CLI before requesting recovery.
        # Some Claude versions report max_tokens with is_error=false and exit 0.
        if self._reasoning_extraction_refusal:
            # The provider declined continuation. Keep the terminal usage, but
            # never enter generic CLI recovery for this refusal category.
            result = AgentEvent(
                type="result", content="", session_id=self.session_id,
                raw={"refusal_category": "reasoning_extraction"},
            )
            self._observe(result)
            yield result
        elif output_limited:
            raise OutputTokenLimitError(
                self.build_command(message, resume=resume, session_id=session_id, last=last),
                1,
                "Claude stopped at max_output_tokens before completing the response.",
            )

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        if _is_reasoning_extraction_refusal(payload):
            self._reasoning_extraction_refusal = True
        super()._capture_model_response(path, payload)

    def _stream_with_capture(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        auth_mode = getattr(self.model, "auth_mode", None)
        upstream_headers: Mapping[str, str] = {}
        if auth_mode == "api":
            endpoint_for = getattr(self.model, "endpoint_for", None)
            if not callable(endpoint_for):
                yield from super()._stream_once(
                    message, resume=resume, session_id=session_id, last=last
                )
                return
            endpoint = endpoint_for("anthropic")
            upstream_url = str(endpoint.url)
            upstream_headers = endpoint.headers
        elif auth_mode == "oauth" and getattr(self.model, "provider", None) == "anthropic":
            upstream_url = "https://api.anthropic.com"
        else:
            yield from super()._stream_once(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        override_provider = getattr(self.model, "request_overrides", None)
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                upstream_url,
                self._capture_model_request,
                upstream_headers=upstream_headers,
                on_response=self._capture_model_response,
                request_overrides=override_provider() if callable(override_provider) else {},
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

    def _recover_cli_failure(
        self,
        error: CLIProcessError,
        *,
        tried_models: set[int],
    ) -> str | None:
        # Claude exhausts its own output-limit retries before exiting. Resume
        # only a known session, using the wrapper's existing recovery budget.
        if self.session_id is not None and isinstance(error, OutputTokenLimitError):
            return "output_limit_resume"
        return super()._recover_cli_failure(error, tried_models=tried_models)

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
            "-p",
            "--output-format",
            "stream-json",
            "--verbose",
            "--model",
            self.model_name,
        ]
        if self.model_name.startswith("anthropic/claude-"):
            # Let Claude Code recognize the model's capabilities while keeping
            # its OpenRouter ID on the wire (including web-tool helper calls).
            canonical = self.model_name.removeprefix("anthropic/").replace(".", "-")
            command.extend(("--settings", compact_json({
                "modelOverrides": {canonical: self.model_name},
            })))
        command.extend(self.model_cli_args())
        if self.minimal_context:
            command.extend(
                (
                    "--safe-mode",
                    "--disable-slash-commands",
                    "--no-chrome",
                    "--strict-mcp-config",
                    "--mcp-config",
                    compact_json({"mcpServers": {}}),
                    "--system-prompt",
                    self.network_instructions(_MINIMAL_INSTRUCTIONS),
                    "--tools",
                    "Bash,WebSearch,WebFetch" if self.internet_enabled else "Bash",
                    "--prompt-suggestions",
                    "false",
                )
            )
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            command.extend(("--effort", str(reasoning)))
        command.append("--dangerously-skip-permissions")
        if self.subagents:
            command.extend(("--agents", compact_json(self.subagent_payload())))
        if resume:
            if session_id is not None:
                command.extend(("--resume", session_id))
            else:
                command.append("--continue")
        if message is not None:
            command.extend(("--", message))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        if _is_reasoning_extraction_refusal(event):
            self._reasoning_extraction_refusal = True
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

        if native_type == "system" and event.get("subtype") == "init":
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
                    event_type = "error" if event.get("isApiErrorMessage") else "message"
                    normalized.append(make(event_type, block.get("text", ""), role=role))
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
                            role=role,
                            tool_call_id=block.get("tool_use_id"),
                        )
                    )
                else:
                    normalized.append(make(str(block_type or "message"), block, role=role))
            return normalized or [make("message", message, role=role)]
        if native_type == "result":
            event_type = "error" if event.get("is_error") else "result"
            result = [make(event_type, event.get("result"))]
            if event.get("usage") is not None:
                result.append(make("usage", event["usage"]))
            return result
        return super().normalize_event(event)


__all__ = ["ClaudeCodeAgent"]
