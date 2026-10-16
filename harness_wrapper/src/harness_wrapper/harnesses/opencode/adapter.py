"""OpenCode CLI adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, compact_json

_PROVIDER_ID = "harness-wrapper"
_MINIMAL_INSTRUCTIONS = (
    "Use the available local file and shell tools to solve the user's task. "
    "Do not use internet or remote-search tools."
)


class OpenCodeAgent(Agent):
    harness_name = "opencode"
    aliases = ("open-code",)
    accepted_api_types = ("openai", "gemini")
    installation = CLIInstallation(
        executable="opencode",
        package="opencode-ai",
    )

    def __init__(self, *args: Any, model_context_window: int | None = None, **kwargs: Any) -> None:
        if model_context_window is not None and (
            type(model_context_window) is not int or model_context_window <= 0
        ):
            raise ValueError("model_context_window must be a positive integer")
        self.model_context_window = model_context_window
        super().__init__(*args, **kwargs)

    @property
    def _qualified_model(self) -> str:
        return f"{_PROVIDER_ID}/{self.model_name}"

    def _provider_config(self, base_url: str, api_key: str) -> dict[str, Any]:
        model_config: dict[str, Any] = {
            "name": self.model_name,
            "tool_call": True,
        }
        protocol = self._model_protocol() or "openai"
        override_provider = getattr(self.model, "request_overrides", None)
        overrides = override_provider() if callable(override_provider) else {}
        reasoning = overrides.get("reasoning", {})
        if getattr(self.model, "reasoning", None) is not None or (
            isinstance(reasoning, Mapping) and reasoning.get("enabled") is True
        ):
            model_config["reasoning"] = True
        if self.model_context_window is not None:
            output_limit = (
                overrides.get("generationConfig", {}).get("maxOutputTokens")
                if protocol == "gemini"
                else overrides.get("max_completion_tokens", overrides.get("max_tokens"))
            )
            if type(output_limit) is not int or not 0 < output_limit < self.model_context_window:
                raise ValueError(
                    "OpenCode model_context_window requires a positive API output limit "
                    "smaller than the context window"
                )
            # The proxy can request more output than OpenCode's internal output cap.
            # Set input explicitly so compaction reserves the actual API allowance.
            model_config["limit"] = {
                "context": self.model_context_window,
                "input": self.model_context_window - output_limit,
                "output": output_limit,
            }
        provider_package = "@ai-sdk/google" if protocol == "gemini" else "@ai-sdk/openai-compatible"
        provider_base_url = f"{base_url.rstrip('/')}/v1beta" if protocol == "gemini" else base_url
        config: dict[str, Any] = {
            "provider": {
                _PROVIDER_ID: {
                    "npm": provider_package,
                    "name": "harness-wrapper",
                    "options": {"baseURL": provider_base_url, "apiKey": api_key},
                    "models": {self.model_name: model_config},
                }
            },
            "model": self._qualified_model,
            "enabled_providers": [_PROVIDER_ID],
            "autoupdate": False,
            "share": "disabled",
            "agent": {"title": {"disable": True}, "summary": {"disable": True}},
        }
        if self.model_context_window is not None:
            config["compaction"] = {"auto": True, "reserved": min(20000, output_limit)}
        if self.minimal_context:
            config.update(
                {
                    "snapshot": False,
                    "plugin": [],
                    "tools": {
                        "task": False, "skill": False,
                        "webfetch": self.internet_enabled, "websearch": self.internet_enabled,
                    },
                    "instructions": [],
                    "mcp": {},
                    "formatter": False,
                    "lsp": False,
                    "subagent_depth": 0,
                    "default_agent": "build",
                    "agent": {
                        "title": {"disable": True},
                        "summary": {"disable": True},
                        "build": {
                            "prompt": self.network_instructions(_MINIMAL_INSTRUCTIONS),
                            "mode": "primary",
                            "permission": {
                                "*": "allow",
                                "task": "deny",
                                "skill": "deny",
                                "webfetch": "allow" if self.internet_enabled else "deny",
                                "websearch": "allow" if self.internet_enabled else "deny",
                                "external_directory": "deny",
                            },
                        },
                    },
                }
            )
        return config

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        endpoint_for = getattr(self.model, "endpoint_for", None)
        protocol = self._model_protocol() or "openai"
        endpoint = endpoint_for(protocol) if callable(endpoint_for) else None
        base_url = getattr(self, "_request_capture_url", None) or getattr(endpoint, "url", None)
        api_key = getattr(endpoint, "api_key", None)
        if base_url and api_key:
            environment["OPENCODE_CONFIG_CONTENT"] = compact_json(
                self._provider_config(str(base_url), str(api_key))
            )
        environment.update(
            {
                "OPENCODE_DISABLE_AUTOUPDATE": "true",
                "OPENCODE_DISABLE_MODELS_FETCH": "true",
                "OPENCODE_AUTO_SHARE": "false",
                "OPENCODE_ENABLE_EXA": "true" if self.internet_enabled else "false",
            }
        )
        if self.minimal_context:
            environment.update(
                {
                    "OPENCODE_DISABLE_DEFAULT_PLUGINS": "true",
                    "OPENCODE_DISABLE_LSP_DOWNLOAD": "true",
                    "OPENCODE_DISABLE_CLAUDE_CODE": "true",
                    "OPENCODE_DISABLE_CLAUDE_CODE_PROMPT": "true",
                    "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS": "true",
                }
            )
        return environment

    def _stream_once(
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
        protocol = self._model_protocol() or "openai"
        endpoint = endpoint_for(protocol)
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
        command = [self.executable, "run", "--format", "json", "--model", self._qualified_model]
        if self.minimal_context:
            command.extend(("--pure", "--agent", "build", "--auto"))
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            command.extend(("--variant", str(reasoning)))
        if resume:
            if session_id is not None:
                command.extend(("--session", session_id))
            else:
                command.append("--continue")
        if message is not None:
            command.extend(("--", message))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        native_type = str(event.get("type", "event"))
        session_id = event.get("sessionID") or event.get("session_id")
        normalized_session_id = str(session_id) if session_id else None
        part = event.get("part")
        part = part if isinstance(part, Mapping) else {}

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

        if native_type == "step_start":
            return [make("session", event)]
        if native_type == "text":
            return [make("message", part.get("text", ""), role="assistant")]
        if native_type == "reasoning":
            return [make("reasoning", part.get("text", ""), role="assistant")]
        if native_type == "tool_use":
            state = part.get("state")
            state = state if isinstance(state, Mapping) else {}
            call_id = part.get("callID") or part.get("call_id")
            result = [
                make(
                    "tool_call",
                    state.get("input"),
                    role="assistant",
                    tool_name=str(part.get("tool") or ""),
                    tool_call_id=str(call_id) if call_id else None,
                )
            ]
            if state.get("status") == "completed":
                result.append(
                    make(
                        "tool_result",
                        state.get("output", state.get("error")),
                        role="tool",
                        tool_call_id=str(call_id) if call_id else None,
                    )
                )
            return result
        if native_type == "step_finish":
            tokens = part.get("tokens")
            tokens = tokens if isinstance(tokens, Mapping) else {}
            cache = tokens.get("cache")
            cache = cache if isinstance(cache, Mapping) else {}
            cache_read = self._usage_count(cache, "read")
            usage = {
                # OpenCode reports fresh input separately from cache reads;
                # the wrapper convention stores total input including cache.
                "input_tokens": self._usage_count(tokens, "input") + cache_read,
                # OpenCode reports visible output and reasoning separately.
                "output_tokens": (
                    self._usage_count(tokens, "output") + self._usage_count(tokens, "reasoning")
                ),
                "cache_read_tokens": cache_read,
                "cache_write_tokens": self._usage_count(cache, "write"),
            }
            return [make("usage", usage)]
        if native_type == "error":
            return [make("error", event.get("error", event))]
        return super().normalize_event(event)


__all__ = ["OpenCodeAgent"]
