"""OpenAI Codex CLI adapter."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any
from urllib.parse import urlsplit

from ...agent import Agent, AgentEvent
from ...models.oauth.openai_bridge import (
    CodexOAuthResponsesProxy,
)
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, compact_json

_MINIMAL_INSTRUCTIONS = "Use the available shell tools to solve the user's task."
_TOOL_FREE_INSTRUCTIONS = (
    "Solve the user's task using reasoning alone. No tools are available: "
    "you cannot execute code, access files, browse the internet, or delegate to agents. "
    "Follow the requested answer format."
)
_TOOL_FREE_DISABLED_FEATURES = (
    "code_mode",
    "code_mode_only",
    "code_mode_host",
    "shell_tool",
    "unified_exec",
    "apply_patch_freeform",
    "js_repl",
    "js_repl_tools_only",
)
_MINIMAL_CONFIG: tuple[tuple[str, Any], ...] = (
    ("instructions", _MINIMAL_INSTRUCTIONS),
    ("include_permissions_instructions", False),
    ("include_apps_instructions", False),
    ("include_collaboration_mode_instructions", False),
    ("include_environment_context", False),
    ("skills.include_instructions", False),
    ("skills.bundled.enabled", False),
    ("project_doc_max_bytes", 0),
    ("project_doc_fallback_filenames", []),
    ("agents.enabled", False),
    ("memories.use_memories", False),
    ("memories.generate_memories", False),
    ("orchestrator.skills.enabled", False),
    ("orchestrator.mcp.enabled", False),
    ("mcp_servers", {}),
    ("personality", "none"),
    ("tools.update_plan.enabled", False),
    ("tools.experimental_request_user_input.enabled", False),
)
_MINIMAL_REQUIRED_FEATURES = (
    # GPT-5.6 Codex models are code-mode-only. Keep the local executor alive;
    # disabling it leaves the model's advertised `functions.exec` tool present
    # but guarantees that every attempted shell call fails closed.
    "code_mode_host",
    "shell_tool",
    "unified_exec",
)
_WEB_FEATURES = (
    "browser_use",
    "browser_use_external",
    "browser_use_full_cdp_access",
    "in_app_browser",
    "standalone_web_search",
)
_MINIMAL_DISABLED_FEATURES = (
    "apps",
    "auth_elicitation",
    "computer_use",
    "goals",
    "hooks",
    "image_generation",
    "in_app_updates",
    "mentions_v2",
    "multi_agent",
    "multi_agent_v2",
    "personality",
    "plugin_sharing",
    "plugins",
    "recommended_plugins",
    "remote_plugin",
    "shell_snapshot",
    "skill_mcp_dependency_install",
    "skill_search",
    "tool_call_mcp_elicitation",
    "tool_suggest",
    "view_image",
    "workspace_dependencies",
)


class CodexCLIAgent(Agent):
    harness_name = "codex-cli"
    prompt_via_stdin = True
    aliases = ("codex", "openai-codex")
    accepted_api_types = ("openai",)
    installation = CLIInstallation(
        executable="codex",
        package="@openai/codex",
    )

    def __init__(
        self,
        *args: Any,
        tools_enabled: bool = True,
        model_context_window: int | None = None,
        **kwargs: Any,
    ) -> None:
        if model_context_window is not None and (
            type(model_context_window) is not int or model_context_window <= 0
        ):
            raise ValueError("model_context_window must be a positive integer")
        if not isinstance(tools_enabled, bool):
            raise TypeError("tools_enabled must be a boolean")
        self._turn_final_text = ""
        self.tools_enabled = tools_enabled
        self.model_context_window = model_context_window
        if not tools_enabled:
            if kwargs.get("subagents"):
                raise ValueError("Tool-free Codex cannot configure subagents")
            kwargs["subagents"] = {}
            kwargs["minimal_context"] = True
        super().__init__(*args, **kwargs)

    def _tool_free_catalog(self) -> str:
        """Override local code-only metadata without changing the requested model."""
        relative_path = ".harness_wrapper/tool-free-models.json"
        catalog = {
            "models": [
                {
                    "slug": self.model_name,
                    "display_name": self.model_name,
                    "description": "Tool-free reasoning model",
                    "supported_reasoning_levels": [],
                    "shell_type": "disabled",
                    "tool_mode": "direct",
                    "visibility": "none",
                    "supported_in_api": True,
                    "priority": 0,
                    "support_verbosity": False,
                    "apply_patch_tool_type": None,
                    "supports_parallel_tool_calls": False,
                    "supports_search_tool": False,
                    "experimental_supported_tools": [],
                    "include_skills_usage_instructions": False,
                    "include_plugin_usage_instructions": False,
                    "include_apps_usage_instructions": False,
                    "model_messages": {"instructions_template": _TOOL_FREE_INSTRUCTIONS},
                    # Match Codex's conservative fallback unless explicitly configured.
                    "context_window": self.model_context_window or 272000,
                    "truncation_policy": {"mode": "bytes", "limit": 10000},
                }
            ],
        }
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(compact_json(catalog) + "\n", encoding="utf-8")
        return relative_path if self.env is not None else str(path)

    def _effective_request_overrides(self) -> dict[str, Any]:
        override_provider = getattr(self.model, "request_overrides", None)
        overrides = dict(override_provider() if callable(override_provider) else {})
        if not self.tools_enabled:
            # Model metadata can force code-mode tools despite CLI feature flags.
            # Enforce the policy on the final API request for every turn/resume.
            overrides.update(tools=[], tool_choice="none", parallel_tool_calls=False)
        return overrides

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if (
            not self.internet_enabled
            or not self.tools_enabled
            or not path.rstrip("/").endswith("/responses")
            or payload.get("tool_choice") == "none"
            or urlsplit(self.model.endpoint_for("openai").url).hostname != "openrouter.ai"
        ):
            return
        tools = payload.get("tools") or []
        if any(tool.get("type") in {"web_search", "web_search_preview"} for tool in tools):
            return
        # Responses Lite advertises local tools in input rather than tools.
        # Leave compaction and other tool-free requests untouched.
        if not tools and not any(
            isinstance(item, Mapping) and item.get("type") == "additional_tools" and item.get("tools")
            for item in payload.get("input") or []
        ):
            return
        # Native search preserves Responses Lite's custom tools and agent messages.
        # OpenRouter's web_fetch server loop drops them when converting requests.
        payload["tools"] = [*tools, {"type": "web_search", "external_web_access": True}]

    def model_cli_args(self) -> list[str]:
        """Create an invocation-local Codex provider for explicit API models."""

        capture_url = getattr(self, "_request_capture_url", None)
        auth_mode = getattr(self.model, "auth_mode", None)
        oauth_bridge_api_key = getattr(self, "_oauth_bridge_api_key", None)
        if capture_url is None and auth_mode != "api":
            return super().model_cli_args()
        values: dict[str, Any] = {
            "model_provider": "harness_wrapper",
            "model_providers.harness_wrapper.name": "harness-wrapper",
            "model_providers.harness_wrapper.wire_api": "responses",
            "model_providers.harness_wrapper.stream_idle_timeout_ms": 28_800_000,
        }
        if capture_url is not None:
            values["model_providers.harness_wrapper.base_url"] = capture_url
        else:
            endpoint_for = getattr(self.model, "endpoint_for", None)
            if not callable(endpoint_for):
                return super().model_cli_args()
            values["model_providers.harness_wrapper.base_url"] = str(endpoint_for("openai").url)
        if auth_mode == "oauth" and oauth_bridge_api_key is None:
            values["model_providers.harness_wrapper.requires_openai_auth"] = True
        else:
            values["model_providers.harness_wrapper.env_key"] = "OPENAI_API_KEY"
        result: list[str] = []
        for key, value in values.items():
            result.extend(("-c", f"{key}={compact_json(value)}"))
        return result

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        oauth_bridge_api_key = getattr(self, "_oauth_bridge_api_key", None)
        if oauth_bridge_api_key is not None:
            environment["OPENAI_API_KEY"] = str(oauth_bridge_api_key)
        return environment

    def minimal_context_cli_args(self) -> list[str]:
        """Return an invocation-local Codex profile with only shell access."""

        result = ["--ignore-user-config", "--ignore-rules", "--strict-config"]
        if not self.tools_enabled:
            result.extend(("-c", f"model_catalog_json={compact_json(self._tool_free_catalog())}"))
        if self.model_context_window is not None:
            result.extend(("-c", f"model_context_window={self.model_context_window}"))
        for key, value in _MINIMAL_CONFIG:
            if key == "instructions":
                value = self.network_instructions(value) if self.tools_enabled else _TOOL_FREE_INSTRUCTIONS
            elif key == "agents.enabled":
                value = bool(self.subagents)
            result.extend(("-c", f"{key}={compact_json(value)}"))
        if self.tools_enabled:
            for feature in _MINIMAL_REQUIRED_FEATURES:
                result.extend(("--enable", feature))
        for feature in _MINIMAL_DISABLED_FEATURES:
            enabled = bool(self.subagents) and feature in {"multi_agent", "multi_agent_v2"}
            result.extend(("--enable" if enabled else "--disable", feature))
        for feature in _WEB_FEATURES:
            result.extend(("--enable" if self.internet_enabled and self.tools_enabled else "--disable", feature))
        if not self.tools_enabled:
            for feature in _TOOL_FREE_DISABLED_FEATURES:
                result.extend(("--disable", feature))
        return result

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        self._turn_final_text = ""
        auth_mode = getattr(self.model, "auth_mode", None)
        if auth_mode == "oauth" and getattr(self.model, "provider", None) == "openai":
            with (
                self._host_service_route() as route,
                CodexOAuthResponsesProxy(
                    on_request=self._capture_model_request,
                    on_response=self._capture_model_response,
                    request_overrides=self._effective_request_overrides(),
                    listen_host=route.listen_host,
                    client_host=route.client_host,
                    allow_remote_clients=route.allow_remote_clients,
                    offline_models=self._offline_models,
                    unix_socket=route.unix_socket,
                    client_port=route.client_port,
                ) as bridge,
            ):
                self._request_capture_url = bridge.base_url
                self._oauth_bridge_api_key = bridge.client_api_key
                try:
                    yield from super()._stream_once(
                        message, resume=resume, session_id=session_id, last=last
                    )
                finally:
                    del self._request_capture_url
                    del self._oauth_bridge_api_key
            return

        upstream_headers: Mapping[str, str] = {}
        if auth_mode == "api":
            endpoint_for = getattr(self.model, "endpoint_for", None)
            if not callable(endpoint_for):
                yield from super()._stream_once(
                    message, resume=resume, session_id=session_id, last=last
                )
                return
            endpoint = endpoint_for("openai")
            upstream_url = str(endpoint.url)
            upstream_headers = endpoint.headers
        else:
            yield from super()._stream_once(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                upstream_url,
                self._capture_model_request,
                upstream_headers=upstream_headers,
                on_response=self._capture_model_response,
                request_overrides=self._effective_request_overrides(),
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
        command = [self.executable, "exec"]
        if resume:
            command.append("resume")
        command.extend(("--json", "--model", self.model_name))
        command.extend(self.model_cli_args())
        if self.minimal_context:
            command.extend(self.minimal_context_cli_args())
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            command.extend(("-c", f"model_reasoning_effort={compact_json(reasoning)}"))
        web_search = "live" if self.internet_enabled and self.tools_enabled else "disabled"
        command.extend(("-c", f"web_search={compact_json(web_search)}"))
        command.append("--dangerously-bypass-approvals-and-sandbox")
        command.append("--skip-git-repo-check")
        if self.subagents:
            command.extend(("-c", "features.multi_agent=true"))
            for name, config in self.subagent_payload().items():
                command.extend(
                    ("-c", f"agents.{name}.description={compact_json(config['description'])}")
                )
                command.extend(("-c", f"agents.{name}.model={compact_json(config['model'])}"))
        if resume:
            if session_id is not None:
                command.append(session_id)
            else:
                command.append("--last")
        else:
            command.extend(("--cd", "." if self.env is not None else str(self.root)))
        if message is not None:
            command.extend(("--", message))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        native_type = str(event.get("type", "event"))
        if native_type in {"thread.started", "turn.started"}:
            self._turn_final_text = ""
        if native_type == "thread.started":
            session_id = event.get("thread_id")
            return [
                AgentEvent(
                    type="session",
                    content=event,
                    session_id=str(session_id) if session_id else None,
                    raw=event,
                )
            ]
        item = event.get("item")
        if native_type in {"item.started", "item.updated", "item.completed"} and isinstance(
            item, Mapping
        ):
            item_type = str(item.get("type", "item"))
            if item_type == "agent_message":
                if native_type == "item.completed" and item.get("phase") != "commentary":
                    self._turn_final_text = str(item.get("text") or "")
                return [
                    AgentEvent(
                        type="message", content=item.get("text", ""), role="assistant", raw=event
                    )
                ]
            if item_type == "reasoning":
                return [
                    AgentEvent(
                        type="reasoning", content=item.get("text", ""), role="assistant", raw=event
                    )
                ]
            if item_type in {"command_execution", "mcp_tool_call", "web_search"}:
                # Older CLI messages have no phase. Text preceding another tool
                # operation is progress, not this turn's final submission.
                self._turn_final_text = ""
                tool_name = item.get("server") or item.get("name") or item_type
                call_id = item.get("id")
                if native_type == "item.started":
                    return [
                        AgentEvent(
                            type="tool_call",
                            content=item.get("command", item.get("arguments", item)),
                            tool_name=str(tool_name),
                            tool_call_id=str(call_id) if call_id else None,
                            raw=event,
                        )
                    ]
                if native_type == "item.updated":
                    return []
                output = item.get("aggregated_output", item.get("result"))
                if output is None:
                    return []
                return [
                    AgentEvent(
                        type="tool_result",
                        content=output,
                        tool_name=str(tool_name),
                        tool_call_id=str(call_id) if call_id else None,
                        raw=event,
                    ),
                ]
            return [AgentEvent(type=item_type, content=item, raw=event)]
        if native_type == "turn.completed":
            events = []
            if event.get("usage") is not None:
                events.append(AgentEvent(type="usage", content=event["usage"], raw=event))
            events.append(AgentEvent(
                type="result", content=self._turn_final_text,
                raw={"type": native_type},
            ))
            self._turn_final_text = ""
            return events
        if native_type in {"error", "turn.failed"}:
            return [
                AgentEvent(
                    type="error", content=event.get("message", event.get("error", event)), raw=event
                ),
                AgentEvent(
                    type="warning",
                    role="meta",
                    content="Codex reported an error; token counts and cost may exclude usage "
                    "from interrupted requests.",
                ),
            ]
        return super().normalize_event(event)


__all__ = ["CodexCLIAgent"]
