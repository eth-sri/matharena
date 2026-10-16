"""Google Antigravity CLI adapter."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterator, Mapping
from typing import Any

from loguru import logger

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, CLIProcessError
from .native_session import last_native_step, read_native_response


class IncompleteAntigravityResponse(CLIProcessError):
    """A zero-exit CLI turn without a verified final answer."""


_AGENT_NAME = "matharena"
_LOCAL_TOOLS = (
    "view_file",
    "write_to_file",
    "replace_file_content",
    "multi_replace_file_content",
    "list_dir",
    "find_by_name",
    "grep_search",
    "run_command",
)
_WEB_TOOLS = ("search_web", "read_url_content", "view_content_chunk", "browser_subagent")
_AGENT_PROMPT = """Use only the local file and shell tools listed for this agent.
The writable project directory is /work. You have no internet access.
Do not invoke subagents, plugins, skills, MCP servers, or remote-search tools.
"""
_EFFORT_ALIASES = {
    "minimal": "low",
    "xhigh": "high",
    "max": "high",
}


class AntigravityCLIAgent(Agent):
    """Run Gemini API models through the native ``agy`` headless client."""

    harness_name = "antigravity-cli"
    aliases = (
        "gravity",
        "gravity-cli",
        "antigravity",
        "agy",
        "google-antigravity",
    )
    accepted_api_types = ("gemini",)
    installation = CLIInstallation(
        executable="agy",
        package="google-antigravity/antigravity-cli",
    )

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._reported_usage: dict[str, dict[str, int]] = {}
        self._api_usage_lock = threading.Lock()
        self._api_usage = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0}
        self._api_usage_responses = 0

    @property
    def _container_auto_approve(self) -> bool:
        # The container, not CLI prompts, enforces the filesystem/network
        # boundary. Never silently bypass permissions for host execution.
        return bool(
            self.env is not None
            and getattr(self.env, "enabled", False)
            and getattr(self.env, "container_root", None) is not None
        )

    def _runtime_paths(self) -> None:
        home = self.root / ".harness-home"
        settings = home / ".gemini" / "antigravity-cli" / "settings.json"
        settings.parent.mkdir(parents=True, exist_ok=True)
        values: dict[str, Any] = {
            "modelProvider": "gemini",
            "enableTelemetry": False,
            "showTips": False,
            "showFeedbackSurvey": False,
            "altScreenMode": "never",
            "allowNonWorkspaceAccess": self._container_auto_approve,
        }
        if self.model.model == "gemini-4-argon" and self.model.auth_mode == "api":
            # The API-key catalog in agy 1.3.5 omits Argon.
            values["customModelsConfig"] = {
                "customModels": {
                    "gemini-4-argon": {"modelName": "gemini-4-argon"}
                }
            }
        if self.minimal_context:
            values["permissions"] = {
                "allow": [
                    "command(*)",
                    f"read_file({getattr(self.env, 'container_root', '/work')}/)",
                    f"write_file({getattr(self.env, 'container_root', '/work')}/)",
                ],
                "deny": ([] if self.internet_enabled else ["read_url(*)", "execute_url(*)"]) + ["mcp(*)"],
            }
            if self.internet_enabled:
                values["permissions"]["allow"].extend(["read_url(*)", "execute_url(*)"])
        settings.write_text(
            json.dumps(values, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if not self.minimal_context:
            return
        agent_file = home / ".gemini" / "config" / "agents" / _AGENT_NAME / "agent.md"
        agent_file.parent.mkdir(parents=True, exist_ok=True)
        tool_names = _LOCAL_TOOLS + (_WEB_TOOLS if self.internet_enabled else ())
        tools = "\n".join(f"  - {name}" for name in tool_names)
        agent_file.write_text(
            "---\n"
            f"name: {_AGENT_NAME}\n"
            "description: Minimal MathArena coding agent.\n"
            "tools:\n"
            f"{tools}\n"
            "mainAgent: true\n"
            "subagent: false\n"
            "model: inherit\n"
            "commandExecutionPolicy: eager\n"
            "mcpServers: []\n"
            "skills: []\n"
            "plugins: []\n"
            "---\n\n"
            + self.network_instructions(_AGENT_PROMPT),
            encoding="utf-8",
        )

    @property
    def _offline_models(self) -> frozenset[str]:
        models = super()._offline_models
        # The pinned CLI uses Flash Lite for housekeeping before its solver turn.
        return models | {"gemini-3.1-flash-lite-preview"} if models else models

    def model_environment(self) -> dict[str, str]:
        self._runtime_paths()
        environment = super().model_environment()
        if "HOME" not in environment:
            container_root = getattr(self.env, "container_root", None)
            runtime_root = container_root if container_root is not None else self.root
            environment["HOME"] = str(runtime_root / ".harness-home")
        capture_url = getattr(self, "_request_capture_url", None)
        if capture_url is not None:
            environment["GOOGLE_GEMINI_BASE_URL"] = str(capture_url)
        environment["NO_BROWSER"] = "true"
        return environment

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        super()._capture_model_response(path, payload)
        usage = payload.get("usageMetadata")
        if not isinstance(usage, Mapping):
            return

        def count(name: str) -> int:
            value = usage.get(name)
            return value if isinstance(value, int) and value >= 0 else 0

        # Gemini's promptTokenCount includes cached input. Output billing also
        # includes separately reported thought tokens.
        observed = {
            "input_tokens": count("promptTokenCount"),
            "output_tokens": count("candidatesTokenCount") + count("thoughtsTokenCount"),
            "cache_read_tokens": count("cachedContentTokenCount"),
        }
        with self._api_usage_lock:
            for field, value in observed.items():
                self._api_usage[field] += value
            self._api_usage_responses += 1

    def _reset_api_usage(self) -> None:
        with self._api_usage_lock:
            self._api_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
            }
            self._api_usage_responses = 0

    def _take_api_usage(self) -> dict[str, int] | None:
        with self._api_usage_lock:
            if self._api_usage_responses == 0:
                return None
            usage = self._api_usage
            self._api_usage = {
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_tokens": 0,
            }
            self._api_usage_responses = 0
            return usage

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if getattr(self.model, "auth_mode", None) != "api":
            yield from self._validated_turn(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        endpoint_for = getattr(self.model, "endpoint_for", None)
        if not callable(endpoint_for):
            yield from self._validated_turn(
                message, resume=resume, session_id=session_id, last=last
            )
            return
        endpoint = endpoint_for("gemini")
        override_provider = getattr(self.model, "request_overrides", None)
        drop_provider = getattr(self.model, "request_drop_fields", None)
        self._reset_api_usage()
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                request_overrides=(override_provider() if callable(override_provider) else {}),
                merge_request_overrides=True,
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
                yield from self._validated_turn(
                    message, resume=resume, session_id=session_id, last=last
                )
            finally:
                del self._request_capture_url

    def _validated_turn(self, message, *, resume, session_id=None, last=False):
        prior_session = session_id or (self.session_id if resume else None)
        after_step = -1
        if prior_session:
            after_step = last_native_step(self.root, prior_session)
        terminal = None
        failure = None
        response_parts = []
        try:
            for event in super()._stream_once(message, resume=resume, session_id=session_id, last=last):
                if event.type == "result":
                    terminal = event
                    continue
                if event.type == "message" and event.role in (None, "assistant"):
                    response_parts.append(str(event.content or ""))
                elif event.type in {"tool_call", "tool_result", "reasoning"}:
                    response_parts.clear()
                yield event
        except CLIProcessError as error:
            # agy can retain an earlier error after completing a newer answer.
            # Check the native terminal step before allowing the recovery loop
            # to send another continuation prompt.
            failure = error
        command = self.build_command(message, resume=resume, session_id=session_id, last=last)
        if terminal is None:
            failure = failure or IncompleteAntigravityResponse(
                command, 1, "Antigravity exited without a terminal result"
            )
        native_payload = terminal.raw.get("result", {}) if terminal else {}
        if native_payload.get("denied_actions"):
            raise IncompleteAntigravityResponse(
                command,
                1,
                "Antigravity stopped after a permission denial without completing the task",
            )
        if terminal and str(native_payload.get("status", "")).upper() != "SUCCESS":
            failure = failure or CLIProcessError(
                command, 1, str(native_payload.get("error") or "Antigravity reported a terminal error")
            )
        content = (terminal.content if terminal else None) or "".join(response_parts)
        recovered = None
        native_session = (terminal.session_id if terminal else None) or self.session_id
        if native_session:
            try:
                recovered = read_native_response(
                    self.root,
                    native_session,
                    after_step=after_step if native_session == prior_session else -1,
                    require_quiescent=failure is not None,
                )
            except (ValueError, OSError, sqlite3.Error) as error:
                logger.warning("Cannot read native Antigravity response: {}", type(error).__name__)
        if failure and (recovered is None or (resume and prior_session is None)):
            raise failure
        # The CLI result can combine the main answer and later background-task
        # follow-ups. Keep that aggregate when its native tail is complete.
        if recovered is not None:
            if not isinstance(content, str) or not content.rstrip().endswith(recovered.text.rstrip()):
                content = recovered.text
            elif failure is None:
                recovered = None
        if not isinstance(content, str) or not content.strip():
            raise IncompleteAntigravityResponse(
                command,
                1,
                "Antigravity reported SUCCESS but no final answer was captured or saved natively",
            )
        raw = dict(terminal.raw) if terminal else {}
        if failure is not None:
            raw["cli_error"] = {"returncode": failure.returncode, "stderr": failure.stderr}
            logger.warning(
                "Accepting fresh native Antigravity answer after CLI error; skipping resume: {}",
                failure,
            )
        if recovered is not None:
            raw["native_response"] = {
                "step_index": recovered.step_index,
                "source": "session_database",
            }
            if terminal is None or not terminal.content:
                logger.info(
                    "Recovered Antigravity final response from native session step {}",
                    recovered.step_index,
                )
        canonical = AgentEvent(
            type="result",
            content=content,
            role="assistant",
            session_id=native_session,
            raw=raw,
        )
        self._observe(canonical)
        yield canonical

    def _recover_cli_failure(self, error, *, tried_models):
        if isinstance(error, IncompleteAntigravityResponse):
            return "incomplete_response"
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
            "--output-format",
            "stream-json",
            "--model",
            self.model_name,
            "--print-timeout",
            "24h",
        ]
        effort = getattr(self.model, "reasoning", None)
        if effort:
            normalized_effort = _EFFORT_ALIASES.get(str(effort), str(effort))
            if normalized_effort not in {"low", "medium", "high"}:
                raise ValueError(
                    f"Antigravity CLI reasoning must be low, medium, or high, got {effort!r}"
                )
            command.extend(("--effort", normalized_effort))
        if self.minimal_context:
            command.extend(
                (
                    "--disable-slash-commands",
                    "--agent",
                    _AGENT_NAME,
                    "--mode",
                    "accept-edits",
                )
            )
        if self._container_auto_approve:
            command.append("--dangerously-skip-permissions")
        if resume:
            if session_id is not None:
                command.extend(("--conversation", session_id))
            else:
                command.append("--continue")
        if message is not None:
            command.extend(("--prompt", message))
        return command

    @staticmethod
    def _event_session_id(event: Mapping[str, Any], payload: Mapping[str, Any]) -> str | None:
        session_id = payload.get("conversation_id") or event.get("conversation_id")
        return str(session_id) if session_id else None

    def _usage_delta(self, session_id: str | None, usage: Mapping[str, Any]) -> dict[str, int]:
        fields = (
            "input_tokens",
            "output_tokens",
            "thinking_tokens",
            "cache_read_tokens",
            "total_tokens",
        )
        current = {
            field: value
            for field in fields
            if isinstance((value := usage.get(field)), int) and value >= 0
        }
        if session_id is None:
            delta = current
        else:
            previous = self._reported_usage.get(session_id, {})
            reset = any(current.get(field, 0) < previous.get(field, 0) for field in current)
            delta = {
                field: value if reset else max(0, value - previous.get(field, 0))
                for field, value in current.items()
            }
            self._reported_usage[session_id] = current
        # Native agy reports uncached and cached prompt tokens separately;
        # harness-wrapper's provider-neutral input count includes both.
        delta["input_tokens"] = delta.get("input_tokens", 0) + delta.get("cache_read_tokens", 0)
        return delta

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        native_type = str(event.get("event", "event"))
        nested = event.get(native_type)
        payload = nested if isinstance(nested, Mapping) else event
        session_id = self._event_session_id(event, payload)

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
                session_id=session_id,
                raw=event,
            )

        if native_type == "init":
            return [make("session", payload)]
        if native_type == "step_update":
            step_type = str(payload.get("step_type", ""))
            if step_type == "agent_response" and payload.get("text_delta") is not None:
                return [make("message", payload.get("text_delta", ""), role="assistant")]
            if step_type != "tool" or payload.get("state") not in {"DONE", "ERROR"}:
                return []
            tool_info = payload.get("tool_info")
            if not isinstance(tool_info, Mapping):
                return []
            tool_name = str(tool_info.get("name") or payload.get("tool_name") or "")
            step_index = payload.get("step_index")
            call_id = f"{session_id or 'antigravity'}:{step_index}"
            result = [
                make(
                    "tool_call",
                    tool_info.get("parameters", {}),
                    role="assistant",
                    tool_name=tool_name,
                    tool_call_id=call_id,
                )
            ]
            # Failed tool output is deliberately omitted from persisted transcripts.
            if tool_info.get("error") is None and tool_info.get("output") is not None:
                result.append(
                    make(
                        "tool_result",
                        tool_info["output"],
                        role="tool",
                        tool_name=tool_name,
                        tool_call_id=call_id,
                    )
                )
            return result
        if native_type == "result":
            result: list[AgentEvent] = []
            usage = payload.get("usage")
            if isinstance(usage, Mapping):
                normalized_usage = self._take_api_usage() or self._usage_delta(session_id, usage)
                result.append(make("usage", normalized_usage))
            if str(payload.get("status", "")).upper() != "SUCCESS":
                result.append(make("error", payload.get("error", payload)))
            # Keep error envelopes too: native validation may recover their
            # fresh answer, and permission denials must remain visible.
            result.append(make("result", payload.get("response")))
            return result
        if native_type == "error":
            return [make("error", payload.get("message", payload))]
        return super().normalize_event(event)


__all__ = ["AntigravityCLIAgent"]
