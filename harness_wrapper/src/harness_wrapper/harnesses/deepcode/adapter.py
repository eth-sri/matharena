"""Deep Code's native --exec mode, with proxied API requests and saved transcripts."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

from ...agent import Agent, AgentEvent
from ...models.request_capture import RequestCaptureProxy
from ...tools import CLIInstallation, CLIProcessError
from ...traces import TokenUsage

_LOCAL_TOOLS = frozenset({"bash", "read", "write", "edit", "UpdatePlan"})
_MINIMAL_PROMPT = (
    "Solve the user's task using mathematical reasoning and, when useful, the available "
    "local file and bash tools. The writable working directory is /work. "
    "There is no internet access. Do not use web search, remote APIs, skills, MCP, "
    "or subagents. Read installed libraries outside /work when needed. "
    "For bash calls, accurately declare sideEffects using the tool schema."
)
_BUNDLED_SKILLS = ("deepcode-self-refer", "image-generator", "skill-digester", "skill-writer")


class DeepCodeAgent(Agent):
    harness_name = "deepcode"
    aliases = ("deepcode-cli", "deep-code")
    accepted_api_types = ("openai",)
    installation = CLIInstallation(executable="deepcode", package="@vegamo/deepcode-cli")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._native_seen: set[str] = set()
        self._provider_usage = TokenUsage()

    @property
    def _private_home(self) -> Path:
        return self.root / ".harness-home"

    def _disabled_skills(self) -> dict[str, bool]:
        names = set(_BUNDLED_SKILLS)
        roots = [
            base / folder / "skills"
            for base in (self.root, self._private_home)
            for folder in (".deepcode", ".agents")
        ]
        # Include new bundled skills when an unpinned/new CLI version is installed.
        for mount in getattr(self.env, "mounts", ()):
            roots.append(Path(mount.source) / "lib/node_modules/@vegamo/deepcode-cli/dist/bundled")
        for root in roots:
            for path in root.glob("*/SKILL.md"):
                match = re.search(
                    r"^name:\s*([^\r\n]+)", path.read_text(encoding="utf-8"), re.MULTILINE
                )
                names.add(match.group(1).strip(" \"'") if match else path.parent.name)
        return dict.fromkeys(sorted(names), False)

    def model_environment(self) -> dict[str, str]:
        environment = super().model_environment()
        endpoint = self.model.endpoint_for("openai")
        settings: dict[str, Any] = {
            "telemetryEnabled": False,
            "filesApiEnabled": False,
            "mcpServers": {},
            "permissions": {
                "defaultMode": "allowAll",
                "allow": [],
                "ask": [],
                "deny": ["mcp"] if self.internet_enabled else ["network", "mcp"],
            },
        }
        if self.minimal_context:
            settings["enabledSkills"] = self._disabled_skills()
        settings_path = self._private_home / ".deepcode/settings.json"
        settings_path.parent.mkdir(parents=True, exist_ok=True)
        settings_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
        runtime_root = getattr(self.env, "container_root", self.root)
        environment.update(
            {
                "HOME": str(runtime_root / ".harness-home"),
                "DEEPCODE_MODEL": self.model_name,
                "DEEPCODE_BASE_URL": getattr(self, "_request_capture_url", endpoint.url),
                "DEEPCODE_API_KEY": endpoint.api_key,
                "DEEPCODE_TELEMETRY_ENABLED": "0",
            }
        )
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            environment["DEEPCODE_THINKING_ENABLED"] = "false" if reasoning == "none" else "true"
            environment["DEEPCODE_REASONING_EFFORT"] = str(reasoning)
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
            raise ValueError("Deep Code requires a prompt")
        command = [
            self.executable,
            "--exec",
            "--prompt",
            message or "Continue the interrupted task from where it stopped.",
        ]
        if resume:
            command.extend(("--resume", session_id) if session_id else ("--last",))
        return command

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        # --exec emits arbitrary final-answer text, not JSON events. In particular,
        # a JSON-looking answer must never be interpreted as a native event.
        return []

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if not path.rstrip("/").endswith("/chat/completions"):
            raise ValueError("Deep Code may only access Chat Completions through this proxy")
        # Deep Code 0.3.1 puts reasoning_effort inside a Python-SDK-style
        # extra_body even though its JS SDK transmits that wrapper literally.
        # Flatten it, with MathArena's top-level model-config overrides winning.
        extra = payload.pop("extra_body", None)
        if extra is not None:
            if not isinstance(extra, dict):
                raise ValueError("extra_body must be an object")
            for key, value in extra.items():
                payload.setdefault(key, value)
        if not self.minimal_context:
            return
        tools = payload.get("tools")
        if isinstance(tools, list):
            payload["tools"] = [
                tool
                for tool in tools
                if isinstance(tool, dict)
                and tool.get("type") == "function"
                and (
                    tool.get("function", {}).get("name") in _LOCAL_TOOLS
                    or self.internet_enabled and tool.get("function", {}).get("name") == "WebSearch"
                )
            ]
        for message in payload.get("messages", []):
            if (
                isinstance(message, dict)
                and message.get("role") == "system"
                and "# Available Tools" in str(message.get("content", ""))
            ):
                message["content"] = self.network_instructions(_MINIMAL_PROMPT)

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        self._touch()
        super()._capture_model_response(path, payload)
        raw_usage = payload.get("usage")
        if not isinstance(raw_usage, Mapping):
            return
        usage = self._normalize_token_usage(raw_usage)
        if usage is None:
            return
        # DeepSeek reports cache hits as a top-level field, not OpenAI details.
        cached = self._usage_count(raw_usage, "prompt_cache_hit_tokens") or usage.cache_read_tokens
        with self._tokens_lock:
            old = self._provider_usage
            self._provider_usage = TokenUsage(
                input_tokens=old.input_tokens + usage.input_tokens,
                output_tokens=old.output_tokens + usage.output_tokens,
                cache_read_tokens=old.cache_read_tokens + cached,
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

    def _native_events(self, *, require_completed: bool = False) -> Iterator[AgentEvent]:
        entries = []
        for path in (self._private_home / ".deepcode/projects").glob("*/sessions-index.json"):
            index = json.loads(path.read_text(encoding="utf-8"))
            entries.extend((entry, path.parent) for entry in index.get("entries", []))
        if not entries:
            if require_completed:
                raise RuntimeError("Deep Code exited without a saved native session")
            return
        entry, directory = max(entries, key=lambda item: item[0].get("updateTime", ""))
        session_id = str(entry["id"])
        yield AgentEvent(
            type="session",
            session_id=session_id,
            raw={"source": "native_session", "status": entry.get("status")},
        )
        path = directory / f"{session_id}.jsonl"
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                native = json.loads(line)
                message_id = str(native["id"])
                if message_id in self._native_seen:
                    continue
                self._native_seen.add(message_id)
                params = native.get("messageParams") or {}
                role = native.get("role")
                content = native.get("content")
                common = {"session_id": session_id, "raw": native}
                # The proxy records the exact post-filter system context, while
                # the wrapper already records each submitted user prompt.
                if role == "assistant":
                    if params.get("reasoning_content"):
                        yield AgentEvent(
                            type="reasoning",
                            content=params["reasoning_content"],
                            role="assistant",
                            **common,
                        )
                    if content:
                        yield AgentEvent(
                            type="message", content=content, role="assistant", **common
                        )
                    for call in params.get("tool_calls") or []:
                        function = call.get("function") or {}
                        arguments = function.get("arguments", "")
                        with suppress(ValueError, TypeError):
                            arguments = json.loads(arguments)
                        yield AgentEvent(
                            type="tool_call",
                            content=arguments,
                            role="assistant",
                            tool_name=function.get("name"),
                            tool_call_id=call.get("id"),
                            **common,
                        )
                elif role == "tool":
                    yield AgentEvent(
                        type="tool_result",
                        content=content,
                        role="tool",
                        tool_call_id=params.get("tool_call_id"),
                        **common,
                    )
        if require_completed:
            if entry.get("status") != "completed":
                raise RuntimeError(
                    f"Deep Code native session did not complete: {entry.get('status')}"
                )
            yield AgentEvent(
                type="result",
                content=entry.get("assistantReply", ""),
                role="assistant",
                session_id=session_id,
                raw={"source": "native_session", "status": "completed"},
            )

    def _emit_native_events(self, *, require_completed: bool = False) -> Iterator[AgentEvent]:
        for event in self._native_events(require_completed=require_completed):
            self._observe(event)
            yield event

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        if getattr(self.model, "auth_mode", None) != "api":
            raise ValueError("Deep Code currently supports API-credit auth only")
        endpoint = self.model.endpoint_for("openai")
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
                try:
                    yield from super()._stream_once(
                        message, resume=resume, session_id=session_id, last=last
                    )
                except CLIProcessError:
                    # Record the native session ID before recovery chooses resume.
                    yield from self._emit_native_events()
                    raise
                else:
                    yield from self._emit_native_events(require_completed=True)
            finally:
                del self._request_capture_url
