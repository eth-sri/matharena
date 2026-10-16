"""Muse Code's headless JSONL interface and Meta Responses API transport."""

from __future__ import annotations

import io
import json
import tempfile
import time
import urllib.error
import urllib.request
import urllib.response
from collections.abc import Iterator, Mapping
from contextlib import suppress
from queue import Empty, SimpleQueue
from threading import local
from typing import Any
from urllib.parse import urljoin, urlsplit

from loguru import logger

from ...agent import Agent, AgentEvent, OutputTokenLimitError
from ...models.request_capture import RequestCaptureProxy, ResponseCapture, _drop_fields, response_chunks
from ...tools import CLIInstallation, CLIProcessError

_LOCAL_TOOLS = frozenset(
    {"bash", "bash_input", "read_file", "search", "write_file", "edit_file", "write_todos"}
)
_WEB_TOOLS = frozenset({"web_search", "web_search_preview", "web_fetch", "server_web_fetch"})
_MINIMAL_PROMPT = (
    "Solve the user's task using mathematical reasoning and, when useful, the available "
    "local file and shell tools. There is no internet access. Do not use web search, "
    "remote APIs, skills, MCP, or subagents."
)


def _select_tools(tools: list[Any], names: frozenset[str]) -> list[dict[str, Any]]:
    kept = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") == "namespace":
            children = _select_tools(tool.get("tools", []), names)
            if children:
                kept.append({**tool, "tools": children})
        elif tool.get("type") == "function" and tool.get("name") in names:
            kept.append(tool)
        elif tool.get("type") in _WEB_TOOLS and tool["type"] in names:
            kept.append(tool)
    return kept


class _MuseTokenLimitError(OutputTokenLimitError):
    """A truncated terminal turn that can continue in Muse's native session."""


class MuseCodeAgent(Agent):
    harness_name = "muse-code"
    aliases = ("muse", "muse-cli")
    accepted_api_types = ("openai",)
    installation = CLIInstallation(executable="muse", package="meta/muse-code")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._provider_events: SimpleQueue[AgentEvent] = SimpleQueue()
        self._seen_items: set[str] = set()
        self._seen_usage: set[str] = set()
        # The proxy transforms and observes each HTTP request on the same thread.
        # Compaction can run concurrently with the main solving request.
        self._request_context = local()
        self._solver_output_budget: int | None = None
        self._last_solver_response_limited = False
        self._last_solver_final_text = ""
        self._resume_catalog: dict[str, Any] | None = None

    def model_environment(self) -> dict[str, str]:
        endpoint = self.model.endpoint_for("openai")
        runtime_root = getattr(self.env, "container_root", self.root)
        home = runtime_root / ".harness-home"
        settings = {
            "schema_version": 1,
            "endpoint_transport": {
                "base_url": getattr(self, "_request_capture_url", endpoint.url),
                "auth": "bearer",
            },
            "telemetry": {"enabled": False},
            "feature_config": {"enabled": False},
        }
        if self.minimal_context:
            settings["run"] = {"reminder_roster": []}
        path = self.root / ".harness-home/.config/muse/settings.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings), encoding="utf-8")
        return {
            "META_API_KEY": endpoint.api_key,
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(home / ".config"),
            "XDG_DATA_HOME": str(home / ".local/share"),
            "XDG_STATE_HOME": str(home / ".local/state"),
            # Prevent the CLI from silently lowering the first turn's effort.
            "MUSE_EXPERIMENTAL_FIRST_TURN_MINIMAL_EFFORT": "0",
            # Long reasoning may produce no model events for several minutes.
            "TBH_STREAM_FIRST_EVENT_TIMEOUT_SECS": "28800",
            "TBH_STREAM_IDLE_TIMEOUT_SECS": "28800",
        }

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
        if resume and not session_id:
            raise ValueError("Muse continuation requires a saved session id")
        if resume:
            command = [self.executable, "serve"]
            if self.env is not None and getattr(self.env, "enabled", False):
                command.append("--disable-sandbox")
            return command
        command = [
            self.executable,
            "exec",
            "--json",
            "--provider",
            "meta",
            "--model",
            self.model_name,
            "--approval-judge",
            "off",
        ]
        reasoning = getattr(self.model, "reasoning", None)
        if reasoning is not None:
            command.extend(("--reasoning-effort", str(reasoning)))
        if self.minimal_context:
            command.append("--no-foreign-personal-context")
            if not self.internet_enabled:
                command.append("--disable-web-tools")
        if self.env is not None and getattr(self.env, "enabled", False):
            # Docker/Podman supplies the outer filesystem and network boundary.
            command.extend(("--disable-approval", "--disable-sandbox"))
        command.extend(("--", message or "Continue the interrupted task."))
        return command

    def _filter_request(self, path: str, payload: dict[str, Any]) -> None:
        if not path.rstrip("/").endswith("/responses"):
            raise ValueError("Muse may only access Responses through this proxy")
        summary_tools = _select_tools(payload.get("tools", []), frozenset({"generate_summary"}))
        is_compaction = bool(summary_tools)
        self._request_context.is_compaction = is_compaction
        overrides = self.model.request_overrides()
        if is_compaction:
            # Muse owns the summary schema, prompt, reasoning, and output budget.
            # Applying the solver's max effort/budget here can exhaust the native
            # five-minute summary timeout. Transport/storage policy still applies.
            overrides = {
                key: value for key, value in overrides.items() if key in {"store", "include"}
            }
        payload.update(overrides)
        _drop_fields(payload, self.model.request_drop_fields())
        if not is_compaction and self._solver_output_budget is not None:
            budget = payload.get("max_output_tokens")
            if isinstance(budget, int) and budget > self._solver_output_budget:
                payload["max_output_tokens"] = self._solver_output_budget
        payload["model"] = self.model_name
        if is_compaction:
            # generate_summary is an internal structured result, not a shell or
            # external tool. Never expose the solver's tool set in this request.
            payload["tools"] = summary_tools
        elif self.minimal_context:
            tools = _LOCAL_TOOLS | (_WEB_TOOLS if self.internet_enabled else frozenset())
            payload["tools"] = _select_tools(payload.get("tools", []), tools)
            payload["instructions"] = self.network_instructions(_MINIMAL_PROMPT)

    def _resume_model_catalog(self) -> dict[str, Any]:
        # Unlike exec with an explicit model, serve needs a catalog entry to
        # initialize its provider. Fetch metadata only, once per Agent; expose
        # only the configured model so a catalog default cannot switch models.
        if self._resume_catalog is None:
            endpoint = self.model.endpoint_for("openai")
            # MathArena keeps the provider key in headers; api_key can be a
            # sandbox-only placeholder. Preserve upstream auth as the proxy does.
            headers = dict(endpoint.headers)
            if not any(name.lower() == "authorization" for name in headers):
                headers["Authorization"] = f"Bearer {endpoint.api_key}"
            request = urllib.request.Request(
                urljoin(endpoint.url, "/muse-code/models"),
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    catalog = json.load(response)
                rows = [row for row in catalog["data"]
                        if isinstance(row, Mapping) and row.get("id") == self.model_name]
                if len(rows) != 1:
                    raise ValueError("configured model missing or duplicated in Muse catalog")
            except (OSError, ValueError, KeyError, TypeError) as error:
                raise CLIProcessError(
                    [self.executable, "serve"], 1,
                    "Could not load the configured model from Muse catalog: "
                    + (f"HTTP {error.code}" if isinstance(error, urllib.error.HTTPError)
                       else type(error).__name__),
                ) from error
            self._resume_catalog = {"object": "list", "data": rows}
        return self._resume_catalog

    def _open_budgeted_request(self, request: urllib.request.Request, *, timeout: float) -> Any:
        """Fit the output reservation without dropping input or restarting the turn.

        Meta returns this generic 400 when input plus max_output_tokens exceeds
        the context window. Rejected requests have generated no output. Other
        invalid requests still fail after the bounded budget retries.
        """
        path = urlsplit(request.full_url).path
        for attempt in range(self.max_recovery_attempts + 1):
            try:
                return urllib.request.urlopen(request, timeout=timeout)
            except urllib.error.HTTPError as error:
                if error.code != 400 or getattr(self._request_context, "is_compaction", False):
                    raise
                body = error.read()
                error.close()
                try:
                    details = json.loads(body).get("error", {})
                    payload = json.loads(request.data or b"{}")
                    budget = payload.get("max_output_tokens")
                    reduce_budget = (
                        details.get("type") == "invalid_request_error"
                        and details.get("message") == "The request contains invalid parameters. Check the request body for any errors or inconsistencies."
                        and isinstance(budget, int) and budget > 1
                        and attempt < self.max_recovery_attempts
                    )
                except (ValueError, AttributeError):
                    reduce_budget = False
                if not reduce_budget:
                    raise urllib.error.HTTPError(
                        error.url, error.code, error.msg, error.headers, io.BytesIO(body)
                    ) from error
                payload["max_output_tokens"] = budget // 2
                self._solver_output_budget = budget // 2
                logger.warning(
                    "Meta rejected the request; retrying with max_output_tokens reduced from {} to {}",
                    budget, budget // 2,
                )
                self._provider_events.put(AgentEvent(type="recovery", role="meta", content={
                    "reason": "output_budget_reduced", "attempt": attempt + 1,
                    "retry_delay_seconds": 60.0,
                    "previous_max_output_tokens": budget, "max_output_tokens": budget // 2,
                }))
                request.data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                request.remove_header("Content-length")
                # Record the actual retried body as well as the rejected one.
                self._record_model_request(path, payload)
                time.sleep(60.0)

    def _open_model_request(self, request: urllib.request.Request, *, timeout: float) -> Any:
        """Retry failed generations without exposing their partial streams to Muse.

        A failure can arrive after text or tool-call deltas. Spool each response
        before forwarding so retries cannot duplicate those events in the native
        conversation. Server-error retries preserve the request bytes and native
        turn; context-budget rejection may first reduce the output reservation.
        Large response bodies spill to disk.
        """
        if not self.max_recovery_attempts:
            return urllib.request.urlopen(request, timeout=timeout)
        path = urlsplit(request.full_url).path
        for attempt in range(self.max_recovery_attempts + 1):
            # Budget rejection is handled before buffering. Other HTTP failures
            # remain on Muse's native retry path; retry SSE server_error here.
            upstream = self._open_budgeted_request(request, timeout=timeout)
            if "text/event-stream" not in upstream.headers.get("content-type", "").lower():
                return upstream
            failed = None

            def inspect(_path: str, payload: Mapping[str, Any]) -> None:
                nonlocal failed
                if payload.get("type") in {"response.failed", "response.completed", "response.incomplete"}:
                    failed = payload if payload.get("type") == "response.failed" else None

            buffered = tempfile.SpooledTemporaryFile(max_size=1024 * 1024)
            try:
                with upstream:
                    headers, status = upstream.headers, upstream.status
                    capture = ResponseCapture(path, headers.get("content-type", ""), inspect,
                                              component="muse_request_retry")
                    for chunk in response_chunks(upstream):
                        self._touch()
                        buffered.write(chunk)
                        capture.feed(chunk)
                    capture.finish()
                error_code = ((failed or {}).get("response", {}).get("error") or {}).get("code")
                if error_code != "server_error" or attempt == self.max_recovery_attempts:
                    buffered.seek(0)
                    return urllib.response.addinfourl(buffered, headers, request.full_url, status)
                # Failed requests still cost tokens. Their partial output items
                # and tool calls never reach the native CLI.
                self._capture_model_response(path, failed)
            except BaseException:
                buffered.close()
                raise
            buffered.close()
            delay = 60.0
            self._provider_events.put(AgentEvent(type="recovery", role="meta", content={
                "reason": "provider_request_retry", "attempt": attempt + 1,
                "retry_delay_seconds": delay,
            }))
            time.sleep(delay)

    def _recover_cli_failure(self, error: CLIProcessError, *, tried_models: set[int]) -> str | None:
        if isinstance(error, _MuseTokenLimitError):
            if self.session_id is not None:
                return "token_limit_resume"
            return None
        return super()._recover_cli_failure(error, tried_models=tried_models)

    def _recovery_message(self, recovery: str) -> str:
        # An empty text part starts a native turn without adding a model-visible
        # user message. Muse supplies its own previous-turn placeholder.
        return "" if recovery == "token_limit_resume" else super()._recovery_message(recovery)

    def _capture_model_response(self, path: str, payload: Mapping[str, Any]) -> None:
        super()._capture_model_response(path, payload)
        response = payload.get("response", {})
        if not isinstance(response, Mapping):
            response = {}
        is_compaction = getattr(self._request_context, "is_compaction", False)
        if payload.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
            if not is_compaction:
                self._last_solver_final_text = "".join(
                    block.get("text", "")
                    for item in response.get("output", [])
                    if isinstance(item, Mapping) and item.get("type") == "message"
                    and item.get("phase") != "commentary"
                    for block in item.get("content", [])
                    if isinstance(block, Mapping) and block.get("type") == "output_text"
                ) if payload.get("type") == "response.completed" else ""
                self._last_solver_response_limited = (
                    payload.get("type") == "response.incomplete"
                    and (response.get("incomplete_details") or {}).get("reason")
                    == "max_output_tokens"
                )
            response_id = str(response.get("id", ""))
            usage = response.get("usage")
            if isinstance(usage, Mapping) and response_id not in self._seen_usage:
                self._seen_usage.add(response_id)
                # Capture immediately: a subsequent CLI error must not lose usage.
                self._observe(AgentEvent(type="usage", content=usage, raw=payload))
        if is_compaction:
            return
        item = payload.get("item")
        if payload.get("type") == "response.output_item.done" and isinstance(item, Mapping):
            item_id = str(item.get("id", ""))
            if item_id in self._seen_items:
                return
            self._seen_items.add(item_id)
            if item.get("type") == "function_call":
                arguments = item.get("arguments", "{}")
                with suppress(ValueError, TypeError):
                    arguments = json.loads(arguments)
                self._provider_events.put(
                    AgentEvent(
                        type="tool_call",
                        content=arguments,
                        role="assistant",
                        tool_name=str(item.get("name", "")).removeprefix("muse."),
                        tool_call_id=item.get("call_id"),
                        raw=payload,
                    )
                )
            elif item.get("type") == "message" and item.get("phase") == "commentary":
                text = "".join(
                    block.get("text", "")
                    for block in item.get("content", [])
                    if isinstance(block, Mapping)
                )
                self._provider_events.put(
                    AgentEvent(
                        type="message",
                        content=text,
                        role="assistant",
                        raw=payload,
                    )
                )

    def _drain_provider_events(self) -> list[AgentEvent]:
        events = []
        while True:
            try:
                events.append(self._provider_events.get_nowait())
            except Empty:
                return events

    def normalize_event(self, event: Mapping[str, Any]) -> list[AgentEvent]:
        events = self._drain_provider_events()
        payload = event.get("payload", {})
        stream = event.get("stream", {})
        if not isinstance(payload, Mapping) or not isinstance(stream, Mapping):
            return events
        session_id = stream.get("id") if stream.get("kind") == "session" else None
        kind = event.get("payload_type")
        fields: dict[str, Any] = {"type": "event", "content": payload}
        if kind == "runtime.command.accepted":
            fields["type"] = "session"
        elif kind == "tool.result":
            fields.update(
                type="tool_result",
                role="tool",
                content=payload.get("text", ""),
                tool_call_id=payload.get("call_id"),
                tool_name=payload.get("correlation_facts", {}).get("tool_name"),
            )
        elif kind == "run.terminal.completed":
            fields.update(type="result", content=payload.get("text", ""))
        elif str(kind).startswith("run.terminal."):
            fields.update(type="error", content=payload.get("reason") or payload)
        # Retain native status/delta events without treating partial text as a final answer.
        events.append(AgentEvent(**fields, session_id=session_id, raw=event))
        return events

    def _stream_once(
        self,
        message: str | None,
        *,
        resume: bool,
        session_id: str | None = None,
        last: bool = False,
    ) -> Iterator[AgentEvent]:
        self._last_solver_response_limited = False
        self._last_solver_final_text = ""
        endpoint = self.model.endpoint_for("openai")
        with (
            self._host_service_route() as route,
            RequestCaptureProxy(
                endpoint.url,
                self._capture_model_request,
                upstream_headers=endpoint.headers,
                on_response=self._capture_model_response,
                open_upstream=self._open_model_request,
                # Apply overrides inside the transform, after identifying
                # compaction; the original native settings must remain available.
                request_transform=self._filter_request,
                # Serve requires valid catalog metadata to initialize its provider.
                # Expose only the configured model on the native product route.
                static_get_responses={"/muse-code/models": (
                    self._resume_model_catalog() if resume else {"object": "list", "data": []}
                )},
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
                completed = False
                if resume:
                    if not session_id:
                        raise ValueError("Muse continuation requires a saved session id")
                    from .msp import resume_session

                    stream = resume_session(self, message, session_id)
                else:
                    stream = super()._stream_once(
                        message, resume=False, session_id=session_id, last=last,
                    )
                for event in stream:
                    if resume:
                        self._observe(event)
                    if event.type == "result":
                        completed = True
                    yield event
                if self._last_solver_response_limited:
                    raise _MuseTokenLimitError(
                        [self.executable, "exec"], 1,
                        "Muse exhausted max_output_tokens before completing its response; "
                        "continuing requires the saved native session",
                    )
                if not completed:
                    raise CLIProcessError(
                        [self.executable, "exec"], 1, "Muse ended without a completed final answer"
                    )
            finally:
                del self._request_capture_url
                for event in self._drain_provider_events():
                    self._observe(event)
                    yield event
