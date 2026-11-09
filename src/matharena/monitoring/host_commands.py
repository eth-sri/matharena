"""Record command tool boundaries on the host without altering container commands.

These are whole tool-call observations, not kernel process events. Nested shell
commands have no independent durations. Missing IDs/boundaries stay explicit.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from collections.abc import Mapping
from contextlib import contextmanager

from matharena.monitoring.actions import action_event, summarize_actions

SHELL_TOOLS = {
    "bash", "shell", "shell_command", "run_shell_command", "command_execution",
    "exec_command", "execute_command", "run_command", "terminal",
}


def command_text(event) -> str | None:
    content = event.content
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, Mapping):
            content = parsed
    if isinstance(content, Mapping):
        for key in ("command", "cmd", "CommandLine", "command_line"):
            if isinstance(content.get(key), str):
                return content[key]
        for key in ("arguments", "input"):
            nested = content.get(key)
            if isinstance(nested, Mapping):
                for field in ("command", "cmd", "CommandLine", "command_line"):
                    if isinstance(nested.get(field), str):
                        return nested[field]
    tool = (event.tool_name or "").rsplit(".", 1)[-1].lower()
    return content if isinstance(content, str) and tool in SHELL_TOOLS else None


class HostCommands:
    def __init__(self, append):
        self.append = append
        self.events = []
        self.pending = defaultdict(list)
        self.session_id = None
        self.commands = {}
        self.unidentified = 0

    def _record(self, boundary):
        self.events.append(boundary)
        self.append("host-commands.jsonl", boundary)

    def on_event(self, event, observed):
        self.session_id = event.session_id or self.session_id
        boundary = action_event(event, observed, self.session_id)
        if boundary is None:
            return
        key = (boundary["session_id"], boundary["tool_call_id"])
        if boundary["event"] == "start":
            command = command_text(event)
            if command is None:
                return
            span_id = uuid.uuid4().hex
            self.commands[span_id] = command
            if key[1] is not None:
                self.pending[key].append(span_id)
            else:
                self.unidentified += 1
            boundary.update(span_id=span_id, command=command, coverage="whole_tool_call")
            self._record(boundary)
        elif self.pending.get(key):
            # Ambiguous reused IDs must not be paired by adjacency.
            candidates = self.pending[key]
            if len(candidates) != 1:
                self.unidentified += 1
                return
            span_id = candidates.pop()
            boundary.update(span_id=span_id, command=self.commands[span_id], coverage="whole_tool_call")
            self._record(boundary)
        elif event.type == "tool_result" and (event.tool_name or "").rsplit(".", 1)[-1].lower() in SHELL_TOOLS:
            # A completed shell tool without a start is evidence of a coverage gap.
            boundary.update(span_id=uuid.uuid4().hex, command=None, coverage="whole_tool_call")
            self._record(boundary)

    @contextmanager
    def lifecycle(self, command, phase):
        span_id = uuid.uuid4().hex
        def boundary(action):
            return {"event": action, "session_id": "host_lifecycle", "tool_call_id": span_id,
                    "span_id": span_id, "tool_name": "sandbox.run", "phase": phase,
                    "command": command, "coverage": "whole_sandbox_invocation",
                    "timestamp_source": "host_observation", "wall_time_ns": time.time_ns(),
                    "observed_monotonic_ns": time.monotonic_ns()}
        self._record(boundary("start"))
        result = {"returncode": None}
        try:
            yield result
        except BaseException as exc:
            result["error_type"] = type(exc).__name__
            raise
        finally:
            self._record({**boundary("end"), **result})

    def summary(self):
        summary = summarize_actions(self.events)
        for span in summary["spans"]:
            event = span["start"] or span["end"]
            span.update(command=event.get("command"), phase=event["phase"], coverage=event["coverage"])
            end = span["end"] or {}
            span.update(returncode=end.get("returncode", end.get("exit_code")), tool_is_error=end.get("tool_is_error"))
        summary.update(
            complete=summary["complete"] and not self.unidentified,
            source="host_harness_events", coverage="whole_tool_calls_and_sandbox_invocations",
            nested_process_timings=False, unidentified_boundaries=self.unidentified,
        )
        return summary
