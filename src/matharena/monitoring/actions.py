"""Pair agent actions without presenting stream receipt times as execution times."""

from __future__ import annotations

import math
from collections import defaultdict


def action_event(event, observed: dict, session_id: str | None) -> dict | None:
    raw = event.raw
    item = raw.get("item") or {}
    native_type = raw.get("type") or raw.get("event")
    if event.type in {"tool_call", "tool_result"}:
        boundary = "start" if event.type == "tool_call" else "end"
    elif event.type == "file_change" and native_type in {"item.started", "item.completed"}:
        boundary = "start" if native_type == "item.started" else "end"
    elif event.type == "file_change" and native_type not in {"item.started", "item.updated"}:
        boundary = "end"
    else:
        return None
    result = {
        **observed,
        "event": boundary,
        "session_id": event.session_id or session_id,
        "tool_call_id": event.tool_call_id or item.get("id"),
        "tool_name": event.tool_name or item.get("type") or event.type,
        "timestamp_source": "host_observation",
        "wall_time_ns": observed["observed_wall_ns"],
        "native_event_type": native_type,
        "native_status": item.get("status"),
        "exit_code": item.get("exit_code"),
    }
    if event.type == "tool_result":
        message = raw.get("message") or {}
        blocks = message.get("content", []) if isinstance(message, dict) else []
        for block in blocks if isinstance(blocks, list) else []:
            if (
                isinstance(block, dict)
                and event.tool_call_id is not None
                and block.get("tool_use_id") == event.tool_call_id
                and isinstance(block.get("is_error"), bool)
            ):
                result["tool_is_error"] = block["is_error"]
    if native_type == "step_update":
        step = raw.get("step_update") or {}
        if step.get("step_type") == "tool" and step.get("state") in {"DONE", "ERROR"}:
            # Antigravity delivers both normalized boundaries at completion.
            # Its native duration is useful; callback arrival spacing is not.
            duration = step.get("duration_seconds")
            valid = (
                isinstance(duration, (int, float)) and not isinstance(duration, bool)
                and math.isfinite(duration) and duration >= 0
            )
            result.update(timestamp_source="completion_observation",
                          native_duration_s=duration if valid else None,
                          native_status=step["state"])
    # OpenCode may emit both normalized boundaries from one completed message.
    # Use its explicit execution times when supplied; never invent a zero-length
    # action from two callbacks created at completion.
    if native_type == "tool_use":
        part = raw.get("part") or {}
        state = part.get("state") or {}
        timing = state.get("time") or {}
        stamp = timing.get(boundary)
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool) and stamp > 0:
            result.update(wall_time_ns=int(stamp * 1_000_000), timestamp_source="native_execution")
        elif boundary == "start" and state.get("status") in {"completed", "error"}:
            return None
    return result


def summarize_actions(events: list[dict]) -> dict:
    pending = defaultdict(list)
    rows = []
    for event in events:
        key = (event["session_id"], event["tool_call_id"])
        if event["event"] == "start":
            row = {"action_index": len(rows) + 1, "start": event, "end": None}
            rows.append(row)
            if key[1] is not None:
                pending[key].append(row)
        elif key[1] is not None and len(pending[key]) == 1:
            pending[key].pop()["end"] = event
        else:
            # Missing or ambiguous IDs must not be paired by adjacency/time.
            rows.append({"action_index": len(rows) + 1, "start": None, "end": event})
    for row in rows:
        start, end = row["start"], row["end"]
        event = start or end
        row.update(session_id=event["session_id"], tool_call_id=event["tool_call_id"], tool_name=event["tool_name"])
        row["started_wall_ns"] = start["wall_time_ns"] if start else None
        row["ended_wall_ns"] = end["wall_time_ns"] if end else None
        row["duration_s"] = None
        if not start or not end:
            row["status"] = "missing_start" if not start else "missing_end"
        elif start["timestamp_source"] != end["timestamp_source"]:
            row["status"] = "mixed_clock_sources"
        elif start["timestamp_source"] == "completion_observation":
            duration = end.get("native_duration_s")
            row.update(started_wall_ns=None, ended_wall_ns=None,
                       observed_completed_wall_ns=end["observed_wall_ns"],
                       timestamp_source="native_reported_duration" if duration is not None else "completion_observation",
                       duration_s=duration, status="complete" if duration is not None else "duration_unavailable")
        else:
            row["timestamp_source"] = start["timestamp_source"]
            elapsed = (
                end["observed_monotonic_ns"] - start["observed_monotonic_ns"]
                if start["timestamp_source"] == "host_observation"
                else end["wall_time_ns"] - start["wall_time_ns"]
            )
            row["status"] = "complete" if elapsed >= 0 else "invalid_clock_order"
            if elapsed >= 0:
                row["duration_s"] = elapsed / 1e9
    return {"spans": rows, "complete": all(row["status"] == "complete" for row in rows)}
