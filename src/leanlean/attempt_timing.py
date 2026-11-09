"""Recover model boundaries while retaining evidence of stalled runtime tails."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path


def rows(path: Path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)


def model_window(directory: Path, observations=None, events=None) -> tuple[int, int, dict | None]:
    """Exclude only a terminal-result tail corroborated by runtime failures.

    Missing samples alone never justify shortening a run. Later model requests,
    commands, source changes or resource samples veto a correction. This is an
    observed harness-result boundary, not a claim of successful completion or
    a measurement of model compute time. Original telemetry is never rewritten.
    """
    observations = list(rows(directory / "observations.jsonl")) if observations is None else observations
    bounds = {r["reason"]: r for r in observations if r["reason"] in {"model_start", "model_end"}}
    start, end = (bounds[key]["started_monotonic_ns"] for key in ("model_start", "model_end"))
    events = list(rows(directory / "harness-events.jsonl")) if events is None else events
    if not events or events[-1]["event"].get("type") != "result":
        return start, end, None
    terminal = events[-1]
    cutoff = terminal["observed_monotonic_ns"]
    if not start <= cutoff < end or end - cutoff < 60 * 10**9:
        return start, end, None
    # Allow clock/observation scheduling skew of at most two seconds.
    latest = cutoff + 2 * 10**9
    samples = [r for r in rows(directory / "resources.jsonl")
               if r.get("phase") == "model" and cutoff < r["observed_monotonic_ns"] <= end]
    timeouts = sum(
        (r.get("status") == "retrying" and r.get("reason") == "container_inspect_timeout")
        or (r.get("status") == "error" and "TimeoutExpired" in r.get("error", "")
            and "inspect" in r.get("error", ""))
        for r in samples
    )
    if timeouts < 2 or any(r.get("status") == "sample" and
                          r.get("sampled_monotonic_ns", r["observed_monotonic_ns"]) > latest for r in samples):
        return start, end, None
    if any(r.get("phase") == "model" and latest < r["monotonic_ns"] <= end
           for r in rows(directory / "commands.jsonl")):
        return start, end, None
    if any(r.get("original_changed", r.get("changed")) and latest < r["started_monotonic_ns"] <= end for r in observations):
        return start, end, None
    # Request capture can reveal work that never reached the harness event log.
    wall_cutoff = terminal.get("observed_wall_ns")
    if wall_cutoff is None:
        return start, end, None
    for path in (directory / "native").rglob("model_requests.jsonl"):
        for record in rows(path):
            event = record.get("event") or {}
            if event.get("type") != "model.request":
                continue
            try:
                stamp = datetime.fromisoformat(event["timestamp"].replace("Z", "+00:00")).timestamp() * 1e9
            except (KeyError, TypeError, ValueError, AttributeError):
                return start, end, None  # Unverifiable request timing is not evidence of inactivity.
            if stamp > wall_cutoff + 2 * 10**9:
                return start, end, None
    result = (terminal["event"].get("raw") or {}).get("result") or {}
    return start, cutoff, dict(
        reason="Runtime stalled after final harness result; no later model requests, commands, source changes or resource samples",
        recorded_elapsed_s=(end - start) / 1e9,
        excluded_runtime_s=(end - cutoff) / 1e9,
        terminal_status=result.get("status"),
        resource_inspect_timeouts=timeouts,
        end_wall_ns=wall_cutoff,
    )
