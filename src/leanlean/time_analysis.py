"""Partition model-attempt time using command, action, and native reasoning spans.

Reasoning is the CLI's reported item lifetime, not a measurement of server GPU
time. Unobserved gaps remain unclassified. Overlapping reasoning/tool execution
has its own category, and nested Lake work is attributed to Comparator first.
"""

from __future__ import annotations

import argparse
import bisect
import json
import re
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path

from leanlean.common import read_json, sha256, write_json
from leanlean.attempt_timing import model_window
from leanlean.lake_analysis import operation, read_lines, write_csv
from matharena.monitoring.attempt import summarize_commands

CATEGORIES = (
    "comparator",
    "build",
    "lean_check",
    "measure",
    "other_tools",
    "reasoning",
    "response",
    "parallel",
    "retry_wait",
    "unclassified",
)


def standalone_measurement(event: dict) -> bool:
    """Only attribute a shell span to measurement when it runs that alone."""
    argv = event["argv"]
    return (
        event["kind"] in {"sh", "bash"}
        and len(argv) == 3
        and argv[1] in {"-c", "-lc"}
        and re.fullmatch(r"\s*python3?\s+(?:/work/|\./)?proof_length\.py\s*", argv[2]) is not None
    )


def partition(start: int, end: int, intervals: list[tuple[int, int, str]]) -> dict[str, float]:
    """Sweep clipped intervals so every instant is assigned exactly once."""
    events = defaultdict(Counter)
    events[start]
    events[end]
    for left, right, kind in intervals:
        if right < left:
            raise ValueError("Invalid clock order")
        left, right = max(left, start), min(right, end)
        if left < right:
            events[left][kind] += 1
            events[right][kind] -= 1
    active, totals = Counter(), dict.fromkeys(CATEGORIES, 0)
    previous = start
    for timestamp, changes in sorted(events.items()):
        tool = next((k for k in ("comparator", "build", "lean_check", "measure", "other_tools") if active[k] > 0), None)
        category = (
            "parallel"
            if tool and active["reasoning"] > 0
            else tool
            or (
                "reasoning" if active["reasoning"] > 0 else "retry_wait" if active["retry_wait"] > 0
                else "response" if active["response"] > 0 else "unclassified"
            )
        )
        totals[category] += timestamp - previous
        active.update(changes)
        previous = timestamp
    if sum(totals.values()) != end - start:
        raise ValueError("Time partition does not reconcile")
    return {key: value / 1e9 for key, value in totals.items()}


def reasoning_intervals(directory: Path, events: list[dict]) -> tuple[list[tuple[int, int, str]], dict]:
    observations = sorted((e["observed_wall_ns"], e["observed_monotonic_ns"]) for e in events)
    walls = [wall for wall, _ in observations]

    def monotonic(wall: int) -> int:
        i = min(bisect.bisect_left(walls, wall), len(walls) - 1)
        if i and abs(walls[i - 1] - wall) < abs(walls[i] - wall):
            i -= 1
        observed_wall, observed_mono = observations[i]
        return wall - observed_wall + observed_mono

    spans, seen, invalid, files = [], set(), 0, []
    for path in sorted((directory / "native").rglob("rollout-*.jsonl")):
        files.append({"path": str(path.relative_to(directory)), "sha256": sha256(path)})
        for line in path.open():
            if '"item_completed"' not in line:
                continue
            record = json.loads(line)
            payload = record.get("payload", {})
            item = payload.get("item", {})
            if payload.get("type") != "item_completed" or item.get("type") != "Reasoning":
                continue
            start, end = payload.get("started_at_ms"), payload.get("completed_at_ms")
            key = (payload.get("thread_id"), payload.get("turn_id"), item.get("id"), start, end)
            if key in seen:
                continue
            seen.add(key)
            if not isinstance(start, int) or not isinstance(end, int) or end < start or not observations:
                invalid += 1
                continue
            left, right = monotonic(start * 10**6), monotonic(end * 10**6)
            if right < left:
                invalid += 1
            else:
                spans.append((left, right, "reasoning"))
    return spans, {
        "source": "native Codex item_completed Reasoning timestamps",
        "valid_items": len(spans),
        "invalid_items": invalid,
        "native_files": files,
        "clock_mapping": "nearest host-observed wall/monotonic timestamp pair",
    }


def response_intervals(directory: Path, events: list[dict], harness: str) -> tuple[list, dict]:
    """Recover response windows, without equating them to pure thinking time.

    Native Gemini planner creation-to-tool-start windows include prefill/output.
    Claude/DeepCode/Kimi/dsh windows pair recorded requests with response markers.
    History echoed in a subsequent request is never treated as live reasoning output.
    Tools and announced retry waits take precedence over these broader windows.
    """
    audit = dict(source="native model-response windows", valid_items=0, native_files=[],
                 includes="thinking, output, prefill and request latency; partial coverage")
    if harness not in {"claude", "gemini", "gravity", "antigravity", "deepcode", "kimi", "dsh"} or not events:
        return [], audit
    observations = sorted((e["observed_wall_ns"], e["observed_monotonic_ns"]) for e in events)
    walls = [wall for wall, _ in observations]

    def stamp(value):
        try:
            return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 10**9)
        except (ValueError, TypeError, AttributeError):
            return None

    def mono(wall):
        i = min(bisect.bisect_left(walls, wall), len(walls) - 1)
        if i and abs(walls[i - 1] - wall) < abs(walls[i] - wall):
            i -= 1
        return wall - observations[i][0] + observations[i][1]

    intervals = set()

    def add(left, right):
        if left is not None and right is not None and left < right:
            intervals.add((mono(left), mono(right), "response"))

    def records(path):
        audit["native_files"].append(dict(path=str(path.relative_to(directory)), sha256=sha256(path)))
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    native = directory / "native"
    if harness in {"gemini", "gravity", "antigravity"}:
        for path in sorted(native.rglob("transcript_full.jsonl")):
            rows = sorted(records(path), key=lambda r: r.get("step_index", -1))
            for current, following in zip(rows, rows[1:]):
                if (current.get("type") == "PLANNER_RESPONSE" and current.get("status") == "DONE"
                        and current.get("tool_calls") and following.get("type") == "GENERIC"):
                    add(stamp(current.get("created_at")), stamp(following.get("created_at")))
        for event in events:
            step = event["event"].get("raw", {}).get("step_update", {})
            seconds = step.get("duration_seconds")
            if (step.get("step_type") == "agent_response" and step.get("state") == "DONE"
                    and isinstance(seconds, (float, int)) and seconds > 0):
                end = event["observed_monotonic_ns"]
                intervals.add((end - int(seconds * 1e9), end, "response"))
    else:
        requests = []
        for path in sorted(native.rglob("model_requests.jsonl")):
            for record in records(path):
                event = record.get("event", {})
                if event.get("type") == "model.request":
                    wall = stamp(event.get("timestamp"))
                    if wall is not None:
                        requests.append(wall)
        requests = sorted(set(requests))
        markers = []
        if harness == "claude":
            for path in sorted((native / ".harness_wrapper/traces").glob("*.jsonl")):
                for record in records(path):
                    event = record.get("event", {})
                    source = event.get("metadata", {}).get("raw", {}).get("source")
                    if event.get("type") == "reasoning_encrypted" and source == "model_response":
                        markers.append(stamp(event.get("timestamp")))
        elif harness in {"kimi", "dsh"}:
            # Kimi Code's proxy records usage when a response completes. dsh emits a
            # step's reasoning, text and tool calls once its stream has finished, so
            # the first of them after a request (the earliest marker wins) ends it.
            kinds = {"usage"} if harness == "kimi" else {"reasoning", "message", "tool_call"}
            for path in sorted((native / ".harness_wrapper/traces").glob("*.jsonl")):
                for record in records(path):
                    event = record.get("event", {})
                    source = (event.get("metadata") or {}).get("raw", {}).get("source")
                    if event.get("type") in kinds and source in ({"model_response"} if harness == "kimi" else {None}):
                        markers.append(stamp(event.get("timestamp")))
        else:
            for path in sorted((native / ".harness-home/.deepcode").rglob("*.jsonl")):
                for record in records(path):
                    if record.get("role") == "assistant":
                        markers.append(stamp(record.get("createTime")))
        paired = set()
        for end in sorted(x for x in markers if x is not None):
            i = bisect.bisect_right(requests, end) - 1
            if i >= 0 and i not in paired:
                add(requests[i], end)
                paired.add(i)
    audit["valid_items"] = len(intervals)
    return sorted(intervals), audit


def analyze(directory: Path) -> dict:
    metadata = read_json(directory / "attempt.json")
    if metadata.get("command_backend") == "host":
        raise ValueError("Process-level time analysis requires wrapper traces; host tool durations are in host-command-summary.json")
    observations = read_lines(directory / "observations.jsonl")
    boundaries = {
        r["reason"]: r["started_monotonic_ns"] for r in observations if r["reason"] in {"model_start", "model_end"}
    }
    if len(boundaries) != 2:
        raise ValueError(f"Missing model attempt boundaries: {directory}")
    events = read_lines(directory / "harness-events.jsonl")
    intervals, provenance = reasoning_intervals(directory, events)
    responses, response_provenance = response_intervals(directory, events, metadata["harness"])
    intervals.extend(responses)
    native = summarize_commands(read_lines(directory / "commands.jsonl"))
    incomplete_commands = harness_processes = 0
    for span in native["spans"]:
        start, end = span["start"], span["end"]
        if (start or end)["phase"] != "model":
            continue
        # A shell-script CLI (dsh) is itself traced; its span is the whole attempt.
        if any(arg.startswith("/opt/harness-cli/bin/") for arg in (start or end)["argv"][:2]):
            harness_processes += 1
            continue
        if span["status"] != "complete":
            incomplete_commands += 1
            continue
        kind = (
            operation(start["argv"])
            if start["kind"] == "lake"
            else ("measure" if standalone_measurement(start) else "other_tools")
        )
        if kind not in {"comparator", "build", "lean_check", "measure"}:
            kind = "other_tools"
        intervals.append(
            (end.get("execution_started_monotonic_ns") or start["monotonic_ns"], end["monotonic_ns"], kind)
        )
    # Action spans also cover file edits and non-shell tools. These are receipt
    # times, distinct from the native process timestamps preferred for Lake.
    for span in read_json(directory / "actions.json")["spans"]:
        if span["status"] == "complete" and span.get("timestamp_source") == "host_observation":
            intervals.append(
                (span["start"]["observed_monotonic_ns"], span["end"]["observed_monotonic_ns"], "other_tools")
            )
    retry_intervals = 0
    for event in events:
        raw = event["event"]
        if raw.get("type") == "recovery" and isinstance(raw.get("content"), dict):
            delay = raw["content"].get("retry_delay_seconds")
            if isinstance(delay, (int, float)) and delay > 0:
                start = event["observed_monotonic_ns"]
                intervals.append((start, start + int(delay * 1e9), "retry_wait"))
                retry_intervals += 1
    start, end, adjustment = model_window(directory, observations, events)
    seconds = partition(start, end, intervals)
    return {
        "problem_idx": metadata["problem_idx"],
        "project_key": metadata["project_key"],
        "harness": metadata["harness"],
        "attempt": str(directory),
        "total_s": (end - start) / 1e9,
        "timing_adjustment": adjustment,
        "boundary_analyzer_sha256": sha256(Path(__file__).with_name("attempt_timing.py")),
        "seconds": seconds,
        "reasoning_provenance": provenance,
        "response_provenance": response_provenance,
        "incomplete_command_spans": incomplete_commands,
        "harness_process_spans": harness_processes,
        "announced_retry_intervals": retry_intervals,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lake-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lake = read_json(args.lake_analysis / "summary.json")
    rows = []
    for attempt in sorted(lake["attempts"], key=lambda a: a["problem_idx"]):
        rows.append(analyze(Path(attempt["attempt"])))
    totals = {key: sum(r["seconds"][key] for r in rows) for key in CATEGORIES}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "breakdown.json",
        {
            "total_s": sum(r["total_s"] for r in rows),
            "seconds": totals,
            "attempts": rows,
            "notes": [
                "Reasoning means native CLI-reported reasoning-item lifetime, not server compute time.",
                "Parallel reasoning and tool activity is counted once in its own category.",
                "Nested tools use Comparator > build > lake env lean > other tools precedence.",
                "Retry intervals use announced delays; other recovery and latency remain unclassified.",
                "Unclassified includes request/prefill latency, output, compaction, setup, gaps and missing telemetry.",
                "Claude, Gemini, DeepCode, Kimi Code and DeepSeek Harness response windows include thinking, output, prefill and request latency; coverage can be partial.",
                "Measure covers standalone proof_length.py commands; mixed shell commands remain other tools.",
            ],
            "lake_analysis_sha256": sha256(args.lake_analysis / "summary.json"),
        },
    )
    write_csv(
        args.output / "breakdown.csv",
        [
            {"problem_idx": r["problem_idx"], "project_key": r["project_key"], "total_s": r["total_s"], **r["seconds"]}
            for r in rows
        ],
    )


if __name__ == "__main__":
    main()
