"""Summarize native Lake process timings from completed evaluation telemetry.

Only instrumented toolchain Lake spans are counted: shell and Elan proxy spans
would duplicate the same execution. Observed interruptions remain separate from
measured command durations. Resource peaks refer to the whole container.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from leanlean.common import read_json, sha256, write_json
from matharena.monitoring.attempt import lake_subcommand, summarize_commands
from matharena.json_zst import load_json_zst

OPERATIONS = {"build", "comparator", "lean_check"}


def read_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def operation(argv: list[str]) -> str:
    args = argv[1:]
    subcommand = lake_subcommand(args)
    if "--help" in args or "-h" in args or "--version" in args:
        return "help"
    if subcommand in {"build", "comparator"}:
        return subcommand
    if subcommand == "env":
        following = args[args.index("env") + 1 :]
        if following and Path(following[0]).name == "lean":
            return "lean_check"
    return "other"


def union_seconds(intervals: list[tuple[int, int]]) -> float:
    merged = []
    for start, end in sorted(intervals):
        if end < start:
            raise ValueError("Invalid interval")
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged) / 1e9


def aggregate(rows: list[dict]) -> dict:
    complete = [r for r in rows if r["status"] == "complete"]
    durations = sorted(r["duration_s"] for r in complete)
    intervals = [(r["execution_start_ns"], r["end_monotonic_ns"]) for r in complete]
    return {
        "commands": len(rows),
        "complete": len(complete),
        "incomplete": len(rows) - len(complete),
        "succeeded": sum(r["returncode"] == 0 for r in complete),
        "nonzero_exit": sum(r["returncode"] is not None and r["returncode"] != 0 for r in complete),
        "process_seconds": sum(durations),
        "busy_wall_seconds": union_seconds(intervals),
        "mean_s": statistics.mean(durations) if durations else None,
        "median_s": statistics.median(durations) if durations else None,
        "p95_s": durations[max(0, (95 * len(durations) + 99) // 100 - 1)] if durations else None,
        "max_s": max(durations) if durations else None,
    }


def utc(ns: int | None) -> str | None:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc).isoformat() if ns is not None else None


def analyze_attempt(directory: Path, interrupted: dict) -> tuple[list[dict], dict]:
    metadata = read_json(directory / "attempt.json")
    if metadata.get("command_backend") == "host":
        raise ValueError("Process-level Lake analysis requires wrapper traces; host tool durations are in host-command-summary.json")
    commands = read_lines(directory / "commands.jsonl")
    summary = summarize_commands(commands)
    resources = [r for r in read_lines(directory / "resources.jsonl") if r.get("status") == "sample"]
    resources.sort(key=lambda r: r["observed_monotonic_ns"])
    sample_times = [r["observed_monotonic_ns"] for r in resources]
    observations = read_lines(directory / "observations.jsonl")
    boundaries = {
        r["reason"]: r["started_monotonic_ns"] for r in observations if r["reason"] in {"model_start", "model_end"}
    }
    model_seconds = (boundaries["model_end"] - boundaries["model_start"]) / 1e9 if len(boundaries) == 2 else None
    base = {
        "problem_idx": metadata["problem_idx"],
        "run_idx": metadata["run_idx"],
        "project_key": metadata["project_key"],
        "harness": metadata["harness"],
        "attempt": str(directory),
    }
    rows = []
    for span in summary["spans"]:
        event = span["start"] or span["end"]
        if event["kind"] != "lake":
            continue
        start, end = span["start"], span["end"]
        execution = (end.get("execution_started_monotonic_ns") if end else None) or (
            start["monotonic_ns"] if start else None
        )
        end_ns = end["monotonic_ns"] if end else None
        selected = (
            resources[bisect.bisect_left(sample_times, execution) : bisect.bisect_right(sample_times, end_ns)]
            if span["status"] == "complete"
            else []
        )
        upper = interrupted.get(span["span_id"], {}) if span["status"] == "missing_end" else {}
        rows.append(
            {
                **base,
                "span_id": span["span_id"],
                "parent_span_id": event.get("parent_span_id"),
                "phase": event["phase"],
                "operation": operation(event["argv"]),
                "argv": json.dumps(event["argv"]),
                "cwd": event["cwd"],
                "status": span["status"],
                "start_utc": utc(span.get("execution_started_wall_ns") or (start["wall_time_ns"] if start else None)),
                "end_utc": utc(end["wall_time_ns"] if end else None),
                "execution_start_ns": execution,
                "end_monotonic_ns": end_ns,
                "duration_s": span.get("duration_s"),
                "returncode": end.get("returncode") if end else None,
                "start_collection_s": span.get("start_collection_s"),
                "check_only": "--no-build" in event["argv"],
                "resource_samples": len(selected),
                "peak_sampled_container_ram_bytes": max((r["memory_current_bytes"] for r in selected), default=None),
                "peak_sampled_container_cpu_percent": max(
                    (r["cpu_percent"] for r in selected if r.get("cpu_percent") is not None), default=None
                ),
                "inferred_container_exit": upper.get("container_exit_zurich"),
                "duration_upper_bound_s": upper.get("seconds_start_to_exit"),
            }
        )
    return rows, {
        **base,
        "model_wall_seconds": model_seconds,
        "commands_sha256": sha256(directory / "commands.jsonl"),
        "duplicate_span_events": summary["duplicate_span_events"],
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="checksum-pinned combined-results.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interrupted-matches", type=Path, help="optional inferred container-exit matches")
    args = parser.parse_args()
    interrupted = (
        {r["span_id"]: r for r in read_json(args.interrupted_matches)["matches"]} if args.interrupted_matches else {}
    )
    directories = set()
    for indexed in read_json(args.index).values():
        if sha256(Path(indexed["path"])) != indexed["sha256"]:
            raise ValueError(f'Result checksum mismatch: {indexed["path"]}')
        result = load_json_zst(indexed["path"])
        for attempt in result["history"]:
            for step in attempt:
                if step.get("monitoring"):
                    directories.add(Path(step["monitoring"]["directory"]))
    rows, attempts = [], []
    for directory in sorted(directories):
        commands, attempt = analyze_attempt(directory, interrupted)
        rows.extend(commands)
        attempts.append(attempt)
    groups = defaultdict(list)
    for row in rows:
        groups[(row["attempt"], row["phase"], row["operation"])].append(row)
    per_repo = []
    for attempt in sorted(attempts, key=lambda a: (a["problem_idx"], a["run_idx"])):
        for phase in ("preflight", "model", "postcheck"):
            for op in sorted(OPERATIONS):
                selected = groups[(attempt["attempt"], phase, op)]
                per_repo.append(
                    {
                        **attempt,
                        "duplicate_span_events": len(attempt["duplicate_span_events"]),
                        "phase": phase,
                        "operation": op,
                        **aggregate(selected),
                    }
                )
    # Add per-attempt unions, not a union across concurrent project workers.
    totals = []
    for phase in ("preflight", "model", "postcheck"):
        for op in sorted(OPERATIONS):
            selected = [r for r in rows if r["phase"] == phase and r["operation"] == op]
            total = aggregate(selected)
            total["busy_wall_seconds"] = sum(
                r["busy_wall_seconds"] for r in per_repo if r["phase"] == phase and r["operation"] == op
            )
            totals.append({"phase": phase, "operation": op, **total})
    model_busy = sum(
        union_seconds(
            [
                (r["execution_start_ns"], r["end_monotonic_ns"])
                for r in rows
                if r["attempt"] == a["attempt"]
                and r["phase"] == "model"
                and r["operation"] in OPERATIONS
                and r["status"] == "complete"
            ]
        )
        for a in attempts
    )
    build_comparator_busy = sum(
        union_seconds(
            [
                (r["execution_start_ns"], r["end_monotonic_ns"])
                for r in rows
                if r["attempt"] == a["attempt"]
                and r["phase"] == "model"
                and r["operation"] in {"build", "comparator"}
                and r["status"] == "complete"
            ]
        )
        for a in attempts
    )
    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "commands.csv", rows)
    write_csv(args.output / "repositories.csv", per_repo)
    write_json(
        args.output / "summary.json",
        {
            "attempts": attempts,
            "totals": totals,
            "model_lake_busy_wall_seconds": model_busy,
            "model_build_comparator_busy_wall_seconds": build_comparator_busy,
            "model_attempt_wall_seconds": sum(a["model_wall_seconds"] or 0 for a in attempts),
            "notes": [
                "Times are native Lake process lifetimes, excluding the initial collector handoff.",
                "Help/version queries and Elan proxy spans are excluded from build/comparator totals.",
                "Busy wall time merges overlaps per attempt; operation totals can overlap each other.",
                "Incomplete spans have no measured duration; inferred exits are separate upper bounds.",
                "Resource peaks are sampled whole-container usage, not per-command CPU/RAM attribution.",
            ],
            "index_sha256": sha256(args.index),
        },
    )


if __name__ == "__main__":
    main()
