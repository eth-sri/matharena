"""Combine replay batches into checkpoint timelines and processing-cost summaries.

Later runs take ownership of overlapping projects only when their captured inputs
and analysis provenance agree. This supports moving queued projects to additional
workers without counting the same state twice.
"""

from __future__ import annotations

import argparse
import math
import re
import statistics
from collections import Counter
from pathlib import Path

from leanlean.common import read_json, sha256, utc_now, within, write_json


def owners(runs: list[Path]) -> dict:
    selected = {}
    provenance = None
    for run in runs:
        plan = read_json(run / "plan.json")
        identity = {
            k: plan[k]
            for k in (
                "postprocess_sha256",
                "runtime",
                "extractor_sha256",
                "extractor_commit",
                "tokens_sha256",
                "pipeline_sha256",
            )
        }
        if provenance is not None and identity != provenance:
            raise ValueError(f"Incompatible replay provenance: {run}")
        provenance = identity
        for record in plan["records"]:
            idx = record["job"]["problem_idx"]
            if idx in selected:
                previous = selected[idx][1]
                for field in ("checkpoints_sha256", "states", "baseline_sources"):
                    if previous[field] != record[field]:
                        raise ValueError(f"Different captured inputs for P{idx}: {field}")
            selected[idx] = run, record
    return selected


def timing(values: list[float]) -> dict:
    ordered = sorted(values)
    return dict(
        count=len(values),
        total_seconds=sum(values),
        mean_seconds=statistics.mean(values) if values else None,
        median_seconds=statistics.median(values) if values else None,
        p95_seconds=ordered[math.ceil(0.95 * len(ordered)) - 1] if ordered else None,
        max_seconds=max(values) if values else None,
    )


def graph_availability(value: dict, result_path: Path) -> dict:
    """Explain missing graphs using checksummed evidence, without rewriting results."""
    if any(name.endswith("/dependency-graph.json") for name in value.get("artifacts", {})):
        return {"status": "available"}
    reason = value.get("graph_unavailable_reason")
    if reason:
        return {"status": reason}
    if value["status"] != "graph_failed":
        return {"status": value["status"]}
    artifacts = value.get("artifacts", {})

    def evidence(suffix):
        matches = [name for name in artifacts if name.endswith("/" + suffix)]
        if len(matches) != 1:
            return None
        path = within(result_path.parent, matches[0])
        if sha256(path) != artifacts[matches[0]]:
            raise ValueError(f"Graph failure evidence changed: {path}")
        return path

    report_path = evidence("report.json")
    if report_path and read_json(report_path).get("stage") == "repository_build":
        comparator = value.get("stages", {}).get("comparator", {})
        errors, logs = [], []
        for name in ("comparator.stdout.log", "comparator.stderr.log"):
            path = evidence(name)
            if path:
                logs.append(str(path))
                errors.extend(
                    line for line in path.read_text().splitlines() if re.match(r"^error: .+\.lean:\d+:\d+:", line)
                )
        if comparator.get("returncode") == 1 and errors:
            return dict(status="solution_build_failed", evidence=[str(report_path), *logs], compiler_errors=errors)
    return {"status": "extraction_failed"}


def report(runs: list[Path], output: Path, verify_artifacts: bool = False, write_timelines: bool = True) -> dict:
    selected = owners(runs)
    projects, gaps = [], []
    overall, availability_counts = Counter(), Counter()
    stage_times = {name: [] for name in ("build", "solution_build", "comparator", "graph")}
    for idx, (run, record) in sorted(selected.items()):
        job = record["job"]
        root = run / "repositories" / job["key"]
        results, counts = {}, Counter()
        per_stage = {name: [] for name in stage_times}
        for state in record["states"]:
            path = root / "states" / state["source_sha256"] / "result.json"
            value = read_json(path) if path.exists() else {"status": "pending", "verified": False}
            if path.exists():
                for field in ("source_sha256", "archive_sha256"):
                    if value[field] != state[field]:
                        raise ValueError(f"Result identity mismatch: {path}")
            if verify_artifacts:
                for relative, digest in value.get("artifacts", {}).items():
                    if sha256(within(path.parent, relative)) != digest:
                        raise ValueError(f"Artifact checksum mismatch: {path.parent / relative}")
            graphs = [
                str(path.parent / name)
                for name in value.get("artifacts", {})
                if name.endswith("/dependency-graph.json")
            ]
            availability = graph_availability(value, path)
            availability_counts[availability["status"]] += 1
            results[state["source_sha256"]] = dict(
                status=value["status"],
                verified=value.get("verified", False),
                result=str(path),
                graph=graphs[0] if graphs else None,
                metrics=value.get("metrics"),
                graph_availability=availability,
            )
            counts[value["status"]] += 1
            for name, stage in value.get("stages", {}).items():
                if name in per_stage:
                    per_stage[name].append((stage["ended_wall_ns"] - stage["started_wall_ns"]) / 1e9)
            if value["status"] in {"error", "source_changed", "graph_failed", "rejected"}:
                gaps.append(
                    dict(
                        problem_idx=idx,
                        source_sha256=state["source_sha256"],
                        status=value["status"],
                        verified=value.get("verified", False),
                        result=str(path),
                        graph_availability=availability,
                    )
                )
        frames = [{**frame, "analysis": results[frame["source_sha256"]]} for frame in record["frames"]]
        if write_timelines:
            write_json(
                output / "timelines" / f"{job['key']}.json", dict(problem_idx=idx, key=job["key"], frames=frames)
            )
        verified = [r for r in results.values() if r["verified"]]
        projects.append(
            dict(
                problem_idx=idx,
                key=job["key"],
                run=str(run),
                states=len(results),
                checkpoints=len(frames),
                statuses=dict(counts),
                verified_states=len(verified),
                baseline_tokens=job["metrics"]["before"]["lean_tokens"],
                best_verified_compression_pct=max(
                    (r["metrics"]["compression_pct"] for r in verified if r["metrics"]), default=None
                ),
                timings={name: timing(values) for name, values in per_stage.items()},
            )
        )
        overall.update(counts)
        for name, values in per_stage.items():
            stage_times[name].extend(values)
    remaining = overall["pending"] + overall["running"]
    summary = dict(
        updated_at=utc_now(),
        runs=[str(run) for run in runs],
        status="running" if remaining else "finished_with_gaps" if gaps else "finished",
        total_states=sum(overall.values()),
        completed=sum(overall.values()) - remaining,
        total_checkpoints=sum(p["checkpoints"] for p in projects),
        statuses=dict(overall),
        graph_availability=dict(availability_counts),
        verified_states=sum(p["verified_states"] for p in projects),
        artifact_checksums_verified=verify_artifacts,
        timings={name: timing(values) for name, values in stage_times.items()},
        projects=projects,
        gaps=gaps,
        scope="Checkpoint verification, token metrics, graphs, and replay processing times. Origin classification is separate.",
    )
    write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True, help="later runs own overlapping projects")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    summary = report([p.resolve() for p in args.run], args.output.resolve(), args.verify_artifacts)
    print(f"STATUS {summary['status']}: {summary['completed']}/{summary['total_states']} states", flush=True)


if __name__ == "__main__":
    main()
