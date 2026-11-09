"""Export verified LeanLean analysis to a portable, source-only website snapshot.

The server never opens experiment directories. Model requests, credentials,
absolute host paths, and build caches are deliberately absent from this format.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import difflib
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile
import tempfile

from leanlean.common import read_json, sha256, utc_now, write_json
from leanlean.definition_savings import origin_categories
from leanlean.graph_identity import checksum
from leanlean.time_analysis import analyze
from leanlean.website_progress import attempt_progress, memoized_counter
from leanlean.postprocess import token_counter


def packed(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode(), mtime=0))


def sources(archive: Path, allowed: set[str]) -> dict[str, str]:
    result = {}
    with tarfile.open(archive) as stream:
        for member in stream:
            name = str(PurePosixPath(member.name))
            if name in allowed:
                if not member.isfile() or member.size > 64 * 1024**2:
                    raise ValueError(f"Unexpected source archive member: {name}")
                result[name] = stream.extractfile(member).read().decode("utf-8")
    return result


def diff_counts(text: str) -> dict:
    added = removed = 0
    in_hunk = False
    for line in text.splitlines():
        if line.startswith("@@ "):
            in_hunk = True
        elif in_hunk:
            added += line.startswith("+")
            removed += line.startswith("-")
    return dict(added=added, removed=removed)


def final_diffs(before: dict, after: dict, output: Path) -> list[dict]:
    files = []
    for path in sorted(before.keys() | after.keys()):
        left, right = before.get(path, ""), after.get(path, "")
        if left == right:
            continue
        diff = "".join(
            line if line.endswith("\n") else line + "\n\\ No newline at end of file\n"
            for line in difflib.unified_diff(
                left.splitlines(True), right.splitlines(True), fromfile=f"a/{path}", tofile=f"b/{path}", n=3
            )
        )
        identifier = "diff-" + hashlib.sha256(path.encode()).hexdigest()[:20]
        packed(output / f"{identifier}.json.gz", dict(path=path, diff=diff))
        files.append(dict(path=path, asset=identifier, **diff_counts(diff)))
    return sorted(files, key=lambda f: (-f["removed"] - f["added"], f["path"]))


def checkpoint_diff(text: str, allowed: set[str]) -> list[dict]:
    """Select scored Lean sources from the monitor's JSON-quoted diff headers."""
    files = []
    for block in text.split("diff --leanlean ")[1:]:
        header, _, body = block.partition("\n")
        path = json.loads(header)
        if path not in allowed:
            continue
        lines = body.splitlines(True)
        diff = "".join(line for line in lines if not line.startswith("metadata "))
        files.append(dict(path=path, diff=diff, **diff_counts(diff)))
    return files


def module_graph(graph: dict, before: dict, after: dict) -> dict:
    """Collapse declaration dependencies into file nodes; never invent edges."""
    if graph.get("sha256") != checksum(graph):
        raise ValueError("Dependency graph checksum mismatch")
    suffixes = defaultdict(list)
    for path in before.keys() | after.keys():
        parts = PurePosixPath(path).with_suffix("").parts
        for i in range(len(parts)):
            suffixes[".".join(parts[i:])].append(path)
    nodes = []
    by_path = {}
    for path in sorted(before.keys() | after.keys()):
        a, b = before.get(path, 0), after.get(path, 0)
        by_path[path] = len(nodes)
        nodes.append(
            dict(
                path=path, before=a, after=b, saved=a - b, compression=100 * (a - b) / a if a else None, protected=False
            )
        )
    providers = []
    unresolved = 0
    protected = set(graph.get("policies", {}).get("production", {}).get("protected", []))
    for i, node in enumerate(graph["nodes"]):
        matches = suffixes.get(node["module"], [])
        path = matches[0] if len(matches) == 1 else None
        providers.append(by_path.get(path))
        unresolved += path is None
        if i in protected and path is not None:
            nodes[by_path[path]]["protected"] = True
    edges = Counter()
    for a, b, *_ in graph["edges"]:
        left, right = providers[a], providers[b]
        if left is not None and right is not None and left != right:
            edges[left, right] += 1
    # Ambiguous dependency candidates are not displayed as definite edges.
    return dict(
        nodes=nodes,
        edges=[[a, b, n] for (a, b), n in sorted(edges.items())],
        unresolved_declarations=unresolved,
        ambiguous_edges=len(graph.get("ambiguous_edges", [])),
        scope="Baseline declaration dependencies aggregated by source file. Color measures file token reduction, not declaration origin.",
    )


def replay_records(runs: list[Path]) -> dict:
    records = {}
    for run in runs:
        for record in read_json(run / "plan.json")["records"]:
            idx = record["job"]["problem_idx"]
            slot = records.setdefault(idx, {"frames": [], "states": {}})
            if len(record["frames"]) > len(slot["frames"]):
                slot["frames"] = record["frames"]
            for state in record["states"]:
                path = run / "repositories" / record["job"]["key"] / "states" / state["source_sha256"] / "result.json"
                if path.exists():
                    value = read_json(path)
                    if (
                        value["source_sha256"] != state["source_sha256"]
                        or value["archive_sha256"] != state["archive_sha256"]
                    ):
                        raise ValueError("Replay source identity changed")
                    slot["states"][state["source_sha256"]] = value
    return records


def timeline(record: dict, attempt: Path, baseline: int, output: Path, allowed: set[str]) -> list:
    observations = [json.loads(line) for line in (attempt / "observations.jsonl").read_text().splitlines()]
    start = next(x["started_monotonic_ns"] for x in observations if x["reason"] == "model_start")
    frames = []
    for f in record.get("frames", []):
        if not f["changed"]:
            continue
        state = record["states"].get(f["source_sha256"], {})
        tokens = state.get("metrics", {}).get("lean_tokens")
        index = f["snapshot_index"]
        asset = None
        if f.get("parent_sha256") and f.get("diff"):
            path = attempt / f["diff"]
            if not path.resolve().is_relative_to(attempt.resolve()):
                raise ValueError("Checkpoint diff escapes telemetry directory")
            files = checkpoint_diff(path.read_text(), allowed)
            if files:
                asset = f"checkpoint-{index}"
                packed(output / f"{asset}.json.gz", dict(files=files))
        frames.append(
            dict(
                index=index,
                elapsed=max(0, (f["started_monotonic_ns"] - start) / 1e9),
                compression=100 * (1 - tokens / baseline) if tokens is not None else None,
                tokens=tokens,
                verified=state.get("verified", False),
                status=state.get("status", "unavailable"),
                reason=f["reason"],
                asset=asset,
            )
        )
    return frames


def export(args) -> None:
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", args.model_id):
        raise ValueError("Model ID must be a lowercase URL slug")
    plan = read_json(args.postprocess / "plan.json")
    count = memoized_counter(token_counter(args.postprocess / "pipeline" / "reference_tokens.py"))
    origins = {r["problem_idx"]: r for r in read_json(args.origins / "summary.json")["results"]}
    origin_sources = {r["problem_idx"]: r for r in read_json(args.origins / "sources.json")["results"]}
    lake = read_json(args.lake_analysis / "summary.json")
    attempts = {r["problem_idx"]: Path(r["attempt"]) for r in lake["attempts"]}
    replay = replay_records(args.replay)
    with (args.lake_analysis / "commands.csv").open() as stream:
        commands = list(csv.DictReader(stream))
    args.output.mkdir(parents=True, exist_ok=True)
    index_path = args.output / "index.json"
    previous = read_json(index_path) if index_path.exists() else None
    if (args.output / args.model_id).exists():
        raise ValueError("Model export already exists; use a fresh output directory")
    staging = Path(tempfile.mkdtemp(prefix=".export-", dir=args.output))
    instances, results = [], []
    try:
        for job in plan["jobs"]:
            idx = job["problem_idx"]
            directory = Path(job["directory"])
            result = read_json(directory / "result.json")
            if result["status"] != "verified":
                raise ValueError(f"P{idx}: final submission is not verified")
            origin = origins[idx]
            source = origin_sources[idx]
            origin_path = Path(source["directory"])
            if sha256(origin_path / "origins.json") != source["origins_sha256"]:
                raise ValueError(f"P{idx}: origin artifact changed")
            if sha256(origin_path / "status.json") != source["status_sha256"]:
                raise ValueError(f"P{idx}: origin provenance changed")
            status = read_json(origin_path / "status.json")
            for name, digest in status["inputs"].items():
                # renames.json is written beside the analysis, not with the job's inputs.
                if sha256((origin_path if name == "renames.json" else directory) / name) != digest:
                    raise ValueError(f"P{idx}: origin inputs changed")
            baseline, final = result["metrics"]["before"], result["metrics"]["after"]
            a, b = baseline["lean_tokens"], final["lean_tokens"]
            if origin["status"] != "passed" or sum(origin["categories"].values()) != a - b:
                raise ValueError(f"P{idx}: origin ledger does not reconcile")
            entry = result["entry"]
            instance = dict(
                id=idx,
                key=job["key"],
                title=entry["title"],
                abstract=entry["abstract"],
                github=entry["github_url"],
                theorems=entry["theorem_names"],
                baseline=a,
                files=baseline["lean_file_count"],
                size="S" if a < 50000 else "M" if a < 200000 else "L" if a < 1000000 else "XL",
            )
            instances.append(instance)
            output = staging / str(idx)
            prefix = entry["project_path"]
            files = {
                phase: {str(PurePosixPath(prefix) / p): n for p, n in result["metrics"][phase]["files"].items()}
                for phase in ("before", "after")
            }
            texts = {
                phase: sources(directory / phase / "original-sources.tar.gz", set(files[phase])) for phase in files
            }
            if any(set(texts[phase]) != set(files[phase]) for phase in files):
                raise ValueError(f"P{idx}: scored source files missing from archive")
            diffs = final_diffs(texts["before"], texts["after"], output)
            graph = module_graph(
                read_json(directory / "before" / "dependency-graph.json"), files["before"], files["after"]
            )
            packed(output / "graph.json.gz", graph)
            timing = analyze(attempts[idx])
            call_stats = {}
            for op in ("build", "comparator", "lean_check"):
                selected = [
                    r
                    for r in commands
                    if int(r["problem_idx"]) == idx and r["phase"] == "model" and r["operation"] == op
                ]
                durations = [float(r["duration_s"]) for r in selected if r["status"] == "complete"]
                call_stats[op] = dict(
                    count=len(selected),
                    complete=len(durations),
                    mean_s=sum(durations) / len(durations) if durations else None,
                    max_s=max(durations, default=None),
                )
            record = replay.get(idx, {})
            # Include intermediate-only files, including ones later deleted,
            # but never publish unscored configuration or telemetry files.
            scored_paths = set(files["before"]) | set(files["after"])
            for state in record.get("states", {}).values():
                scored_paths.update(str(PurePosixPath(prefix) / p) for p in state.get("metrics", {}).get("files", {}))
            frames = timeline(record, attempts[idx], a, output, scored_paths)
            progress = attempt_progress(attempts[idx], entry, a, count)
            row = dict(
                id=idx,
                baseline=a,
                final=b,
                compression=100 * (a - b) / a,
                verified=True,
                elapsed=timing["total_s"],
                checkpoints=len(frames),
                changed_files=len(diffs),
            )
            results.append(row)
            packed(
                output / "detail.json.gz",
                dict(
                    **row,
                    origins=origin_categories(origin_path / "origins.json", origin),
                    time=timing["seconds"],
                    calls=call_stats,
                    incomplete_commands=timing["incomplete_command_spans"],
                    diffs=diffs,
                    timeline=frames,
                    progress=progress,
                    origin_ambiguities=origin.get("unresolved_provider_moves", []),
                    theorem_names=entry["theorem_names"],
                    verification={
                        phase: {
                            name: (v.get("returncode") if isinstance(v, dict) else v)
                            for name, v in read_json(directory / phase / "verification.json").get("stages", {}).items()
                        }
                        for phase in ("before", "after")
                    },
                ),
            )
            print(f"Exported P{idx}: {len(diffs)} changed files, {len(frames)} checkpoints", flush=True)
        model = dict(
            id=args.model_id,
            name=args.model_name,
            profile=args.profile,
            results=results,
            score=sum(r["compression"] for r in results) / len(results),
            verified=len(results),
            elapsed=sum(r["elapsed"] for r in results),
            total_saved=sum(r["baseline"] - r["final"] for r in results),
        )
        if previous and previous["instances"] != instances:
            raise ValueError("Benchmark instances differ from existing export")
        data = dict(
            schema_version=1,
            name="LeanLean",
            edition="October 2026",
            updated_at=utc_now(),
            instances=instances,
            models=[*(previous["models"] if previous else []), model],
            methodology=dict(
                score="Mean per-instance reduction in Lean tokens. Each repository has equal weight; increases remain negative.",
                origin="Heuristic token attribution. Signed categories sum to the verified token reduction.",
                time="Exclusive wall-time allocation during the agent attempt. Thinking is native reasoning-item lifetime, not server compute. Parallel work is counted once. Unobserved latency remains unclassified.",
                measure="Only standalone proof_length.py commands are assigned to Measure. Mixed commands remain Other tools.",
                checkpoints="Saved source changes. Verification is replayed offline and is not included in agent time. Invalid intermediate states are retained.",
                graph="Nodes are source files; edges aggregate baseline declaration dependencies. Green intensity is file token reduction. Relocation can change file size without changing declaration size.",
            ),
        )
        staging.rename(args.output / args.model_id)
        write_json(index_path, data)
        write_json(
            args.output / f"{args.model_id}-provenance.json",
            dict(
                postprocess_plan_sha256=sha256(args.postprocess / "plan.json"),
                origins_sha256=sha256(args.origins / "summary.json"),
                lake_analysis_sha256=sha256(args.lake_analysis / "summary.json"),
                exporter_sha256=sha256(Path(__file__)),
                time_analysis_sha256=sha256(Path(__file__).with_name("time_analysis.py")),
            ),
        )
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("postprocess", "origins", "lake-analysis", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument(
        "--replay",
        type=Path,
        action="append",
        default=[],
        help="Later runs override matching states, including targeted retries",
    )
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--profile", required=True, help="User-visible evaluation conditions")
    export(parser.parse_args())


if __name__ == "__main__":
    main()
