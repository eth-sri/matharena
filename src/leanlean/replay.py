"""Replay captured source states with warm Lake builds and durable per-state results.

One worker owns a project's mutable build cache. Checkpoints keep their original
ordering, including reverts, while repeated archives share a single analysis.
Intermediate build/Comparator failures are observations, not benchmark scores.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import queue
import re
import shutil
import subprocess
import threading
import uuid
from collections import Counter
from pathlib import Path

from leanlean.common import exclusive_lock, read_json, sha256, tree_sha256, utc_now, within, write_json
from leanlean.graphs_container import source_hashes
from leanlean.postprocess import container_command, extract_submission, measure, restore_fixtures, token_counter
from matharena.json_zst import load_json_zst

GENERATED = {".lake", ".git", ".lean-strip"}
REUSABLE = {"complete", "build_failed", "graph_failed", "rejected"}


def source_files(root: Path) -> dict[str, Path]:
    files = {}
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if name not in GENERATED]
        for name in dirs + names:
            path = Path(directory) / name
            if path.is_symlink():
                raise ValueError(f"Unexpected source symlink: {path}")
        for name in names:
            if name not in GENERATED:
                path = Path(directory) / name
                files[str(path.relative_to(root))] = path
    return files


def sync_sources(staged: Path, repository: Path) -> dict:
    """Preserve unchanged mtimes; remove stale modules before incremental builds."""
    old, new = source_files(repository), source_files(staged)
    deleted = sorted(old.keys() - new.keys())
    # Reset *all* local build trees on deletion, including path dependencies.
    # Never walk the pinned external package cache.
    if deleted:
        for directory, dirs, _ in os.walk(repository, followlinks=False):
            dirs[:] = [name for name in dirs if name not in GENERATED]
            build = Path(directory) / ".lake" / "build"
            if build.exists():
                shutil.rmtree(build)
    for relative in deleted:
        old[relative].unlink()
    # Remove empty source directories so directory->file replacements work.
    for directory, dirs, _ in os.walk(repository, topdown=False, followlinks=False):
        path = Path(directory)
        if path == repository or any(part in GENERATED for part in path.relative_to(repository).parts):
            continue
        if not any(path.iterdir()):
            path.rmdir()
    changed = []
    for relative, source in new.items():
        target = repository / relative
        if (
            not target.is_file()
            or sha256(target) != sha256(source)
            or (target.stat().st_mode & 0o777) != (source.stat().st_mode & 0o777)
        ):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            shutil.copymode(source, target)
            changed.append(relative)
    return dict(deleted=deleted, changed=changed, build_cache_reset=bool(deleted))


def verify_archive(state: dict) -> Path:
    archive = Path(state["archive"])
    if sha256(archive) != state["archive_sha256"]:
        raise ValueError(f"Archive changed: {archive}")
    digest = hashlib.sha256()
    with gzip.open(archive, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != state["source_sha256"]:
        raise ValueError(f"Uncompressed snapshot checksum mismatch: {archive}")
    return archive


def cached_result(path: Path, state: dict) -> dict | None:
    if not path.exists():
        return None
    result = read_json(path)
    if result.get("status") not in REUSABLE:
        return None
    if result["source_sha256"] != state["source_sha256"] or result["archive_sha256"] != state["archive_sha256"]:
        raise ValueError(f"Cached state identity changed: {path}")
    for name, digest in result.get("artifacts", {}).items():
        if sha256(within(path.parent, name)) != digest:
            raise ValueError(f"Cached artifact changed: {path.parent / name}")
    return result


def inventory(jobs: list[dict]) -> list[dict]:
    records = []
    for job in jobs:
        verified = read_json(Path(job["directory"]) / "result.json")
        if verified.get("status") != "verified" or verified["result_sha256"] != job["result_sha256"]:
            raise ValueError(f"Missing verified baseline/final pair for {job['key']}")
        result_path = Path(job["result_path"])
        if sha256(result_path) != job["result_sha256"]:
            raise ValueError(f"Evaluation result changed: {result_path}")
        step = load_json_zst(result_path)["history"][0][0]
        if step["leanlean_submission"]["key"] != job["key"]:
            raise ValueError("Checkpoint project does not match baseline")
        telemetry = Path(step["monitoring"]["directory"])
        checkpoints = telemetry / "checkpoints.jsonl"
        frames = [json.loads(line) for line in checkpoints.read_text().splitlines() if line.strip()]
        states = {}
        for frame in frames:
            key = frame["source_sha256"]
            if not re.fullmatch(r"[a-f0-9]{64}", key):
                raise ValueError(f"Invalid state identifier: {key}")
            archive = within(telemetry, frame["archive"])
            if key not in states:
                states[key] = dict(source_sha256=key, archive=str(archive), archive_sha256=sha256(archive))
            frame["result"] = f"states/{key}/result.json"
        if not frames or frames[0]["reason"] != "baseline":
            raise ValueError(f"Missing baseline checkpoint for {job['key']}")
        records.append(
            dict(
                job=job,
                frames=frames,
                states=list(states.values()),
                checkpoints_sha256=sha256(checkpoints),
                baseline_sources=source_hashes(Path(job["directory"]) / "before/repository"),
            )
        )
    return records


def replay_state(record: dict, state: dict, run: Path, plan: dict, worker: str, count) -> dict:
    job = record["job"]
    root = run / "repositories" / job["key"]
    directory = root / "states" / state["source_sha256"]
    result_path = directory / "result.json"
    previous = cached_result(result_path, state)
    if previous is not None:
        return previous
    attempts = directory / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = attempts / f"{len(list(attempts.iterdir())) + 1:04d}"
    attempt.mkdir()
    work = root / "work"
    repository = work / "repository"
    staging = root / "staging"
    status = dict(**state, status="running", verified=False, started_at=utc_now(), attempt=str(attempt))
    write_json(result_path, status)
    name = f"leanlean-replay-{worker}-p{job['problem_idx']}-{uuid.uuid4().hex[:8]}"
    started_container = False
    try:
        archive = verify_archive(state)
        if staging.exists():
            shutil.rmtree(staging)
        extract_submission(archive, staging)
        baseline = Path(job["directory"]) / "before/repository"
        try:
            status["fixtures"] = restore_fixtures(staging, baseline, job["entry"])
        except ValueError as error:
            status.update(status="rejected", error=str(error))
            return status
        status["metrics"] = measure(staging, job["entry"], count)
        baseline_tokens = job["metrics"]["before"]["lean_tokens"]
        status["metrics"].update(
            baseline_tokens=baseline_tokens,
            compression_pct=100 * (1 - status["metrics"]["lean_tokens"] / baseline_tokens),
        )
        status["restored_source_hashes"] = source_hashes(staging)
        if not repository.exists():
            # Fresh mtimes make Lake check restored inputs against the seed's
            # trace hashes, including when resuming at a changed state.
            shutil.copytree(staging, repository, copy_function=shutil.copy)
            project = repository / job["entry"]["project_path"]
            seed = baseline / job["entry"]["project_path"] / ".lake"
            subprocess.run(["cp", "-a", "--reflink=auto", str(seed), str(project / ".lake")], check=True)
            # The seed was compiled from the baseline, not the current archive.
            baseline_files = source_files(baseline)
            if baseline_files.keys() - source_files(staging).keys():
                shutil.rmtree(project / ".lake/build", ignore_errors=True)
            status["cache"] = {"seed": str(seed)}
        else:
            status["cache"] = sync_sources(staging, repository)
        if source_hashes(repository) != status["restored_source_hashes"]:
            raise ValueError("Workspace does not match the restored checkpoint")
        # Do not let an earlier graph or failed scratch build masquerade as this state.
        shutil.rmtree(repository / job["entry"]["project_path"] / ".lean-strip", ignore_errors=True)
        write_json(work / "leanlean-project.json", job["entry"])
        runtime = plan["runtime"]
        command = container_command({**job, "directory": str(root)}, "work", runtime, run / "pipeline", name)
        command = command[: command.index(runtime["docker_image"])] + [
            "--volume",
            f"{attempt}:/output:rw",
            "--volume",
            f"{plan['extractor']}:/extractor:ro",
            "--env",
            "PYTHONPATH=/extractor/src",
            "--env",
            "PYTHONDONTWRITEBYTECODE=1",
            "--env",
            f"LEAN_STRIP_VERSION={plan['extractor_version']}",
            runtime["docker_image"],
            "python3",
            "/postprocess/replay_container.py",
        ]
        write_json(attempt / "container.json", {"command": command})
        started_container = True
        with (attempt / "container.log").open("w") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=68600)
        status.update(read_json(attempt / "checks.json"))
        status["container_returncode"] = completed.returncode
        if completed.returncode:
            status.update(status="error", verified=False)
    except Exception as error:
        status.update(status="error", verified=False, error=f"{type(error).__name__}: {error}")
    finally:
        if started_container:
            try:
                subprocess.run(
                    [plan["runtime"]["container_engine"], "rm", "--force", name],
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
            except Exception as error:
                status.update(status="error", verified=False, cleanup_error=str(error))
        status["finished_at"] = utc_now()
        status["artifacts"] = {
            str(p.relative_to(directory)): sha256(p) for p in sorted(attempt.iterdir()) if p.is_file()
        }
        write_json(result_path, status)
        if staging.exists():
            shutil.rmtree(staging)
    return status


def execute(plan: dict, run: Path, workers: int, limit: int | None) -> None:
    pending = queue.Queue()
    for record in plan["records"]:
        pending.put(record)
    total = sum(min(len(r["states"]), limit) if limit else len(r["states"]) for r in plan["records"])
    results = []
    lock = threading.Lock()

    def worker_loop(number: int) -> None:
        worker = f"w{number}"
        count = token_counter(Path(plan["tokens"]))
        while True:
            try:
                record = pending.get_nowait()
            except queue.Empty:
                return
            job = record["job"]
            root = run / "repositories" / job["key"]
            if source_hashes(Path(job["directory"]) / "before/repository") != record["baseline_sources"]:
                raise ValueError(f"Baseline sources changed: {job['key']}")
            # A previous process may have stopped midway through cache seeding
            # or compilation. Keep completed states, but seed the first pending
            # state from the known baseline rather than that unfinished cache.
            if (root / "work").exists():
                shutil.rmtree(root / "work")
            write_json(
                root / "timeline.json", dict(problem_idx=job["problem_idx"], key=job["key"], frames=record["frames"])
            )
            for state in record["states"][:limit]:
                print(f"[{worker}] picked P{job['problem_idx']}/{state['source_sha256'][:12]}", flush=True)
                result = replay_state(record, state, run, plan, worker, count)
                with lock:
                    results.append(
                        dict(
                            problem_idx=job["problem_idx"],
                            source_sha256=state["source_sha256"],
                            status=result["status"],
                            verified=result.get("verified", False),
                        )
                    )
                    write_json(
                        run / "summary.json",
                        dict(
                            completed=len(results),
                            total=total,
                            statuses=dict(Counter(r["status"] for r in results)),
                            results=results,
                        ),
                    )
                    print(f"STATUS P{job['problem_idx']} {state['source_sha256'][:12]}: {result['status']}", flush=True)
                    print(f"PROGRESS {len(results)}/{total}", flush=True)
                # An infrastructure failure could contaminate the cache. Keep its
                # evidence and start the next state from a fresh baseline build.
                if result["status"] in {"error", "source_changed"}:
                    shutil.rmtree(root / "work", ignore_errors=True)
            print(f"[{worker}] done", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(worker_loop, i + 1) for i in range(workers)]
        for future in futures:
            future.result()
    if any(r["status"] in {"error", "source_changed"} for r in results):
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--problem", type=int, action="append")
    parser.add_argument("--max-states", type=int, help="smoke test: first N distinct states per project")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or (args.max_states is not None and args.max_states < 1):
        parser.error("workers and max-states must be positive")
    import tomllib

    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(run / ".lock"):
        original = read_json(args.postprocess / "plan.json")
        extractor = args.extractor.resolve()
        if subprocess.check_output(["git", "-C", str(extractor), "status", "--porcelain"], text=True).strip():
            raise ValueError("Extractor must have a clean, pinned checkout")
        provenance = dict(
            postprocess_sha256=sha256(args.postprocess / "plan.json"),
            runtime=original["runtime"],
            extractor=str(extractor),
            extractor_sha256=tree_sha256(extractor / "src"),
            extractor_commit=subprocess.check_output(
                ["git", "-C", str(extractor), "rev-parse", "HEAD"], text=True
            ).strip(),
            extractor_version=tomllib.loads((extractor / "pyproject.toml").read_text())["project"]["version"],
            tokens=str(args.tokens.resolve()),
            tokens_sha256=sha256(args.tokens),
            pipeline_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "replay.py",
                    "replay_container.py",
                    "graphs_container.py",
                    "postprocess_container.py",
                    "postprocess.py",
                    "common.py",
                )
            },
            problems=sorted(args.problem) if args.problem else None,
        )
        if provenance["tokens_sha256"] != original["token_module_sha256"]:
            raise ValueError("Token counter differs from final-submission analysis")
        if (run / "plan.json").exists():
            plan = read_json(run / "plan.json")
            if any(plan[k] != value for k, value in provenance.items()):
                raise ValueError("Replay provenance changed; use a new run directory")
        else:
            jobs = [j for j in original["jobs"] if not args.problem or j["problem_idx"] in args.problem]
            if args.problem and {j["problem_idx"] for j in jobs} != set(args.problem):
                raise ValueError("Unknown project index")
            plan = dict(**provenance, records=inventory(jobs), created_at=utc_now())
            (run / "pipeline").mkdir()
            for name in provenance["pipeline_sha256"]:
                shutil.copyfile(Path(__file__).with_name(name), run / "pipeline" / name)
            write_json(run / "plan.json", plan)
        print(
            f"STATUS Replay: {sum(len(r['frames']) for r in plan['records'])} checkpoints, "
            f"{sum(len(r['states']) for r in plan['records'])} distinct states, {args.workers} workers",
            flush=True,
        )
        if not args.prepare_only:
            execute(plan, run, args.workers, args.max_states)


if __name__ == "__main__":
    main()
