"""Run lean-strip's graph-only CLI on retained, verified baseline/final builds.

This step uses the verification image and resource limits. Each invocation writes
its own provenance and logs; successful graphs can be reused on resume only when
their recorded checksums and extractor provenance still match.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import shutil
import subprocess
import uuid
from pathlib import Path

from leanlean.common import exclusive_lock, read_json, sha256, tree_sha256, utc_now, write_json
from leanlean.postprocess import container_command


def extract(job: dict, phase: str, runtime: dict, run: Path, extractor: Path, provenance: dict) -> dict:
    workspace = Path(job["directory"]) / phase
    record = run / f'P{job["problem_idx"]}-{phase}.json'
    graph = workspace / "dependency-graph.json"
    if record.exists():
        previous = read_json(record)
        if previous.get("status") == "passed":
            if previous["provenance"] != provenance or sha256(graph) != previous["graph_sha256"]:
                raise ValueError(f"Graph or provenance changed: {record}")
            return previous
        raise ValueError(f"Previous extraction did not pass: {record}; use a new --run directory")
    name = f'leanlean-graph-p{job["problem_idx"]}-{phase}-{uuid.uuid4().hex[:8]}'
    command = container_command(job, phase, runtime, run / "pipeline", name)
    image_index = command.index(runtime["docker_image"])
    command = command[:image_index] + [
        "--volume",
        f"{extractor}:/extractor:ro",
        "--env",
        "PYTHONPATH=/extractor/src",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f'LEAN_STRIP_VERSION={provenance["version"]}',
        runtime["docker_image"],
        "python3",
        "/postprocess/graphs_container.py",
    ]
    status = dict(
        problem_idx=job["problem_idx"],
        key=job["key"],
        phase=phase,
        provenance=provenance,
        command=command,
        started_at=utc_now(),
        status="running",
    )
    write_json(record, status)
    print(f'STATUS P{job["problem_idx"]} {phase}: extracting graph', flush=True)
    try:
        with record.with_suffix(".log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=43200)
        status.update(read_json(workspace / "graph-status.json"))
        status["container_returncode"] = result.returncode
        if result.returncode != 0 or status["status"] != "passed":
            raise RuntimeError(f"Extraction failed; see {record.with_suffix('.log')}")
        status["graph_path"] = str(graph)
        status["graph_sha256"] = sha256(graph)
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
    finally:
        subprocess.run([runtime["container_engine"], "rm", "--force", name], capture_output=True, timeout=30)
    status["finished_at"] = utc_now()
    write_json(record, status)
    print(f'STATUS P{job["problem_idx"]} {phase}: {status["status"]}', flush=True)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True, help="clean lean-strip release checkout")
    parser.add_argument("--run", type=Path, required=True, help="logs and immutable extraction provenance")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--problem", type=int, action="append", help="restrict to these problem indices")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    import tomllib

    source = args.extractor.resolve()
    version = tomllib.loads((source / "pyproject.toml").read_text())["project"]["version"]
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip():
        raise ValueError("Extractor checkout has uncommitted changes")
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.postprocess / "plan.json")
    jobs = [read_json(Path(j["directory"]) / "result.json") for j in plan["jobs"]]
    if args.problem:
        jobs = [j for j in jobs if j["problem_idx"] in args.problem]
        if {j["problem_idx"] for j in jobs} != set(args.problem):
            raise ValueError("Unknown problem index")
    if any(j["status"] != "verified" for j in jobs):
        raise ValueError("Only verified projects can be extracted")
    with exclusive_lock(args.postprocess / ".lock"):
        provenance = {
            "version": version,
            "commit": commit,
            "source_sha256": tree_sha256(source / "src"),
            "image": plan["runtime"]["docker_image"],
        }
        if not (run / "pipeline").exists():
            (run / "pipeline").mkdir()
            shutil.copyfile(Path(__file__).with_name("graphs_container.py"), run / "pipeline" / "graphs_container.py")
        write_json(run / "provenance.json", provenance)
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(extract, j, phase, plan["runtime"], run, source, provenance)
                for j in jobs
                for phase in ("before", "after")
            ]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                write_json(run / "summary.json", {"completed": len(results), "total": len(futures), "results": results})
                print(f"PROGRESS {len(results)}/{len(futures)}", flush=True)
        passed = all(r["status"] == "passed" for r in results)
        if not args.problem:
            summary = read_json(args.postprocess / "summary.json")
            summary.update(graph_extraction="complete" if passed else "failed", graph_run=str(run))
            write_json(args.postprocess / "summary.json", summary)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
