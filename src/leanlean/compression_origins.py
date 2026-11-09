"""Classify verified compression with the vendored lean-lean classifier."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import shutil
import subprocess
import uuid
from pathlib import Path

from leanlean.checkpoint_renames import rename_map
from leanlean.common import exclusive_lock, read_json, result_attempt, sha256, tree_sha256, utc_now, write_json
from leanlean.postprocess import container_command
from matharena.json_zst import load_json_zst

# lean-lean's classifier closure, copied verbatim; MANIFEST.json pins each file.
REFERENCE = Path(__file__).resolve().parents[2] / "third_party" / "lean-lean"


def check_reference(root: Path) -> dict:
    """Refuse a vendored tree whose files differ from its manifest."""
    manifest = read_json(root / "MANIFEST.json")
    files = {
        p.relative_to(root).as_posix()
        for p in root.rglob("*")
        if p.is_file() and p.name != "MANIFEST.json" and "__pycache__" not in p.parts
    }
    if files != set(manifest["files"]) or any(sha256(root / f) != row["sha256"] for f, row in manifest["files"].items()):
        raise ValueError(f"Vendored lean-lean files differ from {root / 'MANIFEST.json'}")
    return manifest


def graph_position_audit(directory: Path, entry: dict) -> dict:
    """Reject scratch normalization that would shift graph source line ranges."""
    audit = {}
    for phase in ("before", "after"):
        report = read_json(directory / phase / "repository" / entry["project_path"] / ".lean-strip" / "report.json")
        normalization = report["engine_report"]["grind_normalization"]
        changes = []
        for file in normalization["files"]:
            for call in file["calls"]:
                if not call.get("persisted_after_file_decision"):
                    continue
                edit = call["normalization"]
                if edit["original_call"].count("\n") != edit["replacement"].count("\n"):
                    raise ValueError(f'{phase}: graph normalization shifts source lines in {file["path"]}')
                changes.append(
                    {"path": file["path"], "original_call": edit["original_call"], "replacement": edit["replacement"]}
                )
        audit[phase] = {"line_preserving": True, "normalized_calls": changes}
    return audit


def classify(job: dict, runtime: dict, run: Path, reference: Path, provenance: dict,
             program_sites_from: Path | None = None) -> dict:
    output = run / job["key"]
    output.mkdir(exist_ok=True)
    record = output / "status.json"
    inputs = Path(job["directory"])
    position_audit = graph_position_audit(inputs, job["entry"])
    write_json(output / "graph-position-audit.json", position_audit)
    if not record.exists():
        # Namespace-only renames traced through the attempt's checkpoints.
        attempt = result_attempt(load_json_zst(Path(job["result_path"]))["history"][0][0])
        renames = rename_map(attempt, job["entry"]["project_path"]) if (attempt / "checkpoints.jsonl").is_file() else {
            "renames": {}, "checkpoints": 0, "steps": [], "ambiguous_short_names": 0, "attempts": []}
        write_json(output / "renames.json", renames)
        previous = program_sites_from / job["key"] if program_sites_from else None
        if previous and (previous / "status.json").is_file() and (previous / "program-sites").is_dir():
            # Macro/program-site measurement depends only on the after source, not on names.
            if read_json(previous / "status.json")["inputs"].get("result.json") == sha256(inputs / "result.json"):
                shutil.copytree(previous / "program-sites", output / "program-sites", dirs_exist_ok=True)
    checksums = {"renames.json": sha256(output / "renames.json")} | {
        str(path.relative_to(inputs)): sha256(path)
        for path in [
            inputs / "result.json",
            *[
                inputs / phase / name
                for phase in ("before", "after")
                for name in ("dependency-graph.json", "original-sources.tar.gz")
            ],
        ]
    }
    if record.exists():
        previous = read_json(record)
        if previous["status"] == "passed" and previous["inputs"] == checksums and previous["provenance"] == provenance:
            if sha256(output / "origins.json") != previous["origins_sha256"]:
                raise ValueError(f"Origin data changed: {output}")
            return read_json(output / "origins.json")
        raise ValueError(f"Existing analysis cannot be reused: {output}; use a new --run directory")
    name = f'leanlean-origins-p{job["problem_idx"]}-{uuid.uuid4().hex[:8]}'
    command = container_command(job, "after", runtime, run / "pipeline", name)
    # Analysis needs source archives and graphs only, mounted read-only.
    command[command.index(f'{inputs / "after"}:/work:rw')] = f"{inputs}:/inputs:ro"
    command[command.index("--workdir") + 1] = "/analysis"
    index = command.index(runtime["docker_image"])
    command = command[:index] + [
        "--volume",
        f"{output}:/analysis:rw",
        "--volume",
        f"{reference}:/reference:ro",
        "--volume",
        f'{run / "vendor"}:/vendor:ro',
        "--env",
        "PYTHONPATH=/reference:/reference/src:/vendor",
        "--env",
        "PYTHONDONTWRITEBYTECODE=1",
        "--env",
        f'LEANLEAN_ANALYSIS_MODEL={provenance.get("model", "leanlean")}',
        runtime["docker_image"],
        "python3",
        "/postprocess/origins_container.py",
    ]
    status = dict(
        status="running",
        problem_idx=job["problem_idx"],
        key=job["key"],
        directory=str(inputs),
        inputs=checksums,
        provenance=provenance,
        command=command,
        started_at=utc_now(),
    )
    write_json(record, status)
    print(f'STATUS P{job["problem_idx"]}: classifying compression', flush=True)
    try:
        with (output / "container.log").open("w") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=14400)
        if result.returncode != 0:
            raise RuntimeError(f"Analysis exited {result.returncode}; see {output / 'container.log'}")
        origins = read_json(output / "origins.json")
        status.update(status="passed", origins_sha256=sha256(output / "origins.json"))
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        origins = {"status": "failed", "problem_idx": job["problem_idx"], "key": job["key"], "error": status["error"]}
    finally:
        subprocess.run([runtime["container_engine"], "rm", "--force", name], capture_output=True, timeout=30)
    status["finished_at"] = utc_now()
    write_json(record, status)
    print(f'STATUS P{job["problem_idx"]}: {status["status"]}', flush=True)
    return origins


def summarize(rows: list[dict], total: int, output: Path) -> None:
    passed = sorted((r for r in rows if r["status"] == "passed"), key=lambda r: r["problem_idx"])
    summary = {"completed": len(rows), "total": total, "passed": len(passed), "results": rows}
    if passed:
        summary["mean_percentage_points"] = {
            key: sum(r["categories"][key] / r["baseline_tokens"] * 100 for r in passed) / len(passed)
            for key in passed[0]["categories"]
        }
        records = [
            {k: r[k] for k in ("problem_idx", "key", "baseline_tokens", "final_tokens", "tokens_saved")}
            | r["categories"]
            for r in passed
        ]
        with (output / "origins.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
    write_json(output / "summary.json", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postprocess", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--problem", type=int, action="append")
    parser.add_argument("--program-sites-from", type=Path,
                        help="earlier origins run whose program-site measurements of the same submissions are reused")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    import yaml

    run = args.run.resolve()
    reference = run / "pipeline" / "lean-lean"
    run.mkdir(parents=True, exist_ok=True)
    plan = read_json(args.postprocess / "plan.json")
    # A job may name a staged copy of its verification directory, e.g. one holding a regraphed input.
    jobs = [read_json(Path(j["directory"]) / "result.json") | {"directory": j["directory"]} for j in plan["jobs"]]
    if args.problem:
        jobs = [j for j in jobs if j["problem_idx"] in args.problem]
        if {j["problem_idx"] for j in jobs} != set(args.problem):
            raise ValueError("Unknown problem index")
    with exclusive_lock(run / ".lock"):
        if not (run / "pipeline").exists():
            (run / "pipeline").mkdir()
            shutil.copyfile(Path(__file__).with_name("origins_container.py"), run / "pipeline" / "origins_container.py")
            shutil.copyfile(Path(__file__).with_name("graph_identity.py"), run / "pipeline" / "graph_identity.py")
            shutil.copyfile(Path(__file__).with_name("definition_savings.py"), run / "pipeline" / "definition_savings.py")
            shutil.copyfile(Path(__file__).with_name("macro_driver.py"), run / "pipeline" / "macro_driver.py")
            check_reference(REFERENCE)
            shutil.copytree(REFERENCE, reference, ignore=shutil.ignore_patterns("__pycache__"))
            shutil.copytree(
                Path(yaml.__file__).parent,
                run / "vendor" / "yaml",
                ignore=shutil.ignore_patterns("__pycache__", "*.so"),
            )
        manifest = check_reference(reference)
        provenance = {
            "reference_sha256": tree_sha256(reference),
            "reference_lean_lean_head": manifest["head"],
            "image": plan["runtime"]["docker_image"],
            "adapter_sha256": sha256(run / "pipeline" / "origins_container.py"),
            "graph_identity_sha256": sha256(run / "pipeline" / "graph_identity.py"),
            "definition_savings_sha256": sha256(run / "pipeline" / "definition_savings.py"),
            "macro_driver_sha256": sha256(run / "pipeline" / "macro_driver.py"),
            "yaml_version": yaml.__version__,
            "vendor_sha256": tree_sha256(run / "vendor"),
        }
        write_json(run / "provenance.json", provenance)
        rows = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(classify, j, plan["runtime"], run, reference, provenance, args.program_sites_from)
                       for j in jobs]
            for future in concurrent.futures.as_completed(futures):
                rows.append(future.result())
                summarize(rows, len(jobs), run)
                print(f"PROGRESS {len(rows)}/{len(jobs)}", flush=True)
        passed = all(row["status"] == "passed" for row in rows)
        if not args.problem:
            with exclusive_lock(args.postprocess / ".lock"):
                summary = read_json(args.postprocess / "summary.json")
                summary.update(compression_origins="complete" if passed else "failed", origins_run=str(run))
                write_json(args.postprocess / "summary.json", summary)
        if not passed:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
