"""Strip a frozen Palomar snapshot in Podman and export verified candidates.

Each project runs in its own container. Successful projects are copied into
bundle/projects, with their registry metadata and stripping reports beside them.
This command never adds candidates to a production dataset.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from leanlean.common import (
    exclusive_lock,
    python_package_sha256,
    read_json,
    sha256,
    tree_sha256,
    utc_now,
    within,
    write_json,
)
from leanlean.palomar_source import entry_key, source_identity

BUNDLE_EXCLUDES = {".git", ".lake", ".lean-strip", ".lean_strip", "__pycache__"}


def prepare_runtime(output: Path, storage_root: Path | None) -> tuple[dict, dict]:
    """Use the pinned lean-strip installation and its own runtime recipe."""
    try:
        import lean_strip
        from lean_strip import verify
        from lean_strip.container import _ensure_image
    except ImportError as error:
        raise RuntimeError("Install the LeanLean extra first: uv sync --extra leanlean") from error

    engine = shutil.which("podman")
    if engine is None:
        raise RuntimeError("LeanLean preprocessing requires Podman")
    podman = [engine]
    if storage_root:
        podman += ["--root", str(storage_root / "storage"), "--runroot", str(storage_root / "runtime")]
    # Native lean-strip finds Podman on PATH. Forward the selected store and
    # give each job a name so timeouts/cancellation can remove its container.
    binary = output / "bin/podman"
    binary.parent.mkdir(parents=True, exist_ok=True)
    prefix = shlex.join(podman)
    binary.write_text(
        '#!/bin/sh\nif [ "${1-}" = run ] && [ -n "${LEANLEAN_CONTAINER_NAME-}" ]; then\n'
        f'  shift\n  exec {prefix} run --name "$LEANLEAN_CONTAINER_NAME" "$@"\nfi\n'
        f'exec {prefix} "$@"\n'
    )
    binary.chmod(0o755)
    environment = {**os.environ, "PATH": f"{binary.parent}:{os.environ.get('PATH', '')}"}
    # Build once before starting workers, using upstream's recipe in the selected
    # store. This small internal API dependency is covered by the pinned revision.
    tag = _ensure_image(str(binary))
    image = subprocess.check_output([*podman, "image", "inspect", tag, "--format", "{{.Id}}"], text=True).strip()
    package = Path(lean_strip.__file__).resolve().parent
    distribution = importlib.metadata.distribution("lean-strip")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    runtime = {
        "image": image,
        "lean_strip_version": distribution.version,
        "lean_strip_commit": origin.get("vcs_info", {}).get("commit_id"),
        "lean_strip_package": str(package),
        "lean_strip_sha256": python_package_sha256(package),
        "elan_home": str(Path(os.environ.get("ELAN_HOME", str(Path.home() / ".elan"))).resolve()),
        "comparator_commit": verify.COMPARATOR[1],
        "nanoda_commit": verify.NANODA[1],
        "podman": podman,
    }
    return runtime, environment


def container_name(output: Path, key: str) -> str:
    return f"matharena-leanlean-{hashlib.sha256(str(output).encode()).hexdigest()[:10]}-{key.lower()}"


def strip_command(project: Path, comparator: Path, settings: dict) -> list[str]:
    command = [sys.executable, "-m", "lean_strip", str(project)]
    options = {
        "--comparator": comparator,
        "--container": "podman",
        "--image": settings["image"],
        "--memory": settings["memory"],
        "--cpus": settings["cpus"],
        "--jobs": settings["cpus"],
        "--timeout": settings["timeout"],
    }
    for flag, value in options.items():
        command += [flag, str(value)]
    return command + ["--retain-lean-only", "--clean-certify"]


def verification_command(output: Path, entry: dict, settings: dict) -> list[str]:
    """Check the applied tree, using upstream's verifier in the same runtime."""
    key = entry["key"]
    work, cache, result = (output / "work" / key, output / "cache" / key, output / "results" / key)
    project = work / entry["project_path"]
    elan = settings["elan_home"]
    command = [*settings["podman"], "run", "--rm", "--init", "--pids-limit=-1", "--userns=keep-id"]
    for flag, value in {
        "--name": container_name(output, key),
        "--cpus": settings["cpus"],
        "--memory": settings["memory"],
        "--memory-swap": settings["memory"],
        "--workdir": project,
    }.items():
        command += [flag, str(value)]
    for mount in [
        f"{work}:{work}:rw",
        f"{cache}:{cache}:rw",
        f"{elan}:{elan}:rw",
        f"{result}:/result:rw",
        f"{settings['lean_strip_package']}:/pipeline/lean_strip:ro",
        f"{output / 'pipeline/leanlean'}:/pipeline/leanlean:ro",
    ]:
        command += ["--volume", mount]
    for value in [f"ELAN_HOME={elan}", f"XDG_CACHE_HOME={cache}", "PYTHONPATH=/pipeline", "PYTHONDONTWRITEBYTECODE=1"]:
        command += ["--env", value]
    return command + [
        settings["image"],
        "python3",
        "-m",
        "leanlean.verify_project",
        "--project",
        str(project),
        "--comparator",
        str((work / entry["comparator_path"]).relative_to(project)),
        "--jobs",
        str(settings["cpus"]),
        "--timeout",
        str(settings["timeout"]),
        "--output",
        "/result/verification.json",
    ]


def dependency_graph_record(bundle: Path, entry: dict) -> dict | None:
    """Describe a saved graph without conflating its source with a reused tree."""
    provenance = bundle / "provenance" / entry["key"]
    path = provenance / "diagnostics/dependency-graph.json"
    if not path.is_file():
        return None
    graph = read_json(path)
    status = read_json(provenance / "status.json")
    source_commit = entry.get("commit")
    report = provenance / "strip-report.json"
    return {
        "key": entry["key"],
        "repository": entry.get("repository"),
        "source_commit": source_commit,
        "graph_source_commit": status.get("upstream_source_commit", source_commit),
        "path": path.relative_to(bundle).as_posix(),
        "sha256": sha256(path),
        "schema": graph.get("schema"),
        "method": graph.get("graph_method"),
        "nodes": len(graph["nodes"]),
        "edges": len(graph["edges"]),
        "report_path": report.relative_to(bundle).as_posix(),
        "report_sha256": sha256(report),
        "verification_origin": status.get("verification_origin", "applied_tree_check"),
    }


def write_dependency_graph_index(bundle: Path, projects: list[dict]) -> None:
    """Index graphs in project order; make absent diagnostics explicit."""
    write_json(
        bundle / "dependency-graphs.json",
        {
            "schema_version": 1,
            "graphs": [p["dependency_graph"] for p in projects if p.get("dependency_graph")],
            "missing": [p["key"] for p in projects if not p.get("dependency_graph")],
        },
    )

def export_project(snapshot: Path, output: Path, entry: dict) -> dict:
    """Copy sources, not build caches or original backups, into the review bundle."""
    key = entry["key"]
    bundle, result = output / "bundle", output / "results" / key
    destination = bundle / "projects" / key
    if destination.exists():
        raise ValueError(f"Bundle project already exists: {destination}")
    shutil.copytree(
        output / "work" / key,
        destination,
        symlinks=True,
        ignore=lambda directory, names: BUNDLE_EXCLUDES.intersection(names),
    )
    # Retain attribution even when --retain-lean-only removes it from the project.
    provenance = bundle / "provenance" / key
    provenance.mkdir(parents=True, exist_ok=True)
    shutil.copy2(snapshot / "records" / f"{key}.json", provenance / "registry.json")
    source = snapshot / "sources" / key
    metadata = source / entry["metadata_path"]
    shutil.copy2(metadata, provenance / metadata.name)
    license_path = entry.get("source", {}).get("license", {}).get("path")
    if license_path and within(source, license_path).is_file():
        shutil.copy2(within(source, license_path), provenance / "LICENSE")
    for filename in ("strip-report.json", "status.json"):
        shutil.copy2(result / filename, provenance / filename)
    for filename in ("verification.json", "run.log"):
        if (result / filename).is_file():
            shutil.copy2(result / filename, provenance / filename)
    # Keep the CLI's diagnostics outside the stripped project sources.
    diagnostics = output / "work" / key / entry["project_path"] / ".lean-strip/diagnostics"
    if diagnostics.is_dir():
        shutil.copytree(diagnostics, provenance / "diagnostics", symlinks=True)
    descriptor = {
        **entry,
        "project_directory": f"projects/{key}",
        "project_tree_sha256": tree_sha256(destination),
        "strip_status": read_json(result / "strip-report.json")["status"],
        "status": "verified",
    }
    graph = dependency_graph_record(bundle, descriptor)
    if graph is not None:
        descriptor["dependency_graph"] = graph
    write_json(provenance / "entry.json", descriptor)
    return descriptor


def process_entry(snapshot: Path, output: Path, entry: dict, settings: dict, environment: dict, cancelled) -> dict:
    key = entry["key"]
    result_root = output / "results" / key
    result_root.mkdir(parents=True, exist_ok=True)
    status_path = result_root / "status.json"
    descriptor_path = output / "bundle/provenance" / key / "entry.json"
    status = {"key": key, "status": "running", "stage": "strip", "started_at": utc_now()}
    try:
        if descriptor_path.exists():
            descriptor = read_json(descriptor_path)
            if tree_sha256(output / "bundle" / descriptor["project_directory"]) != descriptor["project_tree_sha256"]:
                raise ValueError("Previously exported project changed; use a new output directory")
            return {"key": key, "status": "verified", "reused": True}
        work = output / "work" / key
        if work.exists():
            # Keep unsuccessful attempts' sources and diagnostics when retrying.
            previous = output / "attempts" / key / uuid.uuid4().hex
            previous.mkdir(parents=True)
            work.rename(previous / "work")
            for path in result_root.iterdir():
                if path.is_file():
                    shutil.copy2(path, previous / path.name)
        shutil.copytree(snapshot / "sources" / key, work, symlinks=True)
        project = work / entry["project_path"]
        cache = output / "cache" / key
        cache.mkdir(parents=True, exist_ok=True)
        env = {
            **environment,
            "ELAN_HOME": settings["elan_home"],
            "XDG_CACHE_HOME": str(cache),
            "LEANLEAN_CONTAINER_NAME": container_name(output, key),
            # Source archives have no .git; do not discover the caller's enclosing repository.
            "GIT_CEILING_DIRECTORIES": str(output / "work"),
        }
        write_json(result_root / "input.json", entry)
        write_json(status_path, status)
        with (result_root / "run.log").open("a", buffering=1) as log:
            if cancelled.is_set():
                raise RuntimeError("Batch cancelled")
            result = subprocess.run(
                strip_command(project, work / entry["comparator_path"], settings),
                env=env,
                check=False,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=settings["timeout"] * 4,
            )
            report_path = project / ".lean-strip/report.json"
            report = read_json(report_path) if report_path.exists() else {}
            write_json(result_root / "strip-report.json", report)
            status["strip_status"] = report.get("status")
            if result.returncode or report.get("status") not in {"stripped", "repaired"} or report.get("dry_run"):
                raise RuntimeError(f"lean-strip exited {result.returncode}: {report.get('status', 'no report')}")
            if (report.get("comparator") or {}).get("status") != "accepted":
                raise RuntimeError("lean-strip did not report Comparator acceptance")
            for field in ("challenge_path", "comparator_path", "toolchain_path"):
                relative = entry[field]
                if sha256(work / relative) != entry["file_hashes"][relative]:
                    raise ValueError(f"Stripping changed protected input: {relative}")
            status["stage"] = "verify"
            write_json(status_path, status)
            if cancelled.is_set():
                raise RuntimeError("Batch cancelled")
            subprocess.run(
                verification_command(output, entry, settings),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=settings["timeout"] * 2,
                check=True,
            )
        status["verification"] = read_json(result_root / "verification.json")
        if status["verification"]["status"] != "accepted":
            raise RuntimeError("Applied-tree Comparator did not accept the project")
        status.update(status="verified", finished_at=utc_now())
        write_json(status_path, status)
        export_project(snapshot, output, entry)
    except Exception as error:  # noqa: BLE001 -- keep failures local to one project and retain its diagnostics
        if isinstance(error, subprocess.TimeoutExpired):
            subprocess.run(
                [*settings["podman"], "rm", "--force", container_name(output, key)], capture_output=True, check=False
            )
        status.update(status="failed", error=str(error), finished_at=utc_now())
        write_json(status_path, status)
    return status


def write_bundle_manifest(output: Path, entries: list[dict], statuses: dict[str, dict]) -> dict:
    projects = []
    rows = []
    for entry in entries:
        key = entry["key"]
        status = statuses.get(key, {"status": "pending"})
        descriptor_path = output / "bundle/provenance" / key / "entry.json"
        if status["status"] == "verified" and descriptor_path.exists():
            projects.append(read_json(descriptor_path))
        rows.append(
            {
                "key": key,
                "github_url": entry["github_url"],
                "solution_path": entry["solution_path"],
                "challenge_path": entry["challenge_path"],
                "status": status["status"],
                "error": status.get("error", ""),
            }
        )
    manifest = {
        "schema_version": 1,
        "updated_at": utc_now(),
        "expected_projects": len(entries),
        "status": "complete" if len(projects) == len(entries) else "incomplete",
        "projects": projects,
        "dependency_graph_index": "dependency-graphs.json",
    }
    bundle = output / "bundle"
    write_dependency_graph_index(bundle, projects)
    write_json(bundle / "manifest.json", manifest)
    write_json(output / "summary.json", {"updated_at": utc_now(), "entries": rows})
    temporary = bundle / ".index.csv.tmp"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(bundle / "index.csv")
    return manifest


def strip_snapshot(
    snapshot: Path,
    output: Path,
    *,
    workers: int,
    cpus: int,
    memory: str,
    timeout: int,
    storage_root: Path | None = None,
) -> dict:
    if snapshot == output or snapshot.is_relative_to(output) or output.is_relative_to(snapshot):
        raise ValueError("Snapshot and stripping output must be separate, non-nested directories")
    with exclusive_lock(output / ".strip.lock"), exclusive_lock(snapshot / ".fetch.lock"):
        manifest = read_json(snapshot / "snapshot.json")
        entries = []
        for projection in manifest["entries"]:
            key = entry_key(projection)
            entry = read_json(snapshot / "entries" / f"{key}.json")
            if entry.get("fetch_status") != "ready" or source_identity(entry) != source_identity(projection):
                raise ValueError(f"Source is not ready or does not match the snapshot: {key}")
            if sha256(snapshot / "records" / f"{key}.json") != entry["record_sha256"]:
                raise ValueError(f"Registry record changed: {key}")
            if tree_sha256(snapshot / "sources" / key) != entry["source_tree_sha256"]:
                raise ValueError(f"Source tree changed: {key}")
            entries.append(entry)
        if not entries:
            raise ValueError("Snapshot has no entries")
        runtime, environment = prepare_runtime(output, storage_root)
        code_files = [
            Path(__file__).parent / name
            for name in ("__init__.py", "common.py", "palomar_source.py", "strip.py", "verify_project.py")
        ]
        code_digest = hashlib.sha256(b"".join(p.name.encode() + p.read_bytes() for p in code_files)).hexdigest()
        settings = {
            "schema_version": 2,
            **runtime,
            "snapshot": str(snapshot),
            "snapshot_sha256": sha256(snapshot / "snapshot.json"),
            "storage_root": str(storage_root) if storage_root else None,
            "pipeline_sha256": code_digest,
            "workers": workers,
            "cpus": cpus,
            "memory": memory,
            "timeout": timeout,
        }
        config_path = output / "run.json"
        if config_path.exists() and read_json(config_path) != settings:
            raise ValueError("Run inputs/settings changed; choose a new output directory")
        write_json(config_path, settings)
        package = output / "pipeline/leanlean"
        package.mkdir(parents=True, exist_ok=True)
        for path in code_files:
            if path.resolve() != (package / path.name).resolve():
                shutil.copy2(path, package / path.name)
        cancelled = threading.Event()
        statuses = {}
        write_bundle_manifest(output, entries, statuses)
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        try:
            futures = [
                pool.submit(process_entry, snapshot, output, entry, settings, environment, cancelled)
                for entry in entries
            ]
            for future in concurrent.futures.as_completed(futures):
                status = future.result()
                statuses[status["key"]] = status
                write_bundle_manifest(output, entries, statuses)
                print(
                    f"[{len(statuses)}/{len(entries)}] {status['key']}: {status['status']} {status.get('error', '')}",
                    flush=True,
                )
        except KeyboardInterrupt:
            cancelled.set()
            for future in futures:
                future.cancel()
            for entry in entries:
                subprocess.run(
                    [*runtime["podman"], "rm", "--force", container_name(output, entry["key"])],
                    capture_output=True,
                    check=False,
                )
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
        return write_bundle_manifest(output, entries, statuses)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--storage-root", type=Path, help="Optional dedicated Podman storage on a spacious filesystem")
    parser.add_argument("--cpus", type=int, default=8, help="CPUs per project container")
    parser.add_argument("--memory", default="64g", help="Memory per project container")
    parser.add_argument("--timeout", type=int, default=21600, help="Seconds per strip/build/check command")
    args = parser.parse_args()
    if min(args.workers, args.cpus, args.timeout) < 1 or not re.fullmatch(r"[1-9]\d*[mg]", args.memory):
        parser.error("Workers, CPUs and timeout must be positive; memory must look like 64g or 512m")
    result = strip_snapshot(
        args.snapshot.resolve(),
        args.output.resolve(),
        workers=args.workers,
        cpus=args.cpus,
        memory=args.memory,
        timeout=args.timeout,
        storage_root=args.storage_root.resolve() if args.storage_root else None,
    )
    return int(result["status"] != "complete")


if __name__ == "__main__":
    raise SystemExit(main())
