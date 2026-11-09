"""Verify final submissions and retain source/build pairs for graph extraction.

Grade one repository directory or source archive directly or read MathArena results from a combined-results
index. The raw submission archive is never edited. Verification restores the registered challenge and Comparator
configuration in a fresh workspace; each baseline and final keeps its own Lake
build. No model credentials or network access are supplied to these containers.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import importlib.util
import os
import posixpath
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import uuid
import warnings
from pathlib import Path, PurePosixPath

from leanlean.common import exclusive_lock, read_json, sha256, tree_sha256, utc_now, within, write_json
from leanlean.evaluation import write_submission_archive
from matharena.json_zst import OUTPUT_JSON_SUFFIX, load_json_zst, output_json_stem


def validate_lakefile(repository: Path, baseline: Path, entry: dict) -> dict:
    """Audit layout/build options; warn on options while retaining pinned inputs."""
    relative = entry["lakefile_path"]
    target, source = within(repository, relative), within(baseline, relative)
    if not target.is_file():
        raise ValueError(f"Build configuration changed: {relative} is missing")
    # Lake can prefer an executable lakefile over TOML. Do not let an added
    # alternate file bypass validation of the registered configuration.
    alternate = str(Path(relative).with_name("lakefile.lean" if source.suffix == ".toml" else "lakefile.toml"))
    other, registered = within(repository, alternate), within(baseline, alternate)
    if other.exists() != registered.exists() or (
        other.exists() and (not other.is_file() or sha256(other) != sha256(registered))
    ):
        raise ValueError(f"Build configuration changed: alternate {alternate}")
    submitted, expected = sha256(target), sha256(source)
    audit = {
        "status": "unchanged" if submitted == expected else "modified",
        "submitted_sha256": submitted,
        "registered_sha256": expected,
        "restored": False,
        "policy": "library_layout_with_build_option_warnings",
        "warnings": [],
    }
    if submitted == expected:
        return audit

    def warn(message: str) -> None:
        audit["warnings"].append(message)
        warnings.warn(message, UserWarning, stacklevel=2)

    if source.suffix != ".toml":
        # An executable lakefile is arbitrary unscored Lean code, so verification
        # always uses the registered one.
        shutil.copyfile(source, target)
        audit.update(restored=True, policy="restore_executable")
        warn(f"Build configuration changed: executable {relative}; restored the registered file")
        return audit
    try:
        original, final = (tomllib.loads(p.read_text(encoding="utf-8")) for p in (source, target))
    except tomllib.TOMLDecodeError as error:
        raise ValueError(f"Build configuration changed: invalid {relative}: {error}") from error
    layout = {"defaultTargets", "lean_lib"}
    if original.get("require") != final.get("require"):
        raise ValueError(f"Build configuration changed: {relative} changes pinned dependencies")
    changed_options = sorted(
        k for k in original.keys() | final.keys()
        if k not in layout and original.get(k) != final.get(k)
    )
    if changed_options:
        audit["changed_build_options"] = changed_options
        warn(f"Build options changed in {relative}: {', '.join(changed_options)}; continuing to fresh verification")
    targets = final.get("defaultTargets", [])
    if not isinstance(targets, list) or any(not isinstance(value, str) for value in targets):
        raise ValueError(f"Build configuration changed: invalid defaultTargets in {relative}")

    def libraries(config: dict) -> dict:
        result = {}
        values = config.get("lean_lib", [])
        if not isinstance(values, list):
            raise ValueError(f"Build configuration changed: invalid lean_lib in {relative}")
        for library in values:
            if not isinstance(library, dict) or not isinstance(library.get("name"), str):
                raise ValueError(f"Build configuration changed: invalid library in {relative}")
            name = library["name"]
            if not name or name in result:
                raise ValueError(f"Build configuration changed: duplicate/empty library in {relative}")
            directory = library.get("srcDir", ".")
            if (
                not isinstance(directory, str)
                or "\\" in directory
                or PurePosixPath(directory).is_absolute()
                or any(part.startswith(".") for part in PurePosixPath(directory).parts)
                or not (target.parent / directory).resolve().is_relative_to(target.parent.resolve())
            ):
                raise ValueError(f"Build configuration changed: library {name} srcDir is outside scored sources")
            result[name] = library
        return result

    before, after = libraries(original), libraries(final)
    challenge = entry.get("challenge_module") or str(
        Path(entry["challenge_path"]).relative_to(entry["project_path"] or ".").with_suffix("")
    ).replace("/", ".")

    def includes_challenge(library: dict) -> bool:
        roots = library.get("roots", [library["name"]])
        globs = library.get("globs", [])
        if any(
            not isinstance(values, list) or any(not isinstance(v, str) for v in values)
            for values in (roots, globs)
        ):
            raise ValueError(f"Build configuration changed: invalid library roots/globs in {relative}")
        prefixes = [root.removesuffix(".+").removesuffix(".*") for root in roots + globs]
        return any(
            prefix in {"", "*", "+"} or challenge == prefix or challenge.startswith(prefix + ".")
            for prefix in prefixes
        )

    protected = {name for name, library in before.items() if includes_challenge(library)}
    # Restoring Challenge.lean is insufficient if Lake can resolve the module
    # to a different source directory or a newly introduced library instead.
    if any(after.get(name) != before[name] for name in protected) or any(
        includes_challenge(library) and name not in protected for name, library in after.items()
    ):
        raise ValueError(f"Build configuration changed: {relative} redirects the registered challenge module")
    fields = {"name", "roots", "globs", "srcDir"}
    for name, library in after.items():
        if {k: v for k, v in library.items() if k not in fields} != {
            k: v for k, v in before.get(name, {}).items() if k not in fields
        }:
            warn(f"Build options changed in {relative}: library {name}; continuing to fresh verification")
    audit["libraries_added"] = sorted(after.keys() - before.keys())
    audit["libraries_removed"] = sorted(before.keys() - after.keys())
    audit["libraries_modified"] = sorted(name for name in before.keys() & after.keys() if before[name] != after[name])
    audit["default_targets_changed"] = original.get("defaultTargets") != final.get("defaultTargets")
    return audit


def extract_submission(archive: Path, destination: Path) -> list[dict]:
    """Extract sources, materializing internal file aliases without creating links."""
    destination.mkdir(parents=True, exist_ok=False)
    audit = []
    with tarfile.open(archive) as stream:
        members = {}
        for member in stream:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or path.as_posix() in members:
                raise ValueError(f"Unsafe or duplicate archive path: {member.name}")
            members[path.as_posix()] = member
            if any(part in {".lake", ".git", ".lean-strip"} for part in path.parts):
                raise ValueError(f"Submission contains generated state: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"Submission contains a nonregular entry: {member.name}")
        links = {name for name, member in members.items() if member.issym() or member.islnk()}
        for name in members:
            if any(parent.as_posix() in links for parent in PurePosixPath(name).parents):
                raise ValueError(f"Archive path has a link ancestor: {name}")

        def resolve(name: str, seen: set[str]) -> tarfile.TarInfo:
            if name in seen:
                raise ValueError(f"Cyclic archive link: {name}")
            member = members.get(name)
            if member is None:
                raise ValueError(f"Missing archive link target: {name}")
            if member.isfile():
                return member
            if name not in links:
                raise ValueError(f"Archive link target is not a file: {name}")
            target = member.linkname
            if PurePosixPath(target).is_absolute():
                raise ValueError(f"Unsafe archive link target: {target}")
            if member.issym():
                target = str(PurePosixPath(name).parent / target)
            target = posixpath.normpath(target)
            if target == ".." or target.startswith("../"):
                raise ValueError(f"Unsafe archive link target: {target}")
            return resolve(target, seen | {name})

        resolved = {name: resolve(name, set()) for name in links}
        for name, member in members.items():
            if name not in links:
                stream.extract(member, destination, filter="data")
        for name, member in resolved.items():
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with stream.extractfile(member) as source, target.open("xb") as output:
                shutil.copyfileobj(source, output)
            target.chmod(member.mode & 0o777)
            message = f"Materialized internal archive link {name} -> {members[name].linkname} as a regular file"
            audit.append({"path": name, "target": members[name].linkname, "warning": message})
            warnings.warn(message, UserWarning, stacklevel=2)
    return audit


def restore_fixtures(repository: Path, baseline: Path, entry: dict) -> dict:
    audit = {}
    for field in ("challenge_path", "comparator_path"):
        relative = entry[field]
        target, source = within(repository, relative), within(baseline, relative)
        before = sha256(target) if target.is_file() else None
        expected = sha256(source)
        audit[relative] = {
            "status": "missing" if before is None else "unchanged" if before == expected else "modified",
            "submitted_sha256": before,
            "registered_sha256": expected,
            "restored": before != expected,
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Keep the registered environment; accept declarative source reorganization.
    for relative in (
        entry["toolchain_path"],
        str(Path(entry["project_path"]) / "lake-manifest.json"),
    ):
        target = within(repository, relative)
        if not target.is_file() or sha256(target) != sha256(within(baseline, relative)):
            raise ValueError(f"Build configuration changed: {relative}")
    audit[entry["lakefile_path"]] = validate_lakefile(repository, baseline, entry)
    return audit


def token_counter(source: Path):
    spec = importlib.util.spec_from_file_location("leanlean_reference_tokens", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.count_lean_tokens_in_source


def measure(repository: Path, entry: dict, count) -> dict:
    project = within(repository, entry["project_path"]) if entry["project_path"] else repository
    challenge = within(repository, entry["challenge_path"])
    files = {}
    for directory, dirs, names in os.walk(project, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(names):
            path = Path(directory) / name
            if path.suffix != ".lean" or name == "lakefile.lean" or path == challenge:
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlink: {path}")
            files[str(path.relative_to(project))] = count(path.read_text(encoding="utf-8"))
    return {"lean_tokens": sum(files.values()), "files": files, "lean_file_count": len(files)}


def archive_sources(repository: Path, destination: Path) -> None:
    with tarfile.open(destination, "w:gz") as stream:
        for path in sorted(repository.rglob("*")):
            relative = path.relative_to(repository)
            if any(part in {".lake", ".git", ".lean-strip"} for part in relative.parts):
                continue
            if path.is_file():
                stream.add(path, arcname=str(relative), recursive=False)


def indexed_submissions(index: Path, entries: list[dict]):
    for raw_idx, result in sorted(read_json(index).items(), key=lambda item: int(item[0])):
        idx = int(raw_idx)
        if not 1 <= idx <= len(entries):
            raise ValueError(f"Unknown problem ID: {idx}")
        path = Path(result["path"])
        if sha256(path) != result["sha256"]:
            raise ValueError(f"Result checksum mismatch: {path}")
        runs = load_json_zst(path)["history"]
        # Grading verifies one submission per problem; run with --n 1.
        if len(runs) != 1:
            raise ValueError(f"P{idx} has {len(runs)} runs; LeanLean grades exactly one")
        submission = runs[0][0]["leanlean_submission"]
        entry = entries[idx - 1]
        if entry["key"] != submission["key"] or entry["project_tree_sha256"] != submission["baseline_sha256"]:
            raise ValueError(f"Wrong baseline for P{idx}")
        yield idx, Path(submission["archive"]), submission["sha256"], {
            "result_path": str(path), "result_sha256": result["sha256"],
        }


def prepare(
    index: Path | None, bundle: Path, output: Path, tokens: Path, definition_proofs: Path,
    *, direct: tuple[int, Path, str] | None = None,
) -> list[dict]:
    entries = read_json(bundle / "manifest.json")["projects"]
    count = token_counter(tokens)
    if direct is not None:
        idx, archive, digest = direct
        if not 1 <= idx <= len(entries):
            raise ValueError(f"Unknown problem ID: {idx}")
        submissions = [(idx, archive, digest, {})]
    elif index is not None:
        submissions = indexed_submissions(index, entries)
    else:
        raise ValueError("A result index or a submission archive is required")
    jobs = []
    for idx, archive, digest, result_receipt in submissions:
        entry = entries[idx - 1]
        if sha256(archive) != digest:
            raise ValueError(f"Submission checksum mismatch: {archive}")
        baseline = within(bundle, entry["project_directory"])
        if tree_sha256(baseline) != entry["project_tree_sha256"]:
            raise ValueError(f'Baseline tree changed: {entry["key"]}')
        directory = output / "repositories" / entry["key"]
        directory.mkdir(parents=True)
        archive_warnings = []
        for phase in ("before", "after"):
            workspace = directory / phase
            workspace.mkdir()
            repository = workspace / "repository"
            if phase == "before":
                shutil.copytree(baseline, repository)
            else:
                archive_warnings = extract_submission(archive, repository)
        fixtures = restore_fixtures(directory / "after/repository", baseline, entry)
        # Reviewed equivalence proofs are only tried after `rfl` and are kernel-checked.
        proofs = definition_proofs / f'{entry["key"]}.json'
        write_json(directory / "after/definition-proofs.json", read_json(proofs) if proofs.is_file() else {})
        metrics = {phase: measure(directory / phase / "repository", entry, count) for phase in ("before", "after")}
        before, after = (metrics[phase]["lean_tokens"] for phase in ("before", "after"))
        if before <= 0:
            raise ValueError(f'Empty baseline metric: {entry["key"]}')
        metrics.update(tokens_saved=before - after, compression_pct=100 * (before - after) / before)
        job = {
            "problem_idx": idx,
            "key": entry["key"],
            "entry": entry,
            "directory": str(directory),
            **result_receipt,
            "submission_archive": str(archive),
            "submission_sha256": digest,
            "fixtures": fixtures,
            "archive_warnings": archive_warnings,
            "metrics": metrics,
            "definition_proofs_sha256": sha256(directory / "after/definition-proofs.json"),
            "status": "prepared",
        }
        for phase in ("before", "after"):
            workspace = directory / phase
            repository = workspace / "repository"
            archive_sources(repository, workspace / "original-sources.tar.gz")
            lake = repository / entry["project_path"] / ".lake"
            lake.mkdir()
            (lake / "packages").symlink_to(Path("/opt/leanlean/packages") / entry["key"], target_is_directory=True)
            write_json(workspace / "leanlean-project.json", entry)
        write_json(directory / "result.json", job)
        jobs.append(job)
    return jobs


def container_command(job: dict, phase: str, runtime: dict, code: Path, name: str) -> list[str]:
    workspace = Path(job["directory"]) / phase
    toolchain = job["entry"]["lean_toolchain"].replace("/", "--").replace(":", "---")
    return [
        runtime["container_engine"],
        "run",
        "--rm",
        "--init",
        "--name",
        name,
        "--network",
        "none",
        "--cpus",
        "8",
        "--memory",
        "64g",
        "--memory-swap",
        "64g",
        "--pids-limit=-1",
        "--cap-drop",
        "ALL",
        "--cap-add",
        "SETFCAP",
        "--security-opt",
        "no-new-privileges:true",
        "--security-opt",
        "unmask=/proc/*",
        "--read-only",
        "--user",
        "0:0",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=16g,mode=1777",
        "--volume",
        f"{workspace}:/work:rw",
        "--volume",
        f"{code}:/postprocess:ro",
        "--workdir",
        "/work/repository",
        "--env",
        "HOME=/tmp",
        "--env",
        "LEAN_NUM_THREADS=8",
        "--env",
        f"PATH=/opt/elan/bin:/opt/elan/toolchains/{toolchain}/bin:/usr/local/bin:/usr/bin:/bin",
        runtime["docker_image"],
        "python3",
        "/postprocess/postprocess_container.py",
        "--phase",
        phase,
    ]


def verify(job: dict, runtime: dict, code: Path) -> dict:
    directory = Path(job["directory"])
    job = {**job, "status": "running", "started_at": utc_now(), "checks": {}}
    write_json(directory / "result.json", job)
    try:
        for phase in ("before", "after"):
            name = f'leanlean-postprocess-p{job["problem_idx"]}-{uuid.uuid4().hex[:10]}'
            command = container_command(job, phase, runtime, code, name)
            write_json(directory / phase / "container.json", {"command": command, "image": runtime["docker_image"]})
            print(f'STATUS P{job["problem_idx"]} {phase}: build/verify', flush=True)
            try:
                with (directory / phase / "container.log").open("w") as log:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=25260)
                status = read_json(directory / phase / "verification.json")
                status["container_returncode"] = completed.returncode
            except subprocess.TimeoutExpired:
                status = {"status": "timeout", "container_returncode": 124}
            finally:
                subprocess.run([runtime["container_engine"], "rm", "--force", name], capture_output=True, timeout=30)
            job["checks"][phase] = status
            write_json(directory / "result.json", job)
        job["verified"] = all(
            v.get("status") == "passed" and v.get("container_returncode") == 0 for v in job["checks"].values()
        )
        job["status"] = "verified" if job["verified"] else "verification_failed"
        job["eligible_compression_pct"] = job["metrics"]["compression_pct"] if job["verified"] else 0.0
    except Exception as error:
        job.update(status="error", error=f"{type(error).__name__}: {error}", verified=False)
    job["finished_at"] = utc_now()
    write_json(directory / "result.json", job)
    print(f'STATUS P{job["problem_idx"]} {job["status"]}', flush=True)
    return job


def summarize(jobs: list[dict], output: Path) -> None:
    rows = [
        {
            "problem_idx": job["problem_idx"],
            "key": job["key"],
            "status": job["status"],
            "baseline_tokens": job["metrics"]["before"]["lean_tokens"],
            "final_tokens": job["metrics"]["after"]["lean_tokens"],
            "tokens_saved": job["metrics"]["tokens_saved"],
            "compression_pct": job["metrics"]["compression_pct"],
            "eligible_compression_pct": job.get("eligible_compression_pct", 0.0),
            "restored_files": ";".join(
                k for k, v in job["fixtures"].items() if v.get("restored", v["status"] != "unchanged")
            ),
            "lakefile_status": job["fixtures"].get(job["entry"]["lakefile_path"], {}).get("status", "unchanged"),
            "definitions_status": job.get("checks", {}).get("after", {}).get("definitions", {}).get("status", ""),
        }
        for job in sorted(jobs, key=lambda j: j["problem_idx"])
    ]
    with (output / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        output / "summary.json",
        {
            "projects": len(rows),
            "verified": sum(job.get("verified", False) for job in jobs),
            "baseline_tokens": sum(row["baseline_tokens"] for row in rows),
            "final_tokens": sum(row["final_tokens"] for row in rows),
            "macro_eligible_compression_pct": sum(row["eligible_compression_pct"] for row in rows) / len(rows),
            "results": rows,
        },
    )


def freeze_submission(path: Path, output: Path, *, resume: bool) -> tuple[Path, str]:
    """Turn a directory into a pinned input archive, keeping existing archives as supplied."""
    path = path.resolve()
    if not path.is_dir():
        return path, sha256(path)

    with tempfile.TemporaryDirectory(prefix=".submission-", dir=output) as scratch:
        staged = Path(scratch) / "sources.tar.gz"
        write_submission_archive(path, staged)
        digest = sha256(staged)
        archive = output / f"submission-{digest}.tar.gz"
        if not resume:
            staged.replace(archive)
    return archive, digest


def grade_saved_runs(
    results: Path, bundle: Path, output: Path, *, problems: list[int] | None = None,
    workers: int = 1, resume: bool = False,
) -> int:
    """Discover MathArena result files and pin them for the ordinary LeanLean verifier."""
    index = {}
    for path in sorted(results.glob(f"*{OUTPUT_JSON_SUFFIX}")):
        problem = output_json_stem(path)
        if not problem.isdecimal() or str(int(problem)) != problem or not path.is_file():
            continue
        if problems is None or int(problem) in problems:
            index[problem] = {"path": str(path.resolve()), "sha256": sha256(path)}
    if not index:
        raise ValueError(f"No saved LeanLean results for the selected problems in {results}")

    output.mkdir(parents=True, exist_ok=True)
    # Separate concurrent or changed inputs before the grader takes its output lock.
    digest = hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()
    index_path = output / f"run-index-{digest}.json"
    write_json(index_path, index)
    print(f"LeanLean grades will be written to {output / 'results.csv'}", flush=True)
    argv = ["--index", str(index_path), "--bundle", str(bundle),
            "--output", str(output), "--workers", str(workers)]
    if resume:
        argv.append("--resume")
    return main(argv)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--index", type=Path, help="Checksum-pinned MathArena result index")
    inputs.add_argument("--submission", type=Path, help="Submitted repository directory or source tar archive")
    parser.add_argument("--problem", type=int, help="One-based problem ID for --submission")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--token-module", type=Path,
        default=Path(__file__).resolve().parents[2] / "third_party/lean-lean/src/leanlean/metrics/tokens.py",
        help="Reference Lean token counter (defaults to the vendored copy)",
    )
    parser.add_argument(
        "--definition-proofs",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data/leanlean/definition-proofs",
        help="reviewed definition-equivalence proofs, one <project key>.json per project",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if (args.submission is None) != (args.problem is None):
        parser.error("--submission and --problem must be used together")
    bundle_digest = sha256(args.bundle / "manifest.json")
    output = args.output.resolve()
    if args.submission is not None and args.submission.is_dir() and output.is_relative_to(args.submission.resolve()):
        parser.error("--output must be outside the submitted repository")
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".lock"):
        plan_path = output / "plan.json"
        if not args.resume and (plan_path.exists() or (output / "repositories").exists()):
            raise ValueError("Output already contains work; use --resume or a new directory")
        direct = None
        if args.submission is not None:
            archive, digest = freeze_submission(args.submission, output, resume=args.resume)
            direct = (args.problem, archive, digest)
            input_pin = {"submission_input": {"problem_idx": args.problem, "sha256": digest}}
        else:
            input_pin = {"index_sha256": sha256(args.index)}
        if args.resume:
            plan = read_json(plan_path)
            if (
                any(plan.get(key) != value for key, value in input_pin.items())
                or bundle_digest != plan["bundle_sha256"]
            ):
                raise ValueError("Resume inputs differ from the original plan")
            jobs = [read_json(Path(j["directory"]) / "result.json") for j in plan["jobs"]]
        else:
            runtime = read_json(args.bundle / "eval-runtime.json")
            if runtime["bundle_sha256"] != bundle_digest:
                raise ValueError("Prepared runtime does not match the bundle")
            code = output / "pipeline"
            code.mkdir(exist_ok=True)
            for name in (
                "postprocess.py",
                "postprocess_container.py",
                "definitions_container.py",
                "definition_audit.lean",
            ):
                shutil.copyfile(Path(__file__).with_name(name), code / name)
            shutil.copyfile(args.token_module, code / "reference_tokens.py")
            jobs = prepare(args.index, args.bundle, output, code / "reference_tokens.py",
                           args.definition_proofs, direct=direct)
            plan = {
                "created_at": utc_now(),
                "runtime": runtime,
                **input_pin,
                "bundle_sha256": bundle_digest,
                "token_module_sha256": sha256(args.token_module),
                "metric": "lean-lean non-comment non-import Lean tokens; excludes challenge and lakefile.lean",
                "jobs": jobs,
            }
            write_json(plan_path, plan)
        # A resume continues unfinished jobs; completed failures require a new run
        # so evidence is not overwritten by an automatic retry.
        pending = [j for j in jobs if j["status"] in {"prepared", "running", "error"}]
        done = [j for j in jobs if j not in pending]
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(verify, j, plan["runtime"], output / "pipeline") for j in pending]
            for future in concurrent.futures.as_completed(futures):
                done.append(future.result())
                summarize(
                    done + [j for j in pending if j["problem_idx"] not in {x["problem_idx"] for x in done}], output
                )
        summarize(done, output)
        return int(any(j["status"] != "verified" for j in done))


if __name__ == "__main__":
    raise SystemExit(main())
