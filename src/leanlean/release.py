"""Package a frozen October result catalog for Git and independent restoration."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import gzip
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import uuid
import zipfile

import zstandard

from leanlean.common import attempt_chain, chain_frames, read_json, sha256, submission_attempt, tree_sha256, utc_now, write_json
from leanlean.publication import (Bundle, MODELS, PART_SIZE, Sources, artifact_files,
                                 bundle_checksum, bundle_parts, digest, encoded, open_bundle,
                                 origin_summary, output_tokens_summary, packed, prepared_archives,
                                 read_archive, restore_sources, safe_name, split_bundle)
from matharena.monitoring.attempt import read_tree


def tree_identity(state: dict) -> str:
    entries = {name: [kind, mode, digest(data) if kind == "file" else data]
               for name, (kind, data, mode) in state.items()}
    return digest(encoded(entries))


def post_archive(state: dict, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as raw, zstandard.ZstdCompressor(level=9).stream_writer(raw) as compressed:
        with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as archive:
            for name, (kind, data, mode) in sorted(state.items()):
                safe_name(name)
                info = tarfile.TarInfo(name)
                info.mode = mode
                if kind == "symlink":
                    info.type, info.linkname = tarfile.SYMTYPE, data
                    archive.addfile(info)
                else:
                    info.size = len(data)
                    archive.addfile(info, io.BytesIO(data))
    receipt = dict(sha256=sha256(path), bytes=path.stat().st_size, files=len(state),
                   file_tree_sha256=tree_identity(state), excluded=[".scratch"])
    receipt["parts"] = split_bundle(path)
    return receipt


def read_post(path: Path) -> dict:
    state = {}
    with open_bundle(path) as archive:
        for member in archive:
            name = safe_name(member.name)
            if name in state:
                raise ValueError("Duplicate post source path")
            if member.isfile():
                state[name] = ("file", archive.extractfile(member).read(), member.mode)
            elif member.issym():
                state[name] = ("symlink", member.linkname, member.mode)
            else:
                raise ValueError("Unexpected post source member")
    return state


def write_tree(state: dict, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    for name in state:
        safe_name(name)
        if any(str(parent) in state for parent in PurePosixPath(name).parents):
            raise ValueError("Source entry used as a parent directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".restore-post-", dir=output.parent) as temporary:
        root = Path(temporary) / "repository"
        root.mkdir()
        for kind_to_write in ("file", "symlink"):
            for name, (kind, data, mode) in state.items():
                if kind != kind_to_write:
                    continue
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if kind == "file":
                    target.write_bytes(data)
                    target.chmod(mode)
                else:
                    target.symlink_to(data)
        root.rename(output)


def website_archive(directory: Path) -> dict:
    """Keep frequent summaries loose; index already-gzipped assets for direct reads."""
    paths = sorted(p for p in directory.glob("*.json.gz") if p.name != "detail.json.gz")
    archive = directory / "assets.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as output:
        for path in paths:
            info = zipfile.ZipInfo(path.name, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o100644 << 16
            output.writestr(info, path.read_bytes())
    with zipfile.ZipFile(archive) as output:
        for path in paths:
            if output.read(path.name) != path.read_bytes():
                raise ValueError("Website asset archive failed byte comparison")
    for path in paths:
        path.unlink()
    if archive.stat().st_size > PART_SIZE:
        raise ValueError("Website asset archive exceeds the release file-size budget")
    return dict(files=len(paths), bytes=archive.stat().st_size, sha256=sha256(archive))


def graph_receipt(run: Path, model: str, idx: int, phase: str, graph: Path) -> Path | None:
    """Prefer the completed extraction that produced these exact graph bytes."""
    candidates = list(run.glob(f"*/{model}/graphs/P{idx}-{phase}.json"))
    candidates += list(run.glob(f"graphs-*/{model}/P{idx}-{phase}.json"))
    checksum = sha256(graph) if graph.is_file() else None
    completed, other = [], []
    for path in candidates:
        record = read_json(path)
        if record.get("status") == "passed":
            if checksum and record.get("graph_path") == str(graph) and record.get("graph_sha256") == checksum:
                completed.append(path)
        else:
            other.append(path)
    # Several extractions can describe the same bytes; receipts are copied, so file times say nothing.
    return max(completed or other, key=lambda p: (read_json(p).get("finished_at", ""), str(p)), default=None)


def pack_result(job: dict) -> dict:
    from leanlean.website_export import final_diffs

    run, dataset = Path(job["run"]), Path(job["dataset"])
    target, web = Path(job["target"]), Path(job["website"])
    entry, pinned, submission, row = (job[k] for k in ("entry", "pinned", "submission", "row"))
    slug, model, idx = job["slug"], MODELS[job["slug"]], int(row["id"])
    if (target / "manifest.json").exists():
        saved = read_json(target / "manifest.json")
        if saved.get("original_result_sha256", saved.get("result_sha256")) != pinned["sha256"] or saved["key"] != entry["key"]:
            raise ValueError("Completed bundle belongs to a different frozen result")
        for filename, field in [("post.tar.zst", "post"), ("checkpoints.tar.zst", "checkpoints"), ("analysis.tar.zst", "analysis")]:
            if bundle_checksum(target / filename) != saved[field]["sha256"]:
                raise ValueError("Completed bundle changed before resume")
        if sha256(target / "result.json.zst") != saved["result"]["sha256"]:
            raise ValueError("Completed result changed before resume")
        if sha256(web / "assets.zip") != saved["website"]["sha256"]:
            raise ValueError("Completed website archive changed before resume")
        return saved
    if target.exists():
        failed = target.parents[3] / "failed" / slug / (target.name + "-" + uuid.uuid4().hex)
        failed.parent.mkdir(parents=True, exist_ok=True)
        target.rename(failed)
        if web.exists():
            web.rename(failed / "website")
        shutil.copytree(Path(job["website_source"]), web)
    target.mkdir(parents=True)
    verification = Path(job["verification"])
    attempt = submission_attempt(submission)
    chain = attempt_chain(attempt)
    if not (attempt / "summary.json").exists():
        raise ValueError(f"Attempt is not finished: {slug} P{idx}")
    outcome = read_json(verification / "result.json") if (verification / "result.json").exists() else None
    if outcome and (outcome["submission_sha256"] != submission["sha256"] or outcome["result_sha256"] != pinned["sha256"]):
        raise ValueError("Verification identity mismatch")
    baseline_path = dataset / entry["project_directory"]
    baseline_state = read_tree(baseline_path)
    if tree_sha256(baseline_path) != entry["project_tree_sha256"]:
        raise ValueError("Frozen baseline changed")
    external = {digest(data): name for name, (kind, data, mode) in sorted(baseline_state.items()) if kind == "file"}
    submitted, submitted_identity = read_archive(Path(submission["archive"]), remove_scratch=True, exclude_build=True)
    if submitted_identity["archive_sha256"] != submission["sha256"]:
        raise ValueError("Original submission changed")
    after = verification / "after/original-sources.tar.gz"
    post_state, post_identity = read_archive(after, remove_scratch=True, exclude_build=True) if after.exists() else (submitted, submitted_identity)
    post = post_archive(post_state, target / "post.tar.zst")
    post["excluded_build_artifacts"] = True
    if read_post(target / "post.tar.zst") != post_state:
        raise ValueError("Post source archive failed exact tree comparison")
    frames_path = Path(job.get("checkpoint_snapshot", attempt / "checkpoints.jsonl"))
    frames_bytes = frames_path.read_bytes()
    frames = [json.loads(line) for line in frames_bytes.splitlines() if line.strip()]
    checkpoints = Sources(target / "checkpoints.tar.zst", exclude_scratch=True, external_blobs=external, exclude_build=True)
    parent, known = None, {}
    unique = {}

    def held_by(frame):
        """A continuation chain's rows name their attempt by position; others are the attempt's own."""
        return chain[frame.get("continued_attempt", len(chain) - 1)]

    for frame in frames:
        unique.setdefault(frame["source_sha256"], held_by(frame) / safe_name(frame["archive"]))
    large = sum(path.stat().st_size for path in unique.values()) > 16 * 1024**3
    prepared = iter(prepared_archives(unique.values(), workers=32, remove_scratch=True, exclude_build=True)) if large else None
    for n, frame in enumerate(frames):
        # A continuation opens on its predecessor's final state with a fresh baseline row.
        opens = frame.get("continued_attempt") and frame.get("parent_sha256") is None and frame["source_sha256"] == parent
        if frame.get("parent_sha256") != parent and not opens:
            raise ValueError("Checkpoint source chain is incomplete")
        parent = frame["source_sha256"]
        label = f"checkpoint-{n}"
        if parent not in known:
            if prepared is not None:
                path, state, identity, hashes = next(prepared)
                if path != held_by(frame) / safe_name(frame["archive"]):
                    raise ValueError("Checkpoint preparation order changed")
                known[parent] = checkpoints.prepared_archive(label, path, state, identity,
                                                              file_hashes=hashes, source_sha256=parent)
            else:
                known[parent] = checkpoints.archive(label, held_by(frame) / safe_name(frame["archive"]), source_sha256=parent)
        else:
            checkpoints.states[label] = known[parent]
        checkpoints.timeline.append(dict(frame, tree=known[parent], state=label))
    checkpoint_receipt = checkpoints.finish()
    if frames_path.read_bytes() != frames_bytes:
        raise ValueError("Checkpoint timeline changed during packaging")
    def scored(name):
        return (name.endswith(".lean") and PurePosixPath(name).is_relative_to(PurePosixPath(entry["project_path"] or "."))
                and PurePosixPath(name).name != "lakefile.lean" and name != entry["challenge_path"]
                and not any(part.startswith(".") for part in PurePosixPath(name).parts))
    def texts(state):
        return {name: data.decode("utf-8") for name, (kind, data, mode) in state.items() if kind == "file" and scored(name)}
    files = final_diffs(texts(baseline_state), texts(post_state), web)
    packed(web / "diff-index.json.gz", dict(comparison="before-to-after" if after.exists() else "before-to-submitted",
                                           verified=row.get("status") == "verified", files=files))
    with gzip.open(web / "detail.json.gz", "rt") as stream:
        detail = json.load(stream)
    detail["source_changes"] = dict(files=len(files), evaluated=after.exists())
    packed(web / "detail.json.gz", detail)
    web_receipt = website_archive(web)
    analysis = Bundle(target / "analysis.tar.zst")
    selected = Path(pinned["path"])
    if sha256(selected) != pinned["sha256"]:
        raise ValueError("Selected result changed")
    shutil.copyfile(selected, target / "result.json.zst")
    analysis.add("benchmark-entry.json", encoded(entry))
    analysis.add("checkpoint-identities.json", encoded(dict(checkpoints_sha256=digest(frames_bytes),
                                                          positions=len(frames), archive=checkpoint_receipt)))
    adjustments = []
    for name in sorted(submitted.keys() | post_state.keys()):
        old, new = submitted.get(name), post_state.get(name)
        if old == new:
            continue
        record = dict(path=name, submitted_kind=old[0] if old else None,
                      submitted_mode=old[2] if old else None)
        if old and old[0] == "file":
            member = "submission-adjustments/files/" + name
            analysis.add(member, old[1], old[2])
            record["submitted_content"] = member
        elif old:
            record["submitted_link"] = old[1]
        adjustments.append(record)
    analysis.add("submission-adjustments/manifest.json", encoded(dict(
        original_archive_sha256=submission["sha256"], original_archive_bytes_retained=False,
        excluded=[".scratch"], excluded_build_artifacts=True, excluded_files=submitted_identity["removed_files"],
        excluded_bytes=submitted_identity["removed_bytes"], adjustments=adjustments)))
    for artifact in artifact_files(verification):
        if job.get("defer_analysis") and artifact.name not in {
            "result.json", "leanlean-project.json", "verification.json",
            "build.stdout.log", "build.stderr.log", "comparator.stdout.log", "comparator.stderr.log",
        }:
            continue
        analysis.file("verification/" + artifact.relative_to(verification).as_posix(), artifact)
    if job.get("defer_analysis"):
        analysis.add("analysis/incomplete.json", encoded(dict(
            status="unfinished_at_snapshot", reason="Snapshot released while optional graph/origin analysis continued",
            omitted=["dependency graphs", "graph extraction diagnostics", "origin attribution"],
            preserved=["verification", "final source", "all checkpoints", "result and agent trace", "cost and timing"],
        )))
    # Graph extraction may run after verification, in a separate resumable
    # batch. Keep its receipts, diagnostics and adapter with the graph bytes.
    for phase in ("before", "after"):
        if job.get("defer_analysis"):
            continue
        graph = verification / phase / "dependency-graph.json"
        receipt = graph_receipt(run, model, idx, phase, graph)
        if receipt:
            analysis.file(f"graph-extraction/{phase}.json", receipt)
            log = receipt.with_suffix(".log")
            if log.exists():
                analysis.file(f"graph-extraction/{phase}.log", log)
            adapter = receipt.parent / "pipeline/graphs_container.py"
            if adapter.exists():
                analysis.file(f"graph-extraction/{phase}-adapter.py", adapter)
        report = verification / phase / "repository" / entry["project_path"] / ".lean-strip/report.json"
        if report.is_file():
            analysis.file(f"graph-extraction/{phase}-report.json", report)
    pipeline = verification.parents[1] / "pipeline"
    for artifact in sorted(pipeline.glob("*.py")):
        analysis.file("verification-pipeline/" + artifact.name, artifact)
    charts = run / "charts-20261102" / model
    for group in ("timing", "cost"):
        path = charts / group / f"p{idx}.json"
        if path.exists():
            analysis.file(f"analysis/{group}.json", path)
    corrected = run / "cost-reconciliation-20261102" / model / f"p{idx}.json"
    if corrected.exists():
        analysis.file("analysis/reconciled-cost.json", corrected)
    origins = charts / "origins" / entry["key"]
    if not job.get("defer_analysis"):
        for artifact in artifact_files(origins):
            analysis.file("analysis/origins/" + artifact.relative_to(origins).as_posix(), artifact)
        # An analysis run on a staged copy of the verification directory (e.g. with a
        # regraphed input) ships the graphs and strip reports it read.
        status = origins / "status.json"
        staged = Path(read_json(status).get("directory", verification)) if status.is_file() else verification
        if staged.resolve() != verification.resolve():
            for artifact in artifact_files(staged):
                analysis.file("analysis/origin-inputs/" + artifact.relative_to(staged).as_posix(), artifact)
            for phase in ("before", "after"):
                report = staged / phase / "repository" / entry["project_path"] / ".lean-strip/report.json"
                if report.is_file():
                    analysis.file(f"analysis/origin-inputs/{phase}-report.json", report)
    # Attempts the scored one continues keep their telemetry under continued-<position>/.
    for n, held in enumerate(chain):
        prefix = "" if held == attempt else f"continued-{n}/"
        for path in sorted(held.iterdir()):
            if path.is_file() and not path.is_symlink() and not path.name.startswith(".") and path.name != "harness-events.jsonl":
                analysis.file("telemetry/" + prefix + path.name, path)
        pruning = held / "checkpoint-pruning"
        for path in artifact_files(pruning):
            analysis.file("checkpoint-pruning/" + prefix + path.relative_to(pruning).as_posix(), path)
    for frame in frames:
        member = "telemetry/" + safe_name(frame["diff"])
        if member not in analysis.members:
            analysis.file(member, held_by(frame) / frame["diff"])
    native_receipt = None
    if job.get("include_raw_logs", False):
        native = Bundle(target / "agent-traces.tar.zst")
        for path in sorted((attempt / "native").rglob("*")):
            if path.is_file() and not path.is_symlink():
                native.file(path.relative_to(attempt / "native").as_posix(), path)
        native_receipt = native.finish()
    record = dict(model=slug, id=idx, key=entry["key"], status=row.get("status"),
                  pre=dict(dataset="data/leanlean/oct26", project=entry["project_directory"],
                           tree_sha256=entry["project_tree_sha256"]),
                  post=dict(post, kind="evaluated" if after.exists() else "submitted-unverified",
                            input_archive_sha256=post_identity["archive_sha256"]),
                  checkpoints=checkpoint_receipt, analysis=analysis.finish(), agent_traces=native_receipt,
                  result=dict(path="result.json.zst", sha256=pinned["sha256"], bytes=selected.stat().st_size),
                  raw_harness_events_omitted=True,
                  website=web_receipt, result_sha256=pinned["sha256"],
                  original_submission_sha256=submission["sha256"], original_submission_bytes_retained=False)
    write_json(target / "manifest.json", record)
    return record


def freeze_code(root: Path, run: Path, target: Path) -> dict:
    code = Bundle(target / "code.tar.zst")
    for directory in (root / "src", root / "harness_wrapper/src", root / "website/flaskr"):
        for path in sorted(directory.rglob("*.py")):
            code.file(path.relative_to(root).as_posix(), path)
    for name in ("pyproject.toml", "uv.lock", "website/pyproject.toml", "website/uv.lock"):
        path = root / name
        if path.is_file():
            code.file(name, path)
    for path in sorted((root / "configs/models").rglob("*.yaml")):
        code.file(path.relative_to(root).as_posix(), path)
    code.file("run/config.json", run / "config.json")
    code.file("run/reference_tokens.py", run / "reference_tokens.py")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()
    patch = subprocess.run(["git", "diff", "HEAD", "--binary", "--", "*.py", "*.toml", "*.yaml"],
                           cwd=root, capture_output=True, check=True).stdout
    code.add("working-tree.patch", patch)
    return dict(code.finish(), git_head=git)


def prepare_release(run: Path, website: Path, output: Path, *, include_raw_logs=False) -> dict:
    if output.exists():
        raise FileExistsError("Choose a fresh release directory")
    root = Path(__file__).resolve().parents[2]
    output.parent.mkdir(parents=True, exist_ok=True)
    build = output.parent / ("." + output.name + ".building")
    build.mkdir(exist_ok=True)
    plan_path = build / "build-plan.json"
    staging = build / "release"
    if plan_path.exists():
        plan = read_json(plan_path)
        if plan["include_raw_logs"] != include_raw_logs:
            raise ValueError("Cannot change raw-log policy while resuming a frozen build")
        jobs = plan["jobs"]
        catalog = read_json(staging / "website/index.json")
        dataset = Path(plan["dataset"])
        print(f"Resuming {len(jobs)} frozen result bundles", flush=True)
    else:
        config = read_json(run / "config.json")
        cache = read_json(run / "viewer-export-cache.json")
        dataset = Path(config["bundle"])
        benchmark = read_json(dataset / "manifest.json")
        catalog = read_json(website / "index.json")
        identities = read_json(website / "current-provenance.json")
        staging.mkdir()
        write_json(staging / "benchmark-manifest.json", benchmark)
        write_json(staging / "website/index.json", catalog)
        write_json(staging / "website/current-provenance.json", identities)
        jobs = []
        for model in catalog["models"]:
            slug = model["id"]
            key = MODELS[slug]
            directory = run / config["models"][key].get("postprocess_directory", key)
            selected = read_json(directory / "combined-results.json")
            pins = {row["id"]: row for row in identities["models"][slug]}
            for row in model["results"]:
                idx = int(row["id"])
                pinned = selected[str(idx)]
                saved = cache[f"{key}/{idx}"]
                if pinned["sha256"] != pins[idx]["result_sha256"] or saved["result_sha256"] != pinned["sha256"]:
                    raise ValueError("Selection changed after the frozen website snapshot")
                submission = saved["submission"]
                entry = benchmark["projects"][idx - 1]
                if (submission["key"], submission["baseline_sha256"], submission["sha256"]) != (
                    entry["key"], entry["project_tree_sha256"], pins[idx]["submission_sha256"]):
                    raise ValueError("Submission/baseline identity mismatch")
                web = staging / "website" / slug / str(idx)
                shutil.copytree(website / slug / str(idx), web)
                frozen = build / "inputs" / slug / str(idx)
                frozen.mkdir(parents=True)
                shutil.copyfile(pinned["path"], frozen / "result.json.zst")
                if sha256(frozen / "result.json.zst") != pinned["sha256"]:
                    raise ValueError("Result changed while freezing build inputs")
                pinned = dict(pinned, path=str(frozen / "result.json.zst"))
                chain = attempt_chain(submission_attempt(submission))
                if len(chain) == 1:
                    shutil.copyfile(chain[0] / "checkpoints.jsonl", frozen / "checkpoints.jsonl")
                else:
                    (frozen / "checkpoints.jsonl").write_text("".join(json.dumps(row) + "\n" for row in chain_frames(chain)))
                jobs.append(dict(run=str(run), dataset=str(dataset), target=str(staging / "results" / slug / str(idx)),
                                 website=str(web), entry=entry, pinned=pinned, submission=submission, row=row, slug=slug,
                                 website_source=str(website / slug / str(idx)),
                                 checkpoint_snapshot=str(frozen / "checkpoints.jsonl"),
                                 include_raw_logs=include_raw_logs,
                                 # Real path: graph receipts and origin inputs name the verification directory itself.
                                 verification=str((directory / f"p{idx}/verification/repositories" / entry["key"]).resolve())))
            model["origin_summary"] = origin_summary(model, staging / "website")
            model["output_tokens_summary"] = output_tokens_summary(model, staging / "website")
        write_json(staging / "website/index.json", catalog)
        write_json(plan_path, dict(jobs=jobs, dataset=str(dataset), include_raw_logs=include_raw_logs))
    return dict(root=root, build=build, staging=staging, jobs=jobs, catalog=catalog, dataset=dataset)


def recompress(run: Path, website: Path, output: Path, workers=8, *, include_raw_logs=False) -> dict:
    prepared = prepare_release(run, website, output, include_raw_logs=include_raw_logs)
    root, build, staging, jobs, catalog, dataset = (prepared[k] for k in ("root", "build", "staging", "jobs", "catalog", "dataset"))
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(pack_result, job) for job in jobs]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Packed {len(results)}/{len(jobs)}: {result['model']} P{result['id']}; "
                  f"{result['checkpoints']['checkpoints']} checkpoints; "
                  f"{result['checkpoints']['bytes'] / 1e6:.2f} MB checkpoint bundle", flush=True)
    provenance = freeze_code(root, run, staging / "provenance")
    for path in sorted((root / "readmes").glob("leanlean_checkpoint_pruning*.json")):
        shutil.copyfile(path, staging / "provenance" / path.name)
    manifest = dict(schema="leanlean_release_v2", created_at=utc_now(), edition=catalog["edition"],
                    input_catalog_updated_at=catalog.get("updated_at"),
                    results=sorted(results, key=lambda x: (x["model"], x["id"])), code=provenance,
                    scope=dict(all_selected_checkpoints=True, agent_traces=include_raw_logs,
                               normalized_result_transcripts=True, superseded_attempts=False,
                               excluded=[".lake", ".scratch", ".git", ".lean-strip", "__pycache__"],
                               original_submission_archive_bytes=False, excluded_build_artifacts=True),
                    baseline=dict(path="data/leanlean/oct26", manifest_sha256=sha256(dataset / "manifest.json")),
                    exporter_sha256=sha256(Path(__file__)), source_loader_sha256=sha256(Path(__file__).with_name("publication.py")))
    refresh_files(staging, manifest)
    staging.rename(output)
    from leanlean.result_trace import attach_metrics, separate_release
    manifest = separate_release(output)
    manifest = attach_metrics(output)
    return manifest


def refresh_files(root: Path, manifest: dict) -> None:
    files = {path.relative_to(root).as_posix(): dict(bytes=path.stat().st_size, sha256=sha256(path))
             for path in sorted(root.rglob("*")) if path.is_file() and path != root / "manifest.json"}
    scope = manifest.get("scope", {})
    if scope.get("checkpoints_local_only") or scope.get("telemetry_local_only"):
        def local(name):
            base = Path(name).name
            return ((scope.get("checkpoints_local_only") and base.startswith("checkpoints.tar.zst"))
                    or (scope.get("telemetry_local_only") and base.startswith("telemetry.tar.zst")))
        manifest["local_files"] = {name: receipt for name, receipt in files.items()
                                   if local(name)}
        files = {name: receipt for name, receipt in files.items() if name not in manifest["local_files"]}
        manifest["local_bytes"] = sum(item["bytes"] for item in manifest["local_files"].values())
    manifest["files"] = files
    manifest["total_bytes"] = sum(item["bytes"] for item in manifest["files"].values())
    write_json(root / "manifest.json", manifest)


def finalize_result(job: dict) -> dict:
    """Keep a conventional result file and move extra raw logs outside the Git release."""
    root, local = Path(job["release"]), Path(job["local"])
    row = job["record"]
    target = root / "results" / row["model"] / str(row["id"])
    archive = target / "analysis.tar.zst"
    backup = local / row["model"] / str(row["id"])
    backup.mkdir(parents=True, exist_ok=True)
    if row.get("result") is None or not row.get("raw_harness_events_omitted"):
        converted = backup / "converted"
        converted.mkdir()
        analysis = Bundle(converted / "analysis.tar.zst")
        extracted = False
        with open_bundle(archive) as original:
            for member in original:
                if not member.isfile():
                    raise ValueError("Unexpected analysis archive member")
                if member.name in {"bundle-members.json", "telemetry/harness-events.jsonl"}:
                    continue
                data = original.extractfile(member).read()
                if member.name == "result.json.zst":
                    if digest(data) != row.get("original_result_sha256", row.get("result_sha256")):
                        raise ValueError("Normalized result checksum mismatch")
                    if row.get("result") is None:
                        (target / "result.json.zst").write_bytes(data)
                    extracted = True
                else:
                    analysis.add(member.name, data, member.mode)
        if not extracted and row.get("result") is None:
            raise ValueError("Analysis archive has no normalized result")
        receipt = analysis.finish()
        previous = backup / "analysis-before-finalization"
        previous.mkdir()
        for path in bundle_parts(archive):
            path.rename(previous / path.name)
        for path in converted.iterdir():
            shutil.copyfile(path, target / path.name)
        row["analysis"] = receipt
        if row.get("result") is None:
            row["result"] = dict(path="result.json.zst", sha256=row["result_sha256"],
                                 bytes=(target / "result.json.zst").stat().st_size)
        row["raw_harness_events_omitted"] = True
    if row.get("agent_traces"):
        raw = backup / "agent-traces"
        raw.mkdir()
        for path in bundle_parts(target / "agent-traces.tar.zst"):
            path.rename(raw / path.name)
        write_json(raw / "manifest.json", row["agent_traces"])
        row["agent_traces"] = None
    write_json(target / "manifest.json", row)
    return row


def finalize_for_git(root: Path, local: Path, run: Path, workers=4) -> dict:
    manifest = read_json(root / "manifest.json")
    jobs = [dict(release=str(root), local=str(local), record=row) for row in manifest["results"]]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        manifest["results"] = list(pool.map(finalize_result, jobs))
    manifest["scope"]["agent_traces"] = False
    manifest["scope"]["normalized_result_transcripts"] = True
    manifest["scope"]["raw_harness_events"] = False
    manifest["scope"]["benchmark_telemetry"] = True
    manifest["finalized_at"] = utc_now()
    manifest["finalizer_sha256"] = sha256(Path(__file__))
    old_code = local / "provenance-before-finalization"
    old_code.mkdir(parents=True, exist_ok=True)
    for path in bundle_parts(root / "provenance/code.tar.zst"):
        path.rename(old_code / path.name)
    manifest["code"] = freeze_code(Path(__file__).resolve().parents[2], run, root / "provenance")
    refresh_files(root, manifest)
    return manifest


def verify(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    files = dict(manifest["files"])
    files.update({name: expected for name, expected in manifest.get("local_files", {}).items()
                  if (root / safe_name(name)).exists() or (root / safe_name(name)).is_symlink()})
    for name, expected in files.items():
        path = root / safe_name(name)
        if path.is_symlink() or path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            raise ValueError(f"Release file changed: {name}")
    return dict(results=len(manifest["results"]), files=len(manifest["files"]), bytes=manifest["total_bytes"],
                checkpoints=sum(row["checkpoints"]["checkpoints"] for row in manifest["results"]))


def restore(release: Path, model: str, problem: int, state: str, output: Path, dataset: Path):
    manifest = read_json(release / "manifest.json")
    row = next(row for row in manifest["results"] if (row["model"], row["id"]) == (model, problem))
    directory = release / safe_name(row["artifact_directory"]) if row.get("artifact_directory") else release / "results" / model / str(problem)
    if state == "post":
        path = directory / "post.tar.zst"
        if bundle_checksum(path) != row["post"]["sha256"]:
            raise ValueError("Post archive checksum mismatch")
        source = read_post(path)
        if tree_identity(source) != row["post"]["file_tree_sha256"]:
            raise ValueError("Post file tree checksum mismatch")
        write_tree(source, output)
        return dict(files=len(source), state=state)
    if sha256(dataset / "manifest.json") != manifest["baseline"]["manifest_sha256"]:
        raise ValueError("Wrong baseline dataset")
    baseline = dataset / safe_name(row["pre"]["project"])
    if tree_sha256(baseline) != row["pre"]["tree_sha256"]:
        raise ValueError("Pinned baseline project changed")
    return restore_sources(directory / "checkpoints.tar.zst", state, output,
                           row["checkpoints"]["sha256"], baseline=baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pack = commands.add_parser("recompress")
    for key in ("run", "website", "output"):
        pack.add_argument("--" + key, type=Path, required=True)
    pack.add_argument("--workers", type=int, default=8)
    pack.add_argument("--include-raw-logs", action="store_true", help="Also package optional original harness-native logs")
    finalize = commands.add_parser("finalize-for-git")
    finalize.add_argument("release", type=Path)
    finalize.add_argument("--local-archive", type=Path, required=True)
    finalize.add_argument("--run", type=Path, required=True)
    finalize.add_argument("--workers", type=int, default=4)
    check = commands.add_parser("verify")
    check.add_argument("release", type=Path)
    load = commands.add_parser("restore")
    load.add_argument("release", type=Path)
    load.add_argument("--model", required=True)
    load.add_argument("--problem", required=True, type=int)
    load.add_argument("--state", required=True)
    load.add_argument("--output", required=True, type=Path)
    load.add_argument("--dataset", type=Path, default=Path("data/leanlean/oct26"))
    args = parser.parse_args()
    if args.command == "recompress":
        result = recompress(args.run, args.website, args.output, args.workers, include_raw_logs=args.include_raw_logs)
        print(json.dumps(dict(results=len(result["results"]), bytes=result["total_bytes"])))
    elif args.command == "finalize-for-git":
        result = finalize_for_git(args.release, args.local_archive, args.run, args.workers)
        print(json.dumps(dict(results=len(result["results"]), bytes=result["total_bytes"])))
    elif args.command == "verify":
        print(json.dumps(verify(args.release)))
    else:
        print(json.dumps(restore(args.release, args.model, args.problem, args.state, args.output, args.dataset)))


if __name__ == "__main__":
    main()
