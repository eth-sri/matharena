"""Check a release against its original inputs, including every captured source state."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import tempfile

from leanlean.common import attempt_chain, read_json, sha256, submission_attempt, write_json
from leanlean.publication import digest, encoded, jsonl, open_bundle, read_archive, restore_sources, safe_name
from leanlean.release import read_post, tree_identity, write_tree
from leanlean.result_trace import join_fields
from matharena.json_zst import load_json_zst
from matharena.monitoring.attempt import read_tree
from matharena.monitoring.source_files import is_build_artifact


def checkpoint_identity(args):
    attempt, key, frame, excluded, exclude_build = args
    state, identity = read_archive(Path(attempt) / safe_name(frame["archive"]),
                                   remove_scratch=True, exclude_dirs=excluded, exclude_build=exclude_build)
    if identity["source_sha256"] != key:
        raise ValueError("Original checkpoint hash mismatch")
    return key, tree_identity(state), len(state)


def check_analysis(path: Path):
    actual, expected = {}, None
    with open_bundle(path) as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError("Unexpected analysis archive entry")
            if member.name == "bundle-members.json":
                expected = json.load(archive.extractfile(member))
            else:
                safe_name(member.name)
                if member.name in actual:
                    raise ValueError("Duplicate analysis member")
                actual[member.name] = dict(bytes=member.size,
                    sha256=hashlib.file_digest(archive.extractfile(member), "sha256").hexdigest())
    if actual != expected:
        raise ValueError("Analysis members do not match the recorded bytes")
    if "result.json.zst" in actual:
        raise ValueError("Analysis duplicates the result/agent transcript")
    if "telemetry/harness-events.jsonl" in actual:
        raise ValueError("Analysis duplicates raw harness-event traces")
    return len(actual)


def check_checkpoint_objects(path: Path, baseline: Path):
    blobs, trees, required = set(), set(), set()
    actual, expected, index = {}, None, None
    with open_bundle(path) as archive:
        for member in archive:
            if not member.isfile():
                raise ValueError("Unexpected checkpoint archive entry")
            data = archive.extractfile(member).read()
            if member.name == "bundle-members.json":
                expected = json.loads(data)
                continue
            if member.name in actual:
                raise ValueError("Duplicate checkpoint object")
            checksum = digest(data)
            actual[member.name] = dict(bytes=member.size, sha256=checksum)
            if member.name.startswith("blobs/"):
                if checksum != member.name[6:]:
                    raise ValueError("Checkpoint file content hash mismatch")
                blobs.add(checksum)
            elif member.name.startswith("trees/"):
                if member.name != f"trees/{checksum}.json":
                    raise ValueError("Checkpoint file-tree hash mismatch")
                tree = json.loads(data)
                for name, (kind, mode, value) in tree.items():
                    safe_name(name)
                    if {".scratch", ".lake"}.intersection(Path(name).parts) or is_build_artifact(name):
                        raise ValueError("Checkpoint contains excluded runtime files")
                    if kind == "file":
                        required.add(value)
                    elif kind != "symlink":
                        raise ValueError("Unknown checkpoint source kind")
                trees.add(checksum)
            elif member.name == "sources.json":
                index = json.loads(data)
            else:
                raise ValueError("Unknown checkpoint bundle member")
    if actual != expected or index is None:
        raise ValueError("Checkpoint archive member receipt mismatch")
    external = index.get("external_blobs", {})
    for checksum, name in external.items():
        path = baseline / safe_name(name)
        if path.is_symlink() or not path.resolve().is_relative_to(baseline.resolve()) or sha256(path) != checksum:
            raise ValueError("Pinned external baseline blob changed")
    if required - blobs - external.keys():
        raise ValueError("Checkpoint tree references missing file contents")
    if set(index["states"].values()) - trees:
        raise ValueError("Checkpoint index references missing source trees")
    return index


def check_result(job: dict):
    root = Path(job["release"])
    row, original = job["record"], job["original"]
    folder = root / "results" / row["model"] / str(row["id"])
    baseline = Path(original["dataset"]) / row["pre"]["project"]
    summary = load_json_zst(folder / "result.json.zst")
    trace = load_json_zst(folder / "agent-trace.json.zst")
    source = load_json_zst(Path(original["pinned"]["path"]))
    if encoded(join_fields(summary, trace)) != encoded(source):
        raise ValueError("Separate result and agent trace lost original fields")
    if digest(encoded(source)) != trace["canonical_json_sha256"]:
        raise ValueError("Original result JSON identity mismatch")
    post_path = Path(original["verification"]) / "after/original-sources.tar.gz"
    if not post_path.exists():
        post_path = Path(original["submission"]["archive"])
    excluded = set(row["post"].get("excluded", [])) - {".scratch"}
    exclude_build = row["post"].get("excluded_build_artifacts", False)
    expected_post, _ = read_archive(post_path, remove_scratch=True, exclude_dirs=excluded, exclude_build=exclude_build)
    post = read_post(folder / "post.tar.zst")
    if post != expected_post:
        raise ValueError("Post source tree differs from the original evaluated files")
    index = check_checkpoint_objects(folder / "checkpoints.tar.zst", baseline)
    frames = jsonl(Path(original["checkpoint_snapshot"]))
    if len(frames) != len(index["timeline"]):
        raise ValueError("Checkpoint positions were lost")
    chain = attempt_chain(submission_attempt(original["submission"]))

    def held_by(frame):
        return chain[frame.get("continued_attempt", len(chain) - 1)]

    unique = {frame["source_sha256"]: frame for frame in frames}
    large = row["checkpoints"]["input_archive_bytes"] > 32 * 1024**3
    executor, workers = (ProcessPoolExecutor, 32) if large else (ThreadPoolExecutor, 1)
    with executor(max_workers=workers) as pool:
        identities = list(pool.map(checkpoint_identity,
                          [(str(held_by(frame)), key, frame, excluded, exclude_build) for key, frame in unique.items()]))
    known = {key: identity for key, identity, _count in identities}
    files_checked = sum(count for _key, _identity, count in identities)
    for n, frame in enumerate(frames):
        if any(index["timeline"][n].get(key) != value for key, value in frame.items()):
            raise ValueError("Checkpoint metadata or timestamps changed")
        key = frame["source_sha256"]
        if index["states"][f"checkpoint-{n}"] != known[key]:
            raise ValueError("Checkpoint source tree differs from its original snapshot")
    with tempfile.TemporaryDirectory(prefix="leanlean-release-audit-") as temporary:
        output = Path(temporary)
        write_tree(post, output / "post")
        if read_tree(output / "post") != expected_post:
            raise ValueError("Restored post tree differs from its original")
        n = len(frames) // 2
        restored = output / "checkpoint"
        restore_sources(folder / "checkpoints.tar.zst", f"checkpoint-{n}", restored,
                        row["checkpoints"]["sha256"], baseline=baseline)
        expected, _ = read_archive(held_by(frames[n]) / safe_name(frames[n]["archive"]), remove_scratch=True, exclude_dirs=excluded, exclude_build=exclude_build)
        if read_tree(restored) != expected:
            raise ValueError("Restored checkpoint differs from its original")
    analysis_members = check_analysis(folder / "analysis.tar.zst")
    return dict(model=row["model"], id=row["id"], checkpoints=len(frames), unique_states=len(known),
                source_files_compared=files_checked, analysis_members=analysis_members,
                result_trace_roundtrip=True, restored_post=True, restored_checkpoint=True)


def audit(root: Path, plan: Path, report: Path, workers=8):
    manifest = read_json(root / "manifest.json")
    frozen = read_json(plan)
    jobs = {(job["slug"], int(job["row"]["id"])): job for job in frozen["jobs"]}
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        ordered = sorted(manifest["results"], key=lambda row: row["checkpoints"]["input_archive_bytes"], reverse=True)
        futures = [pool.submit(check_result, dict(release=str(root), record=row,
                                                 original=jobs[(row["model"], row["id"])]))
                   for row in ordered]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(f"Audited {len(results)}/{len(futures)}: {result['model']} P{result['id']}", flush=True)
    summary = dict(results=len(results), checkpoint_positions=sum(row["checkpoints"] for row in results),
                   unique_source_states=sum(row["unique_states"] for row in results),
                   source_files_compared=sum(row["source_files_compared"] for row in results),
                   restored_posts=len(results), restored_checkpoints=len(results),
                   all_analysis_members_verified=True, all_result_trace_fields_preserved=True,
                   details=sorted(results, key=lambda row: (row["model"], row["id"])))
    write_json(report, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    parser.add_argument("--build-plan", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    result = audit(args.release, args.build_plan, args.report, args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "details"}))


if __name__ == "__main__":
    main()
