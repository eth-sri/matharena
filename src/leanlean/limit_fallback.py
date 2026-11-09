"""Score failed submissions at the attempt's latest passing checkpoint.

A submission that passes final verification is scored as submitted. One that
fails is scored at the newest source state the attempt held that passes,
whether the run ended on its own, at the time limit or at the dollar limit
(after its cost-limit grace turn). The cut is the run's final source
observation; no attempt gets extra time. Only states last observed at most
`WINDOW_HOURS` before the cut count; with none passing, the submission scores zero.

Candidates are the attempt's recorded checkpoint states, newest first by when
they were last observed, so a state the model reverted to counts as recent.
Each is checked offline in the evaluation image with the final verification
container (registered fixtures restored, `lake build`, Comparator and the
definition stage) on a warm build tree that only ever held earlier candidates;
no agent build cache is used. The first passing state is selected. A resumed
attempt's history starts from its predecessor's source, so it can happen that
no recorded state passes; the attempt then keeps its failed submission.

A selection that differs from the submitted source gets a copied result record
whose `leanlean_submission` is an archive of that checkpoint; the original
submission stays referenced as `pre_limit_fallback_leanlean_submission`. The
copied records then go through ordinary final verification (`leanlean.postprocess`).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
import uuid
from pathlib import Path

from leanlean.common import exclusive_lock, read_json, sha256, tree_sha256, utc_now, within, write_json
from leanlean.evaluation import capture_submission
from leanlean.postprocess import container_command, extract_submission, restore_fixtures
from leanlean.publication import read_archive
from leanlean.replay import sync_sources
from matharena.json_zst import dump_json_zst, load_json_zst

PIPELINE = ("postprocess.py", "postprocess_container.py", "definitions_container.py", "definition_audit.lean")


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_limits(record: dict) -> dict | None:
    return next((cost["run_limits"] for cost in record.get("detailed_costs", []) if cost.get("run_limits")), None)


def cutoff(limits: dict | None, observations: list[dict]) -> dict:
    """The final observation, which holds the attempt's last source, and how the run ended.

    A time limit reached during a cost-limit grace turn still stopped the model mid-work.
    """
    limits = limits or {}
    if limits.get("exceeded") == "time_limit":
        reason = "time_limit"
    elif limits.get("exceeded") in {"cost_limit", "cost_grace_timeout"} or limits.get("cost_limit_grace"):
        reason = "cost_limit"
    else:
        reason = "submitted"
    return dict(reason=reason, limit_at=None, observation=len(observations) - 1)


WINDOW_HOURS = 3


def candidates(observations: list[dict], cut: int, window_hours: float | None = WINDOW_HOURS) -> list[dict]:
    """Distinct source states up to the cut, most recently observed first.

    With `window_hours`, only states last observed within that long before the cut.
    """
    seen, ordered = set(), []
    since = None if window_hours is None else observations[cut]["started_wall_ns"] - window_hours * 3.6e12
    for row in reversed(observations[: cut + 1]):
        # A failed snapshot (e.g. a file vanished mid-copy) records an error and no state.
        if "source_sha256" not in row:
            continue
        if since is not None and row["started_wall_ns"] < since:
            break
        if row["source_sha256"] not in seen:
            seen.add(row["source_sha256"])
            ordered.append(dict(source_sha256=row["source_sha256"], last_observation=row["observation_index"],
                                last_observed_wall_ns=row["started_wall_ns"]))
    return ordered


def contents(path: Path) -> dict:
    """File contents and links, ignoring scratch files, build output and permissions."""
    state, _ = read_archive(path, remove_scratch=True, exclude_build=True)
    return {name: (kind, hashlib.sha256(data).hexdigest() if kind == "file" else data)
            for name, (kind, data, _mode) in state.items()}


def checkpoint_archive(attempt: Path, frames: list[dict], source: str) -> Path:
    frame = next(frame for frame in frames if frame["source_sha256"] == source)
    archive = within(attempt, frame["archive"])
    with gzip.open(archive, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source:
            raise ValueError(f"Checkpoint archive checksum differs: {archive}")
    return archive


def known_verdict(roots: list[Path], key: str, result_sha256: str) -> dict | None:
    """The final verification of the original submission, if one was run."""
    for root in roots:
        for path in sorted([*root.glob(f"p*/verification/repositories/{key}/result.json"),
                            *root.glob(f"repositories/{key}/result.json")]):
            result = read_json(path)
            if result.get("result_sha256") == result_sha256 and result.get("status") in {"verified", "verification_failed"}:
                return dict(path=str(path), verified=result["status"] == "verified")
    return None


class Screen:
    """Check one attempt's candidates in order on a warm, grader-owned build tree."""

    def __init__(self, job: dict, output: Path, runtime: dict, proofs: Path):
        self.job, self.runtime, self.proofs = job, runtime, proofs
        self.root = output / "screening" / f"p{job['problem_idx']}"
        self.entry = job["entry"]

    def check(self, archive: Path, source: str) -> dict:
        directory = self.root / "states" / source
        if (directory / "result.json").exists():
            saved = read_json(directory / "result.json")
            if saved["status"] in {"passed", "failed", "rejected"}:
                return saved
        directory.mkdir(parents=True, exist_ok=True)
        work, staging = self.root / "work", self.root / "staging"
        repository = work / "repository"
        result = dict(source_sha256=source, archive=str(archive), archive_sha256=sha256(archive), started_at=utc_now())
        shutil.rmtree(staging, ignore_errors=True)
        try:
            extract_submission(archive, staging)
            result["fixtures"] = restore_fixtures(staging, Path(self.job["baseline"]), self.entry)
        except ValueError as error:
            result.update(status="rejected", error=str(error), finished_at=utc_now())
            write_json(directory / "result.json", result)
            return result
        if not repository.exists():
            # Only a tree with its package link becomes the build tree.
            work.mkdir(parents=True, exist_ok=True)
            fresh = work / "repository.partial"
            shutil.rmtree(fresh, ignore_errors=True)
            shutil.copytree(staging, fresh)
            lake = fresh / self.entry["project_path"] / ".lake"
            lake.mkdir(exist_ok=True)
            (lake / "packages").symlink_to(Path("/opt/leanlean/packages") / self.entry["key"], target_is_directory=True)
            fresh.rename(repository)
            result["cache"] = "fresh"
        else:
            result["cache"] = sync_sources(staging, repository)
        shutil.rmtree(staging)
        for name in ("verification.json", "build.stdout.log", "build.stderr.log", "comparator.stdout.log", "comparator.stderr.log"):
            (work / name).unlink(missing_ok=True)
        write_json(work / "leanlean-project.json", self.entry)
        shutil.copyfile(self.proofs, work / "definition-proofs.json")
        name = f"leanlean-fallback-p{self.job['problem_idx']}-{uuid.uuid4().hex[:10]}"
        command = container_command({"directory": str(self.root), "entry": self.entry}, "work", self.runtime,
                                    self.root.parents[1] / "pipeline", name)
        command[-1] = "after"
        try:
            with (directory / "container.log").open("w") as log:
                completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=25260)
            result["returncode"] = completed.returncode
            result["verification"] = read_json(work / "verification.json")
        except subprocess.TimeoutExpired:
            result.update(returncode=124, verification={"status": "timeout"})
        finally:
            subprocess.run([self.runtime["container_engine"], "rm", "--force", name], capture_output=True, timeout=60)
        for log in work.glob("*.log"):
            shutil.copyfile(log, directory / log.name)
        result["status"] = "passed" if result["returncode"] == 0 and result["verification"].get("status") == "passed" else "failed"
        result["finished_at"] = utc_now()
        write_json(directory / "result.json", result)
        return result


def select(job: dict, output: Path, runtime: dict, proofs: Path) -> dict:
    attempt = Path(job["attempt"])
    observations = rows(attempt / "observations.jsonl")
    frames = rows(attempt / "checkpoints.jsonl")
    cut = job["cut"]
    submitted = contents(Path(job["submission"]["archive"]))
    screen = Screen(job, output, runtime, proofs)
    tried = []
    for candidate in candidates(observations, cut["observation"]):
        archive = checkpoint_archive(attempt, frames, candidate["source_sha256"])
        submitted_state = contents(archive) == submitted
        if submitted_state and job.get("known") is not None:
            # The submitted source was already checked by final verification.
            verdict = dict(status="passed" if job["known"]["verified"] else "failed", final_verification=job["known"]["path"])
        else:
            print(f"STATUS P{job['problem_idx']} checking {candidate['source_sha256'][:12]}", flush=True)
            verdict = screen.check(archive, candidate["source_sha256"])
        tried.append(dict(candidate, status=verdict["status"], submitted_state=submitted_state,
                          **({"final_verification": verdict["final_verification"]} if "final_verification" in verdict else {})))
        print(f"STATUS P{job['problem_idx']} {candidate['source_sha256'][:12]}: {verdict['status']}", flush=True)
        if verdict["status"] == "passed":
            return dict(candidate, archive=str(archive), submitted_state=submitted_state, tried=tried)
    return dict(source_sha256=None, submitted_state=True, tried=tried)


def fallback_record(job: dict, chosen: dict, output: Path) -> tuple[Path, str]:
    """Copy the original record, pointing its submission at the selected checkpoint."""
    record = load_json_zst(Path(job["result_path"]))
    step = record["history"][0][0]
    attempt = Path(job["attempt"])
    # Release packing finds the attempt two levels above the submission archive.
    destination = attempt / f"limit-fallback-{chosen['source_sha256'][:16]}"
    with tempfile.TemporaryDirectory(dir=output) as scratch:
        repository = Path(scratch) / "repository"
        repository.mkdir()
        with tarfile.open(chosen["archive"]) as stream:
            stream.extractall(repository, filter="data")
        submission = capture_submission(Path(scratch), job["entry"], output=destination)
    if contents(Path(submission["archive"])) != contents(Path(chosen["archive"])):
        raise ValueError("Fallback submission differs from its checkpoint")
    step["pre_limit_fallback_leanlean_submission"] = step["leanlean_submission"]
    step["leanlean_submission"] = submission
    step["repo_submission"] = {"type": "repository", "grader": "leanlean", **submission}
    record["leanlean_limit_fallback"] = fallback_metadata(job, chosen)
    target = output / "results" / f"{job['problem_idx']}.json.zst"
    target.parent.mkdir(parents=True, exist_ok=True)
    dump_json_zst(record, target)
    return target, sha256(target)


def fallback_metadata(job: dict, chosen: dict) -> dict:
    observations = rows(Path(job["attempt"]) / "observations.jsonl")
    held = observations[job["cut"]["observation"]]
    return dict(
        schema="leanlean_limit_fallback_v1",
        reason=job["cut"]["reason"],
        limit_at=job["cut"]["limit_at"],
        cut_observation=held["observation_index"],
        cut_source_sha256=held["source_sha256"],
        scored_source_sha256=chosen["source_sha256"],
        scored_last_observation=chosen["last_observation"],
        scored_last_observed_wall_ns=chosen["last_observed_wall_ns"],
        scored_is_submission=chosen["submitted_state"],
        candidates=chosen["tried"],
        original_result_path=job["result_path"],
        original_result_sha256=job["result_sha256"],
        original_submission_sha256=job["submission"]["sha256"],
    )


def plan(index: Path, bundle: Path, verified: list[Path], problems: list[int] | None) -> list[dict]:
    entries = read_json(bundle / "manifest.json")["projects"]
    jobs = []
    for raw_idx, pinned in sorted(read_json(index).items(), key=lambda item: int(item[0])):
        idx = int(raw_idx)
        if problems and idx not in problems:
            continue
        path = Path(pinned["path"])
        if sha256(path) != pinned["sha256"]:
            raise ValueError(f"Result checksum mismatch: {path}")
        record = load_json_zst(path)
        submission = record["history"][0][0]["leanlean_submission"]
        entry = entries[idx - 1]
        if (entry["key"], entry["project_tree_sha256"]) != (submission["key"], submission["baseline_sha256"]):
            raise ValueError(f"Wrong baseline for P{idx}")
        if sha256(Path(submission["archive"])) != submission["sha256"]:
            raise ValueError(f"Submission checksum mismatch: P{idx}")
        known = known_verdict(verified, entry["key"], pinned["sha256"])
        if known and known["verified"]:
            continue
        attempt = Path(submission["archive"]).parent.parent
        cut = cutoff(run_limits(record), rows(attempt / "observations.jsonl"))
        baseline = within(bundle, entry["project_directory"])
        if tree_sha256(baseline) != entry["project_tree_sha256"]:
            raise ValueError(f"Baseline tree changed: {entry['key']}")
        jobs.append(dict(problem_idx=idx, entry=entry, baseline=str(baseline), result_path=str(path),
                         result_sha256=pinned["sha256"], pinned=pinned, submission=submission,
                         attempt=str(attempt), cut=cut, known=known))
    return jobs


def run(job: dict, output: Path, runtime: dict, proofs_root: Path) -> dict:
    receipt = output / "selections" / f"{job['problem_idx']}.json"
    if receipt.exists():
        return read_json(receipt)
    proofs = output / "definition-proofs" / f"{job['entry']['key']}.json"
    proofs.parent.mkdir(parents=True, exist_ok=True)
    registered = proofs_root / f"{job['entry']['key']}.json"
    write_json(proofs, read_json(registered) if registered.is_file() else {})
    chosen = select(job, output, runtime, proofs)
    if chosen["submitted_state"]:
        pinned = job["pinned"]
    else:
        path, digest = fallback_record(job, chosen, output)
        pinned = dict(original_path=str(path), path=str(path), sha256=digest)
    value = dict(problem_idx=job["problem_idx"], key=job["entry"]["key"], reason=job["cut"]["reason"],
                 limit_at=job["cut"]["limit_at"], original=job["pinned"], selected=pinned,
                 changed=not chosen["submitted_state"], passed=chosen["source_sha256"] is not None,
                 scored_source_sha256=chosen["source_sha256"],
                 candidates=chosen["tried"], finished_at=utc_now())
    write_json(receipt, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True, help="combined-results.json of the selected results")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified", type=Path, action="append", default=[],
                        help="postprocess output whose verdicts for the submitted sources can be reused")
    parser.add_argument("--definition-proofs", type=Path,
                        default=Path(__file__).resolve().parents[2] / "data/leanlean/definition-proofs")
    parser.add_argument("--problem", type=int, action="append")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(output / ".lock"):
        if (output / "plan.json").exists():
            saved = read_json(output / "plan.json")
            if saved["index_sha256"] != sha256(args.index) or saved["bundle_sha256"] != sha256(args.bundle / "manifest.json"):
                raise ValueError("Inputs differ from the existing plan; use a new output directory")
            jobs = saved["jobs"]
        else:
            jobs = plan(args.index, args.bundle, args.verified, args.problem)
            (output / "pipeline").mkdir(exist_ok=True)
            for name in PIPELINE:
                shutil.copyfile(Path(__file__).with_name(name), output / "pipeline" / name)
            write_json(output / "plan.json", dict(
                created_at=utc_now(), index=str(args.index), index_sha256=sha256(args.index),
                bundle_sha256=sha256(args.bundle / "manifest.json"), runtime=read_json(args.bundle / "eval-runtime.json"),
                pipeline_sha256={name: sha256(output / "pipeline" / name) for name in PIPELINE}, jobs=jobs))
        runtime = read_json(output / "plan.json")["runtime"]
        print(f"STATUS {len(jobs)} failed or unverified submissions", flush=True)
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
            done = list(pool.map(lambda job: run(job, output, runtime, args.definition_proofs), jobs))
        # Passing submissions keep their pin; the index feeds ordinary final verification.
        index = read_json(args.index)
        for row in done:
            index[str(row["problem_idx"])] = row["selected"]
        write_json(output / "combined-results.json", index)
        write_json(output / "summary.json", dict(
            finished_at=utc_now(), attempts=len(done), changed=[row["problem_idx"] for row in done if row["changed"]],
            results=[{k: v for k, v in row.items() if k != "candidates"} for row in done]))


if __name__ == "__main__":
    main()
