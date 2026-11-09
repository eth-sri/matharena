"""Publish validated LeanLean results using the ordinary competition output layout."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import shutil

from leanlean.common import read_json, sha256, utc_now, write_json
from leanlean.publication import digest, encoded
from leanlean.release import refresh_files, verify
from leanlean.result_trace import join_fields
from matharena.json_zst import dump_json_zst, load_json_zst


MODEL_PATHS = {
    "gpt-61-sol-max": "openai/gpt-61-sol",
    "opus-55-high": "anthropic/opus_55_high",
    "gemini-4-argon": "gemini/gemini-4-argon",
    "deepseek-v41-flash-max": "deepseek/deepseek_v41_flash_1017",
    "kimi-k31-pro": "moonshot/k31",
    "deepseek-v41-flash-dsh": "deepseek/deepseek_v41_flash_1017-dsh",
}


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def convert_result(args):
    source, staging, original = args
    row = deepcopy(original)
    model = MODEL_PATHS[row["model"]]
    old = source / "results" / row["model"] / str(row["id"])
    relative = f"{model}/artifacts/{row['id']}"
    artifacts = staging / relative
    artifacts.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(old, artifacts, copy_function=link_or_copy,
                    ignore=shutil.ignore_patterns("result.json.zst", "agent-trace.json.zst", "manifest.json"))
    summary = load_json_zst(old / "result.json.zst")
    trace = load_json_zst(old / "agent-trace.json.zst")
    combined = join_fields(summary, trace)
    if digest(encoded(combined)) != trace["canonical_json_sha256"]:
        raise ValueError("Original result fields changed during output conversion")
    if "leanlean" in summary and "leanlean" not in combined:
        combined["leanlean"] = summary["leanlean"]
    result = staging / model / f"{row['id']}.json.zst"
    dump_json_zst(combined, result)
    restored = load_json_zst(result)
    original_fields = {key: restored[key] for key in trace["source_key_order"]}
    if digest(encoded(original_fields)) != trace["canonical_json_sha256"]:
        raise ValueError("Conventional result lost original transcript fields")
    row.update(model_path=model, artifact_directory=relative,
               original_canonical_json_sha256=trace["canonical_json_sha256"],
               result=dict(path=result.relative_to(staging).as_posix(), sha256=sha256(result), bytes=result.stat().st_size))
    row.pop("agent_trace", None)
    row["checkpoints"]["local_only"] = True
    write_json(artifacts / "manifest.json", row)
    if row.get("analysis"):
        from leanlean.local_telemetry import split_telemetry

        row = split_telemetry(artifacts)
    return row


def prepare_outputs(source: Path, destination: Path, *, workers=4) -> Path:
    """Prepare and verify a replacement without modifying current competition outputs."""
    if not (source / "READY.json").is_file() or not (source / "validation.json").is_file():
        raise ValueError("Source checkpoint audit must finish before publication")
    verify(source)
    manifest = deepcopy(read_json(source / "manifest.json"))
    staging = destination.with_name("." + destination.name + ".building")
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    for name in ("provenance", "audits", "website"):
        if (source / name).is_dir():
            shutil.copytree(source / name, staging / name, copy_function=link_or_copy)
    for name in ("benchmark-manifest.json", "validation.json"):
        shutil.copyfile(source / name, staging / name)
    provenance = staging / "provenance" / "output-layout"
    provenance.mkdir(parents=True, exist_ok=True)
    for name in ("output_layout.py", "release.py", "publication.py", "local_telemetry.py"):
        shutil.copyfile(Path(__file__).with_name(name), provenance / name)
    shutil.copyfile(Path(__file__).parents[1] / "matharena/json_zst.py", provenance / "json_zst.py")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        manifest["results"] = list(pool.map(convert_result, [(source, staging, row) for row in manifest["results"]]))
    manifest.update(schema="leanlean_outputs_v1", layout="competition/provider/model/problem.json.zst",
                    model_paths=MODEL_PATHS, organized_at=utc_now())
    manifest["scope"].update(normalized_traces_separate=False, result_transcripts_inline=True,
                             checkpoints_local_only=True, telemetry_local_only=True)
    manifest["output_layout_exporter_sha256"] = sha256(Path(__file__))
    write_json(staging / "READY.json", dict(finished_at=utc_now(), results=len(manifest["results"]),
                                           validation="validation.json", layout=manifest["layout"]))
    refresh_files(staging, manifest)
    verify(staging)
    return staging


def publish_outputs(source: Path, destination: Path, backup: Path, *, workers=4) -> dict:
    if backup.exists():
        raise FileExistsError(backup)
    staging = prepare_outputs(source, destination, workers=workers)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        shutil.move(str(destination), str(backup))
    try:
        staging.rename(destination)
    except BaseException:
        if backup.exists() and not destination.exists():
            shutil.move(str(backup), str(destination))
        raise
    return verify(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--backup", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    print(publish_outputs(args.source, args.destination, args.backup, workers=args.workers))


if __name__ == "__main__":
    main()
