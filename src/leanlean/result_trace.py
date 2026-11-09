"""Separate result metrics from normalized agent history without losing JSON fields."""

from __future__ import annotations

import json
from pathlib import Path

import zstandard

from leanlean.common import sha256, write_json
from leanlean.publication import digest, encoded
from matharena.json_zst import dump_json_zst, load_json_zst

TRACE_FIELDS = {"messages", "history", "events", "model_requests", "reasoning", "trace"}


def join_fields(result: dict, trace: dict) -> dict:
    fields = trace["fields"]
    order = trace["source_key_order"]
    # Benchmark grading is derived after the original harness record was saved.
    # It belongs in the result summary, not in a reconstructed original trace.
    if "leanlean" not in order and "leanlean" in result:
        result = {key: value for key, value in result.items() if key != "leanlean"}
    if set(result) & set(fields):
        raise ValueError("Trace fields overlap result metadata")
    combined = {**result, **fields}
    if len(order) != len(set(order)) or set(order) != set(combined):
        raise ValueError("Trace/result fields do not reconstruct the source record")
    return {key: combined[key] for key in order}


def split_file(path: Path) -> dict:
    original = path.read_bytes()
    with zstandard.ZstdDecompressor().stream_reader(original) as stream:
        raw = stream.read()
    source = json.loads(raw)
    result = {key: value for key, value in source.items() if key not in TRACE_FIELDS}
    trace = dict(schema="leanlean_agent_trace_v1", fields={key: value for key, value in source.items() if key in TRACE_FIELDS},
                 source_key_order=list(source), original_result_sha256=digest(original), original_json_sha256=digest(raw),
                 canonical_json_sha256=digest(encoded(source)))
    restored = join_fields(result, trace)
    if encoded(restored) != encoded(source):
        raise ValueError("Result/trace split lost source fields")
    encoder = None
    for indent in (None, 2, 4):
        for ensure_ascii in (True, False):
            if json.dumps(restored, indent=indent, ensure_ascii=ensure_ascii).encode() == raw:
                encoder = dict(indent=indent, ensure_ascii=ensure_ascii)
                break
        if encoder is not None:
            break
    trace["source_json_encoder"] = encoder
    trace_path = path.with_name("agent-trace.json.zst")
    if trace_path.exists():
        raise FileExistsError(trace_path)
    dump_json_zst(trace, trace_path)
    temporary = path.with_name(".result-summary.json.zst")
    dump_json_zst(result, temporary)
    if encoded(join_fields(load_json_zst(temporary), load_json_zst(trace_path))) != encoded(source):
        raise ValueError("Compressed result/trace failed round-trip validation")
    temporary.replace(path)
    return dict(result=dict(path=path.name, sha256=sha256(path), bytes=path.stat().st_size),
                agent_trace=dict(path=trace_path.name, sha256=sha256(trace_path), bytes=trace_path.stat().st_size),
                original_result_sha256=digest(original), original_json_sha256=digest(raw),
                json_fields_roundtrip=True, original_json_bytes_reconstructable=encoder is not None)


def separate_release(root: Path) -> dict:
    from leanlean.common import read_json
    from leanlean.release import refresh_files

    manifest = read_json(root / "manifest.json")
    for row in manifest["results"]:
        folder = root / "results" / row["model"] / str(row["id"])
        if row.get("agent_trace"):
            continue
        source_identity = row.get("original_result_sha256", row.get("result_sha256"))
        if sha256(folder / "result.json.zst") != source_identity:
            raise ValueError("Original normalized result changed before trace split")
        record = split_file(folder / "result.json.zst")
        row.update(record)
        row.pop("result_sha256", None)
        write_json(folder / "manifest.json", row)
    manifest["scope"]["normalized_traces_separate"] = True
    manifest["trace_splitter_sha256"] = sha256(Path(__file__))
    refresh_files(root, manifest)
    return manifest


def attach_metrics(root: Path) -> dict:
    import gzip
    from leanlean.common import read_json
    from leanlean.release import refresh_files

    manifest = read_json(root / "manifest.json")
    for row in manifest["results"]:
        folder = root / "results" / row["model"] / str(row["id"])
        with gzip.open(root / "website" / row["model"] / str(row["id"]) / "detail.json.gz", "rt") as stream:
            detail = json.load(stream)
        result = load_json_zst(folder / "result.json.zst")
        if "leanlean" in result and row.get("metrics") != result["leanlean"]:
            raise ValueError("Cannot overwrite an original or modified leanlean field")
        metrics = dict(schema_version=1, model_id=row["model"], problem_id=row["id"], key=row["key"],
                       status=detail.get("status", "verified"), verified=detail.get("verified", False),
                       baseline_tokens=detail["baseline"], post_tokens=detail["final"],
                       compression_percent=detail["compression"],
                       credited_compression_percent=detail.get("credited_compression"),
                       cost_usd=detail.get("cost"), elapsed_seconds=detail["elapsed"])
        result["leanlean"] = metrics
        dump_json_zst(result, folder / "result.json.zst")
        row["metrics"] = metrics
        row["result"].update(sha256=sha256(folder / "result.json.zst"), bytes=(folder / "result.json.zst").stat().st_size)
        write_json(folder / "manifest.json", row)
    manifest["scope"]["result_summaries_include_grading"] = True
    refresh_files(root, manifest)
    return manifest


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("release", type=Path)
    args = parser.parse_args()
    manifest = separate_release(args.release)
    print(json.dumps(dict(results=len(manifest["results"]), bytes=manifest["total_bytes"])))


if __name__ == "__main__":
    main()
