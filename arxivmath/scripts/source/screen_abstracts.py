#!/usr/bin/env python3
"""Cheap, resumable abstract-only triage before source download and analysis."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from matharena.arxivbench_utils import (
    conversation_response_text,
    extract_json,
    list_paper_ids,
    load_metadata,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
)
from matharena.query_client import create_query_client
from matharena.container_runtime import add_container_runtime_argument
from matharena.arxivmath_source import (
    add_source_mode_argument,
    configure_source_mode,
    validate_source_model,
    ABSTRACT_SCREEN_DECISIONS,
    ABSTRACT_SCREEN_FILENAME,
    SCHEMA_VERSION,
    SourceStateError,
    abstract_screen_input,
    abstract_screen_input_sha256,
    abstract_screen_record_is_current,
    abstract_screen_selection,
    atomic_write_json,
    load_json,
    model_record,
    portable_model_config_reference,
    sha256_file,
    sha256_text,
    utc_now,
)


DEFAULT_PROMPT = "arxivmath/prompts/source/abstract_screen.md"


def validate_abstract_screen(parsed: Any) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        raise SourceStateError("abstract screen response must be an object")
    if set(parsed) != {"decision"}:
        raise SourceStateError("abstract screen response must contain only decision")
    decision = parsed.get("decision")
    if decision not in ABSTRACT_SCREEN_DECISIONS:
        raise SourceStateError(f"abstract screen has invalid decision: {decision!r}")
    return {"decision": decision}


def identity_for(
    metadata: dict[str, Any],
    *,
    rendered_prompt: str,
    prompt_sha256: str,
    model_config_sha256: str,
) -> dict[str, str]:
    return {
        "metadata_input_sha256": abstract_screen_input_sha256(metadata),
        "rendered_prompt_sha256": sha256_text(rendered_prompt),
        "prompt_sha256": prompt_sha256,
        "model_config_sha256": model_config_sha256,
    }


def screen_record_is_current(record: Any, metadata: dict[str, Any], identity: dict[str, str]) -> bool:
    return bool(abstract_screen_record_is_current(record, metadata) and record.get("identity") == identity)


def empty_abstract_record(
    *,
    paper_id: str,
    model: str,
    model_config_arg: str,
    model_config_path: str,
    model_config_sha256: str,
    identity: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "paper_id": paper_id,
        "status": "completed",
        "model": model,
        "model_config": model_config_arg,
        "model_config_path": model_config_path,
        "model_config_sha256": model_config_sha256,
        "identity": identity,
        "decision": "reject",
        "raw": "",
        "parsed": None,
        "cost": 0.0,
        "detailed_cost": {"cost": 0.0},
        "updated_at": utc_now(),
    }


def screening_batches(queries, batch_size: int):
    # Short Codex screens hit an observed 60-new-connections/minute limit.
    # Waiting after each completed batch keeps at most two batches (50 papers)
    # in any minute, even when worker startup and response times vary.
    batch_size = min(batch_size, 25)
    for start in range(0, len(queries), batch_size):
        if start:
            print("Screening pacing: waiting 31 seconds before the next batch.", flush=True)
            time.sleep(31)
        yield start, queries[start : start + batch_size]


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Accept or reject ArXivMath papers for full-source processing.")
    parser.add_argument(
        "--model-config",
        default=None,
        help="Model config (default: openai/gpt-6-astra-medium with --false, otherwise openai/gpt-6-astra).",
    )
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--annotation-filename", default=ABSTRACT_SCREEN_FILENAME)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new model queries this run.")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    add_source_mode_argument(parser)
    add_container_runtime_argument(parser)
    args = parser.parse_args()
    configure_source_mode(args, false_model_default="openai/gpt-6-astra-medium")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    prompt_template = load_prompt_template(args.prompt)
    prompt_sha256 = sha256_text(prompt_template)
    resolved_config_path = str(Path(resolve_model_config_path(args.model_config)).resolve())
    stored_config_arg, stored_config_path = portable_model_config_reference(
        args.model_config,
        resolved_config_path,
    )
    model_config_sha256 = sha256_file(resolved_config_path)
    model_config = load_model_config(resolved_config_path)
    validate_source_model(model_config, false_mode=args.false)
    model_name = model_config["model"]
    cost_label = (
        "API-equivalent cost"
        if (model_config.get("harness_config") or {}).get("auth") in {"subscription", "chatgpt", "oauth"}
        else "Model cost"
    )
    client = None

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers is not None:
        paper_ids = paper_ids[: args.max_papers]

    queries: list[list[dict[str, str]]] = []
    query_records: list[dict[str, Any]] = []
    reused = empty_abstract_rejections = 0
    for paper_id in paper_ids:
        paper_dir = Path(args.paper_root) / paper_id
        metadata = load_metadata(args.paper_root, paper_id)
        screen_input = abstract_screen_input(metadata)
        rendered_prompt = prompt_template.format(**screen_input)
        identity = identity_for(
            metadata,
            rendered_prompt=rendered_prompt,
            prompt_sha256=prompt_sha256,
            model_config_sha256=model_config_sha256,
        )
        output_path = paper_dir / args.annotation_filename
        existing = load_json(output_path, {})
        if not args.overwrite and screen_record_is_current(existing, metadata, identity):
            reused += 1
            continue
        if not screen_input["abstract"]:
            atomic_write_json(
                output_path,
                empty_abstract_record(
                    paper_id=paper_id,
                    model=model_name,
                    model_config_arg=stored_config_arg,
                    model_config_path=stored_config_path,
                    model_config_sha256=model_config_sha256,
                    identity=identity,
                ),
            )
            empty_abstract_rejections += 1
            continue
        if args.limit is not None and len(queries) >= args.limit:
            continue
        queries.append([{"role": "user", "content": rendered_prompt}])
        query_records.append(
            {
                "paper_id": paper_id,
                "output_path": output_path,
                "identity": identity,
            }
        )

    total_cost = 0.0
    failed = completed = 0
    for batch_start, batch_queries in screening_batches(queries, args.batch_size):
        batch_records = query_records[batch_start : batch_start + len(batch_queries)]
        received: set[int] = set()
        try:
            if client is None:
                client = create_query_client(model_config, container_runtime=args.container_runtime)
            for index, conversation, cost in client.run_queries(batch_queries):
                if not isinstance(index, int) or not 0 <= index < len(batch_records):
                    continue
                received.add(index)
                context = batch_records[index]
                raw = conversation_response_text(conversation)
                parsed = extract_json(raw)
                record = model_record(
                    model=model_name,
                    config_arg=stored_config_arg,
                    config_path=stored_config_path,
                    config_sha256=model_config_sha256,
                    raw=raw,
                    parsed=parsed,
                    cost=cost,
                )
                record.update(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "paper_id": context["paper_id"],
                        "identity": context["identity"],
                    }
                )
                try:
                    record.update(validate_abstract_screen(parsed))
                    completed += 1
                except SourceStateError as exc:
                    record["status"] = "failed"
                    record["validation_error"] = str(exc)
                    failed += 1
                atomic_write_json(context["output_path"], record)
                total_cost += float(record.get("cost", 0.0) or 0.0)
        except Exception as exc:
            print(f"Abstract-screen batch failed after partial progress: {type(exc).__name__}: {exc}")
        for index, context in enumerate(batch_records):
            if index in received:
                continue
            failed += 1
            atomic_write_json(
                context["output_path"],
                {
                    "schema_version": SCHEMA_VERSION,
                    "paper_id": context["paper_id"],
                    "status": "failed",
                    "identity": context["identity"],
                    "error": "model_response_missing",
                    "updated_at": utc_now(),
                },
            )

    selection = abstract_screen_selection(
        args.paper_root,
        paper_ids=paper_ids,
        screen_filename=args.annotation_filename,
    )
    print(
        f"Abstract screening: {selection['paper_count']} papers, "
        f"{len(selection['accepted'])} accepted, {len(selection['rejected'])} rejected, "
        f"{len(selection['incomplete'])} incomplete. "
        f"Newly completed {completed}, reused {reused}, empty-abstract rejects "
        f"{empty_abstract_rejections}, failed responses {failed}. "
        f"{cost_label} ${total_cost:.2f}."
    )
    return 0 if selection["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
