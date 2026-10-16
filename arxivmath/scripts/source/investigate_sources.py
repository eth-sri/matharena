#!/usr/bin/env python3
"""Generate at most one source-grounded ArXivMath question per accepted paper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from matharena.arxivbench_utils import (
    conversation_response_text,
    extract_json,
    list_paper_ids,
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
    ABSTRACT_SCREEN_FILENAME,
    DEFAULT_INVESTIGATION_MAX_TOKENS,
    INVESTIGATION_FILENAME,
    MODEL_SOURCE_CLEANING_VERSION,
    SCHEMA_VERSION,
    SourceStateError,
    atomic_write_json,
    build_investigation_source,
    investigation_payload_sha256,
    load_json,
    model_record,
    metadata_arxiv_version,
    paper_source_unavailable,
    portable_model_config_reference,
    read_source_artifacts,
    abstract_screen_selection,
    sha256_file,
    sha256_text,
    utc_now,
    validate_investigation_payload,
)


DEFAULT_PROMPT = "arxivmath/prompts/source/investigate_source.md"


def state_is_current(
    state: Any,
    identity: dict[str, Any],
    source_input_manifest: dict[str, Any],
) -> bool:
    if (
        not isinstance(state, dict)
        or state.get("identity") != identity
        or state.get("source_input") != source_input_manifest
    ):
        return False
    if state.get("status") == "excluded":
        return state.get("reason") == "source_too_large"
    if state.get("status") != "completed" or not isinstance(state.get("investigation"), dict):
        return False
    investigation = state["investigation"]
    return investigation.get("investigation_sha256") == investigation_payload_sha256(investigation)


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Investigate each accepted TeX source and generate zero or one ArXivMath question."
    )
    parser.add_argument(
        "--model-config",
        default=None,
        help="Model config (default: openai/gpt-6-astra-high with --false, otherwise openai/gpt-6-astra).",
    )
    parser.add_argument("--paper-root", default="arxivmath/paper")
    parser.add_argument("--abstract-screen-filename", default=ABSTRACT_SCREEN_FILENAME)
    parser.add_argument("--source-cache", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--annotation-filename", default=INVESTIGATION_FILENAME)
    parser.add_argument(
        "--max-source-tokens",
        type=int,
        default=DEFAULT_INVESTIGATION_MAX_TOKENS,
        help="Conservative token ceiling; larger cleaned sources are excluded without an API call.",
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None, help="Maximum new model requests.")
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-source-unavailable", action="store_true")
    add_source_mode_argument(parser)
    add_container_runtime_argument(parser)
    args = parser.parse_args()
    configure_source_mode(args)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.max_source_tokens < 20_000:
        parser.error("--max-source-tokens must be at least 20000")
    if args.max_cost is not None and args.max_cost <= 0:
        parser.error("--max-cost must be positive")

    prompt_template = load_prompt_template(args.prompt)
    prompt_sha256 = sha256_text(prompt_template)
    resolved_config_path = str(Path(resolve_model_config_path(args.model_config)).resolve())
    stored_config_arg, stored_config_path = portable_model_config_reference(args.model_config, resolved_config_path)
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
    screen_summary = abstract_screen_selection(
        args.paper_root,
        paper_ids=paper_ids,
        screen_filename=args.abstract_screen_filename,
    )
    paper_ids = screen_summary["accepted"]
    print(
        f"Abstract screen: {len(paper_ids)} accepted, {len(screen_summary['rejected'])} rejected, "
        f"{len(screen_summary['incomplete'])} not ready (skipped)."
    )

    query_contexts: list[dict[str, Any]] = []
    reused = oversized = unavailable = preflight_failed = 0
    for paper_id in paper_ids:
        paper_dir = Path(args.paper_root) / paper_id
        if args.allow_source_unavailable and paper_source_unavailable(paper_dir):
            unavailable += 1
            continue
        output_path = paper_dir / args.annotation_filename
        try:
            source_text, provenance, _ = read_source_artifacts(paper_dir, args.source_cache)
            if args.false and (
                provenance.get("unresolved_include_count")
                or metadata_arxiv_version(load_json(paper_dir / "metadata.json")).canonical_id != provenance["arxiv_id"]
            ):
                raise SourceStateError("BrokenArXiv requires complete source matching the pinned metadata revision")
            screen_path = paper_dir / args.abstract_screen_filename
            screen = load_json(screen_path)
            if not isinstance(screen, dict):
                raise SourceStateError("abstract screen record is missing")
            source_input = build_investigation_source(source_text)
            source_input_manifest = source_input.manifest_record()
            identity = {
                "source_sha256": provenance["combined_source_sha256"],
                "screen_sha256": sha256_file(screen_path),
                "source_input_sha256": source_input_manifest["source_sha256"],
                "prompt_sha256": prompt_sha256,
                "model_config_sha256": model_config_sha256,
                "max_source_tokens": args.max_source_tokens,
                "cleaning_version": MODEL_SOURCE_CLEANING_VERSION,
            }
            if args.false:
                identity["benchmark"] = "brokenarxiv"
            existing = load_json(output_path, {})
            if not args.overwrite and state_is_current(existing, identity, source_input_manifest):
                reused += 1
                continue
            source_record = {
                key: provenance.get(key)
                for key in (
                    "arxiv_id",
                    "base_id",
                    "version",
                    "cache_key",
                    "combined_source_sha256",
                    "unresolved_include_count",
                    "unresolved_includes_allowed",
                )
            }
            if source_input_manifest["approximate_tokens"] > args.max_source_tokens:
                oversized += 1
                atomic_write_json(
                    output_path,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "paper_id": paper_id,
                        "arxiv_id": provenance["arxiv_id"],
                        "status": "excluded",
                        "reason": "source_too_large",
                        "identity": identity,
                        "source": source_record,
                        "source_input": source_input_manifest,
                        "abstract_screen": {"decision": screen["decision"]},
                        "updated_at": utc_now(),
                    },
                )
                print(
                    f"Excluded oversized source {paper_id}: approximately "
                    f"{source_input_manifest['approximate_tokens']} tokens exceeds "
                    f"the {args.max_source_tokens} token limit."
                )
                continue
            rendered = prompt_template.format(
                arxiv_id=provenance["arxiv_id"],
                source_text=source_input.text,
            )
            query_contexts.append(
                {
                    "paper_id": paper_id,
                    "output_path": output_path,
                    "source_text": source_input.text,
                    "provenance": provenance,
                    "screen": screen,
                    "identity": identity,
                    "source_input_manifest": source_input_manifest,
                    "source_record": source_record,
                    "query": [{"role": "user", "content": rendered}],
                }
            )
            if args.limit is not None and len(query_contexts) >= args.limit:
                break
        except Exception as exc:
            preflight_failed += 1
            atomic_write_json(
                output_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "paper_id": paper_id,
                    "status": "failed",
                    "reason": "investigation_preflight_failed",
                    "error_type": type(exc).__name__,
                    "detail": str(exc)[:2000],
                    "updated_at": utc_now(),
                },
            )
            print(f"Investigation preflight failed for {paper_id}: {type(exc).__name__}: {exc}")

    total_cost = 0.0
    completed = failed = generated = rejected = 0
    budget_stopped = False
    for batch_start in range(0, len(query_contexts), args.batch_size):
        if args.max_cost is not None and total_cost >= args.max_cost:
            budget_stopped = True
            break
        batch = query_contexts[batch_start : batch_start + args.batch_size]
        received: set[int] = set()
        try:
            if client is None:
                client = create_query_client(model_config, container_runtime=args.container_runtime)
            for index, conversation, cost in client.run_queries([value["query"] for value in batch]):
                if not isinstance(index, int) or not 0 <= index < len(batch):
                    continue
                received.add(index)
                context = batch[index]
                raw = conversation_response_text(conversation)
                parsed = extract_json(raw)
                generation = model_record(
                    model=model_name,
                    config_arg=stored_config_arg,
                    config_path=stored_config_path,
                    config_sha256=model_config_sha256,
                    raw=raw,
                    parsed=parsed,
                    cost=cost,
                )
                if args.false:
                    generation["prompt_path"] = args.prompt
                validation = validate_investigation_payload(
                    parsed,
                    context["source_text"],
                    false_mode=args.false,
                )
                state: dict[str, Any] = {
                    "schema_version": SCHEMA_VERSION,
                    "paper_id": context["paper_id"],
                    "arxiv_id": context["provenance"]["arxiv_id"],
                    "identity": context["identity"],
                    "source": context["source_record"],
                    "source_input": context["source_input_manifest"],
                    "abstract_screen": {"decision": context["screen"]["decision"]},
                    "generation": generation,
                    "updated_at": utc_now(),
                }
                if validation.get("valid") is True:
                    state["status"] = "completed"
                    state["investigation"] = validation["investigation"]
                    if "parser_validation" in validation:
                        state["parser_validation"] = validation["parser_validation"]
                    completed += 1
                    if validation["investigation"].get("keep") is True:
                        generated += 1
                    else:
                        rejected += 1
                else:
                    state["status"] = "failed"
                    state["validation_error"] = validation
                    failed += 1
                atomic_write_json(context["output_path"], state)
                total_cost += float(generation.get("cost", 0.0) or 0.0)
        except Exception as exc:
            print(f"Investigation batch failed after partial progress: {type(exc).__name__}: {exc}")
        for index, context in enumerate(batch):
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
                    "reason": "model_response_missing",
                    "updated_at": utc_now(),
                },
            )

    processed_or_reused = completed + reused + oversized
    expected_available = len(paper_ids) - unavailable
    incomplete = max(0, expected_available - processed_or_reused)
    print(
        f"Source investigation complete: generated {generated}, rejected {rejected}, reused {reused}, "
        f"failed {failed + preflight_failed}, oversized {oversized}, "
        f"unavailable {unavailable}, incomplete {incomplete}. "
        f"{cost_label} ${total_cost:.6f}."
    )
    if budget_stopped:
        print(f"Stopped before the next batch after reaching the " f"${args.max_cost:.2f} tracked stage budget.")
    return 0 if not budget_stopped and failed == 0 and preflight_failed == 0 and incomplete == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
