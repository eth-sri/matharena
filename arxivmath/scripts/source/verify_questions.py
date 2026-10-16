#!/usr/bin/env python3
"""Independently verify the single question produced for each investigated paper."""

from __future__ import annotations

import argparse
import json
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
    INVESTIGATION_FILENAME,
    FINAL_ANNOTATION_FILENAME,
    FALSE_FINAL_FILENAME,
    sync_review_annotation,
    SourceStateError,
    VERIFICATION_CHECKS,
    answers_equivalent,
    atomic_write_json,
    build_investigation_source,
    canonical_json_sha256,
    investigation_payload_sha256,
    investigation_verification_is_current,
    load_json,
    model_record,
    paper_source_unavailable,
    portable_model_config_reference,
    read_source_artifacts,
    abstract_screen_selection,
    sha256_file,
    sha256_text,
    source_window,
    utc_now,
    validate_false_payload,
    validate_normalized_investigation,
)


DEFAULT_PROMPT = "arxivmath/prompts/source/verify_question.md"


def build_verification_evidence(
    source_text: str,
    investigation: dict[str, Any],
    *,
    context_lines: int,
) -> str:
    windows: list[str] = []
    seen: set[str] = set()
    for evidence in investigation.get("evidence") or []:
        if not isinstance(evidence, dict):
            continue
        start = evidence.get("start_char")
        end = evidence.get("end_char")
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        window = source_window(source_text, start, end, context_lines=context_lines)
        digest = sha256_text(window)
        if digest not in seen:
            seen.add(digest)
            windows.append(window)
    if not windows:
        raise SourceStateError("investigation has no usable source evidence")
    return "\n\n% MATHARENA_EVIDENCE_WINDOW_BREAK\n\n".join(windows)


def validate_verification(parsed: Any, expected_answer: str) -> dict[str, Any]:
    if not isinstance(parsed, dict):
        return {"valid": False, "reason": "verification_not_object"}
    allowed_fields = {"keep", "derived_answer", "reason", *VERIFICATION_CHECKS}
    unexpected = sorted(set(parsed) - allowed_fields)
    if unexpected:
        return {
            "valid": False,
            "reason": "verification_has_unexpected_fields",
            "fields": unexpected,
        }
    if not isinstance(parsed.get("keep"), bool):
        return {"valid": False, "reason": "verification_keep_not_boolean"}
    if not all(isinstance(parsed.get(name), bool) for name in VERIFICATION_CHECKS):
        return {"valid": False, "reason": "verification_checks_missing_or_non_boolean"}
    derived_answer = parsed.get("derived_answer")
    if derived_answer is not None and not isinstance(derived_answer, str):
        return {"valid": False, "reason": "derived_answer_has_invalid_type"}
    reason = parsed.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        return {"valid": False, "reason": "verification_reason_missing"}
    checks_pass = all(parsed[name] is True for name in VERIFICATION_CHECKS)
    answer_matches = answers_equivalent(derived_answer, expected_answer)
    passed = bool(parsed["keep"] is True and checks_pass and answer_matches)
    if parsed["keep"] is True and not checks_pass:
        return {"valid": False, "reason": "keep_true_with_failed_check"}
    if parsed["keep"] is False and checks_pass:
        return {"valid": False, "reason": "keep_false_without_failed_check"}
    return {
        "valid": True,
        "passed": passed,
        "checks_pass": checks_pass,
        "answer_matches": answer_matches,
        "derived_answer": derived_answer,
        "reason": reason.strip(),
        "checks": {name: parsed[name] for name in VERIFICATION_CHECKS},
    }


def main() -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Independently verify generated ArXivMath questions.")
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
    parser.add_argument("--context-lines", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-papers", type=int, default=None)
    parser.add_argument("--max-cost", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-source-unavailable", action="store_true")
    add_source_mode_argument(parser)
    add_container_runtime_argument(parser)
    args = parser.parse_args()
    configure_source_mode(args)
    if args.context_lines < 0 or args.batch_size < 1:
        parser.error("context lines must be nonnegative and batch size positive")
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

    contexts: list[dict[str, Any]] = []
    reused = skipped_rejections = oversized = unavailable = preflight_failed = deferred = 0
    for paper_id in paper_ids:
        paper_dir = Path(args.paper_root) / paper_id
        if args.allow_source_unavailable and paper_source_unavailable(paper_dir):
            sync_review_annotation(
                paper_dir,
                investigation_filename=args.annotation_filename,
                final_filename=FALSE_FINAL_FILENAME if args.false else FINAL_ANNOTATION_FILENAME,
            )
            unavailable += 1
            continue
        state_path = paper_dir / args.annotation_filename
        try:
            sync_review_annotation(
                paper_dir,
                investigation_filename=args.annotation_filename,
                final_filename=FALSE_FINAL_FILENAME if args.false else FINAL_ANNOTATION_FILENAME,
            )
            state = load_json(state_path)
            if (
                isinstance(state, dict)
                and state.get("status") == "excluded"
                and state.get("reason") == "source_too_large"
            ):
                oversized += 1
                continue
            if not isinstance(state, dict) or state.get("status") != "completed":
                raise SourceStateError("source investigation is missing or incomplete")
            investigation = state.get("investigation")
            if not isinstance(investigation, dict):
                raise SourceStateError("source investigation payload is missing")
            if investigation.get("keep") is not True:
                skipped_rejections += 1
                continue
            raw_source_text, provenance, _ = read_source_artifacts(paper_dir, args.source_cache)
            if (state.get("identity") or {}).get("source_sha256") != provenance["combined_source_sha256"]:
                raise SourceStateError("investigation was generated from a different source snapshot")
            source_input = build_investigation_source(raw_source_text)
            source_input_manifest = source_input.manifest_record()
            if (state.get("identity") or {}).get("source_input_sha256") != source_input_manifest["source_sha256"]:
                raise SourceStateError("investigation used a stale cleaned source")
            if state.get("source_input") != source_input_manifest:
                raise SourceStateError("cleaned source manifest is stale")
            if args.false:
                if (state.get("identity") or {}).get("screen_sha256") != sha256_file(
                    paper_dir / args.abstract_screen_filename
                ):
                    raise SourceStateError("investigation used a stale abstract screen")
                if (
                    investigation.get("benchmark") != "brokenarxiv"
                    or validate_normalized_investigation(investigation, source_input.text).get("valid") is not True
                ):
                    raise SourceStateError("false-statement investigation is invalid or stale")
                rendered = prompt_template.format(
                    arxiv_id=provenance["arxiv_id"],
                    source_text=source_input.text,
                    true_statement=investigation["true_statement"],
                    false_statement=investigation["false_statement"],
                )
            else:
                evidence_packet = build_verification_evidence(
                    source_input.text, investigation, context_lines=args.context_lines
                )
                novelty_record = {
                    key: investigation.get(key)
                    for key in ("novelty_type", "importance", "refutation_status", "prior_claim", "new_result")
                }
                rendered = prompt_template.format(
                    arxiv_id=provenance["arxiv_id"],
                    question=investigation["question"],
                    answer_type=investigation["answer_type"],
                    declared_variables=json.dumps(investigation.get("declared_variables") or []),
                    basis_summary=investigation["basis_summary"],
                    novelty_record=json.dumps(novelty_record, sort_keys=True, ensure_ascii=True),
                    evidence_packet=evidence_packet,
                )
            input_sha256 = canonical_json_sha256(
                {
                    "investigation_sha256": investigation_payload_sha256(investigation),
                    "rendered_prompt_sha256": sha256_text(rendered),
                }
            )
            if not args.overwrite and investigation_verification_is_current(
                state,
                model_config_sha256=model_config_sha256,
                prompt_sha256=prompt_sha256,
                input_sha256=input_sha256,
            ):
                reused += 1
                continue
            if args.limit is not None and len(contexts) >= args.limit:
                deferred += 1
            else:
                contexts.append(
                    {
                        "paper_id": paper_id,
                        "state_path": state_path,
                        "state": state,
                        "investigation": investigation,
                        "source_text": source_input.text if args.false else None,
                        "input_sha256": input_sha256,
                        "query": [{"role": "user", "content": rendered}],
                    }
                )
        except Exception as exc:
            preflight_failed += 1
            print(f"Verification preflight failed for {paper_id}: {type(exc).__name__}: {exc}")

    total_cost = 0.0
    passed = rejected = failed = 0
    budget_stopped = False
    for batch_start in range(0, len(contexts), args.batch_size):
        if args.max_cost is not None and total_cost >= args.max_cost:
            budget_stopped = True
            break
        batch = contexts[batch_start : batch_start + args.batch_size]
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
                model = model_record(
                    model=model_name,
                    config_arg=stored_config_arg,
                    config_path=stored_config_path,
                    config_sha256=model_config_sha256,
                    raw=raw,
                    parsed=parsed,
                    cost=cost,
                )
                validation = (
                    validate_false_payload(parsed, context["source_text"], investigation=context["investigation"])
                    if args.false
                    else validate_verification(parsed, context["investigation"]["answer"])
                )
                verification = {
                    **model,
                    "investigation_sha256": investigation_payload_sha256(context["investigation"]),
                    "prompt_sha256": prompt_sha256,
                    "input_sha256": context["input_sha256"],
                    "context_lines": args.context_lines,
                }
                if args.false:
                    verification["prompt_path"] = args.prompt
                if validation.get("valid") is not True:
                    verification["status"] = "failed"
                    verification["validation_error"] = validation
                    failed += 1
                else:
                    verification.update(validation)
                    verification["status"] = "passed" if validation["passed"] else "rejected"
                    if validation["passed"]:
                        passed += 1
                    else:
                        rejected += 1
                context["state"]["verification"] = verification
                context["state"]["updated_at"] = utc_now()
                atomic_write_json(context["state_path"], context["state"])
                total_cost += float(model.get("cost", 0.0) or 0.0)
                sync_review_annotation(
                    context["state_path"].parent,
                    investigation_filename=args.annotation_filename,
                    final_filename=FALSE_FINAL_FILENAME if args.false else FINAL_ANNOTATION_FILENAME,
                )
        except Exception as exc:
            print(f"Verification batch failed after partial progress: {type(exc).__name__}: {exc}")
        for index, context in enumerate(batch):
            if index in received:
                continue
            failed += 1
            context["state"]["verification"] = {
                "status": "failed",
                "reason": "model_response_missing",
                "investigation_sha256": investigation_payload_sha256(context["investigation"]),
                "model_config_sha256": model_config_sha256,
                "prompt_sha256": prompt_sha256,
                "input_sha256": context["input_sha256"],
                "updated_at": utc_now(),
            }
            atomic_write_json(context["state_path"], context["state"])
            sync_review_annotation(
                context["state_path"].parent,
                investigation_filename=args.annotation_filename,
                final_filename=FALSE_FINAL_FILENAME if args.false else FINAL_ANNOTATION_FILENAME,
            )

    print(
        f"Question verification complete: passed {passed}, rejected {rejected}, reused {reused}, "
        f"source rejections {skipped_rejections}, oversized {oversized}, unavailable {unavailable}, "
        f"deferred {deferred}, failed {failed + preflight_failed}. "
        f"{cost_label} ${total_cost:.6f}."
    )
    if budget_stopped:
        print(f"Stopped before the next batch after reaching the " f"${args.max_cost:.2f} tracked stage budget.")
    return 0 if not budget_stopped and deferred == 0 and failed == 0 and preflight_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
