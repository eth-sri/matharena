#!/usr/bin/env python3
import argparse
from datetime import datetime

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    extract_json,
    get_latest_fields,
    get_latest_pair,
    list_paper_ids,
    load_annotation,
    load_metadata,
    load_model_config,
    load_prompt_template,
    resolve_model_config_path,
    save_annotation,
)
from matharena.utils import normalize_conversation


FINAL_ANNOTATION_FILENAME = "llm_annotation.json"
FALSE_ANNOTATION_FILENAME = "llm_metadata_false.json"
LEAN_ANNOTATION_FILENAME = "metadata_lean_abstract.json"
LEAN_DEFAULT_PROMPT = "arxivmath/prompts/lean/verify_lean_abstract.md"


def default_verification_key(semantic_judge=False):
    return "semantic_verification" if semantic_judge else "verification"


def needs_verification(annotation, overwrite=False, false_mode=False, lean_mode=False, semantic_judge=False, key=None):
    if annotation.get("keep") is not True:
        return False
    if semantic_judge:
        review = annotation.get("review") or {}
        if not (review.get("statement") or annotation.get("statement")):
            return False
        if not (review.get("formalized_statement") or annotation.get("formalized_statement")):
            return False
    elif lean_mode:
        if not annotation.get("statement"):
            return False
    elif false_mode:
        if not annotation.get("original_statement") or not annotation.get("perturbed_statement"):
            return False
    else:
        if not annotation.get("question") or not annotation.get("answer"):
            return False
    if not (semantic_judge or lean_mode) and "review" in annotation and (annotation.get("review") or {}).get("status") != "keep":
        return False
    verification = annotation.get(key or default_verification_key(semantic_judge=semantic_judge))
    if overwrite:
        return True
    if isinstance(verification, dict) and "keep" in verification:
        return False
    return True


def render_prompt(template, annotation, false_mode=False, lean_mode=False, semantic_judge=False):
    if semantic_judge:
        natural_statement, lean_code = get_latest_fields(annotation, ["statement", "formalized_statement"]) or ("", "")
        return template.format(natural_statement=natural_statement, lean_code=lean_code)
    if lean_mode:
        metadata = annotation.get("_metadata") or {}
        return template.format(
            title=(metadata.get("title") or "").strip(),
            abstract=(metadata.get("abstract") or "").strip(),
            statement=(annotation.get("statement") or "").strip(),
        )
    if false_mode:
        original_statement, perturbed_statement, falsity_explanation = get_latest_fields(
            annotation,
            ["original_statement", "perturbed_statement", "falsity_explanation"],
        ) or ("", "", "")
        return template.format(
            original_statement=original_statement,
            perturbed_statement=perturbed_statement,
            falsity_explanation=falsity_explanation,
        )
    question, answer = get_latest_pair(annotation) or ("", "")
    return template.format(question=question, answer=answer)


def coerce_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes"}:
            return True
        if lowered in {"false", "no"}:
            return False
    return None


def main():
    parser = argparse.ArgumentParser(description="Verify kept LLM annotations against the criteria.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to verify.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--false", action="store_true", help="Use the false-statement pipeline.")
    parser.add_argument("--lean", action="store_true", help="Use the Lean-candidate pipeline.")
    parser.add_argument("--semantic-judge", action="store_true", help="Judge whether Lean code faithfully formalizes the natural-language statement.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing verification results.")
    parser.add_argument("--key", default=None, help="Annotation key to store the verification under.")
    args = parser.parse_args()

    prompt_path = args.prompt or (
        "arxivmath/prompts/lean/semantic_judge.md"
        if args.semantic_judge
        else LEAN_DEFAULT_PROMPT
        if args.lean
        else "arxivmath/prompts/broken/verify_false.md"
        if args.false
        else "arxivmath/prompts/arxiv/verify.md"
    )
    annotation_filename = args.annotation_filename or (
        LEAN_ANNOTATION_FILENAME
        if args.semantic_judge or args.lean
        else FALSE_ANNOTATION_FILENAME
        if args.false
        else FINAL_ANNOTATION_FILENAME
    )
    verification_key = args.key or default_verification_key(semantic_judge=args.semantic_judge)
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    client = APIClient(**model_config)

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[:args.max_papers]
    queries = []
    query_paper_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if not needs_verification(
            annotation,
            overwrite=args.overwrite,
            false_mode=args.false,
            lean_mode=args.lean,
            semantic_judge=args.semantic_judge,
            key=verification_key,
        ):
            continue
        if args.lean:
            annotation["_metadata"] = load_metadata(args.paper_root, paper_id)
        prompt = render_prompt(
            prompt_template,
            annotation,
            false_mode=args.false,
            lean_mode=args.lean,
            semantic_judge=args.semantic_judge,
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        return

    total_cost = 0.0
    kept_ids = []
    rejected_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        paper_id = query_paper_ids[idx]
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        verification = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        keep_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            verification["parsed"] = parsed
            if args.lean and keep_value is not None:
                verification["keep"] = keep_value
        if keep_value is not None:
            if "keep_original" not in annotation:
                annotation["keep_original"] = annotation.get("keep")
            annotation["keep"] = keep_value
            if not args.lean:
                verification["keep"] = keep_value
            if keep_value:
                kept_ids.append(paper_id)
            else:
                rejected_ids.append(paper_id)
        annotation[verification_key] = verification
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += verification["cost"]

    print(f"Completed {len(queries)} queries. Total cost: ${total_cost:.6f}")
    print(f"Verified keep: {len(kept_ids)} papers: {', '.join(kept_ids)}")
    print(f"Verified reject: {len(rejected_ids)} papers: {', '.join(rejected_ids)}")


if __name__ == "__main__":
    main()
