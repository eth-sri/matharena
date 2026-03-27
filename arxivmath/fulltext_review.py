#!/usr/bin/env python3
import argparse
from datetime import datetime

from tqdm import tqdm

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    ensure_ocr,
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


def should_review(annotation, overwrite=False, key="full_text_review"):
    if annotation.get("keep") is not True:
        return False
    if not overwrite and key in annotation:
        return False
    review = annotation.get("review") or {}
    if review and review.get("status") != "keep":
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Re-check kept arXiv questions against full paper OCR.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to process.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--redo-ocr", action="store_true", help="Force OCR even if cached markdown exists.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing full-text review results.")
    parser.add_argument("--key", default="full_text_review", help="Annotation key to store the review under.")
    parser.add_argument("--enable-web-search", action="store_true", help="Enable web search for additional context.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip OCR and full text injection.")
    parser.add_argument("--false", action="store_true", help="Use the false-statement pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    args = parser.parse_args()

    prompt_path = args.prompt or ("arxivmath/prompts/prompt_false_fulltext_review.md" if args.false else "arxivmath/prompts/prompt_fulltext_review.md")
    annotation_filename = args.annotation_filename or (FALSE_ANNOTATION_FILENAME if args.false else FINAL_ANNOTATION_FILENAME)
    prompt_template = load_prompt_template(prompt_path)
    model_config_path = resolve_model_config_path(args.model_config)
    model_config = load_model_config(model_config_path)
    model_name = model_config["model"]
    if args.enable_web_search:
        if model_config.get("api") == "google":
            model_config["tools"] = [(None, {"google_search": {}})]
            model_config["use_gdm_tools"] = True
            model_config["max_tool_calls"] = 50
        else:
            model_config["tools"] = [(None, {"type": "web_search"})]
    client = APIClient(**model_config)

    discarded = []
    updated = []
    kept = []
    total_cost = 0.0

    paper_ids = list_paper_ids(args.paper_root)
    if args.max_papers:
        paper_ids = paper_ids[:args.max_papers]
    review_ids = []
    for paper_id in paper_ids:
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        if should_review(annotation, overwrite=args.overwrite, key=args.key):
            review_ids.append(paper_id)

    queries = []
    query_paper_ids = []
    for paper_id in tqdm(review_ids):
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        question = answer = original_statement = perturbed_statement = falsity_explanation = ""
        if args.key != "solid_authors":
            if args.false:
                latest = get_latest_fields(
                    annotation,
                    ["original_statement", "perturbed_statement", "falsity_explanation"],
                )
                if not latest:
                    continue
                original_statement, perturbed_statement, falsity_explanation = latest
            else:
                latest = get_latest_pair(annotation)
                if not latest:
                    continue
                question, answer = latest
        full_text = "" if args.skip_ocr or args.key == "solid_authors" else ensure_ocr(paper_id, redo=args.redo_ocr)
        metadata = load_metadata(args.paper_root, paper_id)
        prompt = prompt_template.format(
            question=question,
            answer=answer,
            original_statement=original_statement,
            perturbed_statement=perturbed_statement,
            falsity_explanation=falsity_explanation,
            full_text=full_text or "",
            title=metadata.get("title") or "",
            authors=", ".join([f"{author['forenames']} {author['keyname']}" for author in metadata.get("authors", [])]),
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)

        if args.limit and len(queries) >= args.limit:
            break
    if not queries:
        print("No papers need review.")
        return

    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        if idx >= len(query_paper_ids):
            continue
        paper_id = query_paper_ids[idx]
        annotation = load_annotation(args.paper_root, paper_id, annotation_filename)
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        action = None
        keep_value = None
        if isinstance(parsed, dict):
            action = parsed.get("action")
            edited_question = parsed.get("question")
            edited_original = parsed.get("original_statement")
            edited_perturbed = parsed.get("perturbed_statement")
            edited_falsity = parsed.get("falsity_explanation")
            keep_value = parsed.get("keep", True)

        review_record = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        }
        if isinstance(parsed, dict):
            review_record["parsed"] = parsed
            if "rationale" in parsed:
                review_record["rationale"] = parsed.get("rationale")
        if action:
            review_record["action"] = action

        review = annotation.get("review") or {}
        if action == "discard" or (action is None and '"action": "discard"' in response) or keep_value is False:
            review["status"] = "discard"
            review["updated_at"] = review_record["updated_at"]
            annotation["keep"] = False
            discarded.append(paper_id)
        elif action == "edit":
            if args.false:
                for field, value in [
                    ("original_statement", edited_original),
                    ("perturbed_statement", edited_perturbed),
                    ("falsity_explanation", edited_falsity),
                ]:
                    if value and str(value).strip():
                        review[field] = str(value).strip()
                        annotation[field] = review[field]
                review["updated_at"] = review_record["updated_at"]
                review["status"] = "keep"
                annotation["keep"] = True
                updated.append(paper_id)
            elif edited_question and str(edited_question).strip():
                review["question"] = str(edited_question).strip()
                annotation["question"] = review["question"]
                review["updated_at"] = review_record["updated_at"]
                review["status"] = "keep"
                annotation["keep"] = True
                updated.append(paper_id)
            else:
                kept.append(paper_id)
        else:
            annotation["keep"] = True
            kept.append(paper_id)

        annotation["review"] = review
        annotation[args.key] = review_record
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += review_record["cost"]

    print(f"Full-text review complete. Total cost: ${total_cost:.6f}")
    print(f"Discarded ({len(discarded)}): {', '.join(discarded)}")
    print(f"Updated ({len(updated)}): {', '.join(updated)}")
    print(f"Kept ({len(kept)}): {', '.join(kept)}")


if __name__ == "__main__":
    main()
