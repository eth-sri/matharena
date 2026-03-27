#!/usr/bin/env python3
import argparse

from matharena.api_client import APIClient
from matharena.arxivbench_utils import (
    extract_json,
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


def needs_annotation(annotation, overwrite=False, false_mode=False):
    if overwrite:
        return True
    if false_mode:
        if annotation.get("keep") is None:
            return True
        if annotation.get("keep") is not True:
            return False
        return not (
            annotation.get("original_statement")
            and annotation.get("perturbed_statement")
            and annotation.get("falsity_explanation")
        )
    keep = annotation.get("keep")
    if keep is None:
        return True
    if keep is True and (not annotation.get("question") or not annotation.get("answer")):
        return True
    return False

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
    parser = argparse.ArgumentParser(description="Generate questions from paper abstracts using an LLM.")
    parser.add_argument("--model-config", required=True, help="Path under ../configs/models (e.g. openai/gpt-5-mini).")
    parser.add_argument("--paper-root", default="arxivmath/paper", help="Root directory containing paper folders.")
    parser.add_argument("--prompt", default=None, help="Prompt template path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on number of papers to query.")
    parser.add_argument("--max-papers", type=int, default=None, help="Optional limit on paper ids to inspect.")
    parser.add_argument("--false", action="store_true", help="Use the false-statement pipeline.")
    parser.add_argument("--annotation-filename", default=None, help="Annotation filename to read/write.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing annotations.")
    args = parser.parse_args()

    prompt_path = args.prompt or ("arxivmath/prompts/prompt_false.md" if args.false else "arxivmath/prompts/prompt.md")
    annotation_filename = args.annotation_filename or (FALSE_ANNOTATION_FILENAME if args.false else FINAL_ANNOTATION_FILENAME)
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
        if not needs_annotation(annotation, overwrite=args.overwrite, false_mode=args.false):
            continue
        metadata = load_metadata(args.paper_root, paper_id)
        prompt = prompt_template.format(
            title=metadata.get("title") or "",
            abstract=metadata.get("abstract") or "",
        )
        queries.append([{"role": "user", "content": prompt}])
        query_paper_ids.append(paper_id)
        if args.limit and len(queries) >= args.limit:
            break

    if not queries:
        print("No papers need annotation.")
        return

    total_cost = 0.0
    kept_ids = []
    for idx, conversation, cost in client.run_queries(queries):
        conversation = normalize_conversation(conversation)
        paper_id = query_paper_ids[idx]
        response = ""
        if conversation and isinstance(conversation[-1], dict):
            response = conversation[-1].get("content", "") or ""
        parsed = extract_json(response)
        annotation = {
            "model": model_name,
            "raw": response,
            "cost": cost.get("cost", 0.0),
        }
        keep_value = None
        if isinstance(parsed, dict):
            keep_value = coerce_bool(parsed.get("keep"))
            annotation["parsed"] = parsed
            if args.false:
                if parsed.get("original_statement"):
                    annotation["original_statement"] = str(parsed["original_statement"]).strip()
                if parsed.get("perturbed_statement"):
                    annotation["perturbed_statement"] = str(parsed["perturbed_statement"]).strip()
                if parsed.get("why_false_given_original"):
                    annotation["falsity_explanation"] = str(parsed["why_false_given_original"]).strip()
            else:
                if parsed.get("question"):
                    annotation["question"] = str(parsed["question"]).strip()
                if parsed.get("answer"):
                    annotation["answer"] = str(parsed["answer"]).strip()
        if keep_value is not None:
            annotation["keep"] = keep_value
            if keep_value is True:
                kept_ids.append(paper_id)
        save_annotation(args.paper_root, paper_id, annotation, annotation_filename)
        total_cost += annotation["cost"]

    print(f"Completed {len(queries)} queries. Total cost: ${total_cost:.6f}")
    print(f"Kept {len(kept_ids)} papers: {', '.join(kept_ids)}")


if __name__ == "__main__":
    main()
