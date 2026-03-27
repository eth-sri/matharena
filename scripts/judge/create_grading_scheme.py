import argparse
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from matharena.api_client import APIClient
from matharena.utils import normalize_conversation

def _sanitize_model_config(model_config, remove_keys):
    cleaned = dict(model_config)
    for key in remove_keys:
        cleaned.pop(key, None)
    return cleaned

def _extract_solution_text(grading_item):
    chunks = []
    for idx, solution in enumerate(grading_item["ground_truth_proofs"], start=1):
        chunks.append(f"### Solution {idx} ###\n{solution}")
    return "\n\n".join(chunks)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--comp", type=str, required=True, help="Competition config path, e.g. usamo/usamo_2025")
    parser.add_argument("--model-configs-dir", type=str, default="configs/models")
    parser.add_argument("--judge-configs-dir", type=str, default="configs/judges")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--redo", action="store_true")
    args = parser.parse_args()

    data_comp_dir = Path(args.data_dir) / args.comp
    grading_scheme_path = data_comp_dir / "grading_scheme.json"

    comp_config = yaml.safe_load((Path("configs/competitions") / f"{args.comp}.yaml").read_text(encoding="utf-8"))

    grading_data = json.loads(grading_scheme_path.read_text(encoding="utf-8"))

    with open(os.path.join(args.judge_configs_dir, comp_config["grading_scheme_creator_config"] + ".yaml"), "r") as f:
        creator_cfg = yaml.safe_load(f)

    with open(os.path.join(args.model_configs_dir, creator_cfg["model_config"] + ".yaml"), "r") as f:
        model_config = yaml.safe_load(f)
    client_args = _sanitize_model_config(
        model_config,
        creator_cfg["api_client_remove_keys"]
    )
    client_args["tools"] = creator_cfg.get("enabled_tools", [])
    client_args["max_tool_calls"] = creator_cfg.get("max_tool_calls", 0)
    client = APIClient(**client_args)

    pending_queries = []
    pending_meta = []

    for item in grading_data:
        problem_idx = int(item["id"])
        if not args.redo and isinstance(item.get("scheme"), str):
            continue

        local_problem_path = data_comp_dir / "problems" / f"{problem_idx}.tex"
        with open(local_problem_path, "r", encoding="utf-8") as f:
            problem_text = f.read()

        solution_text = _extract_solution_text(item)

        prompt = creator_cfg["prompt"].format(problem=problem_text, solution=solution_text)
        pending_queries.append([{"role": "user", "content": prompt}])
        pending_meta.append(item)

    if len(pending_queries) == 0:
        logger.info("No grading schemes to generate.")
        return

    logger.info(f"Generating {len(pending_queries)} grading schemes for {args.comp}.")
    total_cost = 0.0
    for idx, conversation, detailed_cost in client.run_queries(pending_queries):
        conversation = normalize_conversation(conversation)
        generated_scheme = conversation[-1]["content"] if len(conversation) > 0 else ""
        pending_meta[idx]["scheme"] = generated_scheme
        total_cost += float(detailed_cost.get("cost", 0.0))

    grading_scheme_path.write_text(json.dumps(grading_data, ensure_ascii=False, indent=4), encoding="utf-8")
    logger.info(
        f"Updated {grading_scheme_path} with {len(pending_queries)} string-based grading schemes. "
        f"Total generation cost: ${total_cost:.4f}"
    )


if __name__ == "__main__":
    main()
