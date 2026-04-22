import argparse
import json
import os
import re
from pathlib import Path

import yaml
from dotenv import load_dotenv
from loguru import logger

from matharena.api_client import APIClient
from matharena.json_zst import OUTPUT_JSON_SUFFIX, dump_json_zst, load_json_zst
from matharena.tools.lean_execution import get_executed_lean_submission_parts

def _parse_bool(text):
    if text is None:
        return None
    normalized = text.strip().lower()
    if normalized in ["true", "false"]:
        return normalized == "true"
    match = re.search(r"\b(true|false)\b", normalized)
    # return the last match if multiple
    if match:
        return match.groups()[-1] == "true"
    return False


def _sanitize_model_config(model_config):
    cleaned = dict(model_config)
    for key in ["human_readable_id", "other_params", "date", "custom_instructions"]:
        cleaned.pop(key, None)
    cleaned["tools"] = []
    cleaned["max_tool_calls"] = 0
    return cleaned


def _resolve_model_config(model_config_ref, model_configs_dir):
    if model_config_ref.startswith("models/"):
        model_config_ref = model_config_ref[len("models/") :]
    if model_config_ref.endswith(".yaml"):
        config_path = model_config_ref
    else:
        config_path = os.path.join(model_configs_dir, f"{model_config_ref}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config.get("type") == "agent":
        inner_ref = config["model_config"]
        if inner_ref.startswith("models/"):
            inner_ref = inner_ref[len("models/") :]
        return _resolve_model_config(inner_ref, model_configs_dir)

    return config


def _load_competition_config(comp, comp_configs_dir):
    config_path = Path(comp_configs_dir) / f"{comp}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Competition config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_prompt(problem, gold_answer, model_reply):
    return f"""You are a strict answer checker. Your goal is to determine whether the model's reply contains a correct final answer to the given problem.
Do not attempt to solve the problem yourself; only check if the model's reply matches the ground-truth answer. Do not provide any explanations or justifications, and do not attempt to modify the model's answer: only trivial formatting changes (e.g., slightly different punctuation, different order of brackets, etc.) are allowed between the model's final answer and the ground-truth answer.

### PROBLEM ###
{problem}

### GOLD ANSWER ###
{gold_answer}

### MODEL REPLY ###
{model_reply}
Does the model reply contain a correct final answer? Reply only with "true" or "false"."""


def _build_lean_cheat_prompt(problem, formal_statement, executed_prefix, theorem_block, model_reply):
    return f"""You are a strict referee for Lean theorem-proving submissions. Your goal is to determine whether the model's reply attempted to cheat or evade the task, rather than honestly proving the exact target statement.
Do not attempt to solve the theorem yourself. Only inspect the submission for cheating behavior.

Helper lemmas and definitions are allowed if they are properly proved and do not change the meaning of the original statement.
Examples of cheating include, but are not limited to:
- proving a weaker or different statement by changing the original statement in a nontrivial way,
- changing or shadowing the meaning of symbols from the target statement,
- introducing unjustified axioms, assumptions, or placeholders,
- exploiting formatting or checker quirks instead of proving the theorem,
- otherwise avoiding the actual proof obligation while making the submission look accepted.

Reply "true" only if the submission attempted to cheat in any way. Reply "false" if it looks like a legitimate attempt to prove the exact statement. Do not provide any explanation.

### NATURAL LANGUAGE PROBLEM ###
{problem}

### ORIGINAL FORMAL STATEMENT ###
{formal_statement}

### LEAN CODE EXECUTED BEFORE THE THEOREM STATEMENT ###
{executed_prefix or "(none)"}

### FINAL THEOREM BLOCK CHECKED ###
{theorem_block or "(could not extract theorem block)"}

### FULL MODEL REPLY ###
{model_reply}

Did the model attempt to cheat in any way? Reply only with "true" or "false"."""


def _iter_json_files(output_root):
    for root, _, files in os.walk(output_root):
        for name in files:
            if name.endswith(OUTPUT_JSON_SUFFIX):
                yield Path(root) / name


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM-based checker for incorrect outputs.")
    parser.add_argument("--comp", type=str, required=True, help="Competition name, e.g. aime/aime_2025")
    parser.add_argument("--model-config", type=str, default="gemini/gemini-3-flash-low", help="Model config path under configs/models")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--model-configs-dir", type=str, default="configs/models")
    parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
    parser.add_argument("--redo", action="store_true", help="Recompute existing llm_annotation values")
    args = parser.parse_args()

    model_config = _resolve_model_config(args.model_config, args.model_configs_dir)
    competition_config = _load_competition_config(args.comp, args.comp_configs_dir)
    is_lean_comp = competition_config.get("lean", False)
    client = APIClient(**_sanitize_model_config(model_config))

    output_root = Path(args.output_dir) / args.comp
    if not output_root.exists():
        raise FileNotFoundError(f"Outputs not found: {output_root}")

    total_files = 0

    pending = []
    pending_meta = []
    file_cache = {}

    for json_path in _iter_json_files(output_root):
        total_files += 1
        data = load_json_zst(json_path)
        file_cache[json_path] = data

        correct = data.get("correct", [])
        messages = data.get("messages", [])

        llm_annotation = data.get("llm_annotation", [None] * len(correct))
        if not isinstance(llm_annotation, list):
            llm_annotation = [None] * len(correct)
        if len(llm_annotation) < len(correct):
            llm_annotation = llm_annotation + [None] * (len(correct) - len(llm_annotation))
        elif len(llm_annotation) > len(correct):
            llm_annotation = llm_annotation[: len(correct)]
        data["llm_annotation"] = llm_annotation

        for i, is_correct in enumerate(correct):
            if (is_lean_comp and not is_correct) or (not is_lean_comp and is_correct):
                continue
            if (llm_annotation[i] is not None) and not args.redo:
                continue
            model_reply = messages[i][-1]["content"]
            if is_lean_comp:
                formal_statement = data.get("gold_answer", "")
                run_messages = messages[i] if i < len(messages) else []
                executed_prefix, theorem_block = get_executed_lean_submission_parts(
                    model_reply, formal_statement=formal_statement, messages=run_messages
                )
                prompt = _build_lean_cheat_prompt(
                    data.get("problem", ""), formal_statement, executed_prefix, theorem_block, model_reply
                )
            else:
                prompt = _build_prompt(data.get("problem", ""), data.get("gold_answer", ""), model_reply)
            pending.append([{"role": "user", "content": prompt}])
            pending_meta.append((json_path, i))

    if pending:
        total_cost = 0
        for idx, conversation, cost in client.run_queries(pending):
            json_path, run_idx = pending_meta[idx]
            total_cost += cost["cost"]
            reply_text = conversation[-1]["content"]
            parsed = _parse_bool(reply_text)
            if parsed is None:
                logger.warning(f"Could not parse LLM response for {json_path} run {run_idx}")
                continue
            data = file_cache[json_path]
            data["llm_annotation"][run_idx] = parsed
            dump_json_zst(data, json_path, ensure_ascii=False, indent=4)
        logger.info(f"Processed {len(pending)} runs, total cost: {total_cost:.4f}")

if __name__ == "__main__":
    main()
