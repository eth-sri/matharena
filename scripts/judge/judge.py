import argparse
import json
import os
from pathlib import Path

import yaml
from datasets import load_dataset
from dotenv import load_dotenv
from loguru import logger

from matharena.json_zst import OUTPUT_JSON_SUFFIX, dump_json_zst, load_json_zst, output_json_stem
from matharena.solvers.judges import JudgePool
from matharena.tools.code_execution import execute_code


def load_original_statement(dataset_path, problem_idx):
    path = Path(dataset_path) / "original" / f"{problem_idx}.tex"
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def load_grading_map(dataset_path):
    grading_path = Path(dataset_path) / "grading_scheme.json"
    if os.path.exists(dataset_path) and grading_path.exists():
        with open(grading_path, "r", encoding="utf-8") as f:
            return {int(row["id"]): row for row in json.load(f)}

    rows = load_dataset(dataset_path, split="train").to_list()
    grading_map = {}
    for row in rows:
        problem_idx = int(row["problem_idx"])
        grading_map[problem_idx] = {
            "id": problem_idx,
            "points": row.get("points", 7),
            "grading_scheme": row.get("grading_scheme"),
            "sample_solution": row.get("sample_solution"),
            "sample_grading": row.get("sample_grading"),
            "ground_truth_proofs": row.get("ground_truth_proofs", []),
            "ground_truth_solutions": row.get("ground_truth_solutions", []),
        }
    return grading_map


def ensure_judgment_shape(run_data, slot):
    n = len(run_data.get("messages", []))
    judgment = run_data.get("judgment", [])
    if not isinstance(judgment, list):
        judgment = []
    elif judgment and all((item is None or isinstance(item, dict)) for item in judgment):
        judgment = [judgment]

    while len(judgment) <= slot:
        judgment.append([None] * n)
    for i in range(len(judgment)):
        if not isinstance(judgment[i], list):
            judgment[i] = [None] * n
        elif len(judgment[i]) < n:
            judgment[i].extend([None] * (n - len(judgment[i])))
        elif len(judgment[i]) > n:
            judgment[i] = judgment[i][:n]

    run_data["judgment"] = judgment


def find_judgment_slot_for_judge(run_data, run_idx, judge_id):
    for slot_idx, slot_data in enumerate(run_data.get("judgment", [])):
        if not isinstance(slot_data, list) or run_idx >= len(slot_data):
            continue
        entry = slot_data[run_idx]
        if isinstance(entry, dict) and entry.get("judge_id") == judge_id:
            return slot_idx
    return None


def build_judgment_entry(judge_response, max_points, scheme_text, judge_id, judge_points_max):
    if judge_response.points is None:
        return None

    raw_points = judge_response.points
    points = raw_points if abs(max_points - judge_points_max) < 1e-9 else raw_points * max_points / judge_points_max

    return {
        "points": points,
        "max_points": max_points,
        "judge_id": judge_id,
        "cost": judge_response.detailed_cost,
        "additional_info": judge_response.additional_info,
        "details": [
            {
                    "title": "Overall",
                    "points": points,
                    "max_points": max_points,
                    "grading_scheme_desc": scheme_text,
                    "desc": judge_response.explanation,
                }
            ],
        }


def recompute_correct_and_pass_at_1(run_data):
    n = len(run_data["messages"])
    correct = []
    for i in range(n):
        vals = []
        for judge in run_data.get("judgment", []):
            if not isinstance(judge, list) or i >= len(judge) or not isinstance(judge[i], dict):
                continue
            points = judge[i].get("points")
            max_points = judge[i].get("max_points")
            if isinstance(points, (int, float)) and isinstance(max_points, (int, float)) and max_points > 0:
                vals.append(points / max_points)
        correct.append(sum(vals) / len(vals) if vals else "TODO Grading")

    run_data["correct"] = correct
    if n == 0:
        run_data["pass_at_1"] = None
    elif all(isinstance(x, (int, float)) for x in correct):
        run_data["pass_at_1"] = sum(correct) / n
    else:
        run_data["pass_at_1"] = "TODO Grading"


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description="LLM judge for non-final-answer competitions")
    parser.add_argument("--comp", type=str, required=True)
    parser.add_argument("--judge-configs", type=str, nargs="+", default=None)
    parser.add_argument("--problem-ids", type=int, nargs="+", default=None)
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        help="Filter by model path, e.g. gemini/gemini-31-pro openai/gpt-54")
    parser.add_argument("--redo", action="store_true")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
    parser.add_argument("--model-configs-dir", type=str, default="configs/models")
    parser.add_argument("--configs-dir", type=str, default="configs")
    args = parser.parse_args()

    with open(f"{args.comp_configs_dir}/{args.comp}.yaml", "r", encoding="utf-8") as f:
        comp_cfg = yaml.safe_load(f)

    judge_refs = comp_cfg["judge_configs"]

    dataset_path = comp_cfg.get("dataset_path")

    grading_map = load_grading_map(dataset_path)

    output_root = Path(args.output_dir) / args.comp

    runs_cache = {}
    for path in sorted(output_root.glob(f"**/*{OUTPUT_JSON_SUFFIX}"), key=output_json_stem):
        runs_cache[path] = load_json_zst(path)

    problem_filter = set(args.problem_ids) if args.problem_ids else None

    for slot, judge_ref in enumerate(judge_refs):
        with open(f"{args.configs_dir}/{judge_ref}.yaml", "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)
        judge_points_max = full_config.get("judge_points_max", 7)

        with open(f"{args.configs_dir}/{full_config['scaffold_config']}.yaml", "r", encoding="utf-8") as f:
            judge_cfg = yaml.safe_load(f)

        with open(f"{args.model_configs_dir}/{full_config['model_config']}.yaml", "r", encoding="utf-8") as f:
            model_cfg = yaml.safe_load(f)
        
        judge_cfg["model_config"] = model_cfg

        for key in full_config.get("override", {}):
            judge_cfg[key] = full_config["override"][key]

        judge_pool = JudgePool(
            solver_config=judge_cfg
        )

        queries = []
        meta = []

        for path, run_data in runs_cache.items():
            model_path = str(path.relative_to(output_root).parent)
            if args.models and model_path not in args.models:
                continue
            problem_idx = int(run_data.get("idx", output_json_stem(path)))
            if (problem_filter and problem_idx not in problem_filter):
                continue

            record = grading_map.get(problem_idx)

            ensure_judgment_shape(run_data, slot)

            scheme = record.get("scheme", record.get("grading_scheme"))
            max_points = float(record.get("points", 7))
            if isinstance(scheme, list):
                scheme_text = "\n".join(
                            f"{i}. [{item.get('points', 0)} pts] {item.get('title')}: {item.get('desc')}"
                            for i, item in enumerate(scheme, start=1)
                    )
            else:
                scheme_text = str(scheme)

            proofs = record.get("ground_truth_proofs", [])
            original_problem_statement = load_original_statement(dataset_path, problem_idx)

            problem_text = run_data.get("problem")
            for run_idx, conversation in enumerate(run_data.get("messages", [])):
                existing_slot = find_judgment_slot_for_judge(run_data, run_idx, judge_ref)
                if existing_slot is not None and not args.redo:
                    continue

                write_slot = existing_slot if existing_slot is not None else slot
                student_answer = conversation[-1]["content"]

                queries.append(
                    (
                        problem_text,
                        {
                            "guidelines": scheme_text,
                            "ground_truth_solutions": proofs,
                            "original_problem_statement": original_problem_statement,
                            "student_answer": student_answer,
                        },
                    )
                )
                meta.append((path, run_idx, write_slot, max_points, scheme_text, judge_ref, judge_points_max))

        if len(queries) == 0:
            logger.info(f"No pending runs for judge slot {slot + 1} ({judge_ref}).")
            continue

        logger.info(f"Running judge slot {slot + 1}/{len(judge_refs)} ({judge_ref}) on {len(queries)} runs.")
        touched = set()
        total_cost = 0.0
        cost_by_evaluated_model = {}
        cost_by_judge_model = {}

        batch_idx_to_problem_idx = {i: i for i in range(len(queries))}
        batch_idx_to_run_idx = {i: i for i in range(len(queries))}
        for judge_response in judge_pool.solve_batch(queries, batch_idx_to_problem_idx, batch_idx_to_run_idx):
            path, run_idx, write_slot, max_points, scheme_text, judge_id, judge_points_max = meta[judge_response.idx]
            run_data = runs_cache[path]
            ensure_judgment_shape(run_data, write_slot)
            parsed = build_judgment_entry(judge_response, max_points, scheme_text, judge_id, judge_points_max)
            if parsed is None:
                logger.warning(f"Failed to parse judge output for {path} run {run_idx} slot {write_slot}. Skipping. You should rerun the command to retry this run.")
                continue
            run_data["judgment"][write_slot][run_idx] = parsed
            recompute_correct_and_pass_at_1(run_data)
            dump_json_zst(run_data, path, indent=4, ensure_ascii=False)
            touched.add(path)
            cost = float(judge_response.detailed_cost.get("cost", 0.0))
            total_cost += cost
            model_path = str(path.relative_to(output_root).parent)
            problem_idx = int(run_data.get("idx", output_json_stem(path)))
            cost_by_evaluated_model[model_path] = cost_by_evaluated_model.get(model_path, 0.0) + cost
            model_problem_costs = cost_by_evaluated_model.setdefault(f"{model_path}__problems", {})
            model_problem_costs[problem_idx] = model_problem_costs.get(problem_idx, 0.0) + cost
            for jm, jc in judge_response.detailed_cost.get("cost_by_judge_model", {}).items():
                cost_by_judge_model[jm] = cost_by_judge_model.get(jm, 0.0) + jc

        judge_breakdown = ", ".join(f"{m}: ${c:.4f}" for m, c in sorted(cost_by_judge_model.items()))
        logger.info(
            f"Finished judge slot {slot + 1} ({judge_ref}). Updated {len(touched)} files. "
            f"Total cost: ${total_cost:.4f}"
        )
        eval_models = sorted(k for k in cost_by_evaluated_model if not k.endswith("__problems"))
        for m in eval_models:
            problems = cost_by_evaluated_model.get(f"{m}__problems", {})
            problem_str = ", ".join(f"P{p}: ${c:.4f}" for p, c in sorted(problems.items()))
            logger.info(f"  {m}: ${cost_by_evaluated_model[m]:.4f} ({problem_str})")
        if judge_breakdown:
            logger.info(f"  Cost by judge model: {judge_breakdown}")


if __name__ == "__main__":
    main()
