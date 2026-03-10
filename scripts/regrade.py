import argparse
import json
import os

import yaml
from loguru import logger

from matharena.grader import extract_and_grade
from matharena.runner import Runner
from matharena.runs import Runs
from matharena.utils import is_conversation_broken, normalize_conversation

"""
    Loads all .json runs files for a given competition across all models and recomputes some things.
    By default (use flags to disable) it does:
       - Load an existing runs .json file and checks if there are N runs properly recorded
          - This might create fields like judgement or history as empty so they are always present for consistency
       - Fixes common off-by-ones where idx field does not match the file name 
       - Clean the messages using utils.clean_conversation (recomputing: messages) -> warnings where mismatch
       - Rerun the grader (recomputing: answer, is_correct, warning, pass@1) -> warnings where mismatch
       - Rerun the cost computation e.g. if API costs were wrong (recomputing: detailed_cost, cost) -> warnings where mismatch
       - Save back the .json file
         - This might change ordering and formatting and names to new canonical sort/names
"""

parser = argparse.ArgumentParser()
parser.add_argument("--comps", type=str, nargs="+", required=True)
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--no-update-idx", action="store_true", help="Do not fix off-by-one idx")
parser.add_argument("--no-clean-messages", action="store_true", help="Do not clean messages")
parser.add_argument("--no-rerun-grader", action="store_true", help="Do not rerun the grader")
parser.add_argument("--no-rerun-cost", action="store_true", help="Do not recompute costs")
parser.add_argument("--models", type=str, nargs="+", default=None, help="If set, only process these models")

# Ok to keep defaults here
parser.add_argument("--comp-configs-dir", type=str, default="configs/competitions")
parser.add_argument("--model-configs-dir", type=str, default="configs/models")
parser.add_argument("--output-dir", type=str, default="outputs")
args = parser.parse_args()

missing_runs = []

for comp in args.comps:

    logger.info(f"Initializing runner for competition {comp}")
    runner = Runner(comp, args.n, None, args.comp_configs_dir, None, args.output_dir, False)
    for api in os.listdir(f"{args.output_dir}/{comp}"):
        for solver in os.listdir(f"{args.output_dir}/{comp}/{api}"):
            if args.models is not None and f"{api}/{solver}" not in args.models:
                continue

            solver_config_path = f"{args.model_configs_dir}/{api}/{solver}.yaml"
            with open(solver_config_path, "r") as f:
                solver_config = yaml.safe_load(f)
            solver_type = solver_config.get("type", "pure_model")

            for problem in runner.problems:
                logger.info(f"\n--> Processing {solver} @ {comp} P{problem['problem_idx']}")
                output_dir = f"{args.output_dir}/{comp}/{api}/{solver}"
                runs = Runs(comp, runner.is_fa_comp, solver, solver_type, problem, output_dir)
                runs.load_from_file()
                if runs.N != args.n:
                    logger.warning(f"Expected {args.n} runs but found {runs.N}")
                    missing_runs.append((comp, f"{api}/{solver}", problem["problem_idx"], runs.N))

                # Fix IDs
                if not args.no_update_idx:
                    # logger.info("Fixing IDs")
                    if runs.problem_idx + 1 == problem["problem_idx"]:
                        # logger.info(f"[{solver} @ {comp} P{problem['problem_idx']}] Fixing off-by-one idx")
                        runs.problem_idx = problem["problem_idx"]
                    elif runs.problem_idx != problem["problem_idx"]:
                        logger.warning(
                            f"[{solver} @ {comp} P{problem['problem_idx']}] Cannot fix idx mismatch: {runs.problem_idx} vs {problem['problem_idx']}"
                        )

                # Clean messages
                if not args.no_clean_messages:
                    # logger.info("Normalizing messages")
                    for i in range(runs.N):
                        if runs.messages[i] is not None:
                            clean_conversation = normalize_conversation(runs.messages[i])
                            runs.messages[i] = clean_conversation
                            # Check if broken?
                            is_broken, reason = is_conversation_broken(runs.messages[i])
                            if is_broken:
                                raise ValueError(f"Message list is broken: {reason}")

                    # logger.info("Normalizing messages in history")
                    for i in range(runs.N):
                        history = runs.history[i]
                        if history is not None:
                            for step in history:
                                step["messages"] = normalize_conversation(step["messages"])

                # Rerun grading
                if not args.no_rerun_grader:
                    if runner.is_fa_comp:
                        # logger.info("Rerunning grading")
                        for i in range(runs.N):
                            clean_conversation = runs.messages[i]
                            output_tokens = runs.detailed_costs[i]["output_tokens"]
                            gold_answer = str(problem["answer"])
                            comp_config = runner.competition_config
                            debug_info = f"{solver} @ {comp} P{problem['problem_idx']} run {i}"
                            grader_response = extract_and_grade(
                                clean_conversation, output_tokens, gold_answer, comp_config, debug_info=debug_info
                            )
                            runs.update_run_grading(i, grader_response)
                    else:
                        logger.info("Skipping rerunning grading since not final answer competition")

                # Rerun cost
                if not args.no_rerun_cost:
                    # logger.info("Rerunning cost computation")
                    solver_cfg = runner.load_solver_config(solver_config_path)
                    model_config = solver_cfg.get("model_config", {})
                    in_cost = model_config["read_cost"]
                    cache_in_cost = model_config.get("cache_read_cost", in_cost)
                    out_cost = model_config["write_cost"]
                    cache_write_cost = model_config.get("cache_write_cost", out_cost)
                    for i in range(runs.N):
                        runs.update_run_costs(i, in_cost, out_cost, cache_in_cost, cache_write_cost)

                # Save
                logger.info("Saving back to file")
                runs.save_to_file()

print("MISSING RUNS REPORT:")
for comp, solver, pidx, N in missing_runs:
    print(f" - {comp} {solver} P{pidx} has only {N} runs")
