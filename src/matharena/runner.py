"""This module contains the main runner for conducting experiments."""

import base64
import csv
import json
import os
from datetime import datetime

import yaml
from datasets import load_dataset
from loguru import logger

from matharena.grader import extract_and_grade
from matharena.parser import extract_answer
from matharena.request_logger import request_logger
from matharena.runs import Runs
from matharena.solvers import AgentPool, PureModelSolver
from matharena.tools.code_execution import execute_code
from matharena.tools.paper_search import read_paper, query_semantic_scholar, read_pages, find_in_paper
from matharena.utils import normalize_conversation, save_run_for_recovery


class Runner:
    def __init__(
        self, comp_name, runs_per_problem, problem_ids, comp_configs_dir, solver_configs_dir, output_dir, redo_all
    ):
        self.comp_name = comp_name
        self.runs_per_problem = runs_per_problem
        self.problem_ids = problem_ids
        self.comp_configs_dir = comp_configs_dir
        self.solver_configs_dir = solver_configs_dir
        self.base_output_dir = output_dir
        self.redo_all = redo_all

        # Load competition config
        competition_config_path = f"{self.comp_configs_dir}/{self.comp_name}.yaml"
        with open(competition_config_path, "r") as f:
            self.competition_config = yaml.safe_load(f)
        self.is_fa_comp = self.competition_config.get("final_answer", True)
        self.options = self.competition_config.get("options", None)

        # Load problems
        self.problems = self._load_problems(self.problem_ids)
        logger.info(f"Loaded {len(self.problems)} problems for competition {self.comp_name}")

    def _load_problems(self, problem_ids):
        """Loads problems for the competition assigned to this runner.

        Returns:
            list: A list of problems, each is a dict with keys
                  "problem_idx", "problem", "answer", and optionally "source" and "problem_types".
        """
        dataset_path = self.competition_config["dataset_path"]

        if not os.path.exists(dataset_path):
            problems = load_dataset(dataset_path, split="train").to_list()
            for problem in problems:
                if "image" in problem and problem["image"] is not None:
                    image_b64 = base64.b64encode(problem["image"]["bytes"]).decode("utf-8")
                    problem["image"] = image_b64
            if problem_ids is not None:
                problems = [p for p in problems if str(p["problem_idx"]) in [str(pid) for pid in problem_ids]]
            return sorted(problems, key=lambda x: x["problem_idx"])

        if self.competition_config.get("final_answer", True):
            answers_path = os.path.join(dataset_path, "answers.csv")
        else:
            answers_path = os.path.join(dataset_path, "grading_scheme.json")
        source_path = os.path.join(dataset_path, "source.csv")
        type_path = os.path.join(dataset_path, "problem_types.csv")
        problems = []

        problem_types = None
        if os.path.exists(type_path):
            with open(type_path, "r") as f:
                problem_types_reader = csv.DictReader(f)
                problem_types = {int(row["id"]): row["type"] for row in problem_types_reader}
                for problem_id in problem_types:
                    problem_types[problem_id] = (
                        problem_types[problem_id].replace('"', "").replace("[", "").replace("]", "").split(",")
                    )

        with open(answers_path, "r") as f:
            if self.competition_config.get("final_answer", True):
                reader = csv.DictReader(f)
            else:
                grading_scheme = json.load(f)
                reader = []
                for item in grading_scheme:
                    reader.append({"id": str(item["id"]), "answer": item["scheme"]})

            for row in reader:
                id_val = int(row["id"])
                problem_path = os.path.join(dataset_path, "problems", f"{id_val}.tex")
                if os.path.exists(problem_path):
                    with open(problem_path, "r") as f_problem:
                        problem_statement = f_problem.read()
                    must_have_image = False
                else:
                    problem_statement = None
                    must_have_image = True

                problem_type_val = None
                if problem_types and id_val in problem_types:
                    problem_type_val = problem_types[id_val]

                image_path = os.path.join(dataset_path, "problems", f"{id_val}.png")
                if os.path.exists(image_path):
                    with open(image_path, "rb") as image_file:
                        image_b64 = base64.b64encode(image_file.read()).decode("utf-8")
                else:
                    image_b64 = None
                    assert not must_have_image, f"Problem {id_val} has no text and no image."

                problems.append(
                    {
                        "problem_idx": id_val,
                        "problem": problem_statement,
                        "image": image_b64,
                        "answer": row["answer"],
                        "problem_type": problem_type_val,
                    }
                )

        if os.path.exists(source_path):
            with open(source_path, "r") as f:
                source_reader = csv.DictReader(f)
                # Create a mapping from id to source for efficient lookup
                source_map = {int(row["id"]): row["source"] for row in source_reader}
                for p in problems:
                    if p["problem_idx"] in source_map:
                        p["source"] = source_map[p["problem_idx"]]
        sorted_problems = sorted(problems, key=lambda x: x["problem_idx"])

        # Filter problems
        if problem_ids is not None:
            sorted_problems = [p for p in sorted_problems if p["problem_idx"] in problem_ids]

        return sorted_problems

    def load_solver_config(self, solver_config_path):
        """Loads and processes the solver (model/agent) configuration from a YAML file.
        Args:
            solver_config_path (str): The path to the solver config YAML file (under configs/models).
        Returns:
            dict: The processed solver configuration with 4 top-level keys:
                  "human_readable_id",
                  "type" (pure_model or agent),
                  "model_config" (the underlying config dict),
                  and "scaffold_config" (non-null for agents only).
        """

        with open(solver_config_path, "r") as f:
            solver_config = yaml.safe_load(f)
        assert "n" not in solver_config, "Solver config should not try to define the number of runs"
        solver_type = solver_config.get("type", "pure_model")

        if solver_type == "pure_model":
            model_config = solver_config
            solver_config = {
                "human_readable_id": model_config["human_readable_id"],
                "type": "pure_model",
                "model_config": model_config,
                "scaffold_config": None,
            }
            solver_config["model_config"].pop("human_readable_id")
            if "other_params" in solver_config["model_config"]:
                solver_config["model_config"].pop("other_params")
        elif solver_type == "agent":
            # Load the inner configs (model/scaffold)
            scaffold_config_path = os.path.join("configs", solver_config["scaffold_config"] + ".yaml")
            with open(scaffold_config_path, "r") as f:
                scaffold_config = yaml.safe_load(f)

            model_config_path = os.path.join("configs", solver_config["model_config"] + ".yaml")
            with open(model_config_path, "r") as f:
                model_config = yaml.safe_load(f)
            
            if "other_params" in model_config:
                model_config.pop("other_params")

            solver_config = {
                "human_readable_id": solver_config["human_readable_id"],
                "type": "agent",
                "model_config": model_config,
                "scaffold_config": scaffold_config,
            }

        return solver_config

    def _prepare_default_api_client_args(self, model_config):
        """Prepares the default client args including tools.

        Args:
            model_config (dict): The base model configuration.

        Returns:
            dict: The default APIClient arguments with tools integrated.
        """
        tool_descriptions = self.competition_config.get("tools", [])
        POSSIBLE_TOOL_FUNCTIONS = {"execute_code": execute_code, "read_paper": read_paper, 
                                   "query_semantic_scholar": query_semantic_scholar, "read_pages": read_pages, 
                                   "find_in_paper": find_in_paper, "take_a_break": take_a_break}
        tools = []
        for tool_desc in tool_descriptions:
            if model_config.get("use_openai_responses_api_tools", model_config.get("use_openai_responses_api", False)) and "tool_spec_openai_responses_api" in tool_desc:
                tools.append((None, tool_desc["tool_spec_openai_responses_api"]))
            elif model_config.get("use_gdm_tools", False):
                name = tool_desc["tool_spec_gdm"]["name"]
                tools.append((None, {name: {}}))
            else:
                tool_spec = tool_desc["tool_spec"]
                func_name = tool_spec["function"]["name"]
                if func_name in POSSIBLE_TOOL_FUNCTIONS:
                    tools.append((POSSIBLE_TOOL_FUNCTIONS[func_name], tool_spec))
        max_tool_calls = self.competition_config.get("max_tool_calls", 0)
        args = model_config.copy()
        args["tools"] = tools
        args["max_tool_calls"] = max_tool_calls
        API_CLIENT_IRRELEVANT_KEYS = ["date"]
        for key in API_CLIENT_IRRELEVANT_KEYS:
            args.pop(key, None)

        return args

    def _build_last_chance_prompt(self, options):
        prompt = """
            Your last message does not provide a final answer in a way that follows the formatting instructions.
            Please based on the conversation history, report the final answer again within \\boxed{}.
            If you did not find the answer, please use \\boxed{None}.
            Do not reason about the problem again or use tools, simply try to extract the final answer from the previous reasoning.
            Boxed answers in thinking/reasoning stages will be ignored; only the final response message is considered.
            """
        if options is not None:
            option_str = ", ".join([opt for opt in options])
            prompt += f"""
                Recall that this is a multiple choice problem, and the only valid answers to put within \\boxed{{}} are: {option_str}."""
        # Remove spaces at the start of each line
        prompt = "\n".join([line.strip() for line in prompt.split("\n")])
        return prompt

    def _initialize_solver(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        """Initializes the solver (pure_model or agent) based on the configuration.
        Args:
            solver_config (dict): The processed solver configuration.
            default_prompt_template (str): The default prompt template with {problem} templated.
            default_api_client_args (dict): The default APIClient arguments.
            last_chance_prompt (str): The prompt to use for the last chance reprompting.
        Returns:
            BaseSolver: An instance of a solver (PureModelSolver or Agent).
        """
        if solver_config["type"] == "pure_model":
            return PureModelSolver(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        elif solver_config["type"] == "agent":
            return AgentPool(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        else:
            raise ValueError(f"Unknown solver type: {solver_config['type']}")

    def _update_status(self, solver_name, all_problem_runs):
        """Writes a status file to the output directory."""
        status_path = os.path.join("logs", "status", f"{self.comp_name}_{solver_name}.txt")
        os.makedirs(os.path.dirname(status_path), exist_ok=True)
        with open(status_path, "w") as f:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"Status as of {current_time}\n")
            f.write(f"There are {len(self.problems)} problems; doing {self.runs_per_problem} runs per problem.\n\n")
            for problem_idx in sorted(all_problem_runs.keys()):
                problem_runs = all_problem_runs[problem_idx]
                progress = []
                for i in range(self.runs_per_problem):
                    if i < problem_runs.N:
                        if problem_runs.correct[i]:
                            progress.append("O")  # Correct run
                        else:
                            progress.append("X")  # Incorrect run
                    else:
                        progress.append("-")  # Not done yet
                progress = "".join(progress)
                f.write(f"Problem {problem_idx:3}: {progress} ({self.runs_per_problem - problem_runs.N} left)\n")

    def prepare_run(self, solver_name, set_request_metadata=True):
        """Prepare solver, status files, and pending batch for this competition."""
        solver_config_path = f"{self.solver_configs_dir}/{solver_name}.yaml"
        solver_config = self.load_solver_config(solver_config_path)
        output_dir = f"{self.base_output_dir}/{self.comp_name}/{solver_name}"
        os.makedirs(output_dir, exist_ok=True)

        # Init the solver
        logger.info(f"Initializing the solver for {solver_name}")
        if "custom_instructions" in solver_config["model_config"] and self.comp_name in solver_config["model_config"].get("custom_instructions", {}):
            logger.info("Using custom instructions for this competition.")
            default_prompt_template = solver_config["model_config"]["custom_instructions"][self.comp_name] + "\n\n" + "{problem}"
        else:
            default_prompt_template = f"{self.competition_config["instruction"]}\n\n" + "{problem}"
        if "custom_instructions" in solver_config["model_config"]:
            del solver_config["model_config"]["custom_instructions"]
        default_api_client_args = self._prepare_default_api_client_args(solver_config["model_config"])
        last_chance_prompt = self._build_last_chance_prompt(self.options)
        solver = self._initialize_solver(
            solver_config, default_prompt_template, default_api_client_args, last_chance_prompt
        )

        # Load existing runs and prepare one big batch of all new runs we need
        logger.info(f"Loading existing runs for {solver_name}")
        all_runs = {}  # problem_idx -> Runs
        batch = []  # list of (text, image) problem statements
        batch_idx_to_problem_idx = {}  # index in batch -> problem_idx
        batch_idx_to_run_idx = {}  # index in batch -> run_idx
        for problem in self.problems:
            # Initialize or load problem runs for this problem
            runs = Runs(self.comp_name, self.is_fa_comp, solver_name, solver_config["type"], problem, output_dir)
            if self.redo_all:
                logger.info(f"Not skipping existing runs for problem {problem["problem_idx"]} (will overwrite)")
            else:
                runs.load_from_file()
                # logger.info(f"Problem {problem["problem_idx"]}: loaded {runs.N} previous runs")
            all_runs[problem["problem_idx"]] = runs

            # Add to batch if we need more runs
            for run_idx in range(self.runs_per_problem - runs.N):
                batch.append(
                    (
                        problem["problem"] if "problem" in problem else None,
                        problem["image"] if "image" in problem else None,
                    )
                )  # (text, image)
                batch_idx_to_problem_idx[len(batch) - 1] = problem["problem_idx"]
                batch_idx_to_run_idx[len(batch) - 1] = run_idx

        self._update_status(solver_name, all_runs)
        if set_request_metadata:
            request_logger.set_metadata(self.comp_name, solver_name, batch_idx_to_problem_idx)
        logger.info(f"Status file created. Total new runs in the batch sent to solver: {len(batch)}.")
        status_path = os.path.join("logs", "status", f"{self.comp_name}_{solver_name}.txt")
        print(f"Printing initial status from {status_path}.")
        with open(status_path, "r") as f:
            print(f.read())

        if len(batch) == 0:
            logger.info(f"Nothing to do. All problems have {self.runs_per_problem} runs already.")
            return None

        return {
            "solver_name": solver_name,
            "solver_config": solver_config,
            "solver": solver,
            "all_runs": all_runs,
            "batch": batch,
            "batch_idx_to_problem_idx": batch_idx_to_problem_idx,
            "batch_idx_to_run_idx": batch_idx_to_run_idx,
            "status_path": status_path,
        }

    def process_solver_responses(
        self,
        solver_name,
        solver,
        all_runs,
        batch_idx_to_problem_idx,
        status_path,
        solver_responses,
        print_final_status=True,
    ):
        """Grade and save solver responses for an already prepared batch."""
        for solver_response in solver_responses:
            problem_idx = batch_idx_to_problem_idx[solver_response.idx]
            problem_runs = all_runs[problem_idx]
            debug_info = f"{solver_name} @ P{problem_idx} (ridx={solver_response.idx})"
            logger.info(f"[{debug_info}] Received a solver response, analyzing...")
            output_tokens = solver_response.detailed_cost.get("output_tokens", 0)
            try:
                # For FA: If strict parsing finds no answer give the model one last chance to format correctly
                if self.is_fa_comp:
                    try:
                        clean_conversation = normalize_conversation(solver_response.conversation)
                    except Exception as e:  # noqa E722
                        save_run_for_recovery("runner reprompt check", problem_runs.path, solver_response, None)
                        raise
                    last_block = clean_conversation[-1]  # might throw
                    last_role, last_content = last_block.get("role", ""), last_block.get("content", "")
                    logger.info(f"[{debug_info}] Extracted last message role={last_role}.")
                    valid_answer_found = True
                    if last_role != "assistant":
                        valid_answer_found = False
                    else:
                        answer = extract_answer(last_content, True)[0]
                        if answer is None or (self.options is not None and str(answer) not in self.options):
                            valid_answer_found = False
                    if not valid_answer_found:
                        logger.info(
                            "No valid answer found, reprompting the model to report the final answer (last chance)."
                        )
                        solver_response = solver.last_chance(solver_response)
                        logger.info("Done reprompting")

                logger.info(f"[{debug_info}] Extracting and grading the answer...")
                # Extract answer from the run and grade
                if not self.is_fa_comp:
                    grader_response = (None, "TODO Grading", 0)  # answer, is_correct, warnings
                else:
                    try:
                        clean_conversation = normalize_conversation(solver_response.conversation)
                    except Exception as e:  # noqa E722
                        logger.error(f"[{debug_info}] Error during conversation normalization: {e}")
                        save_run_for_recovery("runner pre-grading", problem_runs.path, solver_response, None)
                        raise
                    gold_answer = problem_runs.gold_answer
                    try:
                        grader_response = extract_and_grade(
                            clean_conversation,
                            output_tokens,
                            gold_answer,
                            self.competition_config,
                            debug_info=debug_info,
                        )
                    except Exception as e:  # noqa E722
                        logger.error(f"[{debug_info}] Error during grading: {e}")
                        save_run_for_recovery("runner extract+grading", problem_runs.path, solver_response, None)
                        raise

                # Add run to problem_runs
                logger.info(f"[{debug_info}] Adding the run to problem runs.")
                problem_runs.add_run(solver_response, grader_response)

                # Save and update status
                logger.info(f"[{debug_info}] Successfully added a run and updated status. Saving runs to file.")
                problem_runs.save_to_file()
                self._update_status(solver_name, all_runs)

                # Is this problem done?
                if problem_runs.N == self.runs_per_problem:
                    score = sum(problem_runs.correct) if self.is_fa_comp else problem_runs.N
                    logger.info(
                        f"Problem {str(problem_idx)} is done. Answers: {problem_runs.answers} vs Gold answer: {problem_runs.gold_answer}. #Correct: {score}"

                    )
            except Exception as e:
                logger.opt(exception=True).error(f"[{debug_info}] Error during response analysis, can't add run. {e}")
        if print_final_status:
            self.print_final_status(status_path)

    def print_final_status(self, status_path):
        print(f"Done. Printing final status from {status_path}.")
        with open(status_path, "r") as f:
            print(f.read())

    def run(self, solver_name):
        """Run the specified solver (pure_model or agent) on this Runner's competition.
        Args:
            solver (str): The name of the solver to use.
        """
        prepared = self.prepare_run(solver_name)
        if prepared is None:
            return

        responses = prepared["solver"].solve_batch(
            prepared["batch"], prepared["batch_idx_to_problem_idx"], prepared["batch_idx_to_run_idx"]
        )
        self.process_solver_responses(
            solver_name=prepared["solver_name"],
            solver=prepared["solver"],
            all_runs=prepared["all_runs"],
            batch_idx_to_problem_idx=prepared["batch_idx_to_problem_idx"],
            status_path=prepared["status_path"],
            solver_responses=responses,
        )
