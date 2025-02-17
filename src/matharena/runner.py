import csv
import os
import json
import re
import sympy
from matharena.api import APIQuery
from matharena.cot_solver import CoTSolver
from matharena.parser import extract_answer, parse_answer, check_answers, WarningType
from matharena.possible_issues import check_number_proximity_any_order, check_all_numbers, check_output_length
from loguru import logger
from collections import defaultdict
import yaml

def run(model_config, config_path, competition, skip_existing=False, output_folder="outputs"):
    model = model_config["model"]
    n = model_config["n"]
    api = model_config["api"]

    with open(f"data/{competition}/config.yaml", "r") as f:
        competition_config = yaml.safe_load(f)

    max_tokens = model_config.get("max_tokens", competition_config["default_max_tokens"])
    temperature = model_config.get("temperature", competition_config["default_temperature"])
    kwargs = model_config.copy()
    del kwargs["model"]
    del kwargs["n"]
    del kwargs["api"]
    del kwargs["human_readable_id"]
    kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature

    logger.info(f"New run, model: {model}, competition: {competition}")

    prompt_template = f"{competition_config['instruction']}\n\n" + "{problem_statement}"

    answers_path = os.path.join("data", competition, "answers.csv")
    with open(answers_path, "r") as f:
        reader = csv.DictReader(f)
        problems = list(reader)
    for problem in problems:
        problem_path = os.path.join("data", competition, "problems", problem["id"] + ".tex")
        image_path = os.path.join("data", competition, "images", "problem_" + problem["id"] + ".png")
        with open(problem_path, "r") as f:
            problem["problem_statement"] = f.read()
        problem["image_path"] = None # image_path if os.path.exists(image_path) else None

    output_dir = os.path.join(f"{output_folder}/{competition}/", config_path.replace(".yaml", ""))
    os.makedirs(output_dir, exist_ok=True)

    batch_prompts = []
    batch_idx_to_problem_idx = {}

    all_messages_per_problem = {i: [] for i in range(len(problems))}
    detailed_costs_per_problem = {i: [] for i in range(len(problems))}

    for i, problem in enumerate(problems):
        problem_id = problem["id"]
        output_file = os.path.join(output_dir, f"{problem_id}.json")
        if skip_existing and os.path.exists(output_file):
            data_file = json.load(open(output_file))
            messages = data_file["messages"]
            
            # print all the message lengths
            if "detailed_costs" in data_file:
                detailed_costs = data_file["detailed_costs"]
            else:
                cost = data_file["cost"]
                detailed_costs = [{"cost": cost["cost"] if i == 0 else 0, 
                                   "input_tokens": cost["input_tokens"] if i == 0 else 0, 
                                   "output_tokens": cost["output_tokens"] if i == 0 else 0} 
                                   for i in range(len(messages))]
            detailed_costs = [detailed_costs_one for detailed_costs_one, messages_one in 
                              zip(detailed_costs, messages) if len(messages_one[-1]["content"]) > 0]
            messages = [
                messages_one for messages_one in messages if len(messages_one[-1]["content"]) > 0
            ]
            detailed_costs_per_problem[i] = detailed_costs
            all_messages_per_problem[i] = messages
            logger.info(f"Skipping problem: {problem_id} ({len(messages)} times)")
            if len(messages) == n:
                calculate_problem_results(problem, output_dir, messages,
                                            detailed_costs, i, competition_config["strict_parsing"])
                continue

        problem_statement = problem["problem_statement"]
        problem_prompt = prompt_template.format(problem_statement=problem_statement)
        for _ in range(n - len(all_messages_per_problem[i])):
            batch_idx_to_problem_idx[len(batch_prompts)] = i
            batch_prompts.append((problem_prompt, problem["image_path"]))

    logger.info("Collected all queries, now running")

    if len(batch_prompts) == 0:
        return
    api = APIQuery(
        model=model, 
        api=api,
        **kwargs
    )

    cot_solver = CoTSolver(
        querier=api
    )

    for idx, messages, detailed_cost in cot_solver.solve(batch_prompts):
        problem_idx = batch_idx_to_problem_idx[idx]
        problem = problems[problem_idx]
        all_messages_per_problem[problem_idx].append(messages)
        detailed_costs_per_problem[problem_idx].append(detailed_cost)

        # check if the whole problem is finished
        if len(all_messages_per_problem[problem_idx]) == n:
            calculate_problem_results(problem, output_dir, 
                                      all_messages_per_problem[problem_idx], 
                                      detailed_costs_per_problem[problem_idx], 
                                      problem_idx, competition_config["strict_parsing"])

def calculate_problem_results(problem, output_dir, messages_problem, 
                              costs_problem, problem_idx, strict_parsing):
    problem_id = problem["id"]

    problem_statement = problem["problem_statement"]
    gold_answer, _ = parse_answer(str(problem["answer"]))
    output_file = os.path.join(output_dir, f"{problem_id}.json")
    n = len(messages_problem)
    answers = []
    warnings = []
    corrects = []
    try:
        string_answer = str(model_answer)
    except:
        string_answer = "None"
        warning = WarningType.MAJOR
    for j in range(n):
        model_answer = messages_problem[j][-1]["content"]
        model_answer, warning = extract_answer(model_answer, strict_parsing)
        is_correct = check_answers(model_answer, gold_answer)
        if not is_correct and check_output_length(costs_problem[j]["output_tokens"]):
            logger.warning(f"Model output length is of the form 10**k * 2**n. This might indicate it hit the token limit. Problem: {problem_id}, idx: {j}")
            warning = WarningType.MINOR # model just didnt have time, any error could have been caused by this
        elif not is_correct and check_all_numbers(messages_problem[j][-1]["content"], str(problem["answer"])):
            logger.warning(f"Model answer: {model_answer} is not equal to gold answer: {gold_answer} even though model output contains the gold answer. Problem: {problem_id}, idx: {j}")
            warning = max(warning, WarningType.POSSIBLE)
        elif not is_correct and check_number_proximity_any_order(str(gold_answer), string_answer):
            logger.warning(f"Numbers appearing in gold answer appear close together in model answer, but answer was incorrect. Problem: {problem_id}, idx: {j}")
            warning = max(warning, WarningType.POSSIBLE)
        elif len(messages_problem[j][-1]["content"]) == 0:
            logger.warning(f"Empty message in problem: {problem_id}, idx: {j}")
            warning = WarningType.MAJOR
        answers.append(model_answer)
        warnings.append(warning.value)
        corrects.append(is_correct)

    try:
        logger.info(f"Finished problem: {problem_id}, answers: {answers}, gold answer: {str(problem['answer'])}, #Correct: {sum(corrects)}")
    except:
        pass
    pass_at_1 = sum(x == gold_answer for x in answers)/n
    cost = {
                "cost": sum([d["cost"] for d in costs_problem]),
                "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
                "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
            }

    with open(output_file, "w") as f:
        json.dump({
                    "idx": problem_idx,
                    "problem": problem_statement,
                    "gold_answer": str(problem["answer"]),
                    "messages": messages_problem, 
                    "answers": [convert_answer(answer) for answer in answers],
                    "correct": corrects,
                    "pass_at_1": pass_at_1,
                    "cost": cost,
                    "detailed_costs": costs_problem,
                    "warnings": warnings
                }, f)

def convert_answer(answer):
    if type(answer) == sympy.Integer:
        return int(answer)
    try:
        return str(answer)
    except:
        return "None"