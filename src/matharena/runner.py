import csv
import os
import json
import sympy
from loguru import logger
from datetime import datetime
from datasets import load_dataset
import yaml
import uuid

from matharena.api import APIQuery
from matharena.cot_solver import CoTSolver
from matharena.parser import extract_answer, parse_answer, check_answers, WarningType
from matharena.possible_issues import check_number_proximity_any_order, check_all_numbers, check_output_length
from transformers import AutoTokenizer

def validate_messages(messages, competition):
    if "euler" in competition:
        return True
    
    if not any([message['role'] == 'assistant' for message in messages]):
        return False
    
    for message in messages:
        if message['role'] == 'assistant' and len(message['content']) == 0:
            return False
        
    return True

def run(model_config, config_path, competition, skip_existing=False, output_folder="outputs", 
        competition_config_folder="competition_configs", skip_all=False, recompute_tokens=False):
    model = model_config["model"]
    n = model_config["n"]
    api = model_config["api"]

    with open(f"{competition_config_folder}/{competition}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)

    date_comp = datetime.strptime(competition_config["date"], "%Y-%m-%d")

    max_tokens = model_config.get("max_tokens", competition_config["default_max_tokens"])
    temperature = model_config.get("temperature", competition_config["default_temperature"])
    kwargs = model_config.copy()
    del kwargs["model"]
    del kwargs["n"]
    del kwargs["api"]
    del kwargs["human_readable_id"]

    if "date" in kwargs:
        date_model = datetime.strptime(kwargs["date"], "%Y-%m-%d")
        if date_model > date_comp:
            logger.warning(f"Model date is after competition date. Model: {model}, Competition: {competition}")
        del kwargs["date"]
    kwargs["max_tokens"] = max_tokens
    kwargs["temperature"] = temperature
    
    if "n_code_executions" in competition_config:
        kwargs["n_code_executions"] = competition_config["n_code_executions"]

    logger.info(f"New run, model: {model}, competition: {competition}")

    prompt_template = f"{competition_config['instruction']}\n\n" + "{problem_statement}"

    final_answer_comp = competition_config.get("final_answer", True)
    problem_types = None
    if os.path.exists(competition_config["dataset_path"]):
        answers_path = os.path.join(competition_config["dataset_path"], "answers.csv")
        type_path = os.path.join(competition_config["dataset_path"], "problem_types.csv")
        problems = []
        
        if os.path.exists(type_path):
            with open(type_path, "r") as f:
                problem_types = csv.DictReader(f)
                problem_types = {int(row["id"]): row["type"] for row in problem_types}
                for problem_id in problem_types:
                    problem_types[problem_id] = problem_types[problem_id].replace('"', "").replace("[", "").replace("]", "").split(',')

    if os.path.exists(competition_config["dataset_path"]):
        answers_path = os.path.join(competition_config["dataset_path"], "answers.csv")
        type_path = os.path.join(competition_config["dataset_path"], "problem_types.csv")
        problems = []
        
        if os.path.exists(type_path):
            with open(type_path, "r") as f:
                problem_types = csv.DictReader(f)
                problem_types = {int(row["id"]): row["type"] for row in problem_types}
                for problem_id in problem_types:
                    problem_types[problem_id] = problem_types[problem_id].replace('"', "").replace("[", "").replace("]", "").split(',')

        with open(answers_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                id = row["id"]
                with open(os.path.join(competition_config["dataset_path"], "problems", f"{id}.tex"), "r") as f:
                    problem = f.read()
                problems.append({"problem_idx": int(id), "problem": problem, "answer": row["answer"], "problem_type": problem_types[int(id)] if problem_types is not None else None})

    else:
        problems = load_dataset(competition_config["dataset_path"], split="train").to_list()


    # sort by problem_idx
    problems = sorted(problems, key=lambda x: x["problem_idx"])

    output_dir = os.path.join(f"{output_folder}/{competition}/", config_path.replace(".yaml", ""))
    os.makedirs(output_dir, exist_ok=True)

    batch_prompts = []
    batch_idx_to_problem_idx = {}

    all_messages_per_problem = {i: [] for i in range(len(problems))}
    detailed_costs_per_problem = {i: [] for i in range(len(problems))}    

    for i, problem in enumerate(problems):        
        problem_id = problem["problem_idx"]
        output_file = os.path.join(output_dir, f"{problem_id}.json")
        if skip_existing and os.path.exists(output_file):
            data_file = json.load(open(output_file))
            messages = data_file["messages"]
            
            if recompute_tokens and api=='vllm' :
                tokenizer = AutoTokenizer.from_pretrained(model)
                detailed_costs = [{"cost": len(tokenizer(messages[i][1]['content'], return_tensors="pt")['input_ids'][0])*model_config['read_cost']/1000000 + \
                                   len(tokenizer(messages[i][-1]['content'], return_tensors="pt")['input_ids'][0])*model_config['write_cost']/1000000, 
                                        "input_tokens": len(tokenizer(messages[i][1]['content'], return_tensors="pt")['input_ids'][0]), 
                                        "output_tokens": len(tokenizer(messages[i][-1]['content'], return_tensors="pt")['input_ids'][0])} 
                                        for i in range(len(messages))]
            else:
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
                                zip(detailed_costs, messages) if validate_messages(messages_one, competition) or skip_all
                                ]
            messages = [
                messages_one for messages_one in messages if validate_messages(messages_one, competition) or skip_all
            ]
            detailed_costs_per_problem[i] = detailed_costs
            all_messages_per_problem[i] = messages
            logger.info(f"Skipping problem: {problem_id} ({len(messages)} times)")
            if len(messages) == n or skip_all:
                calculate_problem_results(model_config, problem, output_dir, messages,
                                        detailed_costs, i, competition_config["strict_parsing"], 
                                        final_answer=final_answer_comp)
                continue

        problem_statement = problem["problem"]
        problem_prompt = prompt_template.format(problem_statement=problem_statement)
        for _ in range(n - len(all_messages_per_problem[i])):
            batch_idx_to_problem_idx[len(batch_prompts)] = i
            batch_prompts.append((problem_prompt, None))
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
            calculate_problem_results(model_config, problem, output_dir, 
                                      all_messages_per_problem[problem_idx], 
                                      detailed_costs_per_problem[problem_idx], 
                                      problem_idx, competition_config["strict_parsing"], 
                                      final_answer=final_answer_comp)

def safe_str_int(x, max_digits=4300):
    s = str(x)
    if len(s) > max_digits:
        return f"{s[:20]}...({len(s)} digits)...{s[-20:]}"
    return s

def calculate_problem_results(model_config, problem, output_dir, messages_problem, 
                              costs_problem, problem_idx, strict_parsing, 
                              final_answer=True):
    problem_id = problem["problem_idx"]

    problem_statement = problem["problem"]
    list_answer = "," in str(problem["answer"]) if final_answer else False

    if final_answer:
        gold_answer, _ = parse_answer(str(problem["answer"]), list_answer=list_answer)
    else:
        gold_answer = None
    output_file = os.path.join(output_dir, f"{problem_id}.json")
    n = len(messages_problem)
    answers = []
    warnings = []
    corrects = []
    for j in range(n):
        if final_answer:
            model_answer = messages_problem[j][-1]["content"]
            model_answer, warning = extract_answer(model_answer, strict_parsing, True, list_answer)
            is_correct = check_answers(model_answer, gold_answer)

            if not is_correct and check_output_length(costs_problem[j]["output_tokens"]):
                logger.warning(f"Model output length {costs_problem[j]['output_tokens']} is of the form 10**k * 2**n. This might indicate it hit the token limit. Problem: {problem_id}, idx: {j}")
                warning = WarningType.MINOR # model just didnt have time, any error could have been caused by this
            elif not is_correct and check_all_numbers(messages_problem[j][-1]["content"], str(problem["answer"])):
                logger.warning(f"Model answer: {model_answer} is not equal to gold answer: {gold_answer} even though model output contains the gold answer. Problem: {problem_id}, idx: {j}")
                warning = max(warning, WarningType.POSSIBLE)
            elif not is_correct and check_number_proximity_any_order(str(problem["answer"]), messages_problem[j][-1]["content"]):
                logger.warning(f"Numbers appearing in gold answer appear close together in model answer, but answer was incorrect. Problem: {problem_id}, idx: {j}")
                warning = max(warning, WarningType.POSSIBLE)
            elif len(messages_problem[j][-1]["content"]) == 0:
                logger.warning(f"Empty message in problem: {problem_id}, idx: {j}")
                warning = WarningType.MAJOR
            answers.append(model_answer)
            warnings.append(warning.value)
            corrects.append(is_correct)
        else:
            answers.append(None)
            warnings.append(0)
            corrects.append(0)

    try:
        logger.info(f"Finished problem: {problem_id}, answers: {answers}, gold answer: {str(problem['answer'])}, #Correct: {sum(corrects)}")
    except:
        pass

    if final_answer:
        pass_at_1 = sum(x == gold_answer for x in answers)/n
    else:
        pass_at_1 = 0
    for i in range(len(costs_problem)):
        costs_problem[i]["cost"] = model_config["read_cost"] * costs_problem[i]["input_tokens"] + model_config["write_cost"] * costs_problem[i]["output_tokens"]
        costs_problem[i]["cost"] /= 10 ** 6
    cost = {
        "cost": sum([d["cost"] for d in costs_problem]),
        "input_tokens": sum([d["input_tokens"] for d in costs_problem]),
        "output_tokens": sum([d["output_tokens"] for d in costs_problem]),
    }
    
    if os.path.exists(output_file) and "anonymous_id" in json.load(open(output_file)):
        anonymous_id = json.load(open(output_file))["anonymous_id"]
    else:
        if not os.path.exists("data/ids.txt"):
            os.makedirs("data", exist_ok=True)
            with open("data/ids.txt", "w") as f:
                f.write("")
        all_ids = open("data/ids.txt", "r").read().split("\n")
        anonymous_id = uuid.uuid4().hex[:6]
        while anonymous_id in all_ids:
            anonymous_id = uuid.uuid4().hex[:6]
        with open("data/ids.txt", "a") as f:
            f.write(anonymous_id + "\n")

    with open(output_file, "w") as f:
        json.dump({
                    "idx": problem_idx,
                    "problem": problem_statement,
                    "gold_answer": str(problem.get("answer", "None")),
                    "types": problem.get("problem_type", "None"),
                    "messages": messages_problem, 
                    "answers": [convert_answer(answer) for answer in answers],
                    "correct": corrects,
                    "pass_at_1": pass_at_1,
                    "cost": cost,
                    "detailed_costs": costs_problem,
                    "warnings": warnings,
                    "anonymous_id": anonymous_id
                }, f)

def convert_answer(answer):
    try:
        if type(answer) == sympy.Integer:
            return safe_str_int(int(answer))
        else:
            return safe_str_int(answer)
    except:
        return "None"
