from datasets import Dataset
import pandas as pd
import os
import json
from matharena.configs import load_configs
from loguru import logger
import yaml

def get_as_list(string):
    return string.replace('"', "").replace("[", "").replace("]", "").split(',')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--org", type=str, default="MathArena", help="Hugging Face organization name")
    parser.add_argument("--repo-name", type=str, help="Hugging Face repo name", required=True)
    parser.add_argument("--comp", type=str, help="Competition name", required=True)
    parser.add_argument("--competition-configs-folder", type=str, default="configs/competitions", help="Directory containing the raw data")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (not advised, best to keep it private and manually share)")

    args = parser.parse_args()

    folder = os.path.join("data", args.comp)

    competition_config = yaml.safe_load(open(os.path.join(args.competition_configs_folder, args.comp + ".yaml"), "r"))

    all_data = []

    if competition_config.get("final_answer", True):
        answers = pd.read_csv(os.path.join(folder, "answers.csv"))
        if os.path.exists(os.path.join(folder, "problem_types.csv")):
            problem_types = pd.read_csv(os.path.join(folder, "problem_types.csv"))
            problem_types["type"] = problem_types["type"].apply(get_as_list)
            answers = answers.merge(problem_types, on="id")
        ids = list(answers["id"])
    else:
        answers = json.load(open(os.path.join(folder, "grading_scheme.json"), "r"))
        ids = [grading["id"] for grading in answers]

    for i, idx in enumerate(ids):
        problem_file = os.path.join(folder, "problems", f"{idx}.tex")
        data_dict = dict()
        data_dict["problem_idx"] = idx
        data_dict["problem"] = open(problem_file, "r").read()
        if competition_config.get("final_answer", True):
            data_dict["answer"] = answers.iloc[i]["answer"]
            if "type" in answers.columns:
                data_dict["problem_type"] = answers.iloc[i]["type"]
        else:
            data_dict["points"] = answers[i]["points"]
            data_dict["grading_scheme"] = answers[i]["scheme"]
            sample_solution_file = os.path.join(folder, "solutions", f"{idx}.tex")
            if not os.path.exists(sample_solution_file):
                logger.warning(f"Sample solution file {sample_solution_file} does not exist, skipping.")
                continue
            data_dict["sample_solution"] = open(sample_solution_file, "r").read()
            sample_grading_file = os.path.join(folder, "sample_grading", f"{idx}.txt")
            data_dict["sample_grading"] = open(sample_grading_file, "r").read()
        
        all_data.append(data_dict)
    df = pd.DataFrame(all_data)
    dataset = Dataset.from_pandas(df)
    dataset.push_to_hub(
        os.path.join(args.org, args.repo_name),
        private=not args.public,
    )




