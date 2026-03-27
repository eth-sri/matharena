from datasets import Dataset
from datasets import load_dataset
import pandas as pd
import os
import json
from matharena.configs import load_configs
from loguru import logger
import yaml
import shutil

def get_as_list(string):
    return string.replace('"', "").replace("[", "").replace("]", "").split(',')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--org", type=str, default="MathArena", help="Hugging Face organization name")
    parser.add_argument("--repo-name", type=str, help="Hugging Face repo name", required=True)
    parser.add_argument("--comp", nargs="+", type=str, help="Competition name", required=True)
    parser.add_argument("--competition-configs-folder", type=str, default="configs/competitions", help="Directory containing the raw data")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (not advised, best to keep it private and manually share)")
    parser.add_argument("--add", action="store_true", help="Add to existing dataset instead of overwriting")
    parser.add_argument("--visual-dataset", action="store_true")

    args = parser.parse_args()

    if args.visual_dataset:
        # make temp folder
        os.makedirs("temp", exist_ok=True)

    all_data = []

    for comp in args.comp:
        folder = os.path.join("data", comp)

        competition_config = yaml.safe_load(open(os.path.join(args.competition_configs_folder, comp + ".yaml"), "r"))

        if competition_config.get("final_answer", True):
            answers = pd.read_csv(os.path.join(folder, "answers.csv"))
            if os.path.exists(os.path.join(folder, "problem_types.csv")):
                problem_types = pd.read_csv(os.path.join(folder, "problem_types.csv"))
                problem_types["type"] = problem_types["type"].apply(get_as_list)
                answers = answers.merge(problem_types, on="id")
            ids = list(answers["id"])
        else:
            answers = json.load(open(os.path.join(folder, "grading_scheme.json"), "r"))
            answers = pd.DataFrame(answers)
            answers["id"] = answers["id"].astype(int)
            ids = list(answers["id"])
            
        
        if os.path.exists(os.path.join(folder, "source.csv")):
            source = pd.read_csv(os.path.join(folder, "source.csv"))
            answers = answers.merge(source, on="id")

        for i, idx in enumerate(ids):
            data_dict = dict()
            data_dict["problem_idx"] = int(idx)

            if competition_config.get("final_answer", True):
                if "euler" not in comp:
                    data_dict["answer"] = answers.iloc[i]["answer"] if not args.add else str(answers.iloc[i]["answer"])
                else:
                    data_dict["answer"] = None
                if "type" in answers.columns:
                    data_dict["problem_type"] = answers.iloc[i]["type"]
                if "source" in answers.columns:
                    data_dict["source"] = answers.iloc[i]["source"]
            else:
                data_dict["points"] = answers.iloc[i]["points"]
                data_dict["grading_scheme"] = answers.iloc[i]["scheme"]
                sample_solution_file = os.path.join(folder, "solutions", f"{idx}.tex")
                sample_grading_file = os.path.join(folder, "sample_grading", f"{idx}.txt")

                if os.path.exists(sample_solution_file):
                    data_dict["sample_solution"] = open(sample_solution_file, "r").read()
                if os.path.exists(sample_grading_file):
                    data_dict["sample_grading"] = open(sample_grading_file, "r").read()
            
            if not args.visual_dataset:
                problem_file = os.path.join(folder, "problems", f"{idx}.tex")
                data_dict["problem"] = open(problem_file, "r").read()
            else:
                problem_file = os.path.join(folder, "problems", f"{idx}.png")
                # copy image to temp folder
                
                temp_problem_file = os.path.join("temp", f"{comp.replace("/", "--")}_problem_{idx}.png")
                shutil.copy(problem_file, temp_problem_file)
                data_dict["file_name"] = f"{comp.replace("/", "--")}_problem_{idx}.png"
            
            if len(args.comp) > 1:
                data_dict["competition"] = comp
                data_dict["answer"] = str(data_dict.get("answer", ""))
            all_data.append(data_dict)


    df = pd.DataFrame(all_data)
    if args.add:
        df["competition"] = args.comp
        try:
            existing_dataset = load_dataset(os.path.join(args.org, args.repo_name), split="train")
            existing_df = existing_dataset.to_pandas()
            df = pd.concat([existing_df, df]).drop_duplicates(subset=["problem_idx", "competition"]).reset_index(drop=True)
            logger.info(f"Added {len(df) - len(existing_df)} new samples to existing dataset")
        except Exception as e:
            logger.warning(f"Could not load existing dataset, creating new one. Error: {e}")

    if not args.visual_dataset:
        if len(df) == 0:
            raise ValueError("No data to upload after filtering.")
        logger.info(f"Uploading {len(df)} samples to dataset {args.repo_name} in org {args.org}")
        dataset = Dataset.from_pandas(df)
        # remove __index_level_0__ column if exists
        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])
        dataset.push_to_hub(
            os.path.join(args.org, args.repo_name),
            private=not args.public,
        )
    else:
        df.to_csv(os.path.join("temp", "metadata.csv"), index=False)
        logger.info(f"Uploading visual dataset with {len(df)} samples to dataset {args.repo_name} in org {args.org}")
        dataset = load_dataset("imagefolder", data_dir="temp")
        if "__index_level_0__" in dataset.column_names:
            dataset = dataset.remove_columns(["__index_level_0__"])
        dataset["train"].push_to_hub(
            os.path.join(args.org, args.repo_name),
            private=not args.public,
        )
        # remove temp folder
        shutil.rmtree("temp")



