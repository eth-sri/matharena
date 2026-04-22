from datasets import Dataset
from datasets import load_dataset
import pandas as pd
import os
import json
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
    multi_comp = len(args.comp) > 1

    for comp in args.comp:
        folder = os.path.join("data", comp)
        competition_config = yaml.safe_load(open(os.path.join(args.competition_configs_folder, comp + ".yaml"), "r"))
        problem_types = {}
        source = {}
        source_metadata = {}

        if os.path.exists(os.path.join(folder, "problem_types.csv")):
            problem_types = {
                int(row["id"]): get_as_list(row["type"])
                for row in pd.read_csv(os.path.join(folder, "problem_types.csv"), dtype=str).to_dict(orient="records")
            }

        for filename, target in [("source.csv", source), ("source_metadata.csv", source_metadata)]:
            path = os.path.join(folder, filename)
            if os.path.exists(path):
                target.update({
                    int(row["id"]): {k: v for k, v in row.items() if k != "id" and pd.notna(v)}
                    for row in pd.read_csv(path, dtype=str).to_dict(orient="records")
                })

        if competition_config.get("lean", False):
            original_ids = sorted(
                int(file.removesuffix(".tex"))
                for file in os.listdir(os.path.join(folder, "original"))
                if file.endswith(".tex")
            )
            formal_ids = sorted(
                int(file.removesuffix(".lean"))
                for file in os.listdir(os.path.join(folder, "problems"))
                if file.endswith(".lean")
            )
            if original_ids != formal_ids:
                logger.warning(
                    "Lean statement ids in {} do not match the original statement ids; pairing files by sorted order.",
                    folder,
                )

            for idx, formal_idx in zip(original_ids, formal_ids):
                problem = open(os.path.join(folder, "original", f"{idx}.tex"), "r").read()
                formal_statement = open(os.path.join(folder, "problems", f"{formal_idx}.lean"), "r").read()
                data_dict = {
                    "problem_idx": idx,
                    "problem": problem,
                    "answer": formal_statement,
                    "formal_statement": formal_statement,
                }
                if formal_idx in problem_types:
                    data_dict["problem_type"] = problem_types[formal_idx]
                data_dict.update(source.get(formal_idx, {}))
                data_dict.update(source_metadata.get(formal_idx, {}))
                if multi_comp:
                    data_dict["competition"] = comp
                    data_dict["answer"] = str(data_dict.get("answer", ""))
                all_data.append(data_dict)
            continue

        if competition_config.get("final_answer", True):
            answers = pd.read_csv(os.path.join(folder, "answers.csv"))
        else:
            answers = pd.DataFrame(json.load(open(os.path.join(folder, "grading_scheme.json"), "r")))
        answers["id"] = answers["id"].astype(int)

        for _, row in answers.sort_values("id").iterrows():
            idx = int(row["id"])
            data_dict = {"problem_idx": idx}

            if competition_config.get("final_answer", True):
                data_dict["answer"] = None if "euler" in comp else row["answer"]
            else:
                data_dict["points"] = row["points"]
                data_dict["grading_scheme"] = row["scheme"]
                sample_solution_file = os.path.join(folder, "solutions", f"{idx}.tex")
                sample_grading_file = os.path.join(folder, "sample_grading", f"{idx}.txt")

                if os.path.exists(sample_solution_file):
                    data_dict["sample_solution"] = open(sample_solution_file, "r").read()
                if os.path.exists(sample_grading_file):
                    data_dict["sample_grading"] = open(sample_grading_file, "r").read()

            if not args.visual_dataset:
                data_dict["problem"] = open(os.path.join(folder, "problems", f"{idx}.tex"), "r").read()
            else:
                problem_file = os.path.join(folder, "problems", f"{idx}.png")
                temp_problem_file = os.path.join("temp", f"{comp.replace('/', '--')}_problem_{idx}.png")
                shutil.copy(problem_file, temp_problem_file)
                data_dict["file_name"] = f"{comp.replace('/', '--')}_problem_{idx}.png"

            if idx in problem_types:
                data_dict["problem_type"] = problem_types[idx]
            data_dict.update(source.get(idx, {}))
            data_dict.update(source_metadata.get(idx, {}))
            if multi_comp:
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
