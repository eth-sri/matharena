from datasets import Dataset, load_dataset
import pandas as pd
import os
import json
from matharena.configs import load_configs
from loguru import logger
import yaml
import shutil


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--org", type=str, default="MathArena", help="Hugging Face organization name")
    parser.add_argument("--repo-name", type=str, help="Hugging Face repo name", required=True)
    parser.add_argument("--comp", type=str, help="Competition name", required=True)
    parser.add_argument("--output-folder", type=str, default="outputs", help="Output directory for the dataset")
    parser.add_argument("--data-folder", type=str, default="data", help="Output directory for the dataset")
    parser.add_argument("--local-save-path", type=str, default=None, help="If specified, save the dataset locally to this path instead of uploading to Hugging Face")

    parser.add_argument("--configs-folder", type=str, default="configs/models", help="Directory containing the configs of the models")
    parser.add_argument("--competition-configs-folder", type=str, default="configs/competitions", help="Directory containing the configs")
    parser.add_argument("--public", action="store_true", help="Make the dataset public (not advised, best to keep it private and manually share)")
    parser.add_argument("--visual-dataset", action="store_true", help="Upload visual dataset with images")

    args = parser.parse_args()

    if args.visual_dataset:
        # make temp folder
        os.makedirs("temp", exist_ok=True)
        # add all images from data/<comp>/problems to temp
        problem_folder = os.path.join(args.data_folder, args.comp, "problems")
        for problem_file in os.listdir(problem_folder):
            temp_problem_file = os.path.join("temp", problem_file)
            shutil.copy(os.path.join(problem_folder, problem_file), temp_problem_file)


    folder = os.path.join(args.output_folder, args.comp)

    configs = load_configs(args.configs_folder)

    competition_config = yaml.safe_load(open(os.path.join(args.competition_configs_folder, args.comp + ".yaml"), "r"))

    all_data = []

    for config_path in configs:
        if "tiiuae" in config_path:
            continue # skip tiiuae models for now due to model being private
        folder_model = os.path.join(folder, config_path)
        if not os.path.exists(folder_model):
            logger.info(f"Folder {folder_model} does not exist, skipping...")
            continue
        
        try:
            if "model_config" in configs[config_path]:
                config_model = configs[configs[config_path]["model_config"].replace("models/", "")]
                configs[config_path]["read_cost"] = config_model["read_cost"]
                configs[config_path]["write_cost"] = config_model["write_cost"]
        except:
            continue
        # list all files in the folder
        problem_files = os.listdir(folder_model)
        for file in problem_files:
            file_name = os.path.join(folder_model, file)
            problem_name = file.split(".")[0]
            data = json.load(open(file_name, "r"))
            gold_answer = data["gold_answer"]
            problem = data["problem"] if not args.visual_dataset else f"{problem_name}.png"
            problem_key = "problem" if not args.visual_dataset else "file_name"
            for i in range(len(data["messages"])):
                default_dict = {
                    "problem_idx": problem_name,
                    problem_key: problem,
                    "model_name": configs[config_path]["human_readable_id"],
                    "model_config": config_path,
                    "idx_answer": i,
                    "all_messages": data["messages"][i],
                    "user_message": data["messages"][i][0]["content"],
                    "answer": data["messages"][i][-1]["content"],
                    "input_tokens": data["detailed_costs"][i]["input_tokens"],
                    "output_tokens": data["detailed_costs"][i]["output_tokens"],
                    "cost": data["detailed_costs"][i]["cost"],
                    "input_cost_per_tokens": configs[config_path]["read_cost"],
                    "output_cost_per_tokens": configs[config_path]["write_cost"],
                }

                if "source" in data:
                    default_dict["source"] = data["source"]

                if "history" in data:
                    default_dict["history"] = data["history"]

                if competition_config.get("final_answer", True):
                    extra_dict = {
                        "gold_answer": gold_answer,
                        "parsed_answer": data["answers"][i],
                        "correct": data["correct"][i],
                    }
                else:
                    extra_dict = dict()
                    for j in range(len(data["judgment"])):
                        extra_dict_judge = {
                            f"points_judge_{j+1}": data["judgment"][j][i]["points"],
                            f"grading_details_judge_{j+1}": data["judgment"][j][i]["details"],
                            f"error_judge_{j+1}": data["judgment"][j][i]["error"] if "error" in data["judgment"][j][i] else None,
                            f"max_points_judge_{j+1}": data["judgment"][j][i]["max_points"],
                        }
                        extra_dict.update(extra_dict_judge)


                all_data.append({**default_dict, **extra_dict})
    df = pd.DataFrame(all_data)
    if "parsed_answer" in df.columns:
        df["parsed_answer"] = df["parsed_answer"].astype(str)
    
    if not args.visual_dataset:
        
        if args.local_save_path is not None:
            if len(os.path.dirname(args.local_save_path)) > 0:
                os.makedirs(os.path.dirname(args.local_save_path), exist_ok=True)
            # remove the history col
            logger.info(f"Saving dataset locally to {args.local_save_path}")
            df.to_json(args.local_save_path, index=False, orient="records", lines=True)
            logger.info(f"Saved dataset locally to {args.local_save_path}")
        else:
            logger.info(f"Uploading dataset with {len(df)} samples to dataset {args.repo_name} in org {args.org}")
            if "history" in df.columns:
                df = df.drop(columns=["history"])
            dataset = Dataset.from_pandas(df)
            dataset.push_to_hub(
                os.path.join(args.org, args.repo_name),
                private=not args.public,
            )
    else:
        df.to_csv(os.path.join("temp", "metadata.csv"), index=False)
        logger.info(f"Uploading visual dataset with {len(df)} samples to dataset {args.repo_name} in org {args.org}")
        dataset = load_dataset("imagefolder", data_dir="temp")
        dataset["train"].push_to_hub(
            os.path.join(args.org, args.repo_name),
            private=not args.public,
        )

        # remove temp folder
        shutil.rmtree("temp")




