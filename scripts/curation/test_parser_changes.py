from matharena.parser import parse_answer, extract_answer, check_answers
import glob
import os
import yaml
import json
from loguru import logger
import sys
# only log errors
logger.remove()
logger.add(sys.stderr, level="ERROR")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Upload dataset to Hugging Face Hub")
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--competition-configs-folder", type=str, default="configs/competitions", help="Directory containing the raw data")
    args = parser.parse_args()
    # get all yaml files recursively in the competition config folder
    yaml_files = glob.glob(os.path.join(args.competition_configs_folder, "**/*.yaml"), recursive=True)
    for yaml_file in yaml_files:
        with open(yaml_file, "r") as f:
            competition_config = yaml.safe_load(f)
        comp = yaml_file.replace(".yaml", "").replace(args.competition_configs_folder + "/", "")
        if not competition_config.get("final_answer", True):
            continue
        
        path_to_json_files = os.path.join(args.output_folder, comp, "**/*.json")
        json_files = glob.glob(path_to_json_files, recursive=True)
        for json_file in json_files:
            with open(json_file, "r") as f:
                data = json.load(f)
            gold_answer = data["gold_answer"]
            list_answer = "," in str(gold_answer)
            try:
                parsed_gold_answer, _ = parse_answer(gold_answer, list_answer=list_answer)
            except:
                continue # old format, model not used anymore
            
            for i, solution in enumerate(data['messages']):
                extraction, _ = extract_answer(data["messages"][i][-1]['content'], 
                                               competition_config['strict_parsing'], 
                                               True, list_answer)
                check_answer = check_answers(parsed_gold_answer, extraction)
                if data["correct"][i] != (check_answer == True):
                    logger.error(f"Error in {json_file} for {comp} on problem {i + 1}")
                    logger.error(f"Parsed gold answer: {parsed_gold_answer}")
                    logger.error(f"Extraction: {extraction}")
                    logger.error(f"Currently counted as: {data['correct'][i]}")
                    logger.error(f"Should be: {check_answers(parsed_gold_answer, extraction)}")
