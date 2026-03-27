import pandas as pd
import json
import os
import glob
import numpy as np
import yaml
from matharena.json_zst import OUTPUT_JSON_SUFFIX, load_json_zst
output_folders = [
    "outputs/aime/aime_2025",
    "outputs/hmmt/hmmt_feb_2025",
    "outputs/cmimc/cmimc_2025",
    "outputs/brumo/brumo_2025",
]

input_tokens = dict()
output_tokens = dict()

for output_folder in output_folders:
    for file in glob.glob(os.path.join(output_folder, f"**/*{OUTPUT_JSON_SUFFIX}"), recursive=True):
        data = load_json_zst(file)
        problem_idx = os.path.basename(file).removesuffix(OUTPUT_JSON_SUFFIX)
        config_path_model = os.path.dirname(file).replace(output_folder, "configs/models") + ".yaml"
        with open(config_path_model, "r") as cf:
            model_config = yaml.safe_load(cf)
            model_name = f'\\textsc{{{model_config["human_readable_id"]}}}'
        if model_name not in input_tokens:
            input_tokens[model_name] = dict()
            output_tokens[model_name] = dict()
        if problem_idx not in input_tokens[model_name]:
            input_tokens[model_name][problem_idx] = []
            output_tokens[model_name][problem_idx] = []
        input_tokens[model_name][problem_idx].append(data["cost"]["input_tokens"] / 4)
        output_tokens[model_name][problem_idx].append(data["cost"]["output_tokens"] / 4)

# average over problems
input_tokens_avg = {model: np.mean([np.mean(tokens) for tokens in input_tokens[model].values()]) for model in input_tokens}
output_tokens_avg = {model: np.mean([np.mean(tokens) for tokens in output_tokens[model].values()]) for model in output_tokens}

df = pd.DataFrame([input_tokens_avg, output_tokens_avg], index=["input_tokens", "output_tokens"]).T
# sort by output tokens
df = df.sort_values(by="output_tokens", ascending=False)

print(df.to_latex(float_format="%.2f"))
