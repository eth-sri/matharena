import os
import json
import yaml
import argparse
import pandas as pd
from matharena.configs import extract_existing_configs
from collections import defaultdict
from datasets import load_dataset
import pandas as pd
from collections import Counter
from itertools import chain

from matharena.json_zst import OUTPUT_JSON_SUFFIX, load_json_zst


def get_latex_model_name(model):
    if model == "o4-mini (high)":
        return r"\ofour~\textsc{(high)}"
    elif model == "o4-mini (medium)":
        return r"\ofour~\textsc{(medium)}"
    elif model == "o4-mini (low)":
        return r"\ofour~\textsc{(low)}"
    elif model == "geminipro":
        return r"\geminipro"
    elif model == "o3 (high)":
        return r"\othree~\textsc{(high)}"
    elif model == "Grok 3 Mini (high)":
        return r"\grokthreemini~\textsc{(high)}"
    elif model == "Grok 3 Mini (low)":
        return r"\grokthreemini~\textsc{(low)}"
    elif model == "Qwen3-235B-A22B":
        return r"\qwenthreebig"
    elif model == "Qwen3-30B-A3B":
        return r"\qwenthreesmall"        
    return r"\textsc{" + model + "}"

def generate_type_table(df):
    exploded = df.explode('types')
    avg_by_model = df.groupby(['model', 'comp'])['accuracy'].mean().groupby('model').mean()

    exploded = df.explode('types')
    pivot = exploded.groupby(['model', 'types'])['accuracy'].mean().unstack(fill_value=float('nan'))

    pivot = pivot.loc[avg_by_model.sort_values(ascending=False).index]

    type_distribution = Counter(chain.from_iterable(df['types']))

    total = sum(type_distribution.values())
    dist_str = ', '.join(f"{k}: {v} ({v/total:.1%})" for k, v in sorted(type_distribution.items()))

    latex_rows = []
    for model, row in pivot.iterrows():
        scores = ' & '.join(f'{v*100:.1f}' if pd.notna(v) else '?' for v in row)
        latex_rows.append(f'{get_latex_model_name(model)} & {scores} \\\\')

    types = [f'\\textbf{{{t}}}' for t in pivot.columns]
    header = ' & '.join(['\\textbf{Model}'] + types) + ' \\\\'

    return f"""
\\begin{{table}}[t]
    \\centering
    \\caption{{Average accuracy per model per problem type. Problem type distribution: {dist_str}.}}
    \\resizebox{{\\textwidth}}{{!}}{{
        \\begin{{tabular}}{{l{'x{{2}}{{1}}' * len(types)}}}
            \\toprule
            {header}
            \\midrule
            {'\n            '.join(latex_rows)}
            \\midrule
            \\textsc{{Overall}} & {' & '.join((pivot.mean(0)*100).apply(lambda x: f'{x:.1f}').tolist())}\\\\
            \\bottomrule
        \\end{{tabular}}
    }}
    \\label{{tab:numerical_type}}
\\end{{table}}
"""

def analyze_run(args, competition):
    _, human_readable_ids = extract_existing_configs(competition, args.output_folder, args.config_folder, 
                                                           args.competition_config_folder, 
                                                           allow_non_existing_judgment=True)
    
    with open(f"{args.competition_config_folder}/{competition}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)
        
    data_location = competition_config.get("dataset_path", f'./data/{competition.replace('.yaml', '')}')

    if os.path.exists(data_location):
        out_dir = os.path.join(args.output_folder, competition)
        results = {}
        for config_path in human_readable_ids:
            model_comp_dir = os.path.join(out_dir, config_path)
            results[f"{human_readable_ids[config_path]}"] = {}
            for problem_file in os.listdir(model_comp_dir):
                if not problem_file.endswith(OUTPUT_JSON_SUFFIX):
                    continue
                problem_idx = int(problem_file.removesuffix(OUTPUT_JSON_SUFFIX))
                data = load_json_zst(os.path.join(model_comp_dir, problem_file))
                results[f"{human_readable_ids[config_path]}"][problem_idx] = data
        return results
    else:
        problem_data = load_dataset(data_location)['train'].to_pandas()
        output_data = load_dataset(f'{data_location}_outputs')['train'].to_pandas()
        output_data['problem_idx'] = output_data['problem_idx'].astype('Int64')
        full_data = output_data.merge(problem_data[['problem_idx', 'problem_type']], on='problem_idx', how='left')
        grouped = full_data.groupby(['model_name', 'problem_idx'])
        agg_df = grouped.agg({
            'correct': list,                 
            'problem_type': 'first',        
            'cost': 'sum'                   
        }).reset_index()
        nested_dict = (
            agg_df
            .set_index(['model_name', 'problem_idx'])
            .to_dict(orient='index')
        )

        results = defaultdict(dict)
        for (model_name, problem_idx), values in nested_dict.items():
            results[model_name][problem_idx] = {}
            results[model_name][problem_idx]['cost'] = {'cost': values['cost']}
            results[model_name][problem_idx]['types'] = values['problem_type'].tolist() 
            results[model_name][problem_idx]['correct'] = values['correct']
        return results

def get_problem_stats(results, comp, model, problem):
    if type(problem) == str:
        problem = int(problem)
    res = results[comp][model][problem]
    corrects = res["correct"]
    corrects = [c if c is not None else False for c in corrects]
    warnings = res.get("warnings", [False] * len(corrects))
    types = res.get("types", "None")
    if len(corrects) == 0:
        return {
            "nb_instances": 0,
            "corrects": [],
            "accuracy": 0,
            "cost": 0,
            "types": types
        }
    nb_inst = len(corrects)
    acc = sum(corrects) / nb_inst 
    return {
        "nb_instances": nb_inst,
        "corrects": corrects,
        "accuracy": acc,
        "warnings": warnings,
        "cost": res['cost']['cost'],
        "types": types
    }

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--comps", type=str, nargs='+', required=True)
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--config-folder", type=str, default="configs/models")
    parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
    args = parser.parse_args()

    results = {}
    for competition in args.comps:
        results[competition] = analyze_run(args, competition)

    results_summary = []
    for comp in results:
        for model in results[comp]:
            for problem in results[comp][model]:
                problem_summary = get_problem_stats(results, comp, model, problem)
                problem_summary['comp'] = comp
                problem_summary['model'] = model
                problem_summary['problem'] = problem
                results_summary.append(problem_summary)
    results_summary = pd.DataFrame(results_summary)

    print(results_summary.explode('types').groupby(['comp', 'types'])['problem'].nunique())

    print(generate_type_table(results_summary))
