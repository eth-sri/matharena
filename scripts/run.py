import argparse
import os
from matharena.runner import run
import yaml
from loguru import logger

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=4)
parser.add_argument("--configs", type=str, nargs="+", required=True)
parser.add_argument("--skip-existing", action="store_true")
parser.add_argument("--comp", type=str, required=True)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--configs-folder", type=str, default="configs/models")
parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
args = parser.parse_args()

for config_path in args.configs:
    config_path = config_path + ".yaml" if not config_path.endswith(".yaml") else config_path
    with open(f"{args.configs_folder}/{config_path}", 'r') as f:
        model_config = yaml.safe_load(f)
    
    model_config["n"] = model_config.get("n", args.n)
    logger.info(f"Running config: {config_path}")
    run(model_config, config_path, args.comp, skip_existing=args.skip_existing, 
        output_folder=args.output_folder, competition_config_folder=args.competition_config_folder)