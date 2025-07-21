import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from matharena.configs import load_configs
import re

def matches_models(model_name, models):
    if models is None:
        return True
    for model in models:
        if re.match(model, model_name):
            return True
    return False

def run_configs(config_folder, apis, comp, skip_existing=False, n=4, simul=False, models=None, 
                output_folder="outputs", include_old=False, logs_folder="logs",
                competition_config_folder="competition_configs"):
    """
    Loads configuration files recursively from `config_folder` and runs the appropriate
    command for each valid config file in parallel.

    Each command's stdout/stderr is redirected to a log file under the folder 'logs':
      - If simul is True, the log file is 'logs/{model}.log', where 'model' is obtained from the config.
      - If simul is False, the log file is 'logs/{api}.log' for each api group.

    Args:
        config_folder (str): The root folder where config files are stored.
        apis (list): A list of valid api names. Only configs with an "api" key matching one of these will be used.
        comp (str): Competition parameter passed to the command.
        skip_existing (bool): If True, skip running configs that have already been run.
        n (int): The number of runs to perform for each config.
        simul (bool): If True, run each config separately; if False, group configs by api.
        models (list): A list of model names to filter configs by. Can be a regex. If None, all configs are used.
        output_folder (str): The folder where model answers are stored.
        include_old (bool): If True, include old configs in the config_folder. Old configs are read from configs/exclude.txt
        competition_config_folder (str): The folder where competition configs are stored.
    """
    # Ensure the log directory exists.
    log_dir = Path(logs_folder)
    log_dir.mkdir(exist_ok=True)

    # Convert the config_folder to a Path object.
    configs = load_configs(config_folder, remove_extension=False)

    if not include_old:
        exclude_file = Path(f"{config_folder}/exclude.txt")
        with exclude_file.open("r") as f:
            exclude_regexes = f.read().splitlines()
        
        for config_path in list(configs.keys()):
            for regex in exclude_regexes:
                if re.match(regex, config_path):
                    logger.info(f"Excluding {config_path} due to {regex}")
                    del configs[config_path]
                    break

    valid_configs = []  # Each entry is a tuple: (config_path, api, model)
    for file_path in configs:
        if configs[file_path]['api'] in apis and matches_models(configs[file_path]["model"], models):
            if simul:
                valid_configs.append((file_path, configs[file_path]['api'], configs[file_path]['model']))
            else:
                valid_configs.append((file_path, configs[file_path]['api'], None))

    def run_command(cmd, log_file):
        """Runs a shell command and redirects output to the specified log file."""
        logger.info(f"Running: {cmd} -> logging to {log_file}")
        with open(log_file, "w") as lf:
            subprocess.run(cmd, shell=True, stdout=lf, stderr=subprocess.STDOUT)

    # Use a ThreadPoolExecutor to run commands concurrently.
    with ThreadPoolExecutor() as executor:
        if simul:
            # For simul True, run a separate command for each config.
            for cfg_path, api, model in valid_configs:
                log_file = log_dir / f"{model.replace('/', '-')}.log"
                # Build the command for this config.
                cmd = f"uv run python scripts/run.py --comp {comp} --configs {cfg_path} --n {n} --output-folder {output_folder} --competition-config-folder {competition_config_folder}"
                if skip_existing:
                    cmd += " --skip-existing"
                executor.submit(run_command, cmd, str(log_file))
        else:
            # For simul False, group config files by API.
            api_groups = {}
            for cfg_path, api, _ in valid_configs:
                api_groups.setdefault(api, []).append(str(cfg_path))
            
            # For each API group, build and run a single command.
            for api, cfg_paths in api_groups.items():
                log_file = log_dir / f"{api}.log"
                configs_str = " ".join([path.replace("configs\\models\\", "") for path in cfg_paths])
                cmd = f"uv run python scripts/run.py --comp {comp} --configs {configs_str} --n {n} --output-folder {output_folder}"
                if skip_existing:
                    cmd += " --skip-existing"
                executor.submit(run_command, cmd, str(log_file))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--apis", type=str, nargs="+", required=True)
    parser.add_argument("--comp", type=str, required=True)
    parser.add_argument("--simul", action="store_true")
    parser.add_argument("--models", type=str, nargs="+", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--output-folder", type=str, default="outputs")
    parser.add_argument("--config-folder", type=str, default="configs/models")
    parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
    parser.add_argument("--include-old", action="store_true")
    parser.add_argument("--logs-folder", type=str, default="logs")
    args = parser.parse_args()
    run_configs(args.config_folder, args.apis, args.comp,args.skip_existing, args.n, 
                args.simul, args.models, args.output_folder, include_old=args.include_old, 
                competition_config_folder=args.competition_config_folder, 
                logs_folder=args.logs_folder)
