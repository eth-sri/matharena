"""This module handles loading and managing configurations from YAML files."""

import json
import os
import re
from pathlib import Path
import yaml
from loguru import logger

from matharena.json_zst import load_json_zst


def check_valid_config(config):
    """Checks if a configuration is valid.

    Args:
        config (dict): The configuration to check.

    Raises:
        AssertionError: If the configuration is invalid.
    """
    assert "human_readable_id" in config and isinstance(
        config["human_readable_id"], str
    ), "human_readable_id not found in config"


def load_configs(root_dir, remove_extension=True):
    """Loads all YAML configuration files from a directory.

    Args:
        root_dir (str): The root directory to search for configuration files.
        remove_extension (bool, optional): Whether to remove the file extension from the
            configuration keys. Defaults to True.

    Returns:
        dict: A dictionary of configurations, where the keys are the relative file paths.
    """
    root = Path(root_dir)
    # Find all YAML files (supporting both .yaml and .yml extensions) recursively.
    yaml_files = list(root.rglob("*.yaml")) + list(root.rglob("*.yml"))

    output_configs = dict()

    for file_path in yaml_files:
        with file_path.open("r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        file_path_remove_config = str(file_path).replace("\\", "/").replace(str(root).replace("\\", "/") + "/", "")
        if remove_extension:
            file_path_remove_config = file_path_remove_config.replace(".yaml", "").replace(".yml", "")
        try:
            check_valid_config(config_data)
        except AssertionError as e:
            raise ValueError(f"Config not correct in {file_path}: {e}")
        output_configs[file_path_remove_config] = config_data

    return output_configs


def exclude_configs(configs, human_readable_ids, exclude_file_path, comp):
    """Excludes configurations based on a regex file.

    Args:
        configs (dict): The configurations to filter.
        human_readable_ids (dict): The human-readable IDs of the configurations.
        exclude_file_path (str): The path to the file containing regexes for exclusion.
        comp (str): The competition name.

    Returns:
        tuple: A tuple containing the filtered configurations and human-readable IDs.
    """
    if exclude_file_path is None:
        return configs, human_readable_ids
    with open(exclude_file_path, "r", encoding="utf-8") as f:
        exclude_regexes = [line.strip() for line in f.readlines()]
    for config_path in list(configs.keys()):
        for regex in exclude_regexes:
            if " EXCEPT " in regex:
                regex, competition_exception = regex.split(" EXCEPT ")
                if re.match(competition_exception, comp):
                    continue
            if re.match(regex, config_path):
                logger.info(f"Excluding {config_path} due to {regex}")
                del configs[config_path]
                del human_readable_ids[config_path]
                break
    return configs, human_readable_ids


def extract_existing_configs(
    comp,
    root_dir,
    root_dir_configs,
    root_dir_competition_configs,
    exclude_file_path=None,
    allow_non_existing_judgment=False,
):
    """Extracts existing configurations for a competition.

    Args:
        comp (str): The competition name.
        root_dir (str): The root directory of the competition data.
        root_dir_configs (str): The root directory of the model configurations.
        root_dir_competition_configs (str): The root directory of the competition configurations.
        exclude_file_path (str, optional): The path to the file containing regexes for
            exclusion. Defaults to None.
        allow_non_existing_judgment (bool, optional): Whether to allow non-existing
            judgments. Defaults to False.

    Returns:
        tuple: A tuple containing the filtered configurations and human-readable IDs.
    """
    with open(f"{root_dir_competition_configs}/{comp}.yaml", "r") as f:
        competition_config = yaml.safe_load(f)

    is_final_answer = competition_config.get("final_answer", True)

    all_configs = load_configs(root_dir_configs)

    configs = dict()
    human_readable_ids = dict()
    for config_path in all_configs:
        if os.path.exists(os.path.join(root_dir, comp, config_path)):
            exists = True
            if not is_final_answer:
                for file in os.listdir(os.path.join(root_dir, comp, config_path)):
                    data = load_json_zst(os.path.join(root_dir, comp, config_path, file))
                    if "judgment" not in data and not allow_non_existing_judgment:
                        logger.warning(f"Judgment not found in {file}")
                        exists = False
                        break
            if exists:
                configs[config_path] = all_configs[config_path]
                human_readable_ids[config_path] = all_configs[config_path]["human_readable_id"]

    if len(set(human_readable_ids.values())) != len(human_readable_ids):
        # find for which config the human readable id is duplicated
        duplicated = set()
        for k, v in human_readable_ids.items():
            if v in duplicated:
                logger.error(f"Duplicate human readable id {v} found in {k}")
            duplicated.add(v)
        raise ValueError("Duplicate human readable ids. Website currently does not support this.")

    configs, human_readable_ids = exclude_configs(configs, human_readable_ids, exclude_file_path, comp)
    return configs, human_readable_ids
