import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import yaml
from huggingface_hub import CommitOperationAdd, HfApi
from loguru import logger

from matharena.configs import load_configs


def load_pr_registry(state_file: Path) -> dict:
    if not state_file.exists():
        return {"entries": {}}
    with state_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {"entries": {}}
    if "entries" not in data or not isinstance(data["entries"], dict):
        data["entries"] = {}
    return data


def save_pr_registry(state_file: Path, registry: dict) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    with state_file.open("w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, sort_keys=True)
        f.write("\n")


def load_competitions_with_hf_ids(competition_configs_folder: Path) -> dict[str, dict]:
    competitions = {}
    for competition_path in competition_configs_folder.rglob("*.yaml"):
        with competition_path.open("r", encoding="utf-8") as f:
            competition_config = yaml.safe_load(f)

        if "huggingface_id" not in competition_config:
            continue

        competition_name = competition_path.relative_to(competition_configs_folder).with_suffix("").as_posix()
        competitions[competition_name] = competition_config
    return competitions


def get_model_hf_id(model_config: dict) -> str | None:
    other_params = model_config.get("other_params")
    if not isinstance(other_params, dict):
        return None
    return other_params.get("huggingface_id")


def compute_average_score(output_dir: Path, n_problems: int) -> float | None:
    all_results = []

    for idx in range(1, n_problems + 1):
        result_path = output_dir / f"{idx}.json"
        if not result_path.exists():
            logger.warning(f"Missing output file {result_path}, skipping this model/competition pair.")
            return None

        with result_path.open("r", encoding="utf-8") as f:
            problem_result = json.load(f)

        correct_values = problem_result.get("correct")
        if not isinstance(correct_values, list):
            logger.warning(f"No usable `correct` list in {result_path}, skipping this model/competition pair.")
            return None

        all_results.append(np.mean(correct_values))

    if not all_results:
        return None

    return float(100.0 * np.mean(all_results))


def build_eval_result_yaml(
    competition_hf_id: str,
    competition_name: str,
    model_hf_id: str,
    score: float,
    date_string: str,
    score_decimals: int,
) -> str:
    payload = [
        {
            "dataset": {
                "id": competition_hf_id,
                "task_id": competition_hf_id,
            },
            "value": round(score, score_decimals),
            "date": date_string,
            "source": {
                "url": f"https://matharena.ai/?comp={competition_name.replace('/', '--')}",
                "name": "Official MathArena Evaluation",
            },
        }
    ]
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=False)


def build_pr_description(template: str, substitutions: dict[str, str]) -> str:
    try:
        return template.format(**substitutions)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Missing key `{missing_key}` in PR description template substitutions.") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Hugging Face model-repo PRs that add leaderboard eval result files."
    )
    parser.add_argument("--competition-configs-folder", type=Path, default=Path("configs/competitions"))
    parser.add_argument("--model-configs-folder", type=Path, default=Path("configs/models"))
    parser.add_argument("--outputs-folder", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--prompt-file",
        type=Path,
        default=Path("configs/prompts/hf_eval_pr_description.txt"),
        help="Template used as PR description. Python format placeholders are supported.",
    )
    parser.add_argument("--score-decimals", type=int, default=2)
    parser.add_argument("--token", type=str, default=None, help="Hugging Face token. If omitted, local auth is used.")
    parser.add_argument("--dry-run", action="store_true", help="Compute and print planned PRs without pushing.")
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("scripts/curation/hf_eval_pr_registry.json"),
        help="Local registry file storing already-created PRs.",
    )
    args = parser.parse_args()

    competitions = load_competitions_with_hf_ids(args.competition_configs_folder)
    if not competitions:
        logger.info("No competitions with `huggingface_id` found. Nothing to do.")
        return

    model_configs = load_configs(str(args.model_configs_folder))
    model_hf_ids = {
        model_config_path: get_model_hf_id(model_config)
        for model_config_path, model_config in model_configs.items()
    }
    model_hf_ids = {k: v for k, v in model_hf_ids.items() if v is not None}

    if not model_hf_ids:
        logger.info("No models with `other_params.huggingface_id` found. Nothing to do.")
        return

    with args.prompt_file.open("r", encoding="utf-8") as f:
        pr_description_template = f.read().strip()

    registry = load_pr_registry(args.state_file)
    api = HfApi(token=args.token)

    for competition_name, competition_config in competitions.items():
        competition_hf_id = competition_config["huggingface_id"]
        n_problems = competition_config.get("n_problems")
        if not isinstance(n_problems, int):
            logger.warning(f"Competition {competition_name} has invalid or missing n_problems, skipping.")
            continue

        for model_config_path, model_hf_id in model_hf_ids.items():
            output_dir = args.outputs_folder / competition_name / model_config_path
            if not output_dir.exists():
                logger.info(f"No outputs for {competition_name} x {model_config_path}, skipping.")
                continue

            score = compute_average_score(output_dir, n_problems)
            if score is None:
                continue

            date_string = datetime.date.today().isoformat()

            eval_result_content = build_eval_result_yaml(
                competition_hf_id=competition_hf_id,
                competition_name=competition_name,
                model_hf_id=model_hf_id,
                score=score,
                date_string=date_string,
                score_decimals=args.score_decimals,
            )

            eval_result_path = (
                f".eval_results/{competition_hf_id.replace('/', '--')}.yaml"
            )

            registry_key = f"{model_hf_id}::{competition_name}::{eval_result_path}"
            if registry_key in registry["entries"]:
                logger.info(
                    f"Skipping {model_hf_id} x {competition_name}: PR already tracked in {args.state_file}."
                )
                continue

            if api.file_exists(
                repo_id=model_hf_id,
                filename=eval_result_path,
                repo_type="model",
            ):
                logger.info(
                    f"Skipping {model_hf_id} x {competition_name}: {eval_result_path} already exists in repo."
                )
                continue

            pr_description = build_pr_description(
                pr_description_template,
                {
                    "model_huggingface_id": model_hf_id,
                    "competition_name": competition_name,
                    "competition_huggingface_id": competition_hf_id,
                    "score": f"{round(score, args.score_decimals):.{args.score_decimals}f}",
                    "result_file_path": eval_result_path,
                    "date": date_string,
                    "comp_id": competition_name.replace("/", "--"),
                },
            )

            logger.info(
                f"Preparing PR for model={model_hf_id}, competition={competition_name}, score={score:.{args.score_decimals}f}"
            )

            if args.dry_run:
                logger.info(f"[DRY RUN] Would create PR on {model_hf_id} updating {eval_result_path}")
                logger.debug(f"PR description:\n{pr_description}")
                logger.debug(f"Eval result content:\n{eval_result_content}")
                continue

            commit_info = api.create_commit(
                repo_id=model_hf_id,
                repo_type="model",
                operations=[
                    CommitOperationAdd(
                        path_in_repo=eval_result_path,
                        path_or_fileobj=eval_result_content.encode("utf-8"),
                    )
                ],
                commit_message=f"Add MathArena evaluation result for {competition_name}",
                commit_description=pr_description,
                create_pr=True,
            )

            pr_url = getattr(commit_info, "pr_url", None)
            if pr_url:
                logger.info(f"Created PR: {pr_url}")
            else:
                logger.info(f"Created commit/PR for {model_hf_id} at {commit_info.commit_url}")

            registry["entries"][registry_key] = {
                "competition_name": competition_name,
                "competition_huggingface_id": competition_hf_id,
                "date": date_string,
                "eval_result_path": eval_result_path,
                "model_config_path": model_config_path,
                "model_huggingface_id": model_hf_id,
                "pr_url": pr_url,
                "commit_url": getattr(commit_info, "commit_url", None),
                "score": round(score, args.score_decimals),
                "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            }
            save_pr_registry(args.state_file, registry)


if __name__ == "__main__":
    main()
