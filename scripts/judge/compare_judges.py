"""Replay saved Gemini judge prompts with Astra; never modify benchmark outputs."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from dotenv import load_dotenv
from harness_wrapper.models.oauth.openai_bridge import pace_codex_connections
import yaml

from matharena.json_zst import dump_json_zst, load_json_zst
from matharena.container_runtime import add_container_runtime_argument
from matharena.solvers.judges import JudgePool

ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = "openai/gpt-6-astra-low"
COMPETITIONS = ("arxiv/august", "arxiv_false/august")


def read_config(ref):
    return yaml.safe_load((ROOT / "configs" / f"{ref}.yaml").read_text())


def comparison_config(judge_ref):
    original = read_config(judge_ref)
    config = read_config(original["scaffold_config"])
    config.update(original.get("override", {}))
    config["judge_points_max"] = original["judge_points_max"]
    model = read_config(f"models/{MODEL_CONFIG}")
    # Keep Codex as the transport, including for the tool-free answer checker.
    config["api_client_remove_keys"] = [
        key for key in config.get("api_client_remove_keys", [])
        if key not in {"harness", "harness_version", "harness_config"}
    ]
    model.setdefault("harness_config", {})["tools_enabled"] = bool(
        config.get("enabled_tools") and config.get("max_tool_calls", 0)
    )
    config["model_config"] = model
    # SimpleJudge inserts the archived prompt verbatim, without formatting its braces.
    config["judge_prompt"] = "{student_answer}"
    return config


def saved_prompt(entry):
    for step in entry.get("history", []):
        for message in step.get("messages", []):
            if message.get("role") == "user" and isinstance(message.get("content"), str):
                return message["content"]
    raise ValueError("Saved judgment has no archived user prompt; cannot replay it exactly.")


def collect_cases(comp, config, destination, models=None, problem_ids=None):
    source = ROOT / "outputs" / comp
    judge_ref, = read_config(f"competitions/{comp}")["judge_configs"]
    cases = []
    for path in sorted(source.rglob("*.json.zst")):
        model = path.relative_to(source).parent.as_posix()
        if models and model not in models:
            continue
        data = load_json_zst(path)
        problem = data["idx"]
        if problem_ids and problem not in problem_ids:
            continue
        for run_index in range(len(data.get("messages", []))):
            baseline = next((slot[run_index] for slot in data.get("judgment", [])
                             if isinstance(slot, list) and run_index < len(slot)
                             and isinstance(slot[run_index], dict)
                             and slot[run_index].get("judge_id") == judge_ref), None)
            if baseline is None:
                continue
            # Compare model judgments, including when the viewer saved a manual edit.
            baseline = baseline.get("original_judgment", baseline)
            if not 0 <= baseline["points"] <= baseline["max_points"] or baseline["max_points"] != config["judge_points_max"]:
                raise ValueError(f"Unexpected baseline score scale: {path}, run {run_index + 1}")
            prompt = saved_prompt(baseline)
            fingerprint = hashlib.sha256(json.dumps(
                {"baseline": baseline, "config": config}, sort_keys=True,
            ).encode()).hexdigest()
            result_path = destination / comp / model / f"{problem}_r{run_index + 1}.json.zst"
            case = {
                "competition": comp, "model": model, "problem": problem,
                "run": run_index + 1, "source": str(path), "baseline": baseline,
                "prompt": prompt, "fingerprint": fingerprint,
                "path": result_path, "candidate": None,
            }
            if result_path.exists():
                saved = load_json_zst(result_path)
                if saved["fingerprint"] != fingerprint:
                    raise ValueError(f"Inputs/config changed for {result_path}; use a new --output-dir.")
                case["candidate"] = saved["candidate"]
            cases.append(case)
    return cases


def run_comparison(cases, config, *, container_runtime="docker"):
    pending = [case for case in cases if case["candidate"] is None]
    if not pending:
        return
    queries = [("", {
        "guidelines": "", "ground_truth_solutions": [],
        "student_answer": case["prompt"], "original_problem_statement": "",
    }) for case in pending]
    pool = JudgePool(solver_config=config, container_runtime=container_runtime)
    for response in pool.solve_batch(
        queries,
        {i: case["problem"] for i, case in enumerate(pending)},
        {i: case["run"] - 1 for i, case in enumerate(pending)},
    ):
        if response.points is None:
            continue  # JudgePool already tried three times; leave it pending.
        case = pending[response.idx]
        case["candidate"] = vars(response).copy()
        record = {key: value for key, value in case.items() if key not in {"path", "prompt"}}
        record["judge_model_config"] = MODEL_CONFIG
        record["judge_config"] = config
        case["path"].parent.mkdir(parents=True, exist_ok=True)
        dump_json_zst(record, case["path"], indent=2, ensure_ascii=False)


def summarize(cases, max_points):
    complete = [case for case in cases if case["candidate"] is not None]
    pairs = [(case["baseline"]["points"], case["candidate"]["points"]) for case in complete]
    counts = Counter(pairs)
    n = len(pairs)
    return {
        "total": len(cases), "completed": n, "pending": len(cases) - n,
        "agreement_percent": 100 * sum(a == b for a, b in pairs) / n if n else None,
        "mean_score_change_pp": 100 * sum(b - a for a, b in pairs) / (max_points * n) if n else None,
        "confusion_matrix": [[counts[a, b] for b in range(max_points + 1)] for a in range(max_points + 1)],
    }


def write_report(groups, destination):
    summary = {}
    lines = ["# Gemini versus Astra-low judging", "", "Agreement compares identical archived prompts; it does not establish which judge is correct.", ""]
    for comp, (cases, config) in groups.items():
        scale = config["judge_points_max"]
        overall = summarize(cases, scale)
        by_model = {model: summarize([case for case in cases if case["model"] == model], scale)
                    for model in sorted({case["model"] for case in cases})}
        summary[comp] = {**overall, "by_model": by_model}
        agreement = f'{overall["agreement_percent"]:.2f}%' if overall["completed"] else "pending"
        lines += [f"## {comp}", "", f'{overall["completed"]}/{overall["total"]} judged; exact agreement: {agreement}.', "",
                  "| Evaluated model | Compared | Agreement | Astra − Gemini (pp) |", "|---|---:|---:|---:|"]
        for model, stats in by_model.items():
            values = f'{stats["agreement_percent"]:.2f}% | {stats["mean_score_change_pp"]:+.2f}' if stats["completed"] else "pending | pending"
            lines.append(f'| {model} | {stats["completed"]}/{stats["total"]} | {values} |')
        lines += ["", "Confusion matrix: rows = Gemini score, columns = Astra score.", "",
                  "| Gemini / Astra | " + " | ".join(map(str, range(scale + 1))) + " |",
                  "|---|" + "---:|" * (scale + 1)]
        lines += [f"| {score} | " + " | ".join(map(str, row)) + " |" for score, row in enumerate(overall["confusion_matrix"])]
        lines += ["", "### Disagreements", "", "| Model | Problem | Run | Gemini | Astra | Full judgments |", "|---|---:|---:|---:|---:|---|"]
        for case in cases:
            candidate = case["candidate"]
            if candidate is not None and candidate["points"] != case["baseline"]["points"]:
                lines.append(f'| {case["model"]} | {case["problem"]} | {case["run"]} | {case["baseline"]["points"]:g} | {candidate["points"]} | [details]({case["path"].resolve()}) |')
        lines.append("")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (destination / "comparison.md").write_text("\n".join(lines))
    return summary


def main():
    load_dotenv(ROOT / "scripts/.env")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", help="Run pending Astra judgments; otherwise only summarize/count.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "logs/judge_comparison/astra_low")
    parser.add_argument("--models", nargs="+", help="Optional evaluated-model filter.")
    parser.add_argument("--problem-ids", type=int, nargs="+", help="Optional problem filter, applied to both August benchmarks.")
    add_container_runtime_argument(parser)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    if destination.is_relative_to((ROOT / "outputs").resolve()):
        parser.error("Comparison artifacts must be outside the canonical outputs directory.")
    groups = {}
    for comp in COMPETITIONS:
        judge_ref, = read_config(f"competitions/{comp}")["judge_configs"]
        config = comparison_config(judge_ref)
        cases = collect_cases(comp, config, destination, args.models, args.problem_ids)
        groups[comp] = (cases, config)
        print(f'{comp}: {len(cases)} saved judgments, {sum(case["candidate"] is None for case in cases)} pending comparisons.', flush=True)
    try:
        if args.run:
            print("Judge pacing: one Codex WebSocket connection attempt every 2 seconds, including retries and tool continuations.", flush=True)
            # Scheduling only: keep saved configuration fingerprints valid for resume.
            with pace_codex_connections(2):
                for cases, config in groups.values():
                    run_comparison(cases, config, container_runtime=args.container_runtime)
    finally:
        summary = write_report(groups, destination)
        print(json.dumps(summary, indent=2))
        print(f'Report: {destination / "comparison.md"}')


if __name__ == "__main__":
    main()
