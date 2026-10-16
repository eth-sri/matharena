
import argparse
import copy
import csv
import json
import os
import math
from threading import RLock

import yaml
from flask import Flask, redirect, render_template, url_for, send_from_directory, request, abort, jsonify, g

from matharena.configs import extract_existing_configs
from matharena.json_zst import OUTPUT_JSON_SUFFIX, dump_json_zst, load_json_zst
from matharena.viewer_cache import ViewerResultCache

"""
    A dashboard app that shows all about a run
"""

parser = argparse.ArgumentParser()
parser.add_argument("--comp", type=str)
parser.add_argument("--models", type=str, nargs="+", default=None)
parser.add_argument("--port", type=int, default=5001)
parser.add_argument("--output-folder", type=str, default="outputs")
parser.add_argument("--config-folder", type=str, default="configs/models")
parser.add_argument("--competition-config-folder", type=str, default="configs/competitions")
parser.add_argument(
    "--cache-dir",
    default=os.path.join(os.path.dirname(__file__), ".cache"),
    help="Directory for reusable result summaries (safe to delete).",
)
parser.add_argument(
    "--disable-overwrite",
    action="store_true",
    help="Disable /override routes from writing result files.",
)
parser.add_argument(
    "--disable-debug",
    action="store_true",
    help="Disable Flask debug mode.",
)
parser.add_argument(
    "--edit",
    action="store_true",
    help="Enable editable dataset page at /edit.",
)
args = parser.parse_args()

current_comp = args.comp if args.comp is not None else "imo/imo_2025"  # fast-loading default comp
model_id_to_config_path = {}
result_cache = ViewerResultCache(args.cache_dir)

# Find all comps, directories below outputs of depth 2
all_comps = []
for root, dirs, files in os.walk(args.output_folder):
    rel_path = os.path.relpath(root, args.output_folder)
    depth = rel_path.count(os.sep)
    if depth == 1:
        all_comps.append(rel_path)
        dirs.clear()  # Model outputs below this level cannot add competitions.

# sort by date modified
all_comps = sorted(all_comps, key=lambda x: os.path.getmtime(os.path.join(args.output_folder, x)), reverse=True)


def analyze_run(competition, models):
    global model_id_to_config_path
    configs, human_readable_ids = extract_existing_configs(
        competition,
        args.output_folder,
        args.config_folder,
        args.competition_config_folder,
        allow_non_existing_judgment=True,
    )
    if models is not None:
        for config_path in list(human_readable_ids.keys()):
            if human_readable_ids[config_path] not in models:
                del human_readable_ids[config_path]
                del configs[config_path]
    model_id_to_config_path = {human_readable_ids[k]: k for k in human_readable_ids}
    out_dir = os.path.join(args.output_folder, competition)

    result_paths = {}
    for config_path in human_readable_ids:
        model_comp_dir = os.path.join(out_dir, config_path)
        model = human_readable_ids[config_path]
        result_paths[model] = {}
        for problem_file in os.listdir(model_comp_dir):
            if not problem_file.endswith(OUTPUT_JSON_SUFFIX):
                continue
            problem_idx = int(problem_file.removesuffix(OUTPUT_JSON_SUFFIX))
            result_paths[model][problem_idx] = os.path.realpath(os.path.join(model_comp_dir, problem_file))
    summaries = result_cache.summaries(path for paths in result_paths.values() for path in paths.values())
    return {
        model: {problem: {**summaries[path], "_viewer_result_path": path} for problem, path in paths.items()}
        for model, paths in result_paths.items()
    }


def get_problem_result(model, problem):
    record = results[model][int(problem)]
    path = record.get("_viewer_result_path")
    return result_cache.detail(path) if path else record


def load_sources(comp):
    sources = {}
    with open(f"{args.competition_config_folder}/{comp}.yaml", "r", encoding="utf-8") as f:
        competition_config = yaml.safe_load(f)
    source_path = f"{competition_config['dataset_path']}/source.csv"
    if os.path.exists(source_path):
        with open(source_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            problems = [row for row in reader][1:]
            sources = {int(row[0]): row[1] for row in problems}
    return sources


def _format_grading_scheme_text(scheme):
    if scheme is None:
        return None
    if isinstance(scheme, str):
        return scheme.strip()
    if isinstance(scheme, list):
        return "\n".join(
            f"{i}. [{item.get('points', 0)} pts] {item.get('title', item.get('part_id', f'Part {i}'))}: {item.get('desc', item.get('description', ''))}"
            for i, item in enumerate(scheme, start=1)
        ).strip()
    return json.dumps(scheme, ensure_ascii=False, indent=2)


def _extract_ground_truth_proofs(record):
    proofs = []
    for key in ["ground_truth_proofs", "ground_truth_proof", "ground_truth_solutions", "ground_truth_solution"]:
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, list):
            proofs.extend(str(x).strip() for x in value if x is not None and str(x).strip())
        elif str(value).strip():
            proofs.append(str(value).strip())
    return proofs


def load_grading_records(comp):
    with open(f"{args.competition_config_folder}/{comp}.yaml", "r", encoding="utf-8") as f:
        competition_config = yaml.safe_load(f)
    dataset_path = competition_config.get("dataset_path")
    if not dataset_path:
        return {}
    grading_path = os.path.join(dataset_path, "grading_scheme.json")
    if not os.path.exists(grading_path):
        return {}
    with open(grading_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return {int(row["id"]): row for row in rows if isinstance(row, dict) and "id" in row}


def load_competition_config(comp):
    with open(f"{args.competition_config_folder}/{comp}.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _scheme_to_raw_text(scheme):
    if scheme is None:
        return ""
    if isinstance(scheme, (list, dict)):
        return json.dumps(scheme, ensure_ascii=False, indent=4)
    return str(scheme)


def _load_answers_map(dataset_path):
    answers_path = os.path.join(dataset_path, "answers.csv")
    if not os.path.exists(answers_path):
        return {}
    out = {}
    with open(answers_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "id" not in row:
                continue
            out[int(row["id"])] = row.get("answer", "")
    return out


def _load_grading_rows(dataset_path):
    grading_path = os.path.join(dataset_path, "grading_scheme.json")
    if not os.path.exists(grading_path):
        return []
    with open(grading_path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    return rows if isinstance(rows, list) else []


def _save_grading_rows(dataset_path, rows):
    grading_path = os.path.join(dataset_path, "grading_scheme.json")
    with open(grading_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=4)


def _extract_gt_list(row):
    proofs = _extract_ground_truth_proofs(row)
    return proofs if proofs else []


def load_edit_page_data(comp):
    comp_cfg = load_competition_config(comp)
    dataset_path = comp_cfg.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return {"error": f"Local dataset path not found: {dataset_path}", "items": [], "is_final_answer": True}

    is_final_answer = comp_cfg.get("final_answer", True)
    problem_ids = set()
    problems_dir = os.path.join(dataset_path, "problems")
    if os.path.exists(problems_dir):
        for name in os.listdir(problems_dir):
            if name.endswith(".tex") and name[:-4].isdigit():
                problem_ids.add(int(name[:-4]))

    answers_map = _load_answers_map(dataset_path) if is_final_answer else {}
    problem_ids.update(answers_map.keys())

    grading_rows = _load_grading_rows(dataset_path) if not is_final_answer else []
    grading_map = {}
    for row in grading_rows:
        if isinstance(row, dict) and "id" in row:
            pid = int(row["id"])
            grading_map[pid] = row
            problem_ids.add(pid)

    items = []
    for pid in sorted(problem_ids):
        problem_path = os.path.join(problems_dir, f"{pid}.tex")
        problem_text = ""
        if os.path.exists(problem_path):
            with open(problem_path, "r", encoding="utf-8") as f:
                problem_text = f.read()

        row = grading_map.get(pid, {})
        scheme = row.get("scheme", row.get("grading_scheme"))
        items.append(
            {
                "id": pid,
                "problem_text": problem_text,
                "final_answer": answers_map.get(pid, "") if is_final_answer else None,
                "grading_scheme_display": _format_grading_scheme_text(scheme) if not is_final_answer else None,
                "grading_scheme_raw": _scheme_to_raw_text(scheme) if not is_final_answer else None,
                "ground_truth_proofs": _extract_gt_list(row) if not is_final_answer else [],
            }
        )

    return {"error": None, "items": items, "is_final_answer": is_final_answer, "dataset_path": dataset_path}


def _save_problem_text(dataset_path, problem_id, value):
    problems_dir = os.path.join(dataset_path, "problems")
    os.makedirs(problems_dir, exist_ok=True)
    path = os.path.join(problems_dir, f"{problem_id}.tex")
    with open(path, "w", encoding="utf-8") as f:
        f.write(value)


def _save_final_answer(dataset_path, problem_id, value):
    answers_path = os.path.join(dataset_path, "answers.csv")
    if not os.path.exists(answers_path):
        rows = [{"id": str(problem_id), "answer": value}]
        fieldnames = ["id", "answer"]
    else:
        with open(answers_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fieldnames = reader.fieldnames or ["id", "answer"]
        if "id" not in fieldnames:
            fieldnames = ["id"] + [x for x in fieldnames if x != "id"]
        if "answer" not in fieldnames:
            fieldnames.append("answer")
        found = False
        for row in rows:
            if str(row.get("id", "")) == str(problem_id):
                row["answer"] = value
                found = True
                break
        if not found:
            rows.append({"id": str(problem_id), "answer": value})

    with open(answers_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _find_or_create_grading_row(rows, problem_id):
    for row in rows:
        if isinstance(row, dict) and str(row.get("id", "")) == str(problem_id):
            return row
    row = {"id": str(problem_id), "points": 7, "scheme": "", "ground_truth_proofs": []}
    rows.append(row)
    return row


# Analyze run
results = analyze_run(current_comp, args.models)

# Get problem names
sources = load_sources(current_comp)
grading_records = load_grading_records(current_comp)

app = Flask(__name__)
# The viewer caches one competition in globals. Keep selection and use together,
# including writes, so requests from different tabs cannot mix competitions.
competition_lock = RLock()


def select_competition(comp):
    global current_comp, results, sources, grading_records
    if comp not in all_comps and comp != current_comp:
        abort(404, description="Competition not found.")
    results = analyze_run(comp, args.models)
    sources = load_sources(comp)
    grading_records = load_grading_records(comp)
    current_comp = comp


@app.before_request
def restore_competition():
    if request.endpoint in (None, "static", "data_files"):
        return
    competition_lock.acquire()
    g.competition_locked = True
    if request.endpoint == "refresh":
        return
    comp = request.args.get("comp", current_comp)
    if comp != current_comp:
        select_competition(comp)
    model = request.view_args.get("model")
    if model is not None:
        if model not in results:
            abort(404, description="Model not found in this competition. Select the competition again from Home.")
        problem = request.view_args.get("problem_name")
        if problem is not None and (not problem.isdigit() or int(problem) not in results[model]):
            abort(404, description="Problem not found in this competition.")


@app.teardown_request
def release_competition(_error):
    if g.pop("competition_locked", False):
        competition_lock.release()


@app.url_defaults
def retain_competition(endpoint, values):
    if endpoint not in ("static", "data_files", "refresh"):
        values.setdefault("comp", current_comp)


def get_problem_stats(results, model, problem):
    if type(problem) == str:
        problem = int(problem)
    res = results[model][problem]
    corrects = res["correct"]
    warnings = res.get("warnings", [0] * len(corrects))
    llm_annotations = res.get("llm_annotation", [None] * len(corrects))
    manual_overwrite = res.get("manual_overwrite", [False] * len(corrects))
    if not isinstance(llm_annotations, list):
        llm_annotations = [None] * len(corrects)
    if not isinstance(manual_overwrite, list):
        manual_overwrite = [False] * len(corrects)
    if len(llm_annotations) < len(corrects):
        llm_annotations = llm_annotations + [None] * (len(corrects) - len(llm_annotations))
    elif len(llm_annotations) > len(corrects):
        llm_annotations = llm_annotations[: len(corrects)]
    if len(manual_overwrite) < len(corrects):
        manual_overwrite = manual_overwrite + [False] * (len(corrects) - len(manual_overwrite))
    elif len(manual_overwrite) > len(corrects):
        manual_overwrite = manual_overwrite[: len(corrects)]
    llm_annotations = [
        None if manual_overwrite[i] else llm_annotations[i] for i in range(len(corrects))
    ]
    graded = [value for value in corrects if isinstance(value, (int, float)) and math.isfinite(value)]
    return {
        "nb_instances": len(corrects),
        "corrects": corrects,
        "accuracy": sum(graded) / len(graded) if graded else None,
        "warnings": warnings,
        "llm_annotations": llm_annotations,
        "manual_overwrite": manual_overwrite,
    }


def get_tick(is_correct, warning, llm_annotation=None):
    if not isinstance(is_correct, (int, float)) or not math.isfinite(is_correct):
        return "❔"
    if isinstance(is_correct, bool):
        pass
    elif isinstance(is_correct, (int, float)):
        if is_correct >= 0.999:
            return "✅"
        if is_correct <= 0.001:
            return "❌"
        return "🟨"

    if llm_annotation is True:
        tick = "🤖"
    elif warning >= 3:
        tick = "💀"
    elif warning >= 2:
        tick = "⚠️"
    elif is_correct:
        tick = "✅"
    elif not is_correct and warning == 0:
        tick = "❌"
    else:
        # small warning
        tick = "❕"
    return tick


def get_problem_ticks(results, model, problem):
    stat = get_problem_stats(results, model, problem)
    ticks = ""
    for i, correct in enumerate(stat["corrects"]):
        ticks += get_tick(correct, stat["warnings"][i], stat["llm_annotations"][i])
    return ticks


TOKEN_FIELDS = ("input_tokens", "cached_input_tokens", "output_tokens")


@app.template_filter("token_count")
def format_token_count(value):
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return f"{value:,.0f}"


def get_token_totals(record, runs):
    totals = {}
    for field in TOKEN_FIELDS:
        value = record.get("cost", {}).get(field)
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            details = record.get("detailed_costs", [])
            values = [cost.get(field) for cost in details if isinstance(cost, dict)]
            value = sum(values) if runs and len(values) == runs and all(
                isinstance(item, (int, float)) and math.isfinite(item) for item in values
            ) else None
        totals[field] = value
    return totals


def tokens_per_run(totals, runs):
    return {field: total / runs if total is not None and runs else None for field, total in totals.items()}


def get_model_stats(results, model):
    records = results[model]
    problems = {problem: get_problem_stats(results, model, problem) for problem in records}
    for problem, stat in problems.items():
        stat["total_cost"] = records[problem]["cost"]["cost"]
        stat["token_totals"] = get_token_totals(records[problem], stat["nb_instances"])
        stat["cost_estimated"] = any(
            cost.get("token_usage_recovery", {}).get("estimated", False)
            for cost in records[problem].get("detailed_costs", [])
        )
    graded = [
        value for stat in problems.values() for value in stat["corrects"]
        if isinstance(value, (int, float)) and math.isfinite(value)
    ]
    accuracies = [stat["accuracy"] for stat in problems.values() if stat["accuracy"] is not None]
    runs = sum(stat["nb_instances"] for stat in problems.values())
    total_cost = sum(stat["total_cost"] for stat in problems.values())
    token_totals = {}
    for field in TOKEN_FIELDS:
        values = [stat["token_totals"][field] for stat in problems.values() if stat["nb_instances"]]
        token_totals[field] = sum(values) if values and all(value is not None for value in values) else None
    return {
        "tokens_per_run": tokens_per_run(token_totals, runs),
        "problem_stats": problems,
        "nb_problems": len(problems),
        "n_solutions": runs,
        "n_correct": sum(graded),
        "n_graded": len(graded),
        "n_pending": runs - len(graded),
        "avg_accuracy": sum(accuracies) / len(accuracies) if accuracies else None,
        "total_cost": total_cost,
        "cost_per_run": total_cost / runs if runs else None,
        "cost_estimated": any(stat["cost_estimated"] for stat in problems.values()),
    }


def model_stats_to_html(stats):
    def accuracy_fields(value):
        return {
            "avg_accuracy": f"{value * 100:.2f}%" if value is not None else "—",
            "accuracy_value": value * 100 if value is not None else None,
        }

    problem_rows = []
    for problem, stat in sorted(stats["problem_stats"].items()):
        pending = sum(not isinstance(value, (int, float)) or not math.isfinite(value) for value in stat["corrects"])
        problem_rows.append({
            "id": problem,
            "source": sources.get(problem),
            **accuracy_fields(stat["accuracy"]),
            "n_solutions": stat["nb_instances"],
            "n_pending": pending,
            "ticks": "".join(
                get_tick(correct, stat["warnings"][i], stat["llm_annotations"][i])
                for i, correct in enumerate(stat["corrects"])
            ),
            "total_cost": f"{stat['total_cost']:,.2f}",
            "cost_estimated": stat["cost_estimated"],
            "cost_per_run": f"{stat['total_cost'] / stat['nb_instances']:,.2f}" if stat["nb_instances"] else None,
            "tokens_per_run": tokens_per_run(stat["token_totals"], stat["nb_instances"]),
        })
    return {
        **stats,
        **accuracy_fields(stats["avg_accuracy"]),
        "cost_value": stats["total_cost"],
        "total_cost": f"{stats['total_cost']:,.2f}",
        "cost_per_run_value": stats["cost_per_run"],
        "cost_per_run": f"{stats['cost_per_run']:,.2f}" if stats["cost_per_run"] is not None else None,
        "problem_rows": problem_rows,
    }


###### results


@app.route("/refresh/<comp>", defaults={"url": ""})
@app.route("/refresh/<comp>/<path:url>")
def refresh(comp, url):
    select_competition(comp.replace("---", "/"))
    print("Refreshed!")
    if not url:
        return redirect(url_for("index"))
    if ">>>" in url:
        model, problem_name = url.split(">>>", 1)
        return redirect(url_for("problem_view", model=model, problem_name=problem_name))
    return redirect(url_for("model_view", model=url))


@app.route("/")
def index():
    sidebar_contents = {
        "dropdown": {"all_comps": all_comps, "current_comp": current_comp},
        "reload_url": f"/refresh/{current_comp.replace('/', '---')}",
        "models": list(results.keys()),
    }

    stats_html = {}
    for model in results.keys():
        stats = get_model_stats(results, model)
        stats_html[model] = model_stats_to_html(stats)

    stats_html = dict(sorted(
        stats_html.items(),
        key=lambda item: (
            -(item[1]["accuracy_value"] if item[1]["accuracy_value"] is not None else -1),
            item[0].casefold(),
        ),
    ))
    previous_accuracy = None
    rank = None
    for position, model_stats in enumerate(stats_html.values(), start=1):
        accuracy = model_stats["accuracy_value"]
        if accuracy is not None and accuracy != previous_accuracy:
            rank = position
        model_stats["rank"] = rank if accuracy is not None else None
        previous_accuracy = accuracy
    sidebar_contents["models"] = list(stats_html)
    return render_template("index.html", title="MathArena", sidebar=sidebar_contents, stats=stats_html)


@app.route("/view/<model>")
def model_view(model):
    sidebar_contents = {
        "dropdown": {"all_comps": all_comps, "current_comp": current_comp},
        "reload_url": f"/refresh/{current_comp.replace('/', '---')}/{model}",
        "current_model": model,
        "problems": {},
    }

    for problem in sorted(results[model].keys(), key=lambda x: int(x)):
        ticks = get_problem_ticks(results, model, problem)
        problem_full_name = f"{problem}"
        if sources.get(problem):
            problem_full_name = f"{problem} ({sources.get(problem)})"
        sidebar_contents["problems"][problem] = {"name": problem_full_name, "ticks": ticks}

    stats = get_model_stats(results, model)
    stats_html = model_stats_to_html(stats)

    return render_template(
        "model.html",
        title="MathArena App",
        sidebar=sidebar_contents,
        model=model,
        stats=stats_html,
    )


def render_message(message):
    role = message["role"]
    tagline = ""
    content = ""
    code = None
    is_cot = False

    if role == "developer":
        tagline = "System Prompt / Developer Message"
    elif role == "user":
        tagline = "User"
    elif role == "tool_response":
        tool_name = message["tool_name"]
        tool_call_id = message["tool_call_id"]
        tagline = f"Response from Tool {tool_name} (Tool Call ID: {tool_call_id})"
    elif role == "assistant":
        typ = message.get("type")
        if typ == "cot":
            tagline = "Assistant (Chain-of-Thought)"
            is_cot = True
        elif typ == "response":
            tagline = "Assistant"
        elif typ == "tool_call":
            tool_name = message["tool_name"]
            tool_call_id = message["tool_call_id"]
            tagline = f"Assistant (Tool Call to {tool_name}, Tool Call ID: {tool_call_id})"
            arguments = message.get("arguments", {})
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            
            formatted_args = ""
            for k, v in arguments.items():
                if k == "code":
                    code = v
                else:
                    formatted_args += f"### {k}\n{v}\n\n"
            
            if formatted_args:
                content = formatted_args.strip()
        elif typ == "internal_tool_call":
            tool_name = message["tool_name"]
            assert tool_name == "code_interpreter"
            tagline = f"Assistant (Internal Tool Call to {tool_name})"
            code = message.get("code", None)
        else:
            raise ValueError(f"Unknown assistant type: {typ}")
    else:
        raise ValueError(f"Unknown role: {role}")

    if code is not None:
        code = code.replace("```python", "").replace("```", "").strip()
    
    if not content:
        content = message.get("content", "")

    if isinstance(content, list):
        text, img = None, None
        for c in content:
            if c["type"] in ["text", "input_text"]:
                text = c["text"]
            elif c["type"] == "input_image":
                img = c["image_url"]
            elif c["type"] == "image_url":
                img = c["image_url"]["url"]
        content = {"text": text, "img": img}
    else:
        content = {"text": str(content).strip(), "img": None}
        
    return {"tagline": tagline, "content": content, "code": code, "role": role, "is_cot": is_cot}


def final_response_index(conversation):
    return next((
        i for i in range(len(conversation) - 1, -1, -1)
        if conversation[i].get("role") == "assistant" and conversation[i].get("type") == "response"
    ), None)


def render_conversation_html(conversation):
    fragments = []
    cot_parts = []

    def flush_cot():
        if cot_parts:
            fragments.append(render_template("message.html", **cot_parts[0], cot_parts=cot_parts))
            cot_parts.clear()

    for i, message in enumerate(conversation):
        msg_data = render_message(message)
        if not msg_data["is_cot"]:
            flush_cot()
        if (
            message.get("role") == "assistant"
            and not msg_data["content"]["text"]
            and not msg_data["content"]["img"]
            and not msg_data["code"]
            and i != len(conversation) - 1
        ):
            continue
        if msg_data["is_cot"]:
            cot_parts.append(msg_data)
        else:
            fragments.append(render_template("message.html", **msg_data))
    flush_cot()
    return "".join(fragments)


@app.route("/modelinteraction/<id>")
def model_interaction(id):
    tokens = id.split(">>")
    if len(tokens) == 4:
        model, problem_name, i, extra = tokens
        assert extra == "history"
        history = get_problem_result(model, problem_name)["history"][int(i)]

        options = ""
        for step in history:
            stp = step["step"]
            timestep = step["timestep"]
            options += f"<option value=\"{model}>>{problem_name}>>{i}>>{timestep}>>{stp}\">TIME={timestep} 🕐 {stp}</option>"

        return f"""
            <select id="history-step-selector-{i}" onchange="loadHistoryStep(this.value)" class="history-step-selector">
                <option value="">Select a step...</option>
                {options}
            </select>
            <div id="history-step-content-{i}" style="margin-top: 1rem;"></div>
        """
    else:
        model, problem_name, i = tokens
        conversation = get_problem_result(model, problem_name)["messages"][int(i)]
        if request.args.get("trace_only") == "1":
            final_idx = final_response_index(conversation)
            conversation = [message for idx, message in enumerate(conversation) if idx != final_idx]
        return render_conversation_html(conversation)


@app.route("/historystep/<id>")
def history_step(id):
    tokens = id.split(">>")
    model, problem_name, i, timestep_str, step_name = tokens
    timestep = int(timestep_str)
    history = get_problem_result(model, problem_name)["history"][int(i)]

    target_step = None
    for step in history:
        if step["timestep"] == timestep and step["step"] == step_name:
            target_step = step
            break

    if target_step is None:
        return "<div class=\"error\">Step not found</div>"

    return render_conversation_html(target_step["messages"])

@app.route('/data/<path:filename>')
def data_files(filename):
    print(os.path.join(app.root_path, 'data'))
    return send_from_directory(os.path.join(os.path.dirname(app.root_path), 'data'), filename)


def get_instance_metadata(history_run):
    tool_calls = {}
    if not history_run:
        return {"tool_calls": tool_calls}

    for step in history_run:
        for message in step.get("messages", []):
            if "tool_call" in message.get("type", ""):
                tool_name = message.get("tool_name")
                if tool_name:
                    tool_calls[tool_name] = tool_calls.get(tool_name, 0) + 1
    return {"tool_calls": tool_calls}


def _format_score_value(value):
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else round(float(value), 4)
    return value


def _judgment_rows(raw_judgment):
    if not isinstance(raw_judgment, list):
        return [], False
    legacy_shape = bool(raw_judgment) and all(item is None or isinstance(item, dict) for item in raw_judgment)
    return ([raw_judgment] if legacy_shape else raw_judgment), legacy_shape


def _judgment_entry_to_text(entry):
    blocks = []
    if entry.get("error"):
        blocks.append(str(entry["error"]).strip())

    for detail in entry.get("details", []):
        desc = str(detail.get("desc", "")).strip()
        if desc:
            blocks.append(desc)

    return "\n\n".join(block for block in blocks if block).strip()


def _format_additional_info_value(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2).strip()
    return str(value).strip()


def _judgment_additional_info(entry):
    raw_info = entry.get("additional_info") or {}
    if isinstance(raw_info, dict):
        items = raw_info.items()
    else:
        items = [("value", raw_info)]

    return [
        {
            "key": str(key),
            "value": _format_additional_info_value(value),
        }
        for key, value in items
        if _format_additional_info_value(value)
    ]


def _judgment_without_cost(entry):
    entry = copy.deepcopy(entry)
    entry.pop("cost", None)
    return entry


def get_run_judgments(res, run_idx):
    judgment_rows, _ = _judgment_rows(res.get("judgment", []))
    return [
        {
            "judge_idx": judge_idx,
            "points": _format_score_value(entry.get("points")),
            "max_points": _format_score_value(entry.get("max_points")),
            "details": [
                {
                    **detail,
                    "points": _format_score_value(detail.get("points")) if "points" in detail else detail.get("points"),
                    "max_points": _format_score_value(detail.get("max_points")) if "max_points" in detail else detail.get("max_points"),
                }
                for detail in entry.get("details", [])
            ],
            "error": entry.get("error"),
            "editable_text": _judgment_entry_to_text(entry),
            "additional_info": _judgment_additional_info(entry),
            "manual_overwrite": "original_judgment" in entry,
            "original_judgment": entry.get("original_judgment"),
        }
        for judge_idx, judge_runs in enumerate(judgment_rows, start=1)
        # Runs can be appended before the corresponding judgments are ready.
        if isinstance(judge_runs, list) and 0 <= run_idx < len(judge_runs)
        for entry in [judge_runs[run_idx]]
        if isinstance(entry, dict)
    ]


@app.route("/view/<model>/<problem_name>")
def problem_view(model, problem_name):
    sidebar_contents = {
        "dropdown": {"all_comps": all_comps, "current_comp": current_comp},
        "reload_url": f"/refresh/{current_comp.replace('/', '---')}/{model}>>>{problem_name}",
        "current_model": model,
        "problems": {},
    }

    for problem_name_for in sorted(results[model].keys(), key=lambda x: int(x)):
        ticks = get_problem_ticks(results, model, problem_name_for)
        cls = "current" if str(problem_name_for) == problem_name else ""
        problem_full_name = f"{problem_name_for}"
        if sources.get(problem_name_for):
            problem_full_name = f"{problem_name_for} ({sources.get(problem_name_for)})"
        sidebar_contents["problems"][problem_name_for] = {"name": problem_full_name, "ticks": ticks, "class": cls}

    res = get_problem_result(model, problem_name)
    grading_record = grading_records.get(int(problem_name), {})
    problem_statement = res["problem"]
    img_path = f"/data/{current_comp}/problems/{problem_name}.png"
    if not os.path.exists(img_path[1:]):
        img_path = None

    is_final_answer = res.get("gold_answer", None) is not None
    solution = res["gold_answer"] if is_final_answer else None
    grading_scheme_text = _format_grading_scheme_text(grading_record.get("scheme", grading_record.get("grading_scheme")))
    ground_truth_proofs = _extract_ground_truth_proofs(grading_record)
    instances = []
    llm_annotations = res.get("llm_annotation", [None] * len(res["correct"]))
    manual_overwrite = res.get("manual_overwrite", [False] * len(res["correct"]))
    if not isinstance(llm_annotations, list):
        llm_annotations = [None] * len(res["correct"])
    if not isinstance(manual_overwrite, list):
        manual_overwrite = [False] * len(res["correct"])
    if len(llm_annotations) < len(res["correct"]):
        llm_annotations = llm_annotations + [None] * (len(res["correct"]) - len(llm_annotations))
    elif len(llm_annotations) > len(res["correct"]):
        llm_annotations = llm_annotations[: len(res["correct"])]
    if len(manual_overwrite) < len(res["correct"]):
        manual_overwrite = manual_overwrite + [False] * (len(res["correct"]) - len(manual_overwrite))
    elif len(manual_overwrite) > len(res["correct"]):
        manual_overwrite = manual_overwrite[: len(res["correct"])]
    llm_annotations = [
        None if manual_overwrite[i] else llm_annotations[i] for i in range(len(res["correct"]))
    ]
    for i, messages in enumerate(res["messages"]):
        answer, is_correct = res["answers"][i], res["correct"][i]
        warning = res["warnings"][i] if "warnings" in res else 0
        verdict = get_tick(is_correct, warning, llm_annotations[i])
        if verdict == "❔":
            correct_cls = "pending"
        elif 0.001 < is_correct < 0.999:
            correct_cls = "semicorrect"
        else:
            correct_cls = "correct" if is_correct >= 0.999 else "incorrect"
        history = res.get("history", [])
        run_history = history[i] if i < len(history) else None
        manual_value = "auto"
        if manual_overwrite[i]:
            manual_value = "correct" if is_correct else "incorrect"
        
        metadata = get_instance_metadata(run_history if run_history is not None else [{"messages": messages}])
        metadata['cost'] = res["detailed_costs"][i].copy()
        metadata["cost"]["cost"] = round(metadata["cost"]["cost"], 4)
        metadata["cost"]["time"] = round(metadata["cost"]["time"], 0) if metadata["cost"].get("time") is not None else None
        final_idx = final_response_index(messages)
        final_response = render_template(
            "message.html", **render_message(messages[final_idx]), show_tagline=False,
        ) if final_idx is not None else None
        instances.append(
            {
                "final_response": final_response,
                "run": i,
                "verdict": verdict,
                "warnings": warning,
                "answer": answer or "No answer found in \\boxed{}. Model was instructed to output answer in \\boxed{}.",
                "correct_cls": correct_cls,
                "history": run_history,
                "conversation_id": f"{model}>>{problem_name}>>{i}",
                "metadata": metadata,
                "manual_value": manual_value,
                "judgments": get_run_judgments(res, i),
            }
        )

    problem_full_name = f"{problem_name}"
    if sources.get(int(problem_name)):
        problem_full_name = f"{problem_name} ({sources.get(int(problem_name))})"
    ticks = get_problem_ticks(results, model, problem_name)

    return render_template(
        "problem.html",
        title="MathArena App",
        sidebar=sidebar_contents,
        model=model,
        problem_name=problem_full_name,
        problem_id=problem_name,
        ticks=ticks,
        problem_statement=problem_statement,
        img_path=img_path,
        solution=solution,
        grading_scheme_text=grading_scheme_text,
        ground_truth_proofs=ground_truth_proofs,
        instances=instances,
        is_final_answer=is_final_answer,
        is_checkpoint=False,
        overwrite_enabled=not args.disable_overwrite,
    )


@app.route("/override/<model>/<problem_name>/<int:run_idx>", methods=["POST"])
def override_result(model, problem_name, run_idx):
    global results
    if args.disable_overwrite:
        abort(403)
    action = request.form.get("manual_correct", "auto")
    config_path = model_id_to_config_path.get(model)
    if config_path is None:
        return redirect(url_for("problem_view", model=model, problem_name=problem_name))

    json_path = os.path.join(args.output_folder, current_comp, config_path, f"{problem_name}{OUTPUT_JSON_SUFFIX}")
    if not os.path.exists(json_path):
        return redirect(url_for("problem_view", model=model, problem_name=problem_name))

    data = load_json_zst(json_path)

    correct = data.get("correct", [])
    if not isinstance(correct, list) or run_idx >= len(correct):
        return redirect(url_for("problem_view", model=model, problem_name=problem_name))

    manual_overwrite = data.get("manual_overwrite", [False] * len(correct))
    if not isinstance(manual_overwrite, list):
        manual_overwrite = [False] * len(correct)
    if len(manual_overwrite) < len(correct):
        manual_overwrite = manual_overwrite + [False] * (len(correct) - len(manual_overwrite))
    elif len(manual_overwrite) > len(correct):
        manual_overwrite = manual_overwrite[: len(correct)]

    llm_annotation = data.get("llm_annotation", [None] * len(correct))
    if not isinstance(llm_annotation, list):
        llm_annotation = [None] * len(correct)
    if len(llm_annotation) < len(correct):
        llm_annotation = llm_annotation + [None] * (len(correct) - len(llm_annotation))
    elif len(llm_annotation) > len(correct):
        llm_annotation = llm_annotation[: len(correct)]

    if action in ["correct", "incorrect"]:
        correct[run_idx] = action == "correct"
        manual_overwrite[run_idx] = True
        llm_annotation[run_idx] = None
    else:
        manual_overwrite[run_idx] = False

    data["correct"] = correct
    data["manual_overwrite"] = manual_overwrite
    data["llm_annotation"] = llm_annotation
    try:
        data["pass_at_1"] = sum(correct) / len(correct) if correct else 0
    except Exception:
        data["pass_at_1"] = data.get("pass_at_1")

    dump_json_zst(data, json_path, ensure_ascii=False, indent=4)

    if model in results and int(problem_name) in results[model]:
        results[model][int(problem_name)]["correct"] = correct
        results[model][int(problem_name)]["manual_overwrite"] = manual_overwrite
        results[model][int(problem_name)]["llm_annotation"] = llm_annotation
        results[model][int(problem_name)]["pass_at_1"] = data["pass_at_1"]

    return redirect(url_for("problem_view", model=model, problem_name=problem_name))


def _parse_manual_points(raw_value, fallback):
    if raw_value is None or str(raw_value).strip() == "":
        return fallback
    value = float(str(raw_value).strip())
    return int(value) if value.is_integer() else value


def _build_manual_judgment_details(source_entry, manual_points, max_points, manual_desc):
    source_details = source_entry.get("details")
    source_detail = source_details[0] if isinstance(source_details, list) and source_details else {}
    if not isinstance(source_detail, dict):
        source_detail = {}

    detail = copy.deepcopy(source_detail)
    detail["points"] = manual_points
    detail["max_points"] = max_points
    detail["desc"] = manual_desc
    return [detail]


def _normalize_run_field(values, default_value, length):
    if not isinstance(values, list):
        values = [default_value] * length
    if len(values) < length:
        values = values + [default_value] * (length - len(values))
    elif len(values) > length:
        values = values[:length]
    return values


def _as_float(value):
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _sync_manual_judgment_override(data, run_idx, judgment_entry):
    correct = data.get("correct", [])
    if not isinstance(correct, list) or run_idx >= len(correct):
        return

    manual_overwrite = _normalize_run_field(data.get("manual_overwrite"), False, len(correct))

    points = _as_float(judgment_entry.get("points"))
    max_points = _as_float(judgment_entry.get("max_points"))
    if points is not None and max_points not in (None, 0):
        correct[run_idx] = min(max(points / max_points, 0.0), 1.0)

    manual_overwrite[run_idx] = True

    data["correct"] = correct
    data["manual_overwrite"] = manual_overwrite
    try:
        data["pass_at_1"] = sum(correct) / len(correct) if correct else 0
    except Exception:
        data["pass_at_1"] = data.get("pass_at_1")


@app.route("/override_judgment/<model>/<problem_name>/<int:run_idx>/<int:judge_idx>", methods=["POST"])
def override_judgment(model, problem_name, run_idx, judge_idx):
    global results
    if args.disable_overwrite:
        abort(403)

    config_path = model_id_to_config_path[model]
    json_path = os.path.join(args.output_folder, current_comp, config_path, f"{problem_name}{OUTPUT_JSON_SUFFIX}")
    data = load_json_zst(json_path)

    judgment_rows, legacy_shape = _judgment_rows(data.get("judgment", []))
    judge_pos = judge_idx - 1
    current_entry = judgment_rows[judge_pos][run_idx]

    original_entry = current_entry.get("original_judgment")
    source_entry = original_entry or current_entry
    max_points = source_entry.get("max_points", current_entry.get("max_points", 7))
    fallback_points = current_entry.get("points", source_entry.get("points"))
    manual_points = _parse_manual_points(request.form.get("points"), fallback_points)
    manual_desc = request.form.get("desc", "")

    updated_entry = {
        "judge_id": source_entry.get("judge_id", f"judges/manual_{judge_idx}"),
        "points": manual_points,
        "max_points": max_points,
        "details": _build_manual_judgment_details(source_entry, manual_points, max_points, manual_desc),
        "error": None,
        "cost": current_entry.get("cost"),
        "additional_info": source_entry.get("additional_info", current_entry.get("additional_info", {})),
    }
    if original_entry is None:
        updated_entry["original_judgment"] = _judgment_without_cost(current_entry)
    else:
        updated_entry["original_judgment"] = original_entry
    judgment_rows[judge_pos][run_idx] = updated_entry

    data["judgment"] = judgment_rows[0] if legacy_shape else judgment_rows
    data.pop("judgment_original", None)
    data.pop("judgment_manual_overwrite", None)
    _sync_manual_judgment_override(data, run_idx, updated_entry)

    dump_json_zst(data, json_path, ensure_ascii=False, indent=4)

    if model in results and int(problem_name) in results[model]:
        results[model][int(problem_name)]["judgment"] = data["judgment"]
        results[model][int(problem_name)]["correct"] = data.get("correct", results[model][int(problem_name)].get("correct"))
        results[model][int(problem_name)]["manual_overwrite"] = data.get(
            "manual_overwrite", results[model][int(problem_name)].get("manual_overwrite")
        )
        results[model][int(problem_name)]["pass_at_1"] = data.get("pass_at_1", results[model][int(problem_name)].get("pass_at_1"))
        results[model][int(problem_name)].pop("judgment_original", None)
        results[model][int(problem_name)].pop("judgment_manual_overwrite", None)

    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        saved_entry = judgment_rows[judge_pos][run_idx]
        return jsonify(
            {
                "ok": True,
                "points": _format_score_value(saved_entry.get("points")),
                "max_points": _format_score_value(saved_entry.get("max_points")),
                "desc": _judgment_entry_to_text(saved_entry),
                "manual_overwrite": "original_judgment" in saved_entry,
            }
        )

    return redirect(url_for("problem_view", model=model, problem_name=problem_name))


@app.route("/edit")
def edit_view():
    if not args.edit:
        abort(404)
    data = load_edit_page_data(current_comp)
    page = max(request.args.get("page", 1, type=int), 1)
    per_page = 10
    items = data["items"]
    total_pages = max(1, math.ceil(len(items) / per_page)) if len(items) > 0 else 1
    page = min(page, total_pages)
    start = (page - 1) * per_page
    end = start + per_page
    page_items = items[start:end]
    return render_template(
        "edit.html",
        title="MathArena Edit",
        current_comp=current_comp,
        page=page,
        per_page=per_page,
        total_pages=total_pages,
        has_prev=page > 1,
        has_next=page < total_pages,
        prev_page=page - 1,
        next_page=page + 1,
        is_final_answer=data["is_final_answer"],
        items=page_items,
        error_message=data["error"],
    )


@app.route("/edit/save", methods=["POST"])
def edit_save():
    if not args.edit or args.disable_overwrite:
        abort(403)
    payload = request.get_json(silent=True) or {}
    field = payload.get("field")
    value = payload.get("value", "")
    problem_id = payload.get("problem_id")
    proof_idx = payload.get("proof_idx", None)

    if field is None or problem_id is None:
        return jsonify({"ok": False, "error": "Missing field/problem_id"}), 400

    try:
        problem_id = int(problem_id)
    except Exception:
        return jsonify({"ok": False, "error": "Invalid problem_id"}), 400

    comp_cfg = load_competition_config(current_comp)
    dataset_path = comp_cfg.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        return jsonify({"ok": False, "error": f"Local dataset path not found: {dataset_path}"}), 400

    if field == "problem":
        _save_problem_text(dataset_path, problem_id, value)
    elif field == "final_answer":
        _save_final_answer(dataset_path, problem_id, value)
    elif field == "grading_scheme":
        rows = _load_grading_rows(dataset_path)
        row = _find_or_create_grading_row(rows, problem_id)
        stripped = value.strip()
        parsed = None
        if stripped.startswith("[") or stripped.startswith("{"):
            try:
                parsed = json.loads(value)
            except Exception:
                parsed = None
        row["scheme"] = parsed if parsed is not None else value
        _save_grading_rows(dataset_path, rows)
    elif field == "ground_truth_proof":
        if proof_idx is None:
            return jsonify({"ok": False, "error": "Missing proof_idx"}), 400
        try:
            proof_idx = int(proof_idx)
        except Exception:
            return jsonify({"ok": False, "error": "Invalid proof_idx"}), 400
        rows = _load_grading_rows(dataset_path)
        row = _find_or_create_grading_row(rows, problem_id)
        proofs = row.get("ground_truth_proofs", [])
        if not isinstance(proofs, list):
            proofs = _extract_ground_truth_proofs(row)
        while len(proofs) <= proof_idx:
            proofs.append("")
        proofs[proof_idx] = value
        row["ground_truth_proofs"] = proofs
        _save_grading_rows(dataset_path, rows)
    else:
        return jsonify({"ok": False, "error": f"Unsupported field: {field}"}), 400

    global sources, grading_records
    sources = load_sources(current_comp)
    grading_records = load_grading_records(current_comp)
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(debug=not args.disable_debug, port=args.port)
