
import argparse
import csv
import json
import os
import math

import yaml
from flask import Flask, redirect, render_template, url_for, send_from_directory, request, abort, jsonify
from pyparsing import srange
from torch import ScriptDict

from matharena.configs import extract_existing_configs
from matharena.utils import normalize_conversation

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

# Find all comps, directories below outputs of depth 2
all_comps = []
for root, dirs, files in os.walk(args.output_folder):
    rel_path = os.path.relpath(root, args.output_folder)
    depth = rel_path.count(os.sep)
    if depth == 1:
        all_comps.append(rel_path)

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

    results = {}
    for config_path in human_readable_ids:
        model_comp_dir = os.path.join(out_dir, config_path)
        results[f"{human_readable_ids[config_path]}"] = {}
        for problem_file in os.listdir(model_comp_dir):
            if not problem_file.endswith(".json"):
                continue
            problem_idx = int(problem_file.split(".")[0])
            with open(os.path.join(model_comp_dir, problem_file), "r", encoding="utf-8") as f:
                data = json.load(f)
                results[f"{human_readable_ids[config_path]}"][problem_idx] = data
    return results

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
boxes_expanded = False

# Get problem names
sources = load_sources(current_comp)
grading_records = load_grading_records(current_comp)

app = Flask(__name__)


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
    if len(corrects) == 0:
        return {
            "nb_instances": 0,
            "corrects": [],
            "accuracy": 0,
            "warnings": [],
            "llm_annotations": [],
            "manual_overwrite": [],
        }
    nb_inst = len(corrects)
    try:
        acc = sum(corrects) / nb_inst
    except Exception:
        acc = 0
    return {
        "nb_instances": nb_inst,
        "corrects": corrects,
        "accuracy": acc,
        "warnings": warnings,
        "llm_annotations": llm_annotations,
        "manual_overwrite": manual_overwrite,
    }


def get_tick(is_correct, warning, llm_annotation=None):
    if isinstance(is_correct, bool):
        pass
    elif isinstance(is_correct, (int, float)):
        if is_correct >= 0.999:
            return "✅"
        if is_correct <= 0.001:
            return "❌"
        return "🟨"
    elif isinstance(is_correct, str):
        return "❔"

    if (not is_correct) and llm_annotation is True:
        tick = "🤖"
    elif is_correct:
        tick = "✅"
    elif not is_correct and warning == 0:
        tick = "❌"
    elif warning >= 3:
        tick = "💀"
    elif warning >= 2:
        tick = "⚠️"
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


def get_model_stats(results, model):
    res = results[model]
    nb_problems = len(res)
    problem_stats = {problem: get_problem_stats(results, model, problem) for problem in res.keys()}
    stats = {"problem_stats": problem_stats.copy()}
    stats["nb_problems"] = len(res)
    stats["n_solutions"] = sum([stat["nb_instances"] for stat in problem_stats.values()])
    try:
        stats["n_correct"] = sum([sum(stat["corrects"]) for stat in problem_stats.values()])
    except Exception:
        stats["n_correct"] = 0
    stats["total_cost"] = sum([res[problem]["cost"]["cost"] for problem in res.keys()])
    if nb_problems == 0:
        stats["avg_accuracy"] = 0
    else:
        stats["avg_accuracy"] = sum([stat["accuracy"] for stat in problem_stats.values()]) / nb_problems
    return stats


def model_stats_to_html(stats):
    problem_stats_html = []
    for problem, stat in sorted(stats["problem_stats"].items(), key=lambda x: x[0]):
        problem_full_name = str(problem)
        if problem in sources:
            problem_full_name += f" ({sources[problem]})"
        p = f"{problem_full_name}:{' '*(30-len(str(problem_full_name)))}"
        p += f"{stat['accuracy']*100:6.2f}% "
        p += f"{stat['nb_instances']} instances: "
        for i, correct in enumerate(stat["corrects"]):
            p += get_tick(correct, stat["warnings"][i], stat["llm_annotations"][i])
        problem_stats_html.append(p)
    return {
        "avg_accuracy": f"{stats['avg_accuracy']*100:.2f}%",
        "nb_problems": stats["nb_problems"],
        "n_solutions": stats["n_solutions"],
        "n_correct": stats["n_correct"] if isinstance(stats["n_correct"], int) else round(stats["n_correct"], 2),
        "total_cost": f"{stats['total_cost']:.2f}",
        "problem_stats": problem_stats_html,
    }


def parse_messages_response(response):
    # This is a list of messages
    response_str = response[0]["content"]
    for i in range(1, len(response)):
        if response[i]["role"] == "assistant":
            response_str += "\n\n" + 30 * "=" + "Assistant" + 30 * "=" + "\n\n" + response[i]["content"]
        else:
            response_str += "\n\n" + 30 * "=" + "User" + 30 * "=" + "\n\n" + response[i]["content"]
    return response_str


def sanitize_response(response):
    response = response.replace("\\( ", "$")
    response = response.replace(" \\)", "$")
    response = response.replace("\\(", "$")
    response = response.replace("\\)", "$")

    response = response.replace("\\[ ", "$$")
    response = response.replace(" \\]", "$$")
    response = response.replace("\\[", "$$")
    response = response.replace("\\]", "$$")
    return response


###### results


@app.route("/expand/<path:url>/<int:expanded>")
def expand(url, expanded):
    global boxes_expanded
    boxes_expanded = bool(expanded)
    if not url:
        return redirect(url_for("index"))
    url = "/view/" + url.replace(">>>", "/")
    return redirect(url)


@app.route("/refresh/<comp>", defaults={"url": ""})
@app.route("/refresh/<comp>/<path:url>")
def refresh(comp, url):
    global current_comp, results, sources, grading_records
    current_comp = comp.replace("---", "/")
    results = analyze_run(current_comp, args.models)
    sources = load_sources(current_comp)
    grading_records = load_grading_records(current_comp)
    print("Refreshed!")
    if not url:
        return redirect(url_for("index"))
    return redirect("/view/" + url.replace(">>>", "/"))


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

    # sort models by avg accuracy
    stats_html = dict(
        sorted(
            stats_html.items(),
            key=lambda x: float(x[1]["avg_accuracy"].replace("%", "")),
            reverse=True,
        )
    )
    # sort models in sidebar_contents
    sidebar_contents["models"] = sorted(
        sidebar_contents["models"],
        key=lambda x: float(stats_html[x]["avg_accuracy"].replace("%", "")),
        reverse=True,
    )

    return render_template("index.html", title="MathArena App", sidebar=sidebar_contents, stats=stats_html)

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
        boxes_expanded=boxes_expanded,
    )


def render_message(message):
    role = message["role"]
    tagline = ""
    content = ""
    code = None

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
        
    return {"tagline": tagline, "content": content, "code": code, "role": role}



@app.route("/modelinteraction/<id>")
def model_interaction(id):
    tokens = id.split(">>")
    if len(tokens) == 4:
        model, problem_name, i, extra = tokens
        assert extra == "history"
        history = results[model][int(problem_name)]["history"][int(i)]

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
        conversation = results[model][int(problem_name)]["messages"][int(i)]
        messages_html = ""
        for i, message in enumerate(conversation):
            is_last_message = i == len(conversation) - 1
            msg_data = render_message(message)

            if (
                message.get("role") == "assistant"
                and not msg_data["content"]["text"]
                and not msg_data["content"]["img"]
                and not msg_data["code"]
                and not is_last_message
            ):
                continue

            messages_html += render_template("message.html", **msg_data)
        return messages_html


@app.route("/historystep/<id>")
def history_step(id):
    tokens = id.split(">>")
    model, problem_name, i, timestep_str, step_name = tokens
    timestep = int(timestep_str)
    history = results[model][int(problem_name)]["history"][int(i)]

    target_step = None
    for step in history:
        if step["timestep"] == timestep and step["step"] == step_name:
            target_step = step
            break

    if target_step is None:
        return "<div class=\"error\">Step not found</div>"

    conversation = target_step["messages"]
    messages_html = ""
    for i, message in enumerate(conversation):
        is_last_message = i == len(conversation) - 1
        msg_data = render_message(message)

        if (
            message.get("role") == "assistant"
            and not msg_data["content"]["text"]
            and not msg_data["content"]["img"]
            and not msg_data["code"]
            and not is_last_message
        ):
            continue

        messages_html += render_template("message.html", **msg_data)
    return messages_html

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


def get_run_judgments(res, run_idx):
    raw_judgment = res.get("judgment", [])
    if not isinstance(raw_judgment, list):
        return []

    # Legacy shape: judgment is a single list of per-run entries.
    if raw_judgment and all(item is None or isinstance(item, dict) for item in raw_judgment):
        raw_judgment = [raw_judgment]

    output = []
    for judge_idx, judge_runs in enumerate(raw_judgment, start=1):
        if not isinstance(judge_runs, list) or run_idx >= len(judge_runs):
            continue
        entry = judge_runs[run_idx]
        if not isinstance(entry, dict):
            continue
        details = []
        for detail in entry.get("details", []):
            if not isinstance(detail, dict):
                continue
            detail_copy = detail.copy()
            if "points" in detail_copy:
                detail_copy["points"] = _format_score_value(detail_copy["points"])
            if "max_points" in detail_copy:
                detail_copy["max_points"] = _format_score_value(detail_copy["max_points"])
            details.append(detail_copy)
        output.append(
            {
                "judge_idx": judge_idx,
                "points": _format_score_value(entry.get("points")),
                "max_points": _format_score_value(entry.get("max_points")),
                "details": details,
                "error": entry.get("error"),
            }
        )
    return output


@app.route("/view/<model>/<problem_name>")
def problem_view(model, problem_name):
    sidebar_contents = {
        "dropdown": {"all_comps": all_comps, "current_comp": current_comp},
        "toggle_text": "Make Boxes Scrollable" if boxes_expanded else "Make Boxes Very Tall",
        "expand_url": f"/expand/{model}>>>{problem_name}/{int(not boxes_expanded)}",
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

    res = results[model][int(problem_name)]
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
        correct_cls = "correct" if is_correct else "incorrect"
        history = results[model][int(problem_name)].get("history", [])
        run_history = history[i] if i < len(history) else None
        manual_value = "auto"
        if manual_overwrite[i]:
            manual_value = "correct" if is_correct else "incorrect"
        
        metadata = get_instance_metadata(run_history if run_history is not None else [{"messages": messages}])
        metadata['cost'] = res["detailed_costs"][i]
        metadata["cost"]["cost"] = round(metadata["cost"]["cost"], 4)
        metadata["cost"]["time"] = round(metadata["cost"]["time"], 0) if metadata["cost"].get("time") is not None else None
        instances.append(
            {
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
        boxes_expanded=boxes_expanded,
        is_final_answer=is_final_answer,
        is_checkpoint=False,
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

    json_path = os.path.join(args.output_folder, current_comp, config_path, f"{problem_name}.json")
    if not os.path.exists(json_path):
        return redirect(url_for("problem_view", model=model, problem_name=problem_name))

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

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

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)

    if model in results and int(problem_name) in results[model]:
        results[model][int(problem_name)]["correct"] = correct
        results[model][int(problem_name)]["manual_overwrite"] = manual_overwrite
        results[model][int(problem_name)]["llm_annotation"] = llm_annotation
        results[model][int(problem_name)]["pass_at_1"] = data["pass_at_1"]

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
