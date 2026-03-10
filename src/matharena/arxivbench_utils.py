import json
import os

import yaml
import requests
from loguru import logger

from matharena.tools.paper_search import ocr_paper, STORE_FOLDER


def load_metadata(paper_root, paper_id):
    path = os.path.join(paper_root, paper_id, "metadata.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    
def resolve_model_config_path(model_arg, config_root="configs/models"):
    if model_arg.endswith(".yaml"):
        candidate = model_arg
    else:
        candidate = model_arg + ".yaml"
    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return candidate
    relative = os.path.join(config_root, candidate)
    if os.path.isfile(relative):
        return relative
    raise FileNotFoundError(f"Model config not found: {model_arg}")


def load_model_config(path):
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict) or "model" not in config:
        raise ValueError(f"Invalid model config: {path}")
    config = config.copy()
    config.pop("human_readable_id", None)
    config.pop("other_params", None)
    config.pop("date", None)
    return config


def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def list_paper_ids(paper_root):
    if not os.path.isdir(paper_root):
        return []
    paper_ids = []
    for name in os.listdir(paper_root):
        meta_path = os.path.join(paper_root, name, "metadata.json")
        if os.path.isfile(meta_path):
            paper_ids.append(name)
    return sorted(paper_ids)


def load_annotation(paper_root, paper_id, annotation_filename="llm_annotation.json"):
    path = os.path.join(paper_root, paper_id, annotation_filename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_annotation(paper_root, paper_id, data, annotation_filename="llm_annotation.json"):
    path = os.path.join(paper_root, paper_id, annotation_filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def download_pdf(paper_id):
    os.makedirs(STORE_FOLDER, exist_ok=True)
    pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")
    url = f"https://arxiv.org/pdf/{paper_id}.pdf"
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    with open(pdf_path, "wb") as f:
        f.write(response.content)
    return pdf_path


def ensure_ocr(paper_id, redo=False):
    pdf_path = os.path.join(STORE_FOLDER, f"{paper_id}.pdf")
    md_path = os.path.join(STORE_FOLDER, f"{paper_id}.md")
    if redo or not os.path.exists(pdf_path):
        download_pdf(paper_id)
    if redo or not os.path.exists(md_path):
        ocr_paper(paper_id)
    with open(md_path, "r", encoding="utf-8") as f:
        return f.read()


def _decode_json_escapes(value):
    try:
        return bytes(value, "utf-8").decode("unicode_escape")
    except UnicodeDecodeError:
        return None


def _try_json_loads(value):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def _repair_invalid_json_backslashes(text):
    valid_escapes = {'"', "\\", "/", "b", "f", "n", "r", "t"}
    out = []
    in_string = False
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            idx += 1
            continue

        if ch == '"':
            out.append(ch)
            in_string = False
            idx += 1
            continue

        if ch == "\\":
            if idx + 1 >= len(text):
                out.append("\\\\")
                idx += 1
                continue
            nxt = text[idx + 1]
            if nxt in valid_escapes:
                out.append(ch)
                out.append(nxt)
                idx += 2
                continue
            if nxt == "u":
                hex_digits = text[idx + 2:idx + 6]
                if len(hex_digits) == 4 and all(c in "0123456789abcdefABCDEF" for c in hex_digits):
                    out.append(text[idx:idx + 6])
                    idx += 6
                    continue
            out.append("\\\\")
            out.append(nxt)
            idx += 2
            continue

        if ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        else:
            out.append(ch)
        idx += 1
    return "".join(out)


def _try_json_loads_with_repair(value):
    parsed = _try_json_loads(value)
    if parsed is not None:
        return parsed
    repaired = _repair_invalid_json_backslashes(value)
    if repaired == value:
        return None
    return _try_json_loads(repaired)


def _iter_string_literals(text):
    in_string = False
    escape = False
    start = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                yield text[start:idx]
                in_string = False
                start = None
            continue
        if ch == '"':
            in_string = True
            start = idx + 1


def _find_json_fragment(text):
    stack = []
    in_string = False
    escape = False
    start = None
    for idx, ch in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch in "{[":
            if start is None:
                start = idx
            stack.append("}" if ch == "{" else "]")
            continue
        if ch in "}]":
            if stack and ch == stack[-1]:
                stack.pop()
                if not stack:
                    return text[start:idx + 1]
    return None


def _parse_nested_json(value):
    if not isinstance(value, str):
        return None
    parsed = _try_json_loads_with_repair(value)
    if parsed is not None:
        return parsed
    decoded = _decode_json_escapes(value)
    if decoded is None:
        return None
    return _try_json_loads_with_repair(decoded)


def extract_json(text):
    text = text.strip()
    if not text:
        return None
    parsed = _try_json_loads_with_repair(text)
    if parsed is not None:
        if isinstance(parsed, dict):
            nested = _parse_nested_json(parsed.get("raw"))
            if nested is not None:
                return nested
        if isinstance(parsed, str):
            nested = _parse_nested_json(parsed)
            if nested is not None:
                return nested
        return parsed
    fragment = _find_json_fragment(text)
    if fragment:
        parsed = _try_json_loads_with_repair(fragment)
        if parsed is not None:
            return parsed
        decoded = _decode_json_escapes(fragment)
        if decoded is not None:
            parsed = _try_json_loads_with_repair(decoded)
            if parsed is not None:
                return parsed
    for literal in _iter_string_literals(text):
        if literal.lstrip().startswith(("{", "[")):
            parsed = _parse_nested_json(literal)
            if parsed is not None:
                return parsed
    if "json" in text.lower():
        logger.warning(f"Failed to extract JSON from text that mentions JSON. {text}")
    return None

def get_latest_fields(annotation, fields):
    review = annotation.get("review") or {}
    values = []
    for field in fields:
        value = review.get(field) or annotation.get(field)
        if not value:
            return None
        values.append(str(value).strip())
    return tuple(values)


def get_latest_pair(annotation):
    return get_latest_fields(annotation, ["question", "answer"])
