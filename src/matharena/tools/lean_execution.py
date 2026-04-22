import asyncio
import json
import logging
import os
import re
import subprocess
import threading
import time
import shutil
import tempfile
from pathlib import Path

from axle import AxleClient, CheckResponse
from matharena.utils import normalize_conversation


DEFAULT_LEAN_ENVIRONMENT = "lean-4.29.0"
LOOGLE_DIR_ENV = "MATHARENA_LOOGLE_DIR"
REPO_ROOT = Path(__file__).resolve().parents[3]
COMPARATOR_BIN = REPO_ROOT / "external" / "comparator" / ".lake" / "build" / "bin" / "comparator"
LEAN4EXPORT_BIN = REPO_ROOT / "external" / "lean4export" / ".lake" / "build" / "bin" / "lean4export"
LANDRUN_BIN = REPO_ROOT / "external" / "landrun" / "bin" / "landrun"
COMPARATOR_PROJECT_DIR = REPO_ROOT / "external" / "comparator_project"
COMPARATOR_LOCK = threading.Lock()
COMPARATOR_AXIOMS = ["propext", "Quot.sound", "Classical.choice"]
COMPARATOR_TIMEOUT_WARNING = "Comparator timed out; accepted based on Axle only."
LOOGLE_PROCESS = None
LEAN_EXPLORE_SERVICE = None
LOOGLE_LOCK = threading.Lock()
LEAN_EXPLORE_LOCK = threading.Lock()
LEAN_CODE_BLOCK_RE = re.compile(r"```(?:lean)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
LEAN_BY_RE = re.compile(r":=\s*by\b")
LEAN_DECL_NAME_RE = re.compile(
    r"(?m)^\s*(?:theorem|lemma|def|definition|abbrev|opaque|instance)\s+([A-Za-z_][\w.']*)"
)
ADDED_TO_FILE_HEADER = "### Added To File ###"


async def _check_with_axle(content, environment=DEFAULT_LEAN_ENVIRONMENT):
    content = "\n".join(line for line in content.splitlines() if not line.lstrip().startswith("import "))
    async with AxleClient() as client:
        return await client.check(content=content, environment=environment, ignore_imports=True, timeout_seconds=600)


def _run_async(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result = {}
    error = {}

    def runner():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result["value"] = loop.run_until_complete(coro)
        except Exception as exc:
            error["value"] = exc
        finally:
            loop.close()

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if "value" in error:
        raise error["value"]
    return result["value"]


def _loogle_bin():
    if path := os.getenv(LOOGLE_DIR_ENV):
        return Path(path).expanduser().resolve() / ".lake" / "build" / "bin" / "loogle"
    raise RuntimeError(f"Set {LOOGLE_DIR_ENV}.")


def _loogle_dir():
    return Path(os.environ[LOOGLE_DIR_ENV]).expanduser().resolve()


def _get_lean_explore_service():
    global LEAN_EXPLORE_SERVICE
    if LEAN_EXPLORE_SERVICE is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        logging.getLogger("lean_explore").setLevel(logging.ERROR)
        from lean_explore.search import SearchEngine, Service

        LEAN_EXPLORE_SERVICE = Service(SearchEngine(use_local_data=False))
    return LEAN_EXPLORE_SERVICE


def _get_loogle_process():
    global LOOGLE_PROCESS
    loogle_bin = _loogle_bin()
    if LOOGLE_PROCESS is None or LOOGLE_PROCESS.poll() is not None:
        LOOGLE_PROCESS = subprocess.Popen(
            [str(loogle_bin), "--json", "--interactive"],
            cwd=_loogle_dir(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        LOOGLE_PROCESS.stdout.readline()
    return LOOGLE_PROCESS


def loogle(query, max_results=10):
    try:
        with LOOGLE_LOCK:
            loogle_process = _get_loogle_process()
            loogle_process.stdin.write(f"{query}\n")
            loogle_process.stdin.flush()
            payload = json.loads(loogle_process.stdout.readline())
    except FileNotFoundError:
        return f"Error: loogle is not installed at {_loogle_bin()}."
    except Exception as exc:
        return f"Error running loogle: {exc}"

    if "error" in payload:
        return payload["error"]

    hits = payload.get("hits", [])[:max_results]
    lines = [payload.get("header", f"Found {payload.get('count', len(hits))} result(s).")]
    for idx, hit in enumerate(hits, start=1):
        lines.append(f"{idx}. {hit['name']} : {hit['type']}")
        if hit.get("module"):
            lines.append(f"   from {hit['module']}")
    return "\n".join(lines)


def lean_explore_search(query, max_results=10):
    try:
        with LEAN_EXPLORE_LOCK:
            service = _get_lean_explore_service()
            response = _run_async(
                service.search(
                    query=str(query),
                    limit=max_results,
                    rerank_top=max_results,
                    packages=["Mathlib", "Lean", "Init", "Std"],
                )
            )
            payload = {
                "count": response.count,
                "results": [
                    {
                        "id": result.id,
                        "name": result.name,
                        "module": result.module,
                        "description": result.informalization or result.docstring,
                    }
                    for result in response.results
                ],
            }
    except Exception as exc:
        return f"Error running LeanExplore: {exc}"

    lines = [f"Found {payload['count']} LeanExplore result(s)."]
    for idx, hit in enumerate(payload["results"], start=1):
        lines.append(f"{idx}. [{hit['id']}] {hit['name']}")
        lines.append(f"   from {hit['module']}")
        if hit.get("description"):
            lines.append(f"   {hit['description'].splitlines()[0]}")
    return "\n".join(lines)


def get_lean_feedback_dict(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    retry = 0
    while retry < 5:
        try:
            result = _run_async(_check_with_axle(statement, environment=environment))
            break
        except Exception as exc:
            print(f"Error checking with Axle: {exc}. Retrying...")
            retry += 1
            time.sleep(60)
    if retry == 5 or not isinstance(result, CheckResponse):
        return {
            "okay": False,
            "errors": ["Failed to get feedback from Axle after 5 retries."],
            "warnings": [],
            "infos": [],
        }
    feedback = {
        "okay": result.okay,
        "errors": result.lean_messages.errors + result.tool_messages.errors,
        "warnings": result.lean_messages.warnings + result.tool_messages.warnings,
        "infos": result.lean_messages.infos + result.tool_messages.infos,
    }
    feedback["warnings"] = [
        warning for warning in feedback["warnings"] if not warning.startswith("Imports mismatch") and warning != "Using defaults..."
    ]
    return feedback


def _format_feedback(feedback):
    valid = feedback["okay"] and not feedback["errors"]
    parts = [f"### Compiles ###\n{feedback['okay']}", f"### Valid Proof ###\n{valid}"]
    for key in ["errors", "warnings", "infos"]:
        part = f"""### {key.capitalize()} ###\n""" + "\n".join(feedback[key])
        parts.append(part)
    return "\n\n".join(parts)


def _add_to_file_succeeded(content):
    lines = content.splitlines()
    return len(lines) >= 2 and lines[0] == ADDED_TO_FILE_HEADER and lines[1] == "True"


def _persistent_lean_prefix(messages):
    if not messages:
        return ""

    clean_messages = normalize_conversation(messages)
    pending = {}
    fallback = []
    blocks = []
    for message in clean_messages:
        if message["role"] == "assistant" and message.get("type") == "tool_call" and message.get("tool_name") == "add_to_file":
            arguments = message.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            code = arguments.get("code", "").strip()
            if not code:
                continue
            if message.get("tool_call_id") is None:
                fallback.append(code)
            else:
                pending[message["tool_call_id"]] = code
        elif (
            message["role"] == "tool_response"
            and message.get("tool_name") == "add_to_file"
            and _add_to_file_succeeded(message.get("content", ""))
        ):
            if message.get("tool_call_id") is None:
                code = fallback.pop(0) if fallback else ""
            else:
                code = pending.pop(message["tool_call_id"], "")
            if code:
                blocks.append(code)
    return "\n\n".join(blocks)


def _dedupe_lean_blocks(blocks):
    seen = set()
    kept = []
    for block in reversed([block for block in blocks if block.strip()]):
        names = set(LEAN_DECL_NAME_RE.findall(block))
        complete_names = {name for name in names if re.search(rf"\b{name}\b[\s\S]*:=", block)}
        if any(name in seen for name in complete_names):
            continue
        seen.update(complete_names)
        kept.append(block.strip())
    return list(reversed(kept))


def _prepend_persistent_lean(code, messages):
    prefix = _persistent_lean_prefix(messages)
    blocks = _dedupe_lean_blocks([prefix, code])
    return "\n\n".join(blocks)


def get_executed_lean_submission_parts(model_output, formal_statement=None, messages=None):
    lean_blocks = _extract_submission_blocks(model_output)
    persistent_prefix = _persistent_lean_prefix(messages)

    prefix_blocks = []
    if persistent_prefix:
        prefix_blocks.append(persistent_prefix)
    if len(lean_blocks) > 1:
        prefix_blocks.append(lean_blocks[0].strip())
    executed_prefix = "\n\n".join(_dedupe_lean_blocks(prefix_blocks))

    theorem_block = lean_blocks[-1].strip() if lean_blocks else ""
    if formal_statement is not None and theorem_block:
        normalized_formal_statement = _normalize_formal_statement(formal_statement)
        block_match = LEAN_BY_RE.search(theorem_block)
        formal_match = LEAN_BY_RE.search(normalized_formal_statement)
        if block_match is not None and formal_match is not None:
            theorem_block = (
                normalized_formal_statement[: formal_match.start()].rstrip() + "\n" + theorem_block[block_match.start() :]
            )

    return executed_prefix, theorem_block


def verify_lean(code, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict(_prepend_persistent_lean(code, messages), environment=environment)
    return _format_feedback(feedback)


def add_to_file(code, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict(_prepend_persistent_lean(code, messages), environment=environment)
    if feedback["okay"] and not feedback["errors"]:
        return f"{ADDED_TO_FILE_HEADER}\nTrue\n\nAdded to file."
    return f"{ADDED_TO_FILE_HEADER}\nFalse\n\n" + _format_feedback(feedback)


def _normalize_formal_statement(formal_statement):
    if "```lean" in formal_statement:
        lean_blocks = LEAN_CODE_BLOCK_RE.findall(formal_statement)
        if len(lean_blocks) > 0:
            return lean_blocks[-1].strip()
    return formal_statement


def _extract_submission_blocks(model_output):
    lean_blocks = [block.strip() for block in LEAN_CODE_BLOCK_RE.findall(model_output)]
    if len(lean_blocks) == 1:
        model_output = lean_blocks[0]
    elif len(lean_blocks) > 1:
        model_output = lean_blocks[-1]

    theorem_idx = model_output.rfind("\ntheorem")
    if theorem_idx == -1 and model_output.lstrip().startswith("theorem"):
        theorem_idx = model_output.find("theorem")
    if theorem_idx == -1:
        return []

    prefix = model_output[:theorem_idx].strip()
    theorem = model_output[theorem_idx:].strip()
    return [prefix, theorem] if prefix else [theorem]


def _run_comparator_check(model_output, formal_statement, messages=None):
    if not all(
        path.exists()
        for path in [COMPARATOR_BIN, LEAN4EXPORT_BIN, LANDRUN_BIN, COMPARATOR_PROJECT_DIR / "lakefile.lean"]
    ):
        return None

    theorem_name_match = LEAN_DECL_NAME_RE.search(formal_statement)
    if theorem_name_match is None:
        return "Comparator could not extract the theorem name from the formal statement."

    executed_prefix, theorem_block = get_executed_lean_submission_parts(
        model_output, formal_statement=formal_statement, messages=messages
    )
    solution_code = "\n\n".join(block for block in [executed_prefix, theorem_block] if block.strip())

    with COMPARATOR_LOCK:
        with tempfile.TemporaryDirectory(prefix="matharena-comparator-") as tmpdir:
            tmpdir_path = Path(tmpdir)
            shutil.copy2(COMPARATOR_PROJECT_DIR / "lean-toolchain", tmpdir_path / "lean-toolchain")
            shutil.copy2(COMPARATOR_PROJECT_DIR / "lakefile.lean", tmpdir_path / "lakefile.lean")
            manifest_path = COMPARATOR_PROJECT_DIR / "lake-manifest.json"
            if manifest_path.exists():
                shutil.copy2(manifest_path, tmpdir_path / "lake-manifest.json")

            tmp_lake_dir = tmpdir_path / ".lake"
            tmp_lake_dir.mkdir(exist_ok=True)
            shared_packages = COMPARATOR_PROJECT_DIR / ".lake" / "packages"
            if shared_packages.exists():
                os.symlink(shared_packages, tmp_lake_dir / "packages", target_is_directory=True)

            (tmpdir_path / "Challenge.lean").write_text(f"import Mathlib\n\n{formal_statement}\n", encoding="utf-8")
            (tmpdir_path / "Solution.lean").write_text(f"import Mathlib\n\n{solution_code}\n", encoding="utf-8")
            (tmpdir_path / "comparator.json").write_text(
                json.dumps(
                    {
                        "challenge_module": "Challenge",
                        "solution_module": "Solution",
                        "theorem_names": [theorem_name_match.group(1)],
                        "permitted_axioms": COMPARATOR_AXIOMS,
                        "enable_nanoda": False,
                    }
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["PATH"] = f"{LANDRUN_BIN.parent}:{LEAN4EXPORT_BIN.parent}:{env.get('PATH', '')}"
            try:
                result = subprocess.run(
                    ["lake", "env", str(COMPARATOR_BIN), "comparator.json"],
                    cwd=tmpdir_path,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired:
                return COMPARATOR_TIMEOUT_WARNING

    if result.returncode == 0:
        return None
    return (result.stderr or result.stdout or "Comparator rejected the submission.").strip()


def get_lean_feedback_dict_with_formal_statement(
    model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None, use_comparator=False
):
    formal_statement = _normalize_formal_statement(formal_statement)
    lean_blocks = _extract_submission_blocks(model_output)
    if len(lean_blocks) == 0:
        return {
            "okay": False,
            "errors": ["No Lean code block found in the model output."],
            "warnings": [],
            "infos": [],
        }

    block_match = LEAN_BY_RE.search(lean_blocks[-1])
    formal_match = LEAN_BY_RE.search(formal_statement)
    if block_match is None or formal_match is None:
        return {
            "okay": False,
            "errors": ["Could not align the final Lean code block with the formal statement."],
            "warnings": [],
            "infos": [],
        }

    lean_blocks[-1] = formal_statement[:formal_match.start()].rstrip() + "\n" + lean_blocks[-1][block_match.start():]
    feedback = get_lean_feedback_dict(_prepend_persistent_lean("\n\n".join(lean_blocks), messages), environment=environment)
    if not feedback["okay"] or feedback["errors"]:
        return feedback

    if use_comparator:
        comparator_error = _run_comparator_check(model_output, formal_statement, messages=messages)
        if comparator_error == COMPARATOR_TIMEOUT_WARNING:
            feedback["warnings"].append(COMPARATOR_TIMEOUT_WARNING)
            return feedback
        if comparator_error:
            feedback["okay"] = False
            feedback["errors"].append(comparator_error)
    return feedback


def verify_lean_with_formal_statement(model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict_with_formal_statement(
        model_output, formal_statement, environment=environment, messages=messages
    )
    return _format_feedback(feedback)

def compiles_with_sorries(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    feedback = get_lean_feedback_dict(statement, environment=environment)
    return feedback["okay"]


def compiles_with_formal_statement(model_output, formal_statement, environment=DEFAULT_LEAN_ENVIRONMENT, messages=None):
    feedback = get_lean_feedback_dict_with_formal_statement(
        model_output, formal_statement, environment=environment, messages=messages, use_comparator=True
    )
    if feedback["errors"]:
        return False
    return feedback["okay"]


def compiles(statement, environment=DEFAULT_LEAN_ENVIRONMENT):
    feedback = get_lean_feedback_dict(statement, environment=environment)
    if feedback["errors"]:
        return False
    return feedback["okay"]
