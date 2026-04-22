import asyncio
import tarfile
import tempfile
import threading
import time
from pathlib import Path

from aristotlelib import Project
from aristotlelib.api_request import AristotleAPIError

from matharena.tools.lean_execution import get_lean_feedback_dict_with_formal_statement


DEFAULT_ARISTOTLE_TOOLCHAIN = "leanprover/lean4:v4.28.0"
DEFAULT_ARISTOTLE_MATHLIB_REV = "v4.28.0"
DEFAULT_ARISTOTLE_POLLING_INTERVAL_SECONDS = 30
ARISTOTLE_PROBLEM_FILE = "Problem.lean"


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


def _build_lakefile(mathlib_rev):
    return f"""import Lake
open Lake DSL

package matharena_aristotle where

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "{mathlib_rev}"

lean_lib Problem where
  roots := #[`Problem]
"""


def _build_problem_file(problem_statement, formal_statement):
    if problem_statement is None:
        problem_comment = ""
    else:
        problem_comment = f"/-\nNatural language problem statement:\n{problem_statement}\n-/\n\n"
    return f"import Mathlib\n\n{problem_comment}{formal_statement.strip()}\n"


def _write_project(project_dir, problem_statement, formal_statement, toolchain, mathlib_rev):
    project_dir = Path(project_dir)
    (project_dir / "lean-toolchain").write_text(f"{toolchain}\n", encoding="utf-8")
    (project_dir / "lakefile.lean").write_text(_build_lakefile(mathlib_rev), encoding="utf-8")
    (project_dir / ARISTOTLE_PROBLEM_FILE).write_text(
        _build_problem_file(problem_statement, formal_statement),
        encoding="utf-8",
    )


def _read_solution_file(extracted_dir, relative_path):
    extracted_dir = Path(extracted_dir)
    relative_path = Path(relative_path)
    direct_path = extracted_dir / relative_path
    if direct_path.exists():
        return direct_path.read_text(encoding="utf-8")

    matches = [path for path in extracted_dir.rglob(relative_path.name) if path.is_file()]
    for path in matches:
        if path.as_posix().endswith(relative_path.as_posix()):
            return path.read_text(encoding="utf-8")
    if len(matches) == 1:
        return matches[0].read_text(encoding="utf-8")
    raise FileNotFoundError(f"Could not find {relative_path} in Aristotle output.")


async def _run_aristotle_async(prompt, project_dir, output_tarball_path, polling_interval_seconds):
    backoff_seconds = 30
    while True:
        try:
            project = await Project.create_from_directory(prompt=prompt, project_dir=project_dir)
            break
        except AristotleAPIError as exc:
            if exc.status_code != 429:
                raise
            await asyncio.sleep(backoff_seconds)
            backoff_seconds = min(backoff_seconds * 2, 300)
    await project.wait_for_completion(
        destination=output_tarball_path,
        polling_interval_seconds=polling_interval_seconds,
    )
    await project.refresh()
    return project


def execute_aristotle(
    prompt,
    problem_statement,
    formal_statement,
    problem_idx=None,
    run_idx=None,
    toolchain=DEFAULT_ARISTOTLE_TOOLCHAIN,
    mathlib_rev=DEFAULT_ARISTOTLE_MATHLIB_REV,
    polling_interval_seconds=DEFAULT_ARISTOTLE_POLLING_INTERVAL_SECONDS,
):
    prefix = "matharena-aristotle-"
    if problem_idx is not None:
        prefix += f"p{problem_idx}-"
    if run_idx is not None:
        prefix += f"r{run_idx}-"

    start = time.time()
    with tempfile.TemporaryDirectory(prefix=prefix) as tmpdir:
        tmpdir_path = Path(tmpdir)
        project_dir = tmpdir_path / "project"
        project_dir.mkdir()
        _write_project(project_dir, problem_statement, formal_statement, toolchain, mathlib_rev)

        output_tarball_path = tmpdir_path / "solution.tar.gz"
        project = _run_async(
            _run_aristotle_async(
                prompt=prompt,
                project_dir=project_dir,
                output_tarball_path=output_tarball_path,
                polling_interval_seconds=polling_interval_seconds,
            )
        )

        extracted_dir = tmpdir_path / "solution"
        extracted_dir.mkdir()
        with tarfile.open(output_tarball_path, "r:gz") as tar:
            tar.extractall(extracted_dir)

        code = _read_solution_file(extracted_dir, ARISTOTLE_PROBLEM_FILE)
        elapsed = time.time() - start
        return {
            "project_id": project.project_id,
            "status": project.status.value,
            "output_summary": project.output_summary,
            "code": code,
            "time": elapsed,
        }


def verify_aristotle_code(model_output, formal_statement, environment, messages=None):
    return get_lean_feedback_dict_with_formal_statement(
        model_output,
        formal_statement,
        environment=environment,
        messages=messages,
        use_comparator=False,
    )
