"""Give attempts that stopped at the dollar limit the cost-limit grace turn they missed.

The live harness resumes a cost-limited session once with the competition's
`cost_limit_grace_prompt`. That needs a session id, and before 2026-11-05 Kimi Code
reported one only on a normal exit, so a Kimi run stopped at the limit got no
grace turn. This script gives such attempts that turn after the fact: the saved
workspace is copied, its Lake build is warmed outside the clock with the
original check command, and the saved native session is resumed once with the
same prompt and the same number of minutes. The new submission replaces the
scored archive in a copied result record; the original record, workspace and
submission are never changed and remain referenced from the new record.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
from datetime import timedelta
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import time

from dotenv import dotenv_values
from harness_wrapper import Agent, TokenUsage
from loguru import logger
import yaml

from leanlean.common import exclusive_lock, read_json, sha256, utc_now, write_json
from leanlean.evaluation import IMAGE_ROOT, capture_submission
from matharena.json_zst import dump_json_zst, load_json_zst
from matharena.runner import Runner
from matharena.solvers.codex_cli_solver import _write_text
from matharena.solvers.harness_solver import _events_to_messages
from matharena.solvers.native_usage import NativeUsageReader
from matharena.solvers.run_budget import RunBudget

ROOT = Path(__file__).resolve().parents[2]
COMPETITION = "leanlean/october-2026"
USAGE_FIELDS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout

    diff = git("diff", "HEAD").encode()
    return {"head": git("rev-parse", "HEAD").strip(), "dirty_diff_sha256": hashlib.sha256(diff).hexdigest()}


def provider_usage(budget: RunBudget) -> TokenUsage:
    """Usage of proxied requests only; native counters can repeat earlier turns."""
    with budget._lock:
        requests = list(budget._requests.values())
    return TokenUsage(**{name: sum(getattr(usage, name) for usage in requests) for name in USAGE_FIELDS})


def usage_dict(usage: TokenUsage) -> dict:
    return {name: getattr(usage, name) for name in USAGE_FIELDS}


def native_session(workspace: Path) -> str:
    """The attempt's only Kimi Code session, as `--session` expects it."""
    # Hidden folders such as `.index-dirty/` hold index markers named after sessions.
    sessions = sorted(path for path in (workspace / ".harness-home" / ".kimi-code" / "sessions").glob("*/session_*")
                      if path.is_dir() and not path.parent.name.startswith("."))
    if len(sessions) != 1:
        raise ValueError(f"Expected one native session in {workspace}, found {len(sessions)}")
    return sessions[0].name


class Salvage:
    def __init__(self, args):
        self.source = args.source_run_root.resolve()
        self.output = args.output.resolve()
        self.solver_name = args.solver
        self.keep_cache = set(args.keep_cache)
        self.configs = self.source / "configs"
        self.workspaces = self.output / "workspaces" / COMPETITION / self.solver_name
        self.results = self.output / "outputs" / COMPETITION / self.solver_name
        competition = yaml.safe_load((self.configs / "competitions" / f"{COMPETITION}.yaml").read_text())
        self.prompt = competition["harness_config"]["cost_limit_grace_prompt"]
        self.seconds = float(competition["harness_config"]["cost_limit_grace_seconds"])

    def original_workspace(self, problem_idx: int) -> Path:
        return self.source / "workspaces" / COMPETITION / self.solver_name / f"p{problem_idx}_r0"

    def original_result(self, problem_idx: int) -> Path:
        return self.source / "outputs" / COMPETITION / self.solver_name / f"{problem_idx}.json.zst"

    def prepare(self, problem_idx: int):
        runner = Runner(
            COMPETITION, 1, [problem_idx], str(self.configs / "competitions"), str(self.configs / "models"),
            str(self.output / "runner-outputs"), True, container_runtime="docker",
        )
        prepared = runner.prepare_run(self.solver_name, set_request_metadata=False, workers=1)
        solver = prepared["solver"]
        solver.workspace_root_template = str(self.workspaces / "p{problem_idx}_r{run_idx}")
        solver._prepare_harness_cli()
        payload, _ = prepared["batch"][0]
        return solver, payload

    @staticmethod
    def agent(solver, workspace, sandbox, environment, budget, monitor):
        """Mirror the Agent configuration of HarnessSolver._run_prepared."""

        def observe(event):
            monitor.on_event(event)
            if event.type == "usage":
                budget.usage(agent.get_tokens())

        agent = Agent(
            type=solver.harness,
            model=solver._build_model(),
            env=sandbox,
            dir=workspace,
            subagents=solver.harness_config.get("subagents", {}),
            executable=solver._agent_executable(),
            validate_version=bool(solver.harness_config.get("validate_version", False)) and solver._harness_cli is None,
            environment=environment,
            minimal_context=solver.minimal_context,
            tools_enabled=solver.tools_enabled,
            model_context_window=solver.harness_config.get("model_context_window"),
            model_metadata=solver.harness_config.get("model_metadata"),
            on_model_response=budget.capture_response,
            on_event=observe,
            should_stop=budget.exceeded,
            auto_wait=bool(solver.harness_config.get("auto_wait", True)),
            auto_fallback=bool(solver.harness_config.get("auto_fallback", True)),
            rate_limit_timeout=solver.harness_config.get("rate_limit_timeout"),
            rate_limit_poll_interval=float(solver.harness_config.get("rate_limit_poll_interval", 30)),
            max_recovery_attempts=int(solver.harness_config.get("max_recovery_attempts", 10)),
        )
        return agent

    def run(self, problem_idx: int) -> dict:
        target = self.results / f"{problem_idx}.json.zst"
        if target.exists():
            return read_json(self.output / "status" / f"{problem_idx}.json")
        original_path = self.original_result(problem_idx)
        original = load_json_zst(original_path)
        main = original["history"][0][0]
        limits = original["detailed_costs"][0]["run_limits"]
        if limits.get("exceeded") != "cost_limit" or limits.get("cost_limit_grace"):
            raise ValueError(f"P{problem_idx} did not stop at the cost limit without a grace turn")
        session_id = main.get("session_id") or native_session(self.original_workspace(problem_idx))
        workspace = self.workspaces / f"p{problem_idx}_r0"
        if workspace.exists():
            # A second resume would not reproduce the interrupted grace turn.
            raise RuntimeError(f"{workspace} exists without a saved result; inspect it before retrying")
        self.workspaces.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.original_workspace(problem_idx), workspace, symlinks=True)
        solver, payload = self.prepare(problem_idx)
        entry = payload["leanlean_project"]
        lake = workspace / "repository" / entry["project_path"] / ".lake"
        lake.mkdir(exist_ok=True)
        if not (lake / "packages").is_symlink():
            (lake / "packages").symlink_to(IMAGE_ROOT / "packages" / entry["key"], target_is_directory=True)
        requests_log = workspace / ".harness_wrapper" / "model_requests.jsonl"
        requests_offset = requests_log.stat().st_size if requests_log.exists() else 0
        sandbox = solver._build_sandbox(workspace)
        with solver._attempt_context(workspace, sandbox, payload, problem_idx, 0) as (monitor, environment):
            try:
                # Not ./check.sh: it stops at its protected-input check before building, and
                # a failed submission often changed one. A failing build still warms the rest.
                project = (Path("/work/repository") / entry["project_path"]).as_posix()
                command = f"cd {project} && lake build"
                if solver.monitoring["commands"] and solver.command_backend == "wrapper":
                    from matharena.monitoring.commands import TRACE_VERSION
                    command = f'test "$(cat /opt/leanlean/command-trace-version)" = {TRACE_VERSION} || exit 1\n' + command
                if problem_idx in self.keep_cache:
                    # The attempt's own build cache survived; resume on exactly that cache.
                    # `--no-build` only reports staleness, so the traced preflight `lake build`
                    # the monitor requires leaves the cache unchanged.
                    if not (lake / "build").is_dir():
                        raise RuntimeError(f"P{problem_idx} has no saved build cache to keep")
                    command = command.replace("lake build", "lake build --no-build || true")
                warm_started = utc_now()
                warm = solver._run_sandbox_command(
                    workspace, sandbox, ["sh", "-c", command], environment,
                    float(solver.harness_config.get("preflight_timeout_s", 1800)),
                )
                _write_text(workspace / "grace_warmup_stdout.log", warm.stdout)
                _write_text(workspace / "grace_warmup_stderr.log", warm.stderr)
                warmup = {"command": command.splitlines()[-1], "started_at": warm_started, "finished_at": utc_now(),
                          "returncode": warm.returncode, "counted_in_grace_time": False,
                          "kept_attempt_cache": problem_idx in self.keep_cache}
                if "materializing new dependency" in warm.stderr or "materializing new dependency" in warm.stdout:
                    raise RuntimeError(f"P{problem_idx} warm-up tried to fetch dependencies; the model was not resumed")
                monitor.assert_ready(require_commands=solver.monitoring["commands"] and solver.command_backend == "wrapper")
                monitor.phase = "model"
                monitor.capture("model_start")
                budget = RunBudget(
                    {"max_time_seconds": self.seconds},
                    lambda usage: solver._detailed_cost(None, usage=usage)["cost"],
                    priced=True,
                    request_level_cost=solver.pricing.threshold is not None,
                    base_cost_for_usage=lambda usage: solver._detailed_cost(None, usage=usage, base_only=True)["cost"],
                    native_usage_reader=NativeUsageReader(workspace, solver.harness_key),
                )
                deadline = budget.started_at + timedelta(seconds=self.seconds)
                prompt = self.prompt.format(
                    max_cost_usd=limits["max_cost_usd"],
                    cost_limit_grace_minutes=f"{self.seconds / 60:g}",
                    cost_limit_deadline_at=deadline.isoformat(timespec="seconds"),
                )
                agent = self.agent(solver, workspace, sandbox, environment, budget, monitor)
                agent._session_id = session_id
                started = time.monotonic()
                events = solver._stream_with_budget(agent, sandbox, budget, prompt, resume=True)
                elapsed_s = time.monotonic() - started
                monitor.capture("model_end")
                monitor.phase = "postcheck"
            finally:
                sandbox.stop()
            submission = capture_submission(workspace, entry, output=monitor.output / "submission")
        provider = provider_usage(budget)
        native = budget.usage(agent.get_tokens())
        cost = solver._detailed_cost(None, usage=provider)
        messages = _events_to_messages(events, harness=solver.harness_key)
        final_text = messages[-1].get("content", "") if messages else ""
        grace = {
            "seconds": self.seconds,
            "started_at": budget.started_at.isoformat(),
            "deadline_at": deadline.isoformat(timespec="seconds"),
            "completed": budget.reason is None and isinstance(final_text, str) and bool(final_text.strip()),
            "post_hoc": True,
            "post_hoc_reason": "Kimi Code reported no session id when stopped at the cost limit, so the live grace turn was skipped",
            "exceeded": budget.reason,
            "elapsed_s": elapsed_s,
            "warmup": warmup,
            "cost_usd": cost["cost"],
            "cost_basis": "proxied request usage",
            "provider_usage": usage_dict(provider),
            "native_usage": usage_dict(native),
        }
        step = {
            "step": "harness_cost_limit_grace",
            "timestep": len(original["history"][0]),
            "elapsed_s": elapsed_s,
            "messages": [{"role": "user", "content": prompt}, *messages],
            "harness": solver.harness,
            "container_runtime": solver.container_runtime,
            "tools_enabled": solver.tools_enabled,
            "harness_version": solver._harness_cli.release.version if solver._harness_cli else solver.harness_version,
            "harness_version_requested": solver.harness_version,
            "session_id": session_id,
            "model_requests": {"path": str(requests_log), "start_byte": requests_offset,
                               "end_byte": requests_log.stat().st_size if requests_log.exists() else 0},
            "workspace": str(workspace),
            "events": [event.to_dict() for event in events],
            "container_image": sandbox.image,
            "monitoring": solver._monitor_history(monitor),
            "cost_limit_grace": grace,
        }
        record = copy.deepcopy(original)
        main = record["history"][0][0]
        main["pre_grace_leanlean_submission"] = main["leanlean_submission"]
        main["leanlean_submission"] = submission
        main["repo_submission"] = {"type": "repository", "grader": "leanlean", **submission}
        record["history"][0].append(step)
        details = record["detailed_costs"][0]
        details["pre_grace"] = {key: details.get(key) for key in ("cost", "input_tokens", "cached_input_tokens",
                                                                   "output_tokens", "time", "request_time")}
        for key in ("cost", "input_tokens", "cached_input_tokens", "cached_write_tokens", "output_tokens"):
            details[key] = (details.get(key) or 0) + cost[key]
        for key in ("time", "request_time"):
            details[key] = (details.get(key) or 0) + elapsed_s
        details["run_limits"]["cost_limit_grace"] = {k: grace[k] for k in ("seconds", "started_at", "deadline_at",
                                                                          "completed", "post_hoc")}
        if grace["exceeded"] == "time_limit":
            details["run_limits"]["exceeded"] = "cost_grace_timeout"
        totals = record["cost"]
        for key in ("cost", "input_tokens", "output_tokens", "time", "request_time"):
            totals[key] = details[key]
        record["leanlean_cost_grace_salvage"] = {
            "original_result": str(original_path),
            "original_result_sha256": sha256(original_path),
            "original_workspace": str(self.original_workspace(problem_idx)),
        }
        # Progress, timing and release packing join this attempt onto the original one.
        write_json(monitor.output / "continues.json", {
            "schema": "leanlean_attempt_continuation_v1",
            "reason": "post_hoc_cost_limit_grace",
            "previous_attempt": str(Path(main["pre_grace_leanlean_submission"]["archive"]).parent.parent),
            "previous_result_sha256": sha256(original_path),
            "previous_submission_sha256": main["pre_grace_leanlean_submission"]["sha256"],
            "grace_seconds": self.seconds,
        })
        self.results.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(".partial")
        dump_json_zst(record, partial)
        partial.replace(target)
        status = {
            "problem_idx": problem_idx, "finished_at": utc_now(), "result": str(target), "result_sha256": sha256(target),
            "exceeded": budget.reason, "completed": grace["completed"], "elapsed_s": elapsed_s, "cost_usd": cost["cost"],
            "warmup_returncode": warm.returncode, "submission_sha256": submission["sha256"],
            "pre_grace_submission_sha256": main["pre_grace_leanlean_submission"]["sha256"],
        }
        (self.output / "status").mkdir(parents=True, exist_ok=True)
        write_json(self.output / "status" / f"{problem_idx}.json", status)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--solver", default="moonshot/k31")
    parser.add_argument("--problems", type=int, nargs="+", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--keep-cache", type=int, nargs="*", default=[],
                        help="problems whose saved .lake/build is resumed as is, without a warm-up build")
    args = parser.parse_args()
    # Same credentials as the Kimi launchers: only the OpenRouter key, no other provider's.
    key = dotenv_values(ROOT / "scripts" / ".env").get("OPENROUTER_API_KEY")
    if not key:
        raise SystemExit("Missing configured OpenRouter credential")
    for name in list(os.environ):
        if name.startswith(("ANTHROPIC_", "CLAUDE_", "DEEPCODE_", "OPENAI_", "KIMI_", "MOONSHOT_", "DSH_", "DEEPSEEK_")):
            os.environ.pop(name, None)
    os.environ.pop("HARNESS_WRAPPER_AUTH_FILE", None)
    os.environ["OPENROUTER_API_KEY"] = key
    (args.output / "auth").mkdir(parents=True, exist_ok=True)
    os.environ["HARNESS_WRAPPER_AUTH_FILE"] = str(args.output.resolve() / "auth" / "auth.json")
    salvage = Salvage(args)
    salvage.output.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(salvage.output / ".salvage.lock"):
        provenance = salvage.output / "provenance.json"
        state = {"script_sha256": sha256(Path(__file__)), "git": git_state(), "source_run_root": str(salvage.source),
                 "grace_seconds": salvage.seconds, "prompt": salvage.prompt}
        runs = read_json(provenance)["runs"] if provenance.exists() else []
        write_json(provenance, {**state, "runs": [*runs, {"started_at": utc_now(), "problems": args.problems}]})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(salvage.run, idx): idx for idx in args.problems}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    logger.info("P{} grace: {}", idx, future.result())
                except Exception:
                    logger.exception("P{} grace failed", idx)


if __name__ == "__main__":
    main()
