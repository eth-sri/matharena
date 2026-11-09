"""LeanLean project attempts using MathArena's shared harness execution."""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from leanlean.evaluation import capture_submission, prepare_workspace
from matharena.solvers.harness_solver import HarnessSolver


class LeanLeanSolver(HarnessSolver):
    """Prepare projects, collect telemetry, and retain files for deferred grading."""

    def __init__(
        self, solver_config, default_prompt_template, default_api_client_args,
        last_chance_prompt, *, container_runtime="docker",
    ):
        config = default_api_client_args.copy()
        harness_config = dict(config.get("harness_config") or {})
        if harness_config.get("leanlean_projects") and "monitoring" not in harness_config:
            # Frozen launches predate the shared flags and originally collected
            # full wrapper telemetry. Preserve that meaning until explicitly migrated.
            harness_config["monitoring"] = dict.fromkeys(("diff", "cpu", "ram", "commands"), True)
            harness_config.setdefault("command_monitoring_backend", "wrapper")
        # Comparator keeps both exports in /tmp; large projects exceed 1 GiB.
        harness_config.setdefault("tmpfs_size", "16g")
        harness_config.setdefault("repo_eval", True)
        harness_config.setdefault("preflight", "python proof_length.py && ./check.sh")
        if harness_config.get("runtime_manifest"):
            runtime = json.loads(Path(harness_config["runtime_manifest"]).read_text())
            prepared_backend = runtime.get("command_monitoring_backend")
            if prepared_backend is not None:
                harness_config.setdefault("command_monitoring_backend", prepared_backend)
                if ((harness_config.get("monitoring") or {}).get("commands")
                        and harness_config["command_monitoring_backend"] != prepared_backend):
                    raise ValueError(f"Prepared eval image uses {prepared_backend} command monitoring")
            if (
                (harness_config.get("monitoring") or {}).get("commands")
                and harness_config.get("command_monitoring_backend", "host") == "host"
                and runtime.get("command_trace_version") is not None
            ):
                raise ValueError(
                    "Host command monitoring requires rebuilding the eval image without command wrappers; "
                    "use leanlean-prepare-eval --command-monitoring host, or explicitly select "
                    "command_monitoring_backend: wrapper for the historical image"
                )
            for key in ("docker_image", "container_engine", "container_user"):
                harness_config[key] = runtime[key]
            harness_config["docker_extra_args"] = [
                *runtime.get("docker_extra_args", ()),
                *harness_config.get("docker_extra_args", ()),
            ]
        config["harness_config"] = harness_config
        super().__init__(
            solver_config, default_prompt_template, config, last_chance_prompt,
            container_runtime=container_runtime,
        )
        if not self.tools_enabled:
            raise ValueError("Lean harness runs require allow_harness: true and tools_enabled: true")
        if not self.keep_workspaces:
            raise ValueError("LeanLean evaluation requires keep_workspaces for deferred scoring")

    def solve_batch(self, stmt_batch, batch_idx_to_problem_idx, batch_idx_to_run_idx):
        completed = set()
        for response in super().solve_batch(stmt_batch, batch_idx_to_problem_idx, batch_idx_to_run_idx):
            completed.add(response.idx)
            yield response
        # Persist successes before reporting missing submissions as a failed batch.
        missing = sorted(set(range(len(stmt_batch))) - completed)
        if missing:
            attempts = sorted((batch_idx_to_problem_idx[i], batch_idx_to_run_idx[i]) for i in missing)
            pending = ", ".join(f"P{problem}/r{run}" for problem, run in attempts)
            raise RuntimeError(f"Harness batch incomplete: {len(missing)} failed attempt(s): {pending}")

    def _prepare_repository(self, workspace, payload):
        prepare_workspace(workspace, payload["leanlean_project"])

    def _agent_environment(self):
        environment = super()._agent_environment()
        if self.harness_key in {"claude", "claude-code", "anthropic"}:
            # Root in this isolated runtime maps to the host user.
            environment.setdefault("IS_SANDBOX", "1")
        return environment

    def _attempt_environment(self, payload):
        environment = super()._attempt_environment(payload)
        toolchain = payload["leanlean_project"]["lean_toolchain"].replace("/", "--").replace(":", "---")
        environment["PATH"] = f"/opt/elan/bin:/opt/elan/toolchains/{toolchain}/bin:" + environment["PATH"]
        return environment

    def _source_directory(self, workspace):
        return workspace / "repository"

    def _monitor_metadata(self, payload):
        return {"project_key": payload["leanlean_project"]["key"]}

    def _finalize_attempt(self, response, payload, workspace, sandbox, monitor):
        # File submissions do not use final-answer reprompting.
        self._response_agents.pop(id(response), None)
        self._response_budgets.pop(id(response), None)
        output = (monitor.output if monitor is not None else
                  workspace.parent / (workspace.name + ".submissions") / uuid.uuid4().hex)
        response.history[0]["leanlean_submission"] = capture_submission(
            workspace, payload["leanlean_project"], output=output / "submission"
        )
        response.history[0]["repo_submission"] = {
            "type": "repository", "grader": "leanlean", **response.history[0]["leanlean_submission"],
        }
        response.history[0]["eval_type"] = "repo"
        response.detailed_cost["deferred_grading"] = True
        response.history[0]["container_image"] = sandbox.image
