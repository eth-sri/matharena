import os
import re
from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from matharena.api_client import APIClient
from matharena.solvers.judges.base_judge import BaseJudge
from matharena.solvers.harness_solver import HarnessSolver
from matharena.utils import normalize_conversation
from matharena.tools.code_execution import execute_code


class SimpleJudge(BaseJudge):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, *, container_runtime="docker"):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config, container_runtime=container_runtime)
        model_config = deepcopy(self.solver_config["model_config"])
        for key in self.solver_config.get("api_client_remove_keys", []):
            model_config.pop(key, None)

        self.harness_solver = None
        if model_config.get("harness"):
            harness_config = model_config.setdefault("harness_config", {})
            log_root = Path(os.environ.get("MATHARENA_REQUEST_LOG_DIR", "logs/harness_workspaces"))
            # Separate judgments and retries must not reset another run's workspace.
            harness_config.setdefault(
                "workspace_root", str(log_root / "judges" / uuid4().hex / "p{problem_idx}_r{run_idx}")
            )
            harness_config.setdefault("auto_fallback", False)
            self.harness_solver = HarnessSolver(
                {}, "{problem}", model_config, "", container_runtime=self.container_runtime,
            )
            return

        tools = []
        for tool_name in self.solver_config["enabled_tools"]:
            if tool_name == "execute_code":
                spec = self.solver_config["tool_specs"]["execute_code"]
                tools.append((execute_code, spec["tool_spec"]))
        model_config["tools"] = tools
        model_config["max_tool_calls"] = self.solver_config["max_tool_calls"]

        self.client = APIClient(**model_config)
        self.RUN_ID = None

    def solve(
        self,
        problem_statement: str,
        guidelines: str,
        ground_truth_solutions: list[str],
        student_answer: str,
        original_problem_statement: str = "",
    ):
        self._start_run(problem_statement)
        if len(ground_truth_solutions) == 0:
            gt_text = "None provided."
        elif len(ground_truth_solutions) == 1:
            gt_text = ground_truth_solutions[0]
        else:
            gt_text = "\n\n".join(
                f"### Ground truth proof {i + 1} ###\n{proof}" for i, proof in enumerate(ground_truth_solutions)
            )

        prompt = self.solver_config["judge_prompt"].format(
            problem_statement=problem_statement,
            original_problem_statement=original_problem_statement,
            guidelines=guidelines,
            student_answer=student_answer,
            ground_truth_solutions=gt_text,
        )
        
        if self.harness_solver is not None:
            responses = list(self.harness_solver.solve_batch(
                [(prompt, None)], {0: self.problem_idx}, {0: self.run_idx}
            ))
            if not responses:
                return self._end_run(None, "Judge harness failed to return a final response; retry this judgment.")
            response = responses[0]
            self.harness_solver._response_agents.pop(id(response), None)
            conversation = normalize_conversation(response.conversation)
            self.detailed_cost = response.detailed_cost
            self.history = response.history
        else:
            conversation = normalize_conversation(self._query(self.client, [{"role": "user", "content": prompt}]))
            self._add_history("judge", 0, conversation)
        text = conversation[-1]["content"]
        fields = re.findall(r"<points>(.*?)</points>", text, flags=re.IGNORECASE | re.DOTALL)
        match = (
            re.fullmatch(r"\s*(?:Final\s+score:\s*)?([0-9]+)\s*", fields[0], re.IGNORECASE)
            if len(fields) == 1 else None
        )
        points = int(match.group(1)) if match else None
        if points is not None and not 0 <= points <= self.solver_config.get("judge_points_max", 7):
            points = None
        if "allowed_points" in self.solver_config and points not in self.solver_config["allowed_points"]:
            points = None
        return self._end_run(points, conversation[-1]["content"])
