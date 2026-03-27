import re

from matharena.api_client import APIClient
from matharena.solvers.judges.base_judge import BaseJudge
from matharena.utils import normalize_conversation
from matharena.tools.code_execution import execute_code


class SimpleJudge(BaseJudge):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config)
        model_config = self.solver_config["model_config"]
        for key in self.solver_config.get("api_client_remove_keys", []):
            model_config.pop(key, None)

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
        
        conversation = normalize_conversation(self._query(self.client, [{"role": "user", "content": prompt}]))
        text = conversation[-1]["content"]
        m_points = re.search(r"<points>\s*([0-9]+)\s*</points>", text, flags=re.IGNORECASE | re.DOTALL)

        points = max(0, min(int(m_points.group(1).strip()), 7)) if m_points else None
        return self._end_run(points, conversation[-1]["content"])
