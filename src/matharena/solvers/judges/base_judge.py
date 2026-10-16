from matharena.solvers.base_agent import BaseAgent
from matharena.solvers.judges.judge_response import JudgeResponse
from matharena.container_runtime import validate_container_runtime


class BaseJudge(BaseAgent):
    def __init__(self, batch_idx, problem_idx, run_idx, solver_config, *, container_runtime="docker"):
        super().__init__(batch_idx, problem_idx, run_idx, solver_config)
        self.container_runtime = validate_container_runtime(container_runtime)

    def _end_run(self, points: int | None, explanation: str, additional_info: dict = None) -> JudgeResponse:
        self.has_finished = True
        return JudgeResponse(
            idx=self.batch_idx,
            points=points,
            explanation=explanation,
            detailed_cost=self.detailed_cost,
            history=self.history,
            additional_info=additional_info,
        )

    def solve(
        self,
        problem_statement: str,
        guidelines: str,
        ground_truth_solutions: list[str],
        student_answer: str,
        original_problem_statement: str = "",
    ):
        raise NotImplementedError("Subclasses should implement this method.")
