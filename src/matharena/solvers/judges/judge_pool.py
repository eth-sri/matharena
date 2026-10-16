from loguru import logger

from matharena.solvers.agent_pool import AgentPool
from matharena.solvers.judges.simple_judge import SimpleJudge
from matharena.solvers.judges.maj_judge import MajorityJudge
from matharena.solvers.judges.norm_judge import NormalizeJudge
from matharena.container_runtime import validate_container_runtime


class JudgePool(AgentPool):
    AGENT_CLASSES = {"simple_judge": SimpleJudge, "maj_judge": MajorityJudge, "norm_judge": NormalizeJudge}

    def __init__(self, solver_config, *, container_runtime="docker"):
        self.container_runtime = validate_container_runtime(container_runtime)
        self.solver_config = solver_config
        self.scaffold_config = solver_config
        self.n_threads = self.scaffold_config.get("n_threads", 1)
        self.AGENT_CLASS = JudgePool.AGENT_CLASSES[self.solver_config["scaffold_name"]]

    def _run_agent(self, batch_idx: int, problem_idx: int, run_idx: int, stmt):
        payload = stmt[1]
        total_cost, history = {}, []
        additive_fields = {
            "cost", "input_tokens", "output_tokens", "cached_input_tokens",
            "cached_write_tokens", "time", "request_time", "n_retries",
        }
        for attempt in range(1, 4):
            agent = self.AGENT_CLASS(
                batch_idx=batch_idx,
                problem_idx=problem_idx,
                run_idx=run_idx,
                solver_config=self.solver_config,
                container_runtime=self.container_runtime,
            )
            response = agent.solve(
                stmt[0], payload["guidelines"], payload["ground_truth_solutions"],
                payload["student_answer"], payload.get("original_problem_statement", ""),
            )
            for key, value in response.detailed_cost.items():
                if key in additive_fields:
                    total_cost[key] = total_cost.get(key, 0) + value
                elif key == "cost_by_judge_model":
                    by_model = total_cost.setdefault(key, {})
                    for model, cost in value.items():
                        by_model[model] = by_model.get(model, 0) + cost
                else:
                    total_cost[key] = value
            history.extend({**entry, "judge_attempt": attempt} for entry in response.history)
            response.history = list(history)
            response.detailed_cost = dict(total_cost)
            response.detailed_cost["n_retries"] = total_cost.get("n_retries", 0) + attempt - 1
            response.additional_info["judge_attempts"] = attempt
            points = response.points
            if (
                points is not None
                and 0 <= points <= self.solver_config.get("judge_points_max", 7)
                and ("allowed_points" not in self.solver_config or points in self.solver_config["allowed_points"])
            ):
                return response
            if attempt < 3:
                logger.warning(
                    "Judge query {} returned no valid score on attempt {}/3; retrying automatically.",
                    batch_idx, attempt,
                )
        response.points = None
        return response
