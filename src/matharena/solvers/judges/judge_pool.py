from matharena.solvers.agent_pool import AgentPool
from matharena.solvers.judges.simple_judge import SimpleJudge
from matharena.solvers.judges.maj_judge import MajorityJudge
from matharena.solvers.judges.norm_judge import NormalizeJudge


class JudgePool(AgentPool):
    AGENT_CLASSES = {"simple_judge": SimpleJudge, "maj_judge": MajorityJudge, "norm_judge": NormalizeJudge}

    def __init__(self, solver_config):
        self.solver_config = solver_config
        self.scaffold_config = solver_config
        self.n_threads = self.scaffold_config.get("n_threads", 1)
        self.AGENT_CLASS = JudgePool.AGENT_CLASSES[self.solver_config["scaffold_name"]]

    def _run_agent(self, batch_idx: int, problem_idx: int, run_idx: int, stmt):
        agent = self.AGENT_CLASS(
            batch_idx=batch_idx,
            problem_idx=problem_idx,
            run_idx=run_idx,
            solver_config=self.solver_config
        )
        payload = stmt[1]
        return agent.solve(stmt[0], payload["guidelines"], payload["ground_truth_solutions"], 
                            payload["student_answer"], payload.get("original_problem_statement", ""))
