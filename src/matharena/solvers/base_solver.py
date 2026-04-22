"""Generic solver to inherit from (for base_model or agent solvers)."""

from typing import Any

from matharena.solvers import SolverResponse


class BaseSolver:
    """
    An abstract solver. It wraps an APIClient and uses it to solve problems.
    Subclassed by PureModelSolver and AgentPool.
    """

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        """
        Initializes the solver.
        """
        self.solver_config = solver_config
        self.default_prompt_template = default_prompt_template
        self.default_api_client_args = default_api_client_args
        self.last_chance_prompt = last_chance_prompt
        self.lean_environment_override = default_api_client_args.get("lean_environment_override")

    def solve_batch(self, stmt_batch: list[tuple[str, Any]], batch_idx_to_problem_idx: dict[int, int], batch_idx_to_run_idx: dict[int, int]):
        """
        Solves a batch of problems.

        Args:
            stmt_batch (list[tuple[str, Any]]): A list of problem statements (text, image) to be solved.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the index, conversation, detailed cost, and history for each problem.
        """
        raise NotImplementedError("Subclasses should implement solver.solve")

    def last_chance(self, response: SolverResponse) -> SolverResponse:
        """
        If the parser did not find the solution, the solver has a last chance to modify its response.

        Args:
            response (SolverResponse): The response this solver previously returned for a problem.

        Returns:
            SolverResponse: The modified response after reprompting the model to report an answer.
        """
        raise NotImplementedError("Subclasses should implement solver.last_chance")
