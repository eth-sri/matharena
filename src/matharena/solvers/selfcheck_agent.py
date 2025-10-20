from typing import Any, override
from matharena.solvers import SolverResponse, BaseAgent
from loguru import logger
from matharena.api_client import APIClient
from matharena.utils import get_substring


class SelfcheckAgent(BaseAgent):
    """
    An agent that solves problems through a series of
    self-correction and verification steps.
    """

    def __init__(self, batch_idx, solver_config, default_prompt_template, default_api_client_args):
        super().__init__(batch_idx, solver_config, default_prompt_template, default_api_client_args)
        self.model_config = solver_config["model_config"]
        self.scaffold_config = solver_config["scaffold_config"]

        # Params
        self.correct_count = self.scaffold_config.get("correct_count", 5)
        self.error_count = self.scaffold_config.get("error_count", 10)
        self.max_iterations = self.scaffold_config.get("max_iterations", 30)
        self.return_if_not_found = self.scaffold_config.get("return_if_not_found", False)
        self.prompts = self.scaffold_config.get("prompts", {})
        expected_prompts = ["system", "self_improvement", "correction", "verification_system", "verification_reminder"]
        assert all(
            p in self.prompts.keys() for p in expected_prompts
        ), f"SelfcheckAgent expects prompts: {expected_prompts}, got {list(self.prompts.keys())}"

        # Clients
        self.client = APIClient(**default_api_client_args)

    @override
    def solve(self, stmt: str) -> SolverResponse:
        """
        Solves a single problem statement.

        Args:
            stmt (str): A problem statement as text.

        Returns:
            SolverResponse: A SolverResponse object (see BaseAgent._end_run).
        """
        # Start run and find the first solution with self improvement.
        self._start_run(stmt)
        timestep = 1
        timestep, solution = self.generate_initial_solution("init", timestep, stmt)
        if solution is None:
            logger.debug(f"[{self.bi}] Agent failed during initialization.")
            return self._end_run("Fail.")

        # Main agent loop
        error_count, correct_count = 0, 0
        for it in range(1, self.max_iterations + 1):
            logger.debug(f"[{self.bi}] Agent iteration {it} (correct count {correct_count}, error count {error_count})")

            # Verify the current solution
            logger.debug(f"[{self.bi}] Verifying solution.")
            timestep, (is_correct, bug_report) = self.verify_solution(f"verify-it={it}", timestep, stmt, solution)
            if "yes" in is_correct.lower() and it > 1:
                logger.debug(f"[{self.bi}] Solution verified.")
                correct_count += 1
                error_count = 0
            else:
                logger.debug(f"[{self.bi}] Verification failed, finding a new solution with correction.")
                correct_count = 0
                error_count += 1

                # Establish a new prompt that contains the solution and the bug report
                convo = [
                    {"role": "developer", "content": self.prompts["system"]},
                    {"role": "user", "content": stmt},
                    {"role": "assistant", "content": solution},
                    {"role": "user", "content": f"{self.prompts["correction"]}\n\n{bug_report}"},
                ]
                convo = self._query(self.client, convo)
                solution = convo[-1]["content"]
                solution = get_substring(solution, ["</think>", "</summary>"], mode="after")
                self._add_history(f"correct-it={it}", timestep, convo)
                timestep += 1

            # If many corrects or many errors, stop
            if correct_count >= self.correct_count:
                logger.success(f"[{self.bi}] Agent found a solution.")
                return self._end_run(solution)
            if error_count >= self.error_count:
                logger.debug(f"[{self.bi}] Too many errors, stopping.")
                return self._end_run(solution) if self.return_if_not_found else self._end_run("Fail.")

        # Also if too many iterations stopping.
        logger.debug(f"[{self.bi}] Max iterations reached, stopping.")
        return self._end_run(solution) if self.return_if_not_found else self._end_run("Fail.")

    def generate_initial_solution(self, step, timestep, stmt):
        logger.debug(f"[{self.bi}] Generating initial solution.")

        # Ask for solution
        convo = [
            {"role": "developer", "content": self.prompts["system"]},
            {"role": "user", "content": stmt},
        ]
        convo = self._query(self.client, convo)

        # Self improve
        logger.debug(f"[{self.bi}] Self improving initial solution.")
        convo.append({"role": "user", "content": self.prompts["self_improvement"]})
        convo = self._query(self.client, convo)

        # Record step to history and return solution
        self._add_history(step, timestep, convo)
        solution = convo[-1]["content"]
        solution = get_substring(solution, ["</think>", "</summary>"], mode="after")
        return timestep + 1, solution

    def verify_solution(self, step, timestep, stmt, solution):
        # Extract detailed solution and get verification verdict
        solution = get_substring(solution, "Detailed Solution", mode="after")
        verification_query = f"""
            ======================================================================
            ### Problem ###

            {stmt}

            ======================================================================
            ### Solution ###

            {solution}

            {self.prompts['verification_reminder']}
        """
        convo = [
            {"role": "developer", "content": self.prompts["verification_system"]},
            {"role": "user", "content": verification_query},
        ]
        convo = self._query(self.client, convo)
        verdict = convo[-1]["content"]
        self._add_history(f"{step}-ver", timestep, convo)
        timestep += 1

        # Ask LLM to classify the verdict: was the solution good or do we need a bug report
        is_correct_query = f"""
            Response in "yes" or "no". Is the following statement saying the solution is correct, or does not contain critical error or a major justification gap?\n\n{verdict}
        """
        convo = self._query(self.client, [{"role": "user", "content": is_correct_query}])
        is_correct = convo[-1]["content"]
        is_correct = get_substring(is_correct, ["</think>", "</summary>"], mode="after")
        self._add_history(f"{step}-cor", timestep, convo)
        timestep += 1

        # Extract bug report if needed
        if "yes" not in is_correct.lower():
            bug_report = get_substring(verdict, "Detailed Verification", mode="before")
        else:
            bug_report = None
        return timestep, (is_correct, bug_report)
