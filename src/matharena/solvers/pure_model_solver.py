"""This module defines a Chain-of-Thought (CoT) solver for math problems."""

from typing import Any, override

from matharena.api_client import APIClient
from matharena.solvers import BaseSolver, SolverResponse


class PureModelSolver(BaseSolver):
    """
    A solver that wraps a pure model, prompting it once with the problem statement.
    """

    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        """
        Initializes the solver.
        """
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        self.client = APIClient(**default_api_client_args)

    def build_query(self, text: str | None, image_b64):
        if text is None:
            text = "See image."
        prompt = self.default_prompt_template.format(problem=text)
        if image_b64 is not None:
            # NOTE: OpenAI format, needs to be mangled inside for Gemini, Grok
            content = [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}", "detail": "high"},
            ]
        else:
            content = prompt
        return [{"role": "user", "content": content}]

    @override
    def solve_batch(self, stmt_batch: list[tuple[str, Any]], batch_idx_to_problem_idx: dict[int, int], batch_idx_to_run_idx: dict[int, int]):
        """
        Solves a batch of problems.

        Args:
            stmt_batch (list[tuple[str, Any]]): A batch of problem statements as (text, image) pairs.
            batch_idx_to_problem_idx (dict[int, int]): A mapping from batch indices to original problem indices.
            batch_idx_to_run_idx (dict[int, int]): A mapping from batch indices to run indices.

        Yields:
            solver_response: A SolverResponse object containing the batch_index, the conversation array, detailed cost, and history for each problem.
        """

        queries = []
        for text, image_b64 in stmt_batch:
            queries.append(self.build_query(text, image_b64))
        for idx, conversation, detailed_cost in self.client.run_queries(queries):
            # History is None for pure model solver
            yield SolverResponse(idx, conversation, detailed_cost, history=None)

    @override
    def last_chance(self, previous_response: SolverResponse) -> SolverResponse:
        """
        If the parser did not find the solution for some problem, the solver has a last chance to modify its response.

        Args:
            previous_response (SolverResponse): The response this solver previously returned for a problem.

        Returns:
            SolverResponse: The modified response after reprompting the model to report an answer.
        """

        # Run queries but there is only one
        old_queries = previous_response.conversation
        if old_queries[-1].get("type") == "cot":
            old_queries = [old_queries + [{"role": "assistent", "content": ""}]]

        new_queries = [old_queries + [{"role": "user", "content": self.last_chance_prompt}]]
        for idx, conversation, detailed_cost in self.client.run_queries(
            new_queries, no_tqdm=True, ignore_tool_calls=True
        ):
            # Important: add old cost to new cost
            for k in detailed_cost.keys():
                detailed_cost[k] += previous_response.detailed_cost.get(k, 0)
            return SolverResponse(idx, conversation, detailed_cost, history=None)  # only one
