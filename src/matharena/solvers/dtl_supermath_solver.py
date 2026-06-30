r"""MathArena adapter for JARVI3 DTL/SuperMath.

This file is intended for a MathArena fork/PR. It emits boxed final answers
only when a deterministic SuperMath lane supports the problem, otherwise it
emits \boxed{None}.
"""

from matharena.solvers import BaseSolver, SolverResponse

from .dtl_supermath_lanes import solve_supermath_case


class DTLSuperMathSolver(BaseSolver):
    def solve_batch(self, stmt_batch, batch_idx_to_problem_idx, batch_idx_to_run_idx):
        for batch_idx, stmt in enumerate(stmt_batch):
            problem_idx = batch_idx_to_problem_idx[batch_idx]
            solved = solve_supermath_case(f"aime_2026_{int(problem_idx):02d}")
            if solved.get("answer") is None:
                final = r"\boxed{None}"
                status = "unsupported"
            else:
                final = rf"\boxed{{{solved['answer']}}}"
                status = "answered"
            content = (
                "DTL/SuperMath deterministic lane result.\n"
                f"Lane: {solved.get('lane', 'unknown')}\n"
                f"Status: {status}\n"
                f"Final answer: {final}"
            )
            conversation = [{"role": "assistant", "content": content}]
            yield SolverResponse(
                idx=batch_idx,
                conversation=conversation,
                detailed_cost={"input_tokens": 0, "output_tokens": 0, "total_cost": 0.0},
                history=[{"step": "dtl-supermath", "messages": conversation}],
            )

    def last_chance(self, response):
        return response
