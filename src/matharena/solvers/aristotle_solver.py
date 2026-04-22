import time
from collections import UserDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, override

from loguru import logger

from matharena.request_logger import request_logger
from matharena.solvers import BaseSolver, SolverResponse
from matharena.tools.aristotle_execution import (
    DEFAULT_ARISTOTLE_MATHLIB_REV,
    DEFAULT_ARISTOTLE_POLLING_INTERVAL_SECONDS,
    DEFAULT_ARISTOTLE_TOOLCHAIN,
    execute_aristotle,
)


class _PromptFields(UserDict):
    def __missing__(self, key):
        return "{" + key + "}"


class AristotleSolver(BaseSolver):
    def __init__(self, solver_config, default_prompt_template, default_api_client_args, last_chance_prompt):
        super().__init__(solver_config, default_prompt_template, default_api_client_args, last_chance_prompt)
        self.concurrent_requests = default_api_client_args.get("concurrent_requests", 1)
        self.polling_interval_seconds = default_api_client_args.get(
            "polling_interval_seconds", DEFAULT_ARISTOTLE_POLLING_INTERVAL_SECONDS
        )
        self.toolchain = default_api_client_args.get("aristotle_toolchain", DEFAULT_ARISTOTLE_TOOLCHAIN)
        self.mathlib_rev = default_api_client_args.get("aristotle_mathlib_revision", DEFAULT_ARISTOTLE_MATHLIB_REV)

    def build_query(self, text: str | dict[str, Any] | None, image_b64):
        if image_b64 is not None:
            raise ValueError("AristotleSolver does not support image inputs.")

        if isinstance(text, dict):
            prompt_fields = {k: v for k, v in text.items() if v is not None}
            prompt_fields.setdefault("problem", prompt_fields.get("problem", ""))
        else:
            prompt_fields = {"problem": "" if text is None else text}

        prompt = self.default_prompt_template.format_map(_PromptFields(prompt_fields))
        return [{"role": "user", "content": prompt, "tool_context": prompt_fields}]

    def _solve_one(self, idx, text, image_b64, batch_idx_to_problem_idx, batch_idx_to_run_idx):
        conversation = self.build_query(text, image_b64)
        tool_context = conversation[0].get("tool_context", {})
        prompt = conversation[0]["content"]

        ts = time.strftime("%m%d-%H:%M:%S", time.localtime(time.time()))
        ts += f".{int((time.time() % 1) * 1_000_000):06d}"
        request_logger.log_request(
            ts=ts,
            batch_idx=idx,
            request={
                "api": "aristotle",
                "prompt": prompt,
                "toolchain": self.toolchain,
                "mathlib_revision": self.mathlib_rev,
            },
            run_idx=batch_idx_to_run_idx[idx],
        )

        result = execute_aristotle(
            prompt=prompt,
            problem_statement=tool_context.get("problem"),
            formal_statement=tool_context.get("formal_statement"),
            problem_idx=batch_idx_to_problem_idx[idx],
            run_idx=batch_idx_to_run_idx[idx],
            toolchain=self.toolchain,
            mathlib_rev=self.mathlib_rev,
            polling_interval_seconds=self.polling_interval_seconds,
        )

        request_logger.log_response(
            ts=ts,
            batch_idx=idx,
            response={
                "project_id": result["project_id"],
                "status": result["status"],
                "output_summary": result["output_summary"],
            },
        )

        assistant_content = f"```lean\n{result['code'].strip()}\n```"
        conversation.append({"role": "assistant", "content": assistant_content})
        detailed_cost = {
            "cost": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cached_write_tokens": 0,
            "output_tokens": 0,
            "n_retries": 0,
            "time": result["time"],
            "request_time": result["time"],
        }
        return SolverResponse(idx, conversation, detailed_cost, history=None)

    @override
    def solve_batch(self, stmt_batch, batch_idx_to_problem_idx, batch_idx_to_run_idx):
        with ThreadPoolExecutor(max_workers=self.concurrent_requests) as executor:
            futures = [
                executor.submit(
                    self._solve_one,
                    idx,
                    text,
                    image_b64,
                    batch_idx_to_problem_idx,
                    batch_idx_to_run_idx,
                )
                for idx, (text, image_b64) in enumerate(stmt_batch)
            ]
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception:
                    logger.exception("Aristotle run failed; continuing.")

    @override
    def last_chance(self, previous_response):
        return previous_response
