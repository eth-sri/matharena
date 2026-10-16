"""Use the normal model backends for curation's batched text queries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from uuid import uuid4


class HarnessQueryClient:
    """Adapt the shared HarnessSolver without duplicating its execution lifecycle."""

    def __init__(self, config: dict[str, Any], *, container_runtime: str = "docker"):
        from matharena.solvers.harness_solver import HarnessSolver

        config = config.copy()
        harness_config = dict(config.get("harness_config") or {})
        # Curation uses only the supplied source and the selected model.
        config["allow_harness"] = False
        harness_config["auto_fallback"] = False
        log_root = Path(os.environ.get("MATHARENA_REQUEST_LOG_DIR", "logs/harness_workspaces"))
        harness_config.setdefault(
            "workspace_root", str(log_root / "curation" / uuid4().hex / "p{problem_idx}_r{run_idx}")
        )
        config["harness_config"] = harness_config
        self.solver = HarnessSolver({}, "{problem}", config, "", container_runtime=container_runtime)
        self.next_query_idx = 0

    def run_queries(self, queries):
        batch = []
        for query in queries:
            if len(query) != 1 or query[0].get("role") != "user":
                raise ValueError("Curation harness queries require exactly one user message")
            prompt = query[0].get("content")
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError("Curation harness queries require a nonempty text prompt")
            batch.append((prompt, None))
        if not batch:
            return
        problem_indices = {i: self.next_query_idx + i for i in range(len(batch))}
        self.next_query_idx += len(batch)
        for response in self.solver.solve_batch(batch, problem_indices, dict.fromkeys(problem_indices, 0)):
            # Keep complete local log references even when the curation cache is
            # outside the repository, where the solver's display paths use basenames.
            agent = self.solver._response_agents.pop(id(response), None)
            if agent is not None and response.history:
                response.history[-1]["workspace"] = str(agent.root)
                request_log = agent.model_request_log_path
                response.history[-1]["model_requests"] = str(request_log) if request_log is not None else None
            cost = {**response.detailed_cost, "history": response.history}
            yield response.idx, response.conversation, cost


def create_query_client(model_config: dict[str, Any], *, container_runtime: str = "docker"):
    config = model_config.copy()
    client_type = config.pop("type", "api")
    if client_type not in {"api", "pure_model"}:
        raise ValueError(f"Unsupported curation query client type: {client_type!r}")
    if config.get("harness"):
        return HarnessQueryClient(config, container_runtime=container_runtime)
    config.pop("harness", None)
    config.pop("harness_version", None)
    config.pop("harness_config", None)
    from matharena.api_client import APIClient

    return APIClient(**config)


__all__ = ["create_query_client"]
