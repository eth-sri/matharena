"""Offline coverage of curation through the normal harness backend."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from harness_wrapper import AgentEvent, TokenUsage
from matharena.arxivbench_utils import conversation_response_text, load_model_config
from matharena.query_client import HarnessQueryClient, create_query_client
from matharena.solvers.harness_solver import HarnessSolver


def query(text):
    return [{"role": "user", "content": text}]


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_curation_uses_normal_harness_and_preserves_batches_logs_and_failures(monkeypatch, tmp_path, runtime):
    monkeypatch.setenv("MATHARENA_REQUEST_LOG_DIR", str(tmp_path))
    config = load_model_config("configs/models/openai/gpt-6-astra.yaml")
    config["allow_harness"] = True
    config["harness_config"].update(tools_enabled=True, auto_fallback=True)
    original = deepcopy(config)
    agents = []

    class FakeAgent:
        def __init__(self, **kwargs):
            self.options = kwargs
            self.root = kwargs["dir"]
            self.session_id = f"session-{len(agents)}"
            self.model_request_log_path = kwargs["dir"] / "model_requests.jsonl"
            agents.append(self)

        def run(self, prompt):
            self.model_request_log_path.write_text(prompt)
            if prompt == "missing":
                raise RuntimeError("offline simulated harness failure")
            return [AgentEvent(type="message", role="assistant", content='{"decision":"accept"}')]

        def get_tokens(self):
            return TokenUsage(input_tokens=10000, cache_read_tokens=500, output_tokens=25)

    # Exercise real HarnessSolver batching, workspaces, responses and costs; replace
    # only CLI preparation, authentication, Docker, and the model process.
    monkeypatch.setattr(HarnessSolver, "_prepare_harness_cli", lambda self: None)
    monkeypatch.setattr(HarnessSolver, "_build_model", lambda self: "offline-model")
    monkeypatch.setattr(HarnessSolver, "_build_sandbox", lambda self, path: SimpleNamespace(root=path))
    monkeypatch.setattr(HarnessSolver, "_agent_executable", lambda self: None)
    monkeypatch.setattr(HarnessSolver, "_agent_environment", lambda self: {})
    monkeypatch.setattr("matharena.solvers.harness_solver.Agent", FakeAgent)
    client = create_query_client(config, container_runtime=runtime)
    assert client.solver.container_runtime == runtime

    assert isinstance(client, HarnessQueryClient)
    assert config == original
    assert client.solver.config["model"] == "gpt-6-astra"
    assert client.solver.config["reasoning_effort"] == "max"
    assert client.solver.harness_version == original["harness_version"]
    assert client.solver.harness_config["auth"] == "subscription"
    assert client.solver.concurrent_requests == original["concurrent_requests"]
    first = list(client.run_queries([query("first"), query("missing"), query("third")]))
    second = list(client.run_queries([query("next batch")]))
    assert {index for index, _, _ in first} == {0, 2}
    assert [index for index, _, _ in second] == [0]
    assert client.solver._response_agents == {}
    assert len({agent.options["dir"] for agent in agents}) == 4
    for agent in agents:
        assert agent.options["tools_enabled"] is False
        assert agent.options["auto_fallback"] is False
        assert agent.options["minimal_context"] is True
        assert agent.options["subagents"] == {}
        assert agent.options["model_context_window"] == original["harness_config"]["model_context_window"]
    for _, conversation, cost in first + second:
        assert conversation_response_text(conversation) == '{"decision":"accept"}'
        assert cost["cost"] == pytest.approx(0.09675)
        assert cost["input_tokens"] == 10000
        assert Path(cost["history"][0]["model_requests"]).is_file()
    assert agents[0].model_request_log_path.read_text() in {"first", "missing", "third"}
    assert agents[-1].model_request_log_path.read_text() == "next batch"
    other_client = create_query_client(config)
    assert client.solver._workspace_for(0, 0) != other_client.solver._workspace_for(0, 0)


def test_empty_or_invalid_queries_never_start_the_harness(monkeypatch, tmp_path):
    monkeypatch.setenv("MATHARENA_REQUEST_LOG_DIR", str(tmp_path))
    client = create_query_client(load_model_config("configs/models/openai/gpt-6-astra.yaml"))

    def forbidden(*args):
        raise AssertionError("No harness execution expected")

    monkeypatch.setattr(client.solver, "solve_batch", forbidden)
    assert list(client.run_queries([])) == []
    for invalid in ([], [{"role": "system", "content": "x"}], query(""), query("  ")):
        with pytest.raises(ValueError):
            list(client.run_queries([invalid]))


def test_ordinary_api_config_keeps_its_backend(monkeypatch):
    observed = {}

    def fake_api(**kwargs):
        observed.update(kwargs)
        return "api-client"

    monkeypatch.setattr("matharena.api_client.APIClient", fake_api)
    config = {"model": "api-model", "api": "openai", "harness": False}
    assert create_query_client(config, container_runtime="podman") == "api-client"
    assert observed == {"model": "api-model", "api": "openai"}
    assert config["harness"] is False
