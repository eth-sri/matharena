from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.deepseek_harness import DeepSeekHarnessAgent
from matharena.runner import Runner
from matharena.solvers.harness_solver import HarnessSolver

MODEL = "deepseek/deepseek-v4.1-flash-1017"


def make_agent(tmp_path, **kwargs):
    return DeepSeekHarnessAgent(
        model=Model(
            MODEL,
            api_url="https://openrouter.ai/api",
            api_key="placeholder",
            api_type="anthropic",
            reasoning="max",
        ),
        executable="/usr/bin/true",
        dir=tmp_path,
        trace=None,
        subagents={},
        minimal_context=True,
        **kwargs,
    )


def test_dsh_routes_openrouter_messages_without_forcing_sampling(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-only-openrouter-key")
    config = {
        "model": MODEL, "api": "openrouter", "harness": "dsh",
        "harness_version": "0.2.1-rc.5",
        "harness_config": {
            "auth": "api", "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
        },
        "reasoning_effort": "max",
        "extra_body": {"provider": {"only": ["together"], "allow_fallbacks": False}},
    }
    scaffold = {"type": "pure_model", "model_config": config, "scaffold_config": None}
    runner = SimpleNamespace(comp_name="test", competition_config={"allow_harness": True})
    assert Runner._resolve_harness(runner, scaffold) == "dsh"
    model = HarnessSolver(scaffold, "{problem}", config, "")._build_model()
    endpoint = model.endpoint_for("anthropic")
    assert endpoint.url == "https://openrouter.ai/api"
    assert endpoint.headers["Authorization"] == "Bearer host-only-openrouter-key"
    assert endpoint.api_key != "host-only-openrouter-key"
    assert model.request_overrides() == {
        "provider": {"only": ["together"], "allow_fallbacks": False}
    }
    assert model.reasoning == "max"


def test_dsh_patch_sets_limits_and_disables_uploads(tmp_path):
    agent = make_agent(
        tmp_path,
        environment={"DSH_MAX_OUTPUT_TOKENS": "384000", "DSH_CONTEXT_WINDOW": "1048576"},
    )
    agent._request_capture_url = "http://127.0.0.1:1234/route"
    env = agent.model_environment()
    assert env["DSH_HOME"] == str(tmp_path / ".harness-home/.dsh")
    assert env["DSH_TELEMETRY_DISABLED"] == "1"
    assert env["DSH_PERMISSION_MODE"] == "danger-full-access"
    patch = json.loads((tmp_path / ".harness-home/dsh-patch.yml").read_text())
    rows = {row["id"]: row for row in patch}
    assert rows["llm-deepseek"]["config"] == {
        "baseURL": "http://127.0.0.1:1234/route",
        "apiKeyEnv": "DEEPSEEK_API_KEY",
        "maxTokens": 384000,
        "models": [{"id": MODEL, "contextWindow": 1048576}],
        "reasoningEffort": "max",
    }
    assert rows["agent-default-model"]["config"]["model"] == MODEL
    for row in (
        "session-log-deepseek", "plugin-package-inventory-deepseek",
        "session-telemetry-otel", "session-title-llm", "tool-web",
        "subagent", "tool-subagent", "tool-subagent-fork", "tool-subagent-control",
        "tool-workflow", "skill", "tool-skill", "mcp-resources", "goal-round-driver",
        "tool-goal", "plan-mode", "commands",
    ):
        assert rows[row]["disabled"] is True
    assert rows["system-prompt"]["config"]["includeHarnessIdentity"] is False
    assert "local file and shell tools" in rows["system-prompt"]["config"]["personaPrefix"]


def test_dsh_commands_and_events(tmp_path):
    agent = make_agent(tmp_path)
    assert {"dsh", "deepseek-harness"}.issubset(Agent.available())
    command = agent.build_command("-")
    assert command[1:3] == ["--profile", "headless"] and command[-3:] == ["--json", "--", "-"]
    with pytest.raises(ValueError):
        agent.build_command("continue", resume=True, last=True)
    [session] = agent.normalize_event({"type": "session", "sessionId": "session-1", "cwd": "/work"})
    agent._observe(session)
    assert agent.build_command("go on", resume=True, last=True)[-4:] == [
        "--session-id", "session-1", "--", "go on",
    ]
    [call] = agent.normalize_event(
        {"type": "tool_call", "callId": "c1", "tool": "bash", "input": {"command": "ls"}}
    )
    assert (call.type, call.tool_name, call.tool_call_id, call.session_id) == (
        "tool_call", "bash", "c1", "session-1",
    )
    [done] = agent.normalize_event(
        {"type": "final", "sessionId": "session-1", "reason": {"kind": "completed"}, "text": "42"}
    )
    assert (done.type, done.content) == ("result", "42")
    [failed] = agent.normalize_event(
        {"type": "final", "sessionId": "session-1", "reason": {"kind": "error"}, "text": ""}
    )
    assert failed.type == "error"
    # dsh 0.2.1 always emits reason null; exit status alone reports failure.
    [legacy] = agent.normalize_event(
        {"type": "final", "sessionId": "session-1", "reason": None, "text": "42"}
    )
    assert (legacy.type, legacy.content) == ("result", "42")


def test_dsh_request_filter_uses_adaptive_thinking(tmp_path):
    agent = make_agent(tmp_path)
    payload = {"thinking": {"type": "enabled"}, "output_config": {"effort": "max"}}
    agent._filter_request("/v1/messages", payload)
    assert payload == {"thinking": {"type": "adaptive"}, "output_config": {"effort": "max"}}
    budgeted = {"thinking": {"type": "enabled", "budget_tokens": 1024}}
    agent._filter_request("/v1/messages", budgeted)
    assert budgeted["thinking"]["type"] == "enabled"
    with pytest.raises(ValueError):
        agent._filter_request("/v1/chat/completions", {})


def test_dsh_request_filter_keeps_only_local_tools(tmp_path):
    agent = make_agent(tmp_path)
    names = ["bash", "read", "subagent", "workflow", "skill", "create_goal", "web_search",
             "mcp__x__y", "todo_write", "job_output"]
    payload = {"tools": [{"name": name} for name in names]}
    agent._filter_request("/v1/messages", payload)
    assert [tool["name"] for tool in payload["tools"]] == ["bash", "read", "todo_write", "job_output"]


def test_dsh_counts_each_streamed_message_once(tmp_path):
    agent = make_agent(tmp_path)
    for message_id, output in (("m1", 10), ("m2", 5)):
        agent._capture_model_response("/v1/messages", {
            "type": "message_start",
            "message": {"id": message_id, "usage": {"input_tokens": 100, "cache_read_input_tokens": 50}},
        })
        for cumulative in (output // 2, output):
            agent._capture_model_response(
                "/v1/messages", {"type": "message_delta", "usage": {"output_tokens": cumulative}}
            )
    tokens = agent.get_tokens()
    assert (tokens.input_tokens, tokens.output_tokens, tokens.cache_read_tokens) == (200, 15, 100)


def test_dsh_stream_drops_null_usage_counters():
    from harness_wrapper.harnesses.deepseek_harness.adapter import _drop_null_usage

    start = (
        b'data: {"type":"message_start","message":{"id":"m","usage":'
        b'{"input_tokens":0,"cache_read_input_tokens":null,"cache_creation":null}}}\n'
    )
    event = json.loads(_drop_null_usage(start)[5:])
    assert event["message"]["usage"] == {"input_tokens": 0, "cache_creation": None}
    assert _drop_null_usage(start).endswith(b"}\n")
    delta = b'data: {"type":"message_delta","usage":{"cache_creation_input_tokens":null}}\r\n'
    assert json.loads(_drop_null_usage(delta)[5:])["usage"] == {}
    untouched = b'data: {"type":"content_block_delta","delta":{"text":"null"}}\n'
    assert _drop_null_usage(untouched) is untouched
