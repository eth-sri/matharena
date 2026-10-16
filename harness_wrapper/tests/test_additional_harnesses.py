from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness_wrapper.agent import Agent
from harness_wrapper.harnesses.antigravity_cli import AntigravityCLIAgent
from harness_wrapper.harnesses.opencode import OpenCodeAgent
from harness_wrapper.harnesses.qwen_code import QwenCodeAgent
from harness_wrapper.model import Model


def _executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _agent(cls, tmp_path: Path, model: Model, **kwargs):
    return cls(
        model=model,
        executable=_executable(tmp_path, cls.installation.executable),
        dir=tmp_path,
        trace=None,
        subagents={},
        **kwargs,
    )


def test_factory_registers_additional_harness_aliases(tmp_path: Path) -> None:
    model = Model(
        "gemini-test",
        api_url="https://generativelanguage.googleapis.com",
        api_key="placeholder",
        api_type="gemini",
    )
    agent = Agent(
        type="gravity",
        model=model,
        executable=_executable(tmp_path, "agy"),
        dir=tmp_path,
        trace=None,
        subagents={},
    )

    assert isinstance(agent, AntigravityCLIAgent)
    available = set(Agent.available())
    assert {"gravity", "antigravity-cli", "qwen", "opencode"}.issubset(available)
    assert "gemini" not in available
    assert "gemini-cli" not in available


def test_antigravity_command_and_minimal_context_are_isolated(tmp_path: Path) -> None:
    model = Model(
        "gemini-test",
        api_url="https://generativelanguage.googleapis.com",
        api_key="placeholder",
        api_type="gemini",
        reasoning="max",
    )
    agent = _agent(AntigravityCLIAgent, tmp_path, model, minimal_context=True)

    command = agent.build_command("solve")
    agent.model_environment()

    assert command[command.index("--model") + 1] == "gemini-test"
    assert command[command.index("--effort") + 1] == "high"
    assert "--disable-slash-commands" in command
    assert command[command.index("--agent") + 1] == "matharena"
    assert command[command.index("--mode") + 1] == "accept-edits"
    assert "--sandbox" not in command
    settings_path = tmp_path / ".harness-home" / ".gemini" / "antigravity-cli" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    assert settings["modelProvider"] == "gemini"
    assert settings["allowNonWorkspaceAccess"] is False
    assert settings["enableTelemetry"] is False
    assert settings["permissions"]["allow"] == [
        "command(*)",
        "read_file(/work/)",
        "write_file(/work/)",
    ]
    agent_file = (
        tmp_path / ".harness-home" / ".gemini" / "config" / "agents" / "matharena" / "agent.md"
    ).read_text(encoding="utf-8")
    assert "commandExecutionPolicy: eager" in agent_file
    assert "mcpServers: []" in agent_file
    assert "skills: []" in agent_file
    assert "plugins: []" in agent_file
    assert "  - run_command" in agent_file
    assert "search_web" not in agent_file
    assert "invoke_subagent" not in agent_file
    resumed = agent.build_command("continue", resume=True, session_id="session-1")
    assert resumed[resumed.index("--conversation") + 1] == "session-1"
    assert resumed[-2:] == ["--prompt", "continue"]


def test_antigravity_normalizes_stream_tools_and_cumulative_usage(tmp_path: Path) -> None:
    model = Model(
        "gemini-test",
        api_url="https://generativelanguage.googleapis.com",
        api_key="placeholder",
        api_type="gemini",
    )
    agent = _agent(AntigravityCLIAgent, tmp_path, model)

    session = agent.normalize_event(
        {"event": "init", "conversation_id": "s1", "init": {"cwd": "/work"}}
    )[0]
    message = agent.normalize_event(
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "s1",
                "step_index": 1,
                "state": "DONE",
                "step_type": "agent_response",
                "text_delta": "done",
            },
        }
    )[0]
    tools = agent.normalize_event(
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "s1",
                "step_index": 2,
                "state": "DONE",
                "step_type": "tool",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "python -V"},
                    "output": "Python 3.13",
                },
            },
        }
    )
    failed_tool = agent.normalize_event(
        {
            "event": "step_update",
            "step_update": {
                "conversation_id": "s1",
                "step_index": 3,
                "state": "ERROR",
                "step_type": "tool",
                "tool_info": {
                    "name": "run_command",
                    "parameters": {"CommandLine": "false"},
                    "error": {"type": "exit", "message": "status 1"},
                },
            },
        }
    )
    result = agent.normalize_event(
        {
            "event": "result",
            "result": {
                "conversation_id": "s1",
                "status": "SUCCESS",
                "response": "done",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 20,
                    "thinking_tokens": 10,
                    "cache_read_tokens": 80,
                    "total_tokens": 120,
                },
            },
        }
    )
    for event in (session, message, *tools, *failed_tool, *result):
        agent._observe(event)

    assert session.session_id == "s1"
    assert message.content == "done"
    assert [event.type for event in tools] == ["tool_call", "tool_result"]
    assert tools[0].tool_name == "run_command"
    assert [event.type for event in failed_tool] == ["tool_call"]
    assert [event.type for event in result] == ["usage", "result"]

    resumed = agent.normalize_event(
        {
            "event": "result",
            "result": {
                "conversation_id": "s1",
                "status": "SUCCESS",
                "response": "continued",
                "usage": {
                    "input_tokens": 150,
                    "output_tokens": 25,
                    "thinking_tokens": 12,
                    "cache_read_tokens": 120,
                    "total_tokens": 175,
                },
            },
        }
    )
    for event in resumed:
        agent._observe(event)
    usage = agent.get_tokens()
    assert usage.input_tokens == 270
    assert usage.output_tokens == 25
    assert usage.cache_read_tokens == 120


def test_qwen_uses_documented_bare_safe_mode_and_result_usage(tmp_path: Path) -> None:
    model = Model(
        "qwen-test",
        api_url="https://example.invalid/v1",
        api_key="placeholder",
        api_type="openai",
    )
    agent = _agent(QwenCodeAgent, tmp_path, model, minimal_context=True)

    command = agent.build_command("solve")
    assert "--safe-mode" in command
    assert "--bare" in command
    assert command[command.index("--auth-type") + 1] == "openai"
    assert command[command.index("--exclude-tools") + 1] == "agent,web_fetch,web_search"
    assert command[-2:] == ["--prompt", "solve"]

    events = agent.normalize_event(
        {
            "type": "result",
            "subtype": "success",
            "session_id": "q1",
            "result": "done",
            "usage": {
                "input_tokens": 50,
                "output_tokens": 10,
                "total_tokens": 85,
                "cache_read_input_tokens": 30,
            },
        }
    )
    assert [event.type for event in events] == ["result", "usage"]
    for event in events:
        agent._observe(event)
    assert agent.get_tokens().cache_read_tokens == 30
    assert agent.get_tokens().input_tokens == 50
    assert agent.get_tokens().output_tokens == 35  # 10 visible + 25 reasoning.


@pytest.mark.parametrize(
    "failure", [{"is_error": True}, {"error": {"message": "Command failed"}}]
)
def test_qwen_uses_native_gemini_proxy_and_preserves_failed_tool_results(
    tmp_path: Path, failure: dict[str, object],
) -> None:
    model = Model(
        "gemini-test",
        api_url="https://generativelanguage.googleapis.com",
        api_key="child-placeholder",
        api_type="gemini",
        reasoning="low",
    )
    agent = _agent(QwenCodeAgent, tmp_path, model, minimal_context=True)
    agent._request_capture_url = "http://127.0.0.1:1234/secret"

    command = agent.build_command("solve")
    environment = agent.model_environment()

    assert command[command.index("--auth-type") + 1] == "gemini"
    assert environment["GOOGLE_GEMINI_BASE_URL"] == "http://127.0.0.1:1234/secret"
    assert environment["GEMINI_API_KEY"] == "child-placeholder"
    raw = {
        "type": "user",
        "session_id": "q1",
        "message": {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "failed-call",
                    "content": "NameError: name 'PbLE' is not defined",
                    **failure,
                }
            ],
        },
    }
    events = agent.normalize_event(raw)
    assert len(events) == 1
    event = events[0]
    assert event.type == "tool_result"
    assert event.role == "tool"
    assert event.tool_call_id == "failed-call"
    assert event.session_id == "q1"
    assert event.content == "NameError: name 'PbLE' is not defined"
    assert event.to_dict()["raw"] == raw


def test_opencode_uses_native_gemini_provider(tmp_path: Path) -> None:
    model = Model(
        "gemini-test",
        api_url="https://generativelanguage.googleapis.com",
        api_key="child-placeholder",
        api_type="gemini",
        reasoning="low",
    )
    agent = _agent(OpenCodeAgent, tmp_path, model, minimal_context=True)
    agent._request_capture_url = "http://127.0.0.1:1234/secret"

    command = agent.build_command("solve")
    config = json.loads(agent.model_environment()["OPENCODE_CONFIG_CONTENT"])
    provider = config["provider"]["harness-wrapper"]

    assert command[command.index("--variant") + 1] == "low"
    assert provider["npm"] == "@ai-sdk/google"
    assert provider["options"]["baseURL"] == ("http://127.0.0.1:1234/secret/v1beta")
    assert provider["options"]["apiKey"] == "child-placeholder"


@pytest.mark.parametrize("reasoning", [0, 625])
def test_opencode_inline_provider_is_minimal_and_usage_includes_cache(
    tmp_path: Path, reasoning: int
) -> None:
    model = Model(
        "vendor/model-test",
        api_url="https://example.invalid/v1",
        api_key="placeholder",
        api_type="openai",
        reasoning="high",
    )
    agent = _agent(OpenCodeAgent, tmp_path, model, minimal_context=True)
    agent._request_capture_url = "http://127.0.0.1:1234/secret/v1"

    command = agent.build_command("solve")
    config = json.loads(agent.model_environment()["OPENCODE_CONFIG_CONTENT"])

    assert "--pure" in command
    assert "--auto" in command
    assert command[command.index("--model") + 1] == "harness-wrapper/vendor/model-test"
    assert command[command.index("--variant") + 1] == "high"
    assert config["enabled_providers"] == ["harness-wrapper"]
    assert "limit" not in config["provider"]["harness-wrapper"]["models"]["vendor/model-test"]
    assert "compaction" not in config
    assert config["plugin"] == []
    assert config["tools"] == {"task": False, "skill": False, "webfetch": False, "websearch": False}
    assert config["mcp"] == {}
    assert config["subagent_depth"] == 0
    assert config["agent"]["title"]["disable"] is True
    assert config["agent"]["summary"]["disable"] is True
    assert config["agent"]["build"]["permission"]["websearch"] == "deny"
    assert config["provider"]["harness-wrapper"]["options"]["baseURL"].startswith(
        "http://127.0.0.1:1234/secret"
    )

    event = agent.normalize_event(
        {
            "type": "step_finish",
            "sessionID": "o1",
            "part": {
                "tokens": {
                    "input": 20,
                    "output": 7,
                    "reasoning": reasoning,
                    "cache": {"read": 80, "write": 3},
                }
            },
        }
    )[0]
    agent._observe(event)
    usage = agent.get_tokens()
    assert usage.input_tokens == 100
    assert usage.output_tokens == 7 + reasoning
    assert usage.cache_read_tokens == 80
    assert usage.cache_write_tokens == 3

    failed_tool = agent.normalize_event(
        {
            "type": "tool_use",
            "sessionID": "o1",
            "part": {
                "callID": "failed-call",
                "tool": "bash",
                "state": {
                    "status": "failed",
                    "input": {"command": "false"},
                    "error": "secret failed output",
                },
            },
        }
    )
    assert [event.type for event in failed_tool] == ["tool_call"]


def test_antigravity_provider_usage_includes_thoughts(tmp_path: Path) -> None:
    model = Model(
        "gemini-test", api_url="https://generativelanguage.googleapis.com",
        api_key="placeholder", api_type="gemini",
    )
    agent = _agent(AntigravityCLIAgent, tmp_path, model)
    agent._capture_model_response("/streamGenerateContent", {
        "usageMetadata": {
            "promptTokenCount": 100, "candidatesTokenCount": 14,
            "thoughtsTokenCount": 625, "cachedContentTokenCount": 30,
            "totalTokenCount": 739,
        },
    })
    assert agent._take_api_usage() == {
        "input_tokens": 100, "output_tokens": 639, "cache_read_tokens": 30,
    }
    assert agent._take_api_usage() is None


@pytest.mark.parametrize("protocol, overrides, output", [
    ("openai", {"max_tokens": 131072, "reasoning": {"enabled": True}}, 131072),
    ("openai", {"max_completion_tokens": 65536}, 65536),
    ("gemini", {"generationConfig": {"maxOutputTokens": 65536}}, 65536),
])
def test_opencode_compaction_reserves_proxy_output_budget(tmp_path, protocol, overrides, output):
    model = Model(
        "vendor/model-test", api_url="https://example.invalid/v1", api_key="placeholder",
        api_type=protocol, request_overrides=overrides,
    )
    agent = _agent(OpenCodeAgent, tmp_path, model, model_context_window=1000000, minimal_context=True)
    config = json.loads(agent.model_environment()["OPENCODE_CONFIG_CONTENT"])
    definition = config["provider"]["harness-wrapper"]["models"]["vendor/model-test"]
    assert definition["limit"] == {"context": 1000000, "input": 1000000 - output, "output": output}
    assert config["compaction"] == {"auto": True, "reserved": 20000}
    # OpenCode's input-limit branch must reserve the actual proxy output cap,
    # even when its internal maxOutputTokens() returns a smaller number.
    threshold = definition["limit"]["input"] - config["compaction"]["reserved"]
    assert threshold + output + 20000 == 1000000
    assert model.request_overrides() == overrides
    if overrides.get("reasoning", {}).get("enabled"):
        assert definition["reasoning"] is True
        assert "--variant" not in agent.build_command("solve")


@pytest.mark.parametrize("context", [0, -1, True, 1000.5])
def test_opencode_rejects_invalid_context_window(tmp_path, context):
    model = Model("test", api_url="https://example.invalid/v1", api_key="placeholder", api_type="openai")
    with pytest.raises(ValueError, match="model_context_window must be a positive integer"):
        _agent(OpenCodeAgent, tmp_path, model, model_context_window=context)


@pytest.mark.parametrize("output", [None, 0, -1, True, 1.5, 1000000])
def test_opencode_requires_output_budget_to_fit_context(tmp_path, output):
    model = Model(
        "test", api_url="https://example.invalid/v1", api_key="placeholder",
        api_type="openai", request_overrides={"max_tokens": output},
    )
    agent = _agent(OpenCodeAgent, tmp_path, model, model_context_window=1000000)
    with pytest.raises(ValueError, match="positive API output limit"):
        agent.model_environment()
