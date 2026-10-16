"""Native web tools follow the sandbox's network permission without model calls."""

import json

import pytest

from harness_wrapper import Model, Sandbox
from harness_wrapper.harnesses import (
    AntigravityCLIAgent, ClaudeCodeAgent, CodexCLIAgent, DeepCodeAgent,
    GrokBuildAgent, KimiCodeAgent, MuseCodeAgent, OpenCodeAgent, QwenCodeAgent,
)


@pytest.fixture(params=[None, False, True], ids=["default", "offline", "online"])
def network(request):
    return request.param


def make_agent(kind, tmp_path, network, **kwargs):
    return kind(
        model=Model("test", api_url="https://example.invalid/v1", api_key="placeholder",
                    api_type=kind.accepted_api_types[0]),
        env=None if network is None else Sandbox(tmp_path, network_enabled=network),
        dir=tmp_path, executable="/usr/bin/true", trace=None, subagents={},
        minimal_context=True, **kwargs,
    )


@pytest.mark.parametrize("kind", [
    AntigravityCLIAgent, ClaudeCodeAgent, CodexCLIAgent, DeepCodeAgent,
    GrokBuildAgent, KimiCodeAgent, MuseCodeAgent, OpenCodeAgent, QwenCodeAgent,
])
def test_native_web_tools_follow_network_permission(tmp_path, network, kind):
    online = network is True
    agent = make_agent(kind, tmp_path, network)
    command = agent.build_command("test the tools")
    environment = agent.model_environment()
    assert bool(agent._offline_models) == (network is False)

    if kind is CodexCLIAgent:
        assert f'web_search="{"live" if online else "disabled"}"' in command
        enabled = {command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--enable"}
        disabled = {command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--disable"}
        for feature in ("standalone_web_search", "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser"):
            assert (feature in enabled) == online
            assert (feature in disabled) != online
        assert "multi_agent" in disabled
    elif kind is ClaudeCodeAgent:
        tools = set(command[command.index("--tools") + 1].split(","))
        assert tools == ({"Bash", "WebSearch", "WebFetch"} if online else {"Bash"})
    elif kind is KimiCodeAgent:
        text = (tmp_path / ".harness_wrapper/minimal-kimi-agent.md").read_text()
        tools = {line.strip()[2:] for line in text.splitlines() if line.startswith("  - ")}
        assert tools == ({"Bash", "WebSearch", "FetchURL"} if online else {"Bash"})
        assert "subagents: []" in text
    elif kind is QwenCodeAgent:
        assert ("--bare" in command) != online
        assert "--safe-mode" in command
        excluded = command[command.index("--exclude-tools") + 1].split(",")
        assert ("web_search" not in excluded) == online
        assert ("web_fetch" not in excluded) == online
        assert "agent" in excluded
    elif kind is OpenCodeAgent:
        config = json.loads(environment["OPENCODE_CONFIG_CONTENT"])
        for tool in ("webfetch", "websearch"):
            assert config["tools"][tool] == online
            assert config["agent"]["build"]["permission"][tool] == ("allow" if online else "deny")
        assert environment["OPENCODE_ENABLE_EXA"] == ("true" if online else "false")
        assert not config["tools"]["task"] and not config["tools"]["skill"]
    elif kind is AntigravityCLIAgent:
        home = tmp_path / ".harness-home/.gemini"
        text = (home / "config/agents/matharena/agent.md").read_text()
        tools = {line.strip()[2:] for line in text.splitlines() if line.startswith("  - ")}
        for tool in ("search_web", "read_url_content", "view_content_chunk", "browser_subagent"):
            assert (tool in tools) == online
        permissions = json.loads((home / "antigravity-cli/settings.json").read_text())["permissions"]
        assert ("read_url(*)" in permissions["allow"]) == online
        assert ("read_url(*)" in permissions["deny"]) != online
        assert "mcp(*)" in permissions["deny"]
    elif kind is GrokBuildAgent:
        config = (tmp_path / ".harness-home/.grok/config.toml").read_text()
        assert f"supports_backend_search = {str(online).lower()}" in config
        assert f"web_fetch = {str(online).lower()}" in config
        assert 'web_search = "harness-wrapper"' in config
        assert environment["GROK_WEB_FETCH"] == ("1" if online else "0")
        assert environment["GROK_BACKEND_SEARCH"] == ("1" if online else "0")
        assert ("--disable-web-search" not in command) == online
        assert "--no-subagents" in command
    elif kind is MuseCodeAgent:
        assert ("--disable-web-tools" not in command) == online
        assert "--no-foreign-personal-context" in command
    elif kind is DeepCodeAgent:
        config = json.loads((tmp_path / ".harness-home/.deepcode/settings.json").read_text())
        assert ("network" not in config["permissions"]["deny"]) == online
        assert "mcp" in config["permissions"]["deny"]


def test_codex_tool_free_stays_tool_free_with_internet(tmp_path, network):
    agent = make_agent(CodexCLIAgent, tmp_path, network, tools_enabled=False)
    command = agent.build_command("reason only")
    assert 'web_search="disabled"' in command
    assert "standalone_web_search" not in [command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--enable"]
    assert agent._effective_request_overrides()["tools"] == []


def test_muse_preserves_web_tools_and_compaction(tmp_path, network):
    agent = make_agent(MuseCodeAgent, tmp_path, network)
    payload = {"tools": [
        {"type": "namespace", "name": "muse", "tools": [
            {"type": "function", "name": name}
            for name in ("bash", "web_search", "web_fetch", "subagent_spawn", "read_skill")
        ]},
        {"type": "web_search"},
        {"type": "web_fetch"},
    ]}
    agent._filter_request("/responses", payload)
    assert [tool["name"] for tool in payload["tools"][0]["tools"]] == (
        ["bash", "web_search", "web_fetch"] if network else ["bash"]
    )
    assert len(payload["tools"]) == (3 if network else 1)
    payload["tools"].append({"type": "function", "name": "generate_summary"})
    agent._filter_request("/responses", payload)
    assert payload["tools"] == [{"type": "function", "name": "generate_summary"}]


def test_grok_preserves_web_tools_and_auxiliary_requests(tmp_path, network):
    agent = make_agent(GrokBuildAgent, tmp_path, network)
    payload = {"tools": [
        {"type": "function", "name": "web_fetch"},
        {"type": "web_search"},
        {"type": "x_search"},
        {"type": "function", "name": "spawn_subagent"},
    ]}
    agent._filter_request("/responses", payload)
    assert len(payload["tools"]) == (3 if network else 0)
    assert all(tool.get("name") != "spawn_subagent" for tool in payload["tools"])
    payload = {"tools": [{"type": "function", "name": "session_title"}]}
    agent._filter_request("/responses", payload)
    assert payload["tools"] == [{"type": "function", "name": "session_title"}]


def test_deepcode_preserves_web_search_and_tool_results(tmp_path, network):
    agent = make_agent(DeepCodeAgent, tmp_path, network)
    messages = [{"role": "tool", "tool_call_id": "call", "content": "search results"}]
    payload = {"messages": messages, "tools": [
        {"type": "function", "function": {"name": name}}
        for name in ("bash", "WebSearch", "skill")
    ]}
    agent._filter_request("/chat/completions", payload)
    assert [tool["function"]["name"] for tool in payload["tools"]] == (
        ["bash", "WebSearch"] if network else ["bash"]
    )
    assert payload["messages"] == messages


@pytest.mark.parametrize("host", ["openrouter.ai", "us.openrouter.ai", "example.invalid"])
def test_qwen_openrouter_search_preserves_cli_tools(tmp_path, network, host):
    agent = make_agent(QwenCodeAgent, tmp_path, network)
    agent.model = Model("qwen/test", api_url=f"https://{host}/api/v1", api_key="placeholder")
    local_tool = {"type": "function", "function": {"name": "run_shell_command"}}
    messages = [{"role": "tool", "tool_call_id": "call", "content": "shell output"}]
    payload = {"tools": [local_tool], "messages": messages, "reasoning": {"effort": "max"}}
    agent._filter_request("/api/v1/chat/completions", payload)
    agent._filter_request("/api/v1/chat/completions", payload)
    expected = [local_tool]
    if network is True and host != "example.invalid":
        expected.append({"type": "openrouter:web_search"})
    assert payload == {"tools": expected, "messages": messages, "reasoning": {"effort": "max"}}


@pytest.mark.parametrize(("protocol", "path", "payload"), [
    ("openai", "/api/v1/chat/completions", {"messages": []}),
    ("openai", "/api/v1/chat/completions", {"tools": []}),
    ("openai", "/api/v1/responses", {"tools": [{"type": "function"}]}),
    ("anthropic", "/api/v1/messages", {"tools": [{"name": "shell"}]}),
    ("gemini", "/v1beta/models/test:generateContent", {"tools": [{"functionDeclarations": []}]}),
])
def test_qwen_search_leaves_auxiliary_and_other_protocols_unchanged(tmp_path, protocol, path, payload):
    agent = make_agent(QwenCodeAgent, tmp_path, True)
    agent.model = Model("test", api_url="https://openrouter.ai/api/v1", api_key="placeholder", api_type=protocol)
    before = json.dumps(payload)
    agent._filter_request(path, payload)
    assert json.dumps(payload) == before


@pytest.mark.parametrize("lite", [False, True])
def test_codex_openrouter_native_search_preserves_local_tools_and_context(tmp_path, network, lite):
    agent = make_agent(CodexCLIAgent, tmp_path, network)
    agent.model = Model("openai/gpt-6-astra", api_url="https://openrouter.ai/api/v1", api_key="placeholder")
    local = [{"type": "custom", "name": "exec", "format": {"type": "text"}}]
    history = [
        {"type": "custom_tool_call_output", "call_id": "prior", "output": "prior result"},
        {"type": "agent_message", "author": "/root/worker", "recipient": "/root",
         "content": [{"type": "encrypted_content", "encrypted_content": "opaque-message"}]},
    ]
    payload = {"input": history.copy()}
    if lite:
        payload["input"].insert(0, {"type": "additional_tools", "tools": local})
    else:
        payload["tools"] = local.copy()
    before = json.dumps(payload)
    agent._filter_request("/api/v1/responses", payload)
    agent._filter_request("/api/v1/responses", payload)
    if network is not True:
        assert json.dumps(payload) == before
        return
    assert payload["input"] == json.loads(before)["input"]
    assert payload["tools"] == ([] if lite else local) + [
        {"type": "web_search", "external_web_access": True},
    ]
    if lite:
        assert payload["input"][0]["tools"] == local


@pytest.mark.parametrize("path,payload", [
    ("/api/v1/responses/compact", {"input": [{"type": "additional_tools", "tools": [{}]}]}),
    ("/api/v1/responses", {"input": "Summarize this conversation."}),
    ("/api/v1/responses", {"input": [], "tools": []}),
    ("/api/v1/responses", {"input": None, "tools": None}),
    ("/api/v1/responses", {"tools": [{"type": "web_search", "external_web_access": True}]}),
    ("/api/v1/responses", {"tools": [{"type": "web_search_preview"}]}),
    ("/api/v1/responses", {"input": [{"type": "additional_tools", "tools": []}]}),
    ("/api/v1/responses", {"tools": [{"type": "function"}], "tool_choice": "none"}),
    ("/api/v1/chat/completions", {"tools": [{"type": "function"}]}),
])
def test_codex_openrouter_web_leaves_auxiliary_requests_unchanged(tmp_path, path, payload):
    agent = make_agent(CodexCLIAgent, tmp_path, True)
    agent.model = Model("openai/gpt-6-astra", api_url="https://openrouter.ai/api/v1", api_key="placeholder")
    before = json.dumps(payload)
    agent._filter_request(path, payload)
    assert json.dumps(payload) == before


@pytest.mark.parametrize("url,tools_enabled", [
    ("https://api.openai.com/v1", True),
    ("https://example.invalid/v1", True),
    ("https://openrouter.ai/api/v1", False),
])
def test_codex_hosted_web_respects_provider_and_tool_free_mode(tmp_path, url, tools_enabled):
    agent = make_agent(CodexCLIAgent, tmp_path, True, tools_enabled=tools_enabled)
    agent.model = Model("openai/gpt-6-astra", api_url=url, api_key="placeholder")
    payload = {"input": [{"type": "additional_tools", "tools": [{"type": "namespace", "name": "functions"}]}]}
    before = json.dumps(payload)
    agent._filter_request("/v1/responses", payload)
    assert json.dumps(payload) == before
