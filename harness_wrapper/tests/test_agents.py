from __future__ import annotations

import io
import json
import os
import signal
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from loguru import logger

from harness_wrapper.agent import Agent
from harness_wrapper.harnesses.claude_code import ClaudeCodeAgent
from harness_wrapper.harnesses.codex_cli import CodexCLIAgent
from harness_wrapper.harnesses.kimi_code import KimiCodeAgent
from harness_wrapper.harnesses.kimi_code.adapter import read_kimi_token_usage
from harness_wrapper.model import Model
from harness_wrapper.models import CredentialStore, OAuthConfig, OAuthCredential
from harness_wrapper.sandbox import HostServiceRoute
from harness_wrapper.tools import (
    CLIInstallation,
    CLIProcessError,
    ExecutableNotFoundError,
    JSONLineParser,
    ModelCompatibilityError,
    SessionCache,
    UnsupportedCLIVersionError,
    extract_version,
    find_executable,
    validate_cli_version,
)
from harness_wrapper.traces import TokenUsage, Trace


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


class FakeModel:
    def __init__(self, model: str = "tiny-model", protocols: tuple[str, ...] = ("openai",)):
        self.model = model
        self.protocols = protocols

    def supported_endpoints(self):
        return frozenset(self.protocols)

    def assert_compatible(self, accepted):
        for protocol in accepted:
            if protocol in self.protocols:
                return protocol
        raise ValueError(f"only {self.protocols}")

    def model_for(self, accepted):
        self.assert_compatible(accepted)
        return self

    def cli_environment(self, protocol):
        return {"TEST_MODEL_PROTOCOL": protocol}

    def cli_args(self, harness):
        return ()


class FakeTrace:
    def __init__(self):
        self.events = []

    def append(self, event_type, content=None, **kwargs):
        self.events.append((event_type, content, kwargs))


class FakeProcess:
    def __init__(self, stdout="", stderr="", returncode=0, *, running=False):
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = None if running else returncode
        self.wait_returncode = returncode
        self.signals = []
        self.terminated = False

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        self.returncode = self.wait_returncode
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)

    def terminate(self):
        self.terminated = True
        self.returncode = -signal.SIGTERM

    def kill(self):
        self.returncode = -signal.SIGKILL


class ProcessFactory:
    def __init__(self, process):
        self.process = process
        self.calls = []
        self.stdin_inputs = []

    def __call__(self, command, **kwargs):
        stdin = kwargs.get("stdin")
        self.stdin_inputs.append(stdin.read() if hasattr(stdin, "read") else None)
        self.calls.append((list(command), kwargs))
        return self.process


class ProcessSequenceFactory:
    def __init__(self, *processes):
        self.processes = iter(processes)
        self.calls = []
        self.stdin_inputs = []

    def __call__(self, command, **kwargs):
        stdin = kwargs.get("stdin")
        self.stdin_inputs.append(stdin.read() if hasattr(stdin, "read") else None)
        self.calls.append((list(command), kwargs))
        return next(self.processes)


def executable(tmp_path: Path, name: str) -> Path:
    path = tmp_path / name
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    return path


def make_agent(
    cls,
    tmp_path,
    *,
    model=None,
    trace=None,
    process_factory=None,
    env=None,
    subagents=None,
    **kwargs,
):
    return cls(
        model=model or FakeModel(protocols=cls.accepted_api_types),
        executable=executable(tmp_path, cls.installation.executable),
        dir=tmp_path,
        trace=trace if trace is not None else FakeTrace(),
        process_factory=process_factory or ProcessFactory(FakeProcess()),
        env=env,
        subagents=subagents,
        **kwargs,
    )


def test_agent_factory_and_aliases(tmp_path):
    codex = Agent(
        type="codex_cli",
        model=FakeModel(),
        executable=executable(tmp_path, "codex"),
        dir=tmp_path,
        trace=FakeTrace(),
    )
    assert isinstance(codex, CodexCLIAgent)
    assert {"claude", "codex", "kimi"}.issubset(Agent.available())
    with pytest.raises(ValueError, match="unknown agent type"):
        Agent(type="missing", model=FakeModel())


def test_model_compatibility_is_validated_and_fallback_selected(tmp_path):
    with pytest.raises(ModelCompatibilityError, match="claude-code accepts anthropic"):
        make_agent(ClaudeCodeAgent, tmp_path, model=FakeModel(protocols=("openai",)))

    fallback = FakeModel("fallback", protocols=("anthropic",))

    class Primary(FakeModel):
        def model_for(self, accepted):
            return fallback

    agent = make_agent(ClaudeCodeAgent, tmp_path, model=Primary())
    assert agent.model is fallback


@pytest.mark.parametrize(
    ("agent_type", "full_access_flag"),
    (
        (ClaudeCodeAgent, "--dangerously-skip-permissions"),
        (CodexCLIAgent, "--dangerously-bypass-approvals-and-sandbox"),
    ),
)
def test_all_clis_use_full_access_by_default(tmp_path, agent_type, full_access_flag):
    agent = make_agent(agent_type, tmp_path)

    assert full_access_flag in agent.build_command("work")
    assert full_access_flag in agent.build_command(None, resume=True)


def test_reasoning_is_translated_for_each_cli(tmp_path):
    codex_model = FakeModel()
    codex_model.reasoning = "high"
    codex = make_agent(CodexCLIAgent, tmp_path, model=codex_model)
    codex_command = codex.build_command("work")
    assert 'model_reasoning_effort="high"' in codex_command
    assert 'web_search="disabled"' in codex_command

    claude_model = FakeModel(protocols=("anthropic",))
    claude_model.reasoning = "xhigh"
    claude = make_agent(ClaudeCodeAgent, tmp_path, model=claude_model)
    claude_command = claude.build_command("work")
    assert claude_command[claude_command.index("--effort") + 1] == "xhigh"

    kimi_model = FakeModel()
    kimi_model.reasoning = "medium"
    kimi = make_agent(KimiCodeAgent, tmp_path, model=kimi_model)
    assert kimi.model_environment()["KIMI_MODEL_THINKING_EFFORT"] == "medium"


def test_minimal_context_uses_only_reproducible_cli_context(tmp_path):
    codex = make_agent(CodexCLIAgent, tmp_path, minimal_context=True, subagents={})
    codex_command = codex.build_command("work")
    assert "--ignore-user-config" in codex_command
    assert "--ignore-rules" in codex_command
    assert "--strict-config" in codex_command
    assert "skills.include_instructions=false" in codex_command
    assert "skills.bundled.enabled=false" in codex_command
    assert "project_doc_max_bytes=0" in codex_command
    assert "include_permissions_instructions=false" in codex_command
    assert "include_environment_context=false" in codex_command
    assert codex_command[codex_command.index("--disable") + 1] == "apps"
    disabled_features = {
        codex_command[index + 1]
        for index, value in enumerate(codex_command[:-1])
        if value == "--disable"
    }
    enabled_features = {
        codex_command[index + 1]
        for index, value in enumerate(codex_command[:-1])
        if value == "--enable"
    }
    assert "multi_agent" in disabled_features
    assert {"code_mode_host", "shell_tool", "unified_exec"} <= enabled_features
    assert not {"code_mode", "code_mode_host", "code_mode_only"} & disabled_features

    claude = make_agent(ClaudeCodeAgent, tmp_path, minimal_context=True, subagents={})
    claude_command = claude.build_command("work")
    assert "--safe-mode" in claude_command
    assert "--disable-slash-commands" in claude_command
    assert claude_command[claude_command.index("--tools") + 1] == "Bash"
    assert json.loads(claude_command[claude_command.index("--mcp-config") + 1]) == {
        "mcpServers": {}
    }

    kimi = make_agent(KimiCodeAgent, tmp_path, minimal_context=True, subagents={})
    kimi_command = kimi.build_command("work")
    assert kimi_command[kimi_command.index("--skills-dir") + 1] == (".harness_wrapper/empty-skills")
    assert kimi_command[kimi_command.index("--agent-file") + 1] == (
        ".harness_wrapper/minimal-kimi-agent.md"
    )
    agent_file = tmp_path / ".harness_wrapper/minimal-kimi-agent.md"
    assert "tools:\n  - Bash" in agent_file.read_text()
    assert "subagents: []" in agent_file.read_text()
    resumed = kimi.build_command("more", resume=True, session_id="session-1")
    assert "--skills-dir" in resumed
    assert "--agent-file" not in resumed


def test_oauth_capture_urls_preserve_native_auth_configuration(tmp_path):
    openai = FakeModel()
    openai.auth_mode = "oauth"
    openai.provider = "openai"
    codex = make_agent(CodexCLIAgent, tmp_path, model=openai)
    codex._request_capture_url = "http://127.0.0.1:1234/secret"
    codex_command = codex.build_command("work")
    assert "model_providers.harness_wrapper.requires_openai_auth=true" in codex_command
    assert 'model_providers.harness_wrapper.base_url="http://127.0.0.1:1234/secret"' in (
        codex_command
    )

    anthropic = FakeModel(protocols=("anthropic",))
    anthropic.auth_mode = "oauth"
    anthropic.provider = "anthropic"
    claude = make_agent(ClaudeCodeAgent, tmp_path, model=anthropic)
    claude._request_capture_url = "http://127.0.0.1:2345/secret"
    environment = claude.model_environment()
    assert environment["ANTHROPIC_BASE_URL"].endswith("/secret")
    assert environment["API_TIMEOUT_MS"] == "28800000"
    assert environment["BUN_CONFIG_HTTP_IDLE_TIMEOUT"] == "0"

    kimi_model = FakeModel(protocols=("openai",))
    kimi_model.auth_mode = "oauth"
    kimi_model.provider = "kimi"
    kimi = make_agent(KimiCodeAgent, tmp_path, model=kimi_model)
    kimi._native_oauth_capture_url = "http://127.0.0.1:3456/secret/coding/v1"
    assert kimi.model_environment()["KIMI_CODE_BASE_URL"].endswith("/coding/v1")


@pytest.mark.parametrize("bearer", [False, True])
def test_claude_api_auth_uses_bearer_for_gateways_only(tmp_path, bearer):
    model = Model(
        "anthropic/claude-opus-5", api_type="anthropic",
        api_url="https://openrouter.ai/api" if bearer else "https://api.anthropic.com",
        api_key="cli-placeholder", reasoning="max",
        headers={"Authorization": "Bearer host-secret"} if bearer else {"x-api-key": "host-secret"},
    )
    agent = make_agent(ClaudeCodeAgent, tmp_path, model=model)
    agent._request_capture_url = "http://127.0.0.1:2345/secret"
    environment = agent.model_environment()
    assert environment["ANTHROPIC_BASE_URL"] == agent._request_capture_url
    assert "host-secret" not in environment.values()
    if bearer:
        assert environment["ANTHROPIC_AUTH_TOKEN"] == "cli-placeholder"
        assert environment["ANTHROPIC_API_KEY"] == ""
    else:
        assert "ANTHROPIC_AUTH_TOKEN" not in environment
        assert environment["ANTHROPIC_API_KEY"] == "cli-placeholder"
    command = agent.build_command("solve")
    assert command[command.index("--effort") + 1] == "max"


def test_claude_commands_cover_resume_sandbox_and_native_subagents(tmp_path):
    sandbox = SimpleNamespace(prepare_command=lambda command, cwd, env: (command, cwd, env))
    agent = make_agent(
        ClaudeCodeAgent,
        tmp_path,
        env=sandbox,
        subagents={"reviewer": FakeModel("small-reviewer", protocols=("anthropic",))},
    )
    fresh = agent.build_command("fix it")
    assert fresh[1:6] == ["-p", "--output-format", "stream-json", "--verbose", "--model"]
    assert "--dangerously-skip-permissions" in fresh
    payload = json.loads(fresh[fresh.index("--agents") + 1])
    assert payload["reviewer"]["model"] == "small-reviewer"
    assert fresh[-2:] == ["--", "fix it"]
    assert agent.build_command(None, resume=True, last=True)[-1] == "--continue"
    by_id = agent.build_command("more", resume=True, session_id="session-1")
    assert by_id[-4:] == ["--resume", "session-1", "--", "more"]


def test_codex_commands_cover_resume_root_and_subagent_config(tmp_path):
    agent = make_agent(
        CodexCLIAgent,
        tmp_path,
        subagents={"explore": FakeModel("tiny-explore")},
    )
    fresh = agent.build_command("inspect")
    assert fresh[:3] == [agent.executable, "exec", "--json"]
    assert fresh[fresh.index("--cd") + 1] == str(tmp_path)
    assert "features.multi_agent=true" in fresh
    assert fresh[-2:] == ["--", "inspect"]
    assert agent.build_command(None, resume=True)[-1] == "--last"
    by_id = agent.build_command("continue", resume=True, session_id="thread-1")
    assert by_id[-3:] == ["thread-1", "--", "continue"]
    assert "--skip-git-repo-check" in by_id
    with pytest.raises(ValueError, match="session selection"):
        agent.build_command("bad", session_id="thread")


def test_oauth_models_become_default_named_subagents_and_can_be_disabled(tmp_path):
    class OAuthModel(FakeModel):
        auth_mode = "oauth"

        def available_models(self):
            return ("gpt-5.6-sol", "vendor/model.v2")

    model = OAuthModel("gpt-5.6-sol")
    agent = make_agent(CodexCLIAgent, tmp_path, model=model)

    assert agent.available_subagent_models() == ("gpt-5.6-sol", "vendor/model.v2")
    assert set(agent.subagents) == {"model-gpt-5-6-sol", "model-vendor-model-v2"}
    command = agent.build_command("delegate")
    assert "features.multi_agent=true" in command
    assert any("agents.model-vendor-model-v2.model" in item for item in command)

    disabled = make_agent(CodexCLIAgent, tmp_path, model=model, subagents={})
    assert disabled.subagents == {}
    assert "features.multi_agent=true" not in disabled.build_command("stay local")


def test_codex_explicit_api_uses_invocation_local_provider(tmp_path):
    model = Model("tiny", api_url="http://localhost:8000/v1", api_key="secret")
    agent = make_agent(CodexCLIAgent, tmp_path, model=model)
    command = agent.build_command("hello")
    assert 'model_provider="harness_wrapper"' in command
    assert 'model_providers.harness_wrapper.base_url="http://localhost:8000/v1"' in command
    assert "model_providers.harness_wrapper.stream_idle_timeout_ms=28800000" in command
    assert "secret" not in " ".join(command)


def test_codex_openai_oauth_uses_host_bridge_without_exposing_token(tmp_path, monkeypatch):
    class OpenAIOAuthModel(FakeModel):
        auth_mode = "oauth"
        provider = "openai"

        def cli_environment(self, protocol):
            return {}

    class FakeBridge:
        base_url = "http://10.200.1.1:43210"
        client_api_key = "one-time-bridge-key"

        def __init__(self, **kwargs):
            self.options = kwargs
            self.entered = False
            self.exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited = True

    bridges = []

    def make_bridge(**kwargs):
        bridge = FakeBridge(**kwargs)
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(
        "harness_wrapper.harnesses.codex_cli.adapter.CodexOAuthResponsesProxy",
        make_bridge,
    )
    process = FakeProcess(
        stdout='{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
    )
    factory = ProcessFactory(process)
    agent = make_agent(
        CodexCLIAgent,
        tmp_path,
        model=OpenAIOAuthModel("gpt-test"),
        process_factory=factory,
    )

    assert [event.content for event in agent.run("hello")] == ["done"]

    bridge = bridges[0]
    assert bridge.entered and bridge.exited
    assert bridge.options["on_request"] == agent._capture_model_request
    assert bridge.options["on_response"] == agent._capture_model_response
    command, options = factory.calls[0]
    assert 'model_provider="harness_wrapper"' in command
    assert f'model_providers.harness_wrapper.base_url="{bridge.base_url}"' in command
    assert "model_providers.harness_wrapper.stream_idle_timeout_ms=28800000" in command
    assert 'model_providers.harness_wrapper.env_key="OPENAI_API_KEY"' in command
    assert not any("requires_openai_auth" in arg for arg in command)
    assert options["env"]["OPENAI_API_KEY"] == bridge.client_api_key
    assert not hasattr(agent, "_request_capture_url")
    assert not hasattr(agent, "_oauth_bridge_api_key")


def test_kimi_commands_cover_resume_with_and_without_message(tmp_path):
    agent = make_agent(KimiCodeAgent, tmp_path)
    fresh = agent.build_command("hello")
    assert fresh[:5] == [agent.executable, "-p", "hello", "--output-format", "stream-json"]
    assert "--yolo" not in fresh
    assert "--auto" not in fresh
    assert fresh[-2:] == ["--model", "tiny-model"]
    assert agent.build_command(None, resume=True) == [agent.executable, "--continue"]
    resumed = agent.build_command("again", resume=True, session_id="ses_1")
    assert resumed[1:3] == ["--session", "ses_1"]
    assert resumed[3:7] == ["-p", "again", "--output-format", "stream-json"]
    assert "--yolo" not in resumed
    assert "--auto" not in resumed


def test_kimi_reads_native_wire_usage_when_stream_json_omits_it(tmp_path):
    session_id = "session_usage-test"
    wire = (
        tmp_path
        / ".harness-home"
        / ".kimi-code"
        / "sessions"
        / "wd_work_test"
        / session_id
        / "agents"
        / "main"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    records = [
        {
            "type": "usage.record",
            "usage": {
                "inputOther": 100,
                "inputCacheRead": 900,
                "inputCacheCreation": 20,
                "output": 30,
            },
            "usageScope": "turn",
        },
        {"type": "not-usage", "usage": {"inputOther": 999_999}},
        {
            "type": "usage.record",
            "usage": {
                "inputOther": "40",
                "inputCacheRead": "60",
                "inputCacheCreation": 0,
                "output": "10",
            },
            "usageScope": "turn",
        },
    ]
    wire.write_text("\n".join(json.dumps(record) for record in records) + "\n{incomplete")
    agent = make_agent(KimiCodeAgent, tmp_path)
    agent._session_id = session_id

    usage = agent.get_tokens()

    assert usage == TokenUsage(
        input_tokens=1_100,
        output_tokens=40,
        cache_read_tokens=960,
        cache_write_tokens=20,
    )


def test_kimi_native_usage_uses_last_session_record_without_turn_records(tmp_path):
    wire = (
        tmp_path
        / ".harness-home"
        / ".kimi-code"
        / "sessions"
        / "wd_work_test"
        / "session_cumulative"
        / "agents"
        / "main"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    wire.write_text(
        "\n".join(
            json.dumps(
                {
                    "type": "usage.record",
                    "usage": {
                        "inputOther": fresh,
                        "inputCacheRead": cached,
                        "output": output,
                    },
                    "usageScope": "session",
                }
            )
            for fresh, cached, output in ((10, 20, 30), (40, 50, 60))
        )
    )

    assert read_kimi_token_usage(tmp_path, "session_cumulative") == TokenUsage(
        input_tokens=90,
        output_tokens=60,
        cache_read_tokens=50,
    )


def test_kimi_merges_provider_usage_without_double_counting_native_usage(tmp_path):
    session_id = "session_provider-usage"
    wire = (
        tmp_path
        / ".harness-home"
        / ".kimi-code"
        / "sessions"
        / "wd_work_test"
        / session_id
        / "agents"
        / "main"
        / "wire.jsonl"
    )
    wire.parent.mkdir(parents=True)
    wire.write_text(
        json.dumps(
            {
                "type": "usage.record",
                "usage": {
                    "inputOther": 25,
                    "inputCacheRead": 75,
                    "output": 50,
                },
                "usageScope": "turn",
            }
        )
    )
    agent = make_agent(KimiCodeAgent, tmp_path)
    agent._begin_token_session(resume=False, session_id=None)
    agent._session_id = session_id
    agent._capture_model_response(
        "/chat/completions",
        {
            "usage": {
                "prompt_tokens": 140,
                "completion_tokens": 80,
                "prompt_tokens_details": {"cached_tokens": 90},
            }
        },
    )

    # Provider and native totals describe overlapping calls, so they must not
    # be added. Provider totals win where they include an internally failed call.
    assert agent.get_tokens() == TokenUsage(
        input_tokens=140,
        output_tokens=80,
        cache_read_tokens=90,
    )


def test_kimi_counts_streams_once_and_includes_unreported_reasoning(tmp_path):
    from harness_wrapper.models.request_capture import ResponseCapture

    trace = FakeTrace()
    agent = make_agent(KimiCodeAgent, tmp_path, trace=trace)
    captures = [ResponseCapture(
        "/v1/chat/completions", "text/event-stream", agent._capture_model_response,
        component="test",
    ) for _ in range(2)]
    # Interleave requests to ensure each response has its own cumulative counter.
    for completed in (1, 4, 4):
        for capture, prompt, reasoning in zip(captures, (20, 30), (7, 11), strict=True):
            payload = {
                "object": "chat.completion.chunk", "choices": [],
                "usage": {
                    "prompt_tokens": prompt, "completion_tokens": completed,
                    "total_tokens": prompt + completed + reasoning,
                    "prompt_tokens_details": {"cached_tokens": 5},
                },
            }
            capture.feed(("data: " + json.dumps(payload) + "\n\n").encode())
    for capture in captures:
        capture.finish()
    assert agent.get_tokens() == TokenUsage(
        input_tokens=50, output_tokens=26, cache_read_tokens=10,
    )
    saved_usage = [entry[1] for entry in trace.events if entry[0] == "usage"]
    assert len(saved_usage) == 2
    assert sum(usage["output_tokens"] for usage in saved_usage) == 26


def test_kimi_logs_only_new_model_provider_failures(tmp_path):
    native_log = (
        tmp_path
        / ".harness-home"
        / ".kimi-code"
        / "sessions"
        / "wd_work_test"
        / "session_errors"
        / "logs"
        / "kimi-code.log"
    )
    native_log.parent.mkdir(parents=True)
    native_log.write_text(
        "2026-01-01T00:00:00Z WARN  llm request failed  turnStep=0.1 "
        "model=test errorName=OldError errorMessage=old\n"
    )
    agent = make_agent(KimiCodeAgent, tmp_path)
    offsets = agent._native_log_offsets()
    with native_log.open("a") as stream:
        stream.write(
            "2026-01-01T00:00:01Z WARN  tool result failed  turnStep=0.2 "
            "errorName=ToolError errorMessage=do-not-log\n"
            "2026-01-01T00:00:02Z WARN  llm request failed  turnStep=0.3 "
            'model=test errorName=APIConnectionError errorMessage="Connection error."\n'
        )
    messages: list[str] = []
    sink = logger.add(messages.append, level="WARNING", format="{message}")
    try:
        agent._report_native_api_failures(offsets)
        agent._report_native_api_failures(offsets)
    finally:
        logger.remove(sink)

    rendered = "".join(messages)
    assert "APIConnectionError" in rendered
    assert "Connection error." in rendered
    assert "OldError" not in rendered
    assert "ToolError" not in rendered
    assert "do-not-log" not in rendered
    assert rendered.count("APIConnectionError") == 1


def test_kimi_explicit_api_uses_temporary_environment_provider(tmp_path):
    model = Model("tiny", api_url="http://localhost:8000/v1", api_key="secret")
    agent = make_agent(KimiCodeAgent, tmp_path, model=model)
    command = agent.build_command("hello")
    assert "--model" not in command
    environment = agent.model_environment()
    assert environment["KIMI_MODEL_NAME"] == "tiny"
    assert environment["KIMI_MODEL_BASE_URL"] == "http://localhost:8000/v1"
    assert environment["KIMI_MODEL_MAX_CONTEXT_SIZE"] == "32768"


def test_kimi_translates_explicit_api_model_environment(tmp_path):
    class APIModel(FakeModel):
        auth_mode = "api"
        api_url = "http://127.0.0.1:8000/v1"
        api_key = "test-key"

        def cli_environment(self, protocol):
            return {"OPENAI_BASE_URL": self.api_url, "OPENAI_API_KEY": self.api_key}

    agent = make_agent(KimiCodeAgent, tmp_path, model=APIModel())
    environment = agent.model_environment()
    assert environment["KIMI_MODEL_NAME"] == "tiny-model"
    assert environment["KIMI_MODEL_PROVIDER_TYPE"] == "openai"
    assert environment["KIMI_MODEL_BASE_URL"] == "http://127.0.0.1:8000/v1"
    assert environment["KIMI_MODEL_API_KEY"] == "test-key"
    assert environment["KIMI_CODE_NO_AUTO_UPDATE"] == "1"


def test_kimi_protects_custom_api_headers_with_invocation_local_bridge(tmp_path, monkeypatch):
    class FakeBridge:
        base_url = "http://127.0.0.1:12345/v1"
        client_api_key = "one-time-api-bridge-key"
        entered = False
        exited = False

        def __init__(
            self,
            upstream_url,
            headers,
            *,
            on_request=None,
            on_response=None,
            request_overrides=None,
            request_drop_fields=(),
            listen_host="127.0.0.1",
            client_host="127.0.0.1",
            allow_remote_clients=False,
            offline_models=frozenset(),
            unix_socket=None,
            client_port=None,
        ):
            self.upstream_url = upstream_url
            self.headers = headers
            self.on_request = on_request
            self.on_response = on_response
            self.request_overrides = request_overrides
            self.request_drop_fields = request_drop_fields
            self.listen_host = listen_host
            self.client_host = client_host
            self.allow_remote_clients = allow_remote_clients
            self.offline_models = offline_models
            self.unix_socket = unix_socket
            self.client_port = client_port

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited = True

    bridges = []

    def make_bridge(
        upstream_url,
        headers,
        *,
        on_request=None,
        on_response=None,
        request_overrides=None,
        request_drop_fields=(),
        listen_host="127.0.0.1",
        client_host="127.0.0.1",
        allow_remote_clients=False,
        offline_models=frozenset(),
        unix_socket=None,
        client_port=None,
    ):
        bridge = FakeBridge(
            upstream_url,
            headers,
            on_request=on_request,
            on_response=on_response,
            request_overrides=request_overrides,
            request_drop_fields=request_drop_fields,
            listen_host=listen_host,
            client_host=client_host,
            allow_remote_clients=allow_remote_clients,
            offline_models=offline_models,
            unix_socket=unix_socket,
            client_port=client_port,
        )
        bridges.append(bridge)
        return bridge

    monkeypatch.setattr(
        "harness_wrapper.harnesses.kimi_code.adapter.APIHeaderProxy",
        make_bridge,
    )
    process = FakeProcess(stdout='{"role":"assistant","content":"done"}\n')
    factory = ProcessFactory(process)
    model = Model(
        "openrouter/moonshotai/kimi-k3",
        api_url="https://srlx.inf.ethz.ch/v1",
        api_key="real-virtual-key",
        headers={"x-bf-vk": "real-virtual-key"},
    )
    agent = make_agent(KimiCodeAgent, tmp_path, model=model, process_factory=factory)

    events = agent.run("hello")

    assert [event.content for event in events] == ["done"]
    bridge = bridges[0]
    assert bridge.entered and bridge.exited
    assert bridge.upstream_url == "https://srlx.inf.ethz.ch/v1"
    assert bridge.headers == {"x-bf-vk": "real-virtual-key"}
    assert bridge.on_request == agent._capture_model_request
    assert bridge.on_response == agent._capture_model_response
    assert bridge.request_overrides == {}
    assert bridge.request_drop_fields == frozenset()
    assert bridge.listen_host == "127.0.0.1"
    assert bridge.client_host == "127.0.0.1"
    assert bridge.allow_remote_clients is False
    _, options = factory.calls[0]
    environment = options["env"]
    assert environment["KIMI_MODEL_BASE_URL"] == bridge.base_url
    assert environment["KIMI_MODEL_API_KEY"] == bridge.client_api_key
    assert "OPENAI_API_KEY" not in environment
    assert "real-virtual-key" not in environment.values()
    assert not hasattr(agent, "_api_header_bridge_url")
    assert not hasattr(agent, "_api_header_bridge_api_key")


def test_kimi_enables_native_subagents_and_selects_secondary_model(tmp_path):
    agent = make_agent(
        KimiCodeAgent,
        tmp_path,
        subagents={"fast": "kimi-code/fast", "large": "kimi-code/large"},
    )

    environment = agent.model_environment()

    assert environment["KIMI_CODE_EXPERIMENTAL_FLAG"] == "1"
    assert environment["KIMI_SECONDARY_MODEL"] == "kimi-code/fast"
    assert json.loads(environment["HARNESS_WRAPPER_SUBAGENTS"])["large"]["model"] == (
        "kimi-code/large"
    )


def test_kimi_openai_oauth_uses_invocation_local_bridge(tmp_path, monkeypatch):
    class OpenAIOAuthModel(FakeModel):
        auth_mode = "oauth"
        provider = "openai"

        def cli_environment(self, protocol):
            return {}

        def available_models(self):
            return (self.model, "gpt-secondary")

    class FakeBridge:
        base_url = "http://127.0.0.1:12345"
        client_api_key = "one-time-bridge-key"
        entered = False
        exited = False

        def __enter__(self):
            self.entered = True
            return self

        def __exit__(self, *args):
            self.exited = True

    bridge = FakeBridge()
    monkeypatch.setattr(
        "harness_wrapper.harnesses.kimi_code.adapter.CodexOAuthResponsesProxy",
        lambda **kwargs: bridge,
    )
    process = FakeProcess(stdout='{"role":"assistant","content":"done"}\n')
    factory = ProcessFactory(process)
    agent = make_agent(
        KimiCodeAgent,
        tmp_path,
        model=OpenAIOAuthModel("gpt-test"),
        process_factory=factory,
    )

    events = agent.run("hello")

    assert [event.content for event in events] == ["done"]
    assert bridge.entered and bridge.exited
    command, options = factory.calls[0]
    assert "--model" not in command
    environment = options["env"]
    assert environment["KIMI_MODEL_PROVIDER_TYPE"] == "openai_responses"
    assert environment["KIMI_MODEL_BASE_URL"] == bridge.base_url
    assert environment["KIMI_MODEL_API_KEY"] == bridge.client_api_key
    assert environment["KIMI_CODE_EXPERIMENTAL_FLAG"] == "1"
    assert "KIMI_SECONDARY_MODEL" not in environment
    assert not hasattr(agent, "_codex_oauth_bridge_url")
    assert not hasattr(agent, "_codex_oauth_bridge_api_key")


def test_kimi_openai_oauth_uses_sandbox_host_service_route(tmp_path, monkeypatch):
    class OpenAIOAuthModel(FakeModel):
        auth_mode = "oauth"
        provider = "openai"

    class Sandbox:
        enabled = True
        network_enabled = False
        root = tmp_path

        @contextmanager
        def expose_host_service(self):
            yield HostServiceRoute()

        def prepare_command(self, command, cwd, env):
            return command, cwd, env

    class Bridge:
        base_url = "http://127.0.0.1:12345"
        client_api_key = "sandbox-bridge-key"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        "harness_wrapper.harnesses.kimi_code.adapter.CodexOAuthResponsesProxy",
        lambda **kwargs: Bridge(),
    )
    process = FakeProcess(stdout='{"role":"assistant","content":"done"}\n')
    agent = make_agent(
        KimiCodeAgent,
        tmp_path,
        model=OpenAIOAuthModel(),
        env=Sandbox(),
        process_factory=ProcessFactory(process),
    )

    assert [event.content for event in agent.run("hello")] == ["done"]


@pytest.mark.parametrize(
    ("agent_class", "api_type"),
    [
        (CodexCLIAgent, "openai"),
        (ClaudeCodeAgent, "anthropic"),
        (KimiCodeAgent, "openai"),
    ],
)
def test_openrouter_provider_restrictions_use_sandbox_host_service_route(
    tmp_path, agent_class, api_type
):
    model = Model(
        "provider/model",
        api_url="https://openrouter.ai/api/v1",
        api_key="key",
        api_type=api_type,
        openrouter_providers=("provider",),
    )

    class Sandbox:
        enabled = True
        network_enabled = False
        root = tmp_path

        @contextmanager
        def expose_host_service(self):
            yield HostServiceRoute()

        def prepare_command(self, command, cwd, env):
            return command, cwd, env

    process = FakeProcess(
        stdout='{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n'
    )
    if agent_class is ClaudeCodeAgent:
        process = FakeProcess(
            stdout='{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"done"}]}}\n'
        )
    elif agent_class is KimiCodeAgent:
        process = FakeProcess(stdout='{"role":"assistant","content":"done"}\n')
    agent = make_agent(
        agent_class,
        tmp_path,
        model=model,
        env=Sandbox(),
        process_factory=ProcessFactory(process),
    )

    assert [event.content for event in agent.run("hello")] == ["done"]


def test_jsonl_parser_handles_chunks_scalars_and_banner_lines():
    parser = JSONLineParser()
    assert parser.feed('{"type":"one"') == []
    assert parser.feed("}\nnot json\n[1,2]\n") == [
        {"type": "one"},
        {"type": "stdout", "text": "not json"},
        {"type": "json", "value": [1, 2]},
    ]
    assert parser.feed('{"tail":true}') == []
    assert parser.finish() == [{"tail": True}]
    with pytest.raises(json.JSONDecodeError):
        JSONLineParser(strict=True).feed("bad\n")


def test_stream_process_lifecycle_normalization_trace_and_activity(tmp_path):
    stdout = "\n".join(
        (
            json.dumps({"type": "thread.started", "thread_id": "th-1"}),
            json.dumps(
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "done"},
                }
            ),
            json.dumps(
                {
                    "type": "turn.completed",
                    "usage": {"input_tokens": 4, "output_tokens": 2},
                }
            ),
            "",
        )
    )
    process = FakeProcess(stdout=stdout)
    factory = ProcessFactory(process)
    trace = FakeTrace()
    activity = []
    observed = []
    agent = CodexCLIAgent(
        model=FakeModel(),
        executable=executable(tmp_path, "codex"),
        dir=tmp_path,
        trace=trace,
        process_factory=factory,
        on_activity=activity.append,
        on_event=observed.append,
    )
    events = agent.run("say done")
    assert [event.type for event in events] == ["session", "message", "usage", "result"]
    assert events[1].content == "done"
    assert agent.session_id == "th-1"
    assert agent.last_activity_at is not None
    assert agent.seconds_since_activity() >= 0
    assert [entry[0] for entry in trace.events] == ["message", "session", "message", "usage", "result"]
    assert trace.events[0][1] == "say done"
    assert trace.events[0][2]["role"] == "user"
    assert trace.events[0][2]["metadata"]["raw"] == {
        "source": "harness_wrapper",
        "kind": "original_message",
    }
    assert activity
    assert [event.type for event in observed] == ["session", "message", "usage", "result"]
    command, kwargs = factory.calls[0]
    assert command[-2:] == ["--", "-"]
    assert factory.stdin_inputs == [b"say done"]
    assert kwargs["cwd"] == tmp_path
    assert kwargs["env"]["TEST_MODEL_PROTOCOL"] == "openai"
    assert SessionCache(tmp_path).last("codex-cli") == "th-1"


def test_get_tokens_accumulates_resumes_and_resets_for_a_new_session(tmp_path):
    def codex_turn(session_id, usage):
        return FakeProcess(
            stdout="\n".join(
                (
                    json.dumps({"type": "thread.started", "thread_id": session_id}),
                    json.dumps({"type": "turn.completed", "usage": usage}),
                    "",
                )
            )
        )

    factory = ProcessSequenceFactory(
        codex_turn(
            "thread-1",
            {"input_tokens": 10, "output_tokens": 2, "cached_input_tokens": 4},
        ),
        codex_turn(
            "thread-1",
            {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "prompt_tokens_details": {"cached_tokens": 2},
            },
        ),
        codex_turn("thread-2", {"input_tokens": 5, "output_tokens": 3}),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    assert agent.get_tokens() == TokenUsage()
    agent.run("first")
    assert agent.get_tokens() == TokenUsage(
        input_tokens=10,
        output_tokens=2,
        cache_read_tokens=4,
    )

    agent.resume("second", session_id="thread-1")
    assert agent.get_tokens() == TokenUsage(
        input_tokens=13,
        output_tokens=3,
        cache_read_tokens=6,
    )

    agent.run("new session")
    assert agent.get_tokens() == TokenUsage(input_tokens=5, output_tokens=3)


@pytest.mark.parametrize(
    ("usage", "expected_output"),
    [
        # Gemini through Qwen's native result or Kimi/DeepCode's provider capture.
        ({"input_tokens": 44152, "output_tokens": 2074, "total_tokens": 109163}, 65011),
        ({"prompt_tokens": 19, "completion_tokens": 14, "total_tokens": 658}, 639),
        # OpenAI/DeepSeek completion totals already include their reasoning subset.
        ({"prompt_tokens": 19, "completion_tokens": 639, "total_tokens": 658,
          "completion_tokens_details": {"reasoning_tokens": 625}}, 639),
        # Responses/Muse and native Codex also include reasoning in output.
        ({"input_tokens": 100, "output_tokens": 250, "total_tokens": 350,
          "output_tokens_details": {"reasoning_tokens": 220}}, 250),
        ({"input_tokens": 100, "output_tokens": 250, "reasoning_output_tokens": 220}, 250),
        # Anthropic's thinking detail is a subset; cached input is separate.
        ({"input_tokens": 6, "output_tokens": 113812,
          "cache_read_input_tokens": 1230,
          "output_tokens_details": {"thinking_tokens": 112085}}, 113812),
        # Missing or inconsistent total fields must not erase reported output.
        ({"input_tokens": 100, "output_tokens": 20}, 20),
        ({"input_tokens": 100, "output_tokens": 20, "total_tokens": 50}, 20),
        ({"output_tokens": 20, "total_tokens": 120}, 20),
        ({"input_tokens": "100", "output_tokens": "20", "total_tokens": "150"}, 50),
    ],
)
def test_usage_includes_reasoning_once(usage, expected_output):
    assert Agent._normalize_token_usage(usage).output_tokens == expected_output


def test_get_tokens_normalizes_anthropic_cache_writes(tmp_path):
    process = FakeProcess(
        stdout=json.dumps(
            {
                "type": "result",
                "session_id": "claude-1",
                "result": "done",
                "usage": {
                    "input_tokens": 7,
                    "output_tokens": 4,
                    "cache_read_input_tokens": 11,
                    "cache_creation_input_tokens": 13,
                },
            }
        )
        + "\n"
    )
    agent = make_agent(
        ClaudeCodeAgent,
        tmp_path,
        process_factory=ProcessFactory(process),
    )

    agent.run("hello")

    assert agent.get_tokens() == TokenUsage(
        input_tokens=7,
        output_tokens=4,
        cache_read_tokens=11,
        cache_write_tokens=13,
    )


def test_model_request_capture_records_privileged_prompts_once(tmp_path):
    trace = FakeTrace()
    agent = make_agent(CodexCLIAgent, tmp_path, trace=trace)
    payload = {
        "instructions": "base system prompt",
        "input": [
            {"role": "developer", "content": [{"type": "text", "text": "policy"}]},
            {"role": "user", "content": "not captured"},
        ],
    }

    agent._capture_model_request("/v1/responses", payload)
    agent._capture_model_request("/v1/responses", payload)

    assert [(entry[2]["role"], entry[1]) for entry in trace.events] == [
        ("system", "base system prompt"),
        ("developer", [{"type": "text", "text": "policy"}]),
    ]
    assert trace.events[0][2]["metadata"]["raw"] == {
        "source": "model_request",
        "field": "instructions",
        "path": "/v1/responses",
    }

    request_path = agent.model_request_log_path
    assert request_path == tmp_path / ".harness_wrapper" / "model_requests.jsonl"
    records = [json.loads(line) for line in request_path.read_text().splitlines()]
    assert len(records) == 2
    assert [record["event"]["content"] for record in records] == [payload, payload]
    assert records[0]["event"]["type"] == "model.request"
    assert records[0]["event"]["metadata"] == {
        "path": "/v1/responses",
        "source": "model_proxy",
        "http_headers_included": False,
    }
    assert "headers" not in records[0]


def test_model_capture_records_plaintext_and_recoverable_reasoning_once(tmp_path):
    trace = FakeTrace()
    agent = make_agent(CodexCLIAgent, tmp_path, trace=trace)
    openai_reasoning = {
        "type": "reasoning",
        "id": "rs_1",
        "summary": [{"type": "summary_text", "text": "checked the constraints"}],
        "encrypted_content": "opaque-openai-state",
    }

    response = {"type": "response.output_item.done", "item": openai_reasoning}
    agent._capture_model_response("/v1/responses", response)
    agent._capture_model_response("/v1/responses", response)
    agent._capture_model_request(
        "/v1/messages",
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "considered the tool result",
                            "signature": "anthropic-signature",
                        }
                    ],
                }
            ]
        },
    )

    reasoning = [entry[1] for entry in trace.events if entry[0] == "reasoning"]
    encrypted = [entry[1] for entry in trace.events if entry[0] == "reasoning_encrypted"]
    assert reasoning == ["checked the constraints", "considered the tool result"]
    assert encrypted == [
        openai_reasoning,
        {
            "type": "thinking",
            "thinking": "considered the tool result",
            "signature": "anthropic-signature",
        },
    ]
    assert trace.events[0][2]["metadata"]["raw"]["source"] == "model_response"


def test_model_response_capture_records_safe_provider_errors(tmp_path):
    trace = FakeTrace()
    agent = make_agent(CodexCLIAgent, tmp_path, trace=trace)

    agent._capture_model_response(
        "/v1beta/models/test:streamGenerateContent",
        {
            "error": {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": "bad request; api_key=do-not-record Bearer also-secret",
                "api_key": "also-do-not-record",
                "details": [{"secret": "not-recorded"}],
            }
        },
    )

    assert trace.events == [
        (
            "error",
            {
                "code": 400,
                "status": "INVALID_ARGUMENT",
                "message": "bad request; api_key=[REDACTED] Bearer [REDACTED]",
            },
            {
                "role": "meta",
                "tool_name": None,
                "tool_call_id": None,
                "metadata": {
                    "harness": "codex-cli",
                    "raw": {
                        "source": "model_response",
                        "path": "/v1beta/models/test:streamGenerateContent",
                    },
                },
            },
        )
    ]


def test_native_redacted_thinking_is_promoted_to_encrypted_reasoning_trace(tmp_path):
    trace = FakeTrace()
    agent = make_agent(ClaudeCodeAgent, tmp_path, trace=trace)
    raw = {
        "type": "assistant",
        "session_id": "claude-1",
        "message": {
            "role": "assistant",
            "content": [{"type": "redacted_thinking", "data": "opaque-state"}],
        },
    }

    for event in agent.normalize_event(raw):
        agent._observe(event)

    encrypted = [entry for entry in trace.events if entry[0] == "reasoning_encrypted"]
    assert len(encrypted) == 1
    assert encrypted[0][1] == {"type": "redacted_thinking", "data": "opaque-state"}
    assert encrypted[0][2]["metadata"]["raw"]["source"] == "native_event"


def test_resume_uses_repo_local_last_session_cache(tmp_path):
    cache = SessionCache(tmp_path)
    cache.record("codex-cli", "cached-thread", "2026-08-13T00:00:00+00:00")
    factory = ProcessFactory(FakeProcess())
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    assert agent.resume("continue") == []
    command = factory.calls[0][0]
    assert "cached-thread" in command
    assert "--last" not in command


def test_resume_without_message_selects_cached_session_without_process(tmp_path):
    cache = SessionCache(tmp_path)
    cache.record("codex-cli", "cached-thread", "2026-08-13T00:00:00+00:00")
    factory = ProcessFactory(FakeProcess())
    trace = FakeTrace()
    agent = make_agent(CodexCLIAgent, tmp_path, trace=trace, process_factory=factory)

    events = agent.resume()

    assert factory.calls == []
    assert len(events) == 1
    assert events[0].type == "session"
    assert events[0].session_id == "cached-thread"
    assert events[0].content["selection"] == "id"
    assert trace.events[-1][0] == "session"


def test_resume_without_message_can_select_native_last_without_process(tmp_path):
    factory = ProcessFactory(FakeProcess())
    agent = make_agent(ClaudeCodeAgent, tmp_path, process_factory=factory)

    events = agent.resume()

    assert factory.calls == []
    assert events[0].session_id is None
    assert events[0].content["selection"] == "last"


def test_dash_prefixed_prompts_are_separated_from_cli_options(tmp_path):
    claude = make_agent(ClaudeCodeAgent, tmp_path)
    codex = make_agent(CodexCLIAgent, tmp_path)
    assert claude.build_command("--version")[-2:] == ["--", "--version"]
    assert codex.build_command("--help")[-2:] == ["--", "--help"]


def test_claude_and_kimi_event_normalization(tmp_path):
    claude = make_agent(ClaudeCodeAgent, tmp_path)
    raw = {
        "type": "assistant",
        "session_id": "c1",
        "message": {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "redacted_thinking", "data": "opaque-claude-state"},
                {"type": "tool_use", "id": "t1", "name": "Read", "input": {"file": "x"}},
                {"type": "text", "text": "answer"},
            ],
        },
    }
    events = claude.normalize_event(raw)
    assert [item.type for item in events] == [
        "reasoning",
        "redacted_thinking",
        "tool_call",
        "message",
    ]
    assert events[2].tool_name == "Read"

    kimi = make_agent(KimiCodeAgent, tmp_path)
    events = kimi.normalize_event(
        {
            "role": "assistant",
            "content": "checking",
            "reasoning_content": "thinking about it",
            "usage": {"prompt_tokens": 8, "completion_tokens": 3},
            "tool_calls": [
                {
                    "id": "tc1",
                    "type": "function",
                    "function": {"name": "Shell", "arguments": '{"command":"ls"}'},
                }
            ],
        }
    )
    assert [item.type for item in events] == ["reasoning", "message", "tool_call", "usage"]
    assert events[2].tool_call_id == "tc1"
    session = kimi.normalize_event(
        {"role": "meta", "type": "session.resume_hint", "session_id": "k1", "content": "resume"}
    )
    assert session[0].session_id == "k1"


def test_nonzero_process_raises_but_intentional_interrupt_does_not(tmp_path):
    failure = ProcessSequenceFactory(*[
        FakeProcess(stderr="bad auth\n", returncode=7) for _ in range(11)
    ])
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=failure)
    with pytest.raises(CLIProcessError) as error:
        agent.run("hello")
    assert error.value.returncode == 7
    assert "bad auth" in str(error.value)
    assert agent.trace.events[-1][0] == "error"

    live = FakeProcess(running=True, returncode=-signal.SIGINT)
    agent._process = live
    assert agent.interrupt() is True
    assert live.signals == [signal.SIGINT]
    live.returncode = 0
    assert agent.interrupt() is False


def test_rate_limit_failure_switches_to_api_fallback_and_resumes_session(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("oauth", expires_at=4_000_000_000))
    fallback = Model(
        "api-fallback",
        api_url="https://fallback.invalid/v1",
        api_key="fallback-secret-that-must-never-be-logged",
    )
    oauth = Model(
        "oauth-primary",
        oauth="openai",
        oauth_config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            rate_limit_hook=lambda credential: {"limits": []},
        ),
        fallbacks=(fallback,),
    )
    failed = FakeProcess(
        stdout='{"type":"thread.started","thread_id":"rate-session"}\n',
        stderr="HTTP 429 rate limit exceeded\n",
        returncode=7,
    )
    succeeded = FakeProcess(
        stdout=('{"type":"item.completed","item":{"type":"agent_message","text":"done"}}\n')
    )
    factory = ProcessSequenceFactory(failed, succeeded)
    agent = make_agent(
        CodexCLIAgent,
        tmp_path,
        model=oauth,
        process_factory=factory,
        auto_wait=False,
    )

    log_messages: list[str] = []
    sink = logger.add(log_messages.append, level="DEBUG", format="{message}")
    try:
        events = agent.run("do the work")
    finally:
        logger.remove(sink)

    assert [event.type for event in events] == ["session", "recovery", "message"]
    assert events[1].content["reason"] == "rate_limit_fallback"
    assert agent.model is fallback
    retry_command = factory.calls[1][0]
    assert retry_command[:3] == [str(agent.executable), "exec", "resume"]
    assert "rate-session" in retry_command
    assert "api-fallback" in retry_command
    logs = "\n".join(log_messages)
    assert "Classified harness CLI failure" in logs
    assert "Switching from failed model to API fallback" in logs
    assert "Retrying harness CLI turn after recovery" in logs
    assert "fallback-secret-that-must-never-be-logged" not in logs
    user_messages = [
        entry
        for entry in agent.trace.events
        if entry[0] == "message" and entry[2].get("role") == "user"
    ]
    assert [entry[1] for entry in user_messages] == ["do the work"]


def test_authentication_failure_relogs_in_and_resumes_session(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("old", expires_at=4_000_000_000))
    logins: list[bool] = []
    oauth = Model(
        "oauth-primary",
        oauth="openai",
        oauth_config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            login_hook=lambda: (
                logins.append(True) or OAuthCredential("fresh", expires_at=4_000_000_000)
            ),
            rate_limit_hook=lambda credential: {"limits": []},
        ),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout='{"type":"thread.started","thread_id":"auth-session"}\n',
            stderr="401 token expired; login required\n",
            returncode=1,
        ),
        FakeProcess(
            stdout=(
                '{"type":"item.completed","item":{"type":"agent_message","text":"recovered"}}\n'
            )
        ),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, model=oauth, process_factory=factory)

    events = agent.run("do the work")

    assert [event.type for event in events] == ["session", "recovery", "message"]
    assert events[1].content["reason"] == "authentication_relogin"
    assert logins == [True]
    assert "auth-session" in factory.calls[1][0]


@pytest.mark.parametrize(
    ("agent_class", "provider", "session_line", "success_line"),
    [
        (
            ClaudeCodeAgent,
            "anthropic",
            '{"type":"system","subtype":"init","session_id":"provider-session"}\n',
            '{"type":"assistant","message":{"role":"assistant","content":'
            '[{"type":"text","text":"recovered"}]}}\n',
        ),
        (
            KimiCodeAgent,
            "kimi",
            '{"role":"meta","type":"session.resume_hint","session_id":"provider-session"}\n',
            '{"role":"assistant","content":"recovered"}\n',
        ),
    ],
)
def test_claude_and_kimi_auth_failures_relogin_and_resume(
    tmp_path, agent_class, provider, session_line, success_line
):
    store = CredentialStore(tmp_path / f"{provider}.json")
    store.save(provider, OAuthCredential("old", expires_at=4_000_000_000))
    logins: list[bool] = []
    oauth = Model(
        "oauth-primary",
        oauth=provider,
        oauth_config=OAuthConfig(
            provider=provider,
            store=store,
            auto_login=False,
            login_hook=lambda: (
                logins.append(True) or OAuthCredential("fresh", expires_at=4_000_000_000)
            ),
            rate_limit_hook=lambda credential: {"limits": []},
        ),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout=session_line,
            stderr="OAuth token expired; login required (401)\n",
            returncode=1,
        ),
        FakeProcess(stdout=success_line),
    )
    agent = make_agent(agent_class, tmp_path, model=oauth, process_factory=factory)

    events = agent.run("do the work")

    assert [event.type for event in events] == ["session", "recovery", "message"]
    assert events[1].content["reason"] == "authentication_relogin"
    assert logins == [True]
    assert "provider-session" in factory.calls[1][0]


def test_rate_limit_failure_waits_and_resumes_without_fallback(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("oauth", expires_at=4_000_000_000))
    polls: list[bool] = []
    fallback = Model(
        "api-fallback",
        api_url="https://fallback.invalid/v1",
        api_key="fallback-secret",
    )
    oauth = Model(
        "oauth-primary",
        oauth="openai",
        oauth_config=OAuthConfig(
            provider="openai",
            store=store,
            auto_login=False,
            rate_limit_hook=lambda credential: polls.append(True) or {"limits": []},
        ),
        fallbacks=(fallback,),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout='{"type":"thread.started","thread_id":"rate-session"}\n',
            stderr="weekly rate limit reached (429)\n",
            returncode=1,
        ),
        FakeProcess(
            stdout=('{"type":"item.completed","item":{"type":"agent_message","text":"resumed"}}\n')
        ),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, model=oauth, process_factory=factory)

    events = agent.run("do the work")

    assert [event.type for event in events] == ["session", "recovery", "message"]
    assert events[1].content["reason"] == "rate_limit_wait"
    assert agent.model is oauth
    assert len(polls) >= 2


def test_kimi_rate_limit_recovery_resumes_last_session_without_resume_hint(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("kimi", OAuthCredential("oauth", expires_at=4_000_000_000))
    polls: list[bool] = []
    oauth = Model(
        "kimi-code/k3",
        oauth="kimi",
        oauth_config=OAuthConfig(
            provider="kimi",
            store=store,
            auto_login=False,
            rate_limit_hook=lambda credential: polls.append(True) or {"limits": []},
        ),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout=(
                '{"role":"meta","type":"system.version","version":"0.35.0"}\n'
                '{"role":"assistant","content":"working"}\n'
            ),
            stderr="five hour rate limit reached (429)\n",
            returncode=1,
        ),
        FakeProcess(stdout='{"role":"assistant","content":"resumed"}\n'),
    )
    agent = make_agent(KimiCodeAgent, tmp_path, model=oauth, process_factory=factory)

    events = agent.run("do the work")

    assert [event.type for event in events] == [
        "system.version",
        "message",
        "recovery",
        "message",
    ]
    assert events[2].content["reason"] == "rate_limit_wait"
    retry_command = factory.calls[1][0]
    assert retry_command[1] == "--continue"
    assert "Continue the interrupted task from where it stopped." in retry_command
    assert "do the work" not in retry_command
    assert len(polls) == 2


@pytest.mark.parametrize("returncode", [129, -signal.SIGHUP])
def test_kimi_sighup_recovery_resumes_api_session_within_budget(tmp_path, returncode):
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout=(
                '{"role":"meta","type":"system.version","version":"0.40.1"}\n'
                '{"role":"assistant","content":"working"}\n'
            ),
            returncode=returncode,
        ),
        FakeProcess(stdout='{"role":"assistant","content":"resumed"}\n'),
    )
    agent = make_agent(
        KimiCodeAgent,
        tmp_path,
        model=FakeModel(protocols=("openai",)),
        process_factory=factory,
        max_recovery_attempts=1,
    )

    events = agent.run("do the work")

    assert [event.type for event in events] == [
        "system.version",
        "message",
        "recovery",
        "message",
    ]
    assert events[2].content["reason"] == "process_interrupted"
    retry_command = factory.calls[1][0]
    assert retry_command[1] == "--continue"
    assert "Continue the interrupted task from where it stopped." in retry_command
    assert "do the work" not in retry_command


@pytest.mark.parametrize(
    ("failed_turns", "budget", "has_session", "output_limit", "calls", "succeeds"),
    [
        (3, 3, True, True, 4, True),
        (4, 3, True, True, 4, False),
        (1, 0, True, True, 1, False),
        (1, 3, False, True, 2, True),
        (1, 3, True, False, 2, True),
    ],
)
@pytest.mark.parametrize(
    ("native_stop", "returncode", "is_error"),
    [(False, 1, True), (True, 0, False), (True, 1, False), (True, 0, True), (True, 1, True)],
)
def test_claude_output_limit_recovery(
    tmp_path, failed_turns, budget, has_session, output_limit, calls, succeeds,
    native_stop, returncode, is_error,
):
    failure = (
        "API Error: Claude's response exceeded the 128000 output token maximum. "
        "To configure this behavior, set CLAUDE_CODE_MAX_OUTPUT_TOKENS."
        if output_limit else "API Error: invalid request"
    )
    session = {"session_id": "claude-retry"} if has_session else {}
    failed_tokens = 256000 if native_stop and output_limit else 128000

    def turn(failed):
        native_failure = failed and output_limit and native_stop
        events = [{"type": "system", "subtype": "init", **session}]
        if failed and not native_failure:
            events.append({
                "type": "assistant", "isApiErrorMessage": True, **session,
                "message": {"role": "assistant", "content": [
                    {"type": "text", "text": failure}
                ]},
            })
        events.append({
            "type": "result", "subtype": "success", **session,
            "is_error": is_error if native_failure else failed,
            "stop_reason": "max_tokens" if native_failure else "end_turn",
            "result": "" if native_failure else failure if failed else "done",
            "usage": {"output_tokens": failed_tokens if failed else 10},
        })
        return FakeProcess(
            stdout="".join(json.dumps(event) + "\n" for event in events),
            returncode=returncode if native_failure else int(failed),
        )

    factory = ProcessSequenceFactory(*(turn(True) for _ in range(failed_turns)), turn(False))
    agent = make_agent(
        ClaudeCodeAgent, tmp_path, process_factory=factory, max_recovery_attempts=budget
    )
    events = []
    if succeeds or output_limit:
        events.extend(agent.stream("original task"))
        assert [e.content for e in events if e.type == "result"] == ["done" if succeeds else ""]
        if succeeds:
            assert not any(e.type == "error" for e in events)
        else:
            assert events[-1].raw["reason"] == "max_output_tokens"
            assert agent.trace.events[-1][0:2] == ("result", "")
    else:
        with pytest.raises(CLIProcessError):
            events.extend(agent.stream("original task"))
        assert not any(e.type == "result" for e in events)
        assert events[-1].type == "error"
    assert len(factory.calls) == calls
    recoveries = [e for e in events if e.type == "recovery"]
    assert len(recoveries) == calls - 1
    reason = "output_limit_resume" if output_limit and has_session else "error_retry"
    assert all(e.content["reason"] == reason for e in recoveries)
    assert not any(e.type == "message" and e.content == failure for e in events)
    for command, _ in factory.calls[1:]:
        if has_session:
            assert command[command.index("--resume") + 1] == "claude-retry"
        else:
            assert "--continue" in command
        assert "original task" not in command
        assert "Continue the interrupted task from where it stopped." in command
    assert agent.get_tokens().output_tokens == failed_tokens * (calls - int(succeeds)) + (
        10 if succeeds else 0
    )



@pytest.mark.parametrize("result", ["", "done"])
def test_claude_completed_response_does_not_resume(tmp_path, result):
    terminal = {
        "type": "result", "is_error": False, "session_id": "claude-completed",
        "stop_reason": "end_turn", "result": result,
        "usage": {"output_tokens": 256000},
    }
    factory = ProcessFactory(FakeProcess(stdout=json.dumps(terminal) + "\n"))
    agent = make_agent(ClaudeCodeAgent, tmp_path, process_factory=factory)

    events = agent.run("original task")

    assert len(factory.calls) == 1
    assert [e.content for e in events if e.type == "result"] == [result]
    assert not any(e.type == "recovery" for e in events)
    assert agent.get_tokens().output_tokens == 256000


def test_sighup_recovery_honors_max_recovery_attempts(tmp_path):
    factory = ProcessSequenceFactory(
        FakeProcess(returncode=129),
        FakeProcess(returncode=129),
        FakeProcess(stdout='{"role":"assistant","content":"must not run"}\n'),
    )
    agent = make_agent(
        KimiCodeAgent,
        tmp_path,
        model=FakeModel(protocols=("openai",)),
        process_factory=factory,
        max_recovery_attempts=1,
    )

    with pytest.raises(CLIProcessError) as error:
        agent.run("do the work")

    assert error.value.returncode == 129
    assert len(factory.calls) == 2


def test_rate_limit_recovery_retries_failed_resume(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("kimi", OAuthCredential("oauth", expires_at=4_000_000_000))
    oauth = Model(
        "kimi-code/k3",
        oauth="kimi",
        oauth_config=OAuthConfig(
            provider="kimi",
            store=store,
            auto_login=False,
            rate_limit_hook=lambda credential: {"limits": []},
        ),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(
            stdout='{"role":"assistant","content":"working"}\n',
            stderr="five hour rate limit reached (429)\n",
            returncode=1,
        ),
        FakeProcess(stderr="native session cannot be resumed\n", returncode=2),
        FakeProcess(stdout='{"role":"assistant","content":"must not run"}\n'),
    )
    agent = make_agent(KimiCodeAgent, tmp_path, model=oauth, process_factory=factory)

    agent.run("do the work")

    assert len(factory.calls) == 3
    assert all(command[1] == "--continue" for command, _ in factory.calls[1:])
    assert all("do the work" not in command for command, _ in factory.calls[1:])


def test_rate_limit_recovery_replays_prompt_when_no_session_started(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("kimi", OAuthCredential("oauth", expires_at=4_000_000_000))
    oauth = Model(
        "kimi-code/k3",
        oauth="kimi",
        oauth_config=OAuthConfig(
            provider="kimi",
            store=store,
            auto_login=False,
            rate_limit_hook=lambda credential: {"limits": []},
        ),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(stderr="five hour rate limit reached (429)\n", returncode=1),
        FakeProcess(stdout='{"role":"assistant","content":"must not run"}\n'),
    )
    agent = make_agent(KimiCodeAgent, tmp_path, model=oauth, process_factory=factory)

    agent.run("do the work")

    assert len(factory.calls) == 2
    assert factory.calls[0][0] == factory.calls[1][0]


def test_disabling_quota_wait_does_not_disable_retries(tmp_path):
    store = CredentialStore(tmp_path / "auth.json")
    store.save("openai", OAuthCredential("oauth", expires_at=4_000_000_000))
    oauth = Model(
        "oauth-primary",
        oauth="openai",
        oauth_config=OAuthConfig(provider="openai", store=store, auto_login=False),
    )
    factory = ProcessFactory(FakeProcess(stderr="HTTP 429 rate limit\n", returncode=1))
    agent = CodexCLIAgent(
        model=oauth,
        executable=executable(tmp_path, "codex"),
        dir=tmp_path,
        trace=FakeTrace(),
        process_factory=factory,
        auto_wait=False,
    )

    with pytest.raises(CLIProcessError):
        agent.run("do the work")

    assert agent.trace.events[-1][0] == "error"
    assert len(factory.calls) == 11


def test_closing_stream_terminates_and_reaps_process(tmp_path):
    process = FakeProcess(stdout='{"type":"thread.started","thread_id":"one"}\n', running=True)
    factory = ProcessFactory(process)
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    stream = agent.stream("hello")
    assert next(stream).type == "session"
    stream.close()
    assert process.terminated is True
    assert agent._process is None


def test_process_slot_is_held_while_spawning(tmp_path):
    agent = make_agent(CodexCLIAgent, tmp_path)
    lock_results = []

    def factory(command, **kwargs):
        def probe_lock():
            acquired = agent._process_lock.acquire(timeout=0.05)
            lock_results.append(acquired)
            if acquired:
                agent._process_lock.release()

        thread = threading.Thread(target=probe_lock)
        thread.start()
        thread.join()
        return FakeProcess()

    agent._process_factory = factory
    assert agent.run("hello") == []
    assert lock_results == [False]


def test_sandbox_prepare_command_is_used(tmp_path, monkeypatch):
    class Sandbox:
        def __init__(self):
            self.calls = []

        def prepare_command(self, command, cwd, env):
            self.calls.append((command, cwd, env))
            return ["sandbox", *command], Path("/tmp"), {**env, "IN_SANDBOX": "1"}

    sandbox = Sandbox()
    monkeypatch.setenv("HOST_ONLY_SECRET", "must-not-enter-sandbox")
    factory = ProcessFactory(FakeProcess())
    agent = make_agent(CodexCLIAgent, tmp_path, env=sandbox, process_factory=factory)
    assert agent.run("work") == []
    assert sandbox.calls[0][1] == tmp_path
    command, kwargs = factory.calls[0]
    assert command[0] == "sandbox"
    assert kwargs["cwd"] == Path("/tmp")
    assert kwargs["env"]["IN_SANDBOX"] == "1"
    assert "HOST_ONLY_SECRET" not in sandbox.calls[0][2]
    assert "--dangerously-bypass-approvals-and-sandbox" in command
    if os.name == "posix":
        assert kwargs["start_new_session"] is True


def test_sandbox_root_is_the_default_agent_root(tmp_path):
    class Sandbox:
        root = tmp_path

        def prepare_command(self, command, cwd, env):
            return command, cwd, env

    agent = CodexCLIAgent(
        model=FakeModel(),
        executable=executable(tmp_path, "codex"),
        env=Sandbox(),
        trace=FakeTrace(),
    )
    assert agent.root == tmp_path


def test_executable_discovery_install_metadata_and_versions(tmp_path, monkeypatch):
    installation = CLIInstallation("agent", "@example/agent", "1.2.3")
    assert installation.package_spec == "@example/agent@1.2.3"
    assert installation.install_command(Path("/prefix")) == [
        "harness-wrapper",
        "install-clis",
        "agent",
        "--prefix",
        "/prefix",
        "--version",
        "agent=1.2.3",
    ]
    binary = executable(tmp_path, "agent")
    assert find_executable("agent", explicit=binary) == str(binary.resolve())
    monkeypatch.setenv("PATH", "")
    with pytest.raises(ExecutableNotFoundError):
        find_executable("missing")
    assert extract_version("agent version v1.2.3 (build x)") == "1.2.3"

    def runner(command, **kwargs):
        return SimpleNamespace(stdout="agent 1.2.3", stderr="")

    assert validate_cli_version(str(binary), installation, runner=runner) == "1.2.3"

    def old_runner(command, **kwargs):
        return SimpleNamespace(stdout="agent 1.2.2", stderr="")

    with pytest.raises(UnsupportedCLIVersionError, match=r"pinned to 1\.2\.3"):
        validate_cli_version(str(binary), installation, runner=old_runner)


def test_adapter_installations_default_to_latest():
    assert ClaudeCodeAgent.installation.package_spec == "@anthropic-ai/claude-code@latest"
    assert CodexCLIAgent.installation.package_spec == "@openai/codex@latest"
    assert KimiCodeAgent.installation.package_spec == "@moonshot-ai/kimi-code@latest"


def test_invalid_subagent_name_is_rejected(tmp_path):
    agent = make_agent(CodexCLIAgent, tmp_path, subagents={'bad.name="injection"': "tiny"})
    with pytest.raises(ValueError, match="invalid subagent name"):
        agent.build_command("hello")


@pytest.mark.parametrize("event_type", ["error", "turn.failed"])
def test_codex_usage_warning_is_persisted_without_changing_counts(tmp_path, event_type):
    usage = {"input_tokens": 20763, "output_tokens": 2626}
    process = FakeProcess(
        stdout="\n".join(
            (
                json.dumps(
                    {"type": event_type, "message": "stream closed before response.completed"}
                ),
                json.dumps({"type": "turn.completed", "usage": usage}),
                "",
            )
        )
    )
    trace = Trace(root=tmp_path)
    agent = make_agent(
        CodexCLIAgent, tmp_path, trace=trace, process_factory=ProcessFactory(process), max_recovery_attempts=0
    )
    events = []
    with pytest.raises(CLIProcessError, match="stream closed"):
        events.extend(agent.stream("test"))
    warnings = [event for event in events if event.type == "warning"]
    assert len(warnings) == 1
    assert "token counts and cost may exclude usage" in warnings[0].content
    saved = [json.loads(line)["event"] for line in trace.path.read_text().splitlines()]
    assert any(
        event["type"] == "warning" and event["content"] == warnings[0].content for event in saved
    )
    assert agent.get_tokens() == TokenUsage(**usage)


@pytest.mark.parametrize("resume", [False, True])
def test_codex_large_prompt_reaches_real_process_via_stdin(tmp_path, resume):
    import hashlib

    prompt = "--paper-source\n" + "Mathematics: ∀n ∈ ℕ, λ(n) ≥ 0.\n" * 10_000
    expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cli = tmp_path / "codex-stdin-test"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import hashlib, json, sys\n"
        "assert sys.argv[-2:] == ['--', '-']\n"
        "digest = hashlib.sha256(sys.stdin.buffer.read()).hexdigest()\n"
        "print(json.dumps({'type': 'item.completed', 'item': "
        "{'type': 'agent_message', 'text': digest}}))\n"
    )
    cli.chmod(0o755)
    agent = CodexCLIAgent(
        model=FakeModel(), executable=cli, dir=tmp_path, trace=FakeTrace(),
        process_factory=subprocess.Popen,
    )
    events = agent.resume(prompt, session_id="test-session") if resume else agent.run(prompt)
    assert [event.content for event in events] == [expected]


def test_codex_prompt_input_is_closed_after_spawn_failure(tmp_path):
    opened = []

    def fail_to_spawn(command, **kwargs):
        opened.append(kwargs["stdin"])
        assert opened[-1].read() == b"full prompt"
        raise OSError("spawn failed")

    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=fail_to_spawn)
    with pytest.raises(OSError, match="spawn failed"):
        agent.run("full prompt")
    assert len(opened) == 11
    assert all(handle.closed for handle in opened)


def test_stopped_agent_does_not_retry_or_forward_requests(tmp_path):
    stopped = False

    def on_event(event):
        nonlocal stopped
        if event.type == "usage":
            stopped = True

    process = FakeProcess(stdout=(
        '{"type":"system","subtype":"init","session_id":"limited"}\n'
        '{"type":"result","is_error":false,"stop_reason":"max_tokens",'
        '"result":"","usage":{"output_tokens":100}}\n'
    ))
    factory = ProcessFactory(process)
    agent = make_agent(
        ClaudeCodeAgent, tmp_path, process_factory=factory,
        on_event=on_event, should_stop=lambda: stopped,
    )
    agent.run("question")
    assert len(factory.calls) == 1
    assert agent.get_tokens().output_tokens == 100
    assert agent._capture_model_request("/v1/messages", {}) is False


@pytest.mark.parametrize("answer", [None, "", " \n", "The answer is 42."])
@pytest.mark.parametrize("with_usage", [False, True])
def test_codex_terminal_result_preserves_answer_and_usage(tmp_path, answer, with_usage):
    native = [] if answer is None else [{
        "type": "item.completed", "item": {"type": "agent_message", "text": answer},
    }]
    terminal = {"type": "turn.completed"}
    if with_usage:
        terminal["usage"] = {"input_tokens": 10, "output_tokens": 25}
    native.append(terminal)
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=ProcessFactory(
        FakeProcess(stdout="".join(json.dumps(event) + "\n" for event in native))
    ))

    events = agent.run("Solve")

    assert events[-1].type == "result"
    assert events[-1].content == (answer or "")
    assert agent.get_tokens() == (TokenUsage(input_tokens=10, output_tokens=25) if with_usage else TokenUsage())


def test_codex_empty_resume_does_not_reuse_previous_answer_or_commentary(tmp_path):
    def turn(items):
        return FakeProcess(stdout="".join(json.dumps(event) + "\n" for event in [
            *items, {"type": "turn.completed", "usage": {"output_tokens": 7}},
        ]))

    factory = ProcessSequenceFactory(
        turn([{"type": "thread.started", "thread_id": "codex-session"}, {
            "type": "item.completed", "item": {"type": "agent_message", "text": "First answer"},
        }]),
        turn([{
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Still thinking", "phase": "commentary"},
        }]),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    assert agent.run("Solve")[-1].content == "First answer"
    resumed = agent.resume("Continue", session_id="codex-session")
    assert resumed[-1].type == "result" and resumed[-1].content == ""
    assert agent.get_tokens().output_tokens == 14


def test_codex_failed_turn_is_not_normalized_as_empty_completion(tmp_path):
    agent = make_agent(CodexCLIAgent, tmp_path)
    events = agent.normalize_event({"type": "turn.failed", "error": "Connection reset"})
    assert not any(event.type == "result" for event in events)


@pytest.mark.parametrize("failure", [
    "Connection reset by peer",
    "Invalid request: max_output_tokens must be a positive integer",
])
def test_claude_non_token_limit_failure_remains_an_error(tmp_path, failure):
    agent = make_agent(ClaudeCodeAgent, tmp_path, max_recovery_attempts=0,
                       process_factory=ProcessFactory(FakeProcess(stderr=failure, returncode=1)))
    with pytest.raises(CLIProcessError):
        agent.run("Solve")
    assert not any(event[0] == "result" for event in agent.trace.events)


def test_codex_progress_before_tool_use_is_not_an_empty_turns_answer(tmp_path):
    agent = make_agent(CodexCLIAgent, tmp_path)
    native = [
        {"type": "item.completed", "item": {"type": "agent_message", "text": "Checking a case"}},
        {"type": "item.completed", "item": {"type": "command_execution", "aggregated_output": "42"}},
        {"type": "turn.completed"},
    ]
    events = [event for raw in native for event in agent.normalize_event(raw)]
    assert events[-1].type == "result" and events[-1].content == ""


def test_codex_minimal_context_allows_explicit_subagents(tmp_path):
    agent = make_agent(
        CodexCLIAgent, tmp_path, minimal_context=True,
        subagents={"worker": FakeModel("gpt-6-astra")},
    )
    command = agent.build_command("delegate")
    assert "agents.enabled=true" in command
    assert "agents.enabled=false" not in command
    enabled = {command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--enable"}
    disabled = {command[i + 1] for i, arg in enumerate(command[:-1]) if arg == "--disable"}
    assert {"multi_agent", "multi_agent_v2"} <= enabled
    assert not {"multi_agent", "multi_agent_v2"} & disabled
    assert "plugins" in disabled
