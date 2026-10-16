from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.mistral_vibe import MistralVibeAgent
from harness_wrapper.harnesses.mistral_vibe.adapter import _ProviderErrorCutoff
from matharena.runner import Runner
from matharena.solvers.harness_solver import HarnessSolver
from test_harness_e2e import _environment, _image_available, _sandbox
from test_harness_e2e import docker_workspace as shared_docker_workspace

CONFIG = Path(__file__).resolve().parents[1] / "configs/models/mistral/mistral_large_4.yaml"


@pytest.fixture
def docker_workspace(tmp_path):
    yield from shared_docker_workspace.__wrapped__(tmp_path)


def make_agent(tmp_path, overrides=None, **kwargs):
    return MistralVibeAgent(
        model=Model(
            "mistralai/mistral-large-4-0",
            api_url="https://openrouter.ai/api/v1",
            api_key="placeholder",
            reasoning="high",
            request_overrides=overrides or {"max_tokens": 262144},
        ),
        executable="/usr/bin/true",
        dir=tmp_path,
        trace=None,
        subagents={},
        minimal_context=True,
        **kwargs,
    )


def test_vibe_routes_openrouter_and_preserves_parameters(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-only-openrouter-key")
    config = yaml.safe_load(CONFIG.read_text())
    scaffold = {"type": "pure_model", "model_config": config, "scaffold_config": None}
    runner = SimpleNamespace(comp_name="test", competition_config={"allow_harness": True})
    assert Runner._resolve_harness(runner, scaffold) == "vibe"
    runner.competition_config["allow_harness"] = False
    assert Runner._resolve_harness(runner, scaffold) is None
    config["harness_config"]["workspace_root"] = str(tmp_path / "p{problem_idx}_r{run_idx}")
    solver = HarnessSolver(scaffold, "{problem}", config, "")
    model = solver._build_model()
    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://openrouter.ai/api/v1"
    assert endpoint.headers["Authorization"] == "Bearer host-only-openrouter-key"
    assert endpoint.api_key != "host-only-openrouter-key"
    overrides = model.request_overrides()
    assert overrides["max_tokens"] == 262144
    assert overrides["reasoning_effort"] == "high"
    assert overrides["provider"] == {"allow_fallbacks": False, "order": ["mistral"]}
    assert "temperature" not in overrides and "extra_body" not in overrides
    assert not {"harness", "harness_version", "harness_config"} & overrides.keys()
    assert {"vibe", "vibe-cli", "mistral-vibe", "mistral"}.issubset(Agent.available())


def test_vibe_configuration_and_command(tmp_path):
    agent = make_agent(tmp_path, model_context_window=524288)
    env = agent.model_environment()
    assert env["HOME"] == str(tmp_path / ".harness-home")
    assert env["VIBE_HOME"] == str(tmp_path / ".harness-home/.vibe")
    assert env["VIBE_CLI"] == "python" and env["VIBE_TEST_DISABLE_KEYRING"] == "1"
    assert env["VIBE_HARNESS_API_KEY"] == "placeholder"
    config = (tmp_path / ".harness-home/.vibe/config.toml").read_text()
    for line in (
        "enable_telemetry = false",
        "enable_connectors = false",
        "enable_subagents = false",
        "include_project_context = false",
        "generate_titles = false",
        'reasoning_field_name = "reasoning"',
        'thinking = "high"',
        "max_context_length = 524288",
        "auto_compact_threshold = 253952",
    ):
        assert line in config
    profile = (tmp_path / ".harness-home/.bash_profile").read_text()
    assert "/proc/$$/environ" in profile and "LD_LIBRARY_PATH_ORIG" in profile
    command = agent.build_command("-")
    assert command[:2] == ["/usr/bin/true", "-p"] and command[2] == "--output"
    assert {"--auto-approve", "--trust", "streaming"}.issubset(command)
    assert command.count("--enabled-tools") == 5
    assert agent.build_command("-", resume=True, session_id="abc")[-2:] == ["--resume", "abc"]
    assert agent.build_command("-", resume=True, last=True)[-1] == "--continue"
    with pytest.raises(ValueError):
        make_agent(
            tmp_path, overrides={"temperature": 1.0}, model_context_window=524288
        ).model_environment()


def test_vibe_request_filter_drops_default_temperature_and_optional_context(tmp_path):
    agent = make_agent(tmp_path)
    payload = {
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You help users get complex tasks done. Use run_typescript."},
            {"role": "assistant", "content": None, "reasoning": "think"},
            {"role": "tool", "tool_call_id": "a", "content": "result"},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in (
                "search_tool_functions", "run_typescript", "read_file", "write_file",
                "edit", "bash", "todo", "unified_harness_scratchpad", "cron", "skill",
            )
        ],
    }
    agent._filter_request("/api/v1/chat/completions", payload)
    assert "temperature" not in payload
    assert [tool["function"]["name"] for tool in payload["tools"]] == [
        "read_file", "write_file", "edit", "bash", "todo",
    ]
    assert "run_typescript" not in payload["messages"][0]["content"]
    assert payload["messages"][1]["reasoning"] == "think"
    assert payload["messages"][2]["content"] == "result"
    compact = {"messages": [{"role": "system", "content": "Summarize this conversation"}]}
    agent._filter_request("/chat/completions", compact)
    assert compact["messages"][0]["content"] == "Summarize this conversation"
    explicit = make_agent(tmp_path, overrides={"max_tokens": 10, "temperature": 0.7})
    sampled = {"temperature": 0.7}
    explicit._filter_request("/chat/completions", sampled)
    assert sampled["temperature"] == 0.7
    with pytest.raises(ValueError):
        agent._filter_request("/responses", {})


class _Stream:
    def __init__(self, chunks):
        self.chunks = list(chunks)
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.closed = False

    def read1(self, _size=-1):
        return self.chunks.pop(0) if self.chunks else b""

    def close(self):
        self.closed = True


def test_vibe_withholds_mid_stream_provider_errors():
    ok = b'data: {"choices":[{"delta":{"reasoning":"partial"},"finish_reason":null}]}\n\n'
    failed = (
        b'data: {"error":{"code":502,"message":"Provider disconnected"},'
        b'"choices":[{"delta":{"content":""},"finish_reason":"error"}]}\n\n'
    )
    errors = []
    stream = _ProviderErrorCutoff(_Stream([ok[:20], ok[20:] + failed, b"data: [DONE]\n\n"]), errors.append)
    forwarded = b""
    while chunk := stream.read1(1024):
        forwarded += chunk
    assert forwarded == ok
    assert errors == [{"code": 502, "message": "Provider disconnected"}]
    healthy = _ProviderErrorCutoff(_Stream([ok, b"data: [DONE]"]), errors.append)
    assert healthy.read1() + healthy.read1() + healthy.read1() == ok + b"data: [DONE]"
    with healthy:
        pass
    assert healthy._response.closed


def test_vibe_normalizes_entries_and_skips_replayed_history(tmp_path):
    agent = make_agent(tmp_path)
    common = {"sessionId": "s1", "generationStatus": "completed"}
    entries = [
        {**common, "id": "u1", "type": "message", "role": "user", "content": [{"type": "text", "text": "q"}]},
        {**common, "id": "r1", "type": "reasoning", "text": "plan"},
        {
            **common, "id": "effect-1", "type": "effect", "title": "file_system.bash",
            "detail": {"toolName": "file_system.bash", "input": {"command": "echo 42"}},
            "state": {"status": "completed", "outputText": "42\n"},
        },
        {**common, "id": "a1", "type": "message", "role": "assistant", "content": [{"type": "text", "text": "\\boxed{42}"}]},
    ]
    events = [event for entry in entries for event in agent.normalize_event(entry)]
    assert [event.type for event in events] == [
        "session", "reasoning", "tool_call", "tool_result", "message",
    ]
    assert events[2].content == {"command": "echo 42"} and events[3].content == "42\n"
    assert events[2].tool_call_id == events[3].tool_call_id == "effect-1"
    assert all(event.session_id == "s1" for event in events)
    assert agent._final_text == "\\boxed{42}"
    assert all(agent.normalize_event(entry) == [] for entry in entries)


class FakeOpenRouter(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []
    fail_first: ClassVar[bool] = True

    def log_message(self, *_args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(
            {"path": self.path, "payload": payload, "authorization": self.headers.get("Authorization")}
        )
        base = {"id": "test", "object": "chat.completion.chunk", "created": 1, "model": payload["model"]}
        has_result = any(message.get("role") == "tool" for message in payload["messages"])
        if FakeOpenRouter.fail_first:
            FakeOpenRouter.fail_first = False
            events = [
                {**base, "choices": [{"index": 0, "delta": {"reasoning": "partial"}, "finish_reason": None}]},
                {**base, "error": {"code": 502, "message": "Provider disconnected"},
                 "choices": [{"index": 0, "delta": {}, "finish_reason": "error"}]},
            ]
        elif not has_result:
            call = {
                "index": 0, "id": "vibe-test-call", "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({
                        "command": "python -c 'import sympy; print(sympy.factorint(2026))' && sage -c 'print(2+2)'"
                    }),
                },
            }
            events = [{**base, "choices": [{"index": 0, "delta": {
                "role": "assistant", "content": None, "reasoning": "Check locally.", "tool_calls": [call],
            }, "finish_reason": "tool_calls"}]}]
        else:
            events = [{**base, "choices": [{"index": 0, "delta": {
                "role": "assistant", "content": "VIBE_OK", "reasoning": "Verified.",
            }, "finish_reason": "stop"}]}]
        if "error" not in events[-1]:
            usage = {"prompt_tokens": 30, "completion_tokens": 7, "total_tokens": 37,
                     "prompt_tokens_details": {"cached_tokens": 12}}
            events.append({**base, "choices": [], "usage": usage})
        body = "".join(f"data: {json.dumps(event)}\n\n" for event in events)
        if "error" not in events[-1]:
            body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


@pytest.mark.skipif(not _image_available(), reason="shared harness Docker image is not built")
def test_vibe_real_cli_tools_retry_resume_and_usage_without_paid_credits(docker_workspace):
    FakeOpenRouter.requests = []
    FakeOpenRouter.fail_first = True
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenRouter)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "mistralai/mistral-large-4-0",
        api_url=f"http://127.0.0.1:{server.server_port}/api/v1",
        api_key="placeholder",
        headers={"Authorization": "Bearer host-only-key"},
        reasoning="high",
        request_overrides={"max_tokens": 512, "reasoning_effort": "high"},
    )
    try:
        agent = Agent(
            type="vibe",
            model=model,
            executable="/usr/bin/true",
            dir=docker_workspace,
            env=_sandbox(docker_workspace, "vibe"),
            environment=_environment(),
            subagents={},
            minimal_context=True,
            model_context_window=524288,
            validate_version=False,
        )
        events = agent.run("Check Python and Sage locally, then reply VIBE_OK.")
        assert any(event.type == "tool_call" and event.tool_name == "file_system.bash" for event in events)
        assert any(event.type == "tool_result" and "1013" in str(event.content) for event in events)
        assert events[-1].type == "result" and events[-1].content == "VIBE_OK"
        first_id = agent.session_id
        assert first_id
        # The withheld provider error is retried natively; no title/telemetry calls.
        assert len(FakeOpenRouter.requests) == 3
        assert agent.get_tokens().input_tokens == 60
        assert agent.get_tokens().cache_read_tokens == 24
        resumed = agent.resume("Continue and confirm the result.", session_id=first_id)
        assert agent.session_id == first_id
        assert not any(event.type in {"tool_call", "tool_result"} for event in resumed)
        assert resumed[-1].type == "result" and resumed[-1].content == "VIBE_OK"
        assert len(FakeOpenRouter.requests) == 4
        assert agent.get_tokens().input_tokens == 90
        assert agent.get_tokens().output_tokens == 21
        capture = [json.loads(line) for line in agent.model_request_log_path.read_text().splitlines()]
        assert len(capture) == 4
        assert "host-only-key" not in json.dumps(capture)
        for request in FakeOpenRouter.requests:
            assert request["path"] == "/api/v1/chat/completions"
            assert request["authorization"] == "Bearer host-only-key"
            payload = request["payload"]
            assert payload["max_tokens"] == 512 and payload["reasoning_effort"] == "high"
            assert "temperature" not in payload
            assert {tool["function"]["name"] for tool in payload["tools"]} == {
                "bash", "read_file", "write_file", "edit", "todo",
            }
            assert payload["messages"][0]["content"].startswith("Solve the user's task")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_vibe_usage_survives_recovery_and_resets_for_new_run(tmp_path):
    agent = make_agent(tmp_path)
    agent._begin_token_session(resume=False, session_id="s1")
    payload = {"usage": {
        "prompt_tokens": 30, "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 12},
    }}
    agent._capture_model_response("/chat/completions", payload)
    agent._begin_token_session(resume=True, session_id="s1")
    agent._capture_model_response("/chat/completions", payload)
    assert agent.get_tokens().input_tokens == 60
    assert agent.get_tokens().output_tokens == 14
    assert agent.get_tokens().cache_read_tokens == 24
    agent._begin_token_session(resume=False, session_id="s2")
    assert agent.get_tokens().input_tokens == 0
    assert agent.get_tokens().output_tokens == 0
