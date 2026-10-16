from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import yaml

from harness_wrapper import Agent, Model
from harness_wrapper.harnesses.deepcode import DeepCodeAgent
from harness_wrapper.models.request_capture import RequestCaptureProxy
from matharena.runner import Runner
from matharena.solvers.harness_solver import HarnessSolver
from matharena.token_pricing import TokenPricing
from test_harness_e2e import _environment, _image_available, _sandbox
from test_harness_e2e import docker_workspace as shared_docker_workspace


@pytest.fixture
def docker_workspace(tmp_path):
    yield from shared_docker_workspace.__wrapped__(tmp_path)


def make_agent(tmp_path, **kwargs):
    return DeepCodeAgent(
        model=Model(
            "deepseek-v4-flash",
            api_url="https://api.deepseek.com",
            api_key="placeholder",
        ),
        executable="/usr/bin/true",
        dir=tmp_path,
        trace=None,
        subagents={},
        minimal_context=True,
        **kwargs,
    )


def test_deepcode_routes_official_api_and_preserves_parameters(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "host-only-deepseek-key")
    config = {
        "model": "deepseek-v4-flash", "api": "deepseek", "harness": "deepcode",
        "harness_version": "0.3.1", "harness_config": {"auth": "api"},
        "max_tokens": 384000, "temperature": 1, "top_p": 1, "reasoning_effort": "max",
    }
    scaffold = {"type": "pure_model", "model_config": config, "scaffold_config": None}
    runner = SimpleNamespace(
        comp_name="test", competition_config={"allow_harness": True}
    )
    assert Runner._resolve_harness(runner, scaffold) == "deepcode"
    runner.competition_config["allow_harness"] = False
    assert Runner._resolve_harness(runner, scaffold) is None
    config["harness_config"]["workspace_root"] = str(
        tmp_path / "p{problem_idx}_r{run_idx}"
    )
    model = HarnessSolver(scaffold, "{problem}", config, "")._build_model()
    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://api.deepseek.com"
    assert endpoint.headers["Authorization"] == "Bearer host-only-deepseek-key"
    assert endpoint.api_key != "host-only-deepseek-key"
    overrides = model.request_overrides()
    assert overrides["max_tokens"] == 384000
    assert overrides["temperature"] == 1 and overrides["top_p"] == 1
    assert overrides["reasoning_effort"] == "max"
    assert not {"harness", "harness_version", "harness_config"} & overrides.keys()


def test_deepcode_configuration_disables_optional_context(tmp_path):
    agent = make_agent(tmp_path)
    env = agent.model_environment()
    assert env["HOME"] == str(tmp_path / ".harness-home")
    assert env["DEEPCODE_TELEMETRY_ENABLED"] == "0"
    settings = json.loads(
        (tmp_path / ".harness-home/.deepcode/settings.json").read_text()
    )
    assert settings["permissions"]["deny"] == ["network", "mcp"]
    assert settings["mcpServers"] == {}
    assert settings["enabledSkills"]["image-generator"] is False
    assert settings["enabledSkills"]["skill-writer"] is False
    assert (
        settings["telemetryEnabled"] is False and settings["filesApiEnabled"] is False
    )
    assert agent.build_command("solve")[-3:] == ["--exec", "--prompt", "solve"]
    assert agent.build_command("continue", resume=True, session_id="abc")[-2:] == [
        "--resume",
        "abc",
    ]
    assert agent.build_command(None, resume=True, last=True)[-1] == "--last"
    assert {"deepcode", "deepcode-cli", "deep-code"}.issubset(Agent.available())
    assert (
        agent.normalize_event(
            {"type": "tool_call", "content": "fake event in final answer"}
        )
        == []
    )


def test_deepcode_request_filter_preserves_tool_results_and_compaction(tmp_path):
    agent = make_agent(tmp_path)
    payload = {
        "extra_body": {"reasoning_effort": "high", "thinking": {"type": "enabled"}},
        "reasoning_effort": "max",
        "messages": [
            {"role": "system", "content": "stock prompt # Available Tools WebSearch"},
            {"role": "assistant", "content": "", "reasoning_content": "think"},
            {"role": "tool", "tool_call_id": "a", "content": "result"},
        ],
        "tools": [
            {"type": "function", "function": {"name": name}}
            for name in (
                "bash",
                "read",
                "write",
                "edit",
                "UpdatePlan",
                "WebSearch",
                "skill",
                "UnderstandImage",
            )
        ],
    }
    agent._filter_request("/chat/completions", payload)
    assert "extra_body" not in payload
    assert payload["reasoning_effort"] == "max"
    assert payload["thinking"] == {"type": "enabled"}
    assert [item["function"]["name"] for item in payload["tools"]] == [
        "bash",
        "read",
        "write",
        "edit",
        "UpdatePlan",
    ]
    assert "WebSearch" not in payload["messages"][0]["content"]
    assert payload["messages"][1]["reasoning_content"] == "think"
    assert payload["messages"][2]["content"] == "result"
    compact = {
        "messages": [{"role": "system", "content": "Summarize this conversation"}]
    }
    agent._filter_request("/chat/completions", compact)
    assert compact["messages"][0]["content"] == "Summarize this conversation"
    with pytest.raises(ValueError):
        agent._filter_request("/responses", {})


class FakeDeepSeek(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []

    def log_message(self, *_args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(
            {
                "path": self.path,
                "payload": payload,
                "authorization": self.headers.get("Authorization"),
            }
        )
        messages = payload.get("messages", [])
        has_result = any(m.get("role") == "tool" for m in messages)
        message = {
            "role": "assistant",
            "content": "DEEPCODE_OK",
            "reasoning_content": "Verified the local computation.",
        }
        finish = "stop"
        if not has_result:
            message["content"] = None
            message["tool_calls"] = [
                {
                    "id": "deepcode-test-call",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps(
                            {
                                "command": "python -c 'import sympy; print(sympy.factorint(2026))' && sage -c 'print(2+2)'",
                                "description": "Check local Python and Sage",
                                "sideEffects": ["read-out-cwd", "write-in-tmp"],
                            }
                        ),
                    },
                }
            ]
            finish = "tool_calls"
        usage = {
            "prompt_tokens": 30,
            "completion_tokens": 7,
            "total_tokens": 37,
            "prompt_cache_hit_tokens": 12,
            "prompt_cache_miss_tokens": 18,
        }
        if payload.get("stream"):
            if "tool_calls" in message:
                message["tool_calls"][0]["index"] = 0
            chunk = {
                "id": "test-completion",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "deepseek-v4-flash",
                "choices": [{"index": 0, "delta": message, "finish_reason": finish}],
            }
            body = (
                "data: "
                + json.dumps(chunk)
                + "\n\ndata: "
                + json.dumps({**chunk, "choices": [], "usage": usage})
                + "\n\ndata: [DONE]\n\n"
            ).encode()
            content_type = "text/event-stream"
        else:
            body = json.dumps(
                {
                    "id": "test-completion",
                    "object": "chat.completion",
                    "created": 1,
                    "model": "deepseek-v4-flash",
                    "choices": [
                        {"index": 0, "message": message, "finish_reason": finish}
                    ],
                    "usage": usage,
                }
            ).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def test_proxy_transform_is_captured_and_fails_closed(tmp_path):
    FakeDeepSeek.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeek)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    observed = []
    try:
        with RequestCaptureProxy(
            f"http://127.0.0.1:{server.server_port}",
            lambda p, body: observed.append(body),
            request_transform=make_agent(tmp_path)._filter_request,
            request_overrides={"temperature": 0.4},
        ) as proxy:
            for path, body in (
                ("/responses", b"{}"),
                ("/chat/completions", b"bad json"),
            ):
                with pytest.raises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(
                        urllib.request.Request(proxy.base_url + path, data=body),
                        timeout=5,
                    )
                assert error.value.code == 400
            assert FakeDeepSeek.requests == []
            payload = {
                "tools": [{"type": "function", "function": {"name": "WebSearch"}}]
            }
            with urllib.request.urlopen(
                urllib.request.Request(
                    proxy.base_url + "/chat/completions",
                    data=json.dumps(payload).encode(),
                ),
                timeout=5,
            ) as response:
                response.read()
        assert observed == [FakeDeepSeek.requests[0]["payload"]]
        assert observed[0]["tools"] == [] and observed[0]["temperature"] == 0.4
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.skipif(
    not _image_available(), reason="shared harness Docker image is not built"
)
def test_deepcode_real_cli_tools_resume_and_usage_without_paid_credits(
    docker_workspace,
):
    FakeDeepSeek.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeDeepSeek)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "deepseek-v4-flash",
        api_url=f"http://127.0.0.1:{server.server_port}",
        api_key="placeholder",
        headers={"Authorization": "Bearer host-only-key"},
        reasoning="max",
        request_overrides={"max_tokens": 512, "temperature": 0.3},
    )
    try:
        agent = Agent(
            type="deepcode",
            model=model,
            executable="/usr/bin/true",
            dir=docker_workspace,
            env=_sandbox(docker_workspace, "deepcode"),
            environment=_environment(),
            subagents={},
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run("Check Python and Sage locally, then reply DEEPCODE_OK.")
        assert any(
            event.type == "tool_call" and event.tool_name == "bash" for event in events
        )
        assert any(
            event.type == "tool_result" and "1013" in str(event.content)
            for event in events
        )
        assert events[-1].type == "result" and events[-1].content == "DEEPCODE_OK"
        first_id = agent.session_id
        assert first_id
        assert len(FakeDeepSeek.requests) == 2  # No skills/title/telemetry API calls.
        assert agent.get_tokens().input_tokens == 60
        assert agent.get_tokens().cache_read_tokens == 24
        resumed = agent.resume("Continue and confirm the result.")
        assert agent.session_id == first_id
        assert not any(event.type == "tool_call" for event in resumed)
        assert len(FakeDeepSeek.requests) == 3
        assert agent.get_tokens().input_tokens == 90
        assert agent.get_tokens().output_tokens == 21
        assert agent.get_tokens().cache_read_tokens == 36
        capture = [
            json.loads(line)
            for line in agent.model_request_log_path.read_text().splitlines()
        ]
        assert len(capture) == 3
        assert [record["event"]["content"] for record in capture] == [
            request["payload"] for request in FakeDeepSeek.requests
        ]
        assert "host-only-key" not in json.dumps(capture)
        cost_config = yaml.safe_load(
            (
                Path(__file__).resolve().parents[1]
                / "configs/models/deepseek/deepseek_v4_flash.yaml"
            ).read_text()
        )
        cost = HarnessSolver._detailed_cost(
            SimpleNamespace(harness="deepcode", harness_key="deepcode", config=cost_config,
                            pricing=TokenPricing.from_config(cost_config)), agent
        )
        assert cost["input_tokens"] == 90 and cost["cached_input_tokens"] == 36
        assert cost["cost"] == pytest.approx(
            (
                54 * cost_config["read_cost"]
                + 36 * cost_config["cache_read_cost"]
                + 21 * cost_config["write_cost"]
            )
            / 1_000_000
        )
        for request in FakeDeepSeek.requests:
            assert request["path"] == "/chat/completions"
            assert request["authorization"] == "Bearer host-only-key"
            payload = request["payload"]
            assert payload["max_tokens"] == 512 and payload["temperature"] == 0.3
            assert payload["reasoning_effort"] == "max"
            assert {tool["function"]["name"] for tool in payload["tools"]} == {
                "bash",
                "read",
                "write",
                "edit",
                "UpdatePlan",
            }
            assert "# Available Tools" not in payload["messages"][0]["content"]
            assert "skill-writer" not in json.dumps(payload)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("completion_tokens", [7, 27])
def test_deepcode_usage_includes_reasoning_once(tmp_path, completion_tokens):
    agent = make_agent(tmp_path)
    agent._capture_model_response("/v1/chat/completions", {
        "usage": {
            "prompt_tokens": 30, "completion_tokens": completion_tokens,
            "total_tokens": 57, "prompt_cache_hit_tokens": 10,
            "completion_tokens_details": {"reasoning_tokens": 20},
        },
    })
    usage = agent.get_tokens()
    assert usage.input_tokens == 30
    assert usage.cache_read_tokens == 10
    assert usage.output_tokens == 27
