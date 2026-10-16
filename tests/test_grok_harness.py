from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
import threading
import tomllib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
import yaml

from harness_wrapper import Agent, CLIProcessError, Model
from harness_wrapper.harnesses.grok_build import GrokBuildAgent
from harness_wrapper.harnesses.grok_build.adapter import _LOCAL_TOOLS, _reduced_output_budget
from matharena.api_client import APIClient
from matharena.runner import Runner
from matharena.solvers.harness_solver import HarnessSolver, _events_to_messages
from test_harness_e2e import _environment, _image_available, _sandbox
from test_harness_e2e import docker_workspace as shared_docker_workspace


@pytest.fixture
def docker_workspace(tmp_path):
    yield from shared_docker_workspace.__wrapped__(tmp_path)


def make_agent(tmp_path):
    return GrokBuildAgent(
        model=Model(
            "x-ai/grok-4.7",
            api_url="https://openrouter.ai/api/v1",
            api_key="placeholder",
            reasoning="xhigh",
        ),
        executable="/usr/bin/true",
        dir=tmp_path,
        subagents={},
        minimal_context=True,
        model_context_window=500000,
        max_recovery_attempts=0,
    )


def test_grok_config_uses_openrouter_and_provider_sampling_defaults(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("OPENROUTER_API_KEY", "host-only-openrouter-key")
    runner = Runner.__new__(Runner)
    path = "configs/models/xai/grok-47.yaml"
    for comp in ("arxiv/may", "arxiv/june", "arxiv/august", "arxivlean/june"):
        runner.comp_name = comp
        runner.competition_config = yaml.safe_load(
            Path(f"configs/competitions/{comp}.yaml").read_text()
        )
        solver_config = runner.load_solver_config(path)
        config = runner._prepare_default_api_client_args(solver_config["model_config"])
        harness = runner._resolve_harness(solver_config)
        if comp in ("arxiv/may", "arxiv/june"):
            assert harness is None
            for key in ("harness", "harness_version", "harness_config"):
                config.pop(key, None)
            client = APIClient(**config)
            assert client.use_openai_responses_api is True
            assert client.kwargs["max_output_tokens"] == 450000
            assert client.kwargs["extra_body"]["store"] is False
            assert client.model == "x-ai/grok-4.7"
            assert client.base_url == "https://openrouter.ai/api/v1"
            assert client.kwargs["reasoning"] == {"effort": "xhigh"}
            assert "temperature" not in client.kwargs and "top_p" not in client.kwargs
        else:
            assert harness == "grok"
            config["harness_config"]["workspace_root"] = str(
                tmp_path / "p{problem_idx}_r{run_idx}"
            )
            model = HarnessSolver(solver_config, "{problem}", config, "")._build_model()
            endpoint = model.endpoint_for("openai")
            assert endpoint.url == "https://openrouter.ai/api/v1"
            assert (
                endpoint.headers["Authorization"] == "Bearer host-only-openrouter-key"
            )
            assert endpoint.api_key != "host-only-openrouter-key"
            overrides = model.request_overrides()
            assert overrides["reasoning"] == {"effort": "xhigh"}
            assert overrides["max_output_tokens"] == 450000
            assert overrides["store"] is False
            assert overrides["provider"] == {"only": ["xai"], "allow_fallbacks": False}
            assert (
                not {"temperature", "top_p", "harness", "harness_config"}
                & overrides.keys()
            )


def test_grok_config_and_prompt_file_are_isolated(tmp_path):
    agent = make_agent(tmp_path)
    env = agent.model_environment()
    home = tmp_path / ".harness-home/.grok"
    assert env["GROK_HOME"] == str(home)
    config = tomllib.loads((home / "config.toml").read_text())
    model = config["model"]["harness-wrapper"]
    assert model["api_backend"] == "responses"
    assert model["context_window"] == 500000
    assert "temperature" not in model and "top_p" not in model
    assert config["compat"]["claude"]["skills"] is False
    assert config["memory"]["enabled"] is False
    assert config["session"]["load_envrc"] is False
    prompt = "A long problem\n" * 20000
    command = agent.build_command(prompt)
    assert prompt not in command
    assert Path(command[command.index("--prompt-file") + 1]).read_text() == prompt
    assert command[command.index("--reasoning-effort") + 1] == "xhigh"
    assert agent.build_command("continue", resume=True, session_id="abc")[-2:] == [
        "--resume",
        "abc",
    ]
    assert agent.build_command(None, resume=True, last=True)[-1] == "--continue"
    assert {"grok", "grok-build", "grok-cli"}.issubset(Agent.available())


def test_grok_filter_keeps_local_tools_title_schema_and_full_history(tmp_path):
    agent = make_agent(tmp_path)
    payload = {
        "tools": [
            {"type": "function", "name": name}
            for name in (
                "run_terminal_command",
                "read_file",
                "web_search",
                "search_tool",
                "image_gen",
            )
        ],
        "input": [{"type": "function_call_output", "call_id": "a", "output": "result"}],
        "max_output_tokens": 256,
        "store": True,
    }
    agent._filter_request("/responses", payload)
    assert [t["name"] for t in payload["tools"]] == [
        "run_terminal_command",
        "read_file",
    ]
    assert payload["input"] == [
        {"type": "function_call_output", "call_id": "a", "output": "result"}
    ]
    assert payload["store"] is False
    title = {"tools": [{"type": "function", "name": "session_title"}]}
    agent._filter_request("/responses", title)
    assert len(title["tools"]) == 1
    with pytest.raises(ValueError):
        agent._filter_request("/chat/completions", {})
    with pytest.raises(ValueError, match="full Responses history"):
        agent._filter_request("/responses", {"previous_response_id": "resp_old"})


def test_grok_results_errors_and_usage(tmp_path):
    agent = make_agent(tmp_path)
    agent.normalize_event({"type": "text", "data": "I will calculate."})
    call = agent.normalize_event(
        {
            "type": "tool_call",
            "toolCallId": "call",
            "toolName": "read_file",
            "rawInput": {},
        }
    )
    assert call[0].type == "tool_call"
    result = agent.normalize_event(
        {
            "type": "tool_call_update",
            "toolCallId": "call",
            "status": "failed",
            "rawOutput": "missing",
        }
    )
    assert result[0].type == "tool_result" and result[0].tool_name == "read_file"
    agent.normalize_event({"type": "text", "data": "Answer"})
    end = agent.normalize_event(
        {"type": "end", "stopReason": "end_turn", "sessionId": "session"}
    )
    assert end[0].content == "Answer" and end[0].session_id == "session"
    agent._final_text = ""
    assert (
        agent.normalize_event({"type": "end", "stopReason": "end_turn"})[0].content
        == ""
    )
    assert (
        agent.normalize_event({"type": "end", "stopReason": "max_tokens"})[0].type
        == "error"
    )
    usage = {
        "input_tokens": 30,
        "output_tokens": 7,
        "total_tokens": 37,
        "input_tokens_details": {"cached_tokens": 12},
        "output_tokens_details": {"reasoning_tokens": 3},
    }
    for _ in range(2):
        agent._capture_model_response(
            "/responses", {"type": "response.created", "response": {"usage": usage}}
        )
        agent._capture_model_response(
            "/responses", {"type": "response.completed", "response": {"usage": usage}}
        )
    assert agent.get_tokens().input_tokens == 60
    assert agent.get_tokens().output_tokens == 14  # Reasoning already included.
    assert agent.get_tokens().cache_read_tokens == 24
    agent._begin_token_session(resume=False, session_id=None)
    assert agent.get_tokens().input_tokens == 0


def test_grok_missing_terminal_event_is_not_an_empty_completion(tmp_path):
    agent = make_agent(tmp_path)
    with pytest.raises(CLIProcessError, match="without a terminal event"):
        agent.run("An invocation that exits before Grok starts.")
    assert agent.session_id  # Available to the generic retry machinery.
    assert not hasattr(agent, "_request_capture_url")


CONTEXT_ERROR = (
    "This endpoint's maximum context length is 500000 tokens. However, you requested about "
    "502422 tokens (49344 of text input, 3078 of tool input, 450000 in the output). "
    "Please reduce the length of either one."
)


@pytest.mark.parametrize("message,budget,expected", [
    (CONTEXT_ERROR, 450000, 439386),
    (CONTEXT_ERROR.replace("49344", "249344"), 450000, 239386),
    (CONTEXT_ERROR.replace("49344", "499344"), 450000, None),
    (CONTEXT_ERROR.replace("500000", "500,000"), 450000, 439386),
    ("Invalid tool schema", 450000, None),
    (CONTEXT_ERROR, 4096, None),
    (CONTEXT_ERROR, None, None),
])
def test_grok_output_budget_uses_provider_counts(message, budget, expected):
    assert _reduced_output_budget(message, budget) == expected


@pytest.mark.parametrize("status,message,expected_calls", [
    (400, CONTEXT_ERROR, 2),
    (400, "Invalid tool schema", 1),
    (400, CONTEXT_ERROR.replace("49344", "499344"), 1),
    (429, CONTEXT_ERROR, 1),
    (500, "Service temporarily unavailable", 1),
])
def test_grok_context_retry_is_bounded_and_preserves_errors(
    tmp_path, monkeypatch, status, message, expected_calls
):
    agent = make_agent(tmp_path)
    requests = []
    body = json.dumps({"error": {"code": status, "message": message}}).encode()

    def fail(request, *, timeout):
        requests.append(json.loads(request.data))
        raise urllib.error.HTTPError(request.full_url, status, "test", {}, io.BytesIO(body))

    monkeypatch.setattr(urllib.request, "urlopen", fail)
    request = urllib.request.Request("http://localhost/responses", data=json.dumps({
        "model": "x-ai/grok-4.7", "max_output_tokens": 450000,
        "input": [{"role": "user", "content": "Preserve this conversation."}],
    }).encode())
    with pytest.raises(urllib.error.HTTPError) as raised:
        agent._open_budgeted_request(request, timeout=10)
    assert raised.value.code == status
    assert raised.value.read() == body
    assert len(requests) == expected_calls
    if expected_calls == 2:
        assert requests[1] == {**requests[0], "max_output_tokens": 439386}
        payload = {"max_output_tokens": 450000}
        agent._filter_request("/responses", payload)
        assert payload["max_output_tokens"] == 439386


def test_grok_context_allowance_shrinks_again_as_history_grows(tmp_path, monkeypatch):
    agent = make_agent(tmp_path)
    requests = []
    errors = [CONTEXT_ERROR, CONTEXT_ERROR.replace("49344", "249344").replace("450000", "439386")]
    success = object()

    def upstream(request, *, timeout):
        requests.append(json.loads(request.data))
        if len(requests) in (1, 3):
            body = json.dumps({"error": {"message": errors.pop(0)}}).encode()
            raise urllib.error.HTTPError(request.full_url, 400, "context", {}, io.BytesIO(body))
        return success

    monkeypatch.setattr(urllib.request, "urlopen", upstream)
    payload = {
        "model": "x-ai/grok-4.7", "max_output_tokens": 450000,
        "reasoning": {"effort": "xhigh"}, "store": False,
        "input": [
            {"type": "reasoning", "summary": [], "encrypted_content": "opaque-history"},
            {"type": "function_call", "call_id": "c1", "name": "read_file", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": "Retained tool result."},
        ],
        "tools": [{"type": "function", "name": "read_file", "parameters": {"type": "object"}}],
    }
    for expected in (439386, 239386):
        agent._filter_request("/responses", payload)
        request = urllib.request.Request("http://localhost/responses", data=json.dumps(payload).encode())
        assert agent._open_budgeted_request(request, timeout=10) is success
        assert requests[-1] == {**requests[-2], "max_output_tokens": expected}
        assert requests[-1]["input"] == payload["input"]
        assert agent._output_token_ceiling == expected
        smaller = {**payload, "max_output_tokens": 4096}
        agent._filter_request("/responses", smaller)
        assert smaller["max_output_tokens"] == 4096


class FakeGrok(BaseHTTPRequestHandler):
    requests: ClassVar[list] = []
    context_overflow: ClassVar[bool] = False

    def log_message(self, *_args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.requests.append(
            {
                "path": self.path,
                "body": payload,
                "authorization": self.headers.get("Authorization"),
            }
        )
        # Reject stateful requests exactly as OpenRouter does.
        if payload.get("store") or payload.get("previous_response_id"):
            self.send_error(400, "Stateless Responses required")
            return
        tools = {t["name"] for t in payload.get("tools", [])}
        has_result = any(
            m.get("type") == "function_call_output" for m in payload.get("input", [])
        )
        if self.context_overflow and has_result and payload.get("max_output_tokens", 0) > 439386:
            self.requests[-1]["rejected"] = True
            body = json.dumps({"error": {"code": 400, "message": CONTEXT_ERROR}}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if tools == {"session_title"}:
            name, args = "session_title", {"session_title": "Check Python and Sage"}
        elif not has_result and "run_terminal_command" in tools:
            name, args = (
                "run_terminal_command",
                {
                    "command": "python -c 'import sympy; print(sympy.factorint(2026))' && sage -c 'print(2+2)'",
                    "description": "Check local Python and Sage",
                },
            )
        else:
            name, args = None, None
        i = len(self.requests)
        if name:
            item = {
                "type": "function_call",
                "id": f"fc_{i}",
                "call_id": f"call_{i}",
                "name": name,
                "arguments": json.dumps(args),
                "status": "completed",
            }
        else:
            item = {
                "type": "message",
                "id": f"msg_{i}",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "GROK_OK", "annotations": []}
                ],
            }
        response = {
            "id": f"resp_{i}",
            "object": "response",
            "created_at": 1,
            "status": "completed",
            "model": "x-ai/grok-4.7",
            "output": [item],
            "usage": {
                "input_tokens": 30,
                "output_tokens": 7,
                "total_tokens": 37,
                "input_tokens_details": {"cached_tokens": 12},
                "output_tokens_details": {"reasoning_tokens": 3},
            },
        }
        events = [
            {
                "type": "response.created",
                "response": {**response, "status": "in_progress", "output": []},
            }
        ]
        if name:
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "arguments": ""},
                },
                {
                    "type": "response.function_call_arguments.delta",
                    "item_id": item["id"],
                    "output_index": 0,
                    "delta": item["arguments"],
                },
            ]
        else:
            events += [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": {**item, "content": []},
                },
                *[
                    {
                        "type": "response.output_text.delta",
                        "item_id": item["id"],
                        "output_index": 0,
                        "content_index": 0,
                        "delta": chunk,
                    }
                    for chunk in ("GROK_", "OK")
                ],
            ]
        events += [
            {"type": "response.output_item.done", "output_index": 0, "item": item},
            {"type": "response.completed", "response": response},
        ]
        body = "".join(
            "data: " + json.dumps({**event, "sequence_number": n}) + "\n\n"
            for n, event in enumerate(events)
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.skipif(
    not _image_available(), reason="shared harness Docker image is not built"
)
@pytest.mark.parametrize("context_overflow", [False, True])
def test_grok_real_cli_tools_resume_and_usage_without_paid_credits(docker_workspace, monkeypatch, context_overflow):
    FakeGrok.requests = []
    monkeypatch.setattr(FakeGrok, "context_overflow", context_overflow)
    output_ceiling = 450000 if context_overflow else 4096
    final_text = "GROK_OK"
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeGrok)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "x-ai/grok-4.7",
        api_url=f"http://127.0.0.1:{server.server_port}",
        api_key="placeholder",
        headers={"Authorization": "Bearer host-only-key"},
        reasoning="xhigh",
        request_overrides={"max_output_tokens": output_ceiling, "reasoning": {"effort": "xhigh"}},
    )
    try:
        agent = Agent(
            type="grok",
            model=model,
            executable="/usr/bin/true",
            dir=docker_workspace,
            env=_sandbox(docker_workspace, "grok"),
            environment=_environment(),
            subagents={},
            minimal_context=True,
            max_recovery_attempts=0,
        )
        events = agent.run("Check Python and Sage locally, then reply GROK_OK.")
        assert any(
            e.type == "tool_call" and e.tool_name == "run_terminal_command"
            for e in events
        )
        assert any(e.type == "tool_result" and "1013" in str(e.content) for e in events)
        assert events[-1].type == "result" and events[-1].content == final_text
        messages = _events_to_messages(events, harness="grok")
        assert [m["content"] for m in messages if m.get("type") == "response"] == [final_text]
        session_id = agent.session_id
        assert session_id and events[0].session_id == session_id
        first_count = sum(not r.get("rejected") for r in FakeGrok.requests)
        assert agent.get_tokens().input_tokens == 30 * first_count
        assert agent.get_tokens().output_tokens == 7 * first_count
        assert agent.get_tokens().cache_read_tokens == 12 * first_count
        first_request_count = len(FakeGrok.requests)
        resumed = agent.resume(
            "Use the previous result and reply GROK_OK.", session_id=session_id
        )
        assert agent.session_id == session_id
        assert resumed[-1].type == "result" and resumed[-1].content == final_text
        resumed_messages = _events_to_messages(resumed, harness="grok")
        assert [m["content"] for m in resumed_messages if m.get("type") == "response"] == [final_text]
        count = sum(not r.get("rejected") for r in FakeGrok.requests)
        assert count > first_count
        assert agent.get_tokens().input_tokens == 30 * count
        assert agent.get_tokens().output_tokens == 7 * count
        assert agent.get_tokens().cache_read_tokens == 12 * count
        for request in FakeGrok.requests:
            payload = request["body"]
            assert request["authorization"] == "Bearer host-only-key"
            assert request["path"].endswith("/responses")
            assert (
                payload["model"] == "x-ai/grok-4.7"
                and payload["max_output_tokens"] <= output_ceiling
            )
            names = {t["name"] for t in payload.get("tools", [])}
            assert names <= _LOCAL_TOOLS or names == {"session_title"}
            if names != {"session_title"}:
                assert "temperature" not in payload and "top_p" not in payload
                assert payload["reasoning"]["effort"] == "xhigh"
        followups = [r["body"] for r in FakeGrok.requests[first_request_count:]]
        assert any(
            any(
                item.get("type") == "function_call_output"
                for item in p.get("input", [])
            )
            for p in followups
        )
        assert any(
            "Check Python and Sage locally" in json.dumps(p.get("input", []))
            for p in followups
        )
        assert all(
            p["store"] is False and "previous_response_id" not in p for p in followups
        )
        if context_overflow:
            rejected = [i for i, r in enumerate(FakeGrok.requests) if r.get("rejected")]
            assert len(rejected) == 1
            failed_body = FakeGrok.requests[rejected[0]]["body"]
            assert any(r["body"] == {**failed_body, "max_output_tokens": 439386}
                       for r in FakeGrok.requests[rejected[0] + 1:])
            assert all(p["max_output_tokens"] <= 439386 for p in followups)
            # Rejection correction must not re-run the successful local tool.
            assert sum(e.type == "tool_call" for e in events) == 1
        request_log = agent.model_request_log_path.read_text()
        assert (
            "host-only-key" not in request_log and "run_terminal_command" in request_log
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
