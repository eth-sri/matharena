"""Retry allowances follow completed provider requests, not replayed CLI output."""

import json

import pytest
from test_agents import FakeProcess, ProcessSequenceFactory, make_agent

from harness_wrapper.harnesses.codex_cli import CodexCLIAgent
from harness_wrapper.models.request_capture import ResponseCapture
from harness_wrapper.tools import CLIProcessError


@pytest.fixture(autouse=True)
def no_delays(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


def capture_response(agent, path, payload, content_type="text/event-stream"):
    capture = ResponseCapture(
        path, content_type, agent._capture_model_response, component="test"
    )
    body = json.dumps(payload).encode()
    capture.feed(b"data: " + body + b"\n\n" if "event-stream" in content_type else body)
    capture.finish()


@pytest.mark.parametrize("path,payload,content_type", [
    ("/v1/chat/completions", {"choices": [{"finish_reason": "stop"}]}, "text/event-stream"),
    ("/v1/chat/completions", {"choices": [{"finish_reason": "tool_calls"}]}, "text/event-stream"),
    ("/v1/chat/completions", {"choices": [{"finish_reason": "function_call"}]}, "application/json"),
    ("/v1/responses", {"type": "response.completed", "response": {"status": "completed"}}, "text/event-stream"),
    ("/v1/responses", {"object": "response", "status": "completed"}, "application/json"),
    ("/v1/messages", {"type": "message_delta", "delta": {"stop_reason": "tool_use"}}, "text/event-stream"),
    ("/v1/messages", {"type": "message", "stop_reason": "end_turn"}, "application/json"),
    ("/streamGenerateContent", {"candidates": [{"finishReason": "STOP"}]}, "text/event-stream"),
])
def test_completed_request_resets_exhausted_retry_allowance(tmp_path, path, payload, content_type):
    processes = ProcessSequenceFactory(
        FakeProcess(stderr="HTTP 504", returncode=1),
        FakeProcess(stderr="HTTP 504", returncode=1),
        FakeProcess(stdout=json.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": "done",
        }}) + "\n"),
    )

    def spawn(command, **kwargs):
        if len(processes.calls) == 1:
            capture_response(agent, path, payload, content_type)
        return processes(command, **kwargs)

    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=spawn, max_recovery_attempts=1
    )
    events = agent.run("solve")

    assert len(processes.calls) == 3
    assert [e.content["attempt"] for e in events if e.type == "recovery"] == [1, 1]
    assert events[-1].content == "done"


def test_failures_after_last_success_still_exhaust_retry_allowance(tmp_path):
    processes = ProcessSequenceFactory(*[
        FakeProcess(stderr="HTTP 504", returncode=1) for _ in range(5)
    ])

    def spawn(command, **kwargs):
        if len(processes.calls) == 2:
            capture_response(agent, "/v1/chat/completions", {
                "choices": [{"finish_reason": "tool_calls"}],
            })
        return processes(command, **kwargs)

    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=spawn, max_recovery_attempts=2
    )
    events = []
    with pytest.raises(CLIProcessError, match="504"):
        events.extend(agent.stream("solve"))

    assert len(processes.calls) == 5
    assert [e.content["attempt"] for e in events if e.type == "recovery"] == [1, 2, 1, 2]
    assert events[-1].type == "error"


@pytest.mark.parametrize("payload", [
    {"choices": [{"delta": {"content": "partial output"}, "finish_reason": None}]},
    {"usage": {"prompt_tokens": 20, "completion_tokens": 5}},
    {"choices": [{"finish_reason": "error"}], "error": {"code": 504}},
    {"choices": [{"finish_reason": "stop"}], "error": {"code": 429}},
    {"choices": [{"finish_reason": "length"}]},
    {"type": "response.incomplete", "response": {"status": "incomplete"}},
    {"type": "response.failed", "response": {"error": {"code": "failed"}}},
    {"type": "response.completed", "response": {"status": "incomplete"}},
    {"type": "response.completed", "response": {"error": {"code": "failed"}}},
    {"object": "response"},
    {"type": "message_delta", "delta": {"stop_reason": "max_tokens"}},
    {"type": "message_stop"},
    {"candidates": [{"finishReason": "MAX_TOKENS"}]},
    {"choices": [{"finish_reason": {"unexpected": "value"}}]},
])
def test_partial_failed_and_limited_responses_do_not_reset_retries(tmp_path, payload):
    processes = ProcessSequenceFactory(
        FakeProcess(stderr="HTTP 504", returncode=1),
        FakeProcess(stderr="HTTP 504", returncode=1),
    )

    def spawn(command, **kwargs):
        if len(processes.calls) == 1:
            capture_response(agent, "/v1/chat/completions", payload)
        return processes(command, **kwargs)

    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=spawn, max_recovery_attempts=1
    )
    with pytest.raises(CLIProcessError, match="504"):
        agent.run("solve")
    assert len(processes.calls) == 2


def test_replayed_native_messages_and_usage_do_not_reset_retries(tmp_path):
    replay = "\n".join(json.dumps(event) for event in [
        {"type": "thread.started", "thread_id": "saved-session"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "old answer"}},
        {"type": "turn.completed", "usage": {"output_tokens": 100}},
    ]) + "\n"
    processes = ProcessSequenceFactory(*[
        FakeProcess(stdout=replay, stderr="HTTP 504", returncode=1) for _ in range(2)
    ])
    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=processes, max_recovery_attempts=1
    )
    with pytest.raises(CLIProcessError, match="504"):
        agent.run("solve")
    assert len(processes.calls) == 2
    assert "saved-session" in processes.calls[1][0]


def test_success_from_previous_run_does_not_reset_retries(tmp_path):
    processes = ProcessSequenceFactory(*[
        FakeProcess(stderr="HTTP 504", returncode=1) for _ in range(2)
    ])
    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=processes, max_recovery_attempts=1
    )
    capture_response(agent, "/v1/chat/completions", {
        "choices": [{"finish_reason": "stop"}],
    })
    with pytest.raises(CLIProcessError, match="504"):
        agent.run("new task")
    assert len(processes.calls) == 2


def test_success_does_not_enable_retries_when_disabled(tmp_path):
    processes = ProcessSequenceFactory(FakeProcess(stderr="HTTP 504", returncode=1))

    def spawn(command, **kwargs):
        capture_response(agent, "/v1/chat/completions", {
            "choices": [{"finish_reason": "stop"}],
        })
        return processes(command, **kwargs)

    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=spawn, max_recovery_attempts=0
    )
    with pytest.raises(CLIProcessError, match="504"):
        agent.run("solve")
    assert len(processes.calls) == 1
