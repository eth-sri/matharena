"""Failure recovery regression tests: fake CLIs, no model requests."""

import json

import pytest
from test_agents import FakeProcess, ProcessSequenceFactory, make_agent

from harness_wrapper.harnesses.codex_cli import CodexCLIAgent
from harness_wrapper.harnesses.qwen_code import QwenCodeAgent
from harness_wrapper.tools import CLIProcessError


@pytest.fixture(autouse=True)
def no_delays(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


def lines(*events):
    return "".join(json.dumps(event) + "\n" for event in events)


@pytest.mark.parametrize("detail", [
    "HTTP 401 invalid API key", "HTTP 403 forbidden", "invalid configuration",
    "HTTP 400 invalid_request_error", "HTTP 429 rate limit exceeded",
    "connection reset by peer", "unrecognized failure",
])
def test_all_failures_use_default_retry_budget(tmp_path, detail):
    factory = ProcessSequenceFactory(*[
        FakeProcess(stderr=detail, returncode=1) for _ in range(11)
    ])
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    events = []
    with pytest.raises(CLIProcessError, match=detail):
        events.extend(agent.stream("solve"))
    assert len(factory.calls) == 11
    assert [e.content["attempt"] for e in events if e.type == "recovery"] == list(range(1, 11))
    assert events[-1].type == "error"


@pytest.mark.parametrize("exception", [OSError, ValueError, RuntimeError])
def test_pre_spawn_errors_retry_and_preserve_exception(tmp_path, exception):
    calls = []

    def spawn(command, **kwargs):
        calls.append(command)
        raise exception("startup failure")

    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=spawn)
    with pytest.raises(exception, match="startup failure"):
        agent.run("solve")
    assert len(calls) == 11
    assert all(command == calls[0] for command in calls)


def test_failed_recovery_action_does_not_disable_retry(tmp_path, monkeypatch):
    factory = ProcessSequenceFactory(
        FakeProcess(stderr="authentication failed", returncode=1),
        FakeProcess(stdout=lines({"type": "item.completed", "item": {
            "type": "agent_message", "text": "done",
        }})),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)

    def failed_relogin(*args, **kwargs):
        raise RuntimeError("login failed")

    monkeypatch.setattr(agent, "_recover_cli_failure", failed_relogin)
    assert agent.run("solve")[-1].content == "done"
    assert len(factory.calls) == 2


@pytest.mark.parametrize("cancel", ["interrupt", "terminate", "budget"])
def test_cancellation_during_recovery_prevents_another_invocation(tmp_path, cancel):
    stopped = False
    factory = ProcessSequenceFactory(FakeProcess(stderr="failed", returncode=1))
    agent = make_agent(
        CodexCLIAgent, tmp_path, process_factory=factory, should_stop=lambda: stopped,
    )
    for event in agent.stream("solve"):
        if event.type == "recovery":
            if cancel == "budget":
                stopped = True
            else:
                getattr(agent, cancel)()
    assert len(factory.calls) == 1


def test_keyboard_interrupt_is_not_retried(tmp_path):
    calls = []

    def spawn(command, **kwargs):
        calls.append(command)
        raise KeyboardInterrupt

    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=spawn)
    with pytest.raises(KeyboardInterrupt):
        agent.run("solve")
    assert len(calls) == 1


@pytest.mark.parametrize("error", [
    {"is_error": True, "error": {"message": "unknown provider error"}},
    {"subtype": "success", "is_error": False, "result": "[API Error: Network connection lost.]"},
    {"subtype": "error_during_execution", "result": "request failed"},
])
def test_qwen_failure_with_success_exit_resumes_and_retains_usage(tmp_path, error):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=lines(
            {"type": "system", "subtype": "session_start", "session_id": "q1"},
            {"type": "assistant", "message": {"content": [{
                "type": "thinking", "thinking": "partial reasoning",
            }]}},
            {"type": "result", "usage": {"output_tokens": 99}, **error},
        )),
        FakeProcess(stdout=lines({
            "type": "result", "result": "done", "usage": {"output_tokens": 110},
        })),
    )
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)
    events = agent.run("solve")
    assert [e.content for e in events if e.type == "result"] == ["done"]
    assert len(factory.calls) == 2
    assert "q1" in factory.calls[1][0]
    assert "solve" not in factory.calls[1][0]
    assert agent.get_tokens().output_tokens == 110
    assert any(e.type == "reasoning" for e in events)


@pytest.mark.parametrize("tail", ["", '{"type":"result","result":"truncated'])
def test_qwen_missing_terminal_result_retries(tmp_path, tail):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=lines({
            "type": "system", "subtype": "session_start", "session_id": "q1",
        }) + tail),
        FakeProcess(stdout=lines({"type": "result", "result": "done"})),
    )
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)
    assert agent.run("solve")[-1].content == "done"
    assert len(factory.calls) == 2
    assert "q1" in factory.calls[1][0]


def test_qwen_tool_failure_and_explicit_empty_completion_do_not_retry(tmp_path):
    factory = ProcessSequenceFactory(FakeProcess(stdout=lines(
        {"type": "user", "message": {"content": [{
            "type": "tool_result", "is_error": True, "tool_use_id": "t1",
            "content": "command failed",
        }]}},
        {"type": "result", "result": "", "is_error": False},
    )))
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)
    events = agent.run("solve")
    assert len(factory.calls) == 1
    assert [e.content for e in events if e.type == "tool_result"] == ["command failed"]
    assert events[-1].type == "result" and events[-1].content == ""


def test_native_cli_recovery_with_successful_answer_does_not_repeat_task(tmp_path):
    factory = ProcessSequenceFactory(FakeProcess(stdout=lines(
        {"type": "error", "message": "temporary failure"},
        {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
        {"type": "turn.completed", "usage": {"output_tokens": 10}},
    )))
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    events = agent.run("solve")
    assert len(factory.calls) == 1
    assert not any(e.type == "recovery" for e in events)
    assert [e.content for e in events if e.type == "result"] == ["done"]


def test_runtime_authentication_errors_are_inside_retry_budget(tmp_path, monkeypatch):
    factory = ProcessSequenceFactory()
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    preparations = []

    def prepare():
        preparations.append(True)
        raise ValueError("invalid credentials")

    monkeypatch.setattr(agent, "_prepare_runtime_model", prepare)
    with pytest.raises(ValueError, match="invalid credentials"):
        agent.run("solve")
    assert len(preparations) == 11
    assert not factory.calls


def test_new_task_failure_does_not_resume_previous_task(tmp_path):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=lines({"type": "thread.started", "thread_id": "old"})),
        FakeProcess(stderr="startup failure", returncode=1),
        FakeProcess(stdout=lines({"type": "thread.started", "thread_id": "new"})),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    agent.run("old problem")
    agent.run("new problem")
    assert agent.session_id == "new"
    assert factory.calls[1][0] == factory.calls[2][0]
    assert factory.stdin_inputs[1:] == [b"new problem", b"new problem"]


def test_unknown_error_with_zero_exit_retries_after_capturing_usage(tmp_path):
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=lines(
            {"type": "thread.started", "thread_id": "c1"},
            {"type": "error", "message": "previously unseen failure"},
            {"type": "turn.completed", "usage": {"output_tokens": 100}},
        )),
        FakeProcess(stdout=lines(
            {"type": "item.completed", "item": {"type": "agent_message", "text": "done"}},
            {"type": "turn.completed", "usage": {"output_tokens": 10}},
        )),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    events = agent.run("solve")
    assert len(factory.calls) == 2
    assert "c1" in factory.calls[1][0]
    assert agent.get_tokens().output_tokens == 110
    assert [e.content for e in events if e.type == "result"][-1] == "done"


@pytest.mark.parametrize("recovery", [
    "error_retry", "authentication_relogin", "rate_limit_wait",
    "authentication_fallback", "rate_limit_fallback", "process_interrupted",
    "output_limit_resume", "token_limit_resume", "incomplete_response",
])
def test_every_recovery_action_waits_at_least_sixty_seconds(tmp_path, monkeypatch, recovery):
    factory = ProcessSequenceFactory(
        FakeProcess(stderr="failure", returncode=1),
        FakeProcess(stdout=lines({"type": "item.completed", "item": {
            "type": "agent_message", "text": "done",
        }})),
    )
    agent = make_agent(CodexCLIAgent, tmp_path, process_factory=factory)
    delays = []
    monkeypatch.setattr(agent, "_recover_cli_failure", lambda *args, **kwargs: recovery)
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", delays.append)
    events = agent.run("solve")
    assert delays == [60]
    assert [e.content["retry_delay_seconds"] for e in events if e.type == "recovery"] == [60]
    assert len(factory.calls) == 2


@pytest.mark.parametrize("source", ["assistant", "synthetic_error", "system", "proxy"])
@pytest.mark.parametrize("returncode", [0, 1])
def test_claude_reasoning_extraction_is_terminal_and_keeps_usage(tmp_path, monkeypatch, source, returncode):
    from harness_wrapper.harnesses.claude_code import ClaudeCodeAgent

    details = {"type": "refusal", "category": "reasoning_extraction"}
    refusal = {
        "type": "assistant", "isApiErrorMessage": source == "synthetic_error",
        "message": {"role": "assistant", "stop_reason": "refusal", "stop_details": details,
                    "content": [{"type": "text", "text": "API refusal"}]},
    }
    if source == "system":
        refusal = {"type": "system", "subtype": "model_refusal_no_fallback",
                   "apiRefusalCategory": "reasoning_extraction"}
    if source == "proxy":
        refusal = {"type": "assistant", "isApiErrorMessage": True,
                   "message": {"content": [{"type": "text", "text": "API refusal"}]}}
    process = FakeProcess(stdout=lines(
        {"type": "system", "subtype": "init", "session_id": "refused-session"},
        refusal,
        {"type": "result", "is_error": True, "stop_reason": "refusal", "result": "API refusal",
         "usage": {"input_tokens": 100, "output_tokens": 128025}},
    ), returncode=returncode)
    factory = ProcessSequenceFactory(process)
    agent = make_agent(ClaudeCodeAgent, tmp_path, process_factory=factory)
    if source == "proxy":
        original = agent._stream_with_capture
        def stream(*args, **kwargs):
            agent._capture_model_response("/v1/messages", {
                "type": "message_delta", "delta": {"stop_reason": "refusal", "stop_details": details},
            })
            yield from original(*args, **kwargs)
        monkeypatch.setattr(agent, "_stream_with_capture", stream)
    events = agent.run("Solve the problem")
    assert len(factory.calls) == 1
    assert not any(e.type == "recovery" for e in events)
    assert events[-1].type == "result"
    assert events[-1].raw["refusal_category"] == "reasoning_extraction"
    assert agent.get_tokens().output_tokens == 128025
    assert agent._process is None
    assert not process.terminated  # Usage drained; CLI reaped normally.


def test_refusal_detector_ignores_answer_text_and_other_categories():
    from harness_wrapper.harnesses.claude_code.adapter import _is_reasoning_extraction_refusal

    for payload in [
        {"type": "assistant", "message": {"content": "reasoning_extraction"}},
        {"type": "user", "content": '{"stop_details":{"category":"reasoning_extraction"}}'},
        {"stop_reason": "max_tokens"},
        {"stop_reason": "refusal", "stop_details": {"category": "other"}},
    ]:
        assert not _is_reasoning_extraction_refusal(payload)
