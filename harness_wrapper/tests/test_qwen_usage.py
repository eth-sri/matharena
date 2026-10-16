"""Qwen's terminal usage includes the whole native session, including retries."""

import json

import pytest
from test_agents import FakeProcess, ProcessSequenceFactory, make_agent

from harness_wrapper.harnesses.qwen_code import QwenCodeAgent
from harness_wrapper.traces import TokenUsage


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


def qwen_result(usage, session_id="q1", *, failed=False, result_session_id=True):
    terminal = {
        "type": "result",
        "subtype": "error_during_execution" if failed else "success",
        "is_error": failed,
        "result": "loop guard stopped the turn" if failed else "done",
        "usage": usage,
    }
    if result_session_id:
        terminal["session_id"] = session_id
    return FakeProcess(stdout="\n".join((
        json.dumps({"type": "system", "subtype": "session_start", "session_id": session_id}),
        json.dumps(terminal),
        "",
    )))


@pytest.mark.parametrize("result_session_id", [True, False])
def test_qwen_p22_usage_counts_each_token_once_across_retries_and_resume(
    tmp_path, result_session_id,
):
    # Actual cumulative snapshots from P22's two loop-guard stops, completion,
    # and final-answer continuation. Previously these added up to $269.965156.
    snapshots = [
        {"input_tokens": 129461426, "output_tokens": 534126,
         "cache_read_input_tokens": 128605568, "total_tokens": 129995552},
        {"input_tokens": 211767573, "output_tokens": 579225,
         "cache_read_input_tokens": 210056448, "total_tokens": 212346798},
        {"input_tokens": 310854950, "output_tokens": 615078,
         "cache_read_input_tokens": 308233728, "total_tokens": 311470028},
        {"input_tokens": 311607496, "output_tokens": 615906,
         "cache_read_input_tokens": 308237824, "total_tokens": 312223402},
    ]
    factory = ProcessSequenceFactory(*(
        qwen_result(usage, failed=i < 2, result_session_id=result_session_id)
        for i, usage in enumerate(snapshots)
    ))
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)

    events = agent.run("solve")
    assert len([e for e in events if e.type == "recovery"]) == 2
    assert agent.get_tokens() == TokenUsage(
        input_tokens=310854950, output_tokens=615078, cache_read_tokens=308233728,
    )
    events.extend(agent.resume("Finish", session_id="q1"))
    assert agent.get_tokens() == TokenUsage(
        input_tokens=311607496, output_tokens=615906, cache_read_tokens=308237824,
    )
    usage_events = [e for e in events if e.type == "usage"]
    assert [e.raw["usage"] for e in usage_events] == snapshots
    assert sum(e.content["input_tokens"] for e in usage_events) == 311607496
    assert sum(e.content["output_tokens"] for e in usage_events) == 615906
    assert sum(e.content["cache_read_tokens"] for e in usage_events) == 308237824
    assert usage_events[-1].content["output_tokens"] == 828


def test_qwen_duplicate_and_lower_snapshots_do_not_spend_tokens_again(tmp_path):
    snapshots = [
        {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 6,
         "cache_creation_input_tokens": 2},
        {"input_tokens": 10, "output_tokens": 20, "cache_read_input_tokens": 6,
         "cache_creation_input_tokens": 2},
        {"input_tokens": 3, "output_tokens": 0},
        {"input_tokens": 12, "output_tokens": 25, "cache_read_input_tokens": 7,
         "cache_creation_input_tokens": 3},
    ]
    factory = ProcessSequenceFactory(*(qwen_result(u) for u in snapshots))
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)
    agent.run("solve")
    expected = TokenUsage(
        input_tokens=10, output_tokens=20, cache_read_tokens=6, cache_write_tokens=2,
    )
    for _ in range(2):
        events = agent.resume("Continue")
        assert agent.get_tokens() == expected
        assert event_usage(events) == TokenUsage()
    events = agent.resume("Continue")
    assert agent.get_tokens() == TokenUsage(
        input_tokens=12, output_tokens=25, cache_read_tokens=7, cache_write_tokens=3,
    )
    assert event_usage(events) == TokenUsage(
        input_tokens=2, output_tokens=5, cache_read_tokens=1, cache_write_tokens=1,
    )


def event_usage(events):
    usage = next(e.content for e in events if e.type == "usage")
    return TokenUsage(**{key: usage[key] for key in (
        "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
    )})


@pytest.mark.parametrize("new_session", [False, True])
def test_qwen_usage_resets_for_fresh_run_or_changed_native_session(tmp_path, new_session):
    factory = ProcessSequenceFactory(
        qwen_result({"input_tokens": 100, "output_tokens": 200}),
        qwen_result({"input_tokens": 5, "output_tokens": 7}, "q2" if new_session else "q1"),
    )
    agent = make_agent(QwenCodeAgent, tmp_path, process_factory=factory)
    agent.run("first")
    if new_session:
        agent.resume("Continue", session_id="q1")
    else:
        agent.run("second")
    assert agent.get_tokens() == TokenUsage(input_tokens=5, output_tokens=7)
