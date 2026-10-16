"""Grok's native streaming events must produce readable, lossless conversations."""

import pytest

from harness_wrapper import AgentEvent
from matharena.solvers.harness_solver import _events_to_messages
from test_grok_harness import make_agent
from test_harness_completion import make_solver


def native_text(kind, text):
    # Splitting at individual characters exercises arbitrary token boundaries,
    # including spaces, newlines, and Unicode.
    return [{"type": kind, "data": char} for char in text]


def tool_events(call_id="call-1"):
    return [
        {"type": "tool_call", "toolCallId": call_id,
         "toolName": "run_terminal_command", "rawInput": {"command": "echo 2"}},
        {"type": "tool_call_update", "toolCallId": call_id,
         "status": "in_progress", "content": []},
        {"type": "tool_call_update", "toolCallId": call_id,
         "status": "completed", "rawOutput": {"output_for_prompt": "2", "exit_code": 0}},
    ]


def normalize(tmp_path, native):
    adapter = make_agent(tmp_path)
    return [event for raw in native for event in adapter.normalize_event(raw)]


@pytest.mark.parametrize("harness", ["grok", "grok-build", "grok-cli", "grok_build"])
def test_native_grok_chunks_are_grouped_in_saved_conversation(tmp_path, monkeypatch, harness):
    events = normalize(tmp_path, [
        *native_text("thought", "Think α.\nCheck the calculation."),
        {"type": "usage", "usage": {"output_tokens": 20}},
        *native_text("thought", " Then run it."),
        *native_text("text", "Checking now."),
        *tool_events(),
        *native_text("thought", "The result is 2."),
        *native_text("text", "Final answer: 2."),
        {"type": "end", "stopReason": "end_turn"},
    ])
    solver = make_solver(tmp_path, monkeypatch, events, harness=harness)
    response = solver._solve_one(0, "Problem", None, 1, 0)

    messages = response.conversation[1:]
    assert [(m["role"], m.get("type")) for m in messages] == [
        ("assistant", "cot"), ("assistant", "response"),
        ("assistant", "tool_call"), ("tool_response", None),
        ("assistant", "cot"), ("assistant", "response"),
    ]
    assert messages[0]["content"] == "Think α.\nCheck the calculation. Then run it."
    assert messages[1]["content"] == "Checking now."
    assert messages[-2]["content"] == "The result is 2."
    # The terminal result repeats the streamed answer; it is stored only once.
    assert messages[-1]["content"] == "Final answer: 2."
    assert response.history[0]["messages"] == response.conversation
    assert response.history[0]["events"] == [e.to_dict() for e in events]
    assert "empty_completion" not in response.history[0]


def test_background_tool_call_preserves_prior_proof_and_gap_admission(tmp_path, monkeypatch):
    # Regression for False August P22: a proof, a background-task continuation,
    # further tools, then a genuinely empty terminal result.
    proof = "A complete proof attempt.\nFinal answer: proved"
    correction = "The previous write-up left a gap. I am rechecking it."
    events = normalize(tmp_path, [
        *native_text("text", proof),
        {"type": "usage", "usage": {}},
        *native_text("thought", "A background command completed."),
        *native_text("text", correction),
        *tool_events(),
        *native_text("thought", "Still checking the gap."),
        {"type": "end", "stopReason": "end_turn"},
    ])
    solver = make_solver(tmp_path, monkeypatch, events, harness="grok")
    response = solver._solve_one(0, "Problem", None, 22, 0)

    visible = [m["content"] for m in response.conversation if m.get("type") == "response"]
    assert visible == [proof, correction, ""]
    assert response.history[0]["empty_completion"] is True
    # Preserving the transcript must not silently submit a retracted proof as
    # the final answer or change the existing empty-completion retry policy.
    assert solver.last_chance(response) is response


def test_grok_last_chance_also_groups_streams(tmp_path, monkeypatch):
    initial = normalize(tmp_path, [
        *native_text("text", "Initial answer."),
        {"type": "end", "stopReason": "end_turn"},
    ])
    solver = make_solver(tmp_path, monkeypatch, initial, harness="grok")
    response = solver._solve_one(0, "Problem", None, 1, 0)
    resumed = normalize(tmp_path, [
        *native_text("thought", "Rechecking."),
        *native_text("text", "Final answer."),
        {"type": "end", "stopReason": "end_turn"},
    ])
    agent = solver._response_agents[id(response)]
    agent.resume = lambda *args, **kwargs: resumed

    response = solver.last_chance(response)

    assert response.conversation[-2:] == [
        {"role": "assistant", "type": "cot", "content": "Rechecking."},
        {"role": "assistant", "type": "response", "content": "Final answer."},
    ]
    assert len(response.conversation) == 5  # Includes the user reprompt.
    assert response.history[-1]["messages"][-2:] == response.conversation[-2:]


@pytest.mark.parametrize("boundary", ["tool_call", "tool_result", "message", "recovery", "session", "error"])
def test_grok_does_not_merge_across_conversation_or_retry_boundaries(boundary):
    events = [
        AgentEvent(type="reasoning", content="Before "),
        AgentEvent(type="reasoning", content="boundary."),
        AgentEvent(type=boundary, role="user" if boundary == "message" else None, content="boundary"),
        AgentEvent(type="reasoning", content="After "),
        AgentEvent(type="reasoning", content="boundary."),
    ]
    messages = _events_to_messages(events, harness="grok")
    assert [m["content"] for m in messages if m.get("type") == "cot"] == [
        "Before boundary.", "After boundary.",
    ]


def test_grok_flushes_partial_stream_without_a_terminal_event():
    events = [AgentEvent(type="reasoning", content=part) for part in ["", "Partial", " reasoning"]]
    assert _events_to_messages(events, harness="grok") == [
        {"role": "assistant", "type": "cot", "content": "Partial reasoning"},
    ]


@pytest.mark.parametrize("harness", [None, "claude", "codex", "kimi", "gravity", "muse"])
def test_other_harnesses_do_not_duplicate_stream_and_snapshot(harness):
    events = [
        AgentEvent(type="text", content="Answer"),
        AgentEvent(type="message", role="assistant", content="Answer"),
        AgentEvent(type="result", content="Answer"),
    ]
    assert _events_to_messages(events, harness=harness) == [
        {"role": "assistant", "type": "response", "content": "Answer"},
    ]
