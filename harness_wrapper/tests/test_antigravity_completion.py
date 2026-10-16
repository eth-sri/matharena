"""Offline regressions for false-success exits and missing stream tails."""

import json
import sqlite3
from pathlib import PurePosixPath
from types import SimpleNamespace

import pytest
from test_agents import FakeProcess, ProcessSequenceFactory, make_agent

from harness_wrapper.tools import CLIProcessError
from harness_wrapper.harnesses.antigravity_cli import AntigravityCLIAgent
from harness_wrapper.harnesses.antigravity_cli.adapter import IncompleteAntigravityResponse
from harness_wrapper.harnesses.antigravity_cli.native_session import read_native_response


@pytest.fixture(autouse=True)
def no_retry_delay(monkeypatch):
    monkeypatch.setattr("harness_wrapper.agent.time.sleep", lambda _: None)


def varint(value):
    data = bytearray()
    while value > 127:
        data.append((value & 127) | 128)
        value >>= 7
    data.append(value)
    return bytes(data)


def field(number, value):
    return varint((number << 3) | 2) + varint(len(value)) + value


def native_database(
    root, *, step=4, step_type=15, status=3, answer="Recovered answer", payload=None
):
    directory = root / ".harness-home/.gemini/antigravity-cli/conversations"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "session-1.db"
    with sqlite3.connect(path) as con:
        con.execute(
            "CREATE TABLE IF NOT EXISTS steps "
            "(idx INTEGER PRIMARY KEY, step_type, status, step_payload)"
        )
        con.execute(
            "INSERT INTO steps VALUES (?, ?, ?, ?)",
            (
                step,
                step_type,
                status,
                payload
                if payload is not None
                else field(20, field(1, answer.encode()) + field(2, b"Private reasoning")),
            ),
        )
    return path


def line(event, **payload):
    return json.dumps({"event": event, event: {"conversation_id": "session-1", **payload}}) + "\n"


def success(answer=""):
    return line("init") + line("result", status="SUCCESS", response=answer)


def test_auto_approval_is_container_only(tmp_path):
    agent = make_agent(AntigravityCLIAgent, tmp_path, minimal_context=True)
    assert "--dangerously-skip-permissions" not in agent.build_command("solve")
    agent.env = SimpleNamespace(enabled=False, container_root=PurePosixPath("/work"))
    assert "--dangerously-skip-permissions" not in agent.build_command("solve")
    agent.env.enabled = True
    for resumed in (False, True):
        assert "--dangerously-skip-permissions" in agent.build_command("solve", resume=resumed)
    agent._runtime_paths()
    settings = json.loads(
        (tmp_path / ".harness-home/.gemini/antigravity-cli/settings.json").read_text()
    )
    assert settings["allowNonWorkspaceAccess"] is True


@pytest.mark.parametrize("denied", [False, True])
def test_zero_exit_without_answer_recovers_same_session(tmp_path, denied):
    broken = line("init") + line(
        "result",
        status="SUCCESS",
        response="",
        **({"denied_actions": [{"action": "read_file"}]} if denied else {}),
    )
    factory = ProcessSequenceFactory(
        FakeProcess(stdout=broken), FakeProcess(stdout=success("Complete answer"))
    )
    agent = make_agent(AntigravityCLIAgent, tmp_path, process_factory=factory)
    events = agent.run("Original problem")
    assert events[-1].content == "Complete answer"
    recovery = [e for e in events if e.type == "recovery"]
    assert len(recovery) == 1 and recovery[0].content["reason"] == "incomplete_response"
    command = factory.calls[1][0]
    assert command[command.index("--conversation") + 1] == "session-1"
    assert "Original problem" not in command


@pytest.mark.parametrize("stdout", [success(), line("init"), success(" ")])
def test_incomplete_runs_exhaust_budget_and_raise(tmp_path, stdout):
    factory = ProcessSequenceFactory(*[FakeProcess(stdout=stdout) for _ in range(3)])
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=2
    )
    with pytest.raises(IncompleteAntigravityResponse):
        agent.run("Original problem")
    assert len(factory.calls) == 3
    assert agent.trace.events[-1][0] == "error"


def test_native_answer_recovers_missing_stream_without_model_retry(tmp_path):
    native_database(tmp_path)
    factory = ProcessSequenceFactory(FakeProcess(stdout=success()))
    agent = make_agent(AntigravityCLIAgent, tmp_path, process_factory=factory)
    events = agent.run("Original problem")
    assert events[-1].content == "Recovered answer"
    assert events[-1].raw["native_response"]["step_index"] == 4
    assert len(factory.calls) == 1
    assert not any(e.type == "recovery" for e in events)


def test_resume_cannot_reuse_previous_native_answer(tmp_path):
    native_database(tmp_path)
    factory = ProcessSequenceFactory(FakeProcess(stdout=success()))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    with pytest.raises(IncompleteAntigravityResponse):
        agent.resume("Next task", session_id="session-1")


def test_trailing_tool_cannot_be_graded_as_earlier_commentary(tmp_path):
    stream = line("init") + line(
        "step_update", step_type="agent_response", text_delta="I will calculate."
    )
    stream += line(
        "step_update",
        step_type="tool",
        state="DONE",
        tool_info={
            "name": "run_command",
            "parameters": {},
            "output": "Not an answer",
        },
    ) + line("result", status="SUCCESS", response="")
    agent = make_agent(
        AntigravityCLIAgent,
        tmp_path,
        process_factory=ProcessSequenceFactory(FakeProcess(stdout=stream)),
        max_recovery_attempts=0,
    )
    with pytest.raises(IncompleteAntigravityResponse):
        agent.run("Original problem")


def test_final_result_preserves_full_answer_not_only_last_delta(tmp_path):
    stream = line("init") + line(
        "step_update", step_type="agent_response", text_delta="Last chunk."
    )
    stream += line("result", status="SUCCESS", response="First chunk. Last chunk.")
    agent = make_agent(
        AntigravityCLIAgent,
        tmp_path,
        process_factory=ProcessSequenceFactory(FakeProcess(stdout=stream)),
    )
    assert agent.run("Original problem")[-1].content == "First chunk. Last chunk."


def test_native_followup_does_not_replace_complete_cli_answer(tmp_path):
    main = "The statement is false. Here is a counterexample."
    followup = "The background computation confirms the counterexample."
    full_answer = f"{main}\n\n{followup}\n"
    native_database(tmp_path, step=44, answer=main)
    native_database(tmp_path, step=50, answer=followup)
    stream = line("init") + line(
        "step_update", step_index=44, step_type="agent_response", text_delta=main
    )
    stream += line(
        "step_update", step_index=50, step_type="agent_response", text_delta=followup
    ) + line("result", status="SUCCESS", response=full_answer)
    factory = ProcessSequenceFactory(FakeProcess(stdout=stream))
    agent = make_agent(AntigravityCLIAgent, tmp_path, process_factory=factory)
    events = agent.run("Prove the statement")
    assert events[-1].content == full_answer
    assert "native_response" not in events[-1].raw
    assert len(factory.calls) == 1


def test_native_answer_still_restores_truncated_cli_result(tmp_path):
    native_database(tmp_path, answer="First chunk. Missing final chunk.")
    factory = ProcessSequenceFactory(FakeProcess(stdout=success("First chunk.")))
    agent = make_agent(AntigravityCLIAgent, tmp_path, process_factory=factory)
    result = agent.run("Prove the statement")[-1]
    assert result.content == "First chunk. Missing final chunk."
    assert result.raw["native_response"]["step_index"] == 4
    assert len(factory.calls) == 1


@pytest.mark.parametrize(("step_type", "status"), [(132, 3), (15, 7), (14, 3)])
def test_native_reader_rejects_incomplete_terminal_steps(tmp_path, step_type, status):
    native_database(tmp_path, step_type=step_type, status=status)
    assert read_native_response(tmp_path, "session-1") is None


def test_native_reader_does_not_search_reasoning_for_answers(tmp_path):
    native_database(tmp_path, payload=field(20, field(2, b"Reasoning that looks like an answer")))
    assert read_native_response(tmp_path, "session-1") is None


@pytest.mark.parametrize("payload", [b"\xa2\x01\x64bad", b"\x00", b"\x80" * 11])
def test_native_reader_rejects_malformed_protobuf(tmp_path, payload):
    native_database(tmp_path, payload=payload)
    with pytest.raises(ValueError):
        read_native_response(tmp_path, "session-1")


def test_native_reader_rejects_unsafe_session_names(tmp_path):
    with pytest.raises(ValueError):
        read_native_response(tmp_path, "../../elsewhere")


def test_native_read_is_non_mutating_and_respects_step_boundary(tmp_path):
    path = native_database(tmp_path)
    before = path.read_bytes()
    assert read_native_response(tmp_path, "session-1").text == "Recovered answer"
    assert read_native_response(tmp_path, "session-1", after_step=4) is None
    assert path.read_bytes() == before


@pytest.mark.parametrize('returncode', [0, 1])
def test_fresh_native_answer_recovers_stale_cli_error_without_resume(tmp_path, returncode):
    native_database(tmp_path, step=4, answer='Previous answer')
    stream = line('init') + line(
        'result', status='ERROR', response='Complete answer',
        error='The stream was interrupted. Please continue the task you were working on.',
    )
    factory = ProcessSequenceFactory(FakeProcess(stdout=stream, returncode=returncode))

    def start(*args, **kwargs):
        native_database(tmp_path, step=5, answer='Complete answer')
        return factory(*args, **kwargs)

    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=start, max_recovery_attempts=0
    )
    events = agent.resume('Continue', session_id='session-1')
    assert events[-1].content == 'Complete answer'
    assert events[-1].raw['native_response']['step_index'] == 5
    assert events[-1].raw['cli_error']['returncode'] == (returncode or 1)
    assert len(factory.calls) == 1
    assert not any(event.type == 'recovery' for event in events)


def test_nonzero_exit_with_missing_result_recovers_fresh_native_answer(tmp_path):
    native_database(tmp_path)
    factory = ProcessSequenceFactory(FakeProcess(stdout=line('init'), returncode=1))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    assert agent.run('Solve')[-1].content == 'Recovered answer'
    assert len(factory.calls) == 1


@pytest.mark.parametrize('returncode', [0, 1])
def test_stale_error_requires_fresh_native_answer(tmp_path, returncode):
    native_database(tmp_path, answer='Previous answer')
    stream = line('init') + line(
        'result', status='ERROR', response='Complete-looking answer', error='Old error'
    )
    factory = ProcessSequenceFactory(FakeProcess(stdout=stream, returncode=returncode))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    with pytest.raises(CLIProcessError):
        agent.resume('New task', session_id='session-1')
    assert len(factory.calls) == 1


@pytest.mark.parametrize('status', [1, 2, 8, 9, 11])
def test_stale_error_cannot_accept_answer_with_unfinished_tool(tmp_path, status):
    native_database(tmp_path, step=3, step_type=132, status=status)
    native_database(tmp_path, step=4, answer='I will calculate.')
    factory = ProcessSequenceFactory(FakeProcess(
        stdout=line('init') + line('result', status='ERROR', response='I will calculate.', error='Old error'),
        returncode=1,
    ))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    with pytest.raises(CLIProcessError):
        agent.run('Solve')
    assert len(factory.calls) == 1


def test_stale_error_cannot_bypass_permission_denial(tmp_path):
    native_database(tmp_path)
    factory = ProcessSequenceFactory(FakeProcess(
        stdout=line('init') + line(
            'result', status='ERROR', response='Recovered answer', error='Permission denied',
            denied_actions=[{'action': 'read_file'}],
        ), returncode=1,
    ))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    with pytest.raises(IncompleteAntigravityResponse, match='permission denial'):
        agent.run('Solve')


def test_stale_error_preserves_complete_answer_with_native_followup(tmp_path):
    main = 'Full proof.'
    followup = 'The computation confirms it.'
    native_database(tmp_path, step=4, answer=followup)
    factory = ProcessSequenceFactory(FakeProcess(
        stdout=line('init') + line('result', status='ERROR', response=main+'\n'+followup, error='Old error'),
        returncode=1,
    ))
    agent = make_agent(
        AntigravityCLIAgent, tmp_path, process_factory=factory, max_recovery_attempts=0
    )
    assert agent.run('Solve')[-1].content == main+'\n'+followup
