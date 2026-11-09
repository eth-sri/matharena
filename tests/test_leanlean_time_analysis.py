import json

import pytest

from leanlean.time_analysis import partition, reasoning_intervals, response_intervals, standalone_measurement


@pytest.mark.parametrize(
    "command", ["python proof_length.py", "python3 /work/proof_length.py", " python ./proof_length.py "]
)
def test_standalone_measurement_is_recognized(command):
    assert standalone_measurement({"kind": "bash", "argv": ["bash", "-lc", command]})


@pytest.mark.parametrize(
    "command",
    ["python proof_length.py; lake build", "echo python proof_length.py", "python proof_length.py && sleep 10"],
)
def test_compound_commands_are_not_misreported_as_measurement(command):
    assert not standalone_measurement({"kind": "bash", "argv": ["bash", "-lc", command]})


def test_measurement_is_counted_once_inside_tool_span():
    result = partition(0, 10**9, [(0, 10**9, "other_tools"), (0, 10**9, "measure")])
    assert result["measure"] == 1
    assert sum(result.values()) == 1


def test_partition_clips_and_counts_parallel_reasoning_once():
    result = partition(
        0,
        10 * 10**9,
        [(0, 5 * 10**9, "build"), (2 * 10**9, 6 * 10**9, "reasoning"), (7 * 10**9, 12 * 10**9, "retry_wait")],
    )
    assert result["build"] == 2
    assert result["parallel"] == 3
    assert result["reasoning"] == 1
    assert result["retry_wait"] == 3
    assert result["unclassified"] == 1
    assert sum(result.values()) == 10


def test_nested_build_is_counted_inside_comparator():
    result = partition(0, 10**9, [(0, 10**9, "other_tools"), (0, 10**9, "comparator"), (0, 10**9, "build")])
    assert result["comparator"] == 1
    assert sum(result.values()) == 1


def test_empty_intervals_are_unclassified_and_reversed_intervals_fail():
    assert partition(0, 10**9, [])["unclassified"] == 1
    with pytest.raises(ValueError):
        partition(0, 10**9, [(2, 1, "build")])


def test_native_reasoning_is_deduplicated_and_mapped_to_monotonic_clock(tmp_path):
    folder = tmp_path / "native"
    folder.mkdir()
    record = {
        "payload": {
            "type": "item_completed",
            "thread_id": "t",
            "turn_id": "one",
            "item": {"type": "Reasoning", "id": "r"},
            "started_at_ms": 1000,
            "completed_at_ms": 2000,
        }
    }
    line = json.dumps(record) + "\n"
    (folder / "rollout-one.jsonl").write_text(line + line)
    events = [{"observed_wall_ns": 10**9, "observed_monotonic_ns": 5 * 10**9}]
    spans, audit = reasoning_intervals(tmp_path, events)
    assert spans == [(5 * 10**9, 6 * 10**9, "reasoning")]
    assert audit["valid_items"] == 1
    assert audit["invalid_items"] == 0


def write_records(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_claude_response_uses_live_signature_and_ignores_echoed_history(tmp_path):
    wrapper = tmp_path / "native/.harness_wrapper"
    write_records(wrapper / "model_requests.jsonl", [
        {"event": {"type": "model.request", "timestamp": "1970-01-01T00:00:01Z"}},
    ])
    write_records(wrapper / "traces/session.jsonl", [
        {"event": {"type": "reasoning_encrypted", "timestamp": "1970-01-01T00:00:02Z",
                   "metadata": {"raw": {"source": "model_request"}}}},
        {"event": {"type": "reasoning_encrypted", "timestamp": "1970-01-01T00:00:04Z",
                   "metadata": {"raw": {"source": "model_response"}}}},
    ])
    spans, audit = response_intervals(tmp_path, [{"observed_wall_ns": 10**9, "observed_monotonic_ns": 5*10**9}], "claude")
    assert spans == [(5*10**9, 8*10**9, "response")]
    assert audit["valid_items"] == 1


def test_gemini_planner_stops_when_tool_starts_and_deduplicates_overlaps(tmp_path):
    write_records(tmp_path / "native/logs/transcript_full.jsonl", [
        dict(step_index=1, type="PLANNER_RESPONSE", status="DONE", tool_calls=[{}], created_at="1970-01-01T00:00:01Z"),
        dict(step_index=2, type="GENERIC", created_at="1970-01-01T00:00:04Z"),
    ])
    events = [{"observed_wall_ns": 4*10**9, "observed_monotonic_ns": 4*10**9,
               "event": {"raw": {"step_update": dict(step_type="agent_response", state="DONE", duration_seconds=3)}}}]
    spans, _ = response_intervals(tmp_path, events, "gravity")
    assert partition(0, 5*10**9, spans + [(4*10**9, 5*10**9, "build")])["response"] == 3


def test_deepcode_uses_creation_time_not_post_tool_message_update(tmp_path):
    write_records(tmp_path / "native/.harness_wrapper/model_requests.jsonl", [
        {"event": {"type": "model.request", "timestamp": "1970-01-01T00:00:01Z"}},
    ])
    write_records(tmp_path / "native/.harness-home/.deepcode/session.jsonl", [
        dict(role="assistant", createTime="1970-01-01T00:00:04Z", updateTime="1970-01-01T01:00:00Z"),
    ])
    events = [{"observed_wall_ns": 10**9, "observed_monotonic_ns": 10**9}]
    spans, _ = response_intervals(tmp_path, events, "deepcode")
    assert spans == [(10**9, 4*10**9, "response")]
    totals = partition(0, 5*10**9, spans + [(2*10**9, 3*10**9, "retry_wait"), (3*10**9, 4*10**9, "build")])
    assert totals["response"] == totals["retry_wait"] == totals["build"] == 1
    assert sum(totals.values()) == 5


def test_kimi_response_ends_at_its_usage_record(tmp_path):
    wrapper = tmp_path / "native/.harness_wrapper"
    write_records(wrapper / "model_requests.jsonl", [
        {"event": {"type": "model.request", "timestamp": "1970-01-01T00:00:01Z"}},
    ])
    write_records(wrapper / "traces/session.jsonl", [
        {"event": {"type": "message", "timestamp": "1970-01-01T00:00:02Z",
                   "metadata": {"raw": {"source": "model_request"}}}},
        {"event": {"type": "usage", "timestamp": "1970-01-01T00:00:04Z",
                   "metadata": {"raw": {"source": "model_response"}}}},
    ])
    spans, _ = response_intervals(tmp_path, [{"observed_wall_ns": 0, "observed_monotonic_ns": 0}], "kimi")
    assert spans == [(10**9, 4*10**9, "response")]


def test_dsh_response_ends_at_first_step_output_not_echoed_history(tmp_path):
    wrapper = tmp_path / "native/.harness_wrapper"
    write_records(wrapper / "model_requests.jsonl", [
        {"event": {"type": "model.request", "timestamp": "1970-01-01T00:00:01Z"}},
    ])
    write_records(wrapper / "traces/session.jsonl", [
        {"event": {"type": "reasoning", "timestamp": "1970-01-01T00:00:02Z",
                   "metadata": {"raw": {"source": "model_request"}}}},
        {"event": {"type": "reasoning", "timestamp": "1970-01-01T00:00:03Z", "metadata": {"raw": {}}}},
        {"event": {"type": "tool_call", "timestamp": "1970-01-01T00:00:03.5Z", "metadata": {"raw": {}}}},
    ])
    spans, _ = response_intervals(tmp_path, [{"observed_wall_ns": 0, "observed_monotonic_ns": 0}], "dsh")
    assert spans == [(10**9, 3*10**9, "response")]
