import json

import pytest

from leanlean.attempt_timing import model_window
from leanlean.time_analysis import analyze


def write(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in records))


@pytest.fixture
def stalled_attempt(tmp_path):
    write(tmp_path / "observations.jsonl", [
        dict(reason="model_start", started_monotonic_ns=10**9),
        dict(reason="model_end", started_monotonic_ns=301 * 10**9),
    ])
    write(tmp_path / "harness-events.jsonl", [dict(
        observed_monotonic_ns=101 * 10**9, observed_wall_ns=101 * 10**9,
        event=dict(type="result", raw=dict(result=dict(status="ERROR"))),
    )])
    write(tmp_path / "resources.jsonl", [dict(
        phase="model", status="error", observed_monotonic_ns=t * 10**9,
        error="TimeoutExpired('podman inspect', 3)",
    ) for t in (105, 108)])
    return tmp_path


def test_runtime_tail_is_separate_and_phase_times_reconcile(stalled_attempt):
    path = stalled_attempt
    (path / "attempt.json").write_text(json.dumps(dict(harness="codex", problem_idx=1, project_key="project")))
    (path / "actions.json").write_text(json.dumps(dict(spans=[])))
    (path / "commands.jsonl").touch()
    before = (path / "observations.jsonl").read_bytes()
    result = analyze(path)
    assert result["total_s"] == sum(result["seconds"].values()) == 100
    audit = result["timing_adjustment"]
    assert audit["recorded_elapsed_s"] == 300
    assert audit["excluded_runtime_s"] == 200
    assert audit["terminal_status"] == "ERROR"
    assert audit["resource_inspect_timeouts"] == 2
    assert (path / "observations.jsonl").read_bytes() == before


@pytest.mark.parametrize("evidence", ["request", "command", "source", "pruned_scratch", "sample", "missing_samples", "later_event"])
def test_do_not_trim_when_work_continues_or_stall_is_unproven(stalled_attempt, evidence):
    path = stalled_attempt
    if evidence == "request":
        write(path / "native/.harness_wrapper/model_requests.jsonl", [dict(event=dict(
            type="model.request", timestamp="1970-01-01T00:02:00Z",
        ))])
    elif evidence == "command":
        write(path / "commands.jsonl", [dict(phase="model", monotonic_ns=120 * 10**9)])
    elif evidence in {"source", "pruned_scratch"}:
        with (path / "observations.jsonl").open("a") as stream:
            row = dict(reason="periodic", changed=evidence == "source", started_monotonic_ns=120 * 10**9)
            if evidence == "pruned_scratch":
                row["original_changed"] = True
            stream.write(json.dumps(row) + "\n")
    elif evidence == "sample":
        with (path / "resources.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(phase="model", status="sample", observed_monotonic_ns=120 * 10**9)) + "\n")
    elif evidence == "missing_samples":
        (path / "resources.jsonl").unlink()
    else:
        with (path / "harness-events.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(event=dict(type="recovery"), observed_monotonic_ns=120 * 10**9)) + "\n")
    start, end, audit = model_window(path)
    assert (end - start) / 1e9 == 300
    assert audit is None


def test_short_exit_latency_is_retained(stalled_attempt):
    path = stalled_attempt
    write(path / "observations.jsonl", [
        dict(reason="model_start", started_monotonic_ns=10**9),
        dict(reason="model_end", started_monotonic_ns=105 * 10**9),
    ])
    assert model_window(path)[2] is None


def test_retryable_inspection_timeouts_are_also_stall_evidence(stalled_attempt):
    write(stalled_attempt / "resources.jsonl", [dict(
        phase="model", status="retrying", reason="container_inspect_timeout", observed_monotonic_ns=t * 10**9,
    ) for t in (105, 108)])
    assert model_window(stalled_attempt)[2]["resource_inspect_timeouts"] == 2
