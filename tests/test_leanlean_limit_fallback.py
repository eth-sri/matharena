"""A failed submission is scored at the attempt's latest passing state."""

import json

from leanlean.limit_fallback import candidates, cutoff
from leanlean.website_progress import limit_fallback_marks, scored_point

SECOND = 10**9
# 2026-11-02T07:25:27Z
LIMIT_WALL = 1793604327 * SECOND


def observation(n, source, wall, reason="periodic"):
    return dict(observation_index=n, reason=reason, source_sha256=source,
                started_wall_ns=wall, started_monotonic_ns=wall - LIMIT_WALL + 1000 * SECOND)


def history():
    return [
        observation(1, "base", LIMIT_WALL - 300 * SECOND, "baseline"),
        observation(2, "base", LIMIT_WALL - 299 * SECOND, "model_start"),
        observation(3, "good", LIMIT_WALL - 200 * SECOND),
        observation(4, "broken", LIMIT_WALL - 100 * SECOND),
        observation(5, "good", LIMIT_WALL - 50 * SECOND),
        observation(6, "half", LIMIT_WALL - 10 * SECOND),
        observation(7, "half", LIMIT_WALL - 6 * SECOND, "harness_result"),
        observation(8, "salvaged", LIMIT_WALL + 60 * SECOND),
        observation(9, "salvaged", LIMIT_WALL + 900 * SECOND, "model_end"),
        observation(10, "salvaged", LIMIT_WALL + 901 * SECOND, "final"),
    ]


def test_every_attempt_cuts_at_its_final_source_and_records_how_it_ended():
    def cut(limits):
        return cutoff(limits, history())
    assert cut(dict(exceeded="time_limit")) == dict(reason="time_limit", limit_at=None, observation=9)
    grace = dict(seconds=900, started_at="2026-11-02T07:25:27+00:00", completed=False)
    # A time limit reached inside a cost-limit grace turn still stopped the model mid-work.
    assert cut(dict(exceeded="time_limit", cost_limit_grace=grace))["reason"] == "time_limit"
    # The grace turn's work is kept: the cut stays at the end of the run.
    for limits in (dict(exceeded="cost_limit"), dict(exceeded="cost_grace_timeout", cost_limit_grace=grace),
                   dict(exceeded=None, cost_limit_grace=dict(grace, completed=True))):
        assert cut(limits) == dict(reason="cost_limit", limit_at=None, observation=9)
    assert cut(dict(exceeded=None))["reason"] == cut(None)["reason"] == "submitted"


def test_candidates_are_distinct_states_by_last_observation():
    order = [row["source_sha256"] for row in candidates(history(), 6)]
    # A reverted-to state counts from when it was last held; the baseline comes last.
    assert order == ["half", "good", "broken", "base"]
    assert candidates(history(), 6)[1]["last_observation"] == 5


def test_candidates_skip_failed_snapshots():
    rows = history()
    failed = {k: v for k, v in rows[4].items() if k not in {"source_sha256", "archive", "snapshot_index"}}
    rows.insert(5, dict(failed, error="FileNotFoundError(2, 'No such file or directory')"))
    assert [row["source_sha256"] for row in candidates(rows, 7)] == ["half", "good", "broken", "base"]


def test_marks_and_checkpoints_follow_a_grace_turn_back_into_the_attempt_it_continues(tmp_path):
    from leanlean.common import chain_frames, submission_attempt

    first, grace = tmp_path / "first", tmp_path / "grace"
    first.mkdir()
    grace.mkdir()
    (first / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history()))
    later = LIMIT_WALL + 2000 * SECOND
    (grace / "observations.jsonl").write_text("".join(json.dumps(row) + "\n" for row in [
        observation(1, "salvaged", later, "baseline"), observation(2, "salvaged", later + SECOND, "model_start"),
        observation(3, "late", later + 31 * SECOND), observation(4, "late", later + 61 * SECOND, "model_end"),
    ]))
    (grace / "continues.json").write_text(json.dumps(dict(previous_attempt=str(first))))
    # The scored state was last held in the attempt the grace turn continues; the cut is the grace submission.
    fallback = dict(reason="cost_limit", limit_at=None, cut_observation=4, scored_source_sha256="good",
                    scored_last_observation=5, scored_attempt=str(first), scored_is_submission=False)
    marks = limit_fallback_marks(grace, fallback)
    assert marks == dict(reason="cost_limit", limit_at=None, cut_elapsed=1259.0, scored_elapsed=249.0,
                         scored_is_cut=False)

    def checkpoints(directory, rows):
        (directory / "checkpoints.jsonl").write_text("".join(
            json.dumps(dict(source_sha256=source, parent_sha256=parent, archive=f"sources/{source}.tar.gz")) + "\n"
            for source, parent in rows))
    checkpoints(first, [("base", None), ("good", "base"), ("salvaged", "good")])
    checkpoints(grace, [("salvaged", None), ("late", "salvaged")])
    frames = chain_frames([first, grace])
    assert [(f["source_sha256"], f["continued_attempt"]) for f in frames] == [
        ("base", 0), ("good", 0), ("salvaged", 0), ("salvaged", 1), ("late", 1)]
    assert chain_frames([grace])[0].keys() == {"source_sha256", "parent_sha256", "archive"}
    checkpoints(grace, [("other", None)])
    try:
        chain_frames([first, grace])
    except ValueError as error:
        assert "does not open" in str(error)
    else:
        raise AssertionError("A continuation must open on its predecessor's final source")
    submission = dict(archive=str(tmp_path / "work/limit-fallback-x/leanlean-submission.tar.gz"))
    assert submission_attempt(submission) == tmp_path / "work"
    assert submission_attempt(dict(submission, attempt=str(grace))) == grace
