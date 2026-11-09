"""A failed submission is scored at the attempt's latest passing state."""

import pytest

from leanlean.limit_fallback import candidates, cutoff, sync_sources

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


def put(root, relative, content):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_deletion_invalidates_local_builds_but_preserves_external_packages(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    put(old, "Removed.lean", "theorem old : True := by trivial")
    put(old, "Kept.lean", "def kept := 1")
    kept = put(new, "Kept.lean", "def kept := 1")
    put(old, ".lake/build/Removed.olean", "stale")
    put(old, "vendor/lib/.lake/build/Removed.olean", "stale")
    package = put(tmp_path, "external/build/Library.olean", "pinned")
    (old / ".lake/packages").symlink_to(package.parent.parent, target_is_directory=True)
    before_mtime = (old / "Kept.lean").stat().st_mtime_ns
    result = sync_sources(new, old)
    assert result == {"deleted": ["Removed.lean"], "changed": [], "build_cache_reset": True}
    assert not (old / ".lake/build").exists()
    assert not (old / "vendor/lib/.lake/build").exists()
    assert package.read_text() == "pinned"
    assert (old / "Kept.lean").stat().st_mtime_ns == before_mtime
    assert (old / "Kept.lean").read_bytes() == kept.read_bytes()


def test_edit_preserves_build_cache_and_reverts_and_path_type_changes(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    put(old, "Solution.lean", "def x := 1")
    artifact = put(old, ".lake/build/Solution.olean", "warm")
    put(new, "Solution.lean", "def x := 2")
    assert not sync_sources(new, old)["build_cache_reset"]
    assert artifact.exists()
    assert (old / "Solution.lean").read_text() == "def x := 2"
    (new / "Solution.lean").write_text("def x := 1")
    sync_sources(new, old)
    assert (old / "Solution.lean").read_text() == "def x := 1"
    (new / "Solution.lean").unlink()
    put(new, "Solution.lean/Child.lean", "def child := 1")
    assert sync_sources(new, old)["build_cache_reset"]
    (new / "Solution.lean/Child.lean").unlink()
    (new / "Solution.lean").rmdir()
    put(new, "Solution.lean", "def x := 3")
    sync_sources(new, old)
    assert (old / "Solution.lean").read_text() == "def x := 3"


def test_unexpected_symlink_cannot_redirect_source_sync(tmp_path):
    old, new = tmp_path / "old", tmp_path / "new"
    target = put(tmp_path, "external/Solution.lean", "original")
    old.mkdir()
    (old / "Solution.lean").symlink_to(target)
    put(new, "Solution.lean", "new")
    with pytest.raises(ValueError, match="symlink"):
        sync_sources(new, old)
    assert target.read_text() == "original"
