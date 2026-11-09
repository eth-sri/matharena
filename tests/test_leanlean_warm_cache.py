"""Image preparation must finish writable cache metadata before evaluation."""

import subprocess

import pytest

from leanlean import warm_cache


@pytest.mark.parametrize("fail", [False, True])
def test_warming_preserves_sources_and_keeps_dependency_metadata(tmp_path, monkeypatch, fail):
    image = tmp_path / "image"
    source = image / "sources/example/nested"
    source.mkdir(parents=True)
    (source / "Solution.lean").write_text("theorem t : True := by trivial\n")
    (source / "comparator.json").write_text('{"solution_module": "Solution"}')
    packages = image / "packages/example"
    packages.mkdir(parents=True)
    monkeypatch.setattr(warm_cache, "IMAGE_ROOT", image)
    visited = []

    def build(command, *, cwd, env, check, timeout):
        visited.append(cwd)
        assert command == ["lake", "build", "+Solution"] and check
        assert timeout == 21600 and env["LEAN_NUM_THREADS"] == "8"
        assert (cwd / ".lake/packages").resolve() == packages
        (cwd / "Solution.lean").write_text("scratch only")
        (cwd / ".lake/packages/artifact.hash").write_text("cache metadata")
        if fail:
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setenv("LEAN_NUM_THREADS", "8")
    monkeypatch.setattr(warm_cache.subprocess, "run", build)
    if fail:
        with pytest.raises(subprocess.CalledProcessError):
            warm_cache.warm_project(
                {"key": "example", "project_path": "nested", "comparator_path": "nested/comparator.json"}
            )
    else:
        warm_cache.warm_project(
            {"key": "example", "project_path": "nested", "comparator_path": "nested/comparator.json"}
        )
    assert (packages / "artifact.hash").read_text() == "cache metadata"
    assert (source / "Solution.lean").read_text() == "theorem t : True := by trivial\n"
    assert not visited[0].exists()
