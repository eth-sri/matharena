import io
import json
import sys
import tarfile
from pathlib import Path

import pytest

from leanlean.postprocess import extract_submission, measure, restore_fixtures, validate_lakefile


@pytest.fixture
def project_pair(tmp_path):
    baseline, repo = tmp_path / "baseline", tmp_path / "repository"
    baseline.mkdir()
    repo.mkdir()
    entry = {
        "challenge_path": "Challenge.lean",
        "comparator_path": "comparator.json",
        "lakefile_path": "lakefile.toml",
        "toolchain_path": "lean-toolchain",
        "project_path": "",
    }
    config = '''name = "project"
defaultTargets = ["Helpers", "Challenge", "Solution"]
requiresModuleSystem = true
[leanOptions]
relaxedAutoImplicit = false
[[require]]
name = "mathlib"
rev = "pinned"
[[lean_lib]]
name = "Helpers"
globs = ["Helpers.+"]
[[lean_lib]]
name = "Challenge"
roots = ["Challenge"]
[[lean_lib]]
name = "Solution"
roots = ["Solution"]
'''
    for root in (baseline, repo):
        for name, content in {
            "lakefile.toml": config,
            "Challenge.lean": "theorem target : True := by sorry",
            "comparator.json": '{"solution_module": "Solution", "theorem_names": ["target"]}',
            "lean-toolchain": "pinned-toolchain",
            "lake-manifest.json": '{"packages": []}',
        }.items():
            (root / name).write_text(content)
    return repo, baseline, entry


def test_library_reorganization_restores_original_fixtures_and_keeps_submitted_lakefile(project_pair):
    repo, baseline, entry = project_pair
    path = repo / "lakefile.toml"
    config = path.read_text().replace('"Helpers", ', "").replace(
        '[[lean_lib]]\nname = "Helpers"\nglobs = ["Helpers.+"]\n', ""
    )
    path.write_text(config)
    (repo / "Challenge.lean").write_text("weakened statement")
    (repo / "comparator.json").write_text('{"theorem_names": []}')
    (repo / "Solution.lean").write_text("submitted proof")
    audit = restore_fixtures(repo, baseline, entry)
    assert audit["lakefile.toml"]["libraries_removed"] == ["Helpers"]
    assert audit["lakefile.toml"]["restored"] is False
    assert path.read_text() == config
    assert (repo / "Solution.lean").read_text() == "submitted proof"
    for name in ("Challenge.lean", "comparator.json"):
        assert (repo / name).read_bytes() == (baseline / name).read_bytes()
        assert audit[name]["restored"] is True


def test_library_roots_globs_and_local_source_directories_can_change(project_pair):
    repo, baseline, entry = project_pair
    path = repo / "lakefile.toml"
    path.write_text(path.read_text().replace('roots = ["Solution"]', 'roots = ["Solution.Main"]\nsrcDir = "src"') +
                    '\n[[lean_lib]]\nname = "Shared"\nglobs = ["Shared.+"]\n')
    audit = validate_lakefile(repo, baseline, entry)
    assert audit["libraries_added"] == ["Shared"]
    assert audit["libraries_modified"] == ["Solution"]


@pytest.mark.parametrize("old,new", [
    ('relaxedAutoImplicit = false', 'relaxedAutoImplicit = true'),
    ('requiresModuleSystem = true', 'requiresModuleSystem = false'),
    ('name = "Helpers"\nglobs', 'name = "Helpers"\nmoreLeanArgs = ["--trust=999"]\nglobs'),
    ('name = "Helpers"\nglobs', 'name = "Helpers"\nplugins = ["custom"]\nglobs'),
])
def test_build_option_changes_warn_and_preserve_submitted_settings(project_pair, old, new):
    repo, baseline, entry = project_pair
    path = repo / "lakefile.toml"
    path.write_text(path.read_text().replace(old, new))
    submitted = path.read_text()
    with pytest.warns(UserWarning, match="Build options changed"):
        audit = restore_fixtures(repo, baseline, entry)
    assert audit["lakefile.toml"]["warnings"]
    assert path.read_text() == submitted


def test_dependency_declarations_remain_pinned(project_pair):
    repo, baseline, entry = project_pair
    path = repo / "lakefile.toml"
    path.write_text(path.read_text().replace('rev = "pinned"', 'rev = "different"'))
    with pytest.raises(ValueError, match="pinned dependencies"):
        restore_fixtures(repo, baseline, entry)


@pytest.mark.parametrize(
    "directory", ["/opt/leanlean/sources", "../original", ".lake/packages/mathlib", ".private", "src/.hidden"]
)
def test_library_cannot_point_outside_scored_submitted_sources(project_pair, directory):
    repo, baseline, entry = project_pair
    if directory.startswith(".lake"):
        (repo / ".lake/packages").mkdir(parents=True)
        (repo / directory).symlink_to(baseline, target_is_directory=True)
    path = repo / "lakefile.toml"
    path.write_text(path.read_text().replace('roots = ["Solution"]', f'roots = ["Solution"]\nsrcDir = "{directory}"'))
    with pytest.raises(ValueError, match="srcDir is outside scored sources"):
        validate_lakefile(repo, baseline, entry)


def test_alternate_executable_lakefile_cannot_bypass_toml_checks(project_pair):
    repo, baseline, entry = project_pair
    (repo / "lakefile.lean").write_text("import Lake\nopen Lake DSL\npackage replacement")
    with pytest.raises(ValueError, match="alternate lakefile.lean"):
        restore_fixtures(repo, baseline, entry)


@pytest.mark.parametrize("change", ["redirect", "remove", "shadow", "wildcard"])
def test_registered_challenge_cannot_resolve_to_a_different_library(project_pair, change):
    repo, baseline, entry = project_pair
    path = repo / "lakefile.toml"
    config = path.read_text()
    if change == "redirect":
        config = config.replace('roots = ["Challenge"]', 'roots = ["Challenge"]\nsrcDir = "replacement"')
    elif change == "remove":
        config = config.replace('[[lean_lib]]\nname = "Challenge"\nroots = ["Challenge"]\n', "")
    elif change == "shadow":
        config += '\n[[lean_lib]]\nname = "Replacement"\nroots = ["Challenge"]\nsrcDir = "replacement"\n'
    else:
        config += '\n[[lean_lib]]\nname = "Replacement"\nglobs = ["*"]\nsrcDir = "replacement"\n'
    path.write_text(config)
    with pytest.raises(ValueError, match="redirects the registered challenge"):
        restore_fixtures(repo, baseline, entry)


def test_changed_executable_lakefile_is_restored(project_pair):
    repo, baseline, entry = project_pair
    for root in (repo, baseline):
        (root / "lakefile.toml").unlink()
        (root / "lakefile.lean").write_text("original executable config")
    entry["lakefile_path"] = "lakefile.lean"
    assert validate_lakefile(repo, baseline, entry)["status"] == "unchanged"
    (repo / "lakefile.lean").write_text("changed executable config")
    with pytest.warns(UserWarning, match="executable lakefile.lean; restored"):
        audit = validate_lakefile(repo, baseline, entry)
    assert audit["status"] == "modified" and audit["restored"] is True
    assert (repo / "lakefile.lean").read_text() == "original executable config"


@pytest.mark.parametrize("name", ["lean-toolchain", "lake-manifest.json"])
def test_toolchain_and_dependency_lock_remain_pinned(project_pair, name):
    repo, baseline, entry = project_pair
    (repo / name).write_text("changed")
    with pytest.raises(ValueError, match="Build configuration changed"):
        restore_fixtures(repo, baseline, entry)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_final_verification_builds_the_registered_solution(tmp_path, monkeypatch, phase):
    from leanlean import postprocess_container as runner

    repository = tmp_path / "repository"
    repository.mkdir()
    (tmp_path / "leanlean-project.json").write_text(
        json.dumps({"project_path": "", "comparator_path": "comparator.json"})
    )
    (repository / "comparator.json").write_text('{"solution_module": "Solution.Main"}')
    calls = []

    def path(value):
        original = Path(value)
        return tmp_path / original.relative_to("/work") if original.is_relative_to("/work") else original

    def stage(name, command, project, timeout):
        calls.append(command)
        return {"returncode": 0}

    monkeypatch.setattr(runner, "Path", path)
    monkeypatch.setattr(runner, "run_stage", stage)
    monkeypatch.setattr(sys, "argv", ["postprocess", "--phase", phase])
    with pytest.raises(SystemExit) as stopped:
        runner.main()
    assert stopped.value.code == 0
    assert calls[0] == ["lake", "build", "+Solution.Main"]


def archive(path, members):
    with tarfile.open(path, "w:gz") as stream:
        for member, body in members:
            stream.addfile(member, io.BytesIO(body) if body else None)


@pytest.mark.parametrize("kind", ["traversal", "symlink", "hardlink", "cache", "duplicate"])
def test_submission_rejects_unsafe_entries(tmp_path, kind):
    member = tarfile.TarInfo("../escape" if kind == "traversal" else ".lake/build/a" if kind == "cache" else "A.lean")
    if kind in {"symlink", "hardlink"}:
        member.type = tarfile.SYMTYPE if kind == "symlink" else tarfile.LNKTYPE
        member.linkname = "/tmp/target"
    members = [(member, b"")]
    if kind == "duplicate":
        members.append((member, b""))
    path = tmp_path / "source.tar.gz"
    archive(path, members)
    with pytest.raises(ValueError):
        extract_submission(path, tmp_path / "repository")


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE])
def test_internal_file_alias_is_materialized_and_audited(tmp_path, kind):
    body = b'{"solution_module":"Solution"}'
    original = tarfile.TarInfo("Protected/comparator.json")
    original.size = len(body)
    alias = tarfile.TarInfo("comparator.json")
    alias.type = kind
    alias.linkname = original.name
    path = tmp_path / "source.tar.gz"
    archive(path, [(alias, b""), (original, body)])
    destination = tmp_path / "repository"
    with pytest.warns(UserWarning, match="Materialized internal archive link"):
        audit = extract_submission(path, destination)
    assert (destination / alias.name).read_bytes() == body
    assert not (destination / alias.name).is_symlink()
    assert audit[0]["target"] == original.name


@pytest.mark.parametrize("kind", ["escape", "cycle", "missing", "ancestor"])
def test_invalid_internal_links_are_rejected_before_extracting_files(tmp_path, kind):
    link = tarfile.TarInfo("alias")
    link.type = tarfile.SYMTYPE
    link.linkname = {"escape": "../outside", "cycle": "alias", "missing": "absent", "ancestor": "file"}[kind]
    body = b"contents"
    regular = tarfile.TarInfo("alias/child" if kind == "ancestor" else "file")
    regular.size = len(body)
    path = tmp_path / "source.tar.gz"
    archive(path, [(regular, body), (link, b"")])
    destination = tmp_path / "repository"
    with pytest.raises(ValueError):
        extract_submission(path, destination)
    assert list(destination.iterdir()) == []


def test_restoration_preserves_solution_and_records_modified_fixture(tmp_path):
    baseline = tmp_path / "baseline"
    repo = tmp_path / "repository"
    baseline.mkdir()
    repo.mkdir()
    entry = {
        "challenge_path": "Challenge.lean",
        "comparator_path": "comparator.json",
        "lakefile_path": "lakefile.toml",
        "toolchain_path": "lean-toolchain",
        "project_path": "",
    }
    for name in ["Challenge.lean", "comparator.json", "lakefile.toml", "lean-toolchain", "lake-manifest.json"]:
        (baseline / name).write_text("registered")
        (repo / name).write_text("registered")
    (repo / "Challenge.lean").write_text("edited")
    (repo / "Solution.lean").write_text("final proof")
    audit = restore_fixtures(repo, baseline, entry)
    assert audit["Challenge.lean"]["status"] == "modified"
    assert audit["comparator.json"]["status"] == "unchanged"
    assert (repo / "Challenge.lean").read_text() == "registered"
    assert (repo / "Solution.lean").read_text() == "final proof"
    (repo / "lean-toolchain").write_text("changed")
    with pytest.raises(ValueError, match="Build configuration changed"):
        restore_fixtures(repo, baseline, entry)


def test_metrics_exclude_challenge_build_config_and_generated_files(tmp_path):
    for name in ["Solution.lean", "Challenge.lean", "lakefile.lean", ".lake/build/Generated.lean", ".private/X.lean"]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("two tokens")
    result = measure(tmp_path, {"project_path": "", "challenge_path": "Challenge.lean"}, lambda text: len(text.split()))
    assert result == {"lean_tokens": 2, "files": {"Solution.lean": 2}, "lean_file_count": 1}
