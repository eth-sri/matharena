import importlib.util
import json
import os
from pathlib import Path

import pytest
import yaml

from matharena.eval_run import prepare_run_directory
from matharena.runner import Runner


@pytest.fixture
def inputs(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "problems").mkdir()
    (dataset / "answers.csv").write_text("id,answer\n1,13\n2,14\n3,15\n4,16\n5,17\n")
    for i in range(1, 6):
        (dataset / "problems" / f"{i}.tex").write_text(f"Compute 12 + {i}.")
    (dataset / "eval-runtime.json").write_text('{"docker_image": "sha256:fixed"}\n')
    comps = tmp_path / "competitions"
    models = tmp_path / "models"
    comps.mkdir()
    models.mkdir()
    (comps / "example.yaml").write_text(yaml.safe_dump({
        "dataset_path": str(dataset), "instruction": "Solve {problem}", "allow_harness": True,
        "harness_config": {"runtime_manifest": str(dataset / "eval-runtime.json")},
    }))
    (models / "example.yaml").write_text(yaml.safe_dump({
        "human_readable_id": "Example", "model": "fake", "api": "openai",
        "concurrent_requests": 16, "harness_version": "1.2.3",
    }))
    return {"competition_configs": comps, "model_configs": models,
            "competitions": ["example"], "models": ["example"]}


def test_snapshot_is_independent_and_rebases_runtime_and_workspaces(tmp_path, inputs):
    root = prepare_run_directory(tmp_path / "run", **inputs, workers=4, shuffle_seed=29)
    frozen = yaml.safe_load((root / "configs/competitions/example.yaml").read_text())
    model = yaml.safe_load((root / "configs/models/example.yaml").read_text())
    harness = frozen["harness_config"]
    assert frozen["dataset_path"] == str(root / "datasets/example")
    assert Path(harness["runtime_manifest"]).read_text() == '{"docker_image": "sha256:fixed"}\n'
    assert harness["workspace_root"] == str(root / "workspaces/example/{solver_name}/p{problem_idx}_r{run_idx}")
    assert frozen["shuffle_seed"] == 29
    assert model["concurrent_requests"] == 4
    source_comp = inputs["competition_configs"] / "example.yaml"
    dataset = Path(yaml.safe_load(source_comp.read_text())["dataset_path"])
    (dataset / "problems/1.tex").write_text("changed")
    source_comp.write_text("changed")
    assert (root / "datasets/example/problems/1.tex").read_text() == "Compute 12 + 1."
    assert yaml.safe_load((root / "configs/competitions/example.yaml").read_text()) == frozen
    assert yaml.safe_load((inputs["model_configs"] / "example.yaml").read_text())["concurrent_requests"] == 16
    with pytest.raises(FileExistsError):
        prepare_run_directory(root, **{**inputs, "competitions": []})


def test_snapshot_keeps_remote_dataset_reference_and_copies_external_runtime(tmp_path, inputs):
    config = inputs["competition_configs"] / "example.yaml"
    values = yaml.safe_load(config.read_text())
    values["dataset_path"] = "organization/remote-dataset"
    config.write_text(yaml.safe_dump(values))
    root = prepare_run_directory(tmp_path / "run", **inputs)
    frozen = yaml.safe_load((root / "configs/competitions/example.yaml").read_text())
    assert frozen["dataset_path"] == "organization/remote-dataset"
    assert Path(frozen["harness_config"]["runtime_manifest"]).is_relative_to(root / "runtime")
    assert Path(frozen["harness_config"]["runtime_manifest"]).exists()


def test_snapshot_rejects_recursive_destination_and_unsafe_config_ids(tmp_path, inputs):
    with pytest.raises(ValueError, match="outside the dataset"):
        prepare_run_directory(tmp_path / "dataset/run", **inputs)
    with pytest.raises(ValueError, match="relative paths"):
        prepare_run_directory(tmp_path / "run", **{**inputs, "models": ["../outside"]})
    assert not (tmp_path / "run").exists()


def _cli():
    path = Path(__file__).resolve().parents[1] / "scripts/run.py"
    spec = importlib.util.spec_from_file_location("eval_cli", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("auth_file", [None, "/explicit/private/auth.json"])
def test_prepare_only_never_constructs_a_solver_and_records_order(tmp_path, inputs, monkeypatch, auth_file):
    def unexpected_solver(*args, **kwargs):
        pytest.fail("Preparing a run must not construct a solver or access a provider")

    monkeypatch.setattr(Runner, "prepare_run", unexpected_solver)
    monkeypatch.delenv("HARNESS_WRAPPER_AUTH_FILE", raising=False)
    if auth_file is not None:
        monkeypatch.setenv("HARNESS_WRAPPER_AUTH_FILE", auth_file)
    root = tmp_path / "run"
    cli = _cli()
    cli.main([
        "--comp", "example", "--models", "example", "--n", "1", "--workers", "4",
        "--shuffle-seed", "29", "--run-root", str(root), "--prepare-only",
        "--comp-configs-dir", str(inputs["competition_configs"]),
        "--model-configs-dir", str(inputs["model_configs"]),
    ])
    metadata = json.loads((root / "run.json").read_text())
    order = metadata["problem_order"]["example"]
    assert sorted(order) == [1, 2, 3, 4, 5]
    assert order != [1, 2, 3, 4, 5]
    runner = Runner("example", 1, None, str(root / "configs/competitions"),
                    str(root / "configs/models"), str(root / "outputs"), False)
    assert [p["problem_idx"] for p in runner.problems] == order
    assert metadata["runs_per_problem"] == {"example": 1}
    assert os.environ["HARNESS_WRAPPER_AUTH_FILE"] == (auth_file or str(root / "auth/auth.json"))


@pytest.mark.parametrize("flags,workers", [([], None), (["--workers", "4", "--shuffle-seed", "29"], 4)])
def test_standard_cli_still_runs_without_snapshot(tmp_path, inputs, monkeypatch, flags, workers):
    seen = []

    def prepare(runner, model, **kwargs):
        seen.append((model, kwargs["workers"], [p["problem_idx"] for p in runner.problems]))
        return None

    monkeypatch.setattr(Runner, "prepare_run", prepare)
    _cli().main([
        "--comp", "example", "--models", "example", "--n", "1",
        "--comp-configs-dir", str(inputs["competition_configs"]),
        "--model-configs-dir", str(inputs["model_configs"]), *flags,
    ])
    assert seen[0][:2] == ("example", workers)
    assert sorted(seen[0][2]) == [1, 2, 3, 4, 5]
    assert (seen[0][2] == [1, 2, 3, 4, 5]) == (workers is None)
    assert yaml.safe_load((inputs["model_configs"] / "example.yaml").read_text())["concurrent_requests"] == 16


@pytest.mark.parametrize("flags", [["--workers", "0"], ["--workers", "-1"], ["--prepare-only"]])
def test_invalid_launch_options_fail_before_loading_runners(flags, monkeypatch):
    cli = _cli()
    monkeypatch.setattr(cli, "Runner", lambda *a, **kw: pytest.fail("loaded a runner"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["--comp", "example", "--models", "example", *flags])
    assert exc.value.code == 2


@pytest.mark.parametrize("workers", [None, 4])
def test_worker_override_reaches_solver_without_mutating_model_file(tmp_path, inputs, monkeypatch, workers):
    runner = Runner("example", 1, None, str(inputs["competition_configs"]),
                    str(inputs["model_configs"]), str(tmp_path / "outputs"), False)

    class SolverReached(Exception):
        pass

    def inspect_solver(config, prompt, client_args, last_chance):
        assert client_args["concurrent_requests"] == (16 if workers is None else workers)
        assert config["model_config"]["concurrent_requests"] == client_args["concurrent_requests"]
        raise SolverReached

    monkeypatch.setattr(runner, "_initialize_solver", inspect_solver)
    with pytest.raises(SolverReached):
        runner.prepare_run("example", workers=workers)
    assert yaml.safe_load((inputs["model_configs"] / "example.yaml").read_text())["concurrent_requests"] == 16


def test_snapshot_freezes_scaffold_agent_dependencies(tmp_path, inputs, monkeypatch):
    monkeypatch.chdir(tmp_path)
    dependencies = tmp_path / "configs"
    dependencies.mkdir()
    (dependencies / "inner.yaml").write_text(yaml.safe_dump({"model": "original", "concurrent_requests": 16}))
    (dependencies / "scaffold.yaml").write_text(yaml.safe_dump({"name": "original scaffold"}))
    (inputs["model_configs"] / "example.yaml").write_text(yaml.safe_dump({
        "type": "agent", "human_readable_id": "Example agent",
        "model_config": "inner", "scaffold_config": "scaffold",
    }))
    root = prepare_run_directory(tmp_path / "run", **inputs, workers=4)
    (dependencies / "inner.yaml").write_text("changed")
    (dependencies / "scaffold.yaml").write_text("changed")
    config = Runner.load_solver_config(Runner.__new__(Runner), root / "configs/models/example.yaml")
    assert config["model_config"]["model"] == "original"
    assert config["model_config"]["concurrent_requests"] == 4
    assert config["scaffold_config"]["name"] == "original scaffold"


def test_explicit_problem_ids_control_schedule_without_changing_ids(tmp_path, inputs):
    runner = Runner("example", 1, [5, 1, 3], str(inputs["competition_configs"]),
                    str(inputs["model_configs"]), str(tmp_path / "outputs"), False)
    assert [p["problem_idx"] for p in runner.problems] == [5, 1, 3]
    assert [p["problem"] for p in runner.problems] == ["Compute 12 + 5.", "Compute 12 + 1.", "Compute 12 + 3."]


def test_harness_environment_combines_model_and_competition_settings(tmp_path, inputs, monkeypatch):
    comp_path = inputs["competition_configs"] / "example.yaml"
    comp = yaml.safe_load(comp_path.read_text())
    comp["harness_config"]["environment"] = {"LEAN_NUM_THREADS": "8"}
    comp_path.write_text(yaml.safe_dump(comp))
    model_path = inputs["model_configs"] / "example.yaml"
    model = yaml.safe_load(model_path.read_text())
    model.update(harness="deepcode", harness_config={"environment": {
        "DEEPCODE_CONTEXT_WINDOW": "1m", "DEEPCODE_AUTO_COMPACT_WINDOW": "512k", "LEAN_NUM_THREADS": "2",
    }})
    model_path.write_text(yaml.safe_dump(model))
    runner = Runner("example", 1, None, str(inputs["competition_configs"]),
                    str(inputs["model_configs"]), str(tmp_path / "outputs"), False)

    class SolverReached(Exception):
        pass

    def inspect_solver(config, prompt, client_args, last_chance):
        assert client_args["harness_config"]["environment"] == {
            "DEEPCODE_CONTEXT_WINDOW": "1m", "DEEPCODE_AUTO_COMPACT_WINDOW": "512k", "LEAN_NUM_THREADS": "8",
        }
        raise SolverReached

    monkeypatch.setattr(runner, "_initialize_solver", inspect_solver)
    with pytest.raises(SolverReached):
        runner.prepare_run("example")


@pytest.mark.parametrize("inside_dataset", [True, False])
def test_snapshot_freezes_dataset_helper_files(tmp_path, inputs, inside_dataset):
    config = inputs["competition_configs"] / "example.yaml"
    values = yaml.safe_load(config.read_text())
    helpers = (Path(values["dataset_path"]) if inside_dataset else tmp_path) / "helpers"
    helpers.mkdir()
    script = helpers / "check.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    values["harness_config"].update(repo_eval=True, helper_files=str(helpers))
    config.write_text(yaml.safe_dump(values))
    root = prepare_run_directory(tmp_path / "run", **inputs)
    frozen = yaml.safe_load((root / "configs/competitions/example.yaml").read_text())
    saved = Path(frozen["harness_config"]["helper_files"]) / "check.sh"
    script.write_text("changed")
    assert saved.is_relative_to(root)
    assert saved.read_text() == "#!/bin/sh\nexit 0\n"
    assert os.access(saved, os.X_OK)
    assert frozen["harness_config"]["repo_eval"] is True


def test_repo_eval_disables_inline_answer_grading(tmp_path, inputs):
    config = inputs["competition_configs"] / "example.yaml"
    values = yaml.safe_load(config.read_text())
    values["harness_config"]["repo_eval"] = True
    config.write_text(yaml.safe_dump(values))
    runner = Runner("example", 1, None, str(inputs["competition_configs"]),
                    str(inputs["model_configs"]), str(tmp_path / "outputs"), False)
    assert runner.is_repo_eval
    assert not runner.is_fa_comp
    assert not runner.is_auto_graded_comp
