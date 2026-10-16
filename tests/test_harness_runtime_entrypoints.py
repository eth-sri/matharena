"""Execution flags reach harnesses without changing model/provider configuration."""

import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from matharena.query_client import create_query_client
from matharena.runner import Runner
from matharena.solvers.harness_solver import PodmanHarnessSandbox
from matharena.solvers.judges.judge_pool import JudgePool
from matharena.solvers.judges.judge_response import JudgeResponse
from matharena.solvers.judges.simple_judge import SimpleJudge

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("runtime", [None, "docker", "podman"])
def test_run_cli_passes_runtime_to_every_competition(monkeypatch, runtime):
    seen = []

    class FakeRunner:
        def __init__(self, comp, *args, container_runtime):
            seen.append((comp, container_runtime))

        def prepare_run(self, model):
            return None

    monkeypatch.setattr("matharena.runner.Runner", FakeRunner)
    argv = ["run.py", "--comp", "example/one", "example/two", "--models", "example/model"]
    if runtime is not None:
        argv += ["--container-runtime", runtime]
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "scripts/run.py"), run_name="__main__")
    assert seen == [("example/one", runtime or "docker"), ("example/two", runtime or "docker")]


def test_runner_simple_judge_and_curation_use_the_execution_argument(tmp_path):
    model = {"harness": "codex", "model": "gpt-test", "harness_config": {"managed_cli": False}}
    runner = Runner.__new__(Runner)
    runner.container_runtime = "podman"
    solver = runner._initialize_solver({"harness": "codex"}, "{problem}", model, "")
    judge = SimpleJudge(0, 1, 0, {"model_config": model}, container_runtime="podman")
    client = create_query_client(model, container_runtime="podman")
    for solver in (solver, judge.harness_solver, client.solver):
        assert isinstance(solver._build_sandbox(tmp_path), PodmanHarnessSandbox)
        assert "container_runtime" not in solver.harness_config
    assert "container_runtime" not in model["harness_config"]


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_judge_pool_preserves_runtime_for_every_retry(runtime):
    seen = []

    class FakeJudge:
        def __init__(self, *, batch_idx, container_runtime, solver_config, **kwargs):
            self.idx = batch_idx
            seen.append(container_runtime)
            assert "container_runtime" not in solver_config

        def solve(self, *args):
            return JudgeResponse(self.idx, None if len(seen) == 1 else 1, "assessment", {}, [], {})

    pool = JudgePool({"scaffold_name": "simple_judge"}, container_runtime=runtime)
    pool.AGENT_CLASS = FakeJudge
    response = pool._run_agent(0, 1, 0, ("problem", {
        "guidelines": "", "ground_truth_solutions": [], "student_answer": "answer",
    }))
    assert response.points == 1
    assert seen == [runtime, runtime]


@pytest.mark.parametrize("script", [
    "scripts/run.py", "scripts/judge/judge.py", "scripts/judge/compare_judges.py",
    "arxivmath/scripts/source/screen_abstracts.py",
    "arxivmath/scripts/source/investigate_sources.py",
    "arxivmath/scripts/source/verify_questions.py",
])
def test_harness_capable_python_commands_expose_runtime_flag(monkeypatch, capsys, script):
    monkeypatch.setattr(sys, "argv", [script, "--help"])
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(ROOT / script), run_name="__main__")
    assert result.value.code == 0
    assert "--container-runtime {docker,podman}" in capsys.readouterr().out


@pytest.fixture
def fake_commands(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "commands.jsonl"
    stub = f"#!{sys.executable}\n" + """import json, os, sys
from pathlib import Path
with open(os.environ['MATHARENA_TEST_COMMAND_LOG'], 'a') as log:
    log.write(json.dumps([Path(sys.argv[0]).name, *sys.argv[1:]]) + '\\n')
"""
    for command in ("uv", "python", "python3", "docker", "podman", "codex"):
        path = bin_dir / command
        path.write_text(stub)
        path.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MATHARENA_TEST_COMMAND_LOG", str(log))
    monkeypatch.setenv("MATHARENA_PYTHON", str(bin_dir / "python"))
    monkeypatch.setenv("OPENAI_API_KEY", "test-only-key")
    monkeypatch.setenv("GLM_API_KEY", "test-only-key")
    return log


@pytest.mark.parametrize("runtime", ["docker", "podman"])
@pytest.mark.parametrize("script, positional", [
    ("run_all_fa.sh", ["example/model"]),
    ("run_all_kangaroo.sh", ["example/model"]),
    ("run_euler.sh", []),
])
def test_shell_launchers_forward_runtime_and_keep_existing_arguments(
    tmp_path, fake_commands, runtime, script, positional
):
    subprocess.run(
        ["bash", str(ROOT / "scripts" / script), "--container-runtime", runtime, *positional],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    calls = [json.loads(line) for line in fake_commands.read_text().splitlines()]
    launches = [call for call in calls if "scripts/run.py" in call]
    assert len(launches) == (4 if script == "run_euler.sh" else 1)
    for call in launches:
        assert call[call.index("--container-runtime") + 1] == runtime
    if script == "run_all_fa.sh":
        judgments = [call for call in calls if "scripts/judge/judge.py" in call]
        assert judgments
        assert all(call[call.index("--container-runtime") + 1] == runtime for call in judgments)



@pytest.mark.parametrize("arguments", [["--container-runtime"], ["--container-runtime", "invalid"]])
def test_shell_runtime_errors_happen_before_any_execution(tmp_path, fake_commands, arguments):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_all_fa.sh"), "example/model", *arguments],
        cwd=tmp_path, capture_output=True, text=True,
    )
    assert result.returncode == 2
    assert not fake_commands.exists()


@pytest.mark.parametrize("runtime", ["docker", "podman"])
@pytest.mark.parametrize("script", ["create.sh", "create_false.sh"])
@pytest.mark.parametrize("mode", ["screen", "prepare"])
def test_curation_shells_forward_runtime_only_to_model_steps(fake_commands, runtime, script, mode):
    subprocess.run(
        ["bash", str(ROOT / "arxivmath/scripts" / script), mode, f"--container-runtime={runtime}"],
        capture_output=True, text=True, check=True,
    )
    calls = [json.loads(line) for line in fake_commands.read_text().splitlines()]
    assert len(calls) == (1 if mode == "screen" else 4)
    for call in calls:
        if "arxivmath/scripts/source/download_sources.py" in call:
            assert "--container-runtime" not in call
        else:
            assert call[call.index("--container-runtime") + 1] == runtime
        if script != "create.sh":
            assert "--false" in call


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_build_cli_overrides_legacy_runtime_environment(monkeypatch, fake_commands, runtime):
    monkeypatch.setenv("MATHARENA_CONTAINER_RUNTIME", "podman" if runtime == "docker" else "docker")
    subprocess.run(
        ["bash", str(ROOT / "scripts/build_harness_arxivlean_image.sh"), "--container-runtime", runtime],
        capture_output=True, text=True, check=True,
    )
    calls = [json.loads(line) for line in fake_commands.read_text().splitlines()]
    assert calls[0][1] == "build"
    assert all(call[0] == runtime for call in calls)


@pytest.mark.parametrize("runtime", ["docker", "podman"])
def test_vllm_evaluation_wrapper_forwards_runtime(tmp_path, monkeypatch, fake_commands, runtime):
    # Use a temporary repository to isolate the training environment and log directories.
    script = tmp_path / "train/scripts/eval_arxiv_march_vllm.sh"
    script.parent.mkdir(parents=True)
    script.write_text((ROOT / "train/scripts/eval_arxiv_march_vllm.sh").read_text())
    helper = tmp_path / "scripts/lib/container_runtime.sh"
    helper.parent.mkdir(parents=True)
    helper.write_text((ROOT / "scripts/lib/container_runtime.sh").read_text())
    monkeypatch.setenv("SOURCE_CONFIG", "false")
    monkeypatch.setenv("SCRATCH", str(tmp_path))
    monkeypatch.setenv("HF_HOME", str(tmp_path / "huggingface"))
    monkeypatch.setenv("EVAL_OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("EVAL_VLLM_LOG", str(tmp_path / "vllm.log"))
    monkeypatch.setenv("EVAL_NUM_SHARDS", "1")
    monkeypatch.setenv("EVAL_SHARD_INDEX", "0")
    subprocess.run(
        ["bash", str(script), "example/competition", "--container-runtime", runtime],
        cwd=tmp_path, capture_output=True, text=True, check=True,
    )
    calls = [json.loads(line) for line in fake_commands.read_text().splitlines()]
    launches = [call for call in calls if "scripts/run.py" in call]
    assert len(launches) == 1
    assert launches[0][launches[0].index("--container-runtime") + 1] == runtime
    assert launches[0][launches[0].index("--comp") + 1] == "example/competition"
