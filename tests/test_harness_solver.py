import tempfile
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import pytest
import yaml

from harness_wrapper import AgentEvent, CLIRelease, InstalledCLI, TokenUsage
from matharena.runner import Runner
from matharena.solvers.harness_solver import (
    CONTAINER_HARNESS_CLI_ROOT,
    CONTAINER_HARNESS_PATH,
    CONTAINER_HARNESS_VENV,
    DEFAULT_HARNESS_DOCKER_IMAGE,
    DockerHarnessSandbox,
    HarnessSolver,
    _events_to_messages,
)


def _solver_config(**model_config):
    return {
        "type": "pure_model",
        "model_config": model_config,
        "scaffold_config": None,
    }


def _runner(allow_harness=False):
    return SimpleNamespace(
        comp_name="example",
        competition_config={"allow_harness": allow_harness},
    )


def test_harness_is_disabled_by_default():
    assert Runner._resolve_harness(_runner(), _solver_config(model="test")) is None


def test_allowed_competition_defaults_to_codex_and_accepts_override():
    runner = _runner(allow_harness=True)

    assert Runner._resolve_harness(runner, _solver_config(model="test")) == "codex"
    assert (
        Runner._resolve_harness(runner, _solver_config(model="test", harness="claude"))
        == "claude"
    )
    assert (
        Runner._resolve_harness(runner, _solver_config(model="test", harness=False))
        is None
    )
    assert (
        Runner._resolve_harness(runner, _solver_config(model="test", harness=True))
        == "codex"
    )


@pytest.mark.parametrize(
    ("default_harness", "expected"),
    [(False, None), (True, "codex"), ("qwen", "qwen")],
)
def test_competition_can_choose_default_without_overriding_models(
    default_harness, expected
):
    runner = _runner(allow_harness=True)
    runner.competition_config["default_harness"] = default_harness

    assert Runner._resolve_harness(runner, _solver_config()) == expected
    assert Runner._resolve_harness(runner, _solver_config(harness="codex")) == "codex"
    assert Runner._resolve_harness(runner, _solver_config(harness="kimi")) == "kimi"
    assert Runner._resolve_harness(runner, _solver_config(harness=False)) is None
    scaffold = {"type": "agent", "model_config": {}, "scaffold_config": {}}
    assert Runner._resolve_harness(runner, scaffold) is None


@pytest.mark.parametrize("default_harness", [None, 0, 1, [], {}, "", " "])
def test_competition_default_harness_is_validated(default_harness):
    runner = _runner(allow_harness=True)
    runner.competition_config["default_harness"] = default_harness
    with pytest.raises(TypeError, match="default_harness"):
        Runner._resolve_harness(runner, _solver_config())


def test_default_harness_cannot_bypass_competition_policy():
    runner = _runner()
    runner.competition_config["default_harness"] = "kimi"
    assert Runner._resolve_harness(runner, _solver_config()) is None
    assert Runner._resolve_harness(runner, _solver_config(harness="codex")) is None
    assert Runner._resolve_harness(runner, _solver_config(harness="kimi")) is None


@pytest.mark.parametrize("allow_harness", [False, None])
@pytest.mark.parametrize(
    "harness",
    ("deepcode", "muse", "kimi", "qwen", "opencode", "claude", "gravity",
     "codex", "codex_cli", "openai-codex", True),
)
def test_disallowed_harness_warns_and_falls_back_to_api(harness, allow_harness, monkeypatch):
    warnings = []
    monkeypatch.setattr("matharena.runner.logger.warning", warnings.append)
    runner = _runner()
    if allow_harness is None:
        runner.competition_config.pop("allow_harness")
    assert Runner._resolve_harness(runner, _solver_config(harness=harness)) is None
    assert len(warnings) == 1
    assert "example" in warnings[0]
    assert "Using the normal model API" in warnings[0]


@pytest.mark.parametrize("harness", [None, False, "off", ""])
def test_no_warning_when_no_harness_is_requested(harness, monkeypatch):
    warnings = []
    monkeypatch.setattr("matharena.runner.logger.warning", warnings.append)
    assert Runner._resolve_harness(_runner(), _solver_config(harness=harness)) is None
    assert warnings == []


@pytest.mark.parametrize("harness", ["deepcode", "codex"])
def test_prepare_run_falls_back_without_sending_harness_parameters(harness, tmp_path, monkeypatch):
    config = {
        "model": "test-model", "api": "deepseek", "human_readable_id": "Test model",
        "harness": harness, "harness_version": "0.3.1",
        "harness_config": {"auth": "subscription", "container_executable": harness},
        "reasoning_effort": "max", "max_tokens": 384000, "read_cost": 0.3,
    }
    (tmp_path / "test.yaml").write_text(yaml.safe_dump(config))
    runner = Runner.__new__(Runner)
    runner.solver_configs_dir = str(tmp_path)
    runner.base_output_dir = str(tmp_path / "outputs")
    runner.comp_name = "arxiv/may"
    runner.competition_config = {"instruction": "Solve {problem}"}
    runner.options = None

    class PreparedAPI(Exception):
        pass

    def prepare_api_solver(solver_config, prompt, client_args, last_chance):
        assert solver_config["harness"] is None
        assert prompt == "Solve {problem}"
        assert not {"harness", "harness_version", "harness_config"} & client_args.keys()
        assert client_args["model"] == "test-model"
        assert client_args["api"] == "deepseek"
        assert client_args["reasoning_effort"] == "max"
        assert client_args["max_tokens"] == 384000
        assert client_args["read_cost"] == 0.3
        raise PreparedAPI

    monkeypatch.setattr("matharena.runner.PureModelSolver", prepare_api_solver)
    with pytest.raises(PreparedAPI):
        runner.prepare_run("test")


def test_failed_harness_future_does_not_discard_other_batch_results(
    monkeypatch, tmp_path
):
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "concurrent_requests": 3,
            "harness_config": {
                "managed_cli": False,
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            },
        },
        "",
    )

    def solve_one(batch_idx, *_args):
        if batch_idx == 0:
            raise RuntimeError("failed result must remain pending")
        return SimpleNamespace(idx=batch_idx)

    monkeypatch.setattr(solver, "_solve_one", solve_one)

    responses = list(
        solver.solve_batch(
            [("zero", None), ("one", None), ("two", None)],
            {0: 10, 1: 11, 2: 12},
            {0: 0, 1: 0, 2: 0},
        )
    )

    assert {response.idx for response in responses} == {1, 2}



@pytest.mark.parametrize("has_terminal_result", [True, False])
@pytest.mark.parametrize("subagents", [{}, {"worker": "gpt-6-astra"}])
def test_empty_completed_harness_attempt_is_saved_with_usage(
    tmp_path, monkeypatch, has_terminal_result, subagents
):
    from matharena.json_zst import load_json_zst
    from matharena.runs import Runs

    events = [AgentEvent(type="reasoning", content="Unfinished reasoning.")]
    if has_terminal_result:
        events.append(AgentEvent(type="result", content="", raw={"stop_reason": "max_tokens"}))
    agent = SimpleNamespace(
        run=lambda prompt: events,
        session_id="empty-answer-session",
        get_tokens=lambda: TokenUsage(input_tokens=6, output_tokens=256000),
    )
    solver = HarnessSolver(
        _solver_config(model="claude-test"),
        "{problem}",
        {
            "model": "claude-test", "api": "anthropic", "harness": "claude",
            "read_cost": 10, "write_cost": 50,
            "harness_config": {
                "managed_cli": False,
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
                **({"subagents": subagents} if subagents else {}),
            },
        },
        "",
    )
    def make_agent(**kwargs):
        assert kwargs["subagents"] == subagents
        return agent

    monkeypatch.setattr("matharena.solvers.harness_solver.Agent", make_agent)
    monkeypatch.setattr(solver, "_prepare_harness_cli", lambda: None)
    monkeypatch.setattr(solver, "_build_model", lambda: None)
    monkeypatch.setattr(solver, "_build_sandbox", lambda workspace: None)

    if not has_terminal_result:
        with pytest.raises(RuntimeError, match="without a final response"):
            solver._solve_one(0, "Problem", None, 55, 0)
        return

    response = solver._solve_one(0, "Problem", None, 55, 0)
    runs = Runs(
        "arxiv/august", True, "claude-test", "agent",
        {"problem_idx": 55, "problem": "Problem", "answer": "42"}, str(tmp_path),
    )
    runs.add_run(response, (None, "TODO Grading", 0))
    runs.save_to_file()
    saved = load_json_zst(runs.path)
    assert saved["N"] == 1
    assert saved["messages"][0][-1] == {
        "role": "assistant", "type": "response", "content": "",
    }
    assert saved["messages"][0][-2]["content"] == "Unfinished reasoning."
    assert saved["cost"]["output_tokens"] == 256000
    assert saved["cost"]["cost"] == pytest.approx(12.80006)
    assert saved["history"][0][0]["events"][-1]["raw"]["stop_reason"] == "max_tokens"


def test_allow_harness_must_be_a_yaml_boolean():
    with pytest.raises(TypeError, match="must be a boolean"):
        Runner._resolve_harness(
            _runner(allow_harness="false"), _solver_config(model="test")
        )


def test_matharena_scaffold_agents_are_not_replaced_by_default_harness():
    config = {
        "type": "agent",
        "model_config": {"model": "test"},
        "scaffold_config": {},
    }

    assert Runner._resolve_harness(_runner(allow_harness=True), config) is None


def test_existing_arxivlean_configs_keep_ordinary_api_models_on_api_path():
    for path in (
        "configs/competitions/arxivlean/march.yaml",
        "configs/competitions/arxivlean/june.yaml",
    ):
        with open(path, encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file)
        runner = SimpleNamespace(comp_name=path, competition_config=config)
        assert Runner._resolve_harness(runner, _solver_config(model="test")) is None


def test_arxivlean_june_accepts_explicit_harness_models():
    config = yaml.safe_load(Path("configs/competitions/arxivlean/june.yaml").read_text())
    assert config["allow_harness"] is True
    assert config["default_harness"] is False
    runner = SimpleNamespace(comp_name="arxivlean/june", competition_config=config)
    assert Runner._resolve_harness(runner, _solver_config(harness="codex")) == "codex"


def test_api_model_uses_existing_matharena_openai_credentials(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://api.openai.com/v1"
    assert endpoint.api_key != "api-credit-key"
    assert endpoint.headers["Authorization"] == "Bearer api-credit-key"
    assert model.cli_environment("openai")["OPENAI_API_KEY"] == endpoint.api_key
    assert "api-credit-key" not in model.cli_environment("openai").values()


@pytest.mark.parametrize("harness", ["claude", "claude-code", "anthropic", "codex"])
def test_openrouter_harness_uses_native_protocol_and_host_credentials(monkeypatch, tmp_path, harness):
    monkeypatch.setenv("OPENROUTER_API_KEY", "openrouter-credit-key")
    path = Path(__file__).resolve().parents[1] / "configs/models/anthropic/opus_5_max.yaml"
    config = yaml.safe_load(path.read_text())
    config["harness"] = harness
    config["harness_config"]["workspace_root"] = str(tmp_path / "p{problem_idx}_r{run_idx}")
    solver = HarnessSolver(_solver_config(**config), "{problem}", config, "")
    model = solver._build_model()
    protocol = "openai" if harness == "codex" else "anthropic"
    endpoint = model.endpoint_for(protocol)
    assert endpoint.url == (
        "https://openrouter.ai/api/v1" if harness == "codex" else "https://openrouter.ai/api"
    )
    assert endpoint.headers["Authorization"] == "Bearer openrouter-credit-key"
    assert endpoint.api_key != "openrouter-credit-key"
    assert "openrouter-credit-key" not in model.cli_environment(protocol).values()
    assert model.reasoning == "max"
    if harness != "codex":
        assert model.request_overrides() == config["extra_body"]
        assert "reasoning_effort" not in model.request_overrides()


def test_kimi_harness_uses_glm_api_credits_through_host_proxy(monkeypatch, tmp_path):
    monkeypatch.setenv("GLM_API_KEY", "glm-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="glm-5.3"),
        "{problem}",
        {
            "model": "glm-5.3",
            "api": "glm",
            "temperature": 0.7,
            "extra_body": {
                "thinking": {"type": "enabled"},
            },
            "harness": "kimi",
            "harness_config": {
                "docker_image": "test-kimi-image",
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            },
        },
        "",
    )

    model = solver._build_model()

    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://api.z.ai/api/paas/v4"
    assert model.request_overrides()["temperature"] == 0.7
    assert model.request_overrides()["thinking"] == {"type": "enabled"}
    assert endpoint.api_key != "glm-api-credit-key"
    assert endpoint.headers["Authorization"] == "Bearer glm-api-credit-key"
    assert "glm-api-credit-key" not in model.cli_environment("openai").values()


def test_kimi_harness_normalizes_google_requests_and_preserves_model_params(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="gemini-test"),
        "{problem}",
        {
            "model": "gemini-test",
            "api": "google",
            "harness": "kimi",
            "harness_version": "0.36.0",
            "max_tokens": 1234,
            "temperature": 0.7,
            "tool_choice": "auto",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    assert model.request_drop_fields() == frozenset({"prompt_cache_key"})
    assert model.request_overrides()["max_tokens"] == 1234
    assert model.request_overrides()["temperature"] == 0.7
    assert model.request_overrides()["tool_choice"] == "auto"
    assert "harness_version" not in model.request_overrides()


@pytest.mark.parametrize("harness", ("qwen", "opencode"))
def test_native_google_harness_preserves_generation_config(
    monkeypatch, tmp_path, harness
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="gemini-test"),
        "{problem}",
        {
            "model": "gemini-test",
            "api": "google",
            "harness": harness,
            "max_tokens": 1234,
            "temperature": 0.25,
            "reasoning_effort": "low",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    endpoint = model.endpoint_for("gemini")
    assert endpoint.url == "https://generativelanguage.googleapis.com"
    assert endpoint.api_key != "google-api-credit-key"
    assert endpoint.headers["x-goog-api-key"] == "google-api-credit-key"
    generation = model.request_overrides()["generationConfig"]
    assert generation["maxOutputTokens"] == 1234
    assert generation["temperature"] == 0.25
    assert generation["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "low",
    }


def test_antigravity_cli_uses_native_google_api_and_preserves_generation_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="gemini-test"),
        "{problem}",
        {
            "model": "gemini-test",
            "api": "google",
            "harness": "gravity",
            "harness_version": "1.1.26",
            "max_tokens": 1234,
            "temperature": 0.25,
            "reasoning_effort": "medium",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    endpoint = model.endpoint_for("gemini")
    assert endpoint.url == "https://generativelanguage.googleapis.com"
    assert endpoint.api_key != "google-api-credit-key"
    assert endpoint.headers["x-goog-api-key"] == "google-api-credit-key"
    generation = model.request_overrides()["generationConfig"]
    assert generation["maxOutputTokens"] == 1234
    assert generation["temperature"] == 0.25
    assert generation["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "medium",
    }
    assert model.request_drop_fields() == {
        "generationConfig.thinkingConfig.thinkingBudget"
    }
    assert "google-api-credit-key" not in model.cli_environment("gemini").values()


def test_antigravity_cli_translates_nested_google_thinking_config(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("GOOGLE_API_KEY", "google-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="gemini-test"),
        "{problem}",
        {
            "model": "gemini-test",
            "api": "google",
            "harness": "gravity",
            "extra_body": {
                "extra_body": {
                    "google": {
                        "thinking_config": {
                            "include_thoughts": True,
                            "thinking_level": "medium",
                        }
                    }
                }
            },
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    assert model.request_overrides()["generationConfig"]["thinkingConfig"] == {
        "includeThoughts": True,
        "thinkingLevel": "medium",
    }


@pytest.mark.parametrize("harness", ["qwen", "opencode"])
def test_additional_openai_compatible_harnesses_preserve_matharena_params(
    monkeypatch, tmp_path, harness
):
    monkeypatch.setenv("GLM_API_KEY", "glm-api-credit-key")
    solver = HarnessSolver(
        _solver_config(model="glm-test"),
        "{problem}",
        {
            "model": "glm-test",
            "api": "glm",
            "harness": harness,
            "temperature": 0.3,
            "max_tokens": 4321,
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    model = solver._build_model()

    assert model.request_overrides()["temperature"] == 0.3
    assert model.request_overrides()["max_tokens"] == 4321
    assert model.endpoint_for("openai").headers["Authorization"] == (
        "Bearer glm-api-credit-key"
    )


def test_harness_version_defaults_to_latest_and_mounts_resolved_cli(
    monkeypatch, tmp_path
):
    prefix = tmp_path / "cache" / "kimi-code" / "0.40.1"
    executable = prefix / "bin" / "kimi"
    executable.parent.mkdir(parents=True)
    executable.write_text("fake")
    installed = InstalledCLI(
        CLIRelease(
            "kimi-code",
            "@moonshot-ai/kimi-code",
            "0.40.1",
            "kimi",
        ),
        "latest",
        prefix,
    )
    calls = []

    def resolve(name, version, *, cache_root=None):
        calls.append((name, version, cache_root))
        return installed

    monkeypatch.setattr("matharena.solvers.harness_solver.ensure_cli", resolve)
    solver = HarnessSolver(
        _solver_config(model="glm-test"),
        "{problem}",
        {
            "model": "glm-test",
            "api": "glm",
            "harness": "kimi",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
                "cli_cache_root": str(tmp_path / "cache"),
            },
        },
        "",
    )

    assert solver.harness_version == "latest"
    assert solver._prepare_harness_cli() == installed
    assert solver._prepare_harness_cli() == installed
    sandbox = solver._build_sandbox(tmp_path)

    assert calls == [("kimi-code", "latest", str(tmp_path / "cache"))]
    assert sandbox.container_executable == CONTAINER_HARNESS_CLI_ROOT / "bin" / "kimi"
    assert any(
        mount.source == prefix and mount.target == CONTAINER_HARNESS_CLI_ROOT
        for mount in sandbox.mounts
    )


def test_claude_harness_uses_the_shared_cli_image(tmp_path):
    solver = HarnessSolver(
        _solver_config(model="claude-test"),
        "{problem}",
        {
            "model": "claude-test",
            "api": "anthropic",
            "harness": "claude",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    sandbox = solver._build_sandbox(tmp_path)
    assert sandbox.image == DEFAULT_HARNESS_DOCKER_IMAGE
    assert sandbox.container_executable == PurePosixPath("claude")


def test_claude_subscription_uses_anthropic_oauth(monkeypatch, tmp_path):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    solver = HarnessSolver(
        _solver_config(model="sonnet"),
        "{problem}",
        {
            "model": "sonnet",
            "api": "anthropic",
            "harness": "claude",
            "harness_config": {
                "auth": "subscription",
                "oauth_auto_login": False,
                "oauth_auto_relogin": False,
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            },
        },
        "",
    )

    model = solver._build_model()

    assert model.auth_mode == "oauth"
    assert model.provider == "anthropic"
    assert model.config.auto_login is False
    assert model.config.auto_relogin is False


def test_container_bundled_kimi_does_not_require_a_host_install(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "matharena.solvers.harness_solver.shutil.which",
        lambda name: "/usr/bin/true" if name == "true" else None,
    )
    solver = HarnessSolver(
        _solver_config(model="glm-test"),
        "{problem}",
        {
            "model": "glm-test",
            "api": "glm",
            "harness": "kimi",
            "harness_config": {
                "docker_image": "test-kimi-image",
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            },
        },
        "",
    )

    assert solver._agent_executable() == "/usr/bin/true"


def test_pinned_image_enforces_no_new_privileges_internally_for_snap_docker(
    tmp_path,
):
    base_config = {
        "model": "gpt-test",
        "api": "openai",
        "harness": "codex",
        "harness_config": {
            "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
        },
    }
    pinned = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        base_config,
        "",
    )
    custom_config = {
        **base_config,
        "harness_config": {
            **base_config["harness_config"],
            "docker_image": "custom-image",
        },
    }
    custom = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        custom_config,
        "",
    )

    assert "no-new-privileges:true" not in pinned._build_sandbox(tmp_path).extra_args
    custom_args = custom._build_sandbox(tmp_path).extra_args
    assert "no-new-privileges:true" in custom_args
    entrypoint = Path("docker/harness-arxivlean/harness-entrypoint").read_text()
    assert "setpriv --no-new-privs" in entrypoint


def test_pinned_image_uses_outer_docker_sandbox_instead_of_nested_bwrap():
    requirements = Path("docker/harness-arxivlean/requirements.toml").read_text()

    assert 'allowed_approval_policies = ["never"]' in requirements
    assert "allowed_web_search_modes = []" in requirements
    assert "default_permissions" not in requirements
    assert "[permissions." not in requirements


def test_isolated_harness_accepts_codex_subscription_without_api_key(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "harness_config": {
                "auth": "subscription",
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
            },
        },
        "",
    )

    model = solver._build_model()

    assert model.auth_mode == "oauth"
    assert model.provider == "openai"
    assert model.config.auto_login is False
    assert model.config.auto_relogin is False


def test_docker_sandbox_passes_secret_by_environment_name_only(tmp_path):
    sandbox = DockerHarnessSandbox(
        root=tmp_path,
        image="test-image",
        network_enabled=False,
        container_root=PurePosixPath("/work"),
        container_executable="codex",
    )

    command, cwd, environment = sandbox.prepare_command(
        ["/host/bin/codex", "exec"],
        cwd=tmp_path,
        env={"OPENAI_API_KEY": "secret-value"},
    )

    assert cwd == tmp_path
    assert "secret-value" not in command
    assert command[-2:] == ["codex", "exec"]
    assert command[command.index("--env") + 1] == "OPENAI_API_KEY"
    assert environment["OPENAI_API_KEY"] == "secret-value"

    utility, _, _ = sandbox.prepare_command(["./check.sh"], cwd=tmp_path)
    assert utility[-1] == "./check.sh"


@pytest.mark.parametrize("fail_during_run", [False, True])
def test_docker_proxy_network_is_isolated_and_cleaned_up(
    tmp_path, monkeypatch, fail_during_run
):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("matharena.solvers.harness_solver.subprocess.run", run)
    sandbox = DockerHarnessSandbox(root=tmp_path, network_enabled=False)

    try:
        with sandbox.expose_host_service() as route:
            create = calls[0]
            assert create[1:4] == ["network", "create", "--internal"]
            assert route.client_host == create[create.index("--gateway") + 1]
            assert route.listen_host == "0.0.0.0"
            assert route.allow_remote_clients is True
            argv, _, _ = sandbox.prepare_command(["claude"], cwd=tmp_path)
            assert argv[argv.index("--network") + 1] == create[-1]
            if fail_during_run:
                raise RuntimeError("simulated harness failure")
    except RuntimeError as error:
        assert fail_during_run and str(error) == "simulated harness failure"

    assert calls[-1][1:] == ["network", "rm", calls[0][-1]]
    argv, _, _ = sandbox.prepare_command(["claude"], cwd=tmp_path)
    assert argv[argv.index("--network") + 1] == "none"


def test_networked_docker_sandbox_adds_linux_host_gateway(tmp_path):
    sandbox = DockerHarnessSandbox(
        root=tmp_path,
        image="test-image",
        network_enabled=True,
        container_executable="codex",
    )

    command, _, _ = sandbox.prepare_command(["codex", "exec"], cwd=tmp_path)

    index = command.index("--add-host")
    assert command[index + 1] == "host.docker.internal:host-gateway"


def test_lean_workspace_adapts_competition_tools_to_files(tmp_path):
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "Prove this.\n{problem}\n{formal_statement}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "is_lean_comp": True,
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )
    payload = {
        "problem": "True is true.",
        "formal_statement": "theorem example : True := by trivial",
    }
    workspace = solver._workspace_for(1, 0)
    workspace.mkdir(parents=True)
    (workspace / "Problem.md").write_text("stale duplicate")
    (workspace / "prompt.txt").write_text("stale duplicate")
    (workspace / "Problem.lean").write_text("stale duplicate")
    (workspace / "TOOLS.md").write_text("stale duplicate")
    prompt = solver.build_prompt(payload)

    solver._prepare_workspace(workspace, payload)

    assert payload["problem"] in prompt
    assert payload["formal_statement"] in prompt
    assert not (workspace / "Problem.md").exists()
    assert not (workspace / "prompt.txt").exists()
    assert not (workspace / "Problem.lean").exists()
    assert not (workspace / "TOOLS.md").exists()
    assert (
        (workspace / "Solution.lean")
        .read_text()
        .endswith("theorem example : True := by trivial\n")
    )
    assert (workspace / "check.sh").stat().st_mode & 0o111
    assert {path.name for path in workspace.iterdir()} == {"Solution.lean", "check.sh"}
    assert prompt == solver.default_prompt_template.format(**payload)


def test_plain_harness_workspace_is_reinitialized_for_every_run(tmp_path):
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "Solve this problem:\n{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )
    workspace = solver._workspace_for(1, 0)
    solver._prepare_workspace(workspace, "first attempt")
    (workspace / "Problem.md").write_text("stale duplicate")
    (workspace / "prompt.txt").write_text("stale duplicate")
    (workspace / "search_delta.py").write_text("stale agent code")
    state = workspace / ".harness-home" / "sessions"
    state.mkdir(parents=True)
    (state / "old-session.json").write_text("stale session")
    payload = "The problem appears only in the initial user message."

    solver._prepare_workspace(workspace, payload)

    assert payload in solver.build_prompt(payload)
    assert list(workspace.iterdir()) == []


def test_harness_workspace_reset_refuses_broad_directories(tmp_path):
    for unsafe in (Path.cwd(), Path.home(), Path(tempfile.gettempdir())):
        with pytest.raises(ValueError, match="unsafe harness workspace"):
            HarnessSolver._reset_workspace(unsafe)


def test_harness_home_persists_native_sessions_in_workspace(tmp_path):
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )

    environment = solver._agent_environment()

    assert environment["HOME"] == "/work/.harness-home"
    assert environment["XDG_CONFIG_HOME"] == "/work/.harness-home/.config"
    assert environment["VIRTUAL_ENV"] == CONTAINER_HARNESS_VENV
    assert environment["PATH"] == CONTAINER_HARNESS_PATH
    assert environment["PATH"].startswith(f"{CONTAINER_HARNESS_CLI_ROOT}/bin:")


def test_matharena_harness_uses_minimal_reproducible_context_by_default(tmp_path):
    base = {
        "model": "gpt-test",
        "api": "openai",
        "harness": "codex",
        "harness_config": {
            "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
        },
    }

    solver = HarnessSolver(_solver_config(model="gpt-test"), "{problem}", base, "")
    assert solver.minimal_context is True

    opted_out = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            **base,
            "harness_config": {
                **base["harness_config"],
                "minimal_context": False,
            },
        },
        "",
    )
    assert opted_out.minimal_context is False


def test_lean_harness_rejects_a_mismatched_toolchain(tmp_path):
    with pytest.raises(ValueError, match="Competition requires Lean 4.29.0"):
        HarnessSolver(
            _solver_config(model="gpt-test"),
            "{problem}",
            {
                "model": "gpt-test",
                "api": "openai",
                "harness": "codex",
                "is_lean_comp": True,
                "lean_environment": "lean-4.29.0",
                "harness_config": {
                    "lean_toolchain_dir": "leanprover--lean4---v4.31.0",
                    "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}"),
                },
            },
            "",
        )


def test_harness_events_convert_to_matharena_conversation():
    events = [
        AgentEvent(type="reasoning", content="thinking", role="assistant"),
        AgentEvent(
            type="tool_call",
            content={"command": "./check.sh"},
            tool_name="shell",
            tool_call_id="call-1",
        ),
        AgentEvent(
            type="tool_result",
            content="ok",
            tool_name="shell",
            tool_call_id="call-1",
        ),
        AgentEvent(type="message", content="done", role="assistant"),
    ]

    messages = _events_to_messages(events)

    assert [message["type"] for message in messages if "type" in message] == [
        "cot",
        "tool_call",
        "response",
    ]
    assert messages[1]["arguments"] == {"command": "./check.sh"}
    assert messages[2]["role"] == "tool_response"
    assert messages[-1]["content"] == "done"


def test_cost_conversion_uses_cached_openai_tokens(tmp_path):
    solver = HarnessSolver(
        _solver_config(model="gpt-test"),
        "{problem}",
        {
            "model": "gpt-test",
            "api": "openai",
            "harness": "codex",
            "read_cost": 5,
            "cache_read_cost": 0.5,
            "write_cost": 30,
            "harness_config": {
                "workspace_root": str(tmp_path / "p{problem_idx}_r{run_idx}")
            },
        },
        "",
    )
    agent = SimpleNamespace(
        get_tokens=lambda: TokenUsage(
            input_tokens=100,
            output_tokens=20,
            cache_read_tokens=40,
        )
    )

    cost = solver._detailed_cost(agent)

    assert cost["input_tokens"] == 100
    assert cost["cached_input_tokens"] == 40
    assert cost["cost"] == pytest.approx(0.00092)


def test_model_request_log_is_linked_from_history_with_safe_path(tmp_path):
    request_log = tmp_path / "run" / ".harness_wrapper" / "model_requests.jsonl"
    request_log.parent.mkdir(parents=True)
    request_log.write_text("{}\n", encoding="utf-8")

    path = HarnessSolver._model_request_history_path(
        SimpleNamespace(model_request_log_path=request_log)
    )

    assert path == request_log.name
    request_log.unlink()
    assert (
        HarnessSolver._model_request_history_path(
            SimpleNamespace(model_request_log_path=request_log)
        )
        is None
    )


def test_muse_uses_meta_key_via_proxy_and_responses_parameters(monkeypatch):
    monkeypatch.setenv("META_API_KEY", "meta-test-secret")
    solver = HarnessSolver(
        _solver_config(model="spark"), "{problem}",
        {"model": "spark", "api": "meta", "harness": "muse",
         "reasoning_effort": "xhigh", "max_tokens": 131072}, "",
    )
    model = solver._build_model()
    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://api.meta.ai/v1"
    assert endpoint.api_key != "meta-test-secret"
    assert endpoint.headers["Authorization"] == "Bearer meta-test-secret"
    overrides = model.request_overrides()
    assert overrides["reasoning"] == {"effort": "xhigh"}
    assert overrides["max_output_tokens"] == 131072
    assert "max_tokens" not in overrides
    assert "reasoning_effort" not in overrides


@pytest.mark.parametrize("allow_harness", [False, True])
def test_muse_config_routes_by_competition_and_preserves_max_reasoning(tmp_path, monkeypatch, allow_harness):
    from harness_wrapper import Agent
    from matharena.solvers.pure_model_solver import PureModelSolver

    monkeypatch.setenv("META_API_KEY", "meta-test-secret")
    root = Path(__file__).resolve().parents[1]
    runner = Runner.__new__(Runner)
    runner.comp_name = "example"
    runner.competition_config = {"instruction": "Solve {problem}", "allow_harness": allow_harness}
    runner.solver_configs_dir = str(root / "configs/models")
    runner.base_output_dir = str(tmp_path / "outputs")
    runner.runs_per_problem = 1
    runner.redo_all = False
    runner.options = None
    runner.is_lean_comp = False
    runner.is_auto_graded_comp = True
    runner.problems = [{"problem_idx": 1, "problem": "Compute 6 + 7.", "answer": "13"}]
    monkeypatch.chdir(tmp_path)

    model_config = yaml.safe_load((root / "configs/models/meta/muse_spark_13.yaml").read_text())
    prepared = runner.prepare_run("meta/muse_spark_13", set_request_metadata=False)
    solver = prepared["solver"]
    if not allow_harness:
        assert isinstance(solver, PureModelSolver)
        assert solver.client.model == "muse-spark-1.3"
        assert solver.client.concurrent_requests == model_config["concurrent_requests"]
        assert solver.client.kwargs["reasoning"] == {"effort": "max"}
        assert solver.client.stream_openai_responses is True
        assert solver.client.kwargs["max_output_tokens"] == model_config["max_tokens"]
        assert not {"harness", "harness_version", "harness_config"} & solver.client.kwargs.keys()
        return

    assert isinstance(solver, HarnessSolver)
    assert solver.harness == "muse"
    assert solver.harness_version == "1.0.3-R2198.1"
    assert solver.concurrent_requests == model_config["concurrent_requests"]
    model = solver._build_model()
    assert model.model == "muse-spark-1.3"
    assert model.reasoning == "max"
    assert model.request_overrides()["reasoning"] == {"effort": "max"}
    assert model.request_overrides()["max_output_tokens"] == model_config["max_tokens"]
    assert model.request_overrides()["store"] is False
    assert model.request_overrides()["include"] == ["reasoning.encrypted_content"]
    executable = tmp_path / "muse"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    agent = Agent(type="muse", model=model, dir=tmp_path, executable=executable, subagents={})
    command = agent.build_command("Compute 6 + 7.")
    assert command[command.index("--reasoning-effort") + 1] == "max"
    env = agent.model_environment()
    assert env["MUSE_EXPERIMENTAL_FIRST_TURN_MINIMAL_EFFORT"] == "0"
    assert env["TBH_STREAM_FIRST_EVENT_TIMEOUT_SECS"] == "28800"
    assert env["TBH_STREAM_IDLE_TIMEOUT_SECS"] == "28800"


def test_mimo_opencode_config_preserves_router_settings_and_context(monkeypatch, tmp_path):
    import json
    from harness_wrapper import Agent

    monkeypatch.setenv("OPENROUTER_API_KEY", "mimo-test-key")
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/models/xiaomi/mimo-v2.6-pro.yaml").read_text())
    solver = HarnessSolver(_solver_config(**config), "{problem}", config, "")
    model = solver._build_model()
    endpoint = model.endpoint_for("openai")
    assert endpoint.url == "https://openrouter.ai/api/v1"
    assert endpoint.headers["Authorization"] == "Bearer mimo-test-key"
    assert endpoint.api_key != "mimo-test-key"
    assert model.request_overrides()["max_tokens"] == 131072
    assert model.request_overrides()["reasoning"] == {"enabled": True}
    assert model.request_overrides()["provider"] == {"only": ["io-net/fp8"], "allow_fallbacks": False}
    executable = tmp_path / "opencode"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    agent = Agent(
        type=solver.harness, model=model, executable=executable, dir=tmp_path,
        subagents={}, minimal_context=True,
        model_context_window=solver.harness_config["model_context_window"],
    )
    generated = json.loads(agent.model_environment()["OPENCODE_CONFIG_CONTENT"])
    definition = generated["provider"]["harness-wrapper"]["models"][config["model"]]
    assert definition["limit"] == {"context": 1000000, "input": 868928, "output": 131072}
    assert definition["reasoning"] is True
    assert generated["compaction"]["auto"] is True
