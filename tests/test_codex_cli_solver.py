import json
from pathlib import Path
import shutil
import subprocess
import uuid

import pytest

from matharena.runner import Runner
from matharena.solvers.codex_cli_solver import (
    DEFAULT_DOCKER_IMAGE,
    CodexCLISolver,
    _codex_history_messages,
    _command_with_codex_options,
    _display_workspace_path,
    _parse_codex_jsonl,
)
from matharena.utils import normalize_conversation


def test_codex_command_auto_uses_workspace_sandbox_before_stdin_prompt():
    cmd = _command_with_codex_options(
        ["codex", "exec", "--json"],
        model="gpt-test",
        reasoning_effort="high",
        codex_sandbox="auto",
    )

    assert cmd == [
        "codex",
        "exec",
        "-c",
        'model_reasoning_effort="high"',
        "-m",
        "gpt-test",
        "--json",
        "--sandbox",
        "workspace-write",
        "-",
    ]


def test_codex_command_can_use_managed_permission_profile():
    cmd = _command_with_codex_options(
        [
            "codex",
            "--ask-for-approval",
            "never",
            "exec",
            "--json",
            "--config",
            'web_search="disabled"',
        ],
        model="gpt-test",
        reasoning_effort="high",
        codex_sandbox="permissions",
    )

    assert cmd == [
        "codex",
        "--ask-for-approval",
        "never",
        "exec",
        "-c",
        'model_reasoning_effort="high"',
        "-m",
        "gpt-test",
        "--json",
        "--config",
        'web_search="disabled"',
        "-",
    ]
    assert "--dangerously-bypass-approvals-and-sandbox" not in cmd
    assert "--sandbox" not in cmd


def test_runner_loads_codex_cli_model_config():
    config = Runner.load_solver_config(object(), "configs/models/openai/codex-pro-arxivlean.yaml")

    assert config["type"] == "codex_cli"
    assert config["human_readable_id"] == "GPT-5.6-Sol (max, via Codex)"
    assert config["model_config"]["model"] == "gpt-5.6-sol"
    assert config["model_config"]["model_reasoning_effort"] == "max"
    assert config["model_config"]["accounting_costs_from_usage"] is True
    assert 'shell_environment_policy.inherit="all"' in config["model_config"]["cmd"]
    assert config["model_config"]["codex_sandbox"] == "permissions"
    assert config["model_config"]["lean_runtime"] == "image"
    assert config["model_config"]["read_only_rootfs"] is True
    assert config["model_config"]["docker_extra_args"] == [
        "--security-opt",
        "seccomp=unconfined",
        "--security-opt",
        "systempaths=unconfined",
    ]
    assert config["model_config"]["docker_image"] == DEFAULT_DOCKER_IMAGE
    assert "arxivlean/june" in config["model_config"]["custom_instructions"]
    assert config["scaffold_config"] is None


def test_runner_loads_codex_harness_model_config():
    config = Runner.load_solver_config(object(), "configs/models/openai/codex-max.yaml")

    assert config["type"] == "pure_model"
    assert config["human_readable_id"] == "GPT-5.6-Sol (max, via Codex subscription)"
    assert config["model_config"]["model"] == "gpt-5.6-sol"
    assert config["model_config"]["reasoning_effort"] == "max"
    assert config["model_config"]["api"] == "openai"
    assert config["model_config"]["harness"] == "codex"
    harness_config = config["model_config"]["harness_config"]
    assert harness_config["auth"] == "subscription"
    assert harness_config["oauth_auto_login"] is False
    assert harness_config["oauth_auto_relogin"] is False
    assert harness_config["read_only_rootfs"] is True
    assert "lean_runtime" not in harness_config
    assert "docker_image" not in harness_config
    assert "custom_instructions" not in config["model_config"]
    assert config["scaffold_config"] is None


def test_workspace_history_path_does_not_expose_host_parent(tmp_path):
    workspace = tmp_path / "private-parent" / "p1_r0"

    assert _display_workspace_path(workspace) == "p1_r0"


def test_codex_jsonl_usage_and_history_messages_are_extracted():
    events = [
        {"type": "thread.started", "thread_id": "t1"},
        {
            "type": "item.completed",
            "item": {
                "id": "item_0",
                "type": "agent_message",
                "text": "Thinking aloud.",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_1",
                "type": "command_execution",
                "command": "./check.sh",
                "aggregated_output": "ok\n",
                "exit_code": 0,
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "item_2",
                "type": "file_change",
                "changes": [{"path": "/work/Solution.lean", "kind": "update"}],
                "status": "completed",
            },
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "cached_input_tokens": 40,
                "output_tokens": 7,
                "reasoning_output_tokens": 3,
            },
        },
    ]
    stdout = "\n".join(json.dumps(event, ensure_ascii=False, separators=(",", ":")) for event in events)

    usage = _parse_codex_jsonl(stdout)
    assert usage.input_tokens == 100
    assert usage.cached_input_tokens == 40
    assert usage.fresh_input_tokens == 60
    assert usage.output_tokens == 7
    assert usage.reasoning_output_tokens == 3
    assert usage.n_turns == 1

    messages = _codex_history_messages(
        "prompt", stdout, "```lean\nexample : True := by trivial\n```"
    )
    assert messages[0] == {"role": "user", "content": "prompt"}
    assert any(m.get("tool_name") == "shell" for m in messages)
    assert any(m.get("tool_name") == "file_change" for m in messages)
    assert messages[-1]["content"].startswith("```lean")
    assert normalize_conversation(messages)[-1]["type"] == "response"


def _docker_available():
    image_result = subprocess.run(
        ["docker", "image", "inspect", DEFAULT_DOCKER_IMAGE],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    volume_result = subprocess.run(
        ["docker", "volume", "inspect", "matharena-lean431-cache"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return image_result.returncode == 0 and volume_result.returncode == 0


def test_codex_cli_solver_smoke_runs_lean_in_isolated_docker_workspace():
    if not _docker_available():
        pytest.skip(f"{DEFAULT_DOCKER_IMAGE} Docker image is not available")

    workspace_root = (
        Path.cwd()
        / "logs"
        / "codex_pytest_smoke"
        / uuid.uuid4().hex
        / "p{problem_idx}_r{run_idx}"
    )
    try:
        solver = CodexCLISolver(
            {"type": "codex_cli"},
            "{problem}\n\n```lean\n{formal_statement}\n```",
            {
                "competition": "smoke",
                "solver_name": "codex-cli-smoke",
                "cmd": [
                    "sh",
                    "-c",
                    "printf '%s\\n' "
                    "'import Mathlib' '' "
                    "'theorem codex_smoke_true : True := by' "
                    "'  trivial' > Solution.lean",
                ],
                "send_prompt_stdin": False,
                "copy_codex_auth": False,
                "docker_network": "none",
                "timeout_s": 60,
                "concurrent_requests": 1,
                "workspace_root": str(workspace_root),
                "run_check_after_codex": True,
                "lean_runtime": "image",
                "lean_cache_volume": "matharena-lean431-cache",
                "lean_toolchain_dir": "leanprover--lean4---v4.31.0",
                "read_only_rootfs": True,
                "docker_extra_args": [
                    "--security-opt",
                    "seccomp=unconfined",
                    "--security-opt",
                    "systempaths=unconfined",
                ],
            },
            "",
        )

        payload = {
            "problem": "Prove True.",
            "formal_statement": "theorem codex_smoke_true : True := by\n  sorry",
        }
        responses = list(solver.solve_batch([(payload, None)], {0: 1}, {0: 0}))

        assert len(responses) == 1
        response = responses[0]
        assert response.idx == 0
        assert (
            "theorem codex_smoke_true : True := by"
            in response.conversation[-1]["content"]
        )
        assert "trivial" in response.conversation[-1]["content"]
        assert response.conversation == response.history[0]["messages"]
        workspace = Path(response.history[0]["workspace"])
        assert (workspace / "check_result.json").exists()
        assert '"returncode": 0' in (workspace / "check_result.json").read_text()
    finally:
        shutil.rmtree(workspace_root.parent, ignore_errors=True)


def test_codex_permission_profile_blocks_credentials_and_network():
    if not _docker_available():
        pytest.skip(f"{DEFAULT_DOCKER_IMAGE} Docker image is not available")

    root = Path.cwd() / "logs" / "codex_sandbox_pytest" / uuid.uuid4().hex
    workspace = root / "work"
    auth_source = root / "host-auth.json"
    workspace.mkdir(parents=True)
    auth_source.write_text('{"secret":"must-not-be-readable"}\n', encoding="utf-8")
    (workspace / "Solution.lean").write_text(
        "import Mathlib\n\nexample : True := by\n  trivial\n",
        encoding="utf-8",
    )
    probe = """set -eu
printf 'workspace-write-ok\n' > workspace-write.txt
if cat /run/codex-home/auth.json >/dev/null 2>&1; then
  echo 'credential read unexpectedly succeeded' >&2
  exit 41
fi
if node -e 'fetch("https://example.com").then(() => process.exit(0)).catch(() => process.exit(1))'; then
  echo 'network access unexpectedly succeeded' >&2
  exit 42
fi
lean Solution.lean
"""

    try:
        solver = CodexCLISolver(
            {"type": "codex_cli"},
            "{problem}",
            {
                "cmd": ["sh", "-c", "true"],
                "send_prompt_stdin": False,
                "copy_codex_auth": True,
                "codex_auth_path": str(auth_source),
                "docker_network": "bridge",
                "docker_extra_args": [
                    "--security-opt",
                    "seccomp=unconfined",
                    "--security-opt",
                    "systempaths=unconfined",
                ],
                "timeout_s": 60,
                "lean_runtime": "image",
                "lean_cache_volume": "matharena-lean431-cache",
                "lean_toolchain_dir": "leanprover--lean4---v4.31.0",
                "read_only_rootfs": True,
            },
            "",
        )
        codex_home = solver._stage_codex_auth(workspace)
        assert codex_home is not None
        assert not codex_home.is_relative_to(workspace)

        result = solver._run_in_docker(
            workspace,
            [
                "codex",
                "sandbox",
                "--include-managed-config",
                "--permission-profile",
                "arxivlean_benchmark",
                "--cd",
                "/work",
                "--",
                "sh",
                "-c",
                probe,
            ],
            None,
            codex_home=codex_home,
        )

        assert result.returncode == 0, result.stderr
        assert (workspace / "workspace-write.txt").read_text() == "workspace-write-ok\n"
    finally:
        shutil.rmtree(root, ignore_errors=True)
