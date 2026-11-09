"""Opt-in real Podman/Codex test with a local fake Responses provider.

Set LEANLEAN_MONITOR_TEST_RUNTIME to a prepared bundle's eval-runtime.json,
LEANLEAN_MONITOR_TEST_CLI_CACHE to a cache containing codex-cli 0.153.3, and
optionally LEANLEAN_MONITOR_TEST_OUTPUT to retain the artifacts somewhere fixed.
No API credentials or paid model requests are used.
"""

import json
import os
import shlex
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_harness_e2e import _FakeCodexResponsesHandler

from harness_wrapper import Model
from leanlean.evaluation import load_problems
from leanlean.solver import LeanLeanSolver


class TelemetryCodexHandler(_FakeCodexResponsesHandler):
    command = ""

    @staticmethod
    def _response(response_id, item):
        if item["type"] == "custom_tool_call":
            command = TelemetryCodexHandler.command
            item["input"] = (
                "let r = await tools.exec_command("
                + json.dumps(
                    {
                        "cmd": command,
                        "yield_time_ms": 1000,
                        "max_output_tokens": 1000,
                    }
                )
                + "); text(r); "
                "while (r.session_id) {r = await tools.write_stdin({"
                'session_id:r.session_id,chars:"",yield_time_ms:1000,max_output_tokens:1000});text(r);}'
            )
        return _FakeCodexResponsesHandler._response(response_id, item)


@pytest.mark.skipif(
    not os.environ.get("LEANLEAN_MONITOR_TEST_RUNTIME"), reason="requires prepared LeanLean Podman runtime"
)
def test_codex_nested_exec_timestamps_and_source_history(tmp_path, monkeypatch):
    runtime = Path(os.environ["LEANLEAN_MONITOR_TEST_RUNTIME"]).resolve()
    root = Path(os.environ.get("LEANLEAN_MONITOR_TEST_OUTPUT", tmp_path)) / uuid.uuid4().hex
    config = {
        "model": "gpt-6-astra",
        "harness": "codex",
        "harness_version": "0.153.3",
        "harness_config": {
            "repo_eval": True,
            "monitoring": {"diff": True, "cpu": True, "ram": True, "commands": True},
            "command_monitoring_backend": "wrapper",
            "runtime_manifest": str(runtime),
            "workspace_root": str(root / "workspace"),
            "cli_cache_root": os.environ.get("LEANLEAN_MONITOR_TEST_CLI_CACHE"),
            "memory_gb": 64,
            "cpu_limit": 8,
            "max_time_seconds": 180,
            "environment": {"LEAN_NUM_THREADS": "8"},
        },
    }
    solver = LeanLeanSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None}, "{problem}", config, ""
    )
    handler = TelemetryCodexHandler
    handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "gpt-6-astra",
        api_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
        api_key="local-test-only",
        reasoning="low",
    )
    monkeypatch.setattr(solver, "_build_model", lambda: model)
    problem = load_problems(str(runtime.parent), [1])[0]
    entry = problem["leanlean_project"]
    project = Path("/work/repository") / entry["project_path"]
    solution = Path("/work/repository") / entry["solution_path"]
    handler.command = (
        f"cd {shlex.quote(str(project))} && printf '\\n-- codex telemetry test\\n' >> {shlex.quote(str(solution))} "
        "&& sleep 2 && lake -q build"
    )
    try:
        response = solver._solve_one(0, problem, None, 1, 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert "CODEX_E2E_OK" in response.conversation[-1]["content"]
    assert len(handler.requests) == 2
    output = Path(response.history[0]["monitoring"]["directory"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["complete"], summary
    assert summary["totals"]["model"]["lake_build"]["commands"] >= 1
    model_spans = [span for span in summary["spans"] if span["start"]["phase"] == "model"]
    assert any("codex telemetry test" in str(span["start"]["argv"]) for span in model_spans)
    assert all(span["status"] == "complete" for span in model_spans)
    checkpoints = [json.loads(line) for line in (output / "checkpoints.jsonl").read_text().splitlines()]
    assert len(checkpoints) >= 2
    assert all(row["changed"] for row in checkpoints)
    observations = [json.loads(line) for line in (output / "observations.jsonl").read_text().splitlines()]
    assert len(observations) > len(checkpoints)
    resources = [json.loads(line) for line in (output / "resources.jsonl").read_text().splitlines()]
    samples = [row for row in resources if row["status"] == "sample"]
    assert {row["phase"] for row in samples} >= {"preflight", "model"}
    assert len({row["container_id"] for row in samples}) >= 2
    assert all(row["memory_current_bytes"] > 0 for row in samples)
    assert any((row["cpu_percent"] or 0) > 0 for row in samples if row["phase"] == "model")
    actions = json.loads((output / "actions.json").read_text())
    assert actions["complete"] and actions["spans"]
    assert any(row["duration_s"] >= 2 for row in actions["spans"])
    assert all(row["started_wall_ns"] <= row["ended_wall_ns"] for row in actions["spans"])
    # A periodic capture may already have recorded the edit while lake builds.
    # The action must still be observed and refer to the preserved changed tree.
    tool_results = [row for row in observations if row["reason"] == "harness_tool_result"]
    changed_trees = {row["source_sha256"] for row in checkpoints[1:]}
    assert any(row["source_sha256"] in changed_trees for row in tool_results)
    assert not any(row["reason"].startswith("command_") for row in checkpoints)
    assert json.loads((output / "attempt.json").read_text())["interval_s"] == 30
    assert list((output / "native").rglob("rollout-*.jsonl"))
    assert (output / "harness-events.jsonl").stat().st_size > 0
    (root / "validation.json").write_text(
        json.dumps({"monitoring": str(output), "summary": summary["totals"]}, indent=2) + "\n"
    )
    print(f"Preserved Codex validation: {root}")


@pytest.mark.skipif(
    not os.environ.get("LEANLEAN_MONITOR_TEST_RUNTIME"), reason="requires prepared LeanLean Podman runtime"
)
@pytest.mark.parametrize("version", ["v4.35.0-rc2", "v4.35.0-rc3"])
def test_bare_native_comparator_with_bubblewrap(tmp_path, version):
    """Exercise both October toolchains, including NanoDa and a rejected proof."""
    from pathlib import PurePosixPath
    from harness_wrapper import SandboxMount

    toolchain = "leanprover--lean4---" + version
    source = Path.home() / ".elan/toolchains" / toolchain
    if not source.is_dir():
        pytest.skip(f"requires cached {version}")
    runtime = Path(os.environ["LEANLEAN_MONITOR_TEST_RUNTIME"]).resolve()
    root = Path(os.environ.get("LEANLEAN_MONITOR_TEST_OUTPUT", tmp_path)) / ("comparator-" + uuid.uuid4().hex)
    root.mkdir(parents=True)
    (root / "lean-toolchain").write_text("leanprover/lean4:" + version + "\n")
    (root / "lakefile.toml").write_text(
        'name = "ComparatorSmoke"\nversion = "0.1.0"\ndefaultTargets = ["Challenge", "Solution"]\n'
        '[[lean_lib]]\nname = "Challenge"\n[[lean_lib]]\nname = "Solution"\n'
    )
    (root / "Challenge.lean").write_text("theorem demo : True := by sorry\n")
    (root / "Solution.lean").write_text("theorem demo : True := True.intro\n")
    (root / "comparator.json").write_text(
        json.dumps(
            {
                "challenge_module": "Challenge",
                "solution_module": "Solution",
                "theorem_names": ["demo"],
                "permitted_axioms": ["propext", "Quot.sound", "Classical.choice"],
                "enable_nanoda": True,
            }
        )
    )
    config = {
        "model": "test",
        "harness": "codex",
        "harness_config": {"leanlean_projects": True, "runtime_manifest": str(runtime), "managed_cli": False},
    }
    solver = LeanLeanSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None}, "{problem}", config, ""
    )
    sandbox = solver._build_sandbox(root)
    target = PurePosixPath("/opt/elan/toolchains") / toolchain
    sandbox.mounts = (*sandbox.mounts, SandboxMount(source, target))
    environment = solver._agent_environment()
    environment["PATH"] = f"/opt/elan/bin:{target}/bin:" + environment["PATH"]
    outcomes = []
    try:
        # Reproduce Lake's env -i plus --clearenv, checking inside its sandbox.
        result = sandbox.run(
            [
                "env",
                "-i",
                "/usr/local/bin/bwrap",
                "--ro-bind",
                "/",
                "/",
                "--unshare-all",
                "--clearenv",
                "--",
                "/usr/bin/printenv",
                "LEAN_NUM_THREADS",
            ],
            env=environment,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "8"
        for command, expected in [(["lake", "build"], 0), (["lake", "comparator"], 0)]:
            result = sandbox.run(command, env=environment, timeout=120)
            outcomes.append(
                {"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
            )
            assert result.returncode == expected, result.stdout + result.stderr
        (root / "Solution.lean").write_text("theorem demo : True := by sorry\n")
        result = sandbox.run(["lake", "comparator"], env=environment, timeout=120)
        outcomes.append(
            {
                "command": ["lake", "comparator"],
                "invalid_proof": True,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        )
        assert result.returncode == 1, result.stdout + result.stderr
        assert "sorryAx" in result.stdout + result.stderr
        assert all("Sandbox disabled" not in row["stdout"] + row["stderr"] for row in outcomes)
    finally:
        sandbox.stop()
        (root / "validation.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    print(f"Preserved native Comparator validation ({version}): {root}")


class TelemetryClaudeHandler(BaseHTTPRequestHandler):
    """Local Anthropic stream that asks the real Claude CLI to run one Bash tool."""

    shell_command = ""
    requests = []

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(payload)
        if "count_tokens" in self.path:
            body = json.dumps({"input_tokens": 100}).encode()
            content_type = "application/json"
        else:
            finished = any(
                isinstance(block, dict) and block.get("type") == "tool_result"
                for message in payload.get("messages", [])
                for block in (message.get("content", []) if isinstance(message.get("content"), list) else [])
            )
            block = (
                {"type": "text", "text": ""}
                if finished
                else {"type": "tool_use", "id": "lake_trace_test", "name": "Bash", "input": {}}
            )
            delta = (
                {"type": "text_delta", "text": "CLAUDE_TELEMETRY_OK"}
                if finished
                else {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(
                        {
                            "command": self.shell_command,
                            "timeout": 120000,
                            "description": "Build and compare Lean proofs",
                        }
                    ),
                }
            )
            events = [
                {
                    "type": "message_start",
                    "message": {
                        "id": "msg_trace_test",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-e2e",
                        "stop_reason": None,
                        "stop_sequence": None,
                        "usage": {"input_tokens": 100, "output_tokens": 1},
                    },
                },
                {"type": "content_block_start", "index": 0, "content_block": block},
                {"type": "content_block_delta", "index": 0, "delta": delta},
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn" if finished else "tool_use", "stop_sequence": None},
                    "usage": {"output_tokens": 20},
                },
                {"type": "message_stop"},
            ]
            body = "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()
            content_type = "text/event-stream"
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.mark.skipif(
    not os.environ.get("LEANLEAN_MONITOR_TEST_RUNTIME"), reason="requires prepared LeanLean Podman runtime"
)
def test_claude_build_and_comparator_timestamps(tmp_path, monkeypatch):
    runtime = Path(os.environ["LEANLEAN_MONITOR_TEST_RUNTIME"]).resolve()
    root = Path(os.environ.get("LEANLEAN_MONITOR_TEST_OUTPUT", tmp_path)) / ("claude-" + uuid.uuid4().hex)
    config = {
        "model": "claude-e2e",
        "harness": "claude",
        "harness_version": "2.1.267",
        "harness_config": {
            "repo_eval": True,
            "monitoring": {"diff": True, "cpu": True, "ram": True, "commands": True},
            "command_monitoring_backend": "wrapper",
            "runtime_manifest": str(runtime),
            "workspace_root": str(root / "workspace"),
            "cli_cache_root": os.environ.get("LEANLEAN_MONITOR_TEST_CLI_CACHE"),
            "memory_gb": 64,
            "cpu_limit": 8,
            "max_time_seconds": 300,
            "max_recovery_attempts": 0,
            "environment": {"LEAN_NUM_THREADS": "8", "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"},
        },
    }
    solver = LeanLeanSolver(
        {"type": "pure_model", "model_config": config, "scaffold_config": None}, "{problem}", config, ""
    )
    handler = TelemetryClaudeHandler
    handler.requests = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "claude-e2e", api_url=f"http://127.0.0.1:{server.server_port}", api_key="local-test-only", api_type="anthropic"
    )
    monkeypatch.setattr(solver, "_build_model", lambda: model)
    problem = load_problems(str(runtime.parent), [1])[0]
    project = Path("/work/repository") / problem["leanlean_project"]["project_path"]
    handler.shell_command = f"cd {shlex.quote(str(project))} && lake build && lake comparator"
    try:
        response = solver._solve_one(0, problem, None, 1, 0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    output = Path(response.history[0]["monitoring"]["directory"])
    summary = json.loads((output / "summary.json").read_text())
    assert "CLAUDE_TELEMETRY_OK" in response.conversation[-1]["content"]
    assert summary["complete"], summary.get("errors")
    lake = [s for s in summary["spans"] if s["start"]["phase"] == "model" and s["start"]["kind"] == "lake"]
    for command in ("build", "comparator"):
        matches = [s for s in lake if s.get("lake_subcommand") == command]
        assert len(matches) == 1, lake
        span = matches[0]
        assert span["status"] == "complete" and span["end"]["returncode"] == 0
        assert span["duration_s"] > 0
        assert span["execution_started_wall_ns"] < span["end"]["wall_time_ns"]
    actions = json.loads((output / "actions.json").read_text())
    assert any(s["tool_name"] == "Bash" and s["status"] == "complete" for s in actions["spans"])
    (root / "validation.json").write_text(json.dumps({"monitoring": str(output), "lake_spans": lake}, indent=2) + "\n")
    print(f"Preserved Claude validation: {root}")
