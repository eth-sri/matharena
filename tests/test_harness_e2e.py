from __future__ import annotations

import json
import os
import shutil
import socketserver
import subprocess
import threading
import time
import urllib.request
from contextlib import suppress
from functools import cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from websockets.sync.server import serve

from harness_wrapper import Agent, DEFAULT_CLI_CACHE, Model, SandboxMount
from harness_wrapper.models.oauth.openai_bridge import (
    CodexOAuthCredentials,
    CodexOAuthResponsesProxy,
)
from harness_wrapper.models.oauth import openai_oauth_config
from matharena.solvers.harness_solver import (
    CONTAINER_HARNESS_PATH,
    CONTAINER_HARNESS_VENV,
    DEFAULT_HARNESS_DOCKER_IMAGE,
    DockerHarnessSandbox,
    PodmanHarnessSandbox,
)


CONTAINER_RUNTIME = os.environ.get("MATHARENA_TEST_CONTAINER_RUNTIME", "docker")
if CONTAINER_RUNTIME not in {"docker", "podman"}:
    raise ValueError("MATHARENA_TEST_CONTAINER_RUNTIME must be docker or podman")


@cache
def _image_available() -> bool:
    try:
        return (
            subprocess.run(
                [CONTAINER_RUNTIME, "image", "inspect", DEFAULT_HARNESS_DOCKER_IMAGE],
                check=False,
                capture_output=True,
                timeout=15,
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.fixture
def docker_workspace(tmp_path):
    workspace = (
        Path.cwd() / ".pytest-harness-e2e" / f"{tmp_path.parent.name}-{tmp_path.name}"
    )
    workspace.mkdir(parents=True)
    try:
        yield workspace
    finally:
        shutil.rmtree(workspace)
        with suppress(OSError):
            workspace.parent.rmdir()



@pytest.fixture
def codex_websocket_upstream():
    """Expose the shared fake Responses handler over the OAuth WebSocket transport."""
    servers = []

    def start(http_url):
        def handler(connection):
            payload = connection.recv()
            request = urllib.request.Request(
                http_url,
                data=payload.encode() if isinstance(payload, str) else payload,
                headers={
                    "Authorization": connection.request.headers["Authorization"],
                    "Content-Type": "application/json",
                },
            )
            with urllib.request.urlopen(request, timeout=10) as response:
                for line in response:
                    if line.startswith(b"data: "):
                        event = line[6:].strip()
                        if event and event != b"[DONE]":
                            connection.send(event.decode())

        server = serve(handler, "127.0.0.1", 0)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        servers.append((server, thread))
        return f"ws://127.0.0.1:{server.socket.getsockname()[1]}/responses"

    try:
        yield start
    finally:
        for server, thread in servers:
            server.shutdown()
            thread.join(timeout=2)


_TEST_CLI_RELEASES = {
    "claude": ("claude-code", "2.1.267"),
    "codex": ("codex-cli", "0.147.0"),
    "kimi": ("kimi-code", "0.40.1"),
    "agy": ("antigravity-cli", "1.1.26"),
    "qwen": ("qwen-code", "0.23.0"),
    "opencode": ("opencode", "1.18.27"),
    "deepcode": ("deepcode", "0.3.1"),
    "grok": ("grok-build", "1.0.40"),
    "vibe": ("mistral-vibe", "2.26.1"),
}


def _sandbox(
    tmp_path, executable: str = "kimi"
) -> DockerHarnessSandbox | PodmanHarnessSandbox:
    mounts: tuple[SandboxMount, ...] = ()
    container_executable: str | PurePosixPath = executable
    release = _TEST_CLI_RELEASES.get(executable)
    if release is not None:
        release_name, version = release
        prefix = DEFAULT_CLI_CACHE / release_name / version
        binary = prefix / "bin" / executable
        if not binary.is_file():
            pytest.skip(
                f"{release_name} {version} is not cached; run a model config that pins it first"
            )
        mounts = (SandboxMount(prefix, PurePosixPath("/opt/harness-cli")),)
        container_executable = PurePosixPath("/opt/harness-cli/bin") / executable
    sandbox_class = PodmanHarnessSandbox if CONTAINER_RUNTIME == "podman" else DockerHarnessSandbox
    userns = os.environ.get("MATHARENA_CONTAINER_USERNS")
    if not userns and CONTAINER_RUNTIME == "podman" and os.getuid() != 0:
        userns = "keep-id"
    return sandbox_class(
        root=tmp_path,
        image=DEFAULT_HARNESS_DOCKER_IMAGE,
        network_enabled=False,
        container_root=PurePosixPath("/work"),
        mounts=mounts,
        container_executable=container_executable,
        extra_args=(
            "--memory",
            "2g",
            "--cpus",
            "2",
            "--pids-limit",
            "256",
            "--cap-drop",
            "ALL",
            "--user",
            f"{os.getuid()}:{os.getgid()}",
            "--read-only",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        ) + (("--userns", userns) if userns else ()),
    )


def _environment() -> dict[str, str]:
    return {
        "HOME": "/work/.harness-home",
        "VIRTUAL_ENV": CONTAINER_HARNESS_VENV,
        "XDG_CONFIG_HOME": "/work/.harness-home/.config",
        "TMPDIR": "/tmp",
        "PATH": CONTAINER_HARNESS_PATH,
    }


@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_container_sandbox_workspace_hardening_and_resource_limits(docker_workspace):
    sandbox = _sandbox(docker_workspace, "sh")
    result = sandbox.run(
        ["python", "-c", """
import errno, os
from pathlib import Path

Path('/work/persisted.txt').write_text('workspace is writable')
try:
    Path('/matharena-rootfs-write-test').write_text('unexpected')
except OSError as error:
    assert error.errno == errno.EROFS, error
else:
    raise AssertionError('root filesystem is writable')
status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines())
assert status['NoNewPrivs'].strip() == '1'
assert int(status['CapEff'].strip(), 16) == 0
assert Path('/sys/fs/cgroup/memory.max').read_text().strip() == '2147483648'
quota, period = map(int, Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert quota / period == 2
assert Path('/sys/fs/cgroup/pids.max').read_text().strip() == '256'
print(os.getuid(), os.getgid())
"""],
        env=_environment(), timeout=30,
    )
    assert result.ok, result.stderr
    assert result.stdout.strip() == f"{os.getuid()} {os.getgid()}"
    path = docker_workspace / "persisted.txt"
    assert path.stat().st_uid == os.getuid()
    resumed = sandbox.run(["cat", "/work/persisted.txt"], env=_environment(), timeout=30)
    assert resumed.ok, resumed.stderr
    assert resumed.stdout == "workspace is writable"


@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_container_sandbox_proxy_access_without_public_network(docker_workspace):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"MODEL_PROXY_OK")

        def log_message(self, *args):
            pass

    sandbox = _sandbox(docker_workspace, "sh")
    with sandbox.expose_host_service() as route:
        if route.unix_socket is not None:
            server = socketserver.ThreadingUnixStreamServer(str(route.unix_socket), Handler)
            port = route.client_port
        else:
            server = ThreadingHTTPServer((route.listen_host, 0), Handler)
            port = server.server_address[1]
        with server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = sandbox.run(
                    ["python", "-c", """
import socket, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
    assert response.read() == b'MODEL_PROXY_OK'
try:
    connection = socket.create_connection(('1.1.1.1', 443), timeout=2)
except OSError:
    pass
else:
    connection.close()
    raise AssertionError('container has public network access')
print('PROXY_ONLY_OK')
""", f"http://{route.client_host}:{port}/"],
                    env=_environment(), timeout=30,
                )
                assert result.ok, result.stderr
                assert result.stdout.strip() == "PROXY_ONLY_OK"
            finally:
                server.shutdown()
                thread.join(timeout=5)


@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_container_sandbox_stop_removes_running_child_tools(docker_workspace):
    sandbox = _sandbox(docker_workspace, "sh")
    argv, cwd, env = sandbox.prepare_command(
        ["sh", "-c", "sleep 60 & wait"], cwd=docker_workspace, env=_environment(),
    )
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            state = subprocess.run(
                [CONTAINER_RUNTIME, "inspect", "--format", "{{.State.Running}}", sandbox.container_name],
                capture_output=True, text=True, timeout=5,
            )
            if state.returncode == 0 and state.stdout.strip() == "true":
                break
            if process.poll() is not None:
                pytest.fail(f"container exited before cancellation: {process.communicate()}")
            time.sleep(0.1)
        else:
            pytest.fail("container did not start within 15 seconds")
        sandbox.stop()
        process.communicate(timeout=10)
        state = subprocess.run(
            [CONTAINER_RUNTIME, "inspect", sandbox.container_name],
            capture_output=True, timeout=5,
        )
        assert state.returncode != 0, "cancelled container still exists"
    finally:
        sandbox.stop()
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=10)


@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_scientific_python_is_the_default_container_python(docker_workspace):
    result = _sandbox(docker_workspace, "sh").run(
        [
            "sh",
            "-c",
            "command -v python; command -v python3; "
            "python -c 'import numpy, scipy, sympy, networkx, pandas, sklearn, gmpy2, z3'; "
            "python3 -c 'import numpy, scipy, sympy, networkx, pandas, sklearn, gmpy2, z3'",
        ],
        env=_environment(),
        timeout=30,
        check=True,
    )

    assert result.stdout.splitlines() == [
        f"{CONTAINER_HARNESS_VENV}/bin/python",
        f"{CONTAINER_HARNESS_VENV}/bin/python3",
    ]


@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_sagemath_works_in_read_only_offline_harness(docker_workspace):
    result = _sandbox(docker_workspace, "sh").run(
        [
            "sh", "-c",
            "set -e; "
            "sage -c 'R = PolynomialRing(QQ, names=\"x\"); x = R.gen(); "
            "assert (x**4 - 1).factor().value() == x**4 - 1; "
            "assert graphs.PetersenGraph().chromatic_number() == 3; print(\"SAGE_OK\")'; "
            "sage -python -c 'from sage.all import matrix, QQ; "
            "assert matrix(QQ, [[1,2],[3,4]]).det() == -2; print(\"SAGE_PYTHON_OK\")'; "
            "python -c 'import numpy, scipy, sympy, networkx, pandas, sklearn, gmpy2, z3; "
            "print(\"SCIENTIFIC_PYTHON_OK\")'",
        ],
        env=_environment(), timeout=60, check=False,
    )
    assert result.ok, result.stderr
    assert "SAGE_OK" in result.stdout
    assert "SAGE_PYTHON_OK" in result.stdout
    assert "SCIENTIFIC_PYTHON_OK" in result.stdout


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "accept_encoding": self.headers.get("accept-encoding"),
                "payload": payload,
            }
        )
        created = int(time.time())
        chunks = [
            {
                "id": "chatcmpl-harness-e2e",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "glm-e2e",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "E2E_OK"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-harness-e2e",
                "object": "chat.completion.chunk",
                "created": created,
                "model": "glm-e2e",
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 8,
                    "completion_tokens": 2,
                    "total_tokens": 10,
                },
            },
        ]
        body = "".join(
            f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks
        )
        body += "data: [DONE]\n\n"
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FakeGeminiHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []
    tool_override: ClassVar[dict[str, Any] | None] = None

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "path": self.path,
                "api_key": self.headers.get("x-goog-api-key"),
                "payload": payload,
            }
        )
        is_title = "conversation title generator" in json.dumps(
            payload.get("systemInstruction", "")
        )
        has_tool_result = "functionResponse" in json.dumps(payload.get("contents", []))
        if is_title:
            response_parts = [{"text": "Harness Smoke Test"}]
        elif has_tool_result:
            response_parts = [{"text": "ANTIGRAVITY_E2E_OK"}]
        else:
            response_parts = [
                {
                    "functionCall": type(self).tool_override or {
                        "name": "run_command",
                        "args": {
                            "CommandLine": 'python -c "print(6 * 7)"',
                            "Cwd": "/work",
                            "WaitMsBeforeAsync": 10000,
                            "toolAction": "Computing answer",
                            "toolSummary": "Python calculation",
                        },
                    }
                }
            ]

        chunks = [
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": response_parts,
                        },
                        "finishReason": "STOP",
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 8,
                    "candidatesTokenCount": 2,
                    "totalTokenCount": 10,
                    "cachedContentTokenCount": 3,
                },
                "modelVersion": "gemini-3.8-flash",
            }
        ]
        body = "".join(
            f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n" for chunk in chunks
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FakeCodexResponsesHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []
    text_only: ClassVar[bool] = False

    @staticmethod
    def _response(response_id: str, item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "created_at": int(time.time()),
            "status": "completed",
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "max_output_tokens": None,
            "model": "gpt-5.6-sol",
            "output": [item],
            "parallel_tool_calls": False,
            "previous_response_id": None,
            "reasoning": {"effort": "low", "summary": None},
            "store": False,
            "temperature": None,
            "text": {"format": {"type": "text"}},
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "truncation": "disabled",
            "usage": {
                "input_tokens": 8,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": 10,
            },
            "metadata": {},
        }

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "payload": payload,
            }
        )
        inputs = payload.get("input", [])
        tool_outputs = [
            item.get("output", "")
            for item in inputs
            if isinstance(item, dict) and item.get("type") == "custom_tool_call_output"
        ]
        if tool_outputs or self.text_only:
            item = {
                "id": "msg_codex_harness_e2e",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": "CODEX_E2E_OK",
                        "annotations": [],
                    }
                ],
            }
            response_id = "resp_codex_harness_e2e_final"
        else:
            item = {
                "id": "ctc_codex_harness_e2e",
                "type": "custom_tool_call",
                "status": "completed",
                "call_id": "call_codex_harness_e2e",
                "name": "exec",
                "input": (
                    "const r = await tools.exec_command({"
                    'cmd:"python -c \\"print(6 * 7)\\"",'
                    "yield_time_ms:10000,max_output_tokens:1000});text(r.output)"
                ),
            }
            response_id = "resp_codex_harness_e2e_tool"
        response = self._response(response_id, item)
        events = [
            {
                "type": "response.output_item.done",
                "sequence_number": 0,
                "output_index": 0,
                "item": item,
            },
            {
                "type": "response.completed",
                "sequence_number": 1,
                "response": response,
            },
        ]
        body = "".join(
            f"data: {json.dumps(event, separators=(',', ':'))}\n\n" for event in events
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FakeAnthropicHandler(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict[str, Any]]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("content-length", "0"))
        payload = json.loads(self.rfile.read(length))
        type(self).requests.append(
            {
                "path": self.path,
                "api_key": self.headers.get("x-api-key"),
                "payload": payload,
            }
        )
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_harness_e2e",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "claude-e2e",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 8, "output_tokens": 1},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "CLAUDE_E2E_OK"},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                "usage": {"output_tokens": 2},
            },
            {"type": "message_stop"},
        ]
        body = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event, separators=(',', ':'))}\n\n"
            for event in events
        )
        encoded = body.encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(encoded)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, format: str, *args: object) -> None:
        return


class _FakeToolFreeCodexHandler(_FakeCodexResponsesHandler):
    text_only = True


class _FakeCapacityCodexHandler(_FakeToolFreeCodexHandler):
    failed_turns_remaining: ClassVar[int] = 2

    def do_POST(self) -> None:
        # Keep failing through Codex's internal transport retries so the outer
        # wrapper must recover twice. Never contact a real provider.
        if not type(self).requests or type(self).failed_turns_remaining == 0:
            return super().do_POST()
        payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
        type(self).requests.append(
            {
                "path": self.path,
                "authorization": self.headers.get("authorization"),
                "payload": payload,
            }
        )
        response = self._response("capacity-error", {})
        response.update(
            status="failed",
            output=[],
            error={
                "code": "server_overloaded",
                "message": "Selected model is at capacity. Please try a different model.",
            },
        )
        body = (
            "data: "
            + json.dumps({"type": "response.failed", "response": response})
            + "\n\n"
        ).encode()
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        self.end_headers()
        self.wfile.write(body)


@pytest.mark.parametrize("auth", ("api", "subscription"))
@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_codex_capacity_retries_native_session_without_paid_credits(
    docker_workspace, monkeypatch, auth, codex_websocket_upstream
):
    monkeypatch.setitem(_TEST_CLI_RELEASES, "codex", ("codex-cli", "0.153.3"))
    waits = []
    monkeypatch.setattr(
        "harness_wrapper.agent.time", SimpleNamespace(sleep=waits.append)
    )
    _FakeCapacityCodexHandler.requests = []
    _FakeCapacityCodexHandler.failed_turns_remaining = 2

    def on_event(event):
        if event.type == "recovery" and event.content["reason"] == "capacity_retry":
            _FakeCapacityCodexHandler.failed_turns_remaining -= 1

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCapacityCodexHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    model = Model(
        "gpt-6-astra",
        api_url=f"{upstream_url}/v1",
        api_key="fake-api-key",
        reasoning="max",
    )
    if auth == "subscription":
        websocket_url = codex_websocket_upstream(f"{upstream_url}/responses")

        def local_bridge(**kwargs):
            return CodexOAuthResponsesProxy(
                credentials=CodexOAuthCredentials(
                    "fake-subscription-token", "fake-account"
                ),
                upstream_url=websocket_url,
                **kwargs,
            )

        monkeypatch.setattr(
            "harness_wrapper.harnesses.codex_cli.adapter.CodexOAuthResponsesProxy",
            local_bridge,
        )
        model = SimpleNamespace(
            model="gpt-6-astra",
            auth_mode="oauth",
            provider="openai",
            reasoning="max",
            supported_endpoints=lambda: frozenset({"openai"}),
            cli_environment=lambda protocol: {},
            cli_args=lambda harness: (),
        )
    try:
        agent = Agent(
            type="codex",
            model=model,
            env=_sandbox(docker_workspace, "codex"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            environment=_environment(),
            tools_enabled=False,
            auto_wait=False,
            auto_fallback=False,
            max_recovery_attempts=3,
            on_event=on_event,
        )
        agent.run("Remember CAPACITY_CONTEXT_MARKER and acknowledge it.")
        original_session = agent.session_id
        assert original_session
        events = agent.resume("Repeat that marker.", session_id=original_session)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert agent.session_id == original_session
    assert agent.model is model
    assert waits == [60, 60]
    recoveries = [event for event in events if event.type == "recovery"]
    assert [event.content["reason"] for event in recoveries] == ["capacity_retry"] * 2
    assert all(event.session_id == original_session for event in recoveries)
    assert any(
        "CODEX_E2E_OK" in str(event.content)
        for event in events
        if event.type == "message"
    )
    assert not any(event.type in {"tool_call", "tool_result"} for event in events)
    requests = _FakeCapacityCodexHandler.requests
    assert len(requests) >= 4  # Includes Codex's own transport retries.
    for request in requests:
        assert request["payload"]["model"] == "gpt-6-astra"
        assert request["payload"]["tools"] == []
        assert request["payload"]["tool_choice"] == "none"
        expected_auth = "fake-api-key" if auth == "api" else "fake-subscription-token"
        assert request["authorization"] == f"Bearer {expected_auth}"
    for request in requests[1:]:
        context = json.dumps(request["payload"]["input"])
        assert context.count("CAPACITY_CONTEXT_MARKER") == 1
        assert "CODEX_E2E_OK" in context  # Prior assistant output was retained.
        assert "Repeat that marker." in context
    assert "Continue the interrupted task" in json.dumps(
        requests[-1]["payload"]["input"]
    )
    recorded = [
        json.loads(line)["event"]["content"]
        for line in agent.model_request_log_path.read_text().splitlines()
    ]
    assert recorded == [request["payload"] for request in requests]


@pytest.mark.parametrize("auth", ("api", "subscription"))
@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_codex_astra_tool_free_start_and_resume_without_paid_credits(
    docker_workspace, monkeypatch, auth, codex_websocket_upstream
):
    # Test the new config's actual pinned CLI, with fake tokens and local HTTP/WebSocket servers.
    monkeypatch.setitem(_TEST_CLI_RELEASES, "codex", ("codex-cli", "0.153.3"))
    _FakeToolFreeCodexHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeToolFreeCodexHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    upstream_url = f"http://127.0.0.1:{upstream.server_address[1]}"
    model = Model(
        "gpt-6-astra",
        api_url=f"{upstream_url}/v1",
        api_key="child-placeholder",
        headers={"Authorization": "Bearer host-only-e2e-key"},
        reasoning="max",
        request_overrides={
            "tools": [{"type": "web_search"}],
            "tool_choice": "required",
        },
    )
    if auth == "subscription":
        websocket_url = codex_websocket_upstream(f"{upstream_url}/responses")

        def local_bridge(**kwargs):
            return CodexOAuthResponsesProxy(
                credentials=CodexOAuthCredentials(
                    "fake-subscription-token", "fake-account"
                ),
                upstream_url=websocket_url,
                **kwargs,
            )

        monkeypatch.setattr(
            "harness_wrapper.harnesses.codex_cli.adapter.CodexOAuthResponsesProxy",
            local_bridge,
        )
        # This fake OAuth model deliberately has no real credential or discovery hooks.
        model = SimpleNamespace(
            model="gpt-6-astra",
            auth_mode="oauth",
            provider="openai",
            reasoning="max",
            supported_endpoints=lambda: frozenset({"openai"}),
            cli_environment=lambda protocol: {},
            cli_args=lambda harness: (),
        )
    try:
        agent = Agent(
            type="codex",
            model=model,
            env=_sandbox(docker_workspace, "codex"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            environment=_environment(),
            minimal_context=False,
            tools_enabled=False,
            model_context_window=1050000,
            validate_version=False,
            auto_wait=False,
            auto_fallback=False,
            max_recovery_attempts=0,
        )
        events = agent.run("Compute 6 * 7 and give the answer.")
        assert agent.session_id
        events += agent.resume("Repeat the answer.", session_id=agent.session_id)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("CODEX_E2E_OK" in str(event.content) for event in events)
    assert not any(
        event.type in {"tool_call", "tool_result", "error"} for event in events
    ), [
        (event.type, event.content)
        for event in events
        if event.type in {"tool_call", "tool_result", "error"}
    ]
    requests = _FakeToolFreeCodexHandler.requests
    assert len(requests) == 2
    for request in requests:
        payload = request["payload"]
        assert payload["model"] == "gpt-6-astra"
        assert payload["reasoning"]["effort"] == "max"
        assert payload["tools"] == []
        assert payload["tool_choice"] == "none"
        assert payload["parallel_tool_calls"] is False
        assert "No tools are available" in payload["instructions"]
        assert "Use the available shell tools" not in payload["instructions"]
        request_context = json.dumps(payload.get("input", []))
        assert "functions.exec" not in request_context
        assert "<skills_instructions>" not in request_context
    recorded = [
        json.loads(line)["event"]["content"]
        for line in agent.model_request_log_path.read_text().splitlines()
    ]
    assert recorded == [request["payload"] for request in requests]
    catalog = json.loads(
        (docker_workspace / ".harness_wrapper/tool-free-models.json").read_text()
    )
    assert catalog["models"][0]["context_window"] == 1050000
    assert catalog["models"][0]["tool_mode"] == "direct"


@pytest.mark.parametrize(
    ("cli_version", "model_name"),
    [("0.147.0", "gpt-5.6-sol"), ("0.153.3", "gpt-6-astra")],
)
@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_codex_code_mode_executes_python_without_paid_credits(
    docker_workspace, monkeypatch, cli_version, model_name
):
    monkeypatch.setitem(_TEST_CLI_RELEASES, "codex", ("codex-cli", cli_version))
    _FakeCodexResponsesHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeCodexResponsesHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        model_name,
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        api_key="child-placeholder",
        headers={"Authorization": "Bearer host-only-e2e-key"},
        reasoning="low",
    )
    try:
        agent = Agent(
            type="codex",
            model=model,
            env=_sandbox(docker_workspace, "codex"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment=_environment(),
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run("Use Python to compute 6 * 7, then reply CODEX_E2E_OK.")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("CODEX_E2E_OK" in str(event.content) for event in events)
    assert any(
        event.type == "tool_result" and "42" in str(event.content) for event in events
    )
    assert len(_FakeCodexResponsesHandler.requests) == 2
    assert all(
        request["path"].endswith("/responses")
        and request["authorization"] == "Bearer host-only-e2e-key"
        for request in _FakeCodexResponsesHandler.requests
    )
    second_input = _FakeCodexResponsesHandler.requests[1]["payload"]["input"]
    assert any(
        item.get("type") == "custom_tool_call_output"
        and "42" in json.dumps(item.get("output", ""))
        for item in second_input
        if isinstance(item, dict)
    )
    assert not any(event.type == "error" for event in events), [
        (event.type, event.content) for event in events if event.type == "error"
    ]


@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_kimi_container_proxy_end_to_end_without_paid_credits(docker_workspace):
    _FakeOpenAIHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "glm-e2e",
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        api_key="child-placeholder",
        headers={"Authorization": "Bearer host-only-e2e-key"},
        request_overrides={"thinking": {"type": "enabled"}},
        request_drop_fields=("prompt_cache_key",),
    )
    try:
        agent = Agent(
            type="kimi",
            model=model,
            env=_sandbox(docker_workspace),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment=_environment(),
            validate_version=False,
        )
        events = agent.run("Reply exactly E2E_OK. Do not use tools.")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("E2E_OK" in str(event.content) for event in events)
    assert _FakeOpenAIHandler.requests
    request = _FakeOpenAIHandler.requests[0]
    assert request["path"].endswith("/chat/completions")
    assert request["authorization"] == "Bearer host-only-e2e-key"
    assert request["accept_encoding"] != "gzip"
    assert "prompt_cache_key" not in request["payload"]
    assert request["payload"]["thinking"] == {"type": "enabled"}


@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
@pytest.mark.parametrize("read_library_source", [False, True])
def test_antigravity_cli_container_proxy_end_to_end_without_paid_credits(
    docker_workspace, monkeypatch, read_library_source,
):
    if read_library_source:
        monkeypatch.setattr(_FakeGeminiHandler, "tool_override", {
            "name": "view_file",
            "args": {
                "AbsolutePath": "/usr/lib/python3/dist-packages/sage/combinat/matrices/hadamard_matrix.py",
                "StartLine": 1,
                "EndLine": 15,
                "toolAction": "Read installed Sage source",
                "toolSummary": "Check library documentation",
            },
        })
    _FakeGeminiHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeGeminiHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "gemini-3.8-flash",
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}",
        api_key="child-placeholder",
        api_type="gemini",
        headers={"x-goog-api-key": "host-only-e2e-key"},
        request_overrides={
            "generationConfig": {
                "temperature": 0.25,
                "maxOutputTokens": 64,
            }
        },
        reasoning="high",
    )
    try:
        agent = Agent(
            type="gravity",
            model=model,
            env=_sandbox(docker_workspace, "agy"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment=_environment(),
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run(
            "Use Python to compute 6 * 7, then reply ANTIGRAVITY_E2E_OK."
        )
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any(
        event.type == "message" and "ANTIGRAVITY_E2E_OK" in str(event.content)
        for event in events
    )
    assert _FakeGeminiHandler.requests
    # view_file streams a size summary; the source itself goes into CLI context.
    expected_tool_output = "lines, " if read_library_source else "42"
    assert any(
        event.type == "tool_result" and expected_tool_output in str(event.content) for event in events
    ), [event.to_dict() for event in events if event.type == "tool_call"]
    assert events[-1].type == "result" and events[-1].content.strip() == "ANTIGRAVITY_E2E_OK"
    assert not any(event.type in {"error", "recovery"} for event in events)
    assert len(_FakeGeminiHandler.requests) == 3
    request_log = agent.model_request_log_path
    assert request_log is not None
    persisted_requests = [
        json.loads(line)
        for line in request_log.read_text(encoding="utf-8").splitlines()
    ]
    assert len(persisted_requests) == 3
    # Title generation and the main request run concurrently. Capture occurs
    # before forwarding, so upstream arrival order may differ from log order.
    assert sorted(
        json.dumps(record["event"]["content"], sort_keys=True) for record in persisted_requests
    ) == sorted(
        json.dumps(request["payload"], sort_keys=True) for request in _FakeGeminiHandler.requests
    )
    assert all("headers" not in record for record in persisted_requests)
    model_requests = [
        item
        for item in _FakeGeminiHandler.requests
        if "gemini-3.8-flash" in item["path"]
    ]
    request = model_requests[0]
    assert request["path"].startswith(
        "/v1beta/models/gemini-3.8-flash:streamGenerateContent"
    )
    assert request["api_key"] == "host-only-e2e-key"
    assert request["payload"]["generationConfig"]["temperature"] == 0.25
    assert request["payload"]["generationConfig"]["maxOutputTokens"] == 64
    tool_names = {
        declaration.get("name")
        for group in request["payload"].get("tools", [])
        for declaration in group.get("functionDeclarations", [])
        if isinstance(group, dict) and isinstance(declaration, dict)
    }
    assert "search_web" not in tool_names
    assert "read_url_content" not in tool_names
    assert "functionResponse" in json.dumps(model_requests[1]["payload"]["contents"])

    assert "invoke_subagent" not in tool_names

    assert agent.get_tokens().input_tokens == 24
    assert agent.get_tokens().output_tokens == 6
    assert agent.get_tokens().cache_read_tokens == 9


@pytest.mark.parametrize(
    ("harness", "executable"),
    [("qwen", "qwen"), ("opencode", "opencode")],
)
@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_additional_openai_cli_container_proxy_end_to_end_without_paid_credits(
    docker_workspace, harness, executable
):
    _FakeOpenAIHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "glm-e2e",
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}/v1",
        api_key="child-placeholder",
        headers={"Authorization": "Bearer host-only-e2e-key"},
        request_overrides={"temperature": 0.3, "max_tokens": 64},
    )
    try:
        agent = Agent(
            type=harness,
            model=model,
            env=_sandbox(docker_workspace, executable),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment={
                **_environment(),
                "XDG_CACHE_HOME": "/work/.harness-home/.cache",
                "XDG_DATA_HOME": "/work/.harness-home/.local/share",
                "XDG_STATE_HOME": "/work/.harness-home/.local/state",
            },
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run("Reply exactly E2E_OK. Do not use tools.")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("E2E_OK" in str(event.content) for event in events)
    assert _FakeOpenAIHandler.requests
    request_summaries = [
        {
            "model": item["payload"].get("model"),
            "messages": [
                {
                    "role": message.get("role"),
                    "content": str(message.get("content"))[:500],
                }
                for message in item["payload"].get("messages", [])
            ],
            "tool_names": [
                tool.get("function", {}).get("name")
                for tool in item["payload"].get("tools", [])
                if isinstance(tool, dict)
            ],
        }
        for item in _FakeOpenAIHandler.requests
    ]
    assert len(_FakeOpenAIHandler.requests) == 1, request_summaries
    request = _FakeOpenAIHandler.requests[0]
    assert request["path"].endswith("/chat/completions")
    assert request["authorization"] == "Bearer host-only-e2e-key"
    assert request["payload"]["temperature"] == 0.3
    assert request["payload"]["max_tokens"] == 64
    tool_names = {
        tool.get("function", {}).get("name")
        for tool in request["payload"].get("tools", [])
        if isinstance(tool, dict)
    }
    assert tool_names.isdisjoint({"agent", "task", "skill", "web_fetch", "web_search"})

    assert agent.get_tokens().input_tokens == 8
    assert agent.get_tokens().output_tokens == 2


@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_qwen_anthropic_container_proxy_end_to_end_without_paid_credits(
    docker_workspace,
):
    _FakeAnthropicHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAnthropicHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "claude-e2e",
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}",
        api_key="child-placeholder",
        api_type="anthropic",
        headers={"x-api-key": "host-only-e2e-key"},
    )
    try:
        agent = Agent(
            type="qwen",
            model=model,
            env=_sandbox(docker_workspace, "qwen"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment={
                **_environment(),
                "XDG_CACHE_HOME": "/work/.harness-home/.cache",
                "XDG_DATA_HOME": "/work/.harness-home/.local/share",
                "XDG_STATE_HOME": "/work/.harness-home/.local/state",
            },
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run("Reply exactly CLAUDE_E2E_OK. Do not use tools.")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("CLAUDE_E2E_OK" in str(event.content) for event in events)
    assert len(_FakeAnthropicHandler.requests) == 1
    request = _FakeAnthropicHandler.requests[0]
    assert request["path"].endswith("/v1/messages")
    assert request["api_key"] == "host-only-e2e-key"
    assert agent.get_tokens().input_tokens == 8
    assert agent.get_tokens().output_tokens == 2


@pytest.mark.skipif(
    not _image_available(),
    reason=f"{DEFAULT_HARNESS_DOCKER_IMAGE} is not built",
)
def test_claude_container_proxy_end_to_end_without_paid_credits(docker_workspace):
    _FakeAnthropicHandler.requests = []
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), _FakeAnthropicHandler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    model = Model(
        "claude-e2e",
        api_url=f"http://127.0.0.1:{upstream.server_address[1]}",
        api_key="child-placeholder",
        api_type="anthropic",
        headers={"x-api-key": "host-only-e2e-key"},
    )
    try:
        agent = Agent(
            type="claude",
            model=model,
            env=_sandbox(docker_workspace, "claude"),
            dir=docker_workspace,
            executable="/usr/bin/true",
            subagents={},
            environment={
                **_environment(),
                "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
            },
            minimal_context=True,
            validate_version=False,
        )
        events = agent.run("Reply exactly CLAUDE_E2E_OK. Do not use tools.")
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)

    assert any("CLAUDE_E2E_OK" in str(event.content) for event in events)
    assert _FakeAnthropicHandler.requests
    request = _FakeAnthropicHandler.requests[0]
    assert request["path"].startswith("/v1/messages")
    assert request["api_key"] == "host-only-e2e-key"
    assert [tool["name"] for tool in request["payload"]["tools"]] == ["Bash"]


@pytest.mark.skipif(
    os.getenv("MATHARENA_RUN_PAID_HARNESS_E2E") != "1"
    or not os.getenv("GLM_API_KEY")
    or not _image_available(),
    reason="set MATHARENA_RUN_PAID_HARNESS_E2E=1 and GLM_API_KEY after building the image",
)
def test_kimi_glm_api_credits_live_end_to_end(docker_workspace):
    model = Model(
        "glm-5.3",
        api_url="https://api.z.ai/api/paas/v4/",
        api_key="child-placeholder",
        headers={"Authorization": f"Bearer {os.environ['GLM_API_KEY']}"},
        request_overrides={"thinking": {"type": "enabled"}, "max_tokens": 128},
        reasoning="max",
    )
    agent = Agent(
        type="kimi",
        model=model,
        env=_sandbox(docker_workspace),
        dir=docker_workspace,
        executable="/usr/bin/true",
        subagents={},
        environment={
            **_environment(),
            "KIMI_MODEL_MAX_CONTEXT_SIZE": "131072",
        },
        validate_version=False,
    )

    events = agent.run("Reply with the exact text GLM_E2E_OK. Do not use tools.")

    assert any("GLM_E2E_OK" in str(event.content) for event in events)


@pytest.mark.skipif(
    os.getenv("MATHARENA_RUN_PAID_HARNESS_E2E") != "1"
    or not os.getenv("GOOGLE_API_KEY")
    or not _image_available(),
    reason=(
        "set MATHARENA_RUN_PAID_HARNESS_E2E=1 and GOOGLE_API_KEY after building the image"
    ),
)
def test_kimi_gemini_api_credits_live_end_to_end(docker_workspace):
    model = Model(
        "gemini-3.8-flash",
        api_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key="child-placeholder",
        headers={"Authorization": f"Bearer {os.environ['GOOGLE_API_KEY']}"},
        request_overrides={
            "max_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.9,
            "tool_choice": "auto",
        },
        request_drop_fields=("prompt_cache_key",),
        reasoning="low",
    )
    agent = Agent(
        type="kimi",
        model=model,
        env=_sandbox(docker_workspace),
        dir=docker_workspace,
        executable="/usr/bin/true",
        subagents={},
        environment={
            **_environment(),
            "KIMI_MODEL_MAX_CONTEXT_SIZE": "1048576",
        },
        minimal_context=True,
        validate_version=False,
        max_recovery_attempts=0,
    )

    events = agent.run(
        "You must use Bash to run python3 -c 'print(6 * 7)'. "
        "After the command succeeds, reply with the exact text GEMINI_E2E_OK."
    )

    assert any(event.type == "tool_call" for event in events)
    assert any(
        event.type == "tool_result" and "42" in str(event.content) for event in events
    )
    assert any("GEMINI_E2E_OK" in str(event.content) for event in events)
    usage = agent.get_tokens()
    assert usage.input_tokens > 0
    assert usage.output_tokens > 0


@pytest.mark.skipif(
    os.getenv("MATHARENA_RUN_SUBSCRIPTION_E2E") != "1" or not _image_available(),
    reason="set MATHARENA_RUN_SUBSCRIPTION_E2E=1 with a saved host Codex login",
)
def test_codex_subscription_live_end_to_end(docker_workspace):
    sandbox = _sandbox(docker_workspace, "codex")
    model = Model(
        "gpt-5.6-sol",
        oauth="openai",
        oauth_config=openai_oauth_config(
            device_auth=True,
            auto_login=False,
            auto_relogin=False,
        ),
        reasoning="low",
    )
    agent = Agent(
        type="codex",
        model=model,
        env=sandbox,
        dir=docker_workspace,
        executable="/usr/bin/true",
        subagents={},
        environment=_environment(),
        validate_version=False,
        auto_wait=False,
    )

    events = agent.run(
        "Use the shell to run `printf CODEX_SHELL_OK`. After the command succeeds, "
        "reply with the exact text CODEX_SUBSCRIPTION_E2E_OK."
    )

    assert any("CODEX_SUBSCRIPTION_E2E_OK" in str(event.content) for event in events)
    tool_results = [event for event in events if event.type == "tool_result"]
    assert any("CODEX_SHELL_OK" in str(event.content) for event in tool_results)


@pytest.mark.skipif(not _image_available(), reason="selected harness runtime image is not built")
def test_codex_openrouter_web_with_subagent_and_resume(docker_workspace, monkeypatch):
    from harness_wrapper.harnesses.codex_cli import adapter
    from harness_wrapper.models.request_capture import RequestCaptureProxy

    monkeypatch.setitem(_TEST_CLI_RELEASES, "codex", ("codex-cli", "0.159.0"))

    class Handler(_FakeCodexResponsesHandler):
        requests = []
        root_calls = 0

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["content-length"])))
            self.requests.append({"payload": payload, "path": self.path})
            meta = json.loads(payload["client_metadata"]["x-codex-turn-metadata"])
            child = meta.get("agent_name", "/root") != "/root"
            if not child:
                type(self).root_calls += 1
            if not child and self.root_calls <= 2:
                name = "spawn_agent" if self.root_calls == 1 else "wait_agent"
                arguments = (
                    {"task_name": "web_check", "message": "Check the search results.", "agent_type": "worker", "fork_turns": "none"}
                    if self.root_calls == 1 else {"timeout_ms": 10000}
                )
                item = {"id": "fc_" + name, "type": "function_call", "status": "completed",
                        "call_id": "call_" + name, "name": name, "namespace": "collaboration",
                        "arguments": json.dumps(arguments)}
            elif not child and self.root_calls == 3:
                item = {"id": "ctc_shell", "type": "custom_tool_call", "status": "completed",
                        "call_id": "call_shell", "name": "exec", "namespace": "functions",
                        "input": 'text(await tools.exec_command({cmd:"printf HOSTED_WEB_SHELL_OK"}));'}
            else:
                item = {"id": "msg_done", "type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": "HOSTED_WEB_OK", "annotations": []}]}
            items = [item]
            if child or self.root_calls == 1:
                # Simulate a provider-executed search before its next client tool/final answer.
                items.insert(0, {"id": "ws_child" if child else "ws_root", "type": "web_search_call",
                                 "status": "completed", "action": {"type": "search", "query": "MathArena"}})
            response = self._response("resp_" + str(len(self.requests)), item)
            response["output"] = items
            events = [{"type": "response.output_item.done", "sequence_number": i,
                       "output_index": i, "item": v} for i, v in enumerate(items)]
            events.append({"type": "response.completed", "sequence_number": len(events), "response": response})
            body = "".join("data: " + json.dumps(event) + "\n\n" for event in events).encode()
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()

    def local_proxy(_url, observer, **kwargs):
        return RequestCaptureProxy(f"http://127.0.0.1:{upstream.server_port}/v1", observer, **kwargs)

    monkeypatch.setattr(adapter, "RequestCaptureProxy", local_proxy)
    sandbox = _sandbox(docker_workspace, "codex")
    sandbox.network_enabled = True
    agent = Agent(
        type="codex", model=Model("openai/gpt-6-astra", api_url="https://openrouter.ai/api/v1",
                                  api_key="local-test-only", reasoning="ultra"),
        env=sandbox, dir=docker_workspace, executable="/usr/bin/true", environment=_environment(),
        minimal_context=True, model_context_window=1050000, subagents={"worker": "openai/gpt-6-astra"},
        auto_wait=False, auto_fallback=False, max_recovery_attempts=0,
    )
    timer = threading.Timer(45, sandbox.stop)
    timer.start()
    try:
        events = agent.run("Search, delegate, and check the shell.")
        events += agent.resume("Continue the same task.")
    finally:
        timer.cancel()
        sandbox.stop()
        agent.terminate()
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=2)
    assert any(e.type == "tool_result" and "HOSTED_WEB_SHELL_OK" in str(e.content) for e in events)
    assert any(e.type == "result" and "HOSTED_WEB_OK" in str(e.content) for e in events)
    assert any(e.tool_name == "web_search" for e in events)
    names = set()
    for request in Handler.requests:
        payload = request["payload"]
        assert request["path"] == "/v1/responses"
        assert payload["tools"] == [{"type": "web_search", "external_web_access": True}]
        meta = json.loads(payload["client_metadata"]["x-codex-turn-metadata"])
        names.add(meta["agent_name"])
        assert meta["reasoning_effort"] == "ultra"
    assert names == {"/root", "/root/web_check"}
    assert "ws_root" in json.dumps(Handler.requests[-1]["payload"]["input"])
