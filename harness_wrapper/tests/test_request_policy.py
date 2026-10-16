from __future__ import annotations

import io
import json
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import Mock

import pytest

from harness_wrapper.agent import Agent
from harness_wrapper.models import api_header_bridge
from harness_wrapper.models.api_header_bridge import APIHeaderProxy
from harness_wrapper.models.oauth import openai_bridge
from harness_wrapper.models.oauth.openai_bridge import (
    CodexOAuthCredentials,
    CodexOAuthResponsesProxy,
)
from harness_wrapper.models.request_capture import RequestCaptureProxy
from harness_wrapper.models.request_policy import (
    OfflinePolicyError,
    open_offline_upstream,
    validate_offline_request,
)

URL = "https://never-fetched.invalid/resource.pdf"
BLOCKED = [
    {"input": [{"role": "user", "content": [{"type": "input_file", "file_url": URL}]}]},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": URL}}]}]},
    {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "document", "source": {"type": "url", "url": URL}}],
            }
        ]
    },
    {"contents": [{"parts": [{"fileData": {"fileUri": "gs://private-bucket/file"}}]}]},
    {
        "messages": [
            {
                "content": [
                    {
                        "type": "tool_result",
                        "content": [{"type": "image", "source": {"type": "url", "url": URL}}],
                    }
                ]
            }
        ]
    },
    {"tools": [{"type": "web_search"}]},
    {"tools": [{"type": "web_fetch_20250910", "name": "web_fetch"}]},
    {"tools": [{"type": "mcp", "server_url": URL}]},
    {"tools": [{"type": "code_interpreter"}]},
    {"tools": [{"type": "namespace", "name": "nested", "tools": [{"type": "web_search"}]}]},
    {"tools": [{"functionDeclarations": [], "googleSearch": {}}]},
    {"tools": [{"urlContext": {}}]},
    {"plugins": [{"id": "web"}]},
    {"extra_body": {"web_search_options": {}}},
    {"tools": [{"type": "shell", "environment": {"type": "container_auto"}}]},
    {"tools": [{"type": "a_future_hosted_tool"}]},
    {"tool_choice": {"type": "web_search"}},
    {"tools": [{"name": "local", "input_schema": {}, "googleSearch": {}}]},
    {"tools": [{"type": "function", "name": "local", "urlContext": {}}]},
    {"tools": [{"type": {}}]},
    {"model": {}},
    {"model": "other-model"},
    {"model": "test:online"},
]


@pytest.mark.parametrize("payload", BLOCKED)
def test_retrieval_payloads_are_rejected(payload):
    with pytest.raises(OfflinePolicyError):
        validate_offline_request(
            "POST",
            "/v1/responses",
            json.dumps({"model": "test", **payload}).encode(),
            models=frozenset({"test"}),
        )


@pytest.mark.parametrize(
    "path,body",
    [
        ("/v1/files", b"{}"),
        ("/v1/batches", b"{}"),
        ("/v1/../files", b"{}"),
        ("/v1/%2e%2e/files", b"{}"),
        ("/v1/responses?file_url=https://never-fetched.invalid", b"{}"),
        ("/v1/responses", b"not json"),
        ("/v1/responses", b"[]"),
        ("/v1/responses", b""),
        ("/v1beta/models/other-model:generateContent", b"{}"),
    ],
)
def test_alternative_endpoints_and_malformed_bodies_are_rejected(path, body):
    with pytest.raises(OfflinePolicyError):
        validate_offline_request("POST", path, body, models=frozenset({"test"}))


@pytest.mark.parametrize("prefix", ["/v1", "/api/v1"])
@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
def test_claude_beta_messages_query_preserves_payload_restrictions(prefix, endpoint):
    path = f"{prefix}/{endpoint}?beta=true"
    validate_offline_request(
        "POST", path, b'{"model":"test","messages":[]}', models=frozenset({"test"})
    )
    for payload in BLOCKED:
        with pytest.raises(OfflinePolicyError):
            validate_offline_request(
                "POST", path, json.dumps({"model": "test", **payload}).encode(),
                models=frozenset({"test"}),
            )


@pytest.mark.parametrize("path", [
    "/v1/messages?beta=arbitrary", "/v1/messages?beta=true&url=https://example.com",
    "/v1/responses?beta=true", "/v1/files?beta=true", "/v1/models?beta=true",
])
def test_beta_query_does_not_enable_other_routes_or_options(path):
    with pytest.raises(OfflinePolicyError):
        validate_offline_request("POST", path, b'{"model":"test"}', models=frozenset({"test"}))


@pytest.mark.parametrize("endpoint", ["messages", "messages/count_tokens"])
def test_claude_beta_request_reaches_openrouter_style_upstream(endpoint):
    forwarded = []

    class Response(io.BytesIO):
        status = 200
        headers = {"content-type": "application/json"}

    def upstream(request, **_):
        forwarded.append((request.full_url, request.get_header("Authorization")))
        return Response(b'{"ok":true}')

    with RequestCaptureProxy(
        "https://never-contacted.invalid/api", lambda *_: None,
        upstream_headers={"Authorization": "Bearer host-key"},
        open_upstream=upstream, offline_models=frozenset({"test"}),
    ) as proxy:
        request = urllib.request.Request(
            proxy.base_url + f"/v1/{endpoint}?beta=true",
            data=b'{"model":"test","messages":[]}',
            headers={"Content-Type": "application/json", "Authorization": "Bearer cli-placeholder"},
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            assert json.loads(response.read()) == {"ok": True}
    assert forwarded == [(f"https://never-contacted.invalid/api/v1/{endpoint}?beta=true", "Bearer host-key")]


@pytest.mark.parametrize(
    "tools",
    [
        [
            {
                "type": "function",
                "function": {
                    "name": "Bash",
                    "parameters": {"properties": {"file_url": {"default": URL}}},
                },
            }
        ],
        [{"name": "Bash", "input_schema": {"type": "object"}}],
        [{"type": "custom", "name": "apply_patch", "format": {"type": "grammar"}}],
        [{"type": "local_shell"}],
        [{"type": "shell", "environment": {"type": "local"}}],
        [{"functionDeclarations": [{"name": "run_command", "parameters": {"type": "object"}}]}],
        [
            {
                "type": "namespace",
                "name": "summary",
                "tools": [{"type": "function", "name": "generate_summary"}],
            }
        ],
    ],
)
def test_local_tools_inline_data_and_compaction_are_preserved(tools):
    payload = {
        "model": "test",
        "tools": tools,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": f"A problem may cite {URL}"},
                    {"type": "input_image", "image_url": "data:image/png;base64,aGVsbG8="},
                ],
            },
            {"type": "function_call", "arguments": json.dumps({"file_url": URL})},
            {"type": "function_call_output", "output": json.dumps({"url": URL})},
            {"type": "compaction", "encrypted_content": "opaque"},
        ],
    }
    original = json.dumps(payload).encode()
    for path in ["/v1/responses", "/v1/responses/compact", "/v1/messages/count_tokens"]:
        validate_offline_request("POST", path, original, models=frozenset({"test"}))
    assert json.dumps(payload).encode() == original
    validate_offline_request(
        "POST",
        "/v1beta/models/test:streamGenerateContent?alt=sse",
        original,
        models=frozenset({"test"}),
    )
    validate_offline_request("GET", "/v1/models", None, models=frozenset({"test"}))


@contextmanager
def _proxy(kind, monkeypatch, forwarded, **kwargs):
    class Response(io.BytesIO):
        status = 200
        headers: ClassVar[dict[str, str]] = {"content-type": "application/json"}

    def upstream(request, **_):
        forwarded.append(json.loads(request.data))
        return Response(b'{"ok":true}')

    class WebSocket:
        response = SimpleNamespace(headers=SimpleNamespace(raw_items=lambda: []))

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def send(self, body):
            forwarded.append(json.loads(body))

        def recv(self, **_):
            return json.dumps({"type": "response.completed", "response": {"id": "test"}})

    observer = kwargs.pop("on_request", lambda *_: None)
    if kind == "capture":
        proxy = RequestCaptureProxy(
            "https://never-contacted.invalid/v1", observer, open_upstream=upstream, **kwargs
        )
    elif kind == "header":
        monkeypatch.setattr(api_header_bridge, "open_offline_upstream", upstream)
        proxy = APIHeaderProxy(
            "https://never-contacted.invalid/v1",
            {"x-api-key": "fake"},
            on_request=observer,
            **kwargs,
        )
    else:
        monkeypatch.setattr(openai_bridge, "connect", lambda *_, **__: WebSocket())
        monkeypatch.setattr(openai_bridge, "_wait_for_connection", None)
        proxy = CodexOAuthResponsesProxy(
            credentials=CodexOAuthCredentials("fake", "fake"), on_request=observer, **kwargs
        )
    with proxy:
        headers = {"Content-Type": "application/json"}
        if kind != "capture":
            headers["Authorization"] = f"Bearer {proxy.client_api_key}"
        yield proxy.base_url + "/responses", headers


@pytest.mark.parametrize("kind", ["capture", "header", "codex"])
def test_every_proxy_rejects_before_upstream_and_still_forwards_local_tools(kind, monkeypatch):
    forwarded = []
    with _proxy(kind, monkeypatch, forwarded, offline_models=frozenset({"test"})) as (url, headers):
        for payload in BLOCKED:
            request = urllib.request.Request(
                url, data=json.dumps({"model": "test", **payload}).encode(), headers=headers
            )
            with pytest.raises(urllib.error.HTTPError) as error:
                urllib.request.urlopen(request, timeout=5)
            assert error.value.code == 403
            assert json.loads(error.value.read())["error"]["code"] == "offline_policy_violation"
        assert forwarded == []
        for raw in [b"{broken", b"[]", b""]:
            with pytest.raises(urllib.error.HTTPError):
                urllib.request.urlopen(
                    urllib.request.Request(url, data=raw, headers=headers), timeout=5
                )
        assert forwarded == []
        payload = {
            "model": "test",
            "tools": [{"type": "function", "name": "generate_summary"}],
            "input": "Summarize work",
        }
        with urllib.request.urlopen(
            urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers),
            timeout=5,
        ) as response:
            response.read()
        assert len(forwarded) == 1
        assert forwarded[0]["tools"] == payload["tools"]


@pytest.mark.parametrize("kind", ["capture", "header", "codex"])
def test_policy_cannot_be_bypassed_by_overrides_or_observer_failure(kind, monkeypatch):
    forwarded = []
    kwargs = {
        "offline_models": frozenset({"test"}),
        "request_overrides": {"plugins": [{"id": "web"}]},
        "on_request": Mock(side_effect=RuntimeError("observer failed")),
    }
    with _proxy(kind, monkeypatch, forwarded, **kwargs) as (url, headers):
        with pytest.raises(urllib.error.HTTPError) as error:
            urllib.request.urlopen(
                urllib.request.Request(url, data=b'{"model":"test"}', headers=headers), timeout=5
            )
        assert error.value.code == 403
    assert forwarded == []


def test_redirects_are_not_followed():
    visits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            visits.append(self.path)
            self.send_response(302)
            self.send_header("Location", "/unexpected-resource")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = urllib.request.Request(f"http://127.0.0.1:{server.server_port}/models")
        with pytest.raises(urllib.error.HTTPError):
            open_offline_upstream(request, timeout=5)
        assert visits == ["/models"]
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("network,expected", [(False, frozenset({"test"})), (True, frozenset())])
def test_policy_follows_existing_sandbox_network_setting(network, expected):
    agent = SimpleNamespace(env=SimpleNamespace(network_enabled=network), model_name="test")
    assert Agent._offline_models.fget(agent) == expected
    assert Agent._offline_models.fget(SimpleNamespace(env=None)) == frozenset()
