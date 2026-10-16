"""Secure, invocation-local bridge from third-party Responses clients to Codex OAuth."""

from __future__ import annotations

import hmac
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any, ClassVar

from loguru import logger
from websockets.exceptions import InvalidStatus, WebSocketException
from websockets.sync.client import ClientConnection, connect

from ..request_capture import (
    RequestObserver,
    ResponseCapture,
    ResponseObserver,
    _ThreadingUnixHTTPServer,
)
from ..request_policy import (
    OfflinePolicyError,
    reject_offline_request,
    validate_offline_request,
)
from .native_credentials import codex_oauth_credential
from .oauth import AuthenticationError, CredentialStore

CODEX_RESPONSES_URL = "wss://chatgpt.com/backend-api/codex/responses"

_wait_for_connection: Callable[[], None] | None = None


@contextmanager
def pace_codex_connections(interval_seconds: float) -> Iterator[None]:
    """Space connection attempts across all OAuth bridges in this process.

    Enter once around a batch, before starting workers. Retries and tool
    continuations share the same pacing; other processes are unaffected.
    """
    if not 0 < interval_seconds < float("inf"):
        raise ValueError("interval_seconds must be positive and finite")
    lock = threading.Lock()
    last_start = float("-inf")

    def wait() -> None:
        nonlocal last_start
        with lock:
            delay = last_start + interval_seconds - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            last_start = time.monotonic()

    global _wait_for_connection
    previous = _wait_for_connection
    _wait_for_connection = wait
    try:
        yield
    finally:
        _wait_for_connection = previous


@dataclass(frozen=True, slots=True)
class CodexOAuthCredentials:
    access_token: str
    account_id: str

    def __repr__(self) -> str:
        return "CodexOAuthCredentials(access_token='***', account_id='***')"


def load_codex_oauth_credentials(
    path: str | os.PathLike[str] | None = None,
) -> CodexOAuthCredentials:
    """Import and normalize the native Codex token bundle."""

    credential = codex_oauth_credential(path)
    account_id = credential.extra.get("account_id")
    if credential.expires_at is not None and credential.expires_at <= time.time() + 30:
        raise AuthenticationError(
            "Codex OAuth access token is expired; run `harness-wrapper login openai --force`"
        )
    if credential.access_token is None or not isinstance(account_id, str) or not account_id:
        raise AuthenticationError("Codex OAuth credential bundle is incomplete")
    return CodexOAuthCredentials(credential.access_token, account_id)


def load_central_openai_oauth_credentials(
    store: CredentialStore | None = None,
) -> CodexOAuthCredentials:
    """Load the wrapper-owned OpenAI token from the central auth.json."""

    credential = (store or CredentialStore()).load("openai")
    if credential is None or credential.access_token is None:
        raise AuthenticationError(
            "central OpenAI OAuth token is unavailable; run `harness-wrapper login openai`"
        )
    account_id = credential.extra.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        raise AuthenticationError("central OpenAI OAuth token has no account ID")
    if credential.expires_at is not None and credential.expires_at <= time.time() + 30:
        raise AuthenticationError(
            "central OpenAI OAuth token is expired; run `harness-wrapper login openai --force`"
        )
    return CodexOAuthCredentials(credential.access_token, account_id)


class _BridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    credentials: CodexOAuthCredentials | None
    client_api_key: str
    upstream_url: str
    on_request: RequestObserver | None
    on_response: ResponseObserver | None
    allow_remote_clients: bool
    offline_models: frozenset[str]
    request_overrides: Mapping[str, object]


class _BridgeUnixServer(_ThreadingUnixHTTPServer):
    credentials: CodexOAuthCredentials | None
    client_api_key: str
    upstream_url: str
    on_request: RequestObserver | None
    on_response: ResponseObserver | None
    allow_remote_clients: bool
    offline_models: frozenset[str]
    request_overrides: Mapping[str, object]


class _BridgeHandler(BaseHTTPRequestHandler):
    server: _BridgeServer | _BridgeUnixServer
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:
        local_client = isinstance(self.client_address, tuple) and self.client_address[0] in {
            "127.0.0.1",
            "::1",
        }
        if (not self.server.allow_remote_clients and not local_client) or self.path != "/responses":
            self.send_error(404)
            return
        authorization = self.headers.get("authorization", "")
        expected = f"Bearer {self.server.client_api_key}"
        if not hmac.compare_digest(authorization, expected):
            self.send_error(403)
            return
        try:
            length = int(self.headers.get("content-length", "0"))
            if not 0 < length <= 32 * 1024 * 1024:
                raise ValueError("invalid request length")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
        except ValueError:
            self.send_error(400, "invalid JSON request")
            return
        payload.update(self.server.request_overrides)
        # The WebSocket create event omits HTTP-only fields. The subscription
        # backend controls output limits itself and rejects max_output_tokens.
        for field in ("max_output_tokens", "stream", "background"):
            payload.pop(field, None)
        payload["type"] = "response.create"
        if self.server.on_request is not None:
            try:
                if self.server.on_request(self.path, payload) is False:
                    self.send_error(409, "harness run limit reached")
                    return
            except Exception as error:
                logger.bind(component="oauth_bridge", provider="openai").warning(
                    "Could not inspect model request: error_type={error_type}",
                    error_type=type(error).__name__,
                )

        if self.server.offline_models:
            try:
                validate_offline_request(
                    "POST",
                    self.path,
                    json.dumps(payload).encode(),
                    models=self.server.offline_models,
                )
            except OfflinePolicyError as error:
                reject_offline_request(self, error)
                return

        credentials = self.server.credentials or load_central_openai_oauth_credentials()
        # Preserve native session/turn routing, but replace credentials and keep
        # transport headers (including compression) under the bridge's control.
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower().startswith("x-codex-")
            or key.lower()
            in {
                "originator",
                "user-agent",
                "session-id",
                "thread-id",
                "x-client-request-id",
                "x-openai-internal-codex-responses-lite",
            }
        }
        headers.update(
            {
                "Authorization": f"Bearer {credentials.access_token}",
                "ChatGPT-Account-Id": credentials.account_id,
            }
        )
        try:
            wait_for_connection = _wait_for_connection
            if wait_for_connection is not None:
                wait_for_connection()
            upstream = connect(
                self.server.upstream_url,
                additional_headers=headers,
                user_agent_header=None,
                open_timeout=28_800,
                ping_timeout=28_800,
                max_size=None,
                # Library debug logs include handshake credentials and frame contents.
                logger=logging.Logger("oauth_websocket", level=logging.WARNING),
            )
        except InvalidStatus as error:
            response = error.response
            self._send_error_response(
                response.status_code,
                bytes(response.body),
                response.headers.get("content-type", "application/json"),
                response.headers,
            )
        except (OSError, WebSocketException) as error:
            self._send_error_response(
                502,
                json.dumps(
                    {"error": {"message": f"Upstream WebSocket failed: {type(error).__name__}"}}
                ).encode(),
            )
        else:
            logger.bind(component="oauth_bridge", provider="openai").debug(
                "OAuth bridge connected to upstream WebSocket"
            )
            with upstream:
                self._send_upstream(upstream, payload)

    def _send_error_response(
        self,
        status: int,
        body: bytes,
        content_type: str = "application/json",
        headers: Any = None,
    ) -> None:
        logger.bind(component="oauth_bridge", provider="openai").warning(
            "OAuth bridge upstream WebSocket connection failed: status={}", status
        )
        self.send_response(status)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "close")
        if headers is not None and headers.get("retry-after"):
            self.send_header("retry-after", headers["retry-after"])
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _send_upstream(self, stream: ClientConnection, payload: dict[str, Any]) -> None:
        # Keep SSE on the protected local hop so existing Responses clients work;
        # the long-lived connection to OpenAI uses the native WebSocket route.
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        assert stream.response is not None
        for key, value in stream.response.headers.raw_items():
            if key.lower().startswith("x-codex-") or key.lower() == "x-request-id":
                self.send_header(key, value)
        self.send_header("connection", "close")
        self.end_headers()
        capture = ResponseCapture(
            self.path, "text/event-stream", self.server.on_response, component="oauth_bridge"
        )
        try:
            stream.send(json.dumps(payload, separators=(",", ":")))
            while True:
                event = json.loads(stream.recv(timeout=28_800))
                if not isinstance(event, dict):
                    raise ValueError("expected a WebSocket event object")
                chunk = ("data: " + json.dumps(event, separators=(",", ":")) + "\n\n").encode()
                capture.feed(chunk)
                self.wfile.write(chunk)
                self.wfile.flush()
                if event.get("type") in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                    "error",
                }:
                    break
        except (BrokenPipeError, ConnectionResetError):
            pass
        except (OSError, WebSocketException, ValueError) as error:
            logger.bind(component="oauth_bridge", provider="openai").warning(
                "Upstream WebSocket interrupted: {}; token usage may be incomplete.",
                type(error).__name__,
            )
        finally:
            capture.finish()
            self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return


class CodexOAuthResponsesProxy:
    """Bridge local Responses/SSE clients to the Codex OAuth WebSocket endpoint."""

    _server_class: ClassVar[type[_BridgeServer]] = _BridgeServer

    def __init__(
        self,
        *,
        credentials: CodexOAuthCredentials | None = None,
        upstream_url: str = CODEX_RESPONSES_URL,
        on_request: RequestObserver | None = None,
        on_response: ResponseObserver | None = None,
        request_overrides: Mapping[str, object] | None = None,
        listen_host: str = "127.0.0.1",
        client_host: str = "127.0.0.1",
        allow_remote_clients: bool = False,
        offline_models: frozenset[str] = frozenset(),
        unix_socket: os.PathLike[str] | None = None,
        client_port: int | None = None,
    ) -> None:
        self.credentials = credentials
        self.upstream_url = upstream_url
        self.on_request = on_request
        self.on_response = on_response
        self.request_overrides = dict(request_overrides or {})
        self.listen_host = listen_host
        self.client_host = client_host
        self.allow_remote_clients = allow_remote_clients
        self.offline_models = offline_models
        self.unix_socket = os.fspath(unix_socket) if unix_socket is not None else None
        self.client_port = client_port
        if (self.unix_socket is None) != (client_port is None):
            raise ValueError("unix_socket and client_port must be provided together")
        self._client_api_key = secrets.token_urlsafe(32)
        self._server: _BridgeServer | _BridgeUnixServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("Codex OAuth bridge is not running")
        if self.client_port is not None:
            port = self.client_port
        else:
            assert isinstance(self._server, _BridgeServer)
            port = self._server.server_address[1]
        return f"http://{self.client_host}:{port}"

    @property
    def client_api_key(self) -> str:
        return self._client_api_key

    def start(self) -> CodexOAuthResponsesProxy:
        if self._server is not None:
            raise RuntimeError("Codex OAuth bridge is already running")
        if self.unix_socket is None:
            server: _BridgeServer | _BridgeUnixServer = self._server_class(
                (self.listen_host, 0), _BridgeHandler
            )
        else:
            server = _BridgeUnixServer(self.unix_socket, _BridgeHandler)
        # Resolve the short-lived host credential only when a request arrives.
        # This keeps startup side-effect free and picks up a recently refreshed
        # Codex login without ever copying auth.json into the sandbox.
        server.credentials = self.credentials
        server.client_api_key = self.client_api_key
        server.upstream_url = self.upstream_url
        server.on_request = self.on_request
        server.on_response = self.on_response
        server.request_overrides = self.request_overrides
        server.allow_remote_clients = self.allow_remote_clients
        server.offline_models = self.offline_models
        thread = threading.Thread(
            target=server.serve_forever,
            name="codex-oauth-responses-bridge",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        logger.bind(component="oauth_bridge", provider="openai").info(
            "Started loopback OAuth Responses bridge with upstream WebSocket"
        )
        return self

    def reload_credentials(self) -> CodexOAuthCredentials:
        """Reload the central token after an interactive OAuth re-login."""

        credentials = load_central_openai_oauth_credentials()
        self.credentials = credentials
        if self._server is not None:
            self._server.credentials = credentials
        logger.bind(component="oauth_bridge", provider="openai").info(
            "Reloaded OAuth bridge credentials"
        )
        return credentials

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)
        if server is not None:
            logger.bind(component="oauth_bridge", provider="openai").info(
                "Stopped loopback OAuth Responses bridge"
            )

    def __enter__(self) -> CodexOAuthResponsesProxy:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "CODEX_RESPONSES_URL",
    "CodexOAuthCredentials",
    "CodexOAuthResponsesProxy",
    "load_central_openai_oauth_credentials",
    "load_codex_oauth_credentials",
    "pace_codex_connections",
]
