"""Loopback reverse proxy for inspecting CLI model requests safely."""

from __future__ import annotations

import http.client
import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator, Mapping
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import ThreadingMixIn, UnixStreamServer
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from .request_policy import (
    OfflinePolicyError,
    open_offline_upstream,
    reject_offline_request,
    validate_offline_request,
)

RequestObserver = Callable[[str, Mapping[str, Any]], bool | None]
ResponseObserver = Callable[[str, Mapping[str, Any]], None]
RequestTransform = Callable[[str, dict[str, Any]], None]


def response_chunks(stream: Any) -> Iterator[bytes]:
    """Forward available bytes immediately, preserving partial data on disconnect."""
    # HTTPResponse.read(n) waits for n bytes or EOF, withholding small SSE events
    # and heartbeats. read1(n) returns after one underlying read instead.
    read = getattr(stream, "read1", stream.read)
    try:
        while chunk := read(64 * 1024):
            yield chunk
    except (http.client.HTTPException, OSError) as error:
        logger.bind(component="model_proxy").warning(
            "Upstream response interrupted: {}; token usage may be incomplete.",
            type(error).__name__,
        )
        partial = getattr(error, "partial", b"")
        if partial:
            yield partial


def _merge_overrides(payload: dict[str, Any], overrides: Mapping[str, object]) -> None:
    """Recursively merge explicit request parameters into a client payload."""

    for key, value in overrides.items():
        current = payload.get(key)
        if isinstance(current, dict) and isinstance(value, Mapping):
            _merge_overrides(current, value)
        else:
            payload[key] = value


def _drop_fields(payload: dict[str, Any], fields: frozenset[str]) -> None:
    """Drop top-level fields or dotted paths from a JSON request body."""

    for field in fields:
        parts = field.split(".")
        current: Any = payload
        for part in parts[:-1]:
            if not isinstance(current, dict):
                break
            current = current.get(part)
        else:
            if isinstance(current, dict):
                current.pop(parts[-1], None)


class _ThreadingUnixHTTPServer(ThreadingMixIn, UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = False


class ResponseCapture:
    """Incrementally inspect JSON and SSE responses while forwarding unchanged bytes."""

    _MAX_JSON_BYTES = 32 * 1024 * 1024

    def __init__(
        self,
        path: str,
        content_type: str,
        observer: ResponseObserver | None,
        *,
        component: str,
    ) -> None:
        self.path = path
        self.content_type = content_type.lower()
        self.observer = observer
        self.component = component
        self._buffer = bytearray()
        self._last_usage: dict[str, dict[str, Any]] = {}

    def feed(self, chunk: bytes) -> None:
        if self.observer is None:
            return
        if "text/event-stream" in self.content_type or "ndjson" in self.content_type:
            self._buffer.extend(chunk)
            self._consume_lines(final=False)
        elif "json" in self.content_type and len(self._buffer) <= self._MAX_JSON_BYTES:
            self._buffer.extend(chunk)

    def finish(self) -> None:
        if self.observer is None:
            return
        if "text/event-stream" in self.content_type or "ndjson" in self.content_type:
            self._consume_lines(final=True)
        elif "json" in self.content_type and self._buffer:
            self._observe(bytes(self._buffer))
        if self._last_usage:
            self.observer(self.path, self._last_usage)
            self._last_usage = {}

    def _consume_lines(self, *, final: bool) -> None:
        while True:
            newline = self._buffer.find(b"\n")
            if newline < 0:
                if final and self._buffer:
                    line = bytes(self._buffer)
                    self._buffer.clear()
                    self._consume_line(line)
                return
            line = bytes(self._buffer[:newline])
            del self._buffer[: newline + 1]
            self._consume_line(line)

    def _consume_line(self, line: bytes) -> None:
        line = line.strip()
        if line.startswith(b"data:"):
            line = line[5:].strip()
        elif "ndjson" not in self.content_type:
            return
        if line and line != b"[DONE]":
            self._observe(line)

    def _observe(self, encoded: bytes) -> None:
        try:
            payload = json.loads(encoded)
            if isinstance(payload, Mapping) and self.observer is not None:
                observed = payload
                usage_fields = ["usageMetadata"]
                if self.path.rstrip("/").endswith("/chat/completions") or payload.get(
                    "object"
                ) in {"chat.completion", "chat.completion.chunk"}:
                    usage_fields.append("usage")
                # Streaming usage snapshots are cumulative within one response.
                # Forward content immediately, but account for usage only once.
                for field in usage_fields:
                    usage = payload.get(field)
                    if isinstance(usage, Mapping):
                        self._last_usage[field] = dict(usage)
                        observed = dict(observed)
                        observed.pop(field, None)
                if observed:
                    self.observer(self.path, observed)
        except Exception as error:
            logger.bind(component=self.component).warning(
                "Could not inspect model response: error_type={error_type}",
                error_type=type(error).__name__,
            )


class _CaptureServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    upstream_origin: str
    upstream_prefix: str
    route_secret: str
    upstream_headers: Mapping[str, str]
    request_overrides: Mapping[str, object]
    request_transform: RequestTransform | None
    static_get_responses: Mapping[str, Any]
    merge_request_overrides: bool
    request_drop_fields: frozenset[str]
    allow_remote_clients: bool
    offline_models: frozenset[str]
    upstream_timeout: float
    on_request: RequestObserver
    on_response: ResponseObserver | None
    open_upstream: Callable[..., Any]


class _CaptureUnixServer(_ThreadingUnixHTTPServer):
    upstream_origin: str
    upstream_prefix: str
    route_secret: str
    upstream_headers: Mapping[str, str]
    request_overrides: Mapping[str, object]
    request_transform: RequestTransform | None
    static_get_responses: Mapping[str, Any]
    merge_request_overrides: bool
    request_drop_fields: frozenset[str]
    allow_remote_clients: bool
    offline_models: frozenset[str]
    upstream_timeout: float
    on_request: RequestObserver
    on_response: ResponseObserver | None
    open_upstream: Callable[..., Any]


class _CaptureHandler(BaseHTTPRequestHandler):
    server: _CaptureServer | _CaptureUnixServer
    protocol_version = "HTTP/1.1"

    _HOP_BY_HOP_HEADERS = frozenset(
        {
            "accept-encoding",
            "connection",
            "content-length",
            "host",
            "keep-alive",
            "proxy-authenticate",
            "proxy-authorization",
            "te",
            "trailer",
            "transfer-encoding",
            "upgrade",
        }
    )

    def do_GET(self) -> None:
        # Some CLIs fetch a fixed catalog path relative to the origin, dropping
        # the secret URL prefix. Serve only caller-supplied, non-secret metadata
        # at these paths; they never forward credentials or requests upstream.
        static = self.server.static_get_responses.get(self.path)
        if static is not None:
            body = json.dumps(static).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        parsed = urlsplit(self.path)
        route = f"/{self.server.route_secret}"
        local_client = isinstance(self.client_address, tuple) and self.client_address[0] in {
            "127.0.0.1",
            "::1",
        }
        if (not self.server.allow_remote_clients and not local_client) or not (
            parsed.path == route or parsed.path.startswith(route + "/")
        ):
            self.send_error(404)
            return
        upstream_path = parsed.path[len(route) :] or "/"
        if self.server.upstream_prefix and not (
            upstream_path == self.server.upstream_prefix
            or upstream_path.startswith(self.server.upstream_prefix + "/")
        ):
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("content-length", "0"))
        except ValueError:
            self.send_error(400, "invalid content length")
            return
        if not 0 <= length <= 32 * 1024 * 1024:
            self.send_error(400, "invalid content length")
            return
        body = self.rfile.read(length) if length else None
        if self.server.request_transform is not None:
            # Security-sensitive transformations must fail closed. Never send
            # the original body upstream when filtering or validation fails.
            try:
                payload = json.loads(body or b"{}")
                if not isinstance(payload, dict):
                    raise ValueError("expected a JSON object")
                if self.server.merge_request_overrides:
                    _merge_overrides(payload, self.server.request_overrides)
                else:
                    payload.update(self.server.request_overrides)
                _drop_fields(payload, self.server.request_drop_fields)
                self.server.request_transform(upstream_path, payload)
                body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            except Exception:
                self.send_error(400, "request rejected by harness policy")
                return
        if body:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    if self.server.request_transform is None:
                        if self.server.merge_request_overrides:
                            _merge_overrides(payload, self.server.request_overrides)
                        else:
                            payload.update(self.server.request_overrides)
                        _drop_fields(payload, self.server.request_drop_fields)
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                    if self.server.on_request(upstream_path, payload) is False:
                        self.send_error(409, "harness run limit reached")
                        return
            except Exception as error:
                logger.bind(component="request_capture").warning(
                    "Could not inspect model request: error_type={error_type}",
                    error_type=type(error).__name__,
                )

        if self.server.offline_models:
            try:
                validate_offline_request(
                    self.command,
                    upstream_path + ("?" + parsed.query if parsed.query else ""),
                    body,
                    models=self.server.offline_models,
                )
            except OfflinePolicyError as error:
                reject_offline_request(self, error)
                return

        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in self._HOP_BY_HOP_HEADERS
        }
        protected_headers = {name.lower() for name in self.server.upstream_headers}
        headers = {
            name: value for name, value in headers.items() if name.lower() not in protected_headers
        }
        headers.update(self.server.upstream_headers)
        query = f"?{parsed.query}" if parsed.query else ""
        request = urllib.request.Request(
            self.server.upstream_origin + upstream_path + query,
            data=body,
            headers=headers,
            method=self.command,
        )
        try:
            upstream = self.server.open_upstream(request, timeout=self.server.upstream_timeout)
        except urllib.error.HTTPError as error:
            with error:
                self._send_upstream(error.code, error.headers, error, upstream_path)
        except OSError as error:
            logger.bind(component="request_capture").warning(
                "Model request proxy failed: error_type={error_type}",
                error_type=type(error).__name__,
            )
            self.send_error(502, "upstream request failed")
        else:
            with upstream:
                self._send_upstream(upstream.status, upstream.headers, upstream, upstream_path)

    def _send_upstream(
        self,
        status_code: int,
        headers: Any,
        stream: Any,
        path: str,
    ) -> None:
        self.send_response(status_code)
        for name in ("content-type", "content-length", "content-encoding"):
            value = headers.get(name)
            if value:
                self.send_header(name, value)
        self.send_header("connection", "close")
        self.end_headers()
        capture = ResponseCapture(
            path,
            headers.get("content-type", ""),
            self.server.on_response,
            component="request_capture",
        )
        for chunk in response_chunks(stream):
            capture.feed(chunk)
            with suppress(BrokenPipeError, ConnectionResetError):
                self.wfile.write(chunk)
                self.wfile.flush()
                continue
            break
        capture.finish()
        self.close_connection = True

    def log_message(self, format: str, *args: object) -> None:
        return


class RequestCaptureProxy:
    """Forward a model endpoint through loopback while observing JSON bodies."""

    def __init__(
        self,
        upstream_base_url: str,
        on_request: RequestObserver,
        *,
        upstream_headers: Mapping[str, str] | None = None,
        upstream_timeout: float = 28_800,
        on_response: ResponseObserver | None = None,
        open_upstream: Callable[..., Any] | None = None,
        request_overrides: Mapping[str, object] | None = None,
        request_transform: RequestTransform | None = None,
        static_get_responses: Mapping[str, Any] | None = None,
        request_drop_fields: tuple[str, ...] | frozenset[str] = (),
        merge_request_overrides: bool = False,
        listen_host: str = "127.0.0.1",
        client_host: str = "127.0.0.1",
        allow_remote_clients: bool = False,
        offline_models: frozenset[str] = frozenset(),
        unix_socket: os.PathLike[str] | None = None,
        client_port: int | None = None,
    ) -> None:
        parsed = urlsplit(upstream_base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("upstream_base_url must be an absolute HTTP(S) URL")
        self._upstream_origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self._upstream_prefix = parsed.path.rstrip("/")
        self._upstream_timeout = upstream_timeout
        self._upstream_headers = dict(upstream_headers or {})
        self._request_overrides = dict(request_overrides or {})
        self._request_transform = request_transform
        self._static_get_responses = dict(static_get_responses or {})
        self._request_drop_fields = frozenset(str(field) for field in request_drop_fields)
        self._merge_request_overrides = bool(merge_request_overrides)
        self._listen_host = listen_host
        self._client_host = client_host
        self._allow_remote_clients = allow_remote_clients
        self._offline_models = offline_models
        self._unix_socket = os.fspath(unix_socket) if unix_socket is not None else None
        self._client_port = client_port
        if (self._unix_socket is None) != (client_port is None):
            raise ValueError("unix_socket and client_port must be provided together")
        self._on_request = on_request
        self._on_response = on_response
        self._open_upstream = open_upstream or (
            open_offline_upstream if offline_models else urllib.request.urlopen
        )
        self._route_secret = secrets.token_urlsafe(24)
        self._server: _CaptureServer | _CaptureUnixServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("request capture proxy is not running")
        prefix = f"/{self._route_secret}{self._upstream_prefix}"
        if self._client_port is not None:
            port = self._client_port
        else:
            assert isinstance(self._server, _CaptureServer)
            port = self._server.server_address[1]
        return f"http://{self._client_host}:{port}{prefix}"

    def start(self) -> RequestCaptureProxy:
        if self._server is not None:
            raise RuntimeError("request capture proxy is already running")
        if self._unix_socket is None:
            server: _CaptureServer | _CaptureUnixServer = _CaptureServer(
                (self._listen_host, 0), _CaptureHandler
            )
        else:
            server = _CaptureUnixServer(self._unix_socket, _CaptureHandler)
        server.upstream_origin = self._upstream_origin
        server.upstream_prefix = self._upstream_prefix
        server.upstream_timeout = self._upstream_timeout
        server.upstream_headers = self._upstream_headers
        server.request_overrides = self._request_overrides
        server.request_transform = self._request_transform
        server.static_get_responses = self._static_get_responses
        server.request_drop_fields = self._request_drop_fields
        server.merge_request_overrides = self._merge_request_overrides
        server.allow_remote_clients = self._allow_remote_clients
        server.offline_models = self._offline_models
        server.route_secret = self._route_secret
        server.on_request = self._on_request
        server.on_response = self._on_response
        server.open_upstream = self._open_upstream
        thread = threading.Thread(
            target=server.serve_forever,
            name="model-request-capture",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        logger.bind(component="request_capture").debug("Started model request capture proxy")
        return self

    def close(self) -> None:
        server, thread = self._server, self._thread
        self._server = None
        self._thread = None
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None:
            thread.join(timeout=2)

    def __enter__(self) -> RequestCaptureProxy:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = [
    "RequestCaptureProxy",
    "RequestObserver",
    "ResponseCapture",
    "ResponseObserver",
]
