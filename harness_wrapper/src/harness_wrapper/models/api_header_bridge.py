"""Invocation-local bridge for API endpoints that require custom headers."""

from __future__ import annotations

import copy
import gzip
import hmac
import io
import json
import os
import secrets
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from loguru import logger

from .request_capture import (
    RequestObserver,
    ResponseCapture,
    ResponseObserver,
    _ThreadingUnixHTTPServer,
    response_chunks,
)
from .request_policy import (
    OfflinePolicyError,
    open_offline_upstream,
    reject_offline_request,
    validate_offline_request,
)


def _error_detail(body: bytes, headers: Any) -> str:
    """Return a short readable provider error without altering forwarded bytes."""

    decoded = body
    if headers.get("content-encoding", "").lower() == "gzip":
        with suppress(OSError):
            decoded = gzip.decompress(body)
    text_value = decoded.decode("utf-8", errors="replace").strip()
    try:
        payload = json.loads(text_value)
    except (json.JSONDecodeError, TypeError):
        detail = text_value or "no body"
    else:
        error = payload.get("error") if isinstance(payload, Mapping) else None
        if isinstance(error, Mapping) and error.get("message"):
            detail = str(error["message"])
        elif error:
            detail = str(error)
        else:
            detail = text_value or "no body"
    detail = " ".join(detail.split())
    return detail if len(detail) <= 1000 else f"{detail[:997]}..."


class _ToolCallMetadataReplay:
    """Restore opaque provider metadata that a client CLI drops from tool calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_call_id: dict[str, Any] = {}

    def observe(
        self,
        payload: Mapping[str, Any],
        stream_slots: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        choices = payload.get("choices")
        if not isinstance(choices, list):
            return
        with self._lock:
            for choice_position, choice in enumerate(choices):
                if not isinstance(choice, Mapping):
                    continue
                raw_choice_index = choice.get("index")
                choice_index = (
                    raw_choice_index if isinstance(raw_choice_index, int) else choice_position
                )
                for envelope_name in ("delta", "message"):
                    envelope = choice.get(envelope_name)
                    if not isinstance(envelope, Mapping):
                        continue
                    tool_calls = envelope.get("tool_calls")
                    if not isinstance(tool_calls, list):
                        continue
                    for call_position, call in enumerate(tool_calls):
                        if not isinstance(call, Mapping):
                            continue
                        raw_call_index = call.get("index")
                        call_index = (
                            raw_call_index if isinstance(raw_call_index, int) else call_position
                        )
                        slot = stream_slots.setdefault((choice_index, call_index), {})
                        call_id = call.get("id")
                        if isinstance(call_id, str) and call_id:
                            slot["id"] = call_id
                        if "extra_content" in call:
                            slot["extra_content"] = copy.deepcopy(call["extra_content"])
                        if "id" in slot and "extra_content" in slot:
                            self._by_call_id[slot["id"]] = copy.deepcopy(slot["extra_content"])

    def apply(self, payload: dict[str, Any]) -> int:
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return 0
        restored = 0
        with self._lock:
            for message in messages:
                if not isinstance(message, Mapping):
                    continue
                tool_calls = message.get("tool_calls")
                if not isinstance(tool_calls, list):
                    continue
                for call in tool_calls:
                    if not isinstance(call, dict) or "extra_content" in call:
                        continue
                    call_id = call.get("id")
                    if isinstance(call_id, str) and call_id in self._by_call_id:
                        call["extra_content"] = copy.deepcopy(self._by_call_id[call_id])
                        restored += 1
        return restored


class _HeaderBridgeServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = False

    upstream_origin: str
    upstream_prefix: str
    upstream_headers: Mapping[str, str]
    request_overrides: Mapping[str, object]
    request_drop_fields: frozenset[str]
    tool_call_metadata: _ToolCallMetadataReplay
    allow_remote_clients: bool
    offline_models: frozenset[str]
    client_api_key: str
    on_request: RequestObserver | None
    on_response: ResponseObserver | None


class _HeaderBridgeUnixServer(_ThreadingUnixHTTPServer):
    upstream_origin: str
    upstream_prefix: str
    upstream_headers: Mapping[str, str]
    request_overrides: Mapping[str, object]
    request_drop_fields: frozenset[str]
    tool_call_metadata: _ToolCallMetadataReplay
    allow_remote_clients: bool
    offline_models: frozenset[str]
    client_api_key: str
    on_request: RequestObserver | None
    on_response: ResponseObserver | None


class _HeaderBridgeHandler(BaseHTTPRequestHandler):
    server: _HeaderBridgeServer | _HeaderBridgeUnixServer
    protocol_version = "HTTP/1.1"

    _HOP_BY_HOP_HEADERS = frozenset(
        {
            "authorization",
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
        self._forward()

    def do_POST(self) -> None:
        self._forward()

    def _forward(self) -> None:
        prefix = self.server.upstream_prefix
        path = urlsplit(self.path).path
        local_client = isinstance(self.client_address, tuple) and self.client_address[0] in {
            "127.0.0.1",
            "::1",
        }
        if (not self.server.allow_remote_clients and not local_client) or (
            prefix and path != prefix and not path.startswith(prefix + "/")
        ):
            self.send_error(404)
            return
        expected = f"Bearer {self.server.client_api_key}"
        if not hmac.compare_digest(self.headers.get("authorization", ""), expected):
            self.send_error(403)
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
        if body:
            try:
                payload = json.loads(body)
                if isinstance(payload, dict):
                    payload.update(self.server.request_overrides)
                    for field in self.server.request_drop_fields:
                        payload.pop(field, None)
                    restored = self.server.tool_call_metadata.apply(payload)
                    if restored:
                        logger.bind(component="api_header_bridge").debug(
                            "Restored opaque metadata for {count} tool call(s)",
                            count=restored,
                        )
                    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
                if (
                    isinstance(payload, Mapping)
                    and self.server.on_request is not None
                    and self.server.on_request(path, payload) is False
                ):
                    self.send_error(409, "harness run limit reached")
                    return
            except Exception as error:
                logger.bind(component="api_header_bridge").warning(
                    "Could not inspect model request: error_type={error_type}",
                    error_type=type(error).__name__,
                )
        if self.server.offline_models:
            try:
                validate_offline_request(
                    self.command, self.path, body, models=self.server.offline_models
                )
            except OfflinePolicyError as error:
                reject_offline_request(self, error)
                return
        headers = {
            name: value
            for name, value in self.headers.items()
            if name.lower() not in self._HOP_BY_HOP_HEADERS
        }
        headers.update(self.server.upstream_headers)
        request = urllib.request.Request(
            self.server.upstream_origin + self.path,
            data=body,
            headers=headers,
            method=self.command,
        )
        bound_logger = logger.bind(component="api_header_bridge")
        bound_logger.debug("Forwarding request through custom-header API bridge")
        try:
            open_upstream = (
                open_offline_upstream if self.server.offline_models else urllib.request.urlopen
            )
            upstream = open_upstream(request, timeout=28_800)
        except urllib.error.HTTPError as error:
            with error:
                error_body = error.read()
                bound_logger.warning(
                    "Custom-header API bridge upstream request failed: "
                    "status={status}, detail={detail}",
                    status=error.code,
                    detail=_error_detail(error_body, error.headers),
                )
                self._send_upstream(
                    error.code,
                    error.headers,
                    io.BytesIO(error_body),
                    path,
                )
        except OSError as error:
            bound_logger.warning(
                "Custom-header API bridge upstream request failed: error_type={error_type}",
                error_type=type(error).__name__,
            )
            self.send_error(502, "upstream request failed")
        else:
            with upstream:
                self._send_upstream(upstream.status, upstream.headers, upstream, path)

    def _observe_response(
        self,
        path: str,
        payload: Mapping[str, Any],
        stream_slots: dict[tuple[int, int], dict[str, Any]],
    ) -> None:
        self.server.tool_call_metadata.observe(payload, stream_slots)
        if self.server.on_response is not None:
            self.server.on_response(path, payload)

    def _send_upstream(
        self,
        status_code: int,
        headers: Any,
        stream: Any,
        path: str,
    ) -> None:
        self.send_response(status_code)
        content_type = headers.get("content-type")
        if content_type:
            self.send_header("content-type", content_type)
        content_length = headers.get("content-length")
        if content_length:
            self.send_header("content-length", content_length)
        content_encoding = headers.get("content-encoding")
        if content_encoding:
            self.send_header("content-encoding", content_encoding)
        self.send_header("connection", "close")
        self.end_headers()
        stream_slots: dict[tuple[int, int], dict[str, Any]] = {}
        capture = ResponseCapture(
            path,
            headers.get("content-type", ""),
            lambda response_path, payload: self._observe_response(
                response_path, payload, stream_slots
            ),
            component="api_header_bridge",
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


class APIHeaderProxy:
    """Protect and inject endpoint headers without exposing them to a child CLI."""

    def __init__(
        self,
        upstream_base_url: str,
        headers: Mapping[str, str],
        *,
        on_request: RequestObserver | None = None,
        on_response: ResponseObserver | None = None,
        request_overrides: Mapping[str, object] | None = None,
        request_drop_fields: tuple[str, ...] | frozenset[str] = (),
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
        if not headers:
            raise ValueError("at least one upstream header is required")
        self._upstream_origin = urlunsplit((parsed.scheme, parsed.netloc, "", "", ""))
        self._upstream_prefix = parsed.path.rstrip("/")
        self._headers = dict(headers)
        self._request_overrides = dict(request_overrides or {})
        self._request_drop_fields = frozenset(str(field) for field in request_drop_fields)
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
        self._client_api_key = secrets.token_urlsafe(32)
        self._server: _HeaderBridgeServer | _HeaderBridgeUnixServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def base_url(self) -> str:
        if self._server is None:
            raise RuntimeError("custom-header API bridge is not running")
        if self._client_port is not None:
            port = self._client_port
        else:
            assert isinstance(self._server, _HeaderBridgeServer)
            port = self._server.server_address[1]
        return f"http://{self._client_host}:{port}{self._upstream_prefix}"

    @property
    def client_api_key(self) -> str:
        return self._client_api_key

    def start(self) -> APIHeaderProxy:
        if self._server is not None:
            raise RuntimeError("custom-header API bridge is already running")
        if self._unix_socket is None:
            server: _HeaderBridgeServer | _HeaderBridgeUnixServer = _HeaderBridgeServer(
                (self._listen_host, 0), _HeaderBridgeHandler
            )
        else:
            server = _HeaderBridgeUnixServer(self._unix_socket, _HeaderBridgeHandler)
        server.upstream_origin = self._upstream_origin
        server.upstream_prefix = self._upstream_prefix
        server.upstream_headers = self._headers
        server.request_overrides = self._request_overrides
        server.request_drop_fields = self._request_drop_fields
        server.tool_call_metadata = _ToolCallMetadataReplay()
        server.allow_remote_clients = self._allow_remote_clients
        server.offline_models = self._offline_models
        server.client_api_key = self.client_api_key
        server.on_request = self._on_request
        server.on_response = self._on_response
        thread = threading.Thread(
            target=server.serve_forever,
            name="api-custom-header-bridge",
            daemon=True,
        )
        thread.start()
        self._server = server
        self._thread = thread
        logger.bind(component="api_header_bridge").info("Started loopback custom-header API bridge")
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
        if server is not None:
            logger.bind(component="api_header_bridge").info(
                "Stopped loopback custom-header API bridge"
            )

    def __enter__(self) -> APIHeaderProxy:
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["APIHeaderProxy"]
