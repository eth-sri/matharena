"""Provider-side network restrictions for requests from offline sandboxes."""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from loguru import logger


class OfflinePolicyError(ValueError):
    """A request would grant an offline sandbox provider-side capabilities."""


# Functions/custom tools execute in the CLI; unknown provider tools fail closed.
_LOCAL_TOOL_TYPES = {"function", "custom", "local_shell", "apply_patch"}
_LOCAL_TOOL_FIELDS = {
    "type",
    "name",
    "description",
    "parameters",
    "function",
    "strict",
    "input_schema",
    "cache_control",
    "defer_loading",
    "eager_input_streaming",
    "format",
    "tools",
    "environment",
    "functionDeclarations",
    "function_declarations",
}
_REMOTE_OPTIONS = {
    "plugins",
    "mcpservers",
    "websearchoptions",
    "searchparameters",
    "enablesearch",
    "googlesearch",
    "googlesearchretrieval",
    "urlcontext",
    "extensions",
}
_SCHEMA_FIELDS = {"parameters", "input_schema", "output_schema", "json_schema"}
_GENERATION_PATH = re.compile(
    r"/(?:[^/]+/)*(?:responses(?:/compact)?|chat/completions|messages(?:/count_tokens)?"
    r"|models/[^/]+:(?:generateContent|streamGenerateContent|countTokens))\Z"
)


def _local_tools(tools: Any) -> None:
    if not isinstance(tools, list):
        raise OfflinePolicyError("Expected a list of local tool definitions")
    for tool in tools:
        if not isinstance(tool, dict):
            raise OfflinePolicyError("Invalid local tool definition")
        if not set(tool) <= _LOCAL_TOOL_FIELDS:
            raise OfflinePolicyError("Provider-hosted tool options are disabled")
        kind = tool.get("type")
        if kind == "namespace":
            _local_tools(tool.get("tools"))
        elif kind in _LOCAL_TOOL_TYPES or (
            kind == "shell" and tool.get("environment", {}).get("type") == "local"
        ):
            continue
        elif kind is None and set(tool) <= {"functionDeclarations", "function_declarations"}:
            continue  # Gemini client-side functions, with no sibling hosted tools.
        elif kind is None and "name" in tool and "input_schema" in tool:
            continue  # Anthropic client-side functions.
        else:
            raise OfflinePolicyError("Only client-side tools are permitted")


def _content(value: Any) -> None:
    if isinstance(value, list):
        for item in value:
            _content(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            normalized = key.replace("_", "").lower()
            if key in _SCHEMA_FIELDS or key in {"arguments", "args"}:
                continue  # Tool schemas/arguments are data, not provider fetch instructions.
            if value.get("type") == "tool_use" and key == "input":
                continue
            if key == "functionResponse":
                # Gemini function results are arbitrary JSON. Multimodal parts, if
                # present, remain provider-interpreted and must still be inspected.
                if isinstance(item, dict):
                    _content(item.get("parts", []))
                continue
            if key == "tools":
                _local_tools(item)
                continue
            if (
                normalized in _REMOTE_OPTIONS
                and item is not None
                and item is not False
                and item != []
            ):
                raise OfflinePolicyError(f"Provider retrieval option {key!r} is disabled")
            if normalized.endswith(("url", "uri")):
                url = item.get("url") if isinstance(item, dict) else item
                if not isinstance(url, str) or not url.startswith("data:"):
                    raise OfflinePolicyError("Remote resource URLs are disabled; use inline data")
            _content(item)


def validate_offline_request(
    method: str, path: str, body: bytes | None, *, models: frozenset[str]
) -> None:
    """Check the final wire body, after adapter transformations, before forwarding."""
    parsed = urlsplit(path)
    # Reject encoded/ambiguous routes and limit access to inference and metadata.
    if any(part in {".", ".."} for part in parsed.path.split("/")) or any(
        char in parsed.path for char in "%\\"
    ):
        raise OfflinePolicyError("Invalid model endpoint path")
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        if not (
            key == "key"
            or (key == "alt" and value == "sse")
            or (key == "stream" and value in {"true", "false"})
            # Claude's beta Messages SDK adds this transport flag. Request-body
            # checks below still prohibit hosted tools and remote resources.
            or (
                key == "beta"
                and value == "true"
                and method == "POST"
                and parsed.path.endswith(("/v1/messages", "/v1/messages/count_tokens"))
            )
        ):
            raise OfflinePolicyError("Unsupported model endpoint query parameter")
    if method == "GET" and parsed.path.rstrip("/").endswith("/models") and not body:
        return
    if method != "POST" or not _GENERATION_PATH.fullmatch(parsed.path):
        raise OfflinePolicyError(
            "Only inference, compaction and model catalog endpoints are permitted"
        )
    try:
        payload = json.loads(body or b"")
    except (ValueError, UnicodeError, RecursionError) as error:
        raise OfflinePolicyError("Expected a JSON model request") from error
    if not isinstance(payload, dict):
        raise OfflinePolicyError("Expected a JSON object")
    path_model = re.search(
        r"/models/([^/]+):(?:generateContent|streamGenerateContent|countTokens)$", parsed.path
    )
    requested_model = path_model.group(1) if path_model else payload.get("model")
    if not isinstance(requested_model, str) or requested_model not in {
        model.removeprefix("models/") for model in models
    }:
        raise OfflinePolicyError("Only the configured model is permitted")
    if ":online" in str(payload.get("model", "")).lower():
        raise OfflinePolicyError("Web-enabled model variants are disabled")
    choice = payload.get("tool_choice")
    if isinstance(choice, dict) and choice.get("type") not in {
        "auto",
        "none",
        "required",
        "any",
        "tool",
        "function",
        "custom",
        "allowed_tools",
    }:
        raise OfflinePolicyError("Provider-hosted tool selection is disabled")
    try:
        _content(payload)
    except (AttributeError, TypeError, RecursionError) as error:
        raise OfflinePolicyError("Invalid model request structure") from error


def reject_offline_request(handler: BaseHTTPRequestHandler, error: OfflinePolicyError) -> None:
    logger.bind(component="model_proxy").warning("Offline model request rejected: {}", error)
    body = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "code": "offline_policy_violation",
                "message": str(error),
            }
        }
    ).encode()
    handler.send_response(403)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Connection", "close")
    handler.end_headers()
    handler.wfile.write(body)
    handler.close_connection = True


class _NoRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(
            req.full_url, code, "Model proxy redirects are disabled", headers, fp
        )


def open_offline_upstream(request: urllib.request.Request, *, timeout: float) -> Any:
    """Never follow an upstream redirect into another internet resource."""
    return urllib.request.build_opener(_NoRedirects()).open(request, timeout=timeout)
