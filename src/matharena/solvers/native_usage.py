"""Read per-request usage from native Codex/Claude session files incrementally."""

import json
from pathlib import Path

from harness_wrapper import Agent, TokenUsage


def request_usage(value):
    """Include OpenAI's cache-write subset of input without adding it twice."""
    usage = Agent._normalize_token_usage(value)
    if usage is None:
        return None
    writes = Agent._usage_count(value, "cache_write_input_tokens")
    for name in ("input_tokens_details", "prompt_tokens_details"):
        details = value.get(name)
        if isinstance(details, dict):
            writes = max(writes, Agent._usage_count(details, "cache_write_tokens", "cache_write_input_tokens"))
    if not writes:
        return usage
    return TokenUsage(
        input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
        cache_read_tokens=usage.cache_read_tokens, cache_write_tokens=writes,
    )


class NativeUsageReader:
    def __init__(self, workspace, harness):
        home = Path(workspace) / ".harness-home"
        if harness in {"codex", "codex-cli", "openai-codex"}:
            self.root = home / ".codex" / "sessions"
            self.provider = "codex"
        elif harness in {"claude", "claude-code", "anthropic"}:
            self.root = home / ".claude" / "projects"
            self.provider = "claude"
        else:
            self.root = None
            self.provider = None
        self._offsets = {}
        self._requests = {}
        self._codex_totals = {}
        # A rerun can reuse a workspace. Its old session records belong to a
        # previous attempt; only new records contribute to this run's cost.
        if self.root is not None and self.root.is_dir():
            for path in self.root.rglob("*.jsonl"):
                try:
                    self._offsets[path] = path.stat().st_size
                except OSError:
                    continue

    def usages(self):
        if self.root is None or not self.root.is_dir():
            return []
        for path in self.root.rglob("*.jsonl"):
            try:
                with path.open("rb") as stream:
                    offset = self._offsets.get(path, 0)
                    stream.seek(offset)
                    while line := stream.readline():
                        # A running CLI may still be appending this record.
                        if not line.endswith(b"\n"):
                            break
                        offset = stream.tell()
                        try:
                            self._record(path, json.loads(line))
                        except (ValueError, TypeError, AttributeError):
                            continue
                    self._offsets[path] = offset
            except OSError:
                continue
        return list(self._requests.values())

    def _record(self, path, event):
        if not isinstance(event, dict):
            return
        if self.provider == "codex":
            payload = event.get("payload", {})
            if event.get("type") != "event_msg" or payload.get("type") != "token_count":
                return
            info = payload.get("info") or {}
            raw = info.get("last_token_usage")
            total = info.get("total_token_usage", {}).get("total_tokens")
            if not isinstance(total, int) or total <= self._codex_totals.get(path, 0):
                return
            usage = request_usage(raw)
            if usage is None:
                return
            self._codex_totals[path] = total
            self._requests[path, total] = usage
        elif event.get("type") == "assistant":
            message = event.get("message") or {}
            message_id = message.get("id")
            usage = Agent._normalize_token_usage(message.get("usage"))
            if not message_id or usage is None:
                return
            key = (path, message_id)
            previous = self._requests.get(key, TokenUsage())
            self._requests[key] = TokenUsage(**{
                name: max(getattr(previous, name), getattr(usage, name))
                for name in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
            })
