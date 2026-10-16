"""Read the terminal response from Antigravity's local session database.

The CLI's stream can omit the tail of a conversation. In CLI 1.1.x the
SQLite ``steps.step_payload`` is a protobuf: field 20 is a planner response,
whose field 1 is the submitted answer (not the reasoning in other fields).
Read only that exact field on a completed, terminal planner step. Unknown
schemas fail closed; never search arbitrary blobs for answer-looking text.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NativeResponse:
    text: str
    step_index: int
    database: Path


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if pos >= len(data):
            raise ValueError("Truncated native-session protobuf")
        byte = data[pos]
        pos += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, pos
    raise ValueError("Invalid native-session protobuf varint")


def _bytes_field(data: bytes, wanted: int) -> bytes | None:
    pos = 0
    found = None
    while pos < len(data):
        key, pos = _varint(data, pos)
        number, wire = key >> 3, key & 7
        if number == 0:
            raise ValueError("Invalid native-session protobuf field")
        if wire == 0:
            _, pos = _varint(data, pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        elif wire == 2:
            length, pos = _varint(data, pos)
            end = pos + length
            if number == wanted:
                if found is not None:
                    raise ValueError("Ambiguous native-session response field")
                found = data[pos:end]
            pos = end
        else:
            raise ValueError("Unsupported native-session protobuf wire type")
        if pos > len(data):
            raise ValueError("Truncated native-session protobuf field")
    return found


def _database(root: Path, session_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", session_id):
        raise ValueError("Invalid Antigravity session ID")
    directory = root / ".harness-home/.gemini/antigravity-cli/conversations"
    if not directory.resolve().is_relative_to(root.resolve()):
        raise ValueError("Native session directory escapes workspace")
    path = directory / f"{session_id}.db"
    if path.resolve().parent != directory.resolve():
        raise ValueError("Native session database escapes conversation directory")
    return path


def _last_step(root: Path, session_id: str, *, require_quiescent: bool = False):
    path = _database(root, session_id)
    if not path.is_file():
        return path, None
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
        # DONE=3, CANCELED=6, ERROR=7. A completed planner message can
        # coexist with a running background tool; it is not a final answer.
        if require_quiescent and connection.execute(
            "SELECT 1 FROM steps WHERE status NOT IN (3, 6, 7) LIMIT 1"
        ).fetchone():
            return path, None
        row = connection.execute(
            "SELECT idx, step_type, status, step_payload FROM steps ORDER BY idx DESC LIMIT 1"
        ).fetchone()
    return path, row


def last_native_step(root: Path, session_id: str) -> int:
    _, row = _last_step(root, session_id)
    return row[0] if row else -1


def read_native_response(
    root: Path, session_id: str, *, after_step: int = -1, require_quiescent: bool = False
) -> NativeResponse | None:
    path, row = _last_step(root, session_id, require_quiescent=require_quiescent)
    if row is None:
        return None
    idx, step_type, status, payload = row
    # 15 = planner response; 3 = completed. A trailing tool, permission denial,
    # or user message must not cause us to reuse an earlier answer.
    if idx <= after_step or step_type != 15 or status != 3:
        return None
    response = _bytes_field(payload, 20)
    if response is None:
        return None
    answer = _bytes_field(response, 1)
    if answer is None:
        return None
    text = answer.decode("utf-8")
    return NativeResponse(text, idx, path) if text.strip() else None
