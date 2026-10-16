"""Deterministic contracts for the ArXivMath source-generation pipeline.

Abstract triage uses mutable title and abstract metadata. All authoritative
question work is bound to immutable, version-pinned TeX artifacts. This module
contains no network or model calls.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from matharena.arxiv_source import ArxivVersion, parse_arxiv_version
from matharena.parser import WarningType, check_answers, extract_answer, parse_answer


SCHEMA_VERSION = 4
ABSTRACT_SCREEN_FILENAME = "abstract_screen.json"
SOURCE_REFERENCE_FILENAME = "source_ref.json"
INVESTIGATION_FILENAME = "source_investigation.json"
FINAL_ANNOTATION_FILENAME = "llm_annotation.json"

MODEL_SOURCE_CLEANING_VERSION = 1
DEFAULT_INVESTIGATION_MAX_TOKENS = 750_000

NOVELTY_TYPES = {
    "counterexample_to_prior_conjecture",
    "resolves_competing_conjectures",
    "negative_answer_to_open_question",
    "different_from_prior_prediction",
    "confirms_prior_conjecture",
    "new_exact_value",
    "tight_bound",
    "classification",
    "new_formula",
    "other_new_result",
}
MANDATORY_REVIEW_NOVELTY_TYPES = {
    "counterexample_to_prior_conjecture",
    "resolves_competing_conjectures",
    "negative_answer_to_open_question",
    "different_from_prior_prediction",
}
IMPORTANCE_TYPES = {"main", "one_of_multiple_main", "secondary", "minor"}
ANSWER_TYPES = {
    "exact_scalar",
    "exact_symbolic",
    "finite_list",
    "ordered_tuple",
    "finite_set",
    "interval",
}
ABSTRACT_SCREEN_DECISIONS = {"accept", "reject"}

_SOURCE_MARKER_RE = re.compile(
    r"^% MATHARENA_SOURCE_(?P<kind>BEGIN|END) file=(?P<file>.+)$",
    re.MULTILINE,
)
_PAPER_REFERENCE_RE = re.compile(
    r"\b(?:the|this)\s+(?:paper|article|source|abstract|work)\b|" r"\baccording to (?:the|this)\b|\bin this work\b",
    re.IGNORECASE,
)
_UNSAFE_ANSWER_RE = re.compile(
    r"\\(?:aleph|beth|cap|ceil|circ|cup|deg|floor|in|infty|land|lceil|lfloor|lor|"
    r"mathbb|mathcal|mathbf|mathrm|neg|oplus|operatorname|otimes|prod|rceil|rfloor|"
    r"sqcup|sum|text|tilde|vee|wedge)(?=[^A-Za-z]|$)|[∪∩∨∧⊗⊕∞¬⌊⌋⌈⌉]"
)
_ANSWER_NUMBER_RE = re.compile(r"(?<![A-Za-z])-?\d+(?![A-Za-z])")
_ANSWER_VARIABLE_RE = re.compile(r"(?<!\\)(?<![A-Za-z])([A-Za-z])(?![A-Za-z])")

REFUTATION_STATUSES = {
    "not_applicable",
    "question_targets_refutation",
    "source_refutation_not_parser_gradable",
}
VERIFICATION_CHECKS = (
    "source_supported",
    "self_contained",
    "unique_and_well_defined",
    "answer_type_supported",
    "no_missing_context",
    "no_answer_leak",
    "research_substantive",
    "novelty_supported",
    "refutation_supported",
)

_SOURCE_MARKER_LINE_RE = re.compile(r"% MATHARENA_SOURCE_(?:BEGIN|END) file=[^\r\n]*(?:\r\n|\r|\n|$)")
_MODEL_LITERAL_ENVIRONMENT_RE = re.compile(
    r"\\begin\s*\{(?P<name>"
    r"verbatim\*?|Verbatim\*?|BVerbatim|LVerbatim|SaveVerbatim|"
    r"lstlisting\*?|minted\*?|filecontents\*?"
    r")\}"
)
_MODEL_COMMENT_ENVIRONMENT_RE = re.compile(r"\\begin\s*\{comment\}")
_MODEL_INLINE_VERB_RE = re.compile(r"\\verb\*?(?![A-Za-z@])")


class SourceStateError(ValueError):
    """A stored source-generation artifact is missing, stale, or malformed."""


@dataclass(frozen=True)
class InvestigationSource:
    text: str
    raw_source_chars: int

    def manifest_record(self) -> dict[str, Any]:
        return {
            "cleaning_version": MODEL_SOURCE_CLEANING_VERSION,
            "complete_source": True,
            "raw_source_chars": self.raw_source_chars,
            "source_chars": len(self.text),
            "source_sha256": sha256_text(self.text),
            "approximate_tokens": approximate_tokens(self.text),
            "removed_comment_chars": self.raw_source_chars - len(self.text),
        }


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return sha256_bytes(encoded)


def atomic_write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, raw_tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    tmp = Path(raw_tmp)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def load_json(path: str | Path, default: Any = None) -> Any:
    target = Path(path)
    if not target.is_file():
        return default
    try:
        return json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SourceStateError(f"cannot read {target}: {exc}") from exc


def abstract_screen_input(metadata: dict[str, Any]) -> dict[str, str]:
    return {
        "title": str(metadata.get("title") or "").strip(),
        "abstract": str(metadata.get("abstract") or "").strip(),
    }


def abstract_screen_input_sha256(metadata: dict[str, Any]) -> str:
    return canonical_json_sha256(abstract_screen_input(metadata))


def abstract_screen_record_is_current(record: Any, metadata: dict[str, Any]) -> bool:
    if (
        not isinstance(record, dict)
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("status") != "completed"
    ):
        return False
    identity = record.get("identity")
    decision = record.get("decision")
    return bool(
        isinstance(identity, dict)
        and identity.get("metadata_input_sha256") == abstract_screen_input_sha256(metadata)
        and all(
            isinstance(identity.get(field), str) and bool(identity[field])
            for field in ("prompt_sha256", "rendered_prompt_sha256", "model_config_sha256")
        )
        and decision in ABSTRACT_SCREEN_DECISIONS
    )


def abstract_screen_selection(
    paper_root: str | Path,
    *,
    paper_ids: Iterable[str] | None = None,
    screen_filename: str = ABSTRACT_SCREEN_FILENAME,
) -> dict[str, Any]:
    root = Path(paper_root)
    if not root.is_dir():
        ids: list[str] = []
    elif paper_ids is None:
        ids = sorted(path.name for path in root.iterdir() if path.is_dir() and (path / "metadata.json").is_file())
    else:
        ids = sorted(paper_ids)

    accepted: list[str] = []
    rejected: list[str] = []
    incomplete: list[dict[str, str]] = []
    for paper_id in ids:
        paper_dir = root / paper_id
        try:
            metadata = load_json(paper_dir / "metadata.json")
            record = load_json(paper_dir / screen_filename, {})
        except SourceStateError:
            incomplete.append({"paper_id": paper_id, "reason": "metadata_or_abstract_screen_unreadable"})
            continue
        if not isinstance(metadata, dict):
            incomplete.append({"paper_id": paper_id, "reason": "metadata_missing_or_malformed"})
        elif not isinstance(record, dict) or not record:
            incomplete.append({"paper_id": paper_id, "reason": "abstract_screen_missing"})
        elif record.get("status") == "failed":
            incomplete.append({"paper_id": paper_id, "reason": "abstract_screen_failed"})
        elif not abstract_screen_record_is_current(record, metadata):
            incomplete.append({"paper_id": paper_id, "reason": "abstract_screen_stale_or_malformed"})
        elif record["decision"] == "accept":
            accepted.append(paper_id)
        else:
            rejected.append(paper_id)
    return {
        "complete": not incomplete,
        "paper_count": len(ids),
        "accepted": accepted,
        "rejected": rejected,
        "incomplete": incomplete,
    }


def require_current_abstract_acceptances(
    paper_root: str | Path,
    *,
    paper_ids: Iterable[str] | None = None,
    screen_filename: str = ABSTRACT_SCREEN_FILENAME,
) -> tuple[list[str], dict[str, Any]]:
    selection = abstract_screen_selection(
        paper_root,
        paper_ids=paper_ids,
        screen_filename=screen_filename,
    )
    if not selection["complete"]:
        examples = ", ".join(f"{item['paper_id']}:{item['reason']}" for item in selection["incomplete"][:5])
        suffix = "" if len(selection["incomplete"]) <= 5 else ", ..."
        raise SourceStateError(
            f"abstract screening is incomplete for {len(selection['incomplete'])} papers" f" ({examples}{suffix})"
        )
    return list(selection["accepted"]), selection


def resolve_source_cache(value: str | Path | None = None) -> Path:
    raw = value or os.environ.get("ARXIV_SOURCE_CACHE") or "arxivmath/source_cache"
    return Path(raw).expanduser().resolve()


def portable_model_config_reference(
    config_arg: str,
    config_path: str | Path,
    *,
    repository_root: str | Path | None = None,
) -> tuple[str, str]:
    argument = Path(config_arg).name if Path(config_arg).is_absolute() else config_arg
    resolved = Path(config_path).expanduser().resolve()
    root = Path(repository_root or Path.cwd()).resolve()
    try:
        stored_path = resolved.relative_to(root).as_posix()
    except ValueError:
        stored_path = resolved.name
    return argument, stored_path


def metadata_arxiv_base_id(metadata: dict[str, Any]) -> str:
    raw = metadata.get("id")
    if not isinstance(raw, str) or not raw.strip():
        raise SourceStateError("metadata does not contain an arXiv id")
    candidate = raw.strip()
    try:
        if re.search(r"v[1-9]\d*$", candidate):
            return parse_arxiv_version(candidate).base_id
        return parse_arxiv_version(f"{candidate}v1").base_id
    except ValueError as exc:
        raise SourceStateError(f"metadata contains an invalid arXiv id: {exc}") from exc


def metadata_arxiv_version(metadata: dict[str, Any]) -> ArxivVersion:
    raw = metadata.get("versioned_id")
    if not isinstance(raw, str) or not raw.strip():
        raise SourceStateError("metadata does not contain an exact versioned_id; resolve the current revision first")
    try:
        version = parse_arxiv_version(raw.strip())
    except ValueError as exc:
        raise SourceStateError(f"metadata contains an invalid versioned_id: {exc}") from exc
    base_id = metadata_arxiv_base_id(metadata)
    if version.base_id != base_id:
        raise SourceStateError(f"metadata id {base_id!r} does not match versioned_id {version.canonical_id!r}")
    raw_id = str(metadata.get("id") or "").strip()
    if re.search(r"v[1-9]\d*$", raw_id):
        try:
            id_version = parse_arxiv_version(raw_id)
        except ValueError as exc:
            raise SourceStateError(f"metadata contains an invalid arXiv id: {exc}") from exc
        if id_version != version:
            raise SourceStateError(
                f"metadata id {id_version.canonical_id!r} does not match " f"versioned_id {version.canonical_id!r}"
            )
    return version


def source_cache_key(version: ArxivVersion) -> str:
    return version.storage_key


def build_source_reference(
    cache_dir: str | Path,
    manifest: dict[str, Any],
    *,
    allow_unresolved_includes: bool = False,
) -> dict[str, Any]:
    directory = Path(cache_dir)
    manifest_path = directory / "source_manifest.json"
    combined_path = directory / "combined_source.tex"
    if not manifest_path.is_file() or not combined_path.is_file():
        raise SourceStateError("prepared source is missing its manifest or combined TeX")
    stored_manifest = load_json(manifest_path)
    if not isinstance(stored_manifest, dict) or stored_manifest != manifest:
        raise SourceStateError("prepared source manifest does not match the stored manifest")
    version = parse_arxiv_version(str(manifest.get("arxiv_id", "")))
    if directory.name != source_cache_key(version):
        raise SourceStateError("source cache directory does not match the pinned arXiv id")
    combined_sha = sha256_file(combined_path)
    if combined_sha != (manifest.get("tex") or {}).get("combined_sha256"):
        raise SourceStateError("combined source hash does not match its manifest")
    unresolved_includes = (manifest.get("tex") or {}).get("unresolved_includes") or []
    if not isinstance(unresolved_includes, list):
        raise SourceStateError("source manifest has malformed unresolved include metadata")
    if unresolved_includes and not allow_unresolved_includes:
        raise SourceStateError(f"combined source has {len(unresolved_includes)} unresolved TeX include directives")
    return {
        "schema_version": SCHEMA_VERSION,
        "arxiv_id": version.canonical_id,
        "base_id": version.base_id,
        "version": version.version,
        "cache_key": directory.name,
        "manifest_sha256": sha256_file(manifest_path),
        "combined_source_sha256": combined_sha,
        "combined_source_bytes": combined_path.stat().st_size,
        "unresolved_include_count": len(unresolved_includes),
        "unresolved_includes": unresolved_includes,
        "unresolved_includes_allowed": bool(unresolved_includes and allow_unresolved_includes),
        "created_at": utc_now(),
    }


def read_source_artifacts(
    paper_dir: str | Path,
    source_cache: str | Path | None = None,
    *,
    reference_filename: str = SOURCE_REFERENCE_FILENAME,
) -> tuple[str, dict[str, Any], Path]:
    directory = Path(paper_dir)
    reference = load_json(directory / reference_filename)
    if not isinstance(reference, dict):
        raise SourceStateError(f"missing {reference_filename}")
    if reference.get("schema_version") != SCHEMA_VERSION:
        raise SourceStateError("source reference uses a stale or unsupported schema")
    cache_key = reference.get("cache_key")
    if not isinstance(cache_key, str) or not cache_key or PurePosixPath(cache_key).name != cache_key:
        raise SourceStateError("source reference has an unsafe cache key")
    cache_dir = resolve_source_cache(source_cache) / cache_key
    manifest_path = cache_dir / "source_manifest.json"
    combined_path = cache_dir / "combined_source.tex"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise SourceStateError("source manifest is missing")
    if sha256_file(manifest_path) != reference.get("manifest_sha256"):
        raise SourceStateError("source manifest changed after the paper reference was written")
    try:
        source_bytes = combined_path.read_bytes()
    except OSError as exc:
        raise SourceStateError(f"prepared combined source is unavailable: {exc}") from exc
    if sha256_bytes(source_bytes) != reference.get("combined_source_sha256"):
        raise SourceStateError("combined source changed after the paper reference was written")
    try:
        source_text = source_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SourceStateError("prepared combined source is not valid UTF-8") from exc
    if manifest.get("arxiv_id") != reference.get("arxiv_id"):
        raise SourceStateError("source reference and manifest disagree on the arXiv version")
    return (
        source_text,
        {**reference, "source_manifest": manifest, "source_path": str(combined_path)},
        cache_dir,
    )


def paper_source_unavailable(paper_dir: str | Path) -> bool:
    directory = Path(paper_dir)
    ingestion = load_json(directory / "source_ingestion.json", {})
    return bool(
        isinstance(ingestion, dict)
        and ingestion.get("status") == "failed"
        and not (directory / SOURCE_REFERENCE_FILENAME).is_file()
    )


def approximate_tokens(text: str) -> int:
    # TeX is denser than ordinary prose, so two characters per token is a
    # deliberately conservative budgeting estimate.
    return max(1, (len(text) + 1) // 2)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(match.end() for match in re.finditer("\n", text))
    return starts


def _line_at_offset(starts: list[int], offset: int) -> int:
    if offset <= 0:
        return 1
    return bisect.bisect_right(starts, offset - 1)


def _mask_tex_comments(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines(keepends=True):
        comment_at = None
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                comment_at = index
                break
        if comment_at is None:
            output.append(line)
            continue
        suffix = line[comment_at:]
        output.append(line[:comment_at] + "".join("\n" if character == "\n" else " " for character in suffix))
    return "".join(output)


def source_file_at_offset(source_text: str, offset: int) -> str | None:
    stack: list[str] = []
    for match in _SOURCE_MARKER_RE.finditer(source_text, 0, max(0, offset) + 1):
        source_file = match.group("file").split(" via=", 1)[0].strip()
        if match.group("kind") == "BEGIN":
            stack.append(source_file)
        elif stack and stack[-1] == source_file:
            stack.pop()
        elif source_file in stack:
            stack = stack[: stack.index(source_file)]
    return stack[-1] if stack else None


def locate_evidence_quote(
    source_text: str,
    quote: Any,
    *,
    require_unique: bool = True,
    min_chars: int = 20,
    max_chars: int = 12_000,
) -> dict[str, Any]:
    if not isinstance(quote, str) or not quote.strip():
        raise SourceStateError("evidence quote must be a non-empty string")
    if len(quote) < min_chars or len(quote) > max_chars:
        raise SourceStateError(f"evidence quote must contain {min_chars}..{max_chars} characters")
    # Accept LF/CRLF differences only; retain exact source spans and detect
    # overlapping matches, including duplicates with different line endings.
    pattern = r"(?:\r\n|(?<!\r)\n)".join(re.escape(part) for part in re.split(r"\r?\n", quote))
    positions: list[tuple[int, int]] = []
    for match in re.finditer(f"(?=({pattern}))", source_text):
        positions.append(match.span(1))
        if not require_unique or len(positions) > 1:
            break
    if not positions:
        raise SourceStateError("evidence quote is not a verbatim source substring")
    if require_unique and len(positions) != 1:
        raise SourceStateError("evidence quote is not unique in the prepared source")
    start, end = positions[0]
    quote = source_text[start:end]
    starts = _line_starts(source_text)
    return {
        "quote": quote,
        "quote_sha256": sha256_text(quote),
        "start_char": start,
        "end_char": end,
        "start_line": _line_at_offset(starts, start),
        "end_line": _line_at_offset(starts, max(start, end - 1)),
        "source_file": source_file_at_offset(source_text, start),
    }


def source_window(
    source_text: str,
    start_char: int,
    end_char: int,
    context_lines: int = 40,
) -> str:
    if not 0 <= start_char < end_char <= len(source_text):
        raise ValueError("invalid source window range")
    if context_lines < 0:
        raise ValueError("context_lines cannot be negative")
    starts = _line_starts(source_text)
    first_line = max(1, _line_at_offset(starts, start_char) - context_lines)
    last_line = min(len(starts), _line_at_offset(starts, end_char - 1) + context_lines)
    first_char = starts[first_line - 1]
    last_char = starts[last_line] if last_line < len(starts) else len(source_text)
    return source_text[first_char:last_char]


def _environment_ranges(text: str, begin_pattern: re.Pattern[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        begin = begin_pattern.search(text, cursor)
        if begin is None:
            break
        name = begin.groupdict().get("name") or "comment"
        end_pattern = re.compile(rf"\\end\s*\{{{re.escape(name)}\}}")
        end = end_pattern.search(text, begin.end())
        stop = len(text) if end is None else end.end()
        ranges.append((begin.start(), stop))
        cursor = stop
    return ranges


def _inline_verb_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = 0
    while True:
        command = _MODEL_INLINE_VERB_RE.search(text, cursor)
        if command is None:
            break
        delimiter_index = command.end()
        if delimiter_index >= len(text):
            ranges.append((command.start(), len(text)))
            break
        delimiter = text[delimiter_index]
        if delimiter.isspace() or delimiter.isalpha():
            cursor = command.end()
            continue
        line_end = len(text)
        for newline in (
            text.find("\n", delimiter_index + 1),
            text.find("\r", delimiter_index + 1),
        ):
            if newline != -1:
                line_end = min(line_end, newline)
        closing = text.find(delimiter, delimiter_index + 1, line_end)
        stop = line_end if closing == -1 else closing + 1
        ranges.append((command.start(), stop))
        cursor = max(stop, command.end())
    return ranges


def _range_containing(ranges: list[tuple[int, int]], offset: int) -> tuple[int, int] | None:
    for start, end in ranges:
        if start <= offset < end:
            return start, end
        if start > offset:
            break
    return None


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if start >= end:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _newlines_only(value: str) -> str:
    return "".join(character for character in value if character in "\r\n")


def clean_tex_for_model(source_text: str) -> str:
    """Remove non-rendered TeX comments while preserving source structure.

    Synthetic file-boundary markers and literal code environments remain so
    evidence can still be attributed deterministically. Newlines inside
    removed comments are retained, keeping cleaned line numbers aligned with
    the immutable combined source.
    """

    if not isinstance(source_text, str) or not source_text:
        raise ValueError("source_text must be non-empty")
    comment_mask = _mask_tex_comments(source_text)
    literal_ranges = _merge_ranges(
        [
            *(
                value
                for value in _environment_ranges(source_text, _MODEL_LITERAL_ENVIRONMENT_RE)
                if comment_mask[value[0]] == source_text[value[0]]
            ),
            *(value for value in _inline_verb_ranges(source_text) if comment_mask[value[0]] == source_text[value[0]]),
        ]
    )
    comment_ranges = [
        value
        for value in _environment_ranges(comment_mask, _MODEL_COMMENT_ENVIRONMENT_RE)
        if comment_mask[value[0]] == source_text[value[0]] and _range_containing(literal_ranges, value[0]) is None
    ]

    output: list[str] = []
    cursor = 0
    literal_index = 0
    comment_index = 0
    while cursor < len(source_text):
        while literal_index < len(literal_ranges) and literal_ranges[literal_index][1] <= cursor:
            literal_index += 1
        while comment_index < len(comment_ranges) and comment_ranges[comment_index][1] <= cursor:
            comment_index += 1
        if cursor == 0 or source_text[cursor - 1] in "\r\n":
            marker = _SOURCE_MARKER_LINE_RE.match(source_text, cursor)
            if marker is not None:
                output.append(marker.group(0))
                cursor = marker.end()
                continue

        comment_range = (
            comment_ranges[comment_index]
            if comment_index < len(comment_ranges) and comment_ranges[comment_index][0] <= cursor
            else None
        )
        if comment_range is not None:
            _, end = comment_range
            output.append(_newlines_only(source_text[cursor:end]))
            cursor = end
            continue

        literal_range = (
            literal_ranges[literal_index]
            if literal_index < len(literal_ranges) and literal_ranges[literal_index][0] <= cursor
            else None
        )
        if literal_range is not None:
            _, end = literal_range
            output.append(source_text[cursor:end])
            cursor = end
            continue

        character = source_text[cursor]
        if character == "%":
            backslashes = 0
            before = cursor - 1
            while before >= 0 and source_text[before] == "\\":
                backslashes += 1
                before -= 1
            if backslashes % 2 == 0:
                newline = cursor
                while newline < len(source_text) and source_text[newline] not in "\r\n":
                    newline += 1
                cursor = newline
                continue
        output.append(character)
        cursor += 1

    cleaned = "".join(output)
    if not cleaned.strip():
        raise SourceStateError("comment cleaning produced an empty TeX source")
    return cleaned


def build_investigation_source(source_text: str) -> InvestigationSource:
    """Build the complete comment-cleaned TeX input sent to the investigator."""

    return InvestigationSource(
        text=clean_tex_for_model(source_text),
        raw_source_chars=len(source_text),
    )


def _answer_mutations(answer: str) -> Iterable[str]:
    for match in _ANSWER_NUMBER_RE.finditer(answer):
        yield answer[: match.start()] + str(int(match.group(0)) + 1) + answer[match.end() :]
    match = _ANSWER_VARIABLE_RE.search(answer)
    if match:
        replacement = "y" if match.group(1) == "x" else "x"
        yield answer[: match.start()] + replacement + answer[match.end() :]


def canonicalize_answer_syntax(answer: str) -> str:
    return re.sub(r"\\(?:left|right)(?=[^A-Za-z]|$)", "", answer).strip()


def validate_parser_safe_answer(answer: Any) -> dict[str, Any]:
    if not isinstance(answer, str) or not answer.strip():
        return {"keep": False, "reason": "empty_answer"}
    value = canonicalize_answer_syntax(answer)
    if not value:
        return {"keep": False, "reason": "empty_answer"}
    if _UNSAFE_ANSWER_RE.search(value):
        return {"keep": False, "reason": "unsupported_answer_notation"}
    if (":" in value and "{" in value) or "\\mid" in value or "\\;|" in value:
        return {"keep": False, "reason": "set_builder_answer"}
    list_answer = "," in value
    try:
        parsed, warning = parse_answer(value, list_answer=list_answer)
        extracted, extract_warning = extract_answer(
            rf"\boxed{{{value}}}",
            strict_parsing=False,
            parse=True,
            list_answer=list_answer,
        )
    except Exception as exc:
        return {"keep": False, "reason": "parse_exception", "detail": repr(exc)}
    if parsed is None or warning >= WarningType.MAJOR:
        return {"keep": False, "reason": "unparseable_answer", "warning": warning.name}
    if extracted is None or extract_warning >= WarningType.MAJOR or not check_answers(extracted, parsed):
        return {
            "keep": False,
            "reason": "exact_self_grade_failed",
            "warning": extract_warning.name,
        }
    checked_mutations = 0
    for mutated in _answer_mutations(value):
        try:
            mutated_parsed, mutated_warning = parse_answer(mutated, list_answer=list_answer)
        except Exception:
            continue
        if mutated_parsed is None or mutated_warning >= WarningType.MAJOR:
            continue
        checked_mutations += 1
        if check_answers(mutated_parsed, parsed):
            return {
                "keep": False,
                "reason": "parser_insensitive_to_answer_mutation",
                "mutated_answer": mutated,
            }
    return {
        "keep": True,
        "reason": "parser_safe",
        "warning": warning.name,
        "parsed_answer": str(parsed),
        "canonical_answer": value,
        "checked_mutations": checked_mutations,
    }


def answers_equivalent(left: Any, right: Any) -> bool:
    if not isinstance(left, str) or not isinstance(right, str):
        return False
    list_answer = "," in left or "," in right
    try:
        left_parsed, left_warning = parse_answer(left.strip(), list_answer=list_answer)
        right_parsed, right_warning = parse_answer(right.strip(), list_answer=list_answer)
    except Exception:
        return False
    return bool(
        left_parsed is not None
        and right_parsed is not None
        and left_warning < WarningType.MAJOR
        and right_warning < WarningType.MAJOR
        and check_answers(left_parsed, right_parsed)
    )


# BrokenArXiv uses the same source stages and records, with a different item contract.
FALSE_SCREEN_FILENAME = "abstract_screen_false.json"
FALSE_INVESTIGATION_FILENAME = "source_false_investigation.json"
FALSE_FINAL_FILENAME = "llm_metadata_false_source.json"
FALSE_FIELDS = (
    "true_statement",
    "false_statement",
    "falsity_explanation",
    "basis_summary",
    "plausibility_rationale",
    "difficulty_rationale",
    "easy_refutation_audit",
)
FALSE_CLAIM_KINDS = {"disproved_conjecture", "negative_answer", "refuted_prediction"}
FALSE_CHECKS = (
    "true_statement_supported",
    "false_statement_refuted",
    "hypotheses_match",
    "self_contained",
    "natural_claim",
    "main_contribution",
    "novelty_supported",
    "research_difficult",
)


def add_source_mode_argument(parser) -> None:
    parser.add_argument("--false", action="store_true", help="Use the BrokenArXiv false-statement contract.")


def configure_source_mode(args, *, false_model_default: str = "openai/gpt-6-astra-high") -> None:
    if hasattr(args, "model_config") and args.model_config is None:
        args.model_config = false_model_default if args.false else "openai/gpt-6-astra"
    if not args.false:
        return
    if hasattr(args, "allow_source_unavailable"):
        args.allow_source_unavailable = True
    replacements = {
        ABSTRACT_SCREEN_FILENAME: FALSE_SCREEN_FILENAME,
        INVESTIGATION_FILENAME: FALSE_INVESTIGATION_FILENAME,
        FINAL_ANNOTATION_FILENAME: FALSE_FINAL_FILENAME,
        "arxivmath/prompts/source/abstract_screen.md": "arxivmath/prompts/broken/source_screen.md",
        "arxivmath/prompts/source/investigate_source.md": "arxivmath/prompts/broken/source_investigate.md",
        "arxivmath/prompts/source/verify_question.md": "arxivmath/prompts/broken/source_verify.md",
    }
    for field in (
        "abstract_screen_filename",
        "annotation_filename",
        "investigation_filename",
        "final_filename",
        "prompt",
    ):
        value = getattr(args, field, None)
        if value in replacements:
            setattr(args, field, replacements[value])


def validate_source_model(config: dict[str, Any], *, false_mode: bool) -> None:
    if false_mode and (
        config.get("harness") not in {"codex", "codex-cli", "openai-codex"}
        or not re.match(r"^(?:openai/)?gpt-6(?:[.-]|$)", str(config.get("model", "")))
    ):
        raise SourceStateError("BrokenArXiv requires a GPT-6 Codex harness configuration")


def validate_false_payload(payload: Any, source_text: str, *, investigation=None) -> dict[str, Any]:
    """Validate the false-item contract, returning the shared stages' result shape."""
    verification = investigation is not None
    try:
        if not isinstance(payload, dict) or type(payload.get("keep")) is not bool:
            raise SourceStateError("keep must be a JSON boolean")
        if verification:
            fields = {"keep", *FALSE_CHECKS, "reason", "evidence_quotes"}
            strings = ("reason",)
        elif payload["keep"]:
            fields = {
                "keep",
                *FALSE_FIELDS,
                "claim_kind",
                "prior_claim",
                "prior_work_status",
                "importance",
                "evidence_quotes",
            }
            strings = (*FALSE_FIELDS, "prior_claim")
        else:
            fields, strings = {"keep", "basis_summary", "rejection_reason"}, ("basis_summary", "rejection_reason")
        if set(payload) != fields:
            raise SourceStateError("unexpected or missing false-item fields")
        if any(not isinstance(payload[key], str) or not payload[key].strip() for key in strings):
            raise SourceStateError("required text must be nonempty")
        result = dict(payload)
        for key in strings:
            result[key] = result[key].strip()
        if payload["keep"] or verification:
            if verification:
                if any(type(payload[key]) is not bool for key in FALSE_CHECKS):
                    raise SourceStateError("verification checks must be JSON booleans")
                if payload["keep"] != all(payload[key] for key in FALSE_CHECKS):
                    raise SourceStateError("verification decision contradicts its checks")
            else:
                if payload["claim_kind"] not in FALSE_CLAIM_KINDS:
                    raise SourceStateError("claim must be a documented conjecture, open question, or prediction")
                if (
                    payload["importance"] not in ("main", "one_of_multiple_main")
                    or payload["prior_work_status"] != "new_refutation"
                ):
                    raise SourceStateError("false claim must depend on a main new refutation")
                if result["true_statement"] == result["false_statement"]:
                    raise SourceStateError("true and false statements are identical")
                if any(_PAPER_REFERENCE_RE.search(result[key]) for key in ("true_statement", "false_statement")):
                    raise SourceStateError("statement refers to external source context")
            quotes = result.pop("evidence_quotes")
            if not isinstance(quotes, list):
                raise SourceStateError("evidence_quotes must be a list")
            roles = {"result", "proof", "prior_claim", "prior_work", "difficulty"}
            evidence = []
            for entry in quotes:
                if not isinstance(entry, dict) or set(entry) != {"role", "quote"} or entry["role"] not in roles:
                    raise SourceStateError("invalid evidence entry")
                quote = entry["quote"]
                if not isinstance(quote, str) or not quote.strip():
                    raise SourceStateError("evidence quote must be a nonempty string")
                try:
                    located = locate_evidence_quote(source_text, quote)
                except SourceStateError:
                    # Quote locations help review; they do not determine eligibility.
                    located = {"quote": quote, "quote_sha256": sha256_text(quote)}
                evidence.append({"role": entry["role"], **located})
            if payload["keep"] and roles != {entry["role"] for entry in evidence}:
                raise SourceStateError("missing result, proof, prior-claim, prior-work or difficulty evidence")
            if len({(entry["role"], entry["quote_sha256"]) for entry in evidence}) != len(evidence):
                raise SourceStateError("duplicate evidence entries")
            result["evidence"] = evidence
        if verification:
            return {
                "valid": True,
                "passed": payload["keep"],
                "checks_pass": payload["keep"],
                "checks": {key: payload[key] for key in FALSE_CHECKS},
                "reason": result["reason"],
                "evidence": result["evidence"],
            }
        result.update(
            benchmark="brokenarxiv",
            refutation_status="question_targets_refutation" if payload["keep"] else "not_applicable",
        )
        if payload["keep"]:
            result.update(novelty_type=result["claim_kind"], new_result=result["true_statement"])
        result["investigation_sha256"] = investigation_payload_sha256(result)
        return {"valid": True, "investigation": result}
    except (SourceStateError, TypeError, KeyError) as exc:
        return {"valid": False, "reason": str(exc)}


def _check_payload_keys(payload: dict[str, Any], *, keep: bool) -> str | None:
    shared = {"keep", "basis_summary", "refutation_status"}
    allowed = (
        shared
        | {
            "question",
            "answer",
            "answer_type",
            "declared_variables",
            "novelty_type",
            "importance",
            "prior_claim",
            "new_result",
            "evidence_quotes",
        }
        if keep
        else shared | {"rejection_reason"}
    )
    unexpected = sorted(set(payload) - allowed)
    return ",".join(unexpected) if unexpected else None


def validate_investigation_payload(
    payload: Any,
    source_text: str,
    *,
    false_mode: bool = False,
) -> dict[str, Any]:
    """Validate a rejection or exactly one complete source-grounded question."""

    if false_mode:
        return validate_false_payload(payload, source_text)

    if not isinstance(payload, dict):
        return {"valid": False, "reason": "investigation_not_object"}
    if not isinstance(payload.get("keep"), bool):
        return {"valid": False, "reason": "investigation_keep_not_boolean"}
    unexpected = _check_payload_keys(payload, keep=payload["keep"])
    if unexpected:
        return {
            "valid": False,
            "reason": "investigation_has_unexpected_fields",
            "fields": unexpected,
        }
    basis_summary = str(payload.get("basis_summary") or "").strip()
    if not 40 <= len(basis_summary.split()) <= 120:
        return {
            "valid": False,
            "reason": "basis_summary_must_contain_40_to_120_words",
        }
    refutation_status = payload.get("refutation_status")
    if refutation_status not in REFUTATION_STATUSES:
        return {"valid": False, "reason": "invalid_refutation_status"}

    if payload["keep"] is False:
        rejection_reason = str(payload.get("rejection_reason") or "").strip()
        if not rejection_reason:
            return {"valid": False, "reason": "rejected_investigation_missing_reason"}
        if refutation_status == "question_targets_refutation":
            return {"valid": False, "reason": "rejection_has_question_refutation_status"}
        investigation = {
            "keep": False,
            "basis_summary": basis_summary,
            "refutation_status": refutation_status,
            "rejection_reason": rejection_reason,
        }
        investigation["investigation_sha256"] = investigation_payload_sha256(investigation)
        return {"valid": True, "investigation": investigation}

    question = payload.get("question")
    answer = payload.get("answer")
    answer_type = payload.get("answer_type")
    declared_variables = payload.get("declared_variables", [])
    novelty_type = payload.get("novelty_type")
    importance = payload.get("importance")
    if not isinstance(question, str) or not question.strip():
        return {
            "valid": False,
            "reason": "question_must_be_nonempty_string",
        }
    question = question.strip()
    if _PAPER_REFERENCE_RE.search(question):
        return {"valid": False, "reason": "question_references_paper"}
    if answer_type not in ANSWER_TYPES:
        return {"valid": False, "reason": "unsupported_answer_type"}
    if novelty_type not in NOVELTY_TYPES:
        return {"valid": False, "reason": "unsupported_novelty_type"}
    if importance not in IMPORTANCE_TYPES:
        return {"valid": False, "reason": "unsupported_importance"}
    if not isinstance(declared_variables, list) or not all(
        isinstance(value, str) and value.strip() for value in declared_variables
    ):
        return {
            "valid": False,
            "reason": "declared_variables_must_be_nonempty_string_list",
        }
    parser_validation = validate_parser_safe_answer(answer)
    if parser_validation.get("keep") is not True:
        return {
            "valid": False,
            "reason": "answer_not_parser_safe",
            "parser_validation": parser_validation,
        }
    canonical_answer = parser_validation["canonical_answer"]
    normalized_answer = re.sub(r"\s+", "", canonical_answer)
    normalized_question = re.sub(r"\s+", "", question)
    if len(normalized_answer) >= 4 and normalized_answer in normalized_question:
        return {"valid": False, "reason": "answer_appears_verbatim_in_question"}

    raw_quotes = payload.get("evidence_quotes")
    if not isinstance(raw_quotes, list) or not 1 <= len(raw_quotes) <= 4:
        return {
            "valid": False,
            "reason": "evidence_quotes_must_contain_one_to_four_quotes",
        }
    evidence: list[dict[str, Any]] = []
    try:
        for quote in raw_quotes:
            evidence.append(
                locate_evidence_quote(
                    source_text,
                    quote,
                    require_unique=False,
                    min_chars=1,
                )
            )
    except SourceStateError as exc:
        return {"valid": False, "reason": "invalid_evidence", "detail": str(exc)}

    prior_claim = payload.get("prior_claim")
    new_result = payload.get("new_result")
    targets_refutation = novelty_type in MANDATORY_REVIEW_NOVELTY_TYPES
    if targets_refutation:
        if refutation_status != "question_targets_refutation":
            return {
                "valid": False,
                "reason": "refutation_question_has_invalid_disposition",
            }
        if not isinstance(prior_claim, str) or not prior_claim.strip():
            return {"valid": False, "reason": "refutation_question_missing_prior_claim"}
        if not isinstance(new_result, str) or not new_result.strip():
            return {"valid": False, "reason": "refutation_question_missing_new_result"}
        prior_claim = prior_claim.strip()
        new_result = new_result.strip()
    elif refutation_status != "not_applicable":
        return {
            "valid": False,
            "reason": "non_refutation_question_has_invalid_disposition",
        }
    else:
        prior_claim = None
        new_result = None

    investigation = {
        "keep": True,
        "question": question,
        "answer": canonical_answer,
        "answer_type": answer_type,
        "declared_variables": [value.strip() for value in declared_variables],
        "basis_summary": basis_summary,
        "novelty_type": novelty_type,
        "importance": importance,
        "refutation_status": refutation_status,
        "prior_claim": prior_claim,
        "new_result": new_result,
        "evidence": evidence,
    }
    investigation["investigation_sha256"] = investigation_payload_sha256(investigation)
    return {
        "valid": True,
        "investigation": investigation,
        "parser_validation": parser_validation,
    }


def investigation_payload_sha256(investigation: dict[str, Any]) -> str:
    if investigation.get("benchmark") == "brokenarxiv":
        return canonical_json_sha256(
            {key: value for key, value in investigation.items() if key != "investigation_sha256"}
        )
    immutable = {
        key: investigation.get(key)
        for key in (
            "keep",
            "question",
            "answer",
            "answer_type",
            "declared_variables",
            "basis_summary",
            "novelty_type",
            "importance",
            "refutation_status",
            "prior_claim",
            "new_result",
            "evidence",
            "rejection_reason",
        )
    }
    return canonical_json_sha256(immutable)


def validate_normalized_investigation(
    investigation: Any,
    source_text: str,
) -> dict[str, Any]:
    """Revalidate a stored normalized investigation using the generation contract."""

    if not isinstance(investigation, dict):
        return {"valid": False, "reason": "investigation_not_object"}
    false_mode = investigation.get("benchmark") == "brokenarxiv"
    excluded_fields = {"investigation_sha256"} | (
        {"benchmark", "refutation_status", "novelty_type", "new_result"} if false_mode else set()
    )
    payload = {key: value for key, value in investigation.items() if key not in excluded_fields}
    if investigation.get("keep") is True:
        evidence = payload.pop("evidence", None)
        if isinstance(evidence, list):
            payload["evidence_quotes"] = [
                (
                    ({"role": record.get("role"), "quote": record.get("quote")} if false_mode else record.get("quote"))
                    if isinstance(record, dict)
                    else record
                )
                for record in evidence
            ]
        else:
            payload["evidence_quotes"] = evidence
    validation = validate_investigation_payload(
        payload,
        source_text,
        false_mode=false_mode,
    )
    if validation.get("valid") is not True:
        return validation
    if validation.get("investigation") != investigation:
        return {
            "valid": False,
            "reason": "normalized_investigation_is_not_canonical",
        }
    return {"valid": True}


def preserve_review(existing: Any, investigation_sha256: str) -> dict[str, Any] | None:
    if not isinstance(existing, dict):
        return None
    source_first = existing.get("source_first") or {}
    review = existing.get("review")
    if (
        isinstance(review, dict)
        and review.get("status") in {"keep", "discard"}
        and source_first.get("investigation_sha256") == investigation_sha256
    ):
        return review
    return None


def final_annotation(
    state: dict[str, Any],
    *,
    state_sha256: str,
    existing: Any,
) -> dict[str, Any]:
    investigation = state["investigation"]
    false_mode = investigation.get("benchmark") == "brokenarxiv"
    investigation_sha256 = investigation_payload_sha256(investigation)
    common_source = {
        "investigation_sha256": investigation_sha256,
        "investigation_state_sha256": state_sha256,
        "source_sha256": (state.get("identity") or {}).get("source_sha256"),
        "source_input": state.get("source_input"),
        "basis_summary": investigation.get("basis_summary"),
        "refutation_status": investigation.get("refutation_status"),
        **({"review_binding": state_sha256} if false_mode else {}),
    }
    if investigation.get("keep") is not True:
        return {
            "source_first_schema_version": SCHEMA_VERSION,
            "keep": False,
            "stage": "no_question",
            "source_first": {
                **common_source,
                "rejection_reason": investigation.get("rejection_reason"),
            },
            "updated_at": utc_now(),
        }
    verification = state.get("verification") or {}
    verification_summary = {
        key: value for key, value in verification.items() if key not in {"raw", "parsed", "detailed_cost"}
    }
    source_first = {
        **common_source,
        "novelty_type": investigation.get("novelty_type"),
        "importance": investigation.get("importance"),
        "prior_claim": investigation.get("prior_claim"),
        "new_result": investigation.get("new_result"),
        "evidence": investigation.get("evidence"),
        "verification": verification_summary,
    }
    if verification.get("status") != "passed":
        return {
            "source_first_schema_version": SCHEMA_VERSION,
            "keep": False,
            "stage": "automatic_verification_rejected",
            "source_first": source_first,
            "updated_at": utc_now(),
        }
    final: dict[str, Any] = {
        "source_first_schema_version": SCHEMA_VERSION,
        "keep": True,
        "stage": "selected_for_human_review",
        **(
            {key: investigation[key] for key in FALSE_FIELDS}
            if false_mode
            else {
                "question": investigation["question"],
                "answer": investigation["answer"],
                "answer_type": investigation["answer_type"],
                "declared_variables": investigation.get("declared_variables") or [],
            }
        ),
        "basis_summary": investigation["basis_summary"],
        "source_first": source_first,
        "updated_at": utc_now(),
    }
    review = preserve_review(existing, investigation_sha256)
    if review is not None and (not false_mode or review.get("binding") == state_sha256):
        final["review"] = review
        final["keep"] = review["status"] == "keep"
        final["stage"] = "human_accepted" if final["keep"] else "human_rejected"
    return final


def sync_review_annotation(
    paper_dir: str | Path,
    *,
    investigation_filename: str = INVESTIGATION_FILENAME,
    final_filename: str = FINAL_ANNOTATION_FILENAME,
) -> dict[str, Any]:
    """Keep the review item in sync with saved generation and verification results."""
    paper_dir = Path(paper_dir)
    final_path = paper_dir / final_filename
    existing = load_json(final_path, {})
    if paper_source_unavailable(paper_dir):
        if existing.get("stage") != "source_unavailable":
            existing = {"source_first_schema_version": SCHEMA_VERSION, "keep": False, "stage": "source_unavailable"}
            atomic_write_json(final_path, existing)
        return existing
    state_path = paper_dir / investigation_filename
    if not state_path.is_file():
        return existing
    state_bytes = state_path.read_bytes()
    state_sha256 = sha256_bytes(state_bytes)
    if (existing.get("source_first") or {}).get("investigation_state_sha256") == state_sha256:
        return existing
    state = json.loads(state_bytes)
    investigation = state.get("investigation") or {}
    verification = state.get("verification") or {}
    source_first = {"investigation_state_sha256": state_sha256}
    if state.get("status") == "excluded" and state.get("reason") == "source_too_large":
        final = {
            "keep": False,
            "stage": "source_oversized",
            "source_first": {
                **source_first,
                "source_sha256": (state.get("identity") or {}).get("source_sha256"),
                "source_input": state.get("source_input"),
                "rejection_reason": "source_too_large",
            },
        }
    elif state.get("status") != "completed" or not investigation:
        final = {"keep": False, "stage": "awaiting_investigation", "source_first": source_first}
    elif investigation.get("investigation_sha256") != investigation_payload_sha256(investigation):
        final = {"keep": False, "stage": "stale_investigation", "source_first": source_first}
    elif investigation.get("keep") and (
        verification.get("status") not in {"passed", "rejected"}
        or verification.get("investigation_sha256") != investigation["investigation_sha256"]
        or (verification.get("status") == "passed" and verification.get("passed") is not True)
    ):
        final = {"keep": False, "stage": "awaiting_verification", "source_first": source_first}
    else:
        final = final_annotation(state, state_sha256=state_sha256, existing=existing)
    final.setdefault("source_first_schema_version", SCHEMA_VERSION)
    final.setdefault("updated_at", utc_now())
    atomic_write_json(final_path, final)
    return final


def investigation_verification_is_current(
    state: Any,
    *,
    model_config_sha256: str,
    prompt_sha256: str,
    input_sha256: str,
) -> bool:
    if not isinstance(state, dict) or not isinstance(state.get("investigation"), dict):
        return False
    investigation = state["investigation"]
    verification = state.get("verification")
    return bool(
        isinstance(verification, dict)
        and verification.get("status") in {"passed", "rejected"}
        and verification.get("investigation_sha256") == investigation_payload_sha256(investigation)
        and verification.get("model_config_sha256") == model_config_sha256
        and verification.get("prompt_sha256") == prompt_sha256
        and verification.get("input_sha256") == input_sha256
    )


def model_record(
    *,
    model: str,
    config_arg: str,
    config_path: str,
    config_sha256: str,
    raw: str,
    parsed: Any,
    cost: Any,
) -> dict[str, Any]:
    detailed = cost if isinstance(cost, dict) else {"cost": cost or 0.0}
    try:
        numeric_cost = float(detailed.get("cost", 0.0) or 0.0)
    except (TypeError, ValueError):
        numeric_cost = 0.0
    return {
        "model": model,
        "model_config": config_arg,
        "model_config_path": config_path,
        "model_config_sha256": config_sha256,
        "status": "completed",
        "raw": raw,
        "parsed": parsed,
        "cost": numeric_cost,
        "detailed_cost": detailed,
        "updated_at": utc_now(),
    }


__all__ = [
    "ABSTRACT_SCREEN_DECISIONS",
    "ABSTRACT_SCREEN_FILENAME",
    "ANSWER_TYPES",
    "DEFAULT_INVESTIGATION_MAX_TOKENS",
    "FINAL_ANNOTATION_FILENAME",
    "IMPORTANCE_TYPES",
    "INVESTIGATION_FILENAME",
    "InvestigationSource",
    "MANDATORY_REVIEW_NOVELTY_TYPES",
    "NOVELTY_TYPES",
    "REFUTATION_STATUSES",
    "SCHEMA_VERSION",
    "SOURCE_REFERENCE_FILENAME",
    "SourceStateError",
    "VERIFICATION_CHECKS",
    "abstract_screen_input",
    "abstract_screen_input_sha256",
    "abstract_screen_record_is_current",
    "abstract_screen_selection",
    "answers_equivalent",
    "approximate_tokens",
    "atomic_write_json",
    "build_investigation_source",
    "build_source_reference",
    "canonicalize_answer_syntax",
    "canonical_json_sha256",
    "clean_tex_for_model",
    "investigation_payload_sha256",
    "investigation_verification_is_current",
    "load_json",
    "locate_evidence_quote",
    "metadata_arxiv_base_id",
    "metadata_arxiv_version",
    "MODEL_SOURCE_CLEANING_VERSION",
    "model_record",
    "paper_source_unavailable",
    "portable_model_config_reference",
    "read_source_artifacts",
    "require_current_abstract_acceptances",
    "resolve_source_cache",
    "sha256_file",
    "sha256_text",
    "source_cache_key",
    "source_file_at_offset",
    "source_window",
    "utc_now",
    "validate_investigation_payload",
    "validate_normalized_investigation",
    "validate_parser_safe_answer",
]
