"""Export source-token and container histories without private telemetry fields."""

from __future__ import annotations

import json
import hashlib
import gzip
from pathlib import Path, PurePosixPath
import re
import tarfile

from leanlean.attempt_timing import model_window
from leanlean.common import attempt_chain


def memoized_counter(count):
    """Share token counts across identical files without retaining their text."""
    counts = {}
    def cached(text):
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        if digest not in counts:
            counts[digest] = count(text)
        return counts[digest]
    return cached


def read_rows(path: Path):
    if path.exists():
        with path.open() as stream:
            for line in stream:
                if line.strip():
                    # A sampler stopped mid-write leaves an unterminated last line; skip only that.
                    if not line.endswith("\n"):
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            pass
                        continue
                    yield json.loads(line)


def snapshot_source(attempt: Path, frame: dict, name: str) -> str:
    """Recover exact text from the independently hashed checkpoint archive."""
    archive = (attempt / frame["archive"]).resolve()
    if not archive.is_relative_to(attempt.resolve()):
        raise ValueError("Checkpoint archive escapes its attempt")
    with gzip.open(archive, "rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != frame["source_sha256"]:
            raise ValueError("Checkpoint source archive checksum differs")
    with tarfile.open(archive) as stream:
        member = stream.getmember(name)
        if not member.isfile():
            raise ValueError("Scored checkpoint source is not a regular file")
        return stream.extractfile(member).read().decode("utf-8")


def apply_diff(before: str, diff: str) -> str:
    """Apply a monitor unified diff, checking every context and removed line."""
    lines = diff.splitlines(keepends=True)
    # The monitor emits the standard marker after a line lacking a newline.
    normalized = []
    for line in lines:
        if line.startswith("\\ No newline at end of file"):
            normalized[-1] = normalized[-1].removesuffix("\n")
        else:
            normalized.append(line)
    old = before.splitlines(keepends=True)
    result, cursor, index = [], 0, 0
    while index < len(normalized):
        match = re.match(r"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", normalized[index])
        if not match:
            index += 1
            continue
        start, size = int(match[1]), int(match[2] or 1)
        offset = start - 1 if size else start
        if not cursor <= offset <= len(old):
            raise ValueError("Checkpoint hunk is outside its source")
        result.extend(old[cursor:offset])
        cursor, index = offset, index + 1
        removed = added = 0
        while index < len(normalized) and not normalized[index].startswith("@@ "):
            line = normalized[index]
            if line.startswith((" ", "-")):
                if cursor >= len(old) or old[cursor] != line[1:]:
                    raise ValueError("Checkpoint context differs from its parent")
                cursor += 1
                removed += 1
            if line.startswith((" ", "+")):
                result.append(line[1:])
                added += 1
            index += 1
        if removed != size or added != int(match[4] or 1):
            raise ValueError("Checkpoint hunk counts do not reconcile")
    result.extend(old[cursor:])
    return "".join(result)


def score_history(attempt: Path, entry: dict, baseline: int, count, start: int, end: int):
    """Replay only scored text, recounting whole changed files for comment safety."""
    sources, totals, points = {}, {}, []
    project = PurePosixPath(entry["project_path"] or ".")
    parent = None
    for frame in read_rows(attempt / "checkpoints.jsonl"):
        if frame["started_monotonic_ns"] > end:
            break
        if frame.get("parent_sha256") != parent:
            raise ValueError("Checkpoint source chain is incomplete")
        path = (attempt / frame["diff"]).resolve()
        if not path.is_relative_to(attempt.resolve()):
            raise ValueError("Checkpoint diff escapes its attempt")
        # The diff encodes original line endings, including lone carriage
        # returns. Universal-newline conversion would corrupt those hunks.
        text = path.read_bytes().decode("utf-8")
        # Split only actual headers, not matching text inside a Lean source line.
        for block in re.split(r"^diff --leanlean ", text, flags=re.MULTILINE)[1:]:
            header, _, body = block.partition("\n")
            name = json.loads(header)
            source = PurePosixPath(name)
            if source.suffix != ".lean" or source.name == "lakefile.lean" or name == entry["challenge_path"]:
                continue
            try:
                relative = source.relative_to(project)
            except ValueError:
                continue
            if any(part.startswith(".") for part in relative.parts):
                continue
            metadata, _, diff = body.partition("\n")
            if "-> ('symlink'," in metadata:
                raise ValueError("Scored source became a symlink")
            if metadata.endswith("-> None"):
                sources.pop(name, None)
                totals.pop(name, None)
            else:
                try:
                    sources[name] = apply_diff(sources.get(name, ""), diff)
                except ValueError:
                    # Exotic source line separators can make unified diffs
                    # ambiguous. The hashed full snapshot is authoritative.
                    sources[name] = snapshot_source(attempt, frame, name)
                totals[name] = count(sources[name])
        parent = frame["source_sha256"]
        elapsed = max(0, (frame["started_monotonic_ns"] - start) / 1e9)
        tokens = sum(totals.values())
        point = dict(elapsed=round(elapsed, 3), compression=round(100 * (baseline - tokens) / baseline, 5))
        if not points or elapsed == 0:
            points = [point]
        else:
            points.append(point)
    # Resumed attempts can start from already compressed source. Keep that
    # measured opening value rather than inventing a zero-score checkpoint.
    # A known source state persists between changes until the attempt ends.
    if points:
        points.append({**points[-1], "elapsed": round((end - start) / 1e9, 3)})
    return points


def chain_offsets(chain: list[Path]) -> dict[Path, float]:
    """Model time at which each attempt of a continuation chain starts."""
    offsets, total = {}, 0.0
    for attempt in chain:
        offsets[attempt] = total
        start, end, _ = model_window(attempt)
        total += (end - start) / 1e9
    return offsets


def limit_fallback_marks(attempt: Path, fallback: dict) -> dict:
    """When the run was cut and when the scored checkpoint was last held, in model time.

    The cut is in the scored attempt; the scored state may come from an attempt it continues.
    """
    offsets = chain_offsets(attempt_chain(attempt))

    def elapsed(directory, index):
        observations = {row["observation_index"]: row for row in read_rows(directory / "observations.jsonl")}
        start, _, _ = model_window(directory)
        return round(offsets[directory] + max(0, (observations[index]["started_monotonic_ns"] - start) / 1e9), 3)

    scored_attempt = Path(fallback.get("scored_attempt", attempt))
    if scored_attempt not in offsets:
        raise ValueError(f"Scored state is outside the attempt's continuation chain: {scored_attempt}")
    # Records whose scored state comes from an earlier attempt name no cut source; the cut is the submission.
    is_cut = (fallback["scored_source_sha256"] == fallback["cut_source_sha256"] if "cut_source_sha256" in fallback
              else fallback["scored_is_submission"])
    return dict(reason=fallback["reason"], limit_at=fallback["limit_at"],
                cut_elapsed=elapsed(attempt, fallback["cut_observation"]),
                scored_elapsed=elapsed(scored_attempt, fallback["scored_last_observation"]),
                scored_is_cut=is_cut)


def scored_point(score: list[dict], marks: dict | None) -> dict | None:
    """The history point holding the scored source: the last change at or before it."""
    if not score:
        return None
    if marks is None:
        return score[-1]
    return [row for row in score if row["elapsed"] <= marks["scored_elapsed"] + 0.001][-1]


def join_restart(restart: dict, progress: dict) -> dict:
    """Prefix a restarted attempt's history with the conversation it replaced.

    A harness restarted after the input context limit continues in the same
    workspace with the remaining budget, so its model time starts where the
    first conversation's ended.
    """
    return dict(joined(restart["progress"], progress, restart["elapsed_s"]), restart_offset_s=restart["elapsed_s"])


def joined(first: dict, progress: dict, offset: float) -> dict:
    """`progress` continued after `first`, which took `offset` seconds of model time."""
    if first["score"] and progress["score"] and abs(first["score"][-1]["compression"] - progress["score"][0]["compression"]) > 1e-4:
        raise ValueError("History does not continue from the earlier history's final source")
    def shift(rows):
        return [dict(row, elapsed=round(row["elapsed"] + offset, 3)) for row in rows]
    limits = {key: max(first["limits"].get(key, 0), progress["limits"].get(key, 0))
              for key in first["limits"].keys() | progress["limits"].keys()}
    # The earlier history's carried endpoint is the continuation's opening point.
    return dict(schema_version=1, score=first["score"][:-1] + shift(progress["score"]),
                resources=first["resources"] + shift(progress["resources"]), limits=limits)


def continued_progress(attempt: Path, entry: dict, baseline: int, count) -> dict:
    """The history of an attempt and the attempts it continues, on one model-time axis."""
    chain = attempt_chain(attempt)
    offsets = chain_offsets(chain)
    progress = attempt_progress(chain[0], entry, baseline, count)
    for later in chain[1:]:
        progress = joined(progress, attempt_progress(later, entry, baseline, count), offsets[later])
    return progress


def attempt_progress(attempt: Path, entry: dict, baseline: int, count) -> dict:
    observations = list(read_rows(attempt / "observations.jsonl"))
    boundaries = {r["reason"]: r["started_monotonic_ns"] for r in observations
                  if r["reason"] in {"model_start", "model_end"}}
    if len(boundaries) != 2:
        return dict(schema_version=1, score=[], resources=[], limits={})
    start, end, _ = model_window(attempt, observations)
    limits, buckets = {}, {}
    for row in read_rows(attempt / "resources.jsonl"):
        stamp = row.get("sampled_monotonic_ns", row.get("observed_monotonic_ns", 0))
        if row.get("phase") != "model" or row.get("status") != "sample" or not start <= stamp <= end:
            continue
        elapsed = (stamp - start) / 1e9
        point = dict(elapsed=round(elapsed, 3),
                     ram_gib=row["memory_current_bytes"] / 2**30 if row.get("memory_current_bytes") is not None else None,
                     cpu_percent=row.get("cpu_percent"))
        if row.get("memory_limit_bytes"):
            limits["ram_gib"] = max(limits.get("ram_gib", 0), row["memory_limit_bytes"] / 2**30)
        if row.get("cpu_limit_cores"):
            limits["cpu_percent"] = max(limits.get("cpu_percent", 0), row["cpu_limit_cores"] * 100)
        # Keep each minute's first/last and both metrics' extrema, preserving
        # short spikes without shipping one-second kernel samples to browsers.
        slot = buckets.setdefault(int(elapsed // 60), {})
        slot.setdefault("first", point)
        slot["last"] = point
        for field in ("ram_gib", "cpu_percent"):
            if point[field] is not None:
                for kind, choose in (("min", min), ("max", max)):
                    key = field + kind
                    slot[key] = choose(slot.get(key, point), point, key=lambda p: p[field])
    resources = {p["elapsed"]: p for slot in buckets.values() for p in slot.values()}
    return dict(schema_version=1, score=score_history(attempt, entry, baseline, count, start, end),
                resources=[resources[t] for t in sorted(resources)], limits=limits)

def recorded_cost_breakdown(details: list[dict], total: float) -> dict | None:
    """Recover a flat-price allocation from the token ledger saved with a result."""
    from math import isclose
    from matharena.token_pricing import TokenPricing

    amounts = dict(uncached_input=0.0, cached_input=0.0, cache_write=0.0, output=0.0)
    tokens = dict.fromkeys(amounts, 0)
    for detail in details:
        required = ("input_tokens", "cached_input_tokens", "output_tokens", "read_cost", "write_cost", "cost")
        if any(detail.get(key) is None for key in required) or detail.get("long_context_pricing"):
            return None
        pricing = TokenPricing.from_config(detail)
        cached = detail["cached_input_tokens"]
        counts = dict(uncached_input=detail["input_tokens"] - cached, cached_input=cached,
                      cache_write=detail.get("cached_write_tokens", 0) or 0, output=detail["output_tokens"])
        if any(count < 0 for count in counts.values()):
            raise ValueError("Invalid recorded token counts")
        rates = dict(uncached_input="read_cost", cached_input="cache_read_cost",
                     cache_write="cache_write_cost", output="write_cost")
        priced = {key: count * float(pricing.base[rates[key]]) / 1e6 for key, count in counts.items()}
        if not isclose(sum(priced.values()), detail["cost"], rel_tol=1e-9, abs_tol=1e-7):
            raise ValueError("Recorded token costs do not reconcile with the saved cost")
        for key in amounts:
            amounts[key] += priced[key]
            tokens[key] += counts[key]
    if not details:
        return None
    if not isclose(sum(amounts.values()), total, rel_tol=1e-9, abs_tol=1e-7):
        raise ValueError("Recorded token costs do not reconcile with the total cost")
    return dict(amounts=amounts, tokens=tokens, reasoning_separate=False)


def cost_breakdown(attempt: Path, total: float, rates: dict) -> dict | None:
    """Price native usage with output and reasoning in one category for every model."""
    from datetime import datetime
    from matharena.token_pricing import TokenPricing

    pricing = TokenPricing.from_config(rates)
    harness = json.loads((attempt / "attempt.json").read_text())["harness"]
    boundaries = {r["reason"]: r["started_wall_ns"] / 1e9 for r in read_rows(attempt / "observations.jsonl")
                  if r["reason"] in {"model_start", "model_end"}}
    if len(boundaries) != 2:
        return None
    records = {}

    def add(key, when, usage, anthropic=False):
        try:
            stamp = datetime.fromisoformat(when.replace("Z", "+00:00")).timestamp()
        except (TypeError, ValueError, AttributeError):
            return
        if boundaries["model_start"] <= stamp <= boundaries["model_end"]:
            records[key] = (stamp, usage, anthropic)

    native = attempt / "native"
    if harness == "codex":
        for path in sorted(native.rglob("rollout-*.jsonl")):
            previous = {}
            for record in read_rows(path):
                payload = record.get("payload") or {}
                if payload.get("type") != "token_count":
                    continue
                usage = (payload.get("info") or {}).get("total_token_usage")
                if not usage:
                    continue
                # Native counters are cumulative within each session.
                delta = {key: max(0, value - previous.get(key, 0)) for key, value in usage.items()
                         if isinstance(value, (int, float))}
                previous = usage
                add((path.name, record.get("timestamp")), record.get("timestamp"), delta)
    elif harness == "claude":
        for path in sorted((native / ".harness-home/.claude").rglob("*.jsonl")):
            for record in read_rows(path):
                message = record.get("message") or {}
                if record.get("type") == "assistant" and message.get("usage"):
                    # Streaming records repeat a message; retain its final usage.
                    add((record.get("sessionId"), message.get("id")), record.get("timestamp"), message["usage"], True)
    elif harness in {"gravity", "gemini", "antigravity"}:
        # Terminal usage covers retries whose native transcripts may be replaced.
        for i, record in enumerate(read_rows(attempt / "harness-events.jsonl")):
            event = record.get("event") or {}
            if event.get("type") == "usage" and isinstance(event.get("content"), dict):
                add(i, event.get("timestamp"), event["content"])
    amounts = dict(uncached_input=0.0, cached_input=0.0, cache_write=0.0, output=0.0)
    tokens = dict.fromkeys(amounts, 0)
    for _, usage, anthropic in records.values():
        cached = usage.get("cache_read_input_tokens", usage.get("cached_input_tokens", usage.get("cache_read_tokens", 0))) or 0
        inputs = (usage.get("input_tokens", 0) or 0) + (cached if anthropic else 0)
        cached = min(inputs, max(0, cached))
        # Codex cache writes are already billed as uncached input.
        writes = usage.get("cache_creation_input_tokens", 0) or 0
        output = usage.get("output_tokens", 0) or 0
        selected = pricing.long_context if pricing.threshold and inputs + writes > pricing.threshold else pricing.base
        counts = dict(uncached_input=max(0, inputs - cached), cached_input=cached, cache_write=writes,
                      output=output)
        for key, count in counts.items():
            rate = selected[{"uncached_input": "read_cost", "cached_input": "cache_read_cost",
                             "cache_write": "cache_write_cost"}.get(key, "write_cost")]
            amounts[key] += count * float(rate) / 1e6
            tokens[key] += count
    measured = sum(amounts.values())
    if measured <= 0:
        return None
    # A partial native capture cannot support a full allocation of recorded cost.
    # Keep the unexplained amount explicit; never silently rescale measured costs.
    if measured > total + max(0.005, total * 0.001):
        return None
    if total > measured + 0.005:
        amounts["other"] = total - measured
    return dict(amounts=amounts, tokens=tokens, reasoning_separate=False)


def cost_history(attempt: Path, total: float, rates: dict) -> dict:
    """Estimate the cumulative shape from priced usage; retain the recorded total."""
    from datetime import datetime
    from matharena.token_pricing import TokenPricing
    pricing = TokenPricing.from_config(rates)
    boundaries = {r['reason']: r for r in read_rows(attempt / 'observations.jsonl')
                  if r['reason'] in {'model_start', 'model_end'}}
    if len(boundaries) != 2:
        return dict(points=[], note='Cost history unavailable', breakdown=None)
    start = boundaries['model_start']['started_wall_ns'] / 1e9
    end = boundaries['model_end']['started_wall_ns'] / 1e9
    adjustment = None
    if all('started_monotonic_ns' in row for row in boundaries.values()):
        _, _, adjustment = model_window(attempt)
    if adjustment:
        end = adjustment['end_wall_ns'] / 1e9
    harness = json.loads((attempt / 'attempt.json').read_text())['harness']
    breakdown = cost_breakdown(attempt, total, rates)
    samples = {}

    def stamp(value):
        try:
            return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()
        except (ValueError, TypeError, AttributeError):
            return None

    def add(key, when, usage, anthropic=False, gemini_native=False):
        t = stamp(when)
        if t is None or not start <= t <= end:
            return
        cached = usage.get('cache_read_input_tokens', usage.get('cached_input_tokens', usage.get('cache_read_tokens', 0))) or 0
        inputs = (usage.get('input_tokens', 0) or 0) + (cached if anthropic or gemini_native else 0)
        amount = pricing.cost(inputs, usage.get('output_tokens', 0) or 0, cached,
                              usage.get('cache_creation_input_tokens', 0) or 0)
        samples[key] = (t - start, amount)

    native = attempt / 'native'
    cumulative = harness == 'codex'
    if cumulative:
        for path in sorted(native.rglob('rollout-*.jsonl')):
            for record in read_rows(path):
                payload = record.get('payload', {})
                if payload.get('type') == 'token_count':
                    info = payload.get('info') or {}
                    usage = info.get('total_token_usage')
                    if usage:
                        add((record.get('timestamp'), json.dumps(usage, sort_keys=True)), record.get('timestamp'), usage)
    elif harness == 'claude':
        for path in sorted((native / '.harness-home/.claude').rglob('*.jsonl')):
            for record in read_rows(path):
                message = record.get('message') or {}
                if record.get('type') == 'assistant' and message.get('usage'):
                    add((record.get('sessionId'), message.get('id')), record.get('timestamp'), message['usage'], True)
    elif harness in {'gravity', 'gemini', 'antigravity'}:
        for path in sorted(native.rglob('transcript_full.jsonl')):
            for record in read_rows(path):
                if record.get('type') == 'PLANNER_RESPONSE':
                    add((str(path), record.get('step_index')), record.get('created_at'), record, gemini_native=True)
    ordered = sorted(samples.values())
    # No intermediate charges were recorded: show the final observation alone.
    if not ordered or sum(v for _, v in ordered) <= 0:
        return dict(points=[dict(elapsed=round(end-start, 3), cost=total)], note='Only final cost recorded', breakdown=breakdown)
    weighted, running = [], 0
    for elapsed, amount in ordered:
        running = max(running, amount) if cumulative else running + amount
        weighted.append((elapsed, running))
    points = [dict(elapsed=0, cost=0)]
    points += [dict(elapsed=round(t, 3), cost=round(total * weight / running, 8)) for t, weight in weighted]
    points.append(dict(elapsed=round(end-start, 3), cost=total))
    return dict(points=points, note='Estimated from token usage · recorded final total', breakdown=breakdown)
