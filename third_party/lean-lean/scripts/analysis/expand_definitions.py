#!/usr/bin/env python3
"""Price the definitions a model introduced, by writing them back out.

A ``def`` is a name for a repeated expression.  Unfolding one is delta-reduction:
``f a b`` and its body are definitionally equal, so the kernel accepts the same
proof either way and nothing that was proved changes.  By that test naming a
repeated expression is a re-spelling, which is why definitions are measured here
rather than left inside structural change -- structural change is then what it
claims to be, new *theorems* proved and reused.

The measurement is the token count of a tree in which every new definition is
inlined at its call sites and the definitions themselves are deleted.  Nothing is
elaborated: the question is how many tokens the abstraction saved, and that is a
property of the text.

Theorems are never inlined.  A theorem is a proved fact, and splicing its proof
into each use site substitutes proofs rather than unfolding a definition -- a
different operation, and the one that would eat the structural-change bucket it
is supposed to be measured against.

Approximations, all of which understate the saving:

* Only non-recursive definitions are handled, because a recursive one has no
  finite unfolding.
* Only explicit binders become parameters.  Implicit and instance arguments do
  not appear at a call site, so they stay as free names in the expansion; the
  token count is close but the text would not elaborate.
* Definitions written as pattern-matching equations, or with a ``where`` block,
  are reported rather than inlined.
* A name mentioned by a tactic -- ``simp [f]``, ``unfold f`` -- is left alone,
  so those sites keep the short spelling.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.metrics.tokens import (  # noqa: E402
    count_lean_tokens_in_source, remove_lean_comments,
)
import scripts.analysis.expand_macros as pattern_pass  # noqa: E402
from scripts.analysis.quantify_compression_categories import (  # noqa: E402
    SourceIndex, leading_command, load_graph, written_declaration_groups,
)

DEFINITION_COMMANDS = frozenset({"def", "abbrev"})
_HEAD_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:@\[[^\]]*\][ \t]*)*"
    r"(?:(?:private|protected|noncomputable|unsafe|partial|local|scoped)[ \t]+)*"
    r"(?P<keyword>def|abbrev)[ \t]+(?P<name>[^\s({\[:]+)"
)
# A tactic block is a proof, not an expression to inline.
_TACTIC_BODY_RE = re.compile(r"^\s*by\b")


def explicit_binders(text: str) -> list[str] | str:
    """Names bound by ``(...)`` groups, in order; a reason string on refusal."""
    names: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char.isspace():
            index += 1
            continue
        if char in "{[⦃":
            # Implicit, instance and strict-implicit binders take no argument at
            # a call site, so they are skipped rather than turned into holes.
            closing = {"{": "}", "[": "]", "⦃": "⦄"}[char]
            depth = 0
            while index < len(text):
                if text[index] == char:
                    depth += 1
                elif text[index] == closing:
                    depth -= 1
                    if depth == 0:
                        index += 1
                        break
                index += 1
            continue
        if char == "(":
            end = pattern_pass.atom_end(text, index)
            if end < 0:
                return "unbalanced binder"
            inner = text[index + 1:end - 1]
            colon = remove_lean_comments(inner, mask_strings=True).find(":")
            declared = inner[:colon] if colon >= 0 else inner
            for name in declared.split():
                if re.fullmatch(r"[A-Za-z_À-￿][\w'À-￿]*", name):
                    names.append(name)
            index = end
            continue
        return "unsupported binder syntax"
    return names


def parse_definition(source: str) -> tuple[list[tuple[str, str, str]], str, str] | str:
    """Return (pattern items, body, name) or a reason the definition is skipped."""
    code = remove_lean_comments(source)
    masked = remove_lean_comments(source, mask_strings=True)
    head = _HEAD_RE.search(masked)
    if head is None:
        return "not a definition command"
    name = head.group("name")
    assign = pattern_pass.find_literal(masked, ":=", head.end())
    if assign < 0:
        return "no definition body"
    signature, body = code[head.end():assign], code[assign + 2:].strip()
    signature_masked = masked[head.end():assign]
    if "|" in signature_masked or re.search(r"(?:^|\n)\s*\|", masked[assign:]):
        return "equation-style definition"
    if re.search(r"(?:^|\s)where(?:\s|$)", masked[assign:]):
        return "where block"
    if _TACTIC_BODY_RE.match(body):
        return "tactic body"
    if not body:
        return "empty body"
    # Stop at the type ascription: everything after the top-level ``:`` is the
    # result type, not a binder.
    colon = pattern_pass.find_literal(signature_masked, ":", 0)
    if colon >= 0:
        signature = signature[:colon]
    binders = explicit_binders(signature)
    if isinstance(binders, str):
        return binders
    if pattern_pass.find_literal(remove_lean_comments(body, mask_strings=True), name, 0) >= 0:
        return "recursive definition"
    items: list[tuple[str, str, str]] = [("lit", name, "")]
    items.extend(("param", binder, "term") for binder in binders)
    return items, body, name


def analyse_repository(model: str, repo: str, repo_dir: Path,
                       record: Mapping[str, Any]) -> dict[str, Any] | None:
    need = [repo_dir / "before.json", repo_dir / "after.json",
            repo_dir / "after" / "original-sources.tar.gz"]
    if not all(path.is_file() for path in need):
        return None
    before, after = load_graph(need[0]), load_graph(need[1])
    sources = SourceIndex.from_archive(need[2])
    baseline_archive = repo_dir / "before" / "original-sources.tar.gz"
    baseline = (SourceIndex.from_archive(baseline_archive)
                if baseline_archive.is_file() else None)
    scope = record.get("metric_scope", {})
    if baseline is not None:
        baseline.restrict(
            exclude_files=scope.get("excluded_files", []),
            exclude_dirs=scope.get("exclude_dirs", []),
            include_prefix=scope.get("include_prefix", ""))
    sources.restrict(exclude_files=scope.get("excluded_files", []),
                     exclude_dirs=scope.get("exclude_dirs", []),
                     include_prefix=scope.get("include_prefix", ""))
    measured = sources.total_tokens()
    recorded = int(record.get("text_accounting", {}).get("after_total_tokens", -1))

    added = {name for name in set(after.nodes) - set(before.nodes)
             if str(after.nodes[name].get("kind")) == "def"}
    rules: list[pattern_pass.Rule] = []
    graph_names: dict[str, set[str]] = {}
    skipped: list[dict[str, Any]] = []
    deletions: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for representative, _members in written_declaration_groups(after, added):
        node = after.nodes[representative]
        source = sources.slice(node) or ""
        keyword = leading_command(source)
        if keyword not in DEFINITION_COMMANDS or not source:
            continue
        tokens = count_lean_tokens_in_source(source)
        parsed = parse_definition(source)
        if isinstance(parsed, str):
            skipped.append({"keyword": str(keyword), "reason": parsed, "tokens": tokens})
            continue
        items, body, name = parsed
        if baseline is not None:
            hits = pattern_pass.baseline_occurrences(baseline, name)
            if hits:
                # A definition named `f` is indistinguishable by spelling from
                # every bound variable called `f`; the baseline proves the clash.
                skipped.append({"keyword": str(keyword),
                                "reason": f"name used in baseline ({hits} occurrences)",
                                "tokens": tokens})
                continue
        path = sources.path_of_module(str(node.get("module", "")))
        graph_names.setdefault(name, set()).add(representative)
        rules.append(pattern_pass.Rule(
            repository=repo, model=model, keyword=str(keyword), path=path,
            start_line=int(node["start_line"]), end_line=int(node["end_line"]),
            items=tuple(items), template=body, file_scoped=False, source=source,
            tokens=tokens,
        ))
        if path:
            deletions[path].append((int(node["start_line"]), int(node["end_line"])))

    # Which declarations actually reference each new definition.  Restricting
    # rewriting to those is what stops a definition named ``f`` from matching
    # every bound variable spelled ``f`` in the repository.
    dependents: dict[str, set[str]] = collections.defaultdict(set)
    for source_name, target_name in after.edges:
        for short, full_names in graph_names.items():
            if target_name in full_names:
                dependents[short].add(source_name)
    allowed_by_file: dict[str, dict[str, list[tuple[int, int]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list))
    for short, names in dependents.items():
        for dependent in names:
            node = after.nodes.get(dependent)
            if node is None:
                continue
            dependent_path = sources.path_of_module(str(node.get("module", "")))
            if dependent_path:
                allowed_by_file[dependent_path][short].append(
                    (int(node["start_line"]), int(node["end_line"])))

    files = dict(sources._files)  # noqa: SLF001 -- the scored file set, already restricted
    for path, ranges in deletions.items():
        if path in files:
            files[path] = pattern_pass.delete_ranges(files[path], ranges)
    passes = 0
    for _ in range(4):
        rewrites = 0
        for path, body in list(files.items()):
            if not rules:
                break
            ordered = sorted(rules, key=lambda r: -len(r.lead))
            updated, count = pattern_pass.expand_once(
                body, ordered, allowed_by_file.get(path, {}))
            if count:
                files[path] = updated
                rewrites += count
        passes += 1
        if not rewrites:
            break
    expanded = SourceIndex(files).total_tokens()
    return {
        "model": model, "repository": repo,
        "measured_after_tokens": measured, "recorded_after_tokens": recorded,
        "harness_ok": measured == recorded,
        "expanded_tokens": expanded,
        "definition_leg_tokens": expanded - measured,
        "definitions": len(rules), "definition_tokens": sum(r.tokens for r in rules),
        "dependent_declarations": sum(len(v) for v in dependents.values()),
        "sites": sum(r.sites for r in rules), "passes": passes,
        "skipped": skipped,
        "detail": [{"name": r.lead, "parameters": list(r.parameters), "sites": r.sites,
                    "tokens": r.tokens}
                   for r in sorted(rules, key=lambda x: -x.sites)[:25]],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--category-dir", type=Path,
                        default=Path("output/analysis/compression_categories/runs"))
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for run in options.run:
        run_path = Path(run)
        model = next((m for m in ("opus", "sol", "luna", "gemini")
                      if f"_{m}_" in run_path.name), None)
        category_path = options.category_dir / f"{run_path.name}.json"
        if model is None or not category_path.is_file():
            continue
        payload = json.loads(category_path.read_text(encoding="utf-8"))
        records = {str(r["repository"]): r for r in payload.get("repositories", [])}
        for repo_dir in sorted((run_path / "repositories").glob("*")):
            record = records.get(repo_dir.name)
            if record is None:
                continue
            try:
                row = analyse_repository(model, repo_dir.name, repo_dir, record)
            except Exception as exc:  # noqa: BLE001
                print(f"error {model}/{repo_dir.name}: {exc}", file=sys.stderr)
                continue
            if row is not None:
                rows.append(row)

    totals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for row in rows:
        counter = totals[row["model"]]
        counter["repositories"] += 1
        counter["definitions"] += row["definitions"]
        counter["sites"] += row["sites"]
        counter["definition_tokens"] += row["definition_tokens"]
        counter["skipped"] += len(row["skipped"])
        if row["harness_ok"]:
            counter["harness_ok"] += 1
            counter["definition_leg_tokens"] += row["definition_leg_tokens"]
        for item in row["skipped"]:
            reasons[row["model"]][item["reason"]] += 1
    payload = {"totals": {m: dict(c) for m, c in totals.items()},
               "skip_reasons": {m: dict(c) for m, c in reasons.items()},
               "repositories": rows}
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"totals": payload["totals"], "skip_reasons": payload["skip_reasons"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
