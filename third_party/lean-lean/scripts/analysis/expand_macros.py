#!/usr/bin/env python3
"""Rewrite a compressed repository as if its new notation and macros never existed.

The compression score counts tokens, so a model that defines ``notation "Af" =>
Amat f`` and then writes ``Af`` three hundred times has compressed the repository
without changing a single proof.  Today that lands in ``proof_rewriting_tokens``,
because a notation leaves no constant behind and therefore no dependency edge --
the graph cannot see it, so the shortened call sites look like rewriting.

This reverses the spelling and prices the difference.  Every syntax-level command
the model introduced is parsed into a rewrite rule, every call site is expanded
back to what it stands for, and the commands themselves are deleted.  The token
difference between that tree and the compressed tree is what the syntax layer
bought, net of what defining it cost.

There is no build here by design, so nothing checks that the expanded tree still
elaborates.  Two things stand in for that check.  First, the unmodified tree is
measured through the benchmark's own metric and compared against the score the
run recorded: if those disagree, the harness is wrong and the leg is not
reported.  Second, every command that cannot be expanded faithfully is counted
and reported by reason rather than silently priced at zero -- an ``elab`` has no
syntactic expansion at all, and pretending otherwise is how the earlier regex
estimate came to claim more savings than the repository had.

Expansions are parenthesised, which costs two tokens a site that the original
spelling would not have paid.  The leg is therefore a slight under-estimate of
the saving, which is the direction an attribution should err in.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.metrics.tokens import (  # noqa: E402
    count_lean_tokens_in_source, remove_lean_comments,
)
from scripts.analysis.quantify_compression_categories import (  # noqa: E402
    SourceIndex, command_class, leading_command, load_graph, written_declaration_groups,
)

# Commands that can carry a syntactic rewrite rule.  ``syntax`` only declares a
# parser -- the rule that goes with it arrives in a separate ``macro_rules`` --
# and ``declare_syntax_cat`` declares a category, so neither expands on its own.
RULE_COMMANDS = frozenset({
    "notation", "notation3", "macro", "macro_rules", "infix", "infixl", "infixr",
    "prefix", "postfix",
})

_KEYWORD_RE = re.compile(
    r"(?:^|\n)[ \t]*(?:(?:local|scoped|private|protected)[ \t]+)*"
    r"(?P<keyword>notation3|notation|macro_rules|macro|infixl|infixr|infix|prefix|postfix"
    r"|syntax|elab|declare_syntax_cat)"
    r"(?P<precedence>:[^\s\"]+)?(?=[\s\"]|$)"
)
# A right-hand side that computes its expansion in ``MacroM`` has no syntactic
# form to substitute.  ``fun``, ``if`` and ``let`` are *not* markers of that: in
# a quotation they are ordinary syntax being quoted, and in a notation template
# they are ordinary terms.  A ``do`` block or a monadic bind is the real signal.
_PROGRAMMATIC_RE = re.compile(r"^\s*do\b|←|<-")
# A splice runs code to build part of the expansion, and antiquotation
# repetition matches a variable number of arguments.  Neither has a fixed form.
_SPLICE_RE = re.compile(r"\$\(|\$\[")
# Repetition and optional items match a variable number of arguments, so a single
# substitution cannot stand in for them.
_VARIADIC_RE = re.compile(r"[*+?]|,\s*[*+]")
_OPENERS = {"(": ")", "[": "]", "{": "}", "⟨": "⟩", "⟦": "⟧", "⟪": "⟫", "«": "»"}
_CLOSERS = {close: open_ for open_, close in _OPENERS.items()}
_IDENT_EXTRA = set("._'!?")


def matching_view(source: str) -> tuple[str, list[bool]]:
    """Comment-free text plus a per-offset "inside a string literal" mask.

    Matching needs string literals present -- a call site can take one as an
    argument, and masking it away truncates the site -- but must never match
    *inside* one.  The mask is the difference between masking comments alone and
    masking comments and strings, so it reuses the scoring module's own scanner
    rather than re-implementing Lean's lexical rules.
    """
    without_comments = remove_lean_comments(source)
    without_strings = remove_lean_comments(source, mask_strings=True)
    return without_comments, [a != b for a, b in zip(without_comments, without_strings)]


def is_ident_char(char: str) -> bool:
    """Identifier characters exactly as the scoring lexer defines them."""
    return char.isalnum() or char in _IDENT_EXTRA


def baseline_occurrences(baseline: "SourceIndex", atom: str) -> int:
    """How often an atom already appeared before the model introduced it.

    An atom absent from the baseline can only mean the new declaration, so
    matching it by text is safe.  One that was already there meant something
    else first -- a bound variable, an existing name -- and text cannot tell the
    two apart.  Rules on such an atom are reported rather than expanded, which
    undercounts their saving; that is the direction to err in.
    """
    total = 0
    for path in baseline.paths():
        masked, in_string = matching_view(baseline._files[path])  # noqa: SLF001
        index = 0
        while True:
            index = find_literal(masked, atom, index, in_string=in_string)
            if index < 0:
                break
            total += 1
            index += len(atom)
    return total


@dataclass
class Rule:
    """One expandable syntax command, as a pattern and a replacement."""

    repository: str
    model: str
    keyword: str
    path: str | None
    start_line: int
    end_line: int
    items: tuple[tuple[str, str, str], ...]  # (kind, value, category)
    template: str
    file_scoped: bool
    source: str
    tokens: int
    sites: int = 0

    @property
    def lead(self) -> str:
        for kind, value, _category in self.items:
            if kind == "lit" and value.strip():
                return value.strip()
        return ""

    @property
    def parameters(self) -> tuple[str, ...]:
        return tuple(value for kind, value, _ in self.items if kind == "param")


@dataclass
class Rejected:
    repository: str
    model: str
    keyword: str
    reason: str
    tokens: int
    source: str


@dataclass
class RepositoryResult:
    repository: str
    model: str
    recorded_after_tokens: int
    measured_after_tokens: int
    expanded_tokens: int
    definition_tokens: int
    rules: list[Rule] = field(default_factory=list)
    rejected: list[Rejected] = field(default_factory=list)
    passes: int = 0

    @property
    def harness_ok(self) -> bool:
        return self.recorded_after_tokens == self.measured_after_tokens

    @property
    def syntax_leg(self) -> int:
        """Tokens the syntax layer saved, net of what the commands cost to write."""
        return self.expanded_tokens - self.measured_after_tokens


# --------------------------------------------------------------------------- #
# Parsing a command into a rule
# --------------------------------------------------------------------------- #


# ``n:max`` pins a parameter's precedence, not its category; ``a:term:max``
# gives both.  Only the category constrains what a capture may be.
_PRECEDENCES = frozenset({"max", "arg", "lead", "min", "min1"})
# Category names that stand alone in a pattern, naming no parameter.
_BARE_CATEGORIES = frozenset({
    "num", "str", "ident", "term", "tactic", "command", "char", "name", "level",
    "numLit", "strLit", "charLit", "tacticSeq", "doElem",
})


def split_items(pattern: str) -> list[tuple[str, str, str]] | str:
    """Split a syntax pattern into literals and named parameters.

    Each item is ``(kind, value, category)``; literals carry an empty category.
    Returns a reason string when the pattern contains something a single
    substitution cannot reproduce.
    """
    items: list[tuple[str, str, str]] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char.isspace():
            index += 1
            continue
        if char == '"':
            end = pattern.find('"', index + 1)
            if end < 0:
                return "unterminated literal"
            items.append(("lit", pattern[index + 1:end], ""))
            index = end + 1
            continue
        if char in "()":
            # A grouped or optional item; no faithful single substitution.
            return "grouped pattern item"
        if char in "*+?":
            return "variadic pattern item"
        match = re.match(r"[A-Za-z_][\w']*", pattern[index:])
        if not match:
            index += 1
            continue
        name = match.group(0)
        index += len(name)
        category = ""
        # ``a:term:max`` -- consume the category and precedence annotations, and
        # keep the first that is a category rather than a precedence.
        while index < len(pattern) and pattern[index] == ":":
            index += 1
            annotation = re.match(r"[\w'.]+", pattern[index:])
            if not annotation:
                break
            text = annotation.group(0)
            index += len(text)
            if not category and text not in _PRECEDENCES and not text.isdigit():
                category = text
        # ``syntax "-" num : tableInt`` names no parameter: ``num`` *is* the
        # category.  Treated as an untyped parameter it would accept anything,
        # so a one-character literal like "-" matches every minus in the file.
        if not category and name in _BARE_CATEGORIES:
            category = name
            name = f"_{len(items)}"
        items.append(("param", name, category))
    return items


def capture_fits(category: str, text: str) -> bool:
    """Whether a captured argument could have been parsed in that category."""
    text = text.strip()
    if not text:
        return False
    if category in ("num", "numLit"):
        return bool(re.fullmatch(r"\d+", text))
    if category in ("str", "strLit"):
        return text.startswith('"') and text.endswith('"') and len(text) >= 2
    if category in ("ident", "declId"):
        return bool(re.fullmatch(r"[A-Za-z_\u00c0-\uffff][\w'.!?\u00c0-\uffff]*", text))
    if category in ("char", "charLit"):
        return text.startswith("'")
    return True


def strip_quotation(rhs: str) -> tuple[str, str | None]:
    """Unwrap `` `(tactic| ...) `` to its body; return (body, reason-if-unusable)."""
    text = rhs.strip()
    if not text.startswith("`("):
        return text, None
    depth = 0
    for position, char in enumerate(text[1:], start=1):
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth -= 1
            if depth == 0:
                body = text[2:position]
                trailing = text[position + 1:].strip()
                if trailing:
                    return text, "quotation followed by more syntax"
                body = re.sub(r"^\s*(?:tactic|term|command|doElem|tacticSeq)\|", "", body)
                return body.strip(), None
    return text, "unbalanced quotation"


def parse_command(source: str, keyword: str) -> tuple[list[tuple[str, str]], str] | str:
    """Return (pattern items, template) or a reason the command cannot expand."""
    code = remove_lean_comments(source)
    masked = remove_lean_comments(source, mask_strings=True)
    match = _KEYWORD_RE.search(masked)
    if match is None:
        return "command keyword not found"
    cursor = match.end()

    if keyword == "macro_rules":
        # ``macro_rules | `(pat) => `(rhs)``; only a single alternative can be
        # expanded, because several alternatives are a case split on the input.
        alternatives = [a for a in re.split(r"\n\s*\|", masked[cursor:]) if a.strip()]
        if len(alternatives) != 1:
            return "multiple macro_rules alternatives"
        bar = masked.find("|", cursor)
        if bar < 0:
            return "macro_rules without an alternative"
        cursor = bar + 1

    arrow = masked.find("=>", cursor)
    if arrow < 0:
        return "no expansion arrow"
    lhs, rhs = code[cursor:arrow], code[arrow + 2:]
    lhs_masked = masked[cursor:arrow]

    if keyword == "macro":
        # ``macro <pattern> : <category> =>`` -- drop the category annotation,
        # which is the last top-level colon before the arrow.
        colon = lhs_masked.rfind(":")
        if colon >= 0 and re.fullmatch(r"\s*[\w'.]+\s*", lhs_masked[colon + 1:]):
            lhs = lhs[:colon]
            lhs_masked = lhs_masked[:colon]
    if keyword == "macro_rules":
        pattern_body, reason = strip_quotation(lhs)
        if reason:
            return reason
        lhs = pattern_body

    template, reason = strip_quotation(rhs)
    if reason:
        return reason
    # ``←`` means a monadic bind only outside a quotation.  Inside one it is
    # ordinary Lean syntax being quoted -- ``rw [← foo]`` is a reverse rewrite --
    # so testing the raw right-hand side would reject perfectly expandable tactic
    # macros.  A quotation is therefore exempt, and only a computed expansion
    # (a ``do`` block, or a bind in a bare term) counts as programmatic.
    if not rhs.strip().startswith("`("):
        if _PROGRAMMATIC_RE.search(remove_lean_comments(rhs, mask_strings=True)):
            return "programmatic expansion"
    elif re.match(r"^\s*do\b", remove_lean_comments(rhs, mask_strings=True).strip()):
        return "programmatic expansion"
    if _SPLICE_RE.search(template):
        return "spliced expansion"
    if not template.strip():
        return "empty expansion"

    items = split_items(lhs)
    if isinstance(items, str):
        return items
    if not any(kind == "lit" and value.strip() for kind, value, _ in items):
        return "no literal to match on"
    return items, template


# --------------------------------------------------------------------------- #
# Matching call sites
# --------------------------------------------------------------------------- #


def literal_at(code: str, literal: str, position: int) -> bool:
    if not code.startswith(literal, position):
        return False
    if is_ident_char(literal[0]):
        if position > 0 and is_ident_char(code[position - 1]):
            return False
    end = position + len(literal)
    if is_ident_char(literal[-1]):
        if end < len(code) and is_ident_char(code[end]):
            return False
    return True


def find_literal(code: str, literal: str, start: int, stop: int | None = None,
                 in_string: Sequence[bool] | None = None) -> int:
    """First occurrence of ``literal`` at bracket depth zero, or -1."""
    limit = len(code) if stop is None else stop
    depth = 0
    index = start
    while index < limit:
        char = code[index]
        if in_string is not None and in_string[index]:
            index += 1
            continue
        if depth == 0 and literal_at(code, literal, index):
            return index
        if char in _OPENERS:
            depth += 1
        elif char in _CLOSERS:
            depth = max(0, depth - 1)
        index += 1
    return -1


def atom_end(code: str, start: int, in_string: Sequence[bool] | None = None) -> int:
    """End of one syntactic argument starting at ``start``."""
    index = start
    while index < len(code) and code[index].isspace():
        index += 1
    if index >= len(code):
        return -1
    char = code[index]
    if char == '"':
        # A string literal is one argument, however much it contains.
        index += 1
        while index < len(code):
            if code[index] == "\\":
                index += 2
                continue
            if code[index] == '"':
                return index + 1
            index += 1
        return -1
    if char in _OPENERS:
        depth = 0
        while index < len(code):
            if code[index] in _OPENERS:
                depth += 1
            elif code[index] in _CLOSERS:
                depth -= 1
                if depth == 0:
                    return index + 1
            index += 1
        return -1
    if is_ident_char(char):
        while index < len(code) and is_ident_char(code[index]):
            index += 1
        return index
    return index + 1


def block_end(code: str, start: int) -> int:
    """End of an indented block beginning at ``start``.

    A ``tacticSeq`` argument is the rest of the line plus every following line
    indented further than the line the call site began on.  Taking a single
    token instead truncates the call site and it no longer parses.
    """
    line_start = code.rfind("\n", 0, start) + 1
    indent = len(code[line_start:]) - len(code[line_start:].lstrip())
    cursor = code.find("\n", start)
    if cursor < 0:
        return len(code)
    while True:
        following = code.find("\n", cursor + 1)
        line = code[cursor + 1:following if following >= 0 else len(code)]
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            return cursor
        if following < 0:
            return len(code)
        cursor = following


def match_rule(code: str, position: int, items: Sequence[tuple[str, str, str]],
               in_string: Sequence[bool] | None = None) -> tuple[int, dict[str, tuple[int, int]]] | None:
    """Match a pattern at ``position``; captures are (start, end) offsets.

    Offsets rather than text, because matching runs over the comment- and
    string-masked copy while the replacement has to be built from the real
    source -- and the two are the same length, so offsets carry across.
    """
    captures: dict[str, tuple[int, int]] = {}
    cursor = position
    for index, (kind, value, category) in enumerate(items):
        if kind == "lit":
            literal = value.strip()
            if not literal:
                continue
            while cursor < len(code) and code[cursor].isspace():
                cursor += 1
            if not literal_at(code, literal, cursor):
                return None
            cursor += len(literal)
            continue
        # A parameter runs to the next literal, or is a single argument when the
        # next item is another parameter or the pattern ends.
        following = next(
            (v.strip() for k, v, _ in list(items)[index + 1:] if k == "lit" and v.strip()), None
        )
        while cursor < len(code) and code[cursor].isspace():
            cursor += 1
        if following is not None:
            end = find_literal(code, following, cursor, in_string=in_string)
            if end < 0:
                return None
            if end <= cursor:
                return None
            start, stop = cursor, end
            cursor = end
        elif category in ("tacticSeq", "doElem"):
            end = block_end(code, cursor)
            if end <= cursor:
                return None
            start, stop = cursor, end
            cursor = end
            captures[value] = (start, stop)
            continue
        else:
            end = atom_end(code, cursor, in_string)
            if end < 0:
                return None
            start, stop = cursor, end
            cursor = end
        while start < stop and code[start].isspace():
            start += 1
        while stop > start and code[stop - 1].isspace():
            stop -= 1
        if start >= stop:
            return None
        # The capture has to be something that category could have parsed.
        if not capture_fits(category, code[start:stop]):
            return None
        captures[value] = (start, stop)
    return cursor, captures


def needs_parentheses(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.fullmatch(r"[\w'.]+", stripped):
        return False
    # A literal is already atomic.  Wrapping one is not merely wasteful -- two
    # tokens a site -- it can be outright invalid, as in a `notation` command
    # whose pattern needs a bare string.
    if len(stripped) >= 2 and stripped[0] == '"' and stripped[-1] == '"':
        return False
    if stripped[0] == "'" and stripped[-1] == "'":
        return False
    if re.fullmatch(r"-?\d+(?:\.\d+)?", stripped):
        return False
    if stripped[0] in _OPENERS and atom_end(stripped, 0) == len(stripped):
        return False
    return True


def substitute(template: str, captures: Mapping[str, str]) -> str:
    """Put the captured arguments into the template.

    Notation templates name parameters directly; macro quotations name them with
    an antiquotation and may pin the category, as ``$h:ident``.
    """
    result = template
    for name, value in sorted(captures.items(), key=lambda item: -len(item[0])):
        argument = f"({value})" if needs_parentheses(value) else value
        result = re.sub(
            r"\$" + re.escape(name) + r"(?::[\w'.]+)?(?![\w'])", lambda _m: argument, result
        )
        result = re.sub(
            r"(?<![\w'.$])" + re.escape(name) + r"(?![\w'])", lambda _m: argument, result
        )
    return result


def effective_rule(rules: Sequence[Rule], line: int) -> Rule:
    """The declaration in force at ``line`` when an atom is declared more than once.

    A repository may redeclare the same notation -- two ``local notation "⚷"``
    in one file, meaning different things in different sections.  Lean uses the
    most recent preceding declaration, so applying whichever one happened to be
    parsed first would expand half the call sites wrongly.
    """
    preceding = [rule for rule in rules if rule.start_line <= line]
    return (preceding or list(rules))[-1]


def expand_once(code: str, rules: Sequence[Rule],
                allowed: Mapping[str, Sequence[tuple[int, int]]] | None = None
                ) -> tuple[int, int] | tuple[str, int]:
    """One left-to-right pass, rewriting every call site that matches.

    ``allowed`` optionally restricts each rule to line ranges in this file, for
    when text alone cannot tell a use of a declaration from a local variable of
    the same name.  A definition called ``f`` is indistinguishable by spelling
    from the ``f`` bound in every second proof; the dependency graph is what
    knows the difference.
    """
    masked, in_string = matching_view(code)
    by_lead: dict[str, list[Rule]] = collections.defaultdict(list)
    for rule in rules:
        by_lead[rule.lead].append(rule)
    for group in by_lead.values():
        group.sort(key=lambda r: r.start_line)
    pieces: list[str] = []
    cursor = 0
    rewrites = 0
    index = 0
    while index < len(code):
        # Never start a match inside a comment or a string literal.
        if in_string[index] or (masked[index].isspace() and not code[index].isspace()):
            index += 1
            continue
        for rule in rules:
            lead = rule.lead
            if not lead or not literal_at(masked, lead, index):
                continue
            line = masked.count("\n", 0, index) + 1
            if allowed is not None:
                ranges = allowed.get(rule.lead)
                if ranges is None or not any(start <= line <= end for start, end in ranges):
                    continue
            matched = match_rule(masked, index, rule.items, in_string)
            if matched is None:
                continue
            end, spans = matched
            captures = {name: code[start:stop] for name, (start, stop) in spans.items()}
            chosen = effective_rule(by_lead[lead], line)
            replacement = substitute(chosen.template, captures)
            pieces.append(code[cursor:index])
            # Parenthesise only when the expansion needs it.  Wrapping every
            # site costs two scored tokens a site that a human writing the
            # expansion out by hand would never have paid, and that inflates the
            # measured saving rather than shrinking it.
            pieces.append(f"({replacement})" if needs_parentheses(replacement)
                          else replacement)
            cursor = end
            index = end
            rewrites += 1
            rule.sites += 1
            break
        else:
            index += 1
    pieces.append(code[cursor:])
    return "".join(pieces), rewrites


def delete_ranges(body: str, ranges: Iterable[tuple[int, int]]) -> str:
    lines = body.splitlines(keepends=True)
    drop: set[int] = set()
    for start, end in ranges:
        drop.update(range(max(1, start), min(len(lines), end) + 1))
    return "".join(line for number, line in enumerate(lines, start=1) if number not in drop)


# --------------------------------------------------------------------------- #
# Per-repository analysis
# --------------------------------------------------------------------------- #


def analyse_repository(model: str, repo: str, repo_dir: Path,
                       record: Mapping[str, Any], max_passes: int) -> RepositoryResult | None:
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
    sources.restrict(
        exclude_files=scope.get("excluded_files", []),
        exclude_dirs=scope.get("exclude_dirs", []),
        include_prefix=scope.get("include_prefix", ""),
    )
    recorded = int(record.get("text_accounting", {}).get("after_total_tokens", -1))
    measured = sources.total_tokens()

    rules: list[Rule] = []
    rejected: list[Rejected] = []
    deletions: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    added = set(after.nodes) - set(before.nodes)
    for representative, _members in written_declaration_groups(after, added):
        node = after.nodes[representative]
        source = sources.slice(node) or ""
        keyword = leading_command(source)
        if command_class(keyword) != "syntax_level" or not source:
            continue
        tokens = count_lean_tokens_in_source(source)
        path = sources.path_of_module(str(node.get("module", "")))
        if keyword not in RULE_COMMANDS:
            rejected.append(Rejected(repo, model, str(keyword), "not a rewrite rule", tokens, source))
            continue
        parsed = parse_command(source, str(keyword))
        if isinstance(parsed, str):
            rejected.append(Rejected(repo, model, str(keyword), parsed, tokens, source))
            continue
        items, template = parsed
        rule = Rule(
            repository=repo, model=model, keyword=str(keyword), path=path,
            start_line=int(node["start_line"]), end_line=int(node["end_line"]),
            items=tuple(items), template=template,
            file_scoped=bool(re.search(r"\b(?:local|scoped)\b", source.split("=>")[0])),
            source=source, tokens=tokens,
        )
        # A rule whose expansion invokes itself has no finite expansion.
        if rule.lead and find_literal(
            remove_lean_comments(template, mask_strings=True), rule.lead, 0
        ) >= 0:
            rejected.append(
                Rejected(repo, model, str(keyword), "self-recursive expansion", tokens, source))
            continue
        if baseline is not None and rule.lead:
            hits = baseline_occurrences(baseline, rule.lead)
            if hits:
                rejected.append(Rejected(
                    repo, model, str(keyword),
                    f"atom used in baseline ({hits} occurrences)", tokens, source))
                continue
        rules.append(rule)
        if path:
            deletions[path].append((rule.start_line, rule.end_line))

    files = dict(sources._files)  # noqa: SLF001 -- the scored file set, already restricted
    definition_tokens = sum(rule.tokens for rule in rules)
    for path, ranges in deletions.items():
        if path in files:
            files[path] = delete_ranges(files[path], ranges)

    passes = 0
    for _ in range(max_passes):
        rewrites = 0
        for path, body in list(files.items()):
            applicable = [r for r in rules if not r.file_scoped or r.path == path]
            if not applicable:
                continue
            # Longest literal first, so a short name cannot match inside a longer one.
            applicable.sort(key=lambda r: -len(r.lead))
            updated, count = expand_once(body, applicable)
            if count:
                files[path] = updated
                rewrites += count
        passes += 1
        if not rewrites:
            break

    expanded = SourceIndex(files)
    return RepositoryResult(
        repository=repo, model=model, recorded_after_tokens=recorded,
        measured_after_tokens=measured, expanded_tokens=expanded.total_tokens(),
        definition_tokens=definition_tokens, rules=rules, rejected=rejected, passes=passes,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True,
                        help="graph run directory, repeatable")
    parser.add_argument("--category-dir", type=Path,
                        default=Path("output/analysis/compression_categories/runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-passes", type=int, default=6)
    parser.add_argument("--limit", type=int)
    options = parser.parse_args()

    results: list[RepositoryResult] = []
    for run in options.run:
        run_path = Path(run)
        category_path = options.category_dir / f"{run_path.name}.json"
        records: dict[str, Any] = {}
        if category_path.is_file():
            payload = json.loads(category_path.read_text(encoding="utf-8"))
            records = {str(row["repository"]): row for row in payload.get("repositories", [])}
        model = next((m for m in ("opus", "sol", "luna", "gemini") if f"_{m}_" in run_path.name), None)
        if model is None:
            continue
        for repo_dir in sorted((run_path / "repositories").glob("*")):
            record = records.get(repo_dir.name)
            if record is None:
                continue
            try:
                result = analyse_repository(model, repo_dir.name, repo_dir, record,
                                            options.max_passes)
            except Exception as exc:  # noqa: BLE001 -- one bad repository must not stop the sweep
                print(f"error {model}/{repo_dir.name}: {exc}", file=sys.stderr)
                continue
            if result is None:
                continue
            results.append(result)
            if options.limit and len(results) >= options.limit:
                break

    payload = {
        "repositories": [
            {
                "model": r.model, "repository": r.repository,
                "recorded_after_tokens": r.recorded_after_tokens,
                "measured_after_tokens": r.measured_after_tokens,
                "harness_ok": r.harness_ok,
                "expanded_tokens": r.expanded_tokens,
                "syntax_leg_tokens": r.syntax_leg,
                "definition_tokens": r.definition_tokens,
                "rules": len(r.rules), "sites": sum(rule.sites for rule in r.rules),
                "rejected": len(r.rejected), "passes": r.passes,
                "rule_detail": [
                    {"keyword": rule.keyword, "lead": rule.lead,
                     "parameters": list(rule.parameters), "sites": rule.sites,
                     "tokens": rule.tokens}
                    for rule in sorted(r.rules, key=lambda x: -x.sites)
                ],
                "rejected_detail": [
                    {"keyword": item.keyword, "reason": item.reason, "tokens": item.tokens}
                    for item in r.rejected
                ],
            }
            for r in results if r.rules or r.rejected
        ],
    }
    totals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    reasons: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in results:
        counter = totals[r.model]
        counter["repositories"] += 1
        if r.rules or r.rejected:
            counter["repositories_with_syntax"] += 1
        counter["rules"] += len(r.rules)
        counter["sites"] += sum(rule.sites for rule in r.rules)
        counter["rejected"] += len(r.rejected)
        counter["definition_tokens"] += r.definition_tokens
        if r.harness_ok:
            counter["harness_ok"] += 1
            counter["syntax_leg_tokens"] += r.syntax_leg
            counter["after_tokens"] += r.measured_after_tokens
        for item in r.rejected:
            reasons[r.model][item.reason] += 1
    payload["totals"] = {m: dict(c) for m, c in totals.items()}
    payload["rejection_reasons"] = {m: dict(c) for m, c in reasons.items()}

    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"totals": payload["totals"],
                      "rejection_reasons": payload["rejection_reasons"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
