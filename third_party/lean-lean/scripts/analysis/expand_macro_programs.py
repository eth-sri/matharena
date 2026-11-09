#!/usr/bin/env python3
"""Measure programmatic macro savings using one native Lean expansion step.

A standalone file registers the repository's syntax commands and expands each
call site without a repository build. Expansion stops after one step so Lean's
own list and numeral syntax remain intact. Hygiene marks are removed before
source-token scoring. Elaborators with environment effects are reported as
uncovered. See docs/compression_categories.md.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.metrics.tokens import (  # noqa: E402
    count_lean_tokens_in_source, remove_lean_comments,
)
import scripts.analysis.expand_macros as pattern_pass  # noqa: E402
from scripts.analysis.quantify_compression_categories import (  # noqa: E402
    SourceIndex, command_class, leading_command, load_graph, written_declaration_groups,
)

# Hygiene marks an identifier a macro introduced.  It is an artefact of
# expansion, never something a human wrote, and the lexer would score each one.
_HYGIENE_RE = re.compile(r"✝[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]*")
# A repository may declare its own category, so anything identifier-shaped is
# taken as written.  Defaulting a custom category to ``term`` makes call sites
# parse in the wrong category, where Lean's own macros -- not the model's --
# are what expand.
_CATEGORY_RE = re.compile(r":\s*([A-Za-z_][\w'.]*)\s*$")
_SITE_RE = re.compile(r"@@SITE (\S+) (OK|NO_STEP|PARSE_ERROR)(.*)")
_OK_TAIL_RE = re.compile(r"^\s*(\d+)\s+(\S*)")
# A quotation that mentions a Mathlib tactic cannot be parsed by core Lean, and
# the whole ``macro_rules`` then fails to register, so every call site of it
# reports NO_STEP.  Lean points at the offending token, which is enough to
# declare a stub for it.
_UNKNOWN_TACTIC_RE = re.compile(r":(\d+):(\d+): error: unknown tactic")
_TOKEN_RE = re.compile(r"[A-Za-z_][\w'!?]*")
# Term notation from Mathlib that a quotation may mention.  The diagnostic names
# the token but not the shape, so these are keyed by token and added only when
# that token actually fails -- declaring them unconditionally is harmful: a
# term-level "|" is ambiguous with the "|" that opens a ``macro_rules``
# alternative, and silently stops the rule parsing at all.
# A syntax declared in a category the repository itself declared is a component
# of a larger construct, never something standing alone in the source.  Its
# saving is measured through the parent that embeds it: `syntax "-" num :
# tableInt` describes one entry of a table, so every negative literal anywhere
# matches it, while only `ints[...]` is really a call site.
TOP_LEVEL_CATEGORIES = frozenset({
    "term", "tactic", "command", "doElem", "tacticSeq", "level", "attr",
})
_UNEXPECTED_TERM_RE = re.compile(r"error: unexpected token '([^']+)'; expected term")
TERM_NOTATION_BY_TOKEN = {
    "|": 'syntax:max "|" term "|" : term',
    "‖": 'syntax:max "‖" term "‖" : term',
    "⌊": 'syntax:max "⌊" term "⌋" : term',
    "⌈": 'syntax:max "⌈" term "⌉" : term',
}


def missing_term_notation(output: str) -> set[str]:
    """Stub declarations for term notation the driver could not parse."""
    return {TERM_NOTATION_BY_TOKEN[m.group(1)]
            for m in _UNEXPECTED_TERM_RE.finditer(output)
            if m.group(1) in TERM_NOTATION_BY_TOKEN}
TOOLCHAIN_ROOT = Path.home() / ".elan" / "toolchains"
# Newest first: a macro written against a recent Lean may use syntax an older
# toolchain cannot parse, and the source archives carry no lean-toolchain file.
PREFERRED_TOOLCHAINS = (
    "leanprover--lean4---v4.33.1", "leanprover--lean4---v4.33.0",
    "leanprover--lean4---v4.32.2", "leanprover--lean4---v4.31.0",
)


@dataclass
class Program:
    """A syntax command whose expansion has to be computed rather than matched."""

    keyword: str
    category: str
    source: str
    path: str | None
    start_line: int
    end_line: int
    items: tuple[tuple[str, str, str], ...]
    tokens: int
    lead: str


@dataclass
class Site:
    """A candidate call site: a start offset and enough following text to parse.

    The end is not known until Lean parses it.  Sizing the fragment in Python is
    what produced every extent bug -- a `tacticSeq` argument truncated, a
    grouped category annotation read as three parameters -- so the slab is
    deliberately generous and the parser decides where the construct stops.
    """

    identifier: str
    path: str
    start: int
    end: int
    text: str
    category: str
    lead: str
    expansion: str | None = None
    status: str = "pending"
    # Filled in from Lean's answer: how much of the slab the construct occupied.
    extent: int = 0

    @property
    def site_text(self) -> str:
        return self.text[:self.extent] if self.extent else self.text

    @property
    def before_tokens(self) -> int:
        return count_lean_tokens_in_source(self.site_text)

    @property
    def after_tokens(self) -> int:
        return count_lean_tokens_in_source(self.expansion or "")


@dataclass
class RepositoryResult:
    model: str
    repository: str
    measured_after_tokens: int
    recorded_after_tokens: int
    expanded_tokens: int
    definition_tokens: int
    programs: list[Program] = field(default_factory=list)
    sites: list[Site] = field(default_factory=list)
    uncovered: list[dict[str, Any]] = field(default_factory=list)
    toolchain: str = ""
    driver_error: str = ""
    tactic_stubs: list[str] = field(default_factory=list)

    @property
    def harness_ok(self) -> bool:
        return self.recorded_after_tokens == self.measured_after_tokens

    @property
    def leg(self) -> int:
        return self.expanded_tokens - self.measured_after_tokens


# These always extend ``term``, and a trailing ``: max`` in one of them is a
# parameter's precedence, not a category.
_TERM_ONLY_COMMANDS = frozenset({
    "notation", "notation3", "infix", "infixl", "infixr", "prefix", "postfix",
})


def declared_category(source: str, keyword: str) -> str:
    """The syntax category a command extends, as written."""
    if keyword in _TERM_ONLY_COMMANDS:
        return "term"
    code = remove_lean_comments(source, mask_strings=True)
    arrow = code.find("=>")
    head = code[:arrow] if arrow >= 0 else code
    match = _CATEGORY_RE.search(head.strip())
    if match and match.group(1) not in pattern_pass._PRECEDENCES:  # noqa: SLF001
        return match.group(1)
    return "term"


def collect_programs(sources: SourceIndex, after, added, *, rules: str = "programs"
                     ) -> tuple[list[Program], list[dict[str, Any]], list[str]]:
    """New syntax commands that the pattern pass could not handle, plus context.

    The third result is every new syntax command's source in file order: the
    driver needs all of them, because a ``macro_rules`` only elaborates once the
    ``syntax`` declaration it answers is in scope.
    """
    programs: list[Program] = []
    uncovered: list[dict[str, Any]] = []
    context: list[tuple[str, int, str]] = []
    groups = list(written_declaration_groups(after, added))
    # A parser answered only by elab_rules has no MacroM expansion. Keep it
    # explicitly uncovered, just like inline elab commands, rather than trying
    # to measure it as a macro and failing the entire repository.
    rule_leads: dict[str, set[tuple[str, str]]] = {"elab_rules": set(), "macro_rules": set()}
    for representative, _members in groups:
        node = after.nodes[representative]
        source = sources.slice(node) or ""
        keyword = leading_command(source)
        if keyword in rule_leads:
            path = sources.path_of_module(str(node.get("module", ""))) or ""
            code = remove_lean_comments(source, mask_strings=True)
            for lead in re.findall(r"\|\s*`\(\s*([^\s$()]+)", code):
                rule_leads[keyword].add((path, lead))
    elaborator_only = rule_leads["elab_rules"] - rule_leads["macro_rules"]
    for representative, _members in groups:
        node = after.nodes[representative]
        source = sources.slice(node) or ""
        keyword = leading_command(source)
        if command_class(keyword) != "syntax_level" or not source:
            continue
        path = sources.path_of_module(str(node.get("module", "")))
        context.append((path or "", int(node["start_line"]), source))
        tokens = count_lean_tokens_in_source(source)
        parsed = pattern_pass.parse_command(source, str(keyword))
        is_template = not isinstance(parsed, str)
        if is_template and rules == "programs":
            continue  # the pattern pass prices this one
        if not is_template and rules == "templates":
            continue
        if is_template:
            # A template has no "reason"; keep the shape the rest of the code
            # expects when it falls through to the uncovered list.
            parsed = "template"
        if keyword in ("elab", "declare_syntax_cat", "macro_rules"):
            # ``elab`` is not MacroM; ``macro_rules`` answers a ``syntax``
            # declaration whose own call sites are counted through that
            # declaration, so counting it again here would double-count.
            uncovered.append({"keyword": str(keyword), "reason": parsed, "tokens": tokens})
            continue
        category = declared_category(source, str(keyword))
        if category not in TOP_LEVEL_CATEGORIES:
            uncovered.append({"keyword": str(keyword),
                              "reason": f"component of custom category {category}",
                              "tokens": tokens})
            continue
        items = extent_items(_pattern_text(source, str(keyword)))
        if isinstance(items, str):
            uncovered.append({"keyword": str(keyword), "reason": items, "tokens": tokens})
            continue
        lead = next(v.strip() for k, v, _ in items if k == "lit" and v.strip())
        if keyword == "syntax" and (path or "", lead) in elaborator_only:
            uncovered.append({"keyword": "syntax", "reason": "elab_rules: not a macro expansion",
                              "tokens": tokens})
            continue
        programs.append(Program(
            keyword=str(keyword), category=category,
            source=source, path=path, start_line=int(node["start_line"]),
            end_line=int(node["end_line"]), items=tuple(items), tokens=tokens, lead=lead,
        ))
    context.sort(key=lambda row: (row[0], row[1]))
    return programs, uncovered, [source for _, _, source in context]


def extent_items(pattern: str) -> list[tuple[str, str, str]] | str:
    """Pattern items for finding a call site's extent, not for substituting it.

    Bucket B never substitutes -- Lean produces the expansion -- so a pattern the
    substituting parser refuses is still usable here.  Repetition and groups
    become one anonymous capture, which is enough: a construct delimited by
    literals, as ``"ints[" tableInt* "]"`` is, has its extent fixed by the
    literals whatever sits between them.
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
        match = re.match(r"[A-Za-z_][\w'.]*", pattern[index:])
        if match:
            index += len(match.group(0))
            category = ""
            while index < len(pattern) and pattern[index] == ":":
                index += 1
                if index < len(pattern) and pattern[index] == "(":
                    # ``x:(colGt term:max)`` -- one parameter whose category is
                    # written as a group.  Scanning into it would read `colGt`,
                    # `term` and `max` as three more parameters, and the extent
                    # would then swallow the two tactics after the real one.
                    depth = 0
                    start = index
                    while index < len(pattern):
                        if pattern[index] == "(":
                            depth += 1
                        elif pattern[index] == ")":
                            depth -= 1
                            if depth == 0:
                                index += 1
                                break
                        index += 1
                    inner = re.search(r"(term|tactic|tacticSeq|ident|num|str)", pattern[start:index])
                    if inner and not category:
                        category = inner.group(1)
                    break
                annotation = re.match(r"[\w'.]+", pattern[index:])
                if not annotation:
                    break
                text = annotation.group(0)
                index += len(text)
                if not category and text not in pattern_pass._PRECEDENCES and not text.isdigit():  # noqa: SLF001
                    category = text
            while index < len(pattern) and pattern[index] in "*+?,":
                index += 1
            items.append(("param", f"_{len(items)}", category or "term"))
            continue
        if char in "()":
            index += 1
            continue
        index += 1
    if not any(kind == "lit" and value.strip() for kind, value, _ in items):
        return "no literal to match on"
    return items


def _pattern_text(source: str, keyword: str) -> str:
    """The pattern part of a command: after the keyword, before ``:`` or ``=>``."""
    code = remove_lean_comments(source)
    masked = remove_lean_comments(source, mask_strings=True)
    match = pattern_pass._KEYWORD_RE.search(masked)  # noqa: SLF001
    start = match.end() if match else 0
    arrow = masked.find("=>", start)
    end = arrow if arrow >= 0 else len(masked)
    head_masked = masked[start:end]
    colon = head_masked.rfind(":")
    if colon >= 0 and re.fullmatch(r"\s*[\w'.]+\s*", head_masked[colon + 1:]):
        end = start + colon
    return code[start:end]


def find_sites(files: Mapping[str, str], programs: Sequence[Program],
               enclosing: Mapping[str, Sequence[tuple[int, int]]] | None = None) -> list[Site]:
    """Candidate call sites: where a program's leading literal occurs.

    Only the start is decided here.  The slab handed to Lean runs to the end of
    the declaration the site sits in, which bounds the driver without risking a
    truncated construct.
    """
    sites: list[Site] = []
    counter = 0
    for path, body in files.items():
        # Strings stay visible: a call site's argument is often a string literal.
        masked, in_string = pattern_pass.matching_view(body)
        line_offsets = [0]
        for position, char in enumerate(body):
            if char == "\n":
                line_offsets.append(position + 1)
        ranges = sorted((enclosing or {}).get(path, []))
        for program in programs:
            index = 0
            while True:
                index = pattern_pass.find_literal(masked, program.lead, index,
                                                  in_string=in_string)
                if index < 0:
                    break
                line = masked.count("\n", 0, index) + 1
                if path == program.path and program.start_line <= line <= program.end_line:
                    index += len(program.lead)
                    continue
                enclosing_end = next(
                    (end for start, end in ranges if start <= line <= end), None)
                if enclosing_end is not None and enclosing_end < len(line_offsets):
                    slab_end = line_offsets[enclosing_end] if enclosing_end < len(line_offsets) \
                        else len(body)
                else:
                    slab_end = min(len(body), index + 6000)
                slab_end = max(slab_end, index + len(program.lead))
                counter += 1
                sites.append(Site(
                    identifier=f"s{counter}", path=path, start=index, end=slab_end,
                    text=body[index:slab_end], category=program.category, lead=program.lead,
                ))
                index += len(program.lead)
    return sites


def unknown_tactics(driver: str, output: str) -> set[str]:
    """Tactic names Lean could not parse, read off the error positions."""
    lines = driver.splitlines()
    names: set[str] = set()
    for match in _UNKNOWN_TACTIC_RE.finditer(output):
        line, column = int(match.group(1)) - 1, int(match.group(2))
        if not 0 <= line < len(lines):
            continue
        token = _token_at(lines[line], column)
        if token:
            names.add(token)
    return names


def _token_at(text: str, column: int) -> str | None:
    """The identifier Lean pointed at, tolerating either column convention.

    Lean reports this diagnostic with a one-based column, but that is not worth
    depending on: both candidates are tried, and from each the scan walks back to
    the start of the identifier, so landing anywhere inside the token works.
    """
    for start in (column - 1, column):
        if not 0 <= start < len(text):
            continue
        index = start
        while index > 0 and is_identifier_char(text[index - 1]):
            index -= 1
        token = _TOKEN_RE.match(text[index:])
        if token:
            return token.group(0)
    return None


def is_identifier_char(char: str) -> bool:
    return char.isalnum() or char in "_'"


def build_driver(context: Sequence[str], sites: Sequence[Site],
                 stubs: Sequence[str] = ()) -> str:
    """A standalone Lean file that expands each call site one step."""
    def escape(text: str) -> str:
        return text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")

    lines = [
        "import Lean",
        "open Lean Elab Command PrettyPrinter",
        "",
        "-- A notation's right-hand side names constants this driver does not have,",
        "-- because Mathlib is not imported and does not need to be: the expansion is",
        "-- measured as text, never elaborated.  Without this the precheck rejects the",
        "-- declaration, the parser registers but the macro rule does not, and every",
        "-- call site of it silently reports NO_STEP.",
        "set_option quotPrecheck false",
        "",
    ]
    if stubs:
        lines.extend([
            "-- Syntax core Lean does not declare, added because this driver asked for",
            "-- it and failed.  Only the syntax is needed: an expansion is measured as",
            "-- text and never run, so a stub that parses and prints is enough.",
        ])
        lines.extend(stubs)
        lines.append("")
    lines.append("-- The repository's new syntax commands, verbatim and in file order.")
    lines.append("")
    lines.extend(context)
    lines.extend([
        "",
        "open Lean Parser Elab Command PrettyPrinter in",
        "/-- Parse a PREFIX of `input`, so the caller need not know where the call",
        "site ends: the parser does, and its answer is Lean's layout rules rather",
        "than a guess. -/",
        "private def parsePrefix (env : Environment) (cat : Name) (input : String) :",
        "    Option (Syntax × Nat) :=",
        "  let p : ParserFn := andthenFn whitespace (categoryParser cat 0).fn",
        "  let ictx := mkInputContext input \"<site>\"",
        "  let s := p.run ictx { env := env, options := {} } (getTokenTable env)",
        "             (mkParserState input)",
        "  if s.hasError then none else some (s.stxStack.back, s.pos.byteIdx)",
        "",
        "open Lean Elab Command PrettyPrinter in",
        "private def expandSite (cat : Name) (ident input : String) : CommandElabM Unit := do",
        "  let env ← getEnv",
        "  match parsePrefix env cat input with",
        "  | none => IO.println s!\"@@SITE {ident} PARSE_ERROR\"",
        "  | .some (stx, consumed) => do",
        "    let alternatives := if stx.getKind == choiceKind then stx.getArgs else #[stx]",
        "    let own := alternatives.filter (fun a => !((`Lean).isPrefixOf a.getKind))",
        "    let core := alternatives.filter (fun a => (`Lean).isPrefixOf a.getKind)",
        "    let mut result : Option Syntax := none",
        "    let mut chosen : Name := Name.anonymous",
        "    for alternative in own ++ core do",
        "      if result.isNone then",
        "        let step ← liftMacroM (Lean.Macro.expandMacro? alternative)",
        "        if step.isSome then",
        "          result := step",
        "          chosen := alternative.getKind",
        "    match result with",
        "    | none => IO.println s!\"@@SITE {ident} NO_STEP\"",
        "    | some out => do",
        "      -- A `tactic` macro usually expands to a *sequence*, and printing",
        "      -- that as a single tactic renders only its first element -- which",
        "      -- silently loses most of the expansion.  Prefer the sequence",
        "      -- category and fall back when it does not apply.",
        "      let printCat := if cat == `tactic then `tacticSeq else cat",
        "      let fmt ← liftCoreM (do",
        "        try ppCategory printCat out catch _ => ppCategory cat out)",
        "      IO.println s!\"@@SITE {ident} OK {consumed} {chosen}\"",
        "      IO.println \"@@BEGIN\"",
        "      IO.println (toString fmt)",
        "      IO.println \"@@END\"",
        "",
    ])
    for site in sites:
        lines.append(
            f'#eval expandSite `{site.category} "{site.identifier}" "{escape(site.text)}"'
        )
    return "\n".join(lines) + "\n"


def is_core_kind(kind: str) -> bool:
    """Whether a syntax kind came from Lean itself rather than the repository."""
    return kind.startswith("Lean.") or kind == "Lean"


def parse_driver_output(text: str) -> dict[str, tuple[str, str, str, int]]:
    """Map site id to (status, expansion, syntax kind, bytes the parser consumed)."""
    results: dict[str, tuple[str, str, str, int]] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = _SITE_RE.search(lines[index])
        if not match:
            index += 1
            continue
        identifier, status = match.group(1), match.group(2)
        kind, consumed = "", 0
        tail = _OK_TAIL_RE.match(match.group(3))
        if tail:
            consumed, kind = int(tail.group(1)), tail.group(2)
        if status != "OK":
            results[identifier] = (status, "", "", 0)
            index += 1
            continue
        body: list[str] = []
        index += 1
        while index < len(lines) and "@@BEGIN" not in lines[index]:
            index += 1
        index += 1
        while index < len(lines) and "@@END" not in lines[index]:
            body.append(lines[index])
            index += 1
        results[identifier] = ("OK", _HYGIENE_RE.sub("", "\n".join(body)).strip(), kind,
                               consumed)
        index += 1
    return results


def _invoke(binary: Path, driver: str, timeout: int) -> tuple[str, str]:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "Driver.lean"
        path.write_text(driver, encoding="utf-8")
        completed = subprocess.run(
            [str(binary), str(path)], capture_output=True, text=True, timeout=timeout,
        )
    return completed.stdout, completed.stderr


def settle_stubs(binary: Path, context: Sequence[str], probe: Sequence[Site],
                 timeout: int, max_repairs: int) -> tuple[set[str], int]:
    """Repair the driver on a small probe until its declarations parse.

    Which stubs a repository needs is a property of its declarations, not of how
    many call sites are measured, so it is settled once on a probe and then
    reused.  Returns the stubs and the error count that remained.
    """
    stubs: set[str] = set()
    errors = 10 ** 9
    for _ in range(max_repairs + 1):
        driver = build_driver(context, probe, sorted(stubs))
        try:
            stdout, stderr = _invoke(binary, driver, timeout)
        except subprocess.TimeoutExpired:
            break
        combined = stdout + stderr
        errors = sum(1 for line in combined.splitlines()
                     if "error:" in line and "@@SITE" not in line)
        if errors == 0:
            break
        discovered = {f'syntax "{name}" (colGt term),* : tactic'
                      for name in unknown_tactics(driver, combined)}
        discovered |= missing_term_notation(combined)
        discovered -= stubs
        if not discovered:
            break
        stubs |= discovered
    return stubs, errors


def run_lean(context: Sequence[str], sites: Sequence[Site], toolchains: Sequence[str],
             timeout: int, max_repairs: int = 8,
             chunk: int = 1200) -> tuple[str, str, str, list[str]]:
    """Run the driver, declaring stubs for unparseable tactics until it settles.

    Returns (toolchain, stdout, error, stubs).  The best attempt wins rather than
    the last, because a repair can in principle expose a different failure, and
    the honest result is the one that expanded the most call sites.
    """
    last_error = ""
    for name in toolchains:
        binary = TOOLCHAIN_ROOT / name / "bin" / "lean"
        if not binary.is_file():
            continue
        # Settle the environment on a probe, then measure every call site with
        # it.  A driver of thousands of #evals does not finish in one run, and
        # repairing on the full set would repeat that cost each attempt.
        probe = list(sites[:min(len(sites), 200)])
        stubs, _errors = settle_stubs(binary, context, probe, timeout, max_repairs)
        outputs: list[str] = []
        expanded = 0
        timed_out = False
        for start in range(0, len(sites), chunk):
            driver = build_driver(context, sites[start:start + chunk], sorted(stubs))
            try:
                stdout, _stderr = _invoke(binary, driver, timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                continue
            outputs.append(stdout)
            expanded += sum(1 for status, body, *_ in parse_driver_output(stdout).values()
                            if status == "OK" and body)
        if expanded:
            note = "" if not timed_out else f"{name}: some chunks timed out"
            return name, "\n".join(outputs), note, sorted(stubs)
        last_error = f"{name}: no call site expanded"
    return "", "", last_error or "no usable toolchain", []


def run_project_lean(context: Sequence[str], sites: Sequence[Site], project: Path,
                     timeout: int) -> tuple[str, str, str, list[str]]:
    """Measure with an explicitly selected, already built Mathlib environment.

    ``lake env lean`` loads existing dependencies; it never runs ``lake build``.
    Keep the same one-step measurement and require an answer for every candidate.
    Unrelated elaborator declarations may report errors: their bodies are not
    executed and only successful macro expansions are used as measurements.
    """
    project = project.resolve()
    toolchain = (project/'lean-toolchain').read_text().strip()
    driver = build_driver(context, sites, [])
    driver = driver.replace('import Lean\n', '\n'.join([
        'import Lean', 'import Mathlib.Util.Notation3', 'import Mathlib.Tactic.NormNum',
        'import Mathlib.Tactic.Linarith', 'import Batteries.Tactic.SeqFocus', '']), 1)
    driver = driver.replace('open Lean Elab Command PrettyPrinter',
                            'open Lean Meta Elab Term Tactic Command PrettyPrinter')
    with tempfile.TemporaryDirectory(prefix='checkpoint-macro-') as temporary:
        path = Path(temporary)/'Driver.lean'
        path.write_text(driver, encoding='utf-8')
        try:
            completed = subprocess.run([str(Path.home()/'.elan/bin/lake'), 'env', 'lean', str(path)],
                cwd=project, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return toolchain, '', 'Mathlib macro measurement timed out', []
    output = completed.stdout
    answers = parse_driver_output(output)
    missing = {s.identifier for s in sites} - answers.keys()
    expanded = any(status == 'OK' and body for status,body,*_ in answers.values())
    error = (f'Mathlib driver missed {len(missing)} candidates' if missing else
             'Mathlib driver expanded no call sites' if not expanded else '')
    if error:
        error += '\n' + (output + completed.stderr)[-1500:]
    return toolchain + ' (existing Mathlib environment)', output, error, []


def analyse_repository(model: str, repo: str, repo_dir: Path, record: Mapping[str, Any],
                       toolchains: Sequence[str], timeout: int,
                       rules: str = "programs", *, lean_project: Path | None = None) -> RepositoryResult | None:
    need = [repo_dir / "before.json", repo_dir / "after.json",
            repo_dir / "after" / "original-sources.tar.gz"]
    if not all(path.is_file() for path in need):
        return None
    before, after = load_graph(need[0]), load_graph(need[1])
    sources = SourceIndex.from_archive(need[2])
    scope = record.get("metric_scope", {})
    sources.restrict(exclude_files=scope.get("excluded_files", []),
                     exclude_dirs=scope.get("exclude_dirs", []),
                     include_prefix=scope.get("include_prefix", ""))
    added = set(after.nodes) - set(before.nodes)
    programs, uncovered, context = collect_programs(sources, after, added, rules=rules)
    measured = sources.total_tokens()
    recorded = int(record.get("text_accounting", {}).get("after_total_tokens", -1))
    if not programs:
        return RepositoryResult(model=model, repository=repo, measured_after_tokens=measured,
                                recorded_after_tokens=recorded, expanded_tokens=measured,
                                definition_tokens=0, uncovered=uncovered)

    files = dict(sources._files)  # noqa: SLF001 -- the scored file set, already restricted
    # Where each declaration begins and ends, per file: the slab handed to Lean
    # stops at the end of the declaration a call site sits in.
    enclosing: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for node in after.nodes.values():
        node_path = sources.path_of_module(str(node.get("module", "")))
        if node_path:
            enclosing[node_path].append((int(node["start_line"]), int(node["end_line"])))
    sites = find_sites(files, programs, enclosing)
    result = RepositoryResult(
        model=model, repository=repo, measured_after_tokens=measured,
        recorded_after_tokens=recorded, expanded_tokens=measured,
        definition_tokens=sum(p.tokens for p in programs), programs=programs,
        sites=sites, uncovered=uncovered,
    )
    if not sites:
        return result

    toolchain, output, error, stubs = (run_project_lean(context, sites, lean_project, timeout)
        if lean_project is not None else run_lean(context, sites, toolchains, timeout))
    result.toolchain, result.driver_error, result.tactic_stubs = toolchain, error, stubs
    if not output:
        return result
    expansions = parse_driver_output(output)
    for site in sites:
        status, body, kind, consumed = expansions.get(
            site.identifier, ("MISSING", "", "", 0))
        if status == "OK" and is_core_kind(kind):
            # Lean expanded its own macro here, not the model's.
            status = "CORE_MACRO"
        site.status = status
        if status == "OK" and body and consumed:
            # Lean reports UTF-8 bytes; the tree is indexed in characters.
            site.extent = len(site.text.encode("utf-8")[:consumed].decode("utf-8", "ignore"))
            site.end = site.start + site.extent
            site.expansion = body
        else:
            site.expansion = None

    # Splice, highest offset first so earlier offsets stay valid, then delete the
    # commands themselves -- the expanded repository is one that never had them.
    by_file: dict[str, list[Site]] = collections.defaultdict(list)
    for site in sites:
        if site.expansion:
            by_file[site.path].append(site)
    for path, group in by_file.items():
        # A call site can sit inside another one's arguments -- `zv ...` within a
        # `zd ... zb ...` command.  Only the outermost is a site in its own
        # right; splicing a nested one as well would rewrite text that the outer
        # expansion has already replaced, and count it twice.
        kept: list[Site] = []
        reach = -1
        for site in sorted(group, key=lambda s: (s.start, -s.end)):
            if site.start < reach:
                site.status = "NESTED"
                site.expansion = None
                continue
            kept.append(site)
            reach = max(reach, site.end)
        body = files[path]
        for site in sorted(kept, key=lambda s: -s.start):
            body = body[:site.start] + f"({site.expansion})" + body[site.end:]
        files[path] = body
    deletions: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for program in programs:
        if program.path:
            deletions[program.path].append((program.start_line, program.end_line))
    for path, ranges in deletions.items():
        if path in files:
            files[path] = pattern_pass.delete_ranges(files[path], ranges)
    result.expanded_tokens = SourceIndex(files).total_tokens()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--category-dir", type=Path,
                        default=Path("output/analysis/compression_categories/runs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--toolchain", action="append", default=None)
    parser.add_argument("--rules", choices=("programs", "templates", "all"),
                        default="programs",
                        help="which declarations to measure: the programmatic ones "
                             "(bucket B), the template ones (bucket A, measured the "
                             "same way for comparison), or both")
    options = parser.parse_args()
    toolchains = options.toolchain or list(PREFERRED_TOOLCHAINS)

    results: list[RepositoryResult] = []
    for run in options.run:
        run_path = Path(run)
        model = next((m for m in ("opus", "sol", "luna", "gemini") if f"_{m}_" in run_path.name), None)
        category_path = options.category_dir / f"{run_path.name}.json"
        if model is None or not category_path.is_file():
            continue
        payload = json.loads(category_path.read_text(encoding="utf-8"))
        records = {str(row["repository"]): row for row in payload.get("repositories", [])}
        for repo_dir in sorted((run_path / "repositories").glob("*")):
            record = records.get(repo_dir.name)
            if record is None:
                continue
            try:
                result = analyse_repository(model, repo_dir.name, repo_dir, record,
                                            toolchains, options.timeout, options.rules)
            except Exception as exc:  # noqa: BLE001
                print(f"error {model}/{repo_dir.name}: {exc}", file=sys.stderr)
                continue
            if result is not None and (result.programs or result.uncovered):
                results.append(result)
                print(f"  {model}/{repo_dir.name}: programs={len(result.programs)} "
                      f"sites={len(result.sites)} "
                      f"expanded={sum(1 for s in result.sites if s.expansion)} "
                      f"leg={result.leg}", file=sys.stderr)

    totals: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    statuses: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in results:
        counter = totals[r.model]
        counter["repositories"] += 1
        counter["programs"] += len(r.programs)
        counter["sites"] += len(r.sites)
        counter["sites_expanded"] += sum(1 for s in r.sites if s.expansion)
        counter["definition_tokens"] += r.definition_tokens
        counter["uncovered_commands"] += len(r.uncovered)
        if r.harness_ok:
            counter["harness_ok"] += 1
            counter["codec_generator_leg_tokens"] += r.leg
        for site in r.sites:
            statuses[r.model][site.status] += 1

    payload = {
        "totals": {m: dict(c) for m, c in totals.items()},
        "site_status": {m: dict(c) for m, c in statuses.items()},
        "repositories": [
            {
                "model": r.model, "repository": r.repository, "toolchain": r.toolchain,
                "harness_ok": r.harness_ok, "driver_error": r.driver_error,
                "tactic_stubs": r.tactic_stubs,
                "measured_after_tokens": r.measured_after_tokens,
                "expanded_tokens": r.expanded_tokens,
                "codec_generator_leg_tokens": r.leg,
                "definition_tokens": r.definition_tokens,
                "programs": [
                    {"keyword": p.keyword, "lead": p.lead, "category": p.category,
                     "tokens": p.tokens,
                     "sites": sum(1 for s in r.sites if s.lead == p.lead),
                     "expanded": sum(1 for s in r.sites if s.lead == p.lead and s.expansion),
                     "mean_before_tokens": round(
                         sum(s.before_tokens for s in r.sites if s.lead == p.lead and s.expansion)
                         / max(1, sum(1 for s in r.sites if s.lead == p.lead and s.expansion)), 1),
                     "mean_after_tokens": round(
                         sum(s.after_tokens for s in r.sites if s.lead == p.lead and s.expansion)
                         / max(1, sum(1 for s in r.sites if s.lead == p.lead and s.expansion)), 1)}
                    for p in r.programs
                ],
                "uncovered": r.uncovered,
            }
            for r in results
        ],
    }
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                                  encoding="utf-8")
    print(json.dumps({"totals": payload["totals"], "site_status": payload["site_status"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
