#!/usr/bin/env python3
"""Quantify what contributes to a repository's Lean-token compression.

Three categories are measured from hash-verified before/after dependency graphs
plus the exact Lean sources the benchmark scores:

*dead code*
    Declarations the original repository still carried even though they are
    unreachable from every protected result.  Preprocessing retains such code
    conservatively, so deleting it costs the model no reasoning.  The
    contribution is the source tokens of the unreachable declarations that the
    model actually removed; the tokens of every unreachable declaration are
    reported alongside it as the dead code that remained available.

*structural change*
    Declarations whose name does not occur in the original repository.  The
    contribution is the tokens spent introducing them minus the tokens saved in
    the declarations that depend on them, so a helper that pays for itself
    scores positively and one that does not scores negatively.

*proof rewriting*
    Retained declarations with an unchanged statement and an identical
    dependency edge set, where no structural change can explain the difference.
    Split into *automation rewriting*, whose new proof invokes more automation
    tactics than the old one, and *explicit rewriting*, whose new proof invokes
    the same number or fewer.

Edge direction: an edge points from a declaration to a declaration it depends
on.  Users are therefore counted by incoming edges and root reachability
follows outgoing edges.  Graph association is never treated as proof of
causation, and every measured saving is reported with the count of declarations
whose source could not be located, so a silent gap cannot inflate a category.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

ROOT = Path(__file__).resolve().parents[2]
for _path in (ROOT, ROOT / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from leanlean.metrics.tokens import (  # noqa: E402
    PROOF_LENGTH_ERROR,
    _equation_proof_bodies,
    split_import_command,
    _token_count,
    count_lean_tokens_in_source,
    proof_length,
    remove_lean_comments,
)
from leanlean.preprocessing.graph_artifact import (  # noqa: E402
    EDGE_KIND_BITS,
    EDGE_KINDS,
    graph_sha256,
)
from leanlean.preprocessing.repositories import (  # noqa: E402
    repository_definition_sha256,
)

DECLARATION_KINDS = frozenset({"theorem", "def"})


class BrokenGraphError(ValueError):
    """A graph pair that cannot describe the repository it claims to describe."""

# Edge-set identity policies.  ``union`` is the graph's canonical dependency
# set; the narrower policies exist because an automation rewrite necessarily
# changes the constants a proof term mentions, so measuring proof rewriting
# under ``union`` alone can define the automation category out of existence.
EDGE_POLICIES = ("union", "static", "source_text")

# Automation tactics: those that close or normalize a goal by search rather than
# by naming the step to take.  ``simp_rw``, ``rw``, ``fin_cases`` and
# ``interval_cases`` are deliberately absent -- they take an explicit lemma or
# case list and direct the proof rather than searching for it.
AUTOMATION_SEARCH_TACTICS = frozenset({
    "aesop", "apply?", "bound", "canonical", "cc", "continuity", "decide",
    "differentiability", "duper", "exact?", "finiteness", "fun_prop", "gcongr",
    "grind", "hammer", "hint", "itauto", "linarith", "measurability",
    "native_decide", "nlinarith", "omega", "plausible", "polyrith",
    "positivity", "tauto", "trivial",
})
AUTOMATION_NORMALIZATION_TACTICS = frozenset({
    "abel", "abel_nf", "field_simp", "group", "module", "norm_cast", "norm_fin",
    "norm_num", "push_cast", "reduce_mod_char", "ring", "ring_nf", "simp",
    "simp_all", "simp_arith", "simpa",
})
AUTOMATION_TACTICS = AUTOMATION_SEARCH_TACTICS | AUTOMATION_NORMALIZATION_TACTICS

AUTOMATION_POLICIES = ("strict", "broad")

# Commands that introduce a constant other declarations can depend on, so their
# in-degree means something.
TERM_COMMANDS = frozenset({
    "abbrev", "axiom", "class", "def", "example", "inductive", "instance", "lemma",
    "opaque", "structure", "theorem",
})
# Commands that introduce syntax.  Using the notation leaves no dependency edge on
# the node that defines it, so their in-degree is always zero and averaging over
# them says nothing about whether the model's abstraction was reused.
SYNTAX_COMMANDS = frozenset({
    "app_unexpander", "binder_predicate", "declare_syntax_cat", "delab", "elab",
    "infix", "infixl", "infixr", "macro", "macro_rules", "notation", "notation3",
    "postfix", "prefix", "syntax", "unif_hint",
})

# Import roots that count as reaching for an external library rather than for
# another part of the repository under edit.
LIBRARY_ROOTS = frozenset({"Mathlib", "Batteries", "Std", "Aesop", "Plausible", "Qq", "ProofWidgets"})

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_.'!?]*")
_SUFFIXES = "?!"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Dependency graphs
# --------------------------------------------------------------------------- #


def _edge_pair(edge: Any, nodes: list[dict[str, Any]], context: str) -> tuple[str, str]:
    if not isinstance(edge, list) or len(edge) < 2:
        raise ValueError(f"{context}: malformed edge {edge!r}")
    source, target = edge[0], edge[1]
    if isinstance(source, bool) or isinstance(target, bool):
        raise ValueError(f"{context}: boolean edge index is invalid")
    if isinstance(source, int) and isinstance(target, int):
        if not (0 <= source < len(nodes) and 0 <= target < len(nodes)):
            raise ValueError(f"{context}: edge index out of range: {edge!r}")
        return str(nodes[source]["name"]), str(nodes[target]["name"])
    if isinstance(source, str) and isinstance(target, str):
        return source, target
    raise ValueError(f"{context}: endpoints must both be indexes or names")


# Schemas whose static edges carry an edge-kind bitmask (v3 adds no new bits);
# the identity-aligned graph checkpoint_identity writes carries them by name.
CHECKSUMMED_GRAPH_SCHEMAS = frozenset({"preprocessing_dependency_graph_v2", "preprocessing_dependency_graph_v3"})
IDENTITY_GRAPH_SCHEMA = "sendov_identity_graph_v2"
MASKED_GRAPH_SCHEMAS = CHECKSUMMED_GRAPH_SCHEMAS | {IDENTITY_GRAPH_SCHEMA}


@dataclass(frozen=True)
class GraphData:
    """Nodes plus the edge sets of every evidence layer that found an edge."""

    nodes: dict[str, dict[str, Any]]
    layers: dict[str, set[tuple[str, str]]]
    static_edges: set[tuple[str, str]]
    grind_edges: set[tuple[str, str]]
    schema: str
    sha256: str | None

    @property
    def edges(self) -> set[tuple[str, str]]:
        """The canonical dependency set: static evidence plus Grind edges."""
        return self.static_edges | self.grind_edges

    @property
    def confirmed_edges(self) -> set[tuple[str, str]]:
        """Edges elaboration confirms (kernel, ilean or Grind).

        Source-text matching alone resolves bare binder names (``U``, ``r``)
        to unrelated globals once they are hoisted into ``variable``.  A graph
        without layer attribution falls back to the canonical set.
        """
        if self.schema not in MASKED_GRAPH_SCHEMAS:
            return self.edges
        return self.layers.get("kernel", set()) | self.layers.get("ilean", set()) | self.grind_edges

    def edge_set(self, policy: str) -> set[tuple[str, str]]:
        if policy == "union":
            return self.edges
        if policy == "static":
            return set(self.static_edges)
        if policy in self.layers:
            return set(self.layers[policy])
        raise ValueError(f"unknown edge policy {policy!r}")


def load_graph(path: Path) -> GraphData:
    """Read legacy named edges or a validated v2 indexed/masked graph artifact."""
    payload = read_json(path)
    raw_nodes, raw_edges = payload.get("nodes"), payload.get("edges")
    if not isinstance(raw_nodes, list) or not isinstance(raw_edges, list):
        raise ValueError(f"{path}: expected nodes and edges arrays")
    nodes: dict[str, dict[str, Any]] = {}
    for node in raw_nodes:
        if not isinstance(node, dict) or not isinstance(node.get("name"), str):
            raise ValueError(f"{path}: malformed node {node!r}")
        if node["name"] in nodes:
            raise ValueError(f"{path}: duplicate node {node['name']!r}")
        nodes[node["name"]] = node

    schema = payload.get("schema")
    if schema in CHECKSUMMED_GRAPH_SCHEMAS:
        if payload.get("sha256") != graph_sha256(payload):
            raise ValueError(f"{path}: invalid or missing graph envelope SHA-256")

    layers: dict[str, set[tuple[str, str]]] = {kind: set() for kind in EDGE_KINDS}
    static_edges: set[tuple[str, str]] = set()
    for edge in raw_edges:
        pair = _edge_pair(edge, raw_nodes, f"{path}:static")
        if pair[0] not in nodes or pair[1] not in nodes:
            raise ValueError(f"{path}: static edge references an unknown node")
        if schema in MASKED_GRAPH_SCHEMAS:
            if len(edge) != 3 or isinstance(edge[2], bool) or not isinstance(edge[2], int):
                raise ValueError(f"{path}: masked static edge must have an integer bitmask")
            mask = edge[2]
            if mask <= 0 or mask & ~sum(EDGE_KIND_BITS.values()):
                raise ValueError(f"{path}: invalid edge-kind bitmask {mask!r}")
            for kind, bit in EDGE_KIND_BITS.items():
                if mask & bit:
                    layers[kind].add(pair)
        else:
            # A legacy artifact records no layer attribution.  Treat its edges
            # as source-text evidence so the narrow policies stay well defined.
            layers["source_text"].add(pair)
        static_edges.add(pair)

    policies = payload.get("policies", {})
    production = policies.get("production", {}) if isinstance(policies, dict) else {}
    raw_grind = production.get("grind_edges", []) if isinstance(production, dict) else []
    if not isinstance(raw_grind, list):
        raise ValueError(f"{path}: production grind_edges must be an array")
    grind_edges: set[tuple[str, str]] = set()
    for edge in raw_grind:
        pair = _edge_pair(edge, raw_nodes, f"{path}:production-grind")
        if pair[0] not in nodes or pair[1] not in nodes:
            raise ValueError(f"{path}: Grind edge references an unknown node")
        grind_edges.add(pair)
    layers["grind"] = set(grind_edges)
    return GraphData(
        nodes=nodes,
        layers=layers,
        static_edges=static_edges,
        grind_edges=grind_edges,
        schema=str(schema or "legacy"),
        sha256=payload.get("sha256") if isinstance(payload.get("sha256"), str) else None,
    )


def graph(path: Path) -> tuple[dict[str, dict[str, Any]], set[tuple[str, str]]]:
    """Compatibility wrapper returning nodes and the static-plus-Grind union."""
    loaded = load_graph(path)
    return loaded.nodes, loaded.edges


def reachable(starts: Iterable[str], edges: set[tuple[str, str]]) -> set[str]:
    """Nodes reachable from ``starts`` by following dependency edges outward."""
    adjacency: dict[str, set[str]] = defaultdict(set)
    for source, target in edges:
        adjacency[source].add(target)
    seen = set(starts)
    pending = list(seen)
    while pending:
        for target in adjacency.get(pending.pop(), ()):
            if target not in seen:
                seen.add(target)
                pending.append(target)
    return seen


def reverse_edges(edges: set[tuple[str, str]]) -> set[tuple[str, str]]:
    return {(target, source) for source, target in edges}


def user_map(edges: set[tuple[str, str]]) -> dict[str, set[str]]:
    """Map each declaration to the declarations that depend on it."""
    result: dict[str, set[str]] = defaultdict(set)
    for user, dependency in edges:
        result[dependency].add(user)
    return result


def dependency_map(edges: set[tuple[str, str]]) -> dict[str, frozenset[str]]:
    """Map each declaration to the set of declarations it depends on."""
    collected: dict[str, set[str]] = defaultdict(set)
    for user, dependency in edges:
        collected[user].add(dependency)
    return {name: frozenset(values) for name, values in collected.items()}


def removed_by_protected_reachability(
    before_names: set[str],
    after_names: set[str],
    protected: Iterable[str],
    before_edges: set[tuple[str, str]],
) -> tuple[set[str], set[str]]:
    """Return (removed and outside the original protected closure, removed inside it)."""
    deleted = before_names - after_names
    original_closure = reachable(protected, before_edges)
    return deleted - original_closure, deleted & original_closure


# --------------------------------------------------------------------------- #
# Lean sources
# --------------------------------------------------------------------------- #


class SourceIndex:
    """Line-addressable Lean sources, from a directory or a source tarball.

    Graph nodes carry ``module``, ``start_line`` and ``end_line``, so every
    measurement slices the exact source the benchmark scored.  Module-to-path
    resolution is driven by the file set actually present rather than by a fixed
    ``src/`` convention, because repository layouts differ: some keep
    ``ZetaLean/Pub1.lean`` at the root and others keep ``src/Langlib.lean``.
    """

    def __init__(self, files: Mapping[str, str]):
        self._files = dict(files)
        self._lines: dict[str, list[str]] = {}
        self._line_tokens: dict[str, list[int]] = {}
        self._resolved: dict[str, str | None] = {}
        self.unresolved_modules: set[str] = set()

    @classmethod
    def from_directory(cls, root: Path) -> SourceIndex:
        files: dict[str, str] = {}
        for path in sorted(root.rglob("*.lean")):
            if ".lake" in path.relative_to(root).parts:
                continue
            files[path.relative_to(root).as_posix()] = path.read_text(
                encoding="utf-8", errors="replace"
            )
        return cls(files)

    @classmethod
    def from_archive(cls, path: Path) -> SourceIndex:
        files: dict[str, str] = {}
        with tarfile.open(path, mode="r:*") as archive:
            for member in archive.getmembers():
                if not member.isfile() or not member.name.endswith(".lean"):
                    continue
                name = member.name[2:] if member.name.startswith("./") else member.name
                if ".lake" in Path(name).parts:
                    continue
                stream = archive.extractfile(member)
                if stream is None:
                    continue
                files[name] = stream.read().decode("utf-8", errors="replace")
        return cls(files)

    def restrict(
        self,
        *,
        exclude_files: Iterable[str] = (),
        exclude_dirs: Iterable[str] = (),
        include_prefix: str = "",
    ) -> None:
        """Narrow the file set to exactly what the benchmark metric scores.

        The Challenge file is written into the container by the harness and is
        excluded from the score, so leaving it in would make a repository look
        like it grew.  Called before anything is read, so no cache can go stale.
        """
        excluded = {name.strip().strip("/") for name in exclude_files if name.strip()}
        directories = [name.strip().strip("/") for name in exclude_dirs if name.strip()]
        prefix = include_prefix.strip().strip("/")

        def in_scope(path: str) -> bool:
            if path in excluded:
                return False
            if any(path == item or path.startswith(item + "/") for item in directories):
                return False
            if prefix and not (path == prefix + ".lean" or path.startswith(prefix + "/")):
                return False
            return True

        self._files = {name: body for name, body in self._files.items() if in_scope(name)}
        self._lines.clear()
        self._line_tokens.clear()
        self._resolved.clear()

    def paths(self) -> list[str]:
        """Scored Lean files, excluding the build script the metric ignores."""
        return sorted(name for name in self._files if name != "lakefile.lean")

    def path_of_module(self, module: str) -> str | None:
        return self._resolve(module)

    def line_tokens(self, path: str) -> list[int] | None:
        """Scored tokens per line of a file.

        Comments are stripped from the whole file first, so a block comment
        spanning lines cannot leak tokens, and import commands are priced at zero.
        Summing this list reproduces ``count_lean_tokens_in_source`` for the file
        exactly, which is what makes a line partition a partition of the score.
        """
        if path not in self._files:
            return None
        if path not in self._line_tokens:
            source = self._files[path]
            without_comments = remove_lean_comments(source).splitlines()
            masked = remove_lean_comments(source, mask_strings=True).splitlines()
            self._line_tokens[path] = [
                _token_count(split_import_command(source_line, masked_line)[1])
                for source_line, masked_line in zip(without_comments, masked)
            ]
        return self._line_tokens[path]

    def total_tokens(self) -> int:
        total = 0
        for path in self.paths():
            per_line = self.line_tokens(path)
            if per_line is not None:
                total += sum(per_line)
        return total

    def imports_by_path(self) -> dict[str, set[str]]:
        """The modules each file imports, keyed by file path."""
        result: dict[str, set[str]] = {}
        for path in self.paths():
            names: set[str] = set()
            masked = remove_lean_comments(self._files[path], mask_strings=True)
            for line in masked.splitlines():
                modules, _ = split_import_command(line, line)
                if modules:
                    names.add(modules[-1])
            result[path] = names
        return result

    @staticmethod
    def module_relative_path(module: str) -> str:
        """Turn a Lean module name into the path its source lives at.

        A component that is not a plain identifier is written between French
        quotes, so a directory with a space in its name reaches us as
        ``«Adic spaces».Presheaf`` and its file is ``Adic spaces/Presheaf.lean``.
        Splitting on every dot and keeping the quotes produces a path that
        matches nothing, and every declaration in such a directory then prices
        at zero.
        """
        components: list[str] = []
        current: list[str] = []
        quoted = 0
        for character in module:
            if character == "\u00ab":
                quoted += 1
            elif character == "\u00bb":
                quoted = max(0, quoted - 1)
            elif character == "." and not quoted:
                components.append("".join(current))
                current = []
            else:
                current.append(character)
        components.append("".join(current))
        return "/".join(part for part in components if part) + ".lean"

    def _resolve(self, module: str) -> str | None:
        if module in self._resolved:
            return self._resolved[module]
        relative = self.module_relative_path(module)
        candidates = [relative, "src/" + relative]
        found = next((name for name in candidates if name in self._files), None)
        if found is None:
            # Fall back to a unique suffix match so an unexpected source root
            # cannot silently zero out a category.
            suffix = "/" + relative
            matches = [name for name in self._files if name.endswith(suffix)]
            found = matches[0] if len(matches) == 1 else None
        if found is None:
            self.unresolved_modules.add(module)
        self._resolved[module] = found
        return found

    def lines(self, module: str) -> list[str] | None:
        name = self._resolve(module)
        if name is None:
            return None
        if name not in self._lines:
            self._lines[name] = self._files[name].splitlines()
        return self._lines[name]

    def slice(self, node: Mapping[str, Any]) -> str | None:
        """Return the source of one graph node, or None if it cannot be located."""
        lines = self.lines(str(node.get("module", "")))
        if lines is None:
            return None
        start, end = int(node["start_line"]), int(node["end_line"])
        if start < 1 or end < start or end > len(lines):
            return None
        return "\n".join(lines[start - 1 : end])

    def range_tokens(self, nodes: Iterable[Mapping[str, Any]]) -> tuple[int, int]:
        """Token count of the union of node source ranges, plus a missing count.

        Overlapping and adjacent ranges are merged first, so an inductive and
        its constructors are never charged twice.
        """
        ranges: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for node in nodes:
            ranges[str(node.get("module", ""))].append(
                (int(node["start_line"]), int(node["end_line"]))
            )
        total = missing = 0
        for module, intervals in ranges.items():
            lines = self.lines(module)
            if lines is None:
                missing += len(intervals)
                continue
            merged: list[list[int]] = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1] + 1:
                    merged[-1][1] = max(merged[-1][1], end)
                else:
                    merged.append([start, end])
            for start, end in merged:
                if start < 1 or end > len(lines):
                    missing += 1
                    continue
                total += count_lean_tokens_in_source("\n".join(lines[start - 1 : end]))
        return total, missing


_LEADING_COMMAND_RE = re.compile(
    r"""^[ \t]*
        (?:@\[[^\n]*\][ \t]*)*
        (?:(?:private|protected|noncomputable|unsafe|partial|local|scoped)[ \t]+)*
        (?P<keyword>[A-Za-z_][A-Za-z0-9_]*)
    """,
    re.VERBOSE,
)


def leading_command(source: str) -> str | None:
    """The command keyword a declaration's source begins with.

    Repositories define their own commands -- one in this benchmark declares
    theorems with ``zn`` -- so the keyword cannot be assumed to be a Lean one.
    Reporting it as found keeps a project DSL visible instead of silently folding
    it in with ``theorem``.
    """
    for line in remove_lean_comments(source, mask_strings=True).splitlines():
        match = _LEADING_COMMAND_RE.match(line)
        if match:
            return match.group("keyword")
    return None


def command_class(keyword: str | None) -> str:
    if keyword in TERM_COMMANDS:
        return "term_level"
    if keyword in SYNTAX_COMMANDS:
        return "syntax_level"
    return "other_command"


_DECLARATION_COMMAND_RE = re.compile(
    r"""^[ \t]*
        (?:@\[[^\n]*\][ \t]*)*
        (?:(?:private|protected|noncomputable|unsafe|partial|local|scoped)[ \t]+)*
        (?:theorem|lemma|example|def|abbrev|opaque|instance|axiom|constant|
           structure|class|inductive)\b
    """,
    re.VERBOSE,
)


def is_declaration_command(source: str) -> bool:
    """True when a node's source range really is a declaration command.

    Structure fields and constructors appear in the graph as nodes whose range
    covers a single field line such as ``scale : 8 <= L``.  Those have no proof
    to rewrite, so they must not be counted as measurable declarations.
    """
    masked = remove_lean_comments(source, mask_strings=True)
    return any(_DECLARATION_COMMAND_RE.match(line) for line in masked.splitlines())


def split_statement_and_proof(source: str) -> tuple[str, str] | None:
    """Split a declaration into statement and proof text.

    The split mirrors ``metrics.tokens.proof_length`` exactly -- first ``:= by``,
    then ``:=``, then equation-style ``| pattern => body`` clauses -- so the
    proof text used for automation counting is the same text whose tokens are
    counted.
    """
    without_comments = remove_lean_comments(source)
    if ":= by" in without_comments:
        statement, proof = without_comments.split(":= by", maxsplit=1)
        return statement, proof.strip()
    if ":=" in without_comments:
        statement, proof = without_comments.split(":=", maxsplit=1)
        return statement, proof.strip()
    bodies = _equation_proof_bodies(source)
    if not bodies:
        return None
    return without_comments, "\n".join(bodies)


def _mask_bracketed(text: str) -> str:
    """Blank the contents of ``[...]`` so lemma lists cannot look like tactics.

    Without this, ``simp only [Nat.decide_eq]`` or ``aesop (rule_sets := [Foo])``
    would contribute spurious automation invocations.  Whitespace is preserved so
    offsets and line structure survive.
    """
    result: list[str] = []
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
            result.append(char)
        elif char == "]":
            depth = max(0, depth - 1)
            result.append(char)
        elif depth and not char.isspace():
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


@dataclass(frozen=True)
class AutomationCount:
    total: int
    search: int
    normalization: int
    targeted: int
    per_tactic: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "search": self.search,
            "normalization": self.normalization,
            "targeted_only_invocations": self.targeted,
            "per_tactic": dict(sorted(self.per_tactic.items())),
        }


def count_automation_tactics(
    proof_source: str,
    *,
    policy: str = "strict",
    extra_tactics: Iterable[str] = (),
) -> AutomationCount:
    """Count automation-tactic invocations in a proof.

    An automation tactic closes or normalizes a goal by search instead of naming
    the step, so a proof that gets shorter by invoking more of them has been
    compressed by automation rather than by explicit reasoning.

    Under the ``strict`` policy an ``X only [...]`` invocation is *not*
    automation: the lemma list makes the step explicit.  Under ``broad`` it is.
    Both are reported for every repository so the choice is visible rather than
    buried.
    """
    if policy not in AUTOMATION_POLICIES:
        raise ValueError(f"unknown automation policy {policy!r}")
    tactics = AUTOMATION_TACTICS | {name.strip() for name in extra_tactics if name.strip()}
    masked = _mask_bracketed(remove_lean_comments(proof_source, mask_strings=True))
    words = [match.group(0) for match in _WORD_RE.finditer(masked)]
    per_tactic: dict[str, int] = defaultdict(int)
    search = normalization = targeted = 0
    for index, raw in enumerate(words):
        name = raw.rstrip(_SUFFIXES)
        if name not in tactics:
            continue
        follows_only = index + 1 < len(words) and words[index + 1] == "only"
        if follows_only:
            targeted += 1
            if policy == "strict":
                continue
        per_tactic[name] += 1
        if name in AUTOMATION_SEARCH_TACTICS:
            search += 1
        else:
            normalization += 1
    return AutomationCount(
        total=search + normalization,
        search=search,
        normalization=normalization,
        targeted=targeted,
        per_tactic=dict(per_tactic),
    )


@dataclass(frozen=True)
class Declaration:
    """One graph node together with everything measured from its source."""

    name: str
    kind: str
    module: str
    located: bool
    is_command: bool
    source: str
    statement: str
    proof: str
    total_tokens: int | None
    proof_tokens: int | None
    automation: dict[str, AutomationCount]

    @property
    def statement_tokens(self) -> int | None:
        if self.total_tokens is None or self.proof_tokens is None:
            return None
        return self.total_tokens - self.proof_tokens


def read_declaration(
    node: Mapping[str, Any],
    source: SourceIndex,
    *,
    extra_tactics: Iterable[str] = (),
) -> Declaration:
    name, kind = str(node["name"]), str(node.get("kind") or "unknown")
    module = str(node.get("module", ""))
    text = source.slice(node)
    if text is None:
        return Declaration(name, kind, module, False, False, "", "", "", None, None, {})
    command = is_declaration_command(text)
    split = split_statement_and_proof(text)
    statement, proof = split if split is not None else ("", "")
    measured = proof_length(text)
    proof_tokens = None if measured >= PROOF_LENGTH_ERROR else measured
    automation = {
        policy: count_automation_tactics(proof, policy=policy, extra_tactics=extra_tactics)
        for policy in AUTOMATION_POLICIES
    }
    return Declaration(
        name=name,
        kind=kind,
        module=module,
        located=True,
        is_command=command,
        source=text,
        statement=statement,
        proof=proof,
        total_tokens=count_lean_tokens_in_source(text),
        proof_tokens=proof_tokens,
        automation=automation,
    )


# --------------------------------------------------------------------------- #
# Pinned inputs
# --------------------------------------------------------------------------- #


def load_protected(dataset_path: Path) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Read protected declarations from the dataset's hash-pinned repository database."""
    dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
    if not isinstance(dataset, dict) or not isinstance(dataset.get("repository_database"), dict):
        raise ValueError(f"{dataset_path}: missing repository database pin")
    pin = dataset["repository_database"]
    db_path = (dataset_path.parent / str(pin.get("path", ""))).resolve()
    if not db_path.is_file():
        raise ValueError(f"{dataset_path}: repository database not found: {db_path}")
    raw = db_path.read_bytes()
    file_hash = hashlib.sha256(raw).hexdigest()
    if file_hash != pin.get("sha256"):
        raise ValueError(f"{dataset_path}: repository database file hash does not match pin")
    database = json.loads(raw)
    definition_hash = repository_definition_sha256(database)
    if definition_hash != pin.get("definition_sha256"):
        raise ValueError(f"{dataset_path}: repository database definition hash does not match pin")
    protected: dict[str, list[str]] = {}
    scopes: dict[str, dict[str, Any]] = {}
    for row in database.get("repositories", []):
        repo = row.get("instance_id")
        declarations = row.get("protected", {}).get("declarations")
        if not isinstance(repo, str) or not isinstance(declarations, list) or not all(
            isinstance(name, str) and name for name in declarations
        ):
            raise ValueError(f"{db_path}: malformed protected-declaration row")
        protected[repo] = sorted(set(declarations))
        scope = row.get("scopes") if isinstance(row.get("scopes"), dict) else {}
        scopes[repo] = {
            "exclude_dirs": list(scope.get("exclude_dirs") or []),
            "include_prefix": str(scope.get("target_dir") or ""),
        }
    return protected, scopes, {"sha256": file_hash, "definition_sha256": definition_hash}


def load_declaration_text(path: Path, expected_graph_sha256: str | None) -> dict[str, dict[str, str]]:
    """Read elaborated statements, checking they were dumped from this graph."""
    payload = read_json(path)
    rows = payload.get("declarations")
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a declarations array")
    recorded = payload.get("graph_sha256")
    if expected_graph_sha256 and isinstance(recorded, str) and recorded != expected_graph_sha256:
        raise ValueError(f"{path}: declaration text was dumped from a different graph artifact")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ValueError(f"{path}: malformed declaration row")
        result[str(row["name"])] = {
            "kind": str(row.get("kind", "")),
            "statement": str(row.get("statement", "")),
            "proof": str(row.get("proof", "")),
        }
    return result


def challenge_source_path(comparator_root: Path, repo: str) -> str | None:
    """The Challenge file the harness registers for a repository, if pinned."""
    manifest = comparator_root / repo / "comparator" / "manifest.json"
    if not manifest.is_file():
        return None
    challenge = read_json(manifest).get("challenge")
    if not isinstance(challenge, dict):
        return None
    source = challenge.get("source_path")
    return str(source) if isinstance(source, str) and source else None


def official_totals(path: Path) -> dict[str, int | float]:
    """Read the authoritative scored token totals from a postprocessed playback."""
    payload = read_json(path)
    missing = [key for key in ("baseline_lean_tokens", "post_lean_tokens", "lean_tokens_saved")
               if not isinstance(payload.get(key), int)]
    if missing:
        raise ValueError(f"{path}: playback is missing scored totals {missing}")
    baseline = int(payload["baseline_lean_tokens"])
    post = int(payload["post_lean_tokens"])
    saved = int(payload["lean_tokens_saved"])
    if baseline - post != saved:
        raise ValueError(f"{path}: baseline minus post ({baseline - post}) disagrees with saved ({saved})")
    selected = payload.get("selected_state") or {}
    return {
        "baseline_lean_tokens": baseline,
        "post_lean_tokens": post,
        "lean_tokens_saved": saved,
        "lean_token_compression_pct": float(selected.get("lean_token_compression_pct", 0.0)),
        "build_passed": bool(payload.get("build_passed")),
    }


# --------------------------------------------------------------------------- #
# Measurement helpers
# --------------------------------------------------------------------------- #


def _normalized_text(text: str) -> str:
    """Whitespace-insensitive, comment-free form for comparing two sources."""
    return " ".join(remove_lean_comments(text).split())


def _delta(before: Declaration | None, after: Declaration | None, *, proof_only: bool) -> int | None:
    """Signed token reduction from before to after, or None when unmeasurable."""
    if before is None or after is None:
        return None
    left = before.proof_tokens if proof_only else before.total_tokens
    right = after.proof_tokens if proof_only else after.total_tokens
    if left is None or right is None:
        return None
    return left - right


def _delta_summary(rows: Sequence[Mapping[str, Any]], key: str = "tokens_saved") -> dict[str, int]:
    measured = [row for row in rows if row.get(key) is not None]
    return {
        "count": len(rows),
        "measured_count": len(measured),
        "unmeasured_count": len(rows) - len(measured),
        "tokens_saved": sum(int(row[key]) for row in measured),
    }


def _kind_counts(names: Iterable[str], nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for name in names:
        counts[str(nodes[name].get("kind") or "unknown")] += 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------- #
# The three categories
# --------------------------------------------------------------------------- #


def dead_code_section(
    before: GraphData,
    after: GraphData,
    protected: Sequence[str],
    before_source: SourceIndex,
) -> dict[str, Any]:
    """Dead code: unreachable from every protected root in the original graph.

    Liveness follows the canonical edge union, Grind edges included: a
    Grind-discovered dependency keeps a declaration alive just as a textual one
    does, so ignoring that layer would overstate how much dead code was on offer.
    """
    before_names, after_names = set(before.nodes), set(after.nodes)
    live = reachable(protected, before.edges)
    dead_available = before_names - live
    deleted = before_names - after_names
    dead_removed = dead_available & deleted
    dead_retained = dead_available - deleted
    live_removed = deleted - dead_available

    available_tokens, available_missing = before_source.range_tokens(
        before.nodes[name] for name in dead_available
    )
    live_removed_tokens, live_removed_missing = before_source.range_tokens(
        before.nodes[name] for name in live_removed
    )
    removed_tokens, removed_missing = before_source.range_tokens(
        before.nodes[name] for name in dead_removed
    )
    retained_tokens, retained_missing = before_source.range_tokens(
        before.nodes[name] for name in dead_retained
    )
    return {
        "definition": (
            "unreachable from every protected root in the original graph; "
            "preprocessing retained it conservatively"
        ),
        "contribution_tokens": removed_tokens,
        "available_nodes": len(dead_available),
        "available_tokens": available_tokens,
        "available_by_kind": _kind_counts(dead_available, before.nodes),
        "removed_nodes": len(dead_removed),
        "removed_tokens": removed_tokens,
        "removed_by_kind": _kind_counts(dead_removed, before.nodes),
        "retained_nodes": len(dead_retained),
        "retained_tokens": retained_tokens,
        "retained_by_kind": _kind_counts(dead_retained, before.nodes),
        "removed_share_of_available_tokens_pct": (
            round(100.0 * removed_tokens / available_tokens, 6) if available_tokens else 0.0
        ),
        "deleted_but_protected_reachable_nodes": len(live_removed),
        "deleted_but_protected_reachable_tokens": live_removed_tokens,
        "deleted_but_protected_reachable_by_kind": _kind_counts(live_removed, before.nodes),
        "deleted_but_protected_reachable_names": sorted(live_removed),
        "live_removal_note": (
            "These declarations were reachable from a protected root in the original "
            "graph, so they were not dead: the model could only delete them by "
            "rewriting the proofs that used them.  They are reported here rather "
            "than folded into dead code, and they are the reason a repository can "
            "compress far beyond its dead-code budget."
        ),
        "unlocated_source_ranges": {
            "available": available_missing,
            "removed": removed_missing,
            "retained": retained_missing,
            "deleted_but_protected_reachable": live_removed_missing,
        },
        "removed_names": sorted(dead_removed),
    }


def structural_section(
    before: GraphData,
    after: GraphData,
    before_decls: Mapping[str, Declaration],
    after_decls: Mapping[str, Declaration],
    after_source: SourceIndex,
    *,
    edge_policy: str,
) -> dict[str, Any]:
    """New declarations: what they cost, and what they saved in their dependents.

    A dependent is a retained declaration that depends on at least one new
    declaration in the final graph.  Because a new name cannot occur in the
    original graph, every such edge is new by construction.  The savings are the
    dependents' token reduction, attributed once per dependent so a proof that
    calls two new helpers is not counted twice.
    """
    before_names, after_names = set(before.nodes), set(after.nodes)
    retained = before_names & after_names
    added = after_names - before_names
    new_decls = {name for name in added if str(after.nodes[name].get("kind")) in DECLARATION_KINDS}

    cost_tokens, cost_missing = after_source.range_tokens(
        after.nodes[name] for name in new_decls
    )
    all_added_tokens, all_added_missing = after_source.range_tokens(
        after.nodes[name] for name in added
    )

    after_edges = after.edge_set(edge_policy)
    before_edges = before.edge_set(edge_policy)
    dependencies = dependency_map(after_edges)
    before_dependencies = dependency_map(before_edges)
    direct = {name for name in retained if dependencies.get(name, frozenset()) & new_decls}
    transitive = (reachable(new_decls, reverse_edges(after_edges)) - new_decls) & retained
    # The wider reading of "gained a dependency": any retained declaration whose
    # dependency set grew, whether or not the new dependency is itself new.  A
    # proof that switched to an existing lemma lands here and not in ``direct``.
    gained_any_edge = {
        name for name in retained
        if dependencies.get(name, frozenset()) - before_dependencies.get(name, frozenset())
    }

    rows: list[dict[str, Any]] = []
    for name in sorted(direct):
        before_declaration, after_declaration = before_decls.get(name), after_decls.get(name)
        rows.append({
            "name": name,
            "kind": str(after.nodes[name].get("kind") or "unknown"),
            "new_dependencies": sorted(dependencies.get(name, frozenset()) & new_decls),
            "tokens_saved": _delta(before_declaration, after_declaration, proof_only=False),
            "proof_tokens_saved": _delta(before_declaration, after_declaration, proof_only=True),
        })
    transitive_rows = [{
        "name": name,
        "tokens_saved": _delta(before_decls.get(name), after_decls.get(name), proof_only=False),
    } for name in sorted(transitive)]
    gained_edge_rows = [{
        "name": name,
        "tokens_saved": _delta(before_decls.get(name), after_decls.get(name), proof_only=False),
        "added_dependencies": sorted(
            dependencies.get(name, frozenset()) - before_dependencies.get(name, frozenset())
        ),
    } for name in sorted(gained_any_edge)]

    direct_savings = _delta_summary(rows)
    transitive_savings = _delta_summary(transitive_rows)
    gained_edge_savings = _delta_summary(gained_edge_rows)
    net = direct_savings["tokens_saved"] - cost_tokens

    helper_rows: list[dict[str, Any]] = []
    users = user_map(after_edges)
    for name in sorted(new_decls):
        declaration = after_decls.get(name)
        helper_rows.append({
            "name": name,
            "kind": str(after.nodes[name].get("kind") or "unknown"),
            "module": str(after.nodes[name].get("module", "")),
            "tokens": None if declaration is None else declaration.total_tokens,
            "direct_retained_users": sorted(users.get(name, set()) & retained),
            "direct_new_declaration_users": sorted(users.get(name, set()) & new_decls),
        })

    return {
        "definition": (
            "declarations whose name does not occur in the original repository; "
            "contribution is dependent savings minus introduction cost"
        ),
        "edge_policy": edge_policy,
        "contribution_tokens": net,
        "new_declarations": len(new_decls),
        "new_declarations_by_kind": _kind_counts(new_decls, after.nodes),
        "introduction_cost_tokens": cost_tokens,
        "all_added_nodes": len(added),
        "all_added_tokens": all_added_tokens,
        "direct_dependents": direct_savings,
        "transitive_dependents": transitive_savings,
        "dependents_that_gained_any_edge": gained_edge_savings,
        "tokens_added_by_new_declarations": cost_tokens,
        "tokens_removed_from_dependents": direct_savings["tokens_saved"],
        "net_tokens_saved": net,
        "cost_minus_savings_tokens": cost_tokens - direct_savings["tokens_saved"],
        "pays_for_itself": net > 0,
        "unlocated_source_ranges": {"new": cost_missing, "all_added": all_added_missing},
        "attribution_note": (
            "Dependent savings are graph associations, not proven causation: an "
            "edge may originate in a declaration's type as well as in its body."
        ),
        "new_declaration_rows": helper_rows,
        "dependent_rows": rows,
        "gained_edge_rows": gained_edge_rows,
    }


REWRITING_SCOPES = (
    # The paragraph's definition: no dependency change can explain the saving.
    "identical_edge_set",
    # Same statement and no reliance on a new declaration, so the saving is not
    # already counted as structural change.  Wider than the edge-set test, which
    # an automation rewrite almost always fails: invoking a search tactic changes
    # which lemmas the proof term mentions.
    "same_statement_without_new_dependency",
    # Every same-statement rewrite, including those that lean on a new helper.
    # Overlaps structural change, so it is an upper bound rather than a category.
    "same_statement_any",
)


def proof_rewriting_section(
    before: GraphData,
    after: GraphData,
    before_decls: Mapping[str, Declaration],
    after_decls: Mapping[str, Declaration],
    before_text: Mapping[str, Mapping[str, str]] | None,
    after_text: Mapping[str, Mapping[str, str]] | None,
    *,
    edge_policy: str,
    new_declarations: set[str],
    scope: str = "identical_edge_set",
) -> dict[str, Any]:
    """Proof rewriting: a retained declaration whose statement did not change.

    Every same-statement rewrite is measured once and then tagged, so the three
    scopes in ``REWRITING_SCOPES`` are views over one row set rather than three
    separate passes.  The tags are what keeps the categories honest: a rewrite
    that started depending on a new declaration is already counted as structural
    change, and a rewrite whose dependency edge set is unchanged is the only kind
    the paragraph's strict definition admits.
    """
    if scope not in REWRITING_SCOPES:
        raise ValueError(f"unknown rewriting scope {scope!r}")
    before_names, after_names = set(before.nodes), set(after.nodes)
    retained = before_names & after_names
    before_dependencies = dependency_map(before.edge_set(edge_policy))
    after_dependencies = dependency_map(after.edge_set(edge_policy))

    rows: list[dict[str, Any]] = []
    statement_changed: list[str] = []
    unmeasurable: list[str] = []
    statement_basis: dict[str, int] = defaultdict(int)

    for name in sorted(retained):
        if str(after.nodes[name].get("kind")) not in DECLARATION_KINDS:
            continue
        before_declaration, after_declaration = before_decls.get(name), after_decls.get(name)
        if before_declaration is None or after_declaration is None:
            unmeasurable.append(name)
            continue
        if not (before_declaration.is_command and after_declaration.is_command):
            # Structure fields and constructors carry no rewritable proof.
            continue

        if (before_text is not None and after_text is not None
                and name in before_text and name in after_text):
            same_statement = before_text[name]["statement"] == after_text[name]["statement"]
            basis = "elaborated"
        else:
            same_statement = _normalized_text(before_declaration.statement) == _normalized_text(
                after_declaration.statement
            )
            basis = "source"
        statement_basis[basis] += 1
        if not same_statement:
            statement_changed.append(name)
            continue

        saved = _delta(before_declaration, after_declaration, proof_only=True)
        if saved is None:
            unmeasurable.append(name)
            continue
        after_dependency_set = after_dependencies.get(name, frozenset())
        row: dict[str, Any] = {
            "name": name,
            "kind": str(after.nodes[name].get("kind") or "unknown"),
            "module": before_declaration.module,
            "statement_basis": basis,
            "proof_tokens_before": before_declaration.proof_tokens,
            "proof_tokens_after": after_declaration.proof_tokens,
            "tokens_saved": saved,
            "declaration_tokens_saved": _delta(
                before_declaration, after_declaration, proof_only=False
            ),
            "proof_text_changed": _normalized_text(before_declaration.proof)
            != _normalized_text(after_declaration.proof),
            "edge_set_identical": (
                before_dependencies.get(name, frozenset()) == after_dependency_set
            ),
            "depends_on_new_declaration": bool(after_dependency_set & new_declarations),
            "new_dependencies": sorted(after_dependency_set & new_declarations),
        }
        for policy in AUTOMATION_POLICIES:
            row[f"automation_{policy}"] = {
                "before": before_declaration.automation[policy].as_dict(),
                "after": after_declaration.automation[policy].as_dict(),
                "increased": (
                    after_declaration.automation[policy].total
                    > before_declaration.automation[policy].total
                ),
            }
        rows.append(row)

    def select(name: str) -> list[dict[str, Any]]:
        if name == "identical_edge_set":
            return [row for row in rows if row["edge_set_identical"]]
        if name == "same_statement_without_new_dependency":
            return [row for row in rows if not row["depends_on_new_declaration"]]
        return list(rows)

    def bucket(selected: list[dict[str, Any]], policy: str | None = None) -> dict[str, Any]:
        summary = _delta_summary(selected)
        if policy is not None:
            summary["automation_invocations_before"] = sum(
                int(row[f"automation_{policy}"]["before"]["total"]) for row in selected
            )
            summary["automation_invocations_after"] = sum(
                int(row[f"automation_{policy}"]["after"]["total"]) for row in selected
            )
        summary["names"] = [str(row["name"]) for row in selected]
        return summary

    scopes: dict[str, Any] = {}
    for scope_name in REWRITING_SCOPES:
        selected = select(scope_name)
        entry: dict[str, Any] = {"totals": bucket(selected)}
        for policy in AUTOMATION_POLICIES:
            increased = [row for row in selected if row[f"automation_{policy}"]["increased"]]
            constant_or_fewer = [
                row for row in selected if not row[f"automation_{policy}"]["increased"]
            ]
            entry[f"automation_policy_{policy}"] = {
                "automation_rewriting": bucket(increased, policy),
                "explicit_rewriting": bucket(constant_or_fewer, policy),
            }
        scopes[scope_name] = entry

    headline = select(scope)
    result: dict[str, Any] = {
        "definition": (
            "retained declarations whose statement did not change; the headline "
            f"scope is {scope}"
        ),
        "edge_policy": edge_policy,
        "scope": scope,
        "candidates": _delta_summary(headline),
        "selected_names": [str(row["name"]) for row in headline],
        "excluded": {
            "statement_changed": len(statement_changed),
            "statement_changed_names": sorted(statement_changed),
            "dependency_edge_set_changed": sum(
                1 for row in rows if not row["edge_set_identical"]
            ),
            "depends_on_new_declaration": sum(
                1 for row in rows if row["depends_on_new_declaration"]
            ),
            "unmeasurable_proof_tokens": len(unmeasurable),
            "unmeasurable_names": sorted(unmeasurable),
        },
        "statement_comparison_basis": dict(statement_basis),
        "by_scope": scopes,
        "rows": rows,
    }
    # Kept for readers that want the headline split without walking ``by_scope``.
    for policy in AUTOMATION_POLICIES:
        result[f"automation_policy_{policy}"] = scopes[scope][f"automation_policy_{policy}"]
    return result


# --------------------------------------------------------------------------- #
# Why each removed declaration could be removed
# --------------------------------------------------------------------------- #

REMOVAL_GROUPS = (
    # Unreachable from every protected root before the edit: free to delete.
    "dead_code",
    # A retained user swapped it for a new declaration.  The deletion is the
    # other half of the structural change that introduced the replacement.
    "structural_removal",
    # A retained user dropped it without taking on any new declaration, so the
    # user's proof was rewritten against code that already existed.
    "rewrite_enabled_removal",
    # No surviving declaration ever dropped it and no removed ancestor explains
    # it either: a cycle of removals with nothing above them.
    "unattributed_removal",
)

# Which category each removal group belongs to.  A dropped dependency that then
# left the repository is the other half of whatever the dropping user did
# instead: taking on a new declaration is a structural change, and dropping it
# without taking on anything new is a proof rewrite.
REMOVAL_CATEGORY = {
    "dead_code": "dead_code",
    "structural_removal": "structural_change",
    "rewrite_enabled_removal": "proof_rewriting",
    "unattributed_removal": "unattributed",
}


def _range_key(node: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(node.get("module", "")), int(node["start_line"]), int(node["end_line"])


def written_declaration_groups(
    graph: GraphData, names: Iterable[str]
) -> list[tuple[str, frozenset[str]]]:
    """Group added nodes by the source range they occupy.

    Elaborating one command mints companions -- macro rules, notation
    delaborators, equation lemmas, recursors -- and each appears in the graph as
    its own node over the *same* source range.  ``notation "E2 " n => ...`` yields
    both ``termE2_`` and ``_aux_..._macroRules_..._1``: one command, two nodes.

    The range is what the author wrote, so each range with no surviving occupant
    counts as one new declaration and its nodes are that declaration's aliases.
    Counting nodes instead would inflate the total several-fold and crush the
    in-degree average; requiring a range to hold exactly one node would instead
    discard every notation and macro command the model wrote.

    This affects counts and in-degree only.  Token pricing is a union over source
    lines, so a range is charged once however many nodes sit on it -- including
    the 21 tokens of that ``notation`` command.
    """
    selected = set(names)
    occupants: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for name, node in graph.nodes.items():
        occupants[_range_key(node)].add(name)
    by_range: dict[tuple[str, int, int], set[str]] = defaultdict(set)
    for name in selected:
        by_range[_range_key(graph.nodes[name])].add(name)

    groups: list[tuple[str, frozenset[str]]] = []
    for key in sorted(by_range):
        members = by_range[key]
        if occupants[key] - members:
            # A declaration that survived the edit sits here too, so this range is
            # existing source that merely gained a companion.
            continue
        # The author's name is the plainest one: companions are decorations of it.
        representative = min(sorted(members), key=lambda name: (len(name), name))
        groups.append((representative, frozenset(members)))
    return groups


def written_declaration_names(graph: GraphData, names: Iterable[str]) -> set[str]:
    """Every alias of every newly written declaration."""
    groups = written_declaration_groups(graph, names)
    return set().union(*(members for _, members in groups)) if groups else set()


def removal_attribution(
    before: GraphData,
    after: GraphData,
    dead_removed: set[str],
    new_declarations: set[str],
    *,
    edge_policy: str,
    drivers: set[str] | None = None,
) -> dict[str, Any]:
    """Assign every removed declaration to the change that made it removable.

    For a retained user, the dependencies it no longer has are the candidates: if
    such a dependency also left the repository, its removal is attributable to
    whatever that user did instead.  Where the user took on a new declaration
    that is a structural change; where it did not, the user was rewritten against
    existing code.  Anything no surviving user ever dropped went because its own
    users went, which is a cascade rather than a decision.

    A declaration is assigned to exactly one group, in the order above, so the
    groups partition the removals and nothing is counted twice.
    """
    before_names, after_names = set(before.nodes), set(after.nodes)
    retained = before_names & after_names
    deleted = before_names - after_names
    before_dependencies = dependency_map(before.edge_set(edge_policy))
    after_dependencies = dependency_map(after.edge_set(edge_policy))

    # Only a theorem or definition can be said to have rewritten its proof, so
    # only those drive the attribution.  What they take down with them may be of
    # any kind: a private helper, an instance, an inductive and its constructors.
    candidates = retained if drivers is None else (retained & drivers)

    structural: set[str] = set()
    rewrite: set[str] = set()
    structural_drivers: set[str] = set()
    rewriting_drivers: set[str] = set()
    droppers: dict[str, set[str]] = defaultdict(set)
    dropper_groups: dict[str, set[str]] = defaultdict(set)
    user_rows: list[dict[str, Any]] = []
    for user in sorted(candidates):
        before_deps = before_dependencies.get(user, frozenset())
        after_deps = after_dependencies.get(user, frozenset())
        dropped = (before_deps - after_deps) & deleted
        gained_new = sorted(after_deps & new_declarations)
        if gained_new:
            structural_drivers.add(user)
        else:
            rewriting_drivers.add(user)
        if not dropped:
            continue
        for name in dropped:
            droppers[name].add(user)
        if gained_new:
            structural |= dropped
        else:
            rewrite |= dropped
        for name in dropped:
            dropper_groups[name].add("structural" if gained_new else "rewriting")
        user_rows.append({
            "name": user,
            "kind": str(after.nodes[user].get("kind") or "unknown"),
            "gained_new_declarations": gained_new,
            "dropped_removed_dependencies": sorted(dropped),
            "attribution": "structural_removal" if gained_new else "rewrite_enabled_removal",
        })

    # Precedence: dead code first, then structural, so a dependency dropped by
    # both a rewired user and a plainly rewritten one counts as structural once.
    direct: dict[str, set[str]] = {
        "dead_code": set(dead_removed),
    }
    direct["structural_removal"] = structural - direct["dead_code"]
    direct["rewrite_enabled_removal"] = (
        rewrite - direct["dead_code"] - direct["structural_removal"]
    )
    direct["unattributed_removal"] = deleted - set().union(*direct.values())

    # A declaration that no survivor ever dropped went because its own users
    # went.  Inherit the attribution of the nearest removed ancestor by walking
    # down the dependency edges of already-attributed removals, highest
    # precedence first, so the reason is the one that started the cascade.
    groups = {label: set(names) for label, names in direct.items()}
    unattributed = set(groups["unattributed_removal"])
    groups["unattributed_removal"] = set()
    propagated: dict[str, int] = {label: 0 for label in REMOVAL_GROUPS}
    dependencies_of = {
        name: before_dependencies.get(name, frozenset()) for name in deleted
    }
    for label in ("dead_code", "structural_removal", "rewrite_enabled_removal"):
        pending = [name for name in groups[label]]
        while pending:
            for dependency in dependencies_of.get(pending.pop(), ()):
                if dependency in unattributed:
                    unattributed.discard(dependency)
                    groups[label].add(dependency)
                    propagated[label] += 1
                    pending.append(dependency)
    groups["unattributed_removal"] = unattributed

    return {
        "definition": (
            "each removed declaration assigned to the change that made it "
            "removable, in the order " + ", ".join(REMOVAL_GROUPS)
        ),
        "edge_policy": edge_policy,
        "drivers": len(candidates),
        "drivers_with_a_new_dependency": sorted(structural_drivers),
        "drivers_without_a_new_dependency": sorted(rewriting_drivers),
        "groups": {label: sorted(groups[label]) for label in REMOVAL_GROUPS},
        "counts": {label: len(groups[label]) for label in REMOVAL_GROUPS},
        "category_of_group": dict(REMOVAL_CATEGORY),
        "counts_attributed_directly": {
            label: len(direct[label]) for label in REMOVAL_GROUPS
        },
        "counts_inherited_from_a_removed_ancestor": dict(propagated),
        "counts_by_kind": {
            label: _kind_counts(groups[label], before.nodes) for label in REMOVAL_GROUPS
        },
        "removed_declarations": len(deleted),
        "removed_dependencies_with_multiple_droppers": sum(
            1 for name, users in droppers.items() if len(users) > 1
        ),
        # The one place the attribution has to choose: a dependency dropped both by
        # a driver that took on a new declaration and by one that did not.  It goes
        # to structural change by precedence, and is reported so the size of the
        # choice is visible rather than assumed away.
        "contested_removals": sorted(
            name for name, labels in dropper_groups.items() if len(labels) > 1
        ),
        "rewired_user_rows": user_rows,
    }


def new_declaration_degree_section(
    after: GraphData,
    before: GraphData,
    new_declarations: set[str],
    *,
    edge_policy: str,
    after_source: SourceIndex | None = None,
) -> dict[str, Any]:
    """How much each new declaration is actually used.

    In-degree is the number of declarations that depend on it, counted on the
    final graph.  A helper introduced once and used once has paid for itself only
    if it was longer than what it replaced; one used many times is the shape of a
    genuine abstraction, so the distribution matters more than the count.
    """
    after_names, before_names = set(after.nodes), set(before.nodes)
    retained = before_names & after_names
    users = user_map(after.edge_set(edge_policy))

    groups = written_declaration_groups(after, new_declarations)

    def summarize(entries: Sequence[tuple[str, frozenset[str]]]) -> dict[str, Any]:
        """In-degree per written declaration, counting users of any of its aliases."""
        names = [representative for representative, _ in entries]
        users_of: dict[str, set[str]] = {}
        for representative, members in entries:
            found: set[str] = set()
            for member in members:
                found |= users.get(member, set())
            users_of[representative] = found - members
        degrees = sorted(len(users_of[name]) for name in names)
        retained_degrees = [len(users_of[name] & retained) for name in names]
        new_degrees = [len(users_of[name] & new_declarations) for name in names]
        total = sum(degrees)
        return {
            "declarations": len(names),
            "total_in_degree": total,
            "mean_in_degree": round(total / len(names), 6) if names else None,
            "median_in_degree": (
                degrees[len(degrees) // 2] if degrees else None
            ),
            "max_in_degree": degrees[-1] if degrees else None,
            "unused": sum(1 for degree in degrees if degree == 0),
            "used_once": sum(1 for degree in degrees if degree == 1),
            "used_at_least_twice": sum(1 for degree in degrees if degree >= 2),
            "mean_retained_in_degree": (
                round(sum(retained_degrees) / len(names), 6) if names else None
            ),
            "mean_new_declaration_in_degree": (
                round(sum(new_degrees) / len(names), 6) if names else None
            ),
        }

    node_entries = [(name, frozenset({name})) for name in sorted(new_declarations)]
    by_kind: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for entry in node_entries:
        by_kind[str(after.nodes[entry[0]].get("kind") or "unknown")].append(entry)
    written_by_kind: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    for entry in groups:
        written_by_kind[str(after.nodes[entry[0]].get("kind") or "unknown")].append(entry)

    by_command: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    by_class: dict[str, list[tuple[str, frozenset[str]]]] = defaultdict(list)
    if after_source is not None:
        for entry in groups:
            keyword = leading_command(after_source.slice(after.nodes[entry[0]]) or "")
            by_command[keyword or "unknown"].append(entry)
            by_class[command_class(keyword)].append(entry)
    return {
        "definition": "in-degree of each new declaration in the final graph",
        "edge_policy": edge_policy,
        "note": (
            "Quote ``term_level``: one entry per source range the model wrote with a "
            "command that introduces a constant, counting users of any of its "
            "aliases.  ``syntax_level`` entries are used syntactically and so always "
            "read as in-degree zero, ``other_command`` is a project's own DSL, and "
            "``nodes`` counts every graph node, so one notation command appears two "
            "or three times there."
        ),
        "written": summarize(groups),
        "nodes": summarize(node_entries),
        "term_level": summarize(by_class.get("term_level", [])),
        "syntax_level": summarize(by_class.get("syntax_level", [])),
        "other_command": summarize(by_class.get("other_command", [])),
        "by_command": {
            keyword: summarize(entries) for keyword, entries in sorted(by_command.items())
        },
        "written_by_kind": {
            kind: summarize(entries) for kind, entries in sorted(written_by_kind.items())
        },
        "by_kind": {kind: summarize(entries) for kind, entries in sorted(by_kind.items())},
        "rows": [
            {
                "name": representative,
                "aliases": sorted(members - {representative}),
                "kind": str(after.nodes[representative].get("kind") or "unknown"),
                "in_degree": len(
                    set().union(*(users.get(member, set()) for member in members)) - members
                ),
            }
            for representative, members in groups
        ],
    }

# --------------------------------------------------------------------------- #
# Exact line-level token accounting
# --------------------------------------------------------------------------- #
#
# Merged per-node ranges are the wrong instrument for a closed budget: where one
# declaration nests inside another, or a deleted declaration sits line-adjacent
# to a retained one, two subsets of nodes can claim the same line.  Pricing each
# *line* once and then assigning it to a class keeps the parts summing to the
# whole, which is what lets the categories be reconciled against the scored
# saving instead of merely compared with it.


def line_owner_sets(
    graph: GraphData, source: SourceIndex, names: Iterable[str]
) -> dict[str, set[int]]:
    """Map each source path to the line numbers covered by the given nodes."""
    covered: dict[str, set[int]] = defaultdict(set)
    for name in names:
        node = graph.nodes[name]
        path = source.path_of_module(str(node.get("module", "")))
        if path is None:
            continue
        start, end = int(node["start_line"]), int(node["end_line"])
        covered[path].update(range(start, end + 1))
    return covered


def tokens_of_lines(source: SourceIndex, lines_by_path: Mapping[str, set[int]]) -> int:
    total = 0
    for path, lines in lines_by_path.items():
        per_line = source.line_tokens(path)
        if per_line is None:
            continue
        total += sum(per_line[line - 1] for line in lines if 1 <= line <= len(per_line))
    return total


def _subtract(
    left: Mapping[str, set[int]], *rights: Mapping[str, set[int]]
) -> dict[str, set[int]]:
    result: dict[str, set[int]] = {}
    for path, lines in left.items():
        remaining = set(lines)
        for right in rights:
            remaining -= right.get(path, set())
        if remaining:
            result[path] = remaining
    return result


def text_accounting_section(
    before: GraphData,
    after: GraphData,
    before_source: SourceIndex,
    after_source: SourceIndex,
    removal_groups: Sequence[tuple[str, set[str]]],
    retained_groups: Sequence[tuple[str, set[str]]],
    added_groups: Sequence[tuple[str, set[str]]],
) -> dict[str, Any]:
    """Partition every scored token on both sides into disjoint classes.

    A line covered by both a retained and a deleted declaration is credited to
    the retained one, so nesting can never manufacture a removal.  Text that
    belongs to no declaration -- ``namespace``, ``section``, ``variable``,
    ``open``, ``attribute`` lines and the files that vanished with them -- is its
    own class, and on a repository that the model rewrites around a library
    import it is a category in its own right rather than a rounding error.
    """
    before_names, after_names = set(before.nodes), set(after.nodes)
    retained = before_names & after_names
    added = after_names - before_names

    retained_before = line_owner_sets(before, before_source, retained)

    # Price the removal groups in order, each one only for the lines no earlier
    # claim already took.  Retained declarations are claimed first, so a line that
    # survives can never be billed as a removal.
    claimed: list[Mapping[str, set[int]]] = [retained_before]
    removal_lines: dict[str, dict[str, set[int]]] = {}
    removal_tokens: dict[str, int] = {}
    for label, names in removal_groups:
        lines = _subtract(line_owner_sets(before, before_source, names), *claimed)
        removal_lines[label] = lines
        removal_tokens[label] = tokens_of_lines(before_source, lines)
        claimed.append(lines)

    # The same partition refined by declaration kind, so the kind totals add up to
    # the group totals instead of overlapping where kinds share a line.
    kind_claimed: list[Mapping[str, set[int]]] = [retained_before]
    removal_tokens_by_kind: dict[str, dict[str, int]] = {}
    for label, names in removal_groups:
        per_kind: dict[str, int] = {}
        by_kind: dict[str, set[str]] = defaultdict(set)
        for name in names:
            by_kind[str(before.nodes[name].get("kind") or "unknown")].add(name)
        for kind in sorted(by_kind):
            lines = _subtract(line_owner_sets(before, before_source, by_kind[kind]), *kind_claimed)
            per_kind[kind] = tokens_of_lines(before_source, lines)
            kind_claimed.append(lines)
        removal_tokens_by_kind[label] = per_kind

    retained_after = line_owner_sets(after, after_source, retained)
    added_lines = _subtract(line_owner_sets(after, after_source, added), retained_after)

    added_claimed: list[Mapping[str, set[int]]] = [retained_after]
    added_group_tokens: dict[str, int] = {}
    for label, names in added_groups:
        lines = _subtract(line_owner_sets(after, after_source, names), *added_claimed)
        added_group_tokens[label] = tokens_of_lines(after_source, lines)
        added_claimed.append(lines)

    before_declaration_lines = {path: set(lines) for path, lines in retained_before.items()}
    for mapping in removal_lines.values():
        for path, lines in mapping.items():
            before_declaration_lines.setdefault(path, set()).update(lines)
    after_declaration_lines = {
        path: set(lines) | added_lines.get(path, set())
        for path, lines in retained_after.items()
    }
    for path, lines in added_lines.items():
        after_declaration_lines.setdefault(path, set()).update(lines)

    added_tokens_by_kind: dict[str, int] = {}
    added_by_kind: dict[str, set[str]] = defaultdict(set)
    for name in added:
        added_by_kind[str(after.nodes[name].get("kind") or "unknown")].add(name)
    added_claimed: list[Mapping[str, set[int]]] = [retained_after]
    for kind in sorted(added_by_kind):
        lines = _subtract(line_owner_sets(after, after_source, added_by_kind[kind]), *added_claimed)
        added_tokens_by_kind[kind] = tokens_of_lines(after_source, lines)
        added_claimed.append(lines)

    def non_declaration(source: SourceIndex, declaration: Mapping[str, set[int]]) -> int:
        total = 0
        for path in source.paths():
            per_line = source.line_tokens(path)
            if per_line is None:
                continue
            owned = declaration.get(path, set())
            total += sum(
                count for index, count in enumerate(per_line, start=1) if index not in owned
            )
        return total

    # Split the retained drop the same way, by line union rather than by summing
    # per-declaration slices: declarations that share lines -- a structure and its
    # generated projections, say -- would otherwise be charged more than once, and
    # the double charge does not cancel between the two sides.
    claimed_retained: set[str] = set()
    resolved_groups: dict[str, set[str]] = {}
    for label, names in retained_groups:
        group = (retained & names) - claimed_retained
        resolved_groups[label] = group
        claimed_retained |= group
    resolved_groups["other_retained"] = retained - claimed_retained
    # Price the groups with precedence on both sides, as the removals are priced.
    # Two retained declarations can share a line -- a structure and its generated
    # projections land in different groups -- and pricing each group's union
    # independently would bill that line twice, which does not cancel between the
    # sides and would leave the categories summing past the saving.
    retained_attribution: dict[str, dict[str, int]] = {}
    claimed_before: list[Mapping[str, set[int]]] = []
    claimed_after: list[Mapping[str, set[int]]] = []
    for label, group in resolved_groups.items():
        before_lines = _subtract(
            line_owner_sets(before, before_source, group), *claimed_before
        ) if claimed_before else line_owner_sets(before, before_source, group)
        after_lines = _subtract(
            line_owner_sets(after, after_source, group), *claimed_after
        ) if claimed_after else line_owner_sets(after, after_source, group)
        claimed_before.append(before_lines)
        claimed_after.append(after_lines)
        first = tokens_of_lines(before_source, before_lines)
        last = tokens_of_lines(after_source, after_lines)
        # The group is a denominator: it holds every retained declaration of its
        # kind, most of which the model never touched.  Count the ones whose source
        # actually changed size, so a rate can be quoted against something real.
        changed = 0
        for name in group:
            own_before = tokens_of_lines(
                before_source, line_owner_sets(before, before_source, {name})
            )
            own_after = tokens_of_lines(
                after_source, line_owner_sets(after, after_source, {name})
            )
            if own_before != own_after:
                changed += 1
        retained_attribution[label] = {
            "declarations": len(group),
            "declarations_with_a_token_change": changed,
            "tokens_before": first,
            "tokens_after": last,
            "tokens_saved": first - last,
        }

    before_total, after_total = before_source.total_tokens(), after_source.total_tokens()
    retained_before_tokens = tokens_of_lines(before_source, retained_before)
    retained_after_tokens = tokens_of_lines(after_source, retained_after)
    removed_tokens_total = sum(removal_tokens.values())
    added_tokens = tokens_of_lines(after_source, added_lines)
    non_declaration_before = non_declaration(before_source, before_declaration_lines)
    non_declaration_after = non_declaration(after_source, after_declaration_lines)

    before_paths, after_paths = set(before_source.paths()), set(after_source.paths())
    removed_paths = sorted(before_paths - after_paths)
    added_paths = sorted(after_paths - before_paths)
    # A cross-cut, not a class: these tokens are already inside the dead, live and
    # non-declaration classes.  Reported because a repository that loses most of
    # its files has been restructured wholesale rather than edited declaration by
    # declaration, and that is the shape of a library substitution.
    removed_file_tokens = sum(
        sum(before_source.line_tokens(path) or []) for path in removed_paths
    )
    added_file_tokens = sum(
        sum(after_source.line_tokens(path) or []) for path in added_paths
    )

    return {
        "definition": (
            "every scored token on each side assigned to exactly one class; a line "
            "shared by a retained and a deleted declaration is credited to the retained one"
        ),
        "before_total_tokens": before_total,
        "after_total_tokens": after_total,
        "total_tokens_saved": before_total - after_total,
        "retained_declaration_tokens_before": retained_before_tokens,
        "retained_declaration_tokens_after": retained_after_tokens,
        "retained_declaration_tokens_saved": retained_before_tokens - retained_after_tokens,
        "removed_declaration_tokens": removal_tokens,
        "removed_declaration_tokens_by_kind": removal_tokens_by_kind,
        "removed_declaration_tokens_total": removed_tokens_total,
        "added_declaration_tokens_by_kind": added_tokens_by_kind,
        "added_declaration_tokens_by_group": added_group_tokens,
        "retained_attribution": retained_attribution,
        "retained_attribution_note": (
            "The groups partition the retained declarations in the order given, so "
            "their line-union savings partition the retained saving."
        ),
        "added_declaration_tokens": added_tokens,
        "non_declaration_tokens_before": non_declaration_before,
        "non_declaration_tokens_after": non_declaration_after,
        "non_declaration_tokens_saved": non_declaration_before - non_declaration_after,
        "files_before": len(before_paths),
        "files_after": len(after_paths),
        "files_removed": len(removed_paths),
        "files_added": len(added_paths),
        "tokens_in_removed_files": removed_file_tokens,
        "tokens_in_added_files": added_file_tokens,
        "removed_file_share_of_saving_pct": (
            round(100.0 * removed_file_tokens / (before_total - after_total), 6)
            if before_total != after_total else None
        ),
        "files_removed_names": removed_paths,
        "files_added_names": added_paths,
        "partition_check": {
            "before_classified": (
                retained_before_tokens + removed_tokens_total + non_declaration_before
            ),
            "before_total": before_total,
            "after_classified": retained_after_tokens + added_tokens + non_declaration_after,
            "after_total": after_total,
        },
    }


# --------------------------------------------------------------------------- #
# Library substitution
# --------------------------------------------------------------------------- #


def import_delta_section(
    before_source: SourceIndex,
    after_source: SourceIndex,
    before: GraphData,
    after: GraphData,
    live_removed: set[str],
) -> dict[str, Any]:
    """Which libraries the edit reached for, and what vanished alongside.

    A model can retire a large part of a repository by importing a library that
    already proves what the repository proved by hand.  The import diff is the
    visible half of that move; the tokens deleted from the modules that gained a
    library import are the other half.  The pairing is evidence of substitution,
    not proof of it -- a module can gain an import for one reason and lose
    declarations for another.
    """
    before_imports = before_source.imports_by_path()
    after_imports = after_source.imports_by_path()
    shared = set(before_imports) & set(after_imports)

    added_by_path: dict[str, list[str]] = {}
    removed_by_path: dict[str, list[str]] = {}
    for path in sorted(shared):
        gained = sorted(after_imports[path] - before_imports[path])
        lost = sorted(before_imports[path] - after_imports[path])
        if gained:
            added_by_path[path] = gained
        if lost:
            removed_by_path[path] = lost

    def roots(names: Iterable[str]) -> dict[str, int]:
        counts: dict[str, int] = defaultdict(int)
        for name in names:
            counts[name.split(".", 1)[0]] += 1
        return dict(sorted(counts.items()))

    all_added = [name for names in added_by_path.values() for name in names]
    all_removed = [name for names in removed_by_path.values() for name in names]
    gained_library_paths = {
        path for path, names in added_by_path.items()
        if any(name.split(".", 1)[0] in LIBRARY_ROOTS for name in names)
    }

    # Price the deletions that sit in the modules that reached for a library.
    live_lines = line_owner_sets(before, before_source, live_removed)
    in_gained = {path: lines for path, lines in live_lines.items() if path in gained_library_paths}
    removed_paths = sorted(set(before_source.paths()) - set(after_source.paths()))
    gained_anywhere = bool(
        any(name.split(".", 1)[0] in LIBRARY_ROOTS for name in all_added)
    )
    return {
        "imports_added": len(all_added),
        "imports_removed": len(all_removed),
        "imports_added_by_root": roots(all_added),
        "imports_removed_by_root": roots(all_removed),
        "modules_that_gained_a_library_import": len(gained_library_paths),
        "library_roots_counted": sorted(LIBRARY_ROOTS),
        "live_declaration_tokens_removed_from_modules_that_gained_a_library_import":
            tokens_of_lines(before_source, in_gained),
        # A module deleted outright cannot be a module that gained an import, so
        # the per-module pairing misses the most drastic form of substitution:
        # the repository folds into the library and its own files disappear.
        "repository_gained_a_library_import": gained_anywhere,
        "files_removed_while_a_library_import_was_gained": len(removed_paths) if gained_anywhere else 0,
        "tokens_in_files_removed_while_a_library_import_was_gained": (
            sum(sum(before_source.line_tokens(path) or []) for path in removed_paths)
            if gained_anywhere else 0
        ),
        "imports_added_by_module": added_by_path,
        "imports_removed_by_module": removed_by_path,
        "attribution_note": (
            "An import gained and declarations lost in the same module is evidence "
            "of library substitution, not proof of it."
        ),
    }

# --------------------------------------------------------------------------- #
# Per-repository orchestration
# --------------------------------------------------------------------------- #


def _first_existing(*candidates: Path | None) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    return None


def _source_index(archive: Path | None, directory: Path | None, label: str) -> SourceIndex:
    if archive is not None:
        return SourceIndex.from_archive(archive)
    if directory is not None and directory.is_dir():
        return SourceIndex.from_directory(directory)
    raise FileNotFoundError(f"{label}: no Lean sources found")


def diff_cross_check(path: Path, added: set[str], deleted: set[str], rewritten: set[str]) -> dict[str, Any]:
    """Compare graph-derived declaration sets against an independent text diff."""
    payload = read_json(path)
    sections = ("theorems", "definitions")
    if not all(section in payload for section in sections):
        raise ValueError(f"{path}: missing theorem/definition sections")

    def collect(field_name: str) -> set[str]:
        values: set[str] = set()
        for section in sections:
            entries = payload[section].get(field_name, [])
            if not isinstance(entries, list):
                raise ValueError(f"{path}: {section}.{field_name} must be an array")
            values |= {str(entry) for entry in entries}
        return values

    diff_added = collect("added")
    diff_deleted = collect("deleted")
    diff_rewritten = collect("proof_changed_with_same_statement")
    return {
        "source": str(path),
        "added_only_in_text_diff": sorted(diff_added - added),
        "added_only_in_graph": sorted(added - diff_added),
        "deleted_only_in_text_diff": sorted(diff_deleted - deleted),
        "deleted_only_in_graph": sorted(deleted - diff_deleted),
        "rewritten_only_in_text_diff": sorted(diff_rewritten - rewritten),
        "rewritten_only_in_graph": sorted(rewritten - diff_rewritten),
        "agrees": (diff_added == added and diff_deleted == deleted),
    }


def classify_repo(
    repo: str,
    options: argparse.Namespace,
    protected: Sequence[str],
    scope: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    repo_dir = options.graph_root / repo
    before_graph = load_graph(repo_dir / "before.json")
    after_graph = load_graph(repo_dir / "after.json")

    absent_before = sorted(set(protected) - before_graph.nodes.keys())
    absent_after = sorted(set(protected) - after_graph.nodes.keys())
    if absent_before or absent_after:
        # A graph missing its protected roots did not capture the repository,
        # however complete its summary claims to be.  Refusing it here keeps a
        # broken extraction out of the benchmark numbers.
        raise BrokenGraphError(
            f"{repo}: protected roots absent from the graphs "
            f"(before {len(absent_before)}/{len(protected)} missing, "
            f"after {len(absent_after)}/{len(protected)} missing); "
            f"before={absent_before[:5]}, after={absent_after[:5]}"
        )

    before_source = _source_index(
        _first_existing(repo_dir / "before" / "original-sources.tar.gz"),
        (options.baseline_root / repo / "stripped") if options.baseline_root else None,
        f"{repo}: original sources",
    )
    after_source = _source_index(
        _first_existing(
            repo_dir / "after" / "original-sources.tar.gz",
            repo_dir / "retained-capture" / "submitted.sources.tar.gz",
            (options.terminal_root / repo / "playback" / "capture_001" / "submitted.sources.tar.gz")
            if options.terminal_root else None,
        ),
        None,
        f"{repo}: final sources",
    )

    excluded_files: list[str] = list(options.exclude_file)
    challenge = None
    if options.comparator_root is not None:
        challenge = challenge_source_path(options.comparator_root, repo)
        if challenge is not None:
            excluded_files.append(challenge)
    scope = scope or {}
    for index in (before_source, after_source):
        index.restrict(
            exclude_files=excluded_files,
            exclude_dirs=scope.get("exclude_dirs") or [],
            include_prefix=str(scope.get("include_prefix") or ""),
        )

    before_text = after_text = None
    if options.declaration_text_root is not None:
        text_dir = options.declaration_text_root / repo
        before_text = load_declaration_text(text_dir / "before-declarations.json", before_graph.sha256)
        after_text = load_declaration_text(text_dir / "after-declarations.json", after_graph.sha256)

    before_names, after_names = set(before_graph.nodes), set(after_graph.nodes)
    retained = before_names & after_names
    added = after_names - before_names
    deleted = before_names - after_names

    # Source is only read for declarations whose text can matter, which keeps the
    # cost proportional to the edit rather than to the repository.
    def measurable(names: set[str], nodes: Mapping[str, Mapping[str, Any]]) -> set[str]:
        return {name for name in names if str(nodes[name].get("kind")) in DECLARATION_KINDS}

    before_wanted = measurable(retained, before_graph.nodes)
    after_wanted = measurable(retained | added, after_graph.nodes)
    before_decls = {
        name: read_declaration(before_graph.nodes[name], before_source,
                               extra_tactics=options.extra_automation_tactic)
        for name in sorted(before_wanted)
    }
    after_decls = {
        name: read_declaration(after_graph.nodes[name], after_source,
                               extra_tactics=options.extra_automation_tactic)
        for name in sorted(after_wanted)
    }

    # A graph that cannot be located in the source prices its declarations at
    # zero and silently pushes their tokens into non-declaration text, which
    # looks like scaffolding removal.  Refuse the repository instead.
    located_paths = {
        before_source.path_of_module(str(node.get("module", "")))
        for node in before_graph.nodes.values()
    }
    located_paths.discard(None)
    covered = sum(
        sum(before_source.line_tokens(path) or [])
        for path in before_source.paths() if path in located_paths
    )
    before_total = before_source.total_tokens()
    coverage = covered / before_total if before_total else 1.0
    if coverage < options.minimum_graph_coverage:
        raise BrokenGraphError(
            f"{repo}: graph covers only {coverage:.1%} of the scored source "
            f"({covered} of {before_total} tokens); unresolved modules: "
            f"{sorted(before_source.unresolved_modules)[:5]}"
        )

    dead = dead_code_section(before_graph, after_graph, protected, before_source)
    structural = structural_section(
        before_graph, after_graph, before_decls, after_decls, after_source,
        edge_policy=options.edge_identity,
    )
    new_declarations = {
        name for name in added
        if str(after_graph.nodes[name].get("kind")) in DECLARATION_KINDS
    }
    rewriting = {
        policy: proof_rewriting_section(
            before_graph, after_graph, before_decls, after_decls, before_text, after_text,
            edge_policy=policy, new_declarations=new_declarations, scope=options.rewriting_scope,
        )
        for policy in EDGE_POLICIES
    }
    headline_rewriting = rewriting[options.edge_identity]

    # Everything else: retained declarations that changed but belong to no
    # category, so the reconciliation residual has a name instead of a gap.
    categorized = set(headline_rewriting["selected_names"])
    structural_dependents = {str(row["name"]) for row in structural["dependent_rows"]}
    other_rows = []
    for name in sorted(measurable(retained, after_graph.nodes) - categorized - structural_dependents):
        delta = _delta(before_decls.get(name), after_decls.get(name), proof_only=False)
        if delta:
            other_rows.append({"name": name, "tokens_saved": delta})
    other = _delta_summary(other_rows)
    other["rows"] = other_rows

    dead_removed = set(dead["removed_names"])
    live_removed = set(dead["deleted_but_protected_reachable_names"])
    # A companion minted over source that survived the edit is not the model
    # reaching for a new helper, so the structural test ignores those.  Aliases of
    # a genuinely new command all count: an edge into any of them is an edge into
    # the declaration the model wrote.
    distinct_new = written_declaration_names(after_graph, new_declarations)
    # A new dependency means a newly written theorem or definition.  Notation,
    # macros and a project's own commands are additions too, but depending on one
    # is not the model factoring a proof through a lemma, so they are priced
    # separately instead of being read as structural change.
    term_new: set[str] = set()
    for representative, members in written_declaration_groups(after_graph, new_declarations):
        keyword = leading_command(after_source.slice(after_graph.nodes[representative]) or "")
        if command_class(keyword) == "term_level":
            term_new |= members
    attribution_new = new_declarations if options.count_co_located_additions else term_new

    # Only term-level theorems and definitions drive the attribution.
    drivers = {
        name for name, declaration in before_decls.items()
        if declaration.located
        and command_class(leading_command(declaration.source)) == "term_level"
    }
    removal = removal_attribution(
        before_graph, after_graph, dead_removed, attribution_new,
        edge_policy=options.edge_identity, drivers=drivers,
    )
    degrees = new_declaration_degree_section(
        after_graph, before_graph, new_declarations, edge_policy=options.edge_identity,
        after_source=after_source,
    )
    removal_groups = [(label, set(removal["groups"][label])) for label in REMOVAL_GROUPS]
    text = text_accounting_section(
        before_graph, after_graph, before_source, after_source, removal_groups,
        retained_groups=[
            ("structural_drivers", set(removal["drivers_with_a_new_dependency"])),
            ("rewriting_drivers", set(removal["drivers_without_a_new_dependency"])),
        ],
        added_groups=[
            ("term_level_declarations", set(term_new)),
            ("other_additions", added - term_new),
        ],
    )
    contested = set(removal["contested_removals"])
    contested_tokens = tokens_of_lines(
        before_source,
        _subtract(
            line_owner_sets(before_graph, before_source, contested),
            line_owner_sets(before_graph, before_source, retained),
        ),
    )
    imports = import_delta_section(
        before_source, after_source, before_graph, after_graph, live_removed
    )

    official = None
    if options.playback_root is not None:
        playback = options.playback_root / options.playback_template.format(repository=repo)
        official = official_totals(playback)

    elif options.use_source_token_totals:
        baseline = int(text["before_total_tokens"])
        post = int(text["after_total_tokens"])
        saved = baseline - post
        official = {
            "baseline_lean_tokens": baseline,
            "post_lean_tokens": post,
            "lean_tokens_saved": saved,
            "lean_token_compression_pct": 100.0 * saved / baseline if baseline else 0.0,
            "build_passed": True,
        }
    # Every figure below is taken from the line partition, so the classes sum to
    # the repository's token saving instead of merely resembling it.
    proof_tokens = int(headline_rewriting["candidates"]["tokens_saved"])
    removed = text["removed_declaration_tokens"]
    dead_tokens = int(removed["dead_code"])
    structural_removal_tokens = int(removed["structural_removal"])
    rewrite_removal_tokens = int(removed["rewrite_enabled_removal"])
    unattributed_removal_tokens = int(removed["unattributed_removal"])
    live_removal_tokens = (
        structural_removal_tokens + rewrite_removal_tokens + unattributed_removal_tokens
    )
    non_declaration_tokens = int(text["non_declaration_tokens_saved"])
    added_declaration_tokens = int(text["added_declaration_tokens"])
    added_by_group = text["added_declaration_tokens_by_group"]
    new_term_cost = int(added_by_group["term_level_declarations"])
    other_addition_cost = int(added_by_group["other_additions"])
    retained_drop = int(text["retained_declaration_tokens_saved"])
    attribution = text["retained_attribution"]
    # Structural change: the drivers that took on a new declaration, the removals
    # that gave way to it, less what writing it cost.
    dependent_tokens = int(attribution["structural_drivers"]["tokens_saved"])
    structural_tokens = dependent_tokens + structural_removal_tokens - new_term_cost
    # Proof rewriting: the drivers that took on nothing new, plus whatever their
    # rewrites made deletable.
    rewriting_declaration_tokens = int(attribution["rewriting_drivers"]["tokens_saved"])
    other_tokens = int(attribution["other_retained"]["tokens_saved"])
    explained = dead_tokens + structural_tokens + rewriting_declaration_tokens + rewrite_removal_tokens
    reconciliation: dict[str, Any] = {
        "dead_code_tokens": dead_tokens,
        "structural_change_tokens": structural_tokens,
        "structural_tokens_added_by_new_declarations": new_term_cost,
        "structural_tokens_removed_from_dependents": dependent_tokens,
        "structural_drivers": int(attribution["structural_drivers"]["declarations"]),
        "rewriting_drivers": int(attribution["rewriting_drivers"]["declarations"]),
        "structural_drivers_changed": int(
            attribution["structural_drivers"]["declarations_with_a_token_change"]
        ),
        "rewriting_drivers_changed": int(
            attribution["rewriting_drivers"]["declarations_with_a_token_change"]
        ),
        "other_addition_cost_tokens": other_addition_cost,
        "contested_removals": len(contested),
        "contested_removal_tokens": contested_tokens,
        "all_added_declaration_tokens": added_declaration_tokens,
        "tokens_removed_from_declarations_that_gained_any_edge": int(
            structural["dependents_that_gained_any_edge"]["tokens_saved"]
        ),
        # The paragraph's measure: proof tokens only, summed per declaration.
        "proof_rewriting_proof_tokens": proof_tokens,
        # The budget's measure: whole declarations priced by line union, plus the
        # dependencies that the rewrites made deletable.
        "proof_rewriting_tokens": rewriting_declaration_tokens + rewrite_removal_tokens,
        "proof_rewriting_retained_tokens": rewriting_declaration_tokens,
        "removed_live_declaration_tokens": live_removal_tokens,
        "structural_removal_tokens": structural_removal_tokens,
        "rewrite_enabled_removal_tokens": rewrite_removal_tokens,
        "unattributed_removal_tokens": unattributed_removal_tokens,
        "non_declaration_text_tokens": non_declaration_tokens,
        "library_substitution_tokens": int(
            imports["live_declaration_tokens_removed_from_modules_that_gained_a_library_import"]
        ),
        "tokens_in_removed_files": int(text["tokens_in_removed_files"]),
        "tokens_in_files_removed_with_a_library_import": int(
            imports["tokens_in_files_removed_while_a_library_import_was_gained"]
        ),
        "uncategorized_retained_change_tokens": other_tokens,
        "categorized_tokens": explained,
        "retained_declaration_tokens_saved": retained_drop,
        # What the partition says the retained declarations gave up, minus what
        # the per-declaration categories claimed of it.  Non-zero means some
        # retained saving is real but unattributed, and it is shown rather than
        # absorbed.
        "retained_attribution_slack_tokens": (
            retained_drop - dependent_tokens - rewriting_declaration_tokens - other_tokens
        ),
        "addition_cost_slack_tokens": (
            added_declaration_tokens - new_term_cost - other_addition_cost
        ),
        "partition_total_tokens_saved": int(text["total_tokens_saved"]),
        "accounted_including_live_removal_tokens": (
            dead_tokens + live_removal_tokens + non_declaration_tokens
            + retained_drop - added_declaration_tokens
        ),
    }
    if official is not None:
        saved = int(official["lean_tokens_saved"])
        reconciliation |= {
            "scored_baseline_tokens": int(official["baseline_lean_tokens"]),
            "scored_post_tokens": int(official["post_lean_tokens"]),
            "scored_tokens_saved": saved,
            "residual_tokens": saved - explained,
            "residual_after_live_removal_tokens": (
                saved - dead_tokens - live_removal_tokens - non_declaration_tokens
                - retained_drop + added_declaration_tokens
            ),
            # The category shares, which partition the scored saving exactly.
            "other_addition_share_of_scored_saving_pct": (
                round(-100.0 * other_addition_cost / saved, 6) if saved else None
            ),
            "scored_minus_partition_tokens": saved - int(text["total_tokens_saved"]),
            "removed_live_share_of_scored_saving_pct": (
                round(100.0 * live_removal_tokens / saved, 6) if saved else None
            ),
            "categorized_share_of_scored_saving_pct": (
                round(100.0 * explained / saved, 6) if saved else None
            ),
            "dead_code_share_of_scored_saving_pct": (
                round(100.0 * int(dead["contribution_tokens"]) / saved, 6) if saved else None
            ),
            "structural_share_of_scored_saving_pct": (
                round(100.0 * int(structural["contribution_tokens"]) / saved, 6) if saved else None
            ),
            "proof_rewriting_share_of_scored_saving_pct": (
                round(100.0 * (rewriting_declaration_tokens + rewrite_removal_tokens) / saved, 6)
                if saved else None
            ),
            "non_declaration_share_of_scored_saving_pct": (
                round(100.0 * non_declaration_tokens / saved, 6) if saved else None
            ),
            "compression_pct": official["lean_token_compression_pct"],
            "build_passed": official["build_passed"],
        }

    row: dict[str, Any] = {
        "repository": repo,
        "protected_roots": sorted(protected),
        "graph_coverage": {
            "located_tokens": covered,
            "total_tokens": before_total,
            "fraction": round(coverage, 6),
        },
        "graph_validation": {
            "before_schema": before_graph.schema,
            "after_schema": after_graph.schema,
            "before_sha256": before_graph.sha256,
            "after_sha256": after_graph.sha256,
            "before_nodes": len(before_names),
            "after_nodes": len(after_names),
            "retained_nodes": len(retained),
            "added_nodes": len(added),
            "deleted_nodes": len(deleted),
            "edge_counts": {
                policy: {
                    "before": len(before_graph.edge_set(policy)),
                    "after": len(after_graph.edge_set(policy)),
                }
                for policy in EDGE_POLICIES
            },
            "unresolved_source_modules": {
                "before": sorted(before_source.unresolved_modules),
                "after": sorted(after_source.unresolved_modules),
            },
        },
        "new_declarations": {
            "added_nodes": len(added),
            "declaration_kinds": len(new_declarations),
            "written_declarations": len(written_declaration_groups(after_graph, new_declarations)),
            "alias_nodes": len(distinct_new),
            "companions_over_surviving_source": len(new_declarations - distinct_new),
            "used_for_removal_attribution": len(attribution_new),
        },
        "metric_scope": {
            "challenge_source_path": challenge,
            "excluded_files": sorted(set(excluded_files)),
            "exclude_dirs": scope.get("exclude_dirs") or [],
            "include_prefix": str(scope.get("include_prefix") or ""),
        },
        "reconciliation": reconciliation,
        "dead_code": dead,
        "structural_change": structural,
        "proof_rewriting": headline_rewriting,
        "proof_rewriting_by_edge_policy": {
            policy: {
                "candidates": section["candidates"],
                "excluded": {
                    key: value for key, value in section["excluded"].items()
                    if not key.endswith("_names")
                },
                "by_scope": {
                    scope_name: {
                        "totals": {
                            key: value for key, value in scope_entry["totals"].items()
                            if key != "names"
                        },
                        **{
                            f"automation_policy_{automation}": {
                                bucket: {
                                    key: value
                                    for key, value in
                                    scope_entry[f"automation_policy_{automation}"][bucket].items()
                                    if key != "names"
                                }
                                for bucket in ("automation_rewriting", "explicit_rewriting")
                            }
                            for automation in AUTOMATION_POLICIES
                        },
                    }
                    for scope_name, scope_entry in section["by_scope"].items()
                },
            }
            for policy, section in rewriting.items()
        },
        "uncategorized_retained_changes": other,
        "removal_attribution": removal,
        "new_declaration_degree": degrees,
        "text_accounting": text,
        "library_substitution": imports,
    }
    if options.node_diff_root is not None:
        diff_path = options.node_diff_root / "repositories" / repo / "diff.json"
        row["text_diff_cross_check"] = diff_cross_check(
            diff_path,
            {name for name in added if str(after_graph.nodes[name].get("kind")) in DECLARATION_KINDS},
            {name for name in deleted if str(before_graph.nodes[name].get("kind")) in DECLARATION_KINDS},
            {str(r["name"]) for r in headline_rewriting["rows"] if r["proof_text_changed"]},
        )
    return row


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def _mean(values: Sequence[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def aggregate(rows: Sequence[Mapping[str, Any]], *, edge_identity: str) -> dict[str, Any]:
    """Pool the categories across repositories and average their shares.

    Pooled totals answer "where did the tokens go across the benchmark"; the
    macro means answer "what does a typical repository look like", and the two
    differ whenever repository sizes do.
    """
    def pooled(section: str, key: str) -> int:
        return sum(int(row[section][key]) for row in rows)

    scored = [row for row in rows if "scored_tokens_saved" in row["reconciliation"]]
    scored_saved = sum(int(row["reconciliation"]["scored_tokens_saved"]) for row in scored)
    dead_tokens = pooled("reconciliation", "dead_code_tokens")
    structural_tokens = pooled("reconciliation", "structural_change_tokens")
    proof_tokens = pooled("reconciliation", "proof_rewriting_tokens")
    categorized = dead_tokens + structural_tokens + proof_tokens

    automation: dict[str, Any] = {}
    for policy in AUTOMATION_POLICIES:
        buckets: dict[str, Any] = {}
        for bucket in ("automation_rewriting", "explicit_rewriting"):
            buckets[bucket] = {
                key: sum(int(row["proof_rewriting"][f"automation_policy_{policy}"][bucket][key])
                         for row in rows)
                for key in ("count", "measured_count", "unmeasured_count", "tokens_saved",
                            "automation_invocations_before", "automation_invocations_after")
            }
        automation[policy] = buckets

    return {
        "repositories": len(rows),
        "repositories_with_scored_totals": len(scored),
        "edge_identity_policy": edge_identity,
        "pooled": {
            "scored_baseline_tokens": sum(
                int(row["reconciliation"]["scored_baseline_tokens"]) for row in scored
            ),
            "scored_tokens_saved": scored_saved,
            "dead_code_tokens": dead_tokens,
            "structural_change_tokens": structural_tokens,
            "structural_tokens_added_by_new_declarations": pooled(
                "reconciliation", "structural_tokens_added_by_new_declarations"
            ),
            "structural_tokens_removed_from_dependents": pooled(
                "reconciliation", "structural_tokens_removed_from_dependents"
            ),
            "tokens_removed_from_declarations_that_gained_any_edge": pooled(
                "reconciliation", "tokens_removed_from_declarations_that_gained_any_edge"
            ),
            "proof_rewriting_tokens": proof_tokens,
            "categorized_tokens": categorized,
            "residual_tokens": scored_saved - categorized if scored else None,
            "uncategorized_retained_change_tokens": pooled(
                "reconciliation", "uncategorized_retained_change_tokens"
            ),
            "removed_live_declaration_tokens": pooled(
                "reconciliation", "removed_live_declaration_tokens"
            ),
            "structural_removal_tokens": pooled("reconciliation", "structural_removal_tokens"),
            "rewrite_enabled_removal_tokens": pooled(
                "reconciliation", "rewrite_enabled_removal_tokens"
            ),
            "unattributed_removal_tokens": pooled(
                "reconciliation", "unattributed_removal_tokens"
            ),
            "non_declaration_text_tokens": pooled("reconciliation", "non_declaration_text_tokens"),
            "other_addition_cost_tokens": pooled("reconciliation", "other_addition_cost_tokens"),
            "structural_drivers": pooled("reconciliation", "structural_drivers"),
            "rewriting_drivers": pooled("reconciliation", "rewriting_drivers"),
            "contested_removals": pooled("reconciliation", "contested_removals"),
            "contested_removal_tokens": pooled("reconciliation", "contested_removal_tokens"),
            "structural_drivers_changed": pooled("reconciliation", "structural_drivers_changed"),
            "rewriting_drivers_changed": pooled("reconciliation", "rewriting_drivers_changed"),
            "library_substitution_tokens": pooled("reconciliation", "library_substitution_tokens"),
            "tokens_in_removed_files": pooled("reconciliation", "tokens_in_removed_files"),
            "tokens_in_files_removed_with_a_library_import": pooled(
                "reconciliation", "tokens_in_files_removed_with_a_library_import"
            ),
            "retained_declaration_tokens_saved": pooled(
                "reconciliation", "retained_declaration_tokens_saved"
            ),
            "retained_attribution_slack_tokens": pooled(
                "reconciliation", "retained_attribution_slack_tokens"
            ),
            "residual_after_live_removal_tokens": (
                sum(int(row["reconciliation"]["residual_after_live_removal_tokens"])
                    for row in scored) if scored else None
            ),
            "removed_live_share_of_saving_pct": (
                round(100.0 * pooled("reconciliation", "removed_live_declaration_tokens")
                      / scored_saved, 6) if scored_saved else None
            ),
            "dead_code_share_of_saving_pct": (
                round(100.0 * dead_tokens / scored_saved, 6) if scored_saved else None
            ),
            "structural_share_of_saving_pct": (
                round(100.0 * structural_tokens / scored_saved, 6) if scored_saved else None
            ),
            "proof_rewriting_share_of_saving_pct": (
                round(100.0 * proof_tokens / scored_saved, 6) if scored_saved else None
            ),
            "categorized_share_of_saving_pct": (
                round(100.0 * categorized / scored_saved, 6) if scored_saved else None
            ),
        },
        "macro_mean_share_of_saving_pct": {
            key: _mean([
                float(row["reconciliation"][key])
                for row in scored if row["reconciliation"].get(key) is not None
            ])
            for key in (
                "dead_code_share_of_scored_saving_pct",
                "structural_share_of_scored_saving_pct",
                "proof_rewriting_share_of_scored_saving_pct",
                "removed_live_share_of_scored_saving_pct",
                "non_declaration_share_of_scored_saving_pct",
                "categorized_share_of_scored_saving_pct",
            )
        },
        "dead_code": {
            key: pooled("dead_code", key)
            for key in ("available_nodes", "available_tokens", "removed_nodes", "removed_tokens",
                        "retained_nodes", "retained_tokens",
                        "deleted_but_protected_reachable_nodes",
                        "deleted_but_protected_reachable_tokens")
        },
        "structural_change": {
            "new_declarations": pooled("structural_change", "new_declarations"),
            "introduction_cost_tokens": pooled("structural_change", "introduction_cost_tokens"),
            "direct_dependent_tokens_saved": sum(
                int(row["structural_change"]["direct_dependents"]["tokens_saved"]) for row in rows
            ),
            "direct_dependents": sum(
                int(row["structural_change"]["direct_dependents"]["count"]) for row in rows
            ),
            "transitive_dependents": sum(
                int(row["structural_change"]["transitive_dependents"]["count"]) for row in rows
            ),
            "transitive_dependent_tokens_saved": sum(
                int(row["structural_change"]["transitive_dependents"]["tokens_saved"])
                for row in rows
            ),
            "dependents_that_gained_any_edge": sum(
                int(row["structural_change"]["dependents_that_gained_any_edge"]["count"])
                for row in rows
            ),
            "tokens_removed_from_declarations_that_gained_any_edge": sum(
                int(row["structural_change"]["dependents_that_gained_any_edge"]["tokens_saved"])
                for row in rows
            ),
            "net_tokens_saved": structural_tokens,
            "repositories_where_new_declarations_pay_for_themselves": sum(
                1 for row in rows if row["structural_change"]["pays_for_itself"]
            ),
        },
        "removal_attribution": {
            "counts": {
                label: sum(int(row["removal_attribution"]["counts"][label]) for row in rows)
                for label in REMOVAL_GROUPS
            },
            "tokens": {
                label: sum(
                    int(row["text_accounting"]["removed_declaration_tokens"][label])
                    for row in rows
                )
                for label in REMOVAL_GROUPS
            },
            "tokens_by_kind": {
                label: {
                    kind: sum(
                        int(row["text_accounting"]["removed_declaration_tokens_by_kind"]
                            [label].get(kind, 0))
                        for row in rows
                    )
                    for kind in sorted({
                        kind
                        for row in rows
                        for kind in row["text_accounting"]["removed_declaration_tokens_by_kind"][label]
                    })
                }
                for label in REMOVAL_GROUPS
            },
            "counts_by_kind": {
                label: {
                    kind: sum(
                        int(row["removal_attribution"]["counts_by_kind"][label].get(kind, 0))
                        for row in rows
                    )
                    for kind in sorted({
                        kind
                        for row in rows
                        for kind in row["removal_attribution"]["counts_by_kind"][label]
                    })
                }
                for label in REMOVAL_GROUPS
            },
        },
        "new_declarations": {
            key: sum(int(row["new_declarations"][key]) for row in rows)
            for key in ("added_nodes", "declaration_kinds", "written_declarations",
                        "alias_nodes", "companions_over_surviving_source",
                        "used_for_removal_attribution")
        },
        "new_declaration_degree": {
            **{
                scope: {
                    "declarations": sum(
                        int(row["new_declaration_degree"][scope]["declarations"]) for row in rows
                    ),
                    "total_in_degree": sum(
                        int(row["new_declaration_degree"][scope]["total_in_degree"]) for row in rows
                    ),
                    "unused": sum(
                        int(row["new_declaration_degree"][scope]["unused"]) for row in rows
                    ),
                    "used_once": sum(
                        int(row["new_declaration_degree"][scope]["used_once"]) for row in rows
                    ),
                    "used_at_least_twice": sum(
                        int(row["new_declaration_degree"][scope]["used_at_least_twice"])
                        for row in rows
                    ),
                    "max_in_degree": max(
                        (int(row["new_declaration_degree"][scope]["max_in_degree"] or 0)
                         for row in rows),
                        default=0,
                    ),
                }
                for scope in ("written", "nodes", "term_level", "syntax_level", "other_command")
            },
            "written_by_kind": {
                kind: {
                    "declarations": sum(
                        int(row["new_declaration_degree"]["written_by_kind"][kind]
                            ["declarations"])
                        for row in rows
                        if kind in row["new_declaration_degree"]["written_by_kind"]
                    ),
                    "total_in_degree": sum(
                        int(row["new_declaration_degree"]["written_by_kind"][kind]
                            ["total_in_degree"])
                        for row in rows
                        if kind in row["new_declaration_degree"]["written_by_kind"]
                    ),
                }
                for kind in sorted({
                    kind for row in rows
                    for kind in row["new_declaration_degree"]["written_by_kind"]
                })
            },
            "declarations": sum(
                int(row["new_declaration_degree"]["written"]["declarations"]) for row in rows
            ),

            "by_kind": {
                kind: {
                    "declarations": sum(
                        int(row["new_declaration_degree"]["by_kind"][kind]["declarations"])
                        for row in rows if kind in row["new_declaration_degree"]["by_kind"]
                    ),
                    "total_in_degree": sum(
                        int(row["new_declaration_degree"]["by_kind"][kind]["total_in_degree"])
                        for row in rows if kind in row["new_declaration_degree"]["by_kind"]
                    ),
                }
                for kind in sorted({
                    kind for row in rows for kind in row["new_declaration_degree"]["by_kind"]
                })
            },
        },
        "proof_rewriting": {
            "candidates": sum(int(row["proof_rewriting"]["candidates"]["count"]) for row in rows),
            "measured_candidates": sum(
                int(row["proof_rewriting"]["candidates"]["measured_count"]) for row in rows
            ),
            "tokens_saved": proof_tokens,
            "statement_changed": sum(
                int(row["proof_rewriting"]["excluded"]["statement_changed"]) for row in rows
            ),
            "dependency_edge_set_changed": sum(
                int(row["proof_rewriting"]["excluded"]["dependency_edge_set_changed"]) for row in rows
            ),
            "by_automation_policy": automation,
        },
        "proof_rewriting_by_edge_policy": {
            policy: {
                "candidates": sum(
                    int(row["proof_rewriting_by_edge_policy"][policy]["candidates"]["count"])
                    for row in rows
                ),
                "tokens_saved": sum(
                    int(row["proof_rewriting_by_edge_policy"][policy]["candidates"]["tokens_saved"])
                    for row in rows
                ),
                "by_scope": {
                    scope: {
                        "count": sum(
                            int(row["proof_rewriting_by_edge_policy"][policy]["by_scope"][scope]
                                ["totals"]["count"])
                            for row in rows
                        ),
                        "tokens_saved": sum(
                            int(row["proof_rewriting_by_edge_policy"][policy]["by_scope"][scope]
                                ["totals"]["tokens_saved"])
                            for row in rows
                        ),
                        **{
                            f"automation_policy_{automation}": {
                                bucket: {
                                    key: sum(
                                        int(row["proof_rewriting_by_edge_policy"][policy]
                                            ["by_scope"][scope][f"automation_policy_{automation}"]
                                            [bucket][key])
                                        for row in rows
                                    )
                                    for key in ("count", "tokens_saved",
                                                "automation_invocations_before",
                                                "automation_invocations_after")
                                }
                                for bucket in ("automation_rewriting", "explicit_rewriting")
                            }
                            for automation in AUTOMATION_POLICIES
                        },
                    }
                    for scope in REWRITING_SCOPES
                },
            }
            for policy in EDGE_POLICIES
        },
    }


def _strip_rows(payload: Any) -> Any:
    """Drop per-declaration row arrays so a summary output stays readable."""
    row_keys = {"rows", "dependent_rows", "new_declaration_rows", "gained_edge_rows",
                "imports_added_by_module", "imports_removed_by_module", "rewired_user_rows",
                "groups"}
    if isinstance(payload, dict):
        return {
            key: _strip_rows(value)
            for key, value in payload.items()
            if key not in row_keys and not key.endswith("_names") and key != "names"
        }
    if isinstance(payload, list):
        return [_strip_rows(item) for item in payload]
    return payload


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--graph-root", type=Path, required=True,
                        help="Endpoint-graph run directory, with or without a repositories/ level")
    parser.add_argument("--dataset", type=Path, required=True,
                        help="Dataset YAML pinning the protected declarations")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path,
                        help="Fallback original sources: <root>/<repo>/stripped")
    parser.add_argument("--terminal-root", type=Path,
                        help="Fallback final sources: <root>/<repo>/playback/capture_001/submitted.sources.tar.gz")
    parser.add_argument("--declaration-text-root", type=Path,
                        help="Elaborated statement dumps: <root>/<repo>/{before,after}-declarations.json")
    parser.add_argument("--playback-root", type=Path,
                        help="Evaluation output root holding the scored postprocessed playbacks")
    parser.add_argument("--playback-template",
                        default="{repository}/playback/capture_001/postprocessed-playback.json")
    parser.add_argument(
        "--use-source-token-totals", action="store_true",
        help=("Use the exact scoped before/after source archives as the scored token totals "
              "when no postprocessed playback root is available. This is intended for "
              "completed-build graph bundles whose terminal build already passed."),
    )

    parser.add_argument("--comparator-root", type=Path,
                        help="Dataset repo root holding <repo>/comparator/manifest.json, used to "
                             "drop the harness-registered Challenge file from the scored file set")
    parser.add_argument("--exclude-file", action="append", default=[],
                        help="Additional repository-relative Lean file to leave out (repeatable)")
    parser.add_argument("--node-diff-root", type=Path,
                        help="Optional independent text diff to cross-check the graph sets against")
    parser.add_argument("--edge-identity", choices=EDGE_POLICIES, default="union",
                        help="Edge set whose identity defines proof rewriting (default: union)")
    parser.add_argument("--rewriting-scope", choices=REWRITING_SCOPES, default="identical_edge_set",
                        help="Which same-statement rewrites count as proof rewriting "
                             "(default: the strict identical-edge-set definition)")
    parser.add_argument("--automation-policy", choices=AUTOMATION_POLICIES, default="strict",
                        help="Headline automation split; both policies are always reported")
    parser.add_argument("--count-co-located-additions", action="store_true",
                        help="Let companions minted over source that survived the edit count as "
                             "new declarations when attributing removals")
    parser.add_argument("--extra-automation-tactic", action="append", default=[],
                        help="Additional tactic name to count as automation (repeatable)")
    parser.add_argument("--repository", action="append", default=[],
                        help="Restrict the analysis to these repositories (repeatable)")
    parser.add_argument("--skip-repository", action="append", default=[],
                        help="Exclude these repositories from the analysis (repeatable)")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="Skip repositories whose graph extraction has not finished, "
                             "recording each one and why, instead of failing")
    parser.add_argument("--minimum-graph-coverage", type=float, default=0.5,
                        help="Refuse a repository whose graph nodes cannot be located in at "
                             "least this fraction of its scored source (default 0.5)")
    parser.add_argument("--allow-broken-graphs", action="store_true",
                        help="Skip repositories whose graphs are missing their protected "
                             "roots, recording each one, instead of failing")
    parser.add_argument("--summary-output", type=Path,
                        help="Also write a row-free summary to this path")
    return parser


def main() -> None:
    options = build_parser().parse_args()
    graph_root = options.graph_root
    if (graph_root / "repositories").is_dir():
        graph_root = graph_root / "repositories"
    options.graph_root = graph_root

    repositories = sorted(path.name for path in graph_root.iterdir() if path.is_dir())
    if options.repository:
        requested = set(options.repository)
        unknown = sorted(requested - set(repositories))
        if unknown:
            raise SystemExit(f"requested repositories absent from {graph_root}: {unknown}")
        repositories = [repo for repo in repositories if repo in requested]
    if options.skip_repository:
        excluded = set(options.skip_repository)
        repositories = [repo for repo in repositories if repo not in excluded]
    if not repositories:
        raise SystemExit(f"no repository graph artifacts under {graph_root}")

    protected_by_repo, scopes_by_repo, database_hashes = load_protected(options.dataset.resolve())
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for repo in repositories:
        repo_dir = graph_root / repo
        reason: str | None = None
        if not (repo_dir / "before.json").is_file() or not (repo_dir / "after.json").is_file():
            reason = "incomplete before/after graph pair"
        else:
            summary_path = repo_dir / "summary.json"
            if summary_path.is_file() and read_json(summary_path).get("status") != "complete":
                reason = "graph extraction has not completed"
        if reason is not None:
            # Fail closed by default: a partial run must not quietly become a
            # smaller benchmark with a different mean.
            if not options.allow_incomplete:
                raise SystemExit(f"{repo}: {reason}")
            skipped.append({"repository": repo, "reason": reason})
            continue
        if repo not in protected_by_repo:
            raise SystemExit(f"{repo}: missing protected-root record in the pinned dataset")
        try:
            rows.append(classify_repo(
                repo, options, protected_by_repo[repo], scopes_by_repo.get(repo)
            ))
        except BrokenGraphError as error:
            if not options.allow_broken_graphs:
                raise
            skipped.append({"repository": repo, "reason": str(error)})
    if not rows:
        raise SystemExit(f"no complete repository graph pairs under {graph_root}")

    result = {
        "schema": "leanlean_compression_categories_v3",
        "dataset": str(options.dataset.resolve()),
        "repository_database_hashes": database_hashes,
        "settings": {
            "edge_identity": options.edge_identity,
            "rewriting_scope": options.rewriting_scope,
            "automation_policy": options.automation_policy,
            "extra_automation_tactics": sorted(options.extra_automation_tactic),
            "automation_search_tactics": sorted(AUTOMATION_SEARCH_TACTICS),
            "automation_normalization_tactics": sorted(AUTOMATION_NORMALIZATION_TACTICS),
        },
        "edge_semantics": (
            "an edge points from a user declaration to a declaration it depends on; "
            "root reachability follows outgoing edges and users are counted by incoming edges"
        ),
        "skipped_repositories": skipped,
        "aggregate": aggregate(rows, edge_identity=options.edge_identity),
        "repositories": rows,
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if options.summary_output is not None:
        options.summary_output.parent.mkdir(parents=True, exist_ok=True)
        options.summary_output.write_text(
            json.dumps(_strip_rows(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(json.dumps(result["aggregate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
