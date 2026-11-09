#!/usr/bin/env python3
"""Estimate source-token savings from new definitions, notation and macros.

Fixed templates are priced as tokens(expansion) - tokens(call), with each
parameter counted once. Repeated parameters are flagged; nested abbreviations
include their own expansion savings. Syntax without a fixed expansion counts
as machinery cost.

Definition uses require dependency evidence and exclude unfolding hint lists.
Notation uses require a distinctive literal and the declaration's source scope.
Supported elaborator factories are handled by generated_syntax.py; other
elaborators and ambiguous notation remain unpriced.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leanlean.metrics.tokens import lean_refactor_lexer, remove_lean_comments  # noqa: E402
import scripts.analysis.expand_definitions as definition_pass  # noqa: E402
import scripts.analysis.generated_syntax as generated_pass
import scripts.analysis.expand_macros as pattern_pass  # noqa: E402
from scripts.analysis.quantify_compression_categories import SourceIndex, load_graph  # noqa: E402

_HEAD = re.compile(
    r"^\s*(?:@\[[^\]]*\]\s*)*(?P<mods>(?:(?:private|protected|noncomputable|unsafe|partial|local|scoped)\s+)*)"
    r"(?P<command>def|abbrev|notation3|notation|infixl|infixr|infix|prefix|postfix|macro_rules|macro"
    r"|syntax|elab_rules|elab|declare_syntax_cat)\b")
_IMPORT = re.compile(r"^\s*import\s+(.+)$", re.M)
# a definition named in one of these steps' hint lists is unfolded, not used
_HINT_HEADS = frozenset({"simp", "simp_all", "dsimp", "simpa", "simp_rw", "rw", "rwa", "erw", "rewrite",
                         "nth_rewrite", "nth_rw", "unfold", "delta", "field_simp", "norm_num", "aesop",
                         "grind", "simp?", "simp!", "fun_prop", "positivity"})


# names Lean code uses for local variables: one letter (with primes, digits or
# subscripts), or `h`/`H`/`ih`/`this`-style hypotheses
_LOCAL_NAME = re.compile(r"^(?:[A-Za-zα-ωΑ-Ω][₀-₉'0-9]*|(?:h|H|ih|hyp|this)[\w'₀-₉]*)$")


def tokens(text: str) -> list[str]:
    return [t for line in remove_lean_comments(text).splitlines()
            for t in lean_refactor_lexer(line.strip()).split()]


@dataclass
class Abbreviation:
    kind: str                     # definition | notation | macro | generated | machinery | automation
    command: str
    names: list[str]              # the graph nodes the command created
    path: str
    line: int
    local: bool
    keys: list[list[str]]         # token sequences that mark one use
    template: list[str]           # expansion tokens, parameters as their names
    params: list[str]
    base_saving: int              # saving of one use, before nested abbreviations
    saving: int = 0               # with nested new abbreviations priced in
    cost: int = 0                 # tokens of the command itself
    repeated: bool = False        # a parameter occurs other than once in the expansion
    nested: dict[str, int] = field(default_factory=dict)
    priced: bool = True           # False: machinery, counted as cost only
    note: str = ""
    generated_category: str = ""
    generated_constant: int | None = None
    generated_weights: list[int] = field(default_factory=list)


@dataclass
class Skipped:
    command: str
    names: list[str]
    path: str
    line: int
    reason: str
    cost: int


def _occurrences(seq: Sequence[str], key: Sequence[str]) -> int:
    n, k = len(seq), len(key)
    if not k or k > n:
        return 0
    return sum(1 for i in range(n - k + 1) if list(seq[i:i + k]) == list(key))


def _pattern(command: str, source: str):
    """(items, template) of a notation/macro, or a reason."""
    if command in ("infix", "infixl", "infixr", "prefix", "postfix"):
        parsed = pattern_pass.parse_command(source, command)
        if isinstance(parsed, str):
            return parsed
        items, template = parsed
        literals = [i for i in items if i[0] == "lit"]
        if len(literals) != 1:
            return "operator without one literal"
        if command.startswith("infix"):
            return [("param", "a", ""), literals[0], ("param", "b", "")], f"{template} a b"
        if command == "prefix":
            return [literals[0], ("param", "a", "")], f"{template} a"
        return [("param", "a", ""), literals[0]], f"{template} a"
    return pattern_pass.parse_command(source, command)


_GENERATED = re.compile(r"(?:^|\.)(?:_aux_\w*?_macroRules_)?«?(term|tactic|command)(?P<sym>[^_»]+)")
_OPTIONAL = re.compile(r"\$\[[^\]]*\]\?")


def _generated_symbol(names: Sequence[str]) -> str | None:
    """`termĀĀ`, `_aux_M___macroRules_tacticZy16_1` -> `ĀĀ`, `Zy16`."""
    for name in names:
        match = _GENERATED.search(name.rsplit(".", 1)[-1] if "_aux_" not in name else name)
        if match and ("macroRules" in name or name.rsplit(".", 1)[-1].lstrip("«").startswith(match.group(1))):
            return match.group("sym")
    return None


def _generated(source: str, symbol: str):
    """(key, expansion) of a DSL line that defines an abbreviation: the token that
    names it (`zq3`, or `ĀĀ` glued to `zn`) and everything after it."""
    toks = tokens(source)
    for index, tok in enumerate(toks):
        low, sym = tok.lower(), symbol.lower()
        if low == sym or (low.endswith(sym) and len(low) > len(sym)):
            key = tok if low == sym else tok[-len(symbol):]
            rest = toks[index + 1:]
            if rest[:1] == ['"']:
                rest = rest[1:]
            return [key], rest
    return None


def _macro_rules(source: str):
    """(items, template) of a single-alternative `macro_rules`, read from its own
    quotations; optional groups `$[..]?` are dropped on both sides."""
    code = _OPTIONAL.sub("", remove_lean_comments(source))
    alternatives = [a for a in re.split(r"\n\s*\|", code.split("macro_rules", 1)[-1]) if a.strip()]
    if len(alternatives) != 1:
        return "several macro_rules alternatives"
    alt = alternatives[0].lstrip().lstrip("|")
    if "=>" not in alt:
        return "no expansion arrow"
    lhs, rhs = alt.split("=>", 1)
    lhs, reason = pattern_pass.strip_quotation(lhs.strip())
    if reason:
        return reason
    template, reason = pattern_pass.strip_quotation(rhs.strip())
    if reason:
        return reason
    if re.search(r"\$\(|\$\[", lhs + template):
        return "spliced expansion"
    items = []
    for tok in lhs.split():
        match = re.fullmatch(r"\$(\w+)(?::[\w.]+)?", tok)
        items.append(("param", match.group(1), "") if match else ("lit", tok, ""))
    if not any(kind == "lit" for kind, _, _ in items):
        return "no literal to match on"
    return items, template


def _any_definition(source: str, reason: str):
    """A definition the template parser refused, priced anyway when it has a
    finite expansion: a tactic body (`by ..`), equations (`| 0 => a | n+1 => b`,
    written back as `fun | ..`), a `where` block, or binders it cannot read (the
    call is then just the name).  A recursive definition has none."""
    masked = remove_lean_comments(source, mask_strings=True)
    code = remove_lean_comments(source)
    head = definition_pass._HEAD_RE.search(masked)  # noqa: SLF001
    if head is None:
        return reason
    name = head.group("name")
    short = name.rsplit(".", 1)[-1]
    assign = pattern_pass.find_literal(masked, ":=", head.end())
    bar = re.search(r"(?:^|\n)\s*\|", masked[head.end():])
    if assign >= 0 and not (bar and head.end() + bar.start() < assign):
        signature, body = code[head.end():assign], code[assign + 2:]
        prefix: list[str] = []
    elif bar:
        cut = head.end() + bar.start()
        signature, body = code[head.end():cut], code[cut:]
        prefix = ["fun"]
    else:
        return reason
    if re.search(r"(?<![\w.'])" + re.escape(short) + r"(?![\w'])", body):
        return "recursive definition"
    colon = pattern_pass.find_literal(remove_lean_comments(signature, mask_strings=True), ":", 0)
    binders = definition_pass.explicit_binders(signature[:colon] if colon >= 0 else signature)
    params = binders if isinstance(binders, list) else []
    items = [("lit", name, "")] + [("param", b, "term") for b in params]
    return items, " ".join(prefix) + " " + body.strip(), name


def automation_only(template: Sequence[str]) -> bool:
    """A tactic expansion made of automation and routing only (parentheses and
    `first | .. | ..` alternatives included)."""
    from scripts.analysis import proof_hunks as ph
    seen = False
    for step in ph._steps([" ".join(template)]):  # noqa: SLF001
        step = [t for t in step if t not in ("(", ")")]
        if not step:
            continue
        label = ph.step_is_automation(step)
        if label is False:
            return False
        seen = seen or bool(label)
    return seen


def build(before, after, before_src: SourceIndex, after_src: SourceIndex,
          after_tokens: Mapping[str, int]) -> tuple[list[Abbreviation], list[Skipped]]:
    """The new abbreviations of a repository, priced, and the commands set aside."""
    before_vocab: set[str] = set()
    for text in before_src._files.values():  # noqa: SLF001
        before_vocab.update(tokens(text))
    new = set(after.nodes) - set(before.nodes)
    ranges: dict[tuple[str, int, int], list[str]] = collections.defaultdict(list)
    for name in new:
        node = after.nodes[name]
        path = after_src.path_of_module(str(node.get("module", "")))
        if path and node.get("start_line"):
            ranges[(path, int(node["start_line"]), int(node["end_line"]))].append(name)

    # Recognize the factory implementation before inspecting its invocations.
    factories = {}
    factory_ranges = set()
    for (path, start, end), names in sorted(ranges.items()):
        source = "".join(after_src._files.get(path, "").splitlines(keepends=True)[start - 1:end])
        spec = generated_pass.factory(source)
        if spec:
            # The quoted binder names in this factory family deliberately use
            # the caller's names. Without this setting, Lean hygienically
            # renames them and token substitution would not be justified.
            prefix = "".join(after_src._files.get(path, "").splitlines(keepends=True)[:start - 1])
            settings = re.findall(r"^\s*set_option\s+hygiene\s+(true|false)\s*$", remove_lean_comments(prefix), re.M)
            if not settings or settings[-1] != "false":
                raise ValueError(f"generated syntax factory lacks explicit hygiene false: {path}:{start}")
            if spec.literal in factories and factories[spec.literal] != spec:
                raise ValueError(f"ambiguous syntax factory: {spec.literal}")
            factories[spec.literal] = spec
            factory_ranges.add((path, start, end))
    found: list[Abbreviation] = []
    skipped: list[Skipped] = []
    for (path, start, end), names in sorted(ranges.items()):
        names = sorted(names)
        lines = after_src._files.get(path, "").splitlines(keepends=True)  # noqa: SLF001
        source = "".join(lines[start - 1:end])
        head = _HEAD.match(remove_lean_comments(source, mask_strings=True))
        cost = sum(after_tokens.get(n, 0) for n in names)

        def machinery(command: str, note: str, keys=()) -> None:
            found.append(Abbreviation("machinery", command, names, path, start, False, [list(k) for k in keys],
                                      [], [], 0, cost=cost, priced=False, note=note))

        if (path, start, end) in factory_ranges:
            machinery("elab", "verified generated-syntax factory")
            continue
        if head is None:
            lead = tokens(source)[:1]
            spec = factories.get(lead[0]) if lead else None
            if spec:
                symbol = generated_pass.symbol(names, spec.category)
                template = generated_pass.template(source, spec, tokens)
                if symbol is None or template is None:
                    raise ValueError(f"unresolved generated syntax: {path}:{start}")
                if symbol in before_vocab:
                    raise ValueError(f"generated literal already in baseline: {symbol}")
                found.append(Abbreviation("generated", "generated", names, path, start, False,
                                          [[symbol]], template, list(spec.params), 0, cost=cost,
                                          generated_category=spec.category,
                                          note="command-generated prefix syntax; graph-attested literal"))
                continue
            symbol = _generated_symbol(names)
            generated = _generated(source, symbol) if symbol else None
            if generated is None:
                continue              # a theorem, instance, structure ...: not syntax
            key, template = generated
            if all(t in before_vocab for t in key):
                skipped.append(Skipped("generated", names, path, start,
                                       "every literal already occurs in the before sources", cost))
                continue
            found.append(Abbreviation("generated", "generated", names, path, start, False, [key],
                                      template, [], len(template) - len(key), cost=cost))
            continue
        command, local = head.group("command"), "local" in head.group("mods").split()
        if command in ("elab", "elab_rules"):
            skipped.append(Skipped(command, names, path, start, "elab: not decided yet", cost))
            continue
        if command in ("syntax", "declare_syntax_cat"):
            machinery(command, "declares a parser; its expansion, if any, is a macro_rules")
            continue
        if command in ("def", "abbrev"):
            parsed = definition_pass.parse_definition(source)
            if isinstance(parsed, str):
                parsed = _any_definition(source, parsed)
            if isinstance(parsed, str):
                named = definition_pass._HEAD_RE.search(remove_lean_comments(source, mask_strings=True))  # noqa: SLF001
                machinery(command, parsed, [[named.group("name")]] if named else [])
                continue
            items, body, name = parsed
            params = [v for kind, v, _ in items if kind == "param"]
            template = tokens(body)
            call = 1                  # the name
            # the key is the qualified graph name: `def f` in `namespace Ns` is `Ns.f`
            short = name.rsplit(".", 1)[-1]
            keys = [[next((n for n in names if n.rsplit(".", 1)[-1] == short), name)]]
            kind = "definition"
        else:
            parsed = _macro_rules(source) if command == "macro_rules" else _pattern(command, source)
            if isinstance(parsed, str):
                machinery(command, parsed)
                continue
            items, body = parsed
            params = [v for kind, v, _ in items if kind == "param"]
            body = re.sub(r"\$(\w+)(?::\w+)?", r"\1", body)       # `$x:term` antiquotations
            template = tokens(body)
            literals = [tokens(v) for kind, v, _ in items if kind == "lit" and v.strip()]
            call = sum(len(l) for l in literals)
            keys = [l for l in literals if l and not all(t in before_vocab for t in l)]
            if not keys:
                skipped.append(Skipped(command, names, path, start,
                                       "every literal already occurs in the before sources", cost))
                continue
            kind = "macro" if command in ("macro", "macro_rules") else "notation"
        occurrences = {p: template.count(p) for p in params}
        base = len(template) - sum(occurrences.values()) - call
        found.append(Abbreviation(kind, command, names, path, start, local, keys, template, params, base,
                                  cost=cost, repeated=any(v != 1 for v in occurrences.values())))

    # price nested abbreviations: an expansion that uses another new one saves that one's saving too
    def uses(template: Sequence[str], other: Abbreviation) -> int:
        if other.kind == "definition":
            short = other.keys[0][0].rsplit(".", 1)[-1]
            return sum(1 for t in template if t == other.keys[0][0] or t.rsplit(".", 1)[-1] == short)
        return min(_occurrences(template, key) for key in other.keys)

    generated_pass.compile_forms(found)
    by_token: dict[str, set[int]] = collections.defaultdict(set)
    for j, other in enumerate(found):
        if other.keys:
            for key in other.keys:
                by_token[key[0]].add(j)
            if other.kind == "definition":
                by_token[other.keys[0][0].rsplit(".", 1)[-1]].add(j)
    # Nested expansions must obey the same visibility rules as direct uses.
    # Local notation literals are routinely reused by unrelated files; globally
    # matching those literals recursively invents enormous expansion costs.
    pricer = Pricer(found, after, after_src)
    indices = {id(a): i for i, a in enumerate(found)}
    visible = [
        {indices[id(other)] for name in a.names for other in pricer.visible(name)}
        for a in found
    ]
    memo: dict[int, int] = {}

    def price(index: int, stack: frozenset[int]) -> int:
        if index in memo:
            return memo[index]
        a = found[index]
        if a.generated_category:
            return a.saving
        total = a.base_saving
        candidates = set().union(*(by_token.get(t, set()) | by_token.get(t.rsplit(".", 1)[-1], set()) for t in a.template))
        for j in sorted(candidates & visible[index]):
            other = found[j]
            if j == index or j in stack or not other.priced or not other.keys:
                continue
            n = uses(a.template, other)
            if n:
                a.nested[" ".join(other.keys[0])] = n
                total += n * price(j, stack | {index})
        memo[index] = total
        return total

    for i, a in enumerate(found):
        if a.priced:
            a.saving = price(i, frozenset())
    # a tactic macro that only runs automation is an automation tactic, not a
    # re-spelling: its steps are automation, and its uses save nothing as syntax
    for a in found:
        if a.kind in ("macro", "generated") and not a.generated_category and a.priced and len(a.keys[0]) == 1 \
                and a.template and automation_only(a.template):
            a.kind, a.priced, a.saving, a.note = "automation", False, 0, "automation-only tactic macro"
    return found, skipped


def automation_macros(abbreviations: Sequence[Abbreviation]) -> set[str]:
    """Heads of the repository's automation-only tactic macros."""
    return {a.keys[0][0].strip() for a in abbreviations if a.kind == "automation" and a.keys}


def _imports(after_src: SourceIndex) -> dict[str, set[str]]:
    """path -> the repository paths it imports, transitively."""
    direct: dict[str, set[str]] = collections.defaultdict(set)
    for path, text in after_src._files.items():  # noqa: SLF001
        for match in _IMPORT.finditer(remove_lean_comments(text)):
            for module in match.group(1).split():
                target = after_src.path_of_module(module)
                if target:
                    direct[path].add(target)
    closure: dict[str, set[str]] = {}
    for path in after_src._files:  # noqa: SLF001
        seen, stack = set(), [path]
        while stack:
            for nxt in direct.get(stack.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        closure[path] = seen
    return closure


class VisibleAbbreviations(list):
    """Cache the generated dispatch table across a declaration's proof steps."""
    def __init__(self, values):
        super().__init__(values)
        self.generated = {a.keys[0][0]: a for a in self if a.generated_category and a.priced}
        self.ordinary = [a for a in self if not a.generated_category]


class Pricer:
    """Prices the abbreviation uses in the tokens of one declaration."""

    def __init__(self, abbreviations: list[Abbreviation], after, after_src: SourceIndex):
        self.abbreviations = abbreviations
        self.after = after
        self.after_src = after_src
        self.imports = _imports(after_src)
        self.depends: dict[str, set[str]] = collections.defaultdict(set)
        for source, target in after.edges:
            self.depends[source].add(target)

    def visible(self, declaration: str) -> list[Abbreviation]:
        node = self.after.nodes.get(declaration, {})
        path = self.after_src.path_of_module(str(node.get("module", "")))
        line = int(node.get("start_line") or 0)
        out = []
        for a in self.abbreviations:
            if declaration in a.names or not a.priced:
                continue
            if a.kind == "definition":
                if a.names and self.depends[declaration] & set(a.names):
                    out.append(a)
                continue
            same_file = a.path == path and a.line < line
            if same_file or (not a.local and a.path in self.imports.get(path or "", ())):
                out.append(a)
        return VisibleAbbreviations(out)

    @staticmethod
    def price_steps(steps: Sequence[Sequence[str]], visible: Sequence[Abbreviation]) -> int:
        """Tokens the abbreviation uses in these steps saved."""
        total = 0
        priced = visible if isinstance(visible, VisibleAbbreviations) else VisibleAbbreviations(visible)
        generated, ordinary = priced.generated, priced.ordinary
        for step in steps:
            if generated:
                total += generated_pass.price(step, generated)
            hint = bool(step) and step[0] in _HINT_HEADS
            for a in ordinary:
                if not a.priced or not a.saving:
                    continue
                if a.kind == "definition":
                    name = a.keys[0][0]
                    short = name.rsplit(".", 1)[-1]
                    # a name that looks like a local (`f`, `t`, `hx`) is only a use
                    # when written qualified: bare, it is the local variable
                    bare_ok = not _LOCAL_NAME.match(short)
                    depth, n = 0, 0
                    for t in step:
                        depth += t == "["
                        depth -= t == "]"
                        qualified = t == name or (t.endswith("." + short) and name.endswith(t))
                        dotted = bare_ok and (t == short or t.endswith("." + short))
                        if (qualified or dotted) and not (hint and depth > 0) \
                                and not (bool(step) and step[0] in ("unfold", "delta")):
                            n += 1
                else:
                    n = min(_occurrences(step, key) for key in a.keys)
                total += n * a.saving
        return total


def repository_table(repo_dir: Path, record: Mapping | None = None):
    from scripts.analysis.classify_compression import declaration_tokens
    before, after = load_graph(repo_dir / "before.json"), load_graph(repo_dir / "after.json")
    srcs = {}
    for side in ("before", "after"):
        index = SourceIndex.from_archive(repo_dir / side / "original-sources.tar.gz")
        scope = (record or {}).get("metric_scope", {})
        index.restrict(exclude_files=scope.get("excluded_files", []), exclude_dirs=scope.get("exclude_dirs", []),
                       include_prefix=scope.get("include_prefix", ""))
        srcs[side] = index
    return build(before, after, srcs["before"], srcs["after"], declaration_tokens(after, srcs["after"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo_dirs", nargs="+", type=Path, help="graph run repository directories")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    out = []
    for repo_dir in options.repo_dirs:
        found, skipped = repository_table(repo_dir)
        out.append({"repository": str(repo_dir), "abbreviations": [asdict(a) for a in found],
                    "skipped": [asdict(s) for s in skipped]})
        kinds = collections.Counter(a.kind for a in found)
        print(f"{repo_dir.name}: {dict(kinds)}  skipped {collections.Counter(s.reason.split(':')[0] for s in skipped)}")
    if options.output:
        options.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
