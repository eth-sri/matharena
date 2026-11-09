"""Split a retained declaration's before/after text into hunks and label them.

A retained declaration is compared line by line (comments stripped, whitespace
normalised).  Each maximal run of changed lines is a hunk.  The scored token
count is a per-line sum, so the hunks' token deltas add up to the declaration's
delta exactly and anything charged to a hunk is a partition, not an estimate.

Two things are read off a hunk, in this order:

  SYNTAX.  Rewrites that change how the proof is spelled but not the proof:
  ``simp only [..]`` -> ``simp``, dropping the type from a ``have``, merging
  consecutive ``rw`` / ``intro`` calls, ``by exact e`` -> ``e``, dot notation,
  redundant parentheses, binders hoisted into ``variable`` ...  Each rule
  rewrites the BEFORE side toward the AFTER side, and a rewrite is kept only if
  it lies on a shortest token-edit path from before to after:

      d(before, rewritten) + d(rewritten, after) == d(before, after)

  so every token a rule removes is one the model removed too, and a rule can
  never claim a saving the model made some other way.  (``simp only [a, b, c]``
  -> ``simp only [a]`` is hint pruning, not syntax: dropping the whole list
  overshoots and is rejected; dropping only ``only`` would not be on the path.)

  AUTOMATION.  After the syntax rules, code that collapsed into automation.  A
  hunk (or an aligned run of its steps) whose new steps are nothing but
  automation (``simp``, ``grind``, ``omega``, ``linarith`` ...) and goal routing,
  and whose old steps did work by hand, is charged its old steps minus its new
  ones -- including the helper claims (``have h1 : T``) that went with them.
  Old hand steps deleted outright next to automation, or dropped beside steps
  they were merely inlined into, are ``absorbed`` by that automation; a helper
  ``have`` deleted in one hunk whose use became ``(by positivity)`` in another
  is ``absorbed_at_use``.  Inlining (the lemma is still named, elsewhere),
  pruning (``<;> ring`` dropped) and a new argument (new lemmas) are not.

What is left of each hunk goes on to the structural/rewriting split, which is
decided per declaration by its dependency set.
"""

from __future__ import annotations

import collections
import difflib
import re
from dataclasses import dataclass, field
from typing import Callable, Iterable, Iterator, Mapping, Sequence

from leanlean.metrics.tokens import _token_count, lean_refactor_lexer, remove_lean_comments

NL = "⏎"  # line-break sentinel in token lists; never a scored token

# Tactics that close or simplify goals by search rather than by citing a proof.
AUTOMATION = frozenset({
    "simp", "simp_all", "simpa", "dsimp", "simp_arith", "norm_num", "norm_num1",
    "grind", "omega", "linarith", "nlinarith", "positivity", "polyrith", "aesop",
    "tauto", "itauto", "decide", "native_decide", "ring", "ring_nf", "ring1",
    "field_simp", "abel", "abel_nf", "group", "noncomm_ring", "module",
    "bound", "fun_prop", "continuity", "measurability",
    "gcongr", "norm_cast", "push_cast", "assumption_mod_cast",
    "trivial", "rfl", "contradiction", "assumption", "cutsat", "lia", "order",
    "infer_instance", "mono", "compute_degree",
    "monicity", "subsingleton", "zify", "qify", "tfae_finish",
    "bv_decide", "bv_omega", "simp_rw", "nlinarith!",
    "linarith!", "positivity!", "simp!", "norm_num!", "field_simp!", "decide!",
    "and_intros", "constructorm", "casesm", "rcases?", "finiteness", "nontriviality",
    "monoidal", "coherence", "cat_disch", "aesop_cat", "fin_omega",
})
# Automation whose argument may be a hand-written proof: `exact_mod_cast h` is
# automation, `linear_combination h + c * two_eq_zero` is the proof itself.
# These count as automation only when the argument cites no lemma.
TERM_AUTOMATION = frozenset({"exact_mod_cast", "exacts", "linear_combination"})
# Routing: these steer goals but neither prove nor cite anything by themselves.
GLUE = frozenset({
    "all_goals", "any_goals", "try", "first", "focus", "next", "case", "repeat",
    "repeat'", "iterate", "intro", "intros", "rintro", "constructor", "skip", "done",
    "on_goal", "pick_goal", "swap", "rotate_left", "rotate_right", "classical",
    "exfalso", "left", "right", "ext", "funext", "by_contra", "by_contra!", "refine",
    "fun", "induction", "cases", "rcases", "obtain", "use", "exists", "split",
    "by_cases", "subst", "congr", "filter_upwards", "with", "at",
    # restructure or restate the goal without closing it
    "split_ifs", "interval_cases", "fin_cases", "push_neg", "unfold", "delta", "change",
    "show",
})
# Glue whose argument may itself be the proof: `refine foo ?_ h`, `use f x`.
# These are routing only when the argument is holes, binders and patterns.
TERM_GLUE = frozenset({"refine", "use", "exists", "rcases", "obtain", "cases",
                       "induction", "filter_upwards", "fun"})
SIMP_FAMILY = frozenset({"simp", "simp_all", "dsimp", "simpa", "norm_num", "field_simp",
                         "simp_arith", "simp!", "norm_num!"})
OPEN = {"(": ")", "[": "]", "{": "}", "⟨": "⟩", "⦃": "⦄"}
CLOSE = {v: k for k, v in OPEN.items()}
ARROWS = frozenset({"=>", "↦"})
_IDENT = re.compile(r"^[A-Za-z_À-῿℀-⅏\U0001D400-\U0001D7FF][\w'.!?À-῿℀-⅏\U0001D400-\U0001D7FF₀-₉]*$")
KEYWORDS = frozenset({"fun", "by", "at", "with", "using", "only", "from", "have", "show",
                      "let", "exact", "apply", "rw", "then", "else", "if", "do", "in",
                      "match", "calc", "this", "where", "λ", "Type", "Prop", "Sort"})


# --- tokens -------------------------------------------------------------------

def tokens(text: str) -> int:
    return sum(_token_count(line) for line in text.splitlines())


def token_list(lines: Sequence[str]) -> list[str]:
    """Scored tokens with a line sentinel between lines."""
    out: list[str] = []
    for line in lines:
        out.extend(lean_refactor_lexer(line.strip()).split())
        out.append(NL)
    return out


def scored(toks: Sequence[str]) -> int:
    return sum(1 for t in toks if t != NL)


def lcs(a: Sequence[str], b: Sequence[str]) -> int:
    """Longest common subsequence length, bit-parallel (Hyyro 2004)."""
    if not a or not b:
        return 0
    masks: dict[str, int] = collections.defaultdict(int)
    for index, item in enumerate(a):
        masks[item] |= 1 << index
    full = (1 << len(a)) - 1
    row = full
    for item in b:
        u = row & masks.get(item, 0)
        row = ((row + u) | (row - u)) & full
    return len(a) - bin(row).count("1")


def distance(a: Sequence[str], b: Sequence[str]) -> int:
    return len(a) + len(b) - 2 * lcs(a, b)


def is_ident(token: str) -> bool:
    return bool(_IDENT.match(token)) and token not in KEYWORDS


# --- hunks --------------------------------------------------------------------

def clean_lines(text: str) -> list[str]:
    return [line.rstrip() for line in remove_lean_comments(text).splitlines()]


def header_end(lines: Sequence[str]) -> int:
    """Index of the line on which the declaration's statement ends (its `:=`,
    `where` or first `|` clause); lines after it are proof."""
    depth = 0
    for index, line in enumerate(lines):
        stripped = line.strip()
        if index and (stripped.startswith("|") or stripped == "where") and depth == 0:
            return max(0, index - 1)
        for tok in lean_refactor_lexer(stripped).split():
            if tok in OPEN:
                depth += 1
            elif tok in CLOSE:
                depth = max(0, depth - 1)
            elif depth == 0 and tok in (":=", "where"):
                return index
    return len(lines) - 1


@dataclass
class Hunk:
    before: list[str]
    after: list[str]
    before_start: int            # 0-based line index into the declaration
    after_start: int = 0         # the hunk's lines [after_start, after_end) in the new text
    after_end: int = 0
    header: bool = False         # touches the statement, not just the proof
    syntax: int = 0              # tokens saved by accepted syntax rules
    rules: dict[str, int] = field(default_factory=dict)
    rewritten: list[str] = field(default_factory=list)
    automation: int = 0
    kind: str = "rewrite"        # rewrite | automation | syntax | partial_automation
    automated_runs: list = field(default_factory=list)  # (before steps, after steps, kind)
    upgrade: int = 0             # part of `automation` from automation upgrades
    pure: bool = False           # its declaration's whole proof became automation

    @property
    def delta(self) -> int:
        return tokens("\n".join(self.before)) - tokens("\n".join(self.after))

    @property
    def residual(self) -> int:
        return self.delta - self.syntax - self.automation


def hunks(before_text: str, after_text: str) -> list[Hunk]:
    b, a = clean_lines(before_text), clean_lines(after_text)
    b_head = header_end(b)
    key_b = [" ".join(x.split()) for x in b]
    key_a = [" ".join(x.split()) for x in a]
    matcher = difflib.SequenceMatcher(None, key_b, key_a, autojunk=False)
    found: list[Hunk] = []
    current: Hunk | None = None
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            if current is not None and all(not x for x in key_b[i1:i2]):
                continue  # blank lines do not separate hunks
            current = None
            continue
        if current is None:
            current = Hunk(before=[], after=[], before_start=i1, after_start=j1)
            found.append(current)
        current.before.extend(b[i1:i2])
        current.after.extend(a[j1:j2])
        current.after_end = j2
        if i1 <= b_head or (i1 == i2 and i1 <= b_head + 1):
            current.header = True
    return [h for h in found if tokens("\n".join(h.before)) or tokens("\n".join(h.after))]


# --- syntax rules -------------------------------------------------------------
# Each generator yields (rule, candidate, anchors): the token list with one
# rewrite applied, and groups of indices into the CURRENT list of which at least
# one token per group must survive into the after side.  The anchors are what
# makes a rule a re-spelling rather than a deletion: `have h : T := e` -> `have
# h := e` only counts if that `have` is still there afterwards.

Candidate = tuple[str, list[str], list[list[int]]]


def _match(toks: Sequence[str], start: int) -> int | None:
    """Index of the bracket closing the one opened at `start`."""
    depth = 0
    for index in range(start, len(toks)):
        tok = toks[index]
        if tok in OPEN:
            depth += 1
        elif tok in CLOSE:
            depth -= 1
            if depth == 0:
                return index
    return None


def _until(toks: Sequence[str], start: int, stops: set[str]) -> int | None:
    """First index >= start of a depth-0 token in `stops`."""
    depth = 0
    for index in range(start, len(toks)):
        tok = toks[index]
        if tok in OPEN:
            depth += 1
        elif tok in CLOSE:
            depth -= 1
            if depth < 0:
                return None
        elif depth == 0 and tok in stops:
            return index
    return None


def _depths(toks: Sequence[str]) -> Iterator[tuple[str, int]]:
    depth = 0
    for tok in toks:
        if tok in CLOSE:
            depth -= 1
        yield tok, depth
        if tok in OPEN:
            depth += 1


def _real(indices: Iterable[int], toks: Sequence[str]) -> list[int]:
    return [i for i in indices if 0 <= i < len(toks) and toks[i] != NL]


def _kept_items(items: list[str], after_tokens: set[str]) -> list[str] | None:
    """The simp-set entries the after side still names, comma-separated."""
    groups: list[list[str]] = [[]]
    for tok, depth in _depths(items):
        if tok == "," and depth == 0:
            groups.append([])
        else:
            groups[-1].append(tok)
    kept = [g for g in groups if g and all(t in after_tokens for t in g if is_ident(t))]
    if not kept or len(kept) == len(groups):
        return None
    out: list[str] = []
    for g in kept:
        out.extend(([","] if out else []) + g)
    return out


def rule_simp_only(toks: list[str], ctx) -> Iterator[Candidate]:
    """`simp only [xs]` -> `simp`, or -> `simp [xs]`."""
    for i, tok in enumerate(toks):
        if tok in SIMP_FAMILY and i + 1 < len(toks) and toks[i + 1] == "only":
            if i + 2 < len(toks) and toks[i + 2] == "[":
                end = _match(toks, i + 2)
                if end is not None:
                    yield "simp_only_list", toks[:i + 1] + toks[end + 1:], [[i]]
                    kept = _kept_items(toks[i + 3:end], ctx["after_tokens"])
                    if kept is not None:
                        yield ("simp_only_list", toks[:i + 1] + ["["] + kept + ["]"] + toks[end + 1:],
                               [[i]])
            yield "simp_only_keyword", toks[:i + 1] + toks[i + 2:], [[i]]


def rule_have_type(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`have h : T := e` -> `have h := e` (also let/obtain/set/replace)."""
    for i, tok in enumerate(toks):
        if tok not in ("have", "haveI", "let", "letI", "obtain", "set", "replace"):
            continue
        j = i + 1
        if j < len(toks) and toks[j] != ":":
            if toks[j] in OPEN:
                close = _match(toks, j)
                if close is None:
                    continue
                j = close + 1
            else:
                j += 1
        if j < len(toks) and toks[j] == ":":
            end = _until(toks, j + 1, {":=", NL, "with", "by"})
            if end is not None and toks[end] == ":=":
                yield "have_type", toks[:j] + toks[end:], [[i], [end]]


def rule_show(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`show T from e` -> `e`; a bare `show T` step dropped."""
    for i, tok in enumerate(toks):
        if tok != "show":
            continue
        end = _until(toks, i + 1, {"from", NL, ";", "by"})
        if end is None:
            continue
        if toks[end] == "from":
            yield "show_type", toks[:i] + toks[end + 1:], [_real([end + 1], toks)]
        elif toks[end] in (NL, ";") and (i == 0 or toks[i - 1] in (NL, ";", "·")):
            # the step before and the step after must both survive
            yield "show_type", toks[:i] + toks[end + 1:], [
                _real(range(max(0, i - 3), i), toks), _real(range(end + 1, end + 4), toks)]


def rule_merge(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`rw [a]` then `rw [b]` -> `rw [a, b]`; `intro x` then `intro y` -> `intro x y`."""
    for i, tok in enumerate(toks):
        if tok in ("rw", "rewrite", "simp_rw") and i + 1 < len(toks) and toks[i + 1] == "[":
            close = _match(toks, i + 1)
            loc_end = None if close is None else _until(toks, close + 1, {NL, ";", "<;>"})
            if loc_end is None:
                continue
            loc = toks[close + 1:loc_end]
            k = loc_end + 1
            if k + 1 >= len(toks) or toks[k] not in (tok, "rwa") or toks[k + 1] != "[":
                continue
            if tok == "simp_rw" and toks[k] != "simp_rw":
                continue
            close2 = _match(toks, k + 1)
            if close2 is None:
                continue
            end2 = _until(toks, close2 + 1, {NL, ";", "<;>"})
            end2 = len(toks) if end2 is None else end2
            if toks[close2 + 1:end2] != loc:
                continue
            merged = [toks[k], "["] + toks[i + 2:close] + [","] + toks[k + 2:close2] + ["]"] + loc
            yield "merge_rw", toks[:i] + merged + toks[end2:], [
                _real(range(i + 2, close), toks), _real(range(k + 2, close2), toks)]
        if tok in ("intro", "intros", "rintro"):
            end = _until(toks, i + 1, {NL, ";", "<;>"})
            if end is None or end + 1 >= len(toks) or toks[end + 1] != tok:
                continue
            yield "merge_intro", toks[:end] + toks[end + 2:], [[i]]


def rule_by_exact(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`by exact e` / `by refine ⟨..⟩` / `by apply f ..` -> the term; `exact rfl` -> `rfl`."""
    for i, tok in enumerate(toks):
        if tok == "by":
            j = i + 1
            while j < len(toks) and toks[j] == NL:
                j += 1
            if j < len(toks) and toks[j] in ("exact", "refine", "apply"):
                yield "by_exact", toks[:i] + toks[j + 1:], [_real(range(j + 1, j + 4), toks)]
        if tok in ("exact", "apply") and i + 1 < len(toks) and toks[i + 1] in ("rfl", "trivial"):
            yield "exact_rfl", toks[:i] + toks[i + 1:], [[i + 1]]


def rule_parens(toks: list[str], _ctx) -> Iterator[Candidate]:
    """Redundant parentheses around a term (not binders, tuples or named args)."""
    for i, tok in enumerate(toks):
        if tok != "(":
            continue
        close = _match(toks, i)
        if close is None or close == i + 1:
            continue
        inner = toks[i + 1:close]
        top = [t for t, d in _depths(inner) if d == 0]
        if any(t in (",", ":", ":=") for t in top):
            continue
        yield "redundant_parens", toks[:i] + inner + toks[close + 1:], [
            _real(range(i + 1, close), toks), _real(range(max(0, i - 2), i), toks)]


def rule_named_args(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`f (x := a) b` -> `f b`; `@f _ _ b` -> `f b`."""
    for i, tok in enumerate(toks):
        if tok == "(" and i + 2 < len(toks) and is_ident(toks[i + 1]) and toks[i + 2] == ":=":
            close = _match(toks, i)
            if close is not None and i > 0:
                yield "named_arg", toks[:i] + toks[close + 1:], [_real([i - 1], toks)]
        if tok == "@" and i + 1 < len(toks):
            j = i + 2
            yield "explicit_at", toks[:i] + toks[i + 1:], [[i + 1]]
            while j < len(toks) and toks[j] == "_":
                j += 1
                yield "explicit_at", toks[:i] + toks[i + 1:i + 2] + toks[j:], [[i + 1]]


def rule_dot_notation(toks: list[str], ctx) -> Iterator[Candidate]:
    """`Foo.bar_baz h x` -> `h.baz x` / `h.bar_baz x`, when the after side has it."""
    after_tokens: set[str] = ctx["after_tokens"]
    for i, tok in enumerate(toks[:-1]):
        arg = toks[i + 1]
        if not is_ident(tok) or not is_ident(arg) or "." in arg:
            continue
        last = tok.rsplit(".", 1)[-1]
        for candidate in after_tokens:
            if not candidate.startswith(arg + "."):
                continue
            field_name = candidate[len(arg) + 1:]
            if field_name and (last == field_name or last.endswith("_" + field_name)):
                yield "dot_notation", toks[:i] + [candidate] + toks[i + 2:], []


def rule_eta(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`fun x => f x` -> `f`."""
    for i, tok in enumerate(toks):
        if tok not in ("fun", "λ") or i + 3 >= len(toks):
            continue
        var, arrow = toks[i + 1], toks[i + 2]
        if not is_ident(var) or arrow not in ARROWS:
            continue
        j, depth = i + 3, 0
        while j < len(toks):
            t = toks[j]
            if t in OPEN:
                depth += 1
            elif t in CLOSE:
                if depth == 0:
                    break
                depth -= 1
            elif depth == 0 and t in (NL, ",", ";"):
                break
            j += 1
        if j - 1 > i + 3 and toks[j - 1] == var and var not in toks[i + 3:j - 1]:
            yield "eta", toks[:i] + toks[i + 3:j - 1] + toks[j:], [_real(range(i + 3, j - 1), toks)]


def rule_constructor(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`refine ⟨?_, ?_⟩` / `apply And.intro` -> `constructor`."""
    for i, tok in enumerate(toks):
        if tok in ("refine", "refine'") and i + 1 < len(toks) and toks[i + 1] == "⟨":
            close = _match(toks, i + 1)
            if close is None:
                continue
            inner = [t for t in toks[i + 2:close] if t != ","]
            if inner and all(t in ("?_", "?", "_") for t in inner):
                yield "refine_constructor", toks[:i] + ["constructor"] + toks[close + 1:], []
        if tok == "apply" and i + 1 < len(toks) and toks[i + 1] in ("And.intro", "Iff.intro"):
            yield "refine_constructor", toks[:i] + ["constructor"] + toks[i + 2:], []


def rule_bullets(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`· t` repeated k times -> `<;> t` on the previous step (or `all_goals t`)."""
    lines: list[list[str]] = [[]]
    for tok in toks:
        if tok == NL:
            lines.append([])
        else:
            lines[-1].append(tok)
    for start in range(len(lines)):
        if not lines[start] or lines[start][0] != "·":
            continue
        body = lines[start][1:]
        end = start + 1
        while end < len(lines) and lines[end] == lines[start]:
            end += 1
        if end - start < 2 or not body:
            continue
        prefix = sum(len(x) + 1 for x in lines[:start])
        suffix = sum(len(x) + 1 for x in lines[:end])
        body_index = list(range(prefix + 1, prefix + 1 + len(body)))
        if start > 0 and lines[start - 1]:
            yield ("bullets_combinator", toks[:prefix - 1] + ["<;>"] + body + [NL] + toks[suffix:],
                   [body_index])
        yield ("bullets_combinator", toks[:prefix] + ["all_goals"] + body + [NL] + toks[suffix:],
               [body_index])


def rule_binders(toks: list[str], ctx) -> Iterator[Candidate]:
    """A statement binder dropped because it now comes from a `variable` command
    in the same file: the statement is the same, only spelled once."""
    if not ctx["header"]:
        return
    hoisted: set[str] = ctx["variable_binders"]
    stop = next((i for i, t in enumerate(toks) if t in (":=", "where")), len(toks))
    # hoisted into `variable` AND still used by the declaration
    hoisted = {h for h in hoisted if h in ctx["after_decl_idents"] or " " in h}
    singles = list(_binder_spans(toks, stop, hoisted))
    if len(singles) > 1:
        keep = [t for k, t in enumerate(toks) if not any(a <= k <= b for a, b in singles)]
        yield "binder_hoisted", keep, []
    for a, b in singles:
        yield "binder_hoisted", toks[:a] + toks[b + 1:], [_real(range(max(0, a - 2), a), toks)]


def _binder_spans(toks: list[str], stop: int, hoisted: set[str]) -> Iterator[tuple[int, int]]:
    for i, tok in enumerate(toks[:stop]):
        if tok not in ("(", "{", "[", "⦃"):
            continue
        close = _match(toks, i)
        if close is None or close > stop:
            continue
        inner = toks[i + 1:close]
        if tok == "[":
            key = " ".join(inner)
            if key in hoisted:
                yield i, close
            continue
        if ":" not in inner:
            continue
        names = inner[:inner.index(":")]
        if names and all(is_ident(n) and n in hoisted for n in names):
            yield i, close


def _in_stated_type(toks: Sequence[str], index: int) -> bool:
    start = index
    while start > 0 and toks[start - 1] != NL:
        start -= 1
    head = toks[start] if start < len(toks) else ""
    if head in ("·",) and start + 1 < len(toks):
        start, head = start + 1, toks[start + 1]
    if head not in ("have", "haveI", "let", "show", "suffices", "obtain", "replace"):
        return False
    end = _until(toks, start + 1, {":=", "from", "by", NL})
    return end is None or index < end


def rule_ascription(toks: list[str], ctx) -> Iterator[Candidate]:
    """`(e : T)` -> `e` inside a proof term.

    Not inside the stated type of a `have`/`let`/`show`/`suffices`: there the
    ascription picks the type the claim is about -- `0 < (k.factorial : ℝ)` and
    `0 < k.factorial` are different propositions -- so dropping it is a
    different intermediate fact, not a re-spelling."""
    if ctx["header"]:
        return
    for i, tok in enumerate(toks):
        if tok != "(" or _in_stated_type(toks, i):
            continue
        close = _match(toks, i)
        if close is None:
            continue
        inner = toks[i + 1:close]
        top = [k for k, (t, d) in enumerate(_depths(inner)) if d == 0 and t == ":"]
        if len(top) != 1 or top[0] == 0:
            continue
        colon = top[0]
        term = inner[:colon]
        if any(t in (",", ":=") for t, d in _depths(term) if d == 0):
            continue
        keep = [_real(range(i + 1, i + 1 + colon), toks)]
        yield "type_ascription", toks[:i] + ["("] + term + [")"] + toks[close + 1:], keep
        yield "type_ascription", toks[:i] + term + toks[close + 1:], keep


BINDER_HEADS = frozenset({"∀", "∃", "∃!", "∑", "∏", "⋃", "⋂", "⨆", "⨅", "fun", "λ", "∫"})


def rule_binder_type(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`∑ x : T, f x` -> `∑ x, f x`; `fun (x : T) => ..` -> `fun x => ..`;
    `[inst : C α]` -> `[C α]`."""
    for i, tok in enumerate(toks):
        if tok in BINDER_HEADS:
            j = i + 1
            while j < len(toks) and is_ident(toks[j]):
                j += 1
            if j > i + 1 and j < len(toks) and toks[j] == ":":
                end = _until(toks, j + 1, {",", "=>", "↦", "∈", NL})
                if end is not None and toks[end] in (",", "=>", "↦"):
                    yield "binder_type", toks[:j] + toks[end:], [[i], [end]]
            if j == i + 1 and j < len(toks) and toks[j] == "(":
                close = _match(toks, j)
                if close is not None and ":" in toks[j + 1:close]:
                    colon = toks.index(":", j + 1)
                    names = toks[j + 1:colon]
                    if names and all(is_ident(n) for n in names):
                        yield "binder_type", toks[:j] + names + toks[close + 1:], [[i]]
        if tok == "[" and i + 2 < len(toks) and is_ident(toks[i + 1]) and toks[i + 2] == ":":
            yield "binder_type", toks[:i + 1] + toks[i + 3:], [[i]]


def rule_cdot(toks: list[str], _ctx) -> Iterator[Candidate]:
    """`(fun x => f x y)` -> `(f · y)`: the variable used once becomes `·`."""
    for i, tok in enumerate(toks):
        if tok not in ("fun", "λ") or i == 0 or toks[i - 1] != "(" or i + 3 >= len(toks):
            continue
        var, arrow = toks[i + 1], toks[i + 2]
        if not is_ident(var) or arrow not in ARROWS:
            continue
        close = _match(toks, i - 1)
        if close is None:
            continue
        body = toks[i + 3:close]
        if body.count(var) != 1 or any(t in ("fun", "λ", "·") for t in body):
            continue
        body = [("·" if t == var else t) for t in body]
        yield "cdot", toks[:i] + body + toks[close:], [_real([i - 1, close], toks)]


def rule_named_args_run(toks: list[str], _ctx) -> Iterator[Candidate]:
    """Several named arguments dropped from one application at once."""
    i = 0
    while i < len(toks):
        if toks[i] == "(" and i + 2 < len(toks) and is_ident(toks[i + 1]) and toks[i + 2] == ":=":
            start, end = i, i
            while (end < len(toks) and toks[end] == "(" and end + 2 < len(toks)
                   and is_ident(toks[end + 1]) and toks[end + 2] == ":="):
                close = _match(toks, end)
                if close is None:
                    break
                end = close + 1
            if end > start and start > 0:
                yield "named_arg", toks[:start] + toks[end:], [_real([start - 1], toks)]
            i = max(end, i + 1)
        else:
            i += 1


RULES: tuple[Callable, ...] = (
    rule_simp_only, rule_have_type, rule_show, rule_merge, rule_by_exact,
    rule_named_args, rule_dot_notation, rule_eta, rule_constructor, rule_bullets,
    rule_binders, rule_ascription, rule_binder_type, rule_cdot, rule_named_args_run,
    rule_parens,
)


def matched_flags(current: Sequence[str], after: Sequence[str]) -> list[bool]:
    """Which tokens of `current` an alignment with `after` keeps."""
    flags = [False] * len(current)
    matcher = difflib.SequenceMatcher(None, current, after, autojunk=False)
    for block in matcher.get_matching_blocks():
        for k in range(block.a, block.a + block.size):
            flags[k] = current[k] != NL
    return flags


def joined(current: Sequence[str], cand: Sequence[str], after_flat: Sequence[str]) -> bool:
    """The seam a rewrite creates lies inside one matched run of the after side.

    Deleting `: T` from `have h : T := e` joins `h` to `:=`; the rewrite is only
    evidence of a re-spelling if `h :=` sits, with a neighbour either side, in a
    run of tokens the after side shares.  A single shared token elsewhere in a
    heavily rewritten hunk is not enough.
    """
    prefix = 0
    limit = min(len(current), len(cand))
    while prefix < limit and current[prefix] == cand[prefix]:
        prefix += 1
    suffix = 0
    while (suffix < limit - prefix
           and current[len(current) - 1 - suffix] == cand[len(cand) - 1 - suffix]):
        suffix += 1
    flat = [t for t in cand if t != NL]
    lo = sum(1 for t in cand[:prefix] if t != NL) - 1          # last kept token before
    hi = len(flat) - sum(1 for t in cand[len(cand) - suffix:] if t != NL)  # first after
    lo, hi = max(lo, 0), min(hi, len(flat) - 1)
    if lo == hi:
        lo, hi = max(lo - 1, 0), min(hi + 1, len(flat) - 1)
    if hi <= lo:
        return False
    matcher = difflib.SequenceMatcher(None, flat, after_flat, autojunk=False)
    for block in matcher.get_matching_blocks():
        if (block.size >= min(3, len(flat), len(after_flat))
                and block.a <= lo and hi < block.a + block.size):
            return True
    return False


def apply_syntax(hunk: Hunk, variable_binders: set[str], after_decl_idents: set[str] = frozenset(),
                 *, max_steps: int = 200) -> None:
    before = token_list(hunk.before)
    after = token_list(hunk.after)
    if len(before) > 3000 or len(after) > 3000:
        hunk.rewritten = hunk.before
        return
    after_flat = [t for t in after if t != NL]
    ctx = {"header": hunk.header, "after_tokens": set(after),
           "variable_binders": variable_binders, "after_decl_idents": after_decl_idents}
    current = before
    d_current = distance(current, after)
    flags = matched_flags(current, after)
    steps = 0
    progress = True
    while progress and steps < max_steps and d_current:
        progress = False
        for rule in RULES:
            for name, cand, anchors in rule(current, ctx):
                saved = scored(current) - scored(cand)
                if saved <= 0:
                    continue
                if any(not any(flags[k] for k in group) for group in anchors):
                    continue
                if not joined(current, cand, after_flat):
                    continue
                d_new = distance(cand, after)
                if distance(current, cand) + d_new != d_current:
                    continue
                hunk.rules[name] = hunk.rules.get(name, 0) + saved
                hunk.syntax += saved
                current, d_current = cand, d_new
                flags = matched_flags(current, after)
                steps += 1
                progress = True
                break
            if progress:
                break
    hunk.rewritten = _reindent([" ".join(x) for x in _split_lines(current)], hunk.before) \
        if hunk.syntax else list(hunk.before)
    if hunk.syntax and d_current == 0:
        hunk.kind = "syntax"


def _split_lines(toks: Sequence[str]) -> list[list[str]]:
    lines: list[list[str]] = [[]]
    for tok in toks:
        if tok == NL:
            lines.append([])
        else:
            lines[-1].append(tok)
    return [x for x in lines if x]


# --- automation ---------------------------------------------------------------

def _reindent(lines: list[str], original: Sequence[str]) -> list[str]:
    """Give rewritten lines the indentation of the lines they came from, when
    the rewrite kept the line count (step splitting reads indentation)."""
    source = [line for line in original if line.strip()]
    if len(source) != len(lines):
        return lines
    return [src[:len(src) - len(src.lstrip())] + line for src, line in zip(source, lines)]


def _proof_lines(lines: Sequence[str], header: bool) -> list[str]:
    """The proof part of a hunk's lines: for a header hunk, what follows `:=`.
    Lines keep their indentation."""
    if not header:
        return list(lines)
    depth = 0
    for index, line in enumerate(lines):
        toks = lean_refactor_lexer(line.strip()).split()
        for k, tok in enumerate(toks):
            if tok in OPEN:
                depth += 1
            elif tok in CLOSE:
                depth -= 1
            elif depth == 0 and tok == ":=":
                rest = " ".join(toks[k + 1:])
                indent = line[:len(line) - len(line.lstrip())]
                return ([indent + rest] if rest else []) + list(lines[index + 1:])
    return []


# A line continues the step above it when that line ended mid-term, or when it
# opens with something no tactic starts with.
_CONTINUE_AFTER = frozenset({"using", ":=", ",", "←", "at", "+", "-", "*", "/", "=", "≤", "<",
                             "≥", ">", "∧", "∨", "→", "↔", "∘", "•", "^", "$", "|>.", "<|"})
_CONTINUE_WITH = frozenset({"(", "[", "⟨", "+", "*", "/", "=", "≤", "<", "≥", ">", "∧", "∨",
                            "→", "↔", ",", ")", "]", "⟩", "•", "^", "∘", "$"})
_BLOCK_OPENERS = frozenset({"by", "=>", "·", "with", "do", "then", "else"})


def _steps(lines: Sequence[str]) -> Iterator[list[str]]:
    """Tactic steps: split at `;`, `<;>`, `·`, `by`, `=>` and line breaks.

    A line break does not split a step when the step is inside brackets, when
    the line above ended mid-term (`simpa [..] using`, `have h : T :=`, a
    trailing operator), when the new line starts with something no tactic
    starts with (`(`, `⟨`, `+` ...), or when it is indented past the column
    the step started at -- Lean's own rule for an argument continuing on the
    next line.  Without this, `simpa [..] using` + `  foo_le h` reads as a
    manual step `foo_le h`."""
    step: list[str] = []
    step_col = 0
    depth = 0
    last = ""
    for raw in lines:
        toks = lean_refactor_lexer(raw.strip()).split()
        if not toks:
            continue
        indent = len(raw) - len(raw.lstrip())
        continues = bool(step) and (
            depth > 0 or last in _CONTINUE_AFTER or toks[0] in _CONTINUE_WITH
            or (indent > step_col and last not in _BLOCK_OPENERS and toks[0] not in ("·", "|")))
        if not continues and step:
            yield step
            step = []
        col = indent
        for tok in toks:
            if tok in OPEN:
                depth += 1
            elif tok in CLOSE:
                depth = max(0, depth - 1)
            if depth == 0 and tok in (";", "<;>", "·", "by", "=>", "|", "{", "}"):
                if step:
                    yield step
                step = []
                col = indent + 2 if tok == "·" else indent
                continue
            if not step:
                step_col = col
            step.append(tok)
        last = toks[-1]
    if step:
        yield step


_LOCAL = re.compile(r"^(?:h|H|this|ih|hyp|_)[\w'₀-₉!?]*(?:\.[\w'₀-₉]+)*$|^[A-Za-zα-ωΑ-Ω][₀-₉'0-9]*(?:\.[\w'₀-₉]+)*$")
_TACTIC_WORDS = AUTOMATION | TERM_AUTOMATION | GLUE | frozenset({
    "exact", "apply", "rw", "rewrite", "erw", "rwa", "calc", "have", "show", "change",
    "unfold", "delta", "conv", "refine'", "specialize", "let", "set", "suffices",
    "exists", "at", "with", "using", "only", "from", "fun", "λ", "then", "else", "if",
    "in", "true", "false", "True", "False", "Type", "Prop", "rfl", "id", "calc",
    "replace", "generalize", "clear", "revert", "exacts", "nth_rw", "nth_rewrite",
})


def citations(step: Sequence[str]) -> set[str]:
    """What the step names that looks like a lemma: not a tactic, not a local
    hypothesis, and named like one -- Lean and Mathlib lemma names carry `_` or
    `.` (`le_trans`, `Nat.succ_le`), local variables usually don't (`lambda`,
    `hscale`, `u`), so `linear_combination lambda ^ 10 * h1` cites nothing."""
    return {tok for tok in step
            if is_ident(tok) and len(tok) > 1 and tok not in _TACTIC_WORDS
            and not _LOCAL.match(tok) and ("_" in tok or "." in tok)}


def cites_lemma(step: Sequence[str]) -> bool:
    return bool(citations(step))


def _routing_only(step: Sequence[str]) -> bool:
    """A glue step that carries no term: binders, patterns and holes only."""
    if step[0] not in TERM_GLUE:
        return True
    return not cites_lemma(step[1:])


_STATES = frozenset({"have", "haveI", "obtain", "suffices", "show", "replace"})


def _states_a_goal(step: Sequence[str]) -> bool:
    """`have h : T :=` (its proof follows after `by`) or `have h : T` (a new goal):
    stating what to prove is routing; the proof is judged as its own steps.
    `have h := foo x` carries its proof as a term and stays manual."""
    if step[0] not in _STATES:
        return False
    depth = 0
    for index, tok in enumerate(step):
        if tok in OPEN:
            depth += 1
        elif tok in CLOSE:
            depth -= 1
        elif depth == 0 and tok == ":=":
            return index == len(step) - 1
    return True


# Tactic macros a repository defines whose expansion is automation only
# (`macro "∎" : tactic => `(tactic| all_goals first | rfl | omega | grind)`):
# set per repository by abbreviations.automation_macros, read by step labels.
EXTRA_AUTOMATION: set[str] = set()

# Combinators that run the tactic they wrap: `all_goals grind` is automation.
_COMBINATORS = frozenset({"all_goals", "any_goals", "try", "repeat", "repeat'", "focus"})


def step_is_automation(step: Sequence[str]) -> bool | None:
    """True for automation, None for routing, False for a manual step."""
    if not step:
        return None
    if _states_a_goal(step):
        return None
    head = step[0]
    if head in _COMBINATORS and len(step) > 1:
        return step_is_automation(step[1:])
    if head == "iterate" and len(step) > 2:
        return step_is_automation(step[2:])
    if head in AUTOMATION or head in EXTRA_AUTOMATION:
        return True
    if head in TERM_AUTOMATION:
        return not cites_lemma(step[1:])
    if head in GLUE and _routing_only(step):
        return None
    return False


def automatic(lines: Sequence[str]) -> bool:
    """Every step is automation or routing, and at least one is automation."""
    saw = False
    for step in _steps(lines):
        verdict = step_is_automation(step)
        if verdict is False:
            return False
        saw = saw or bool(verdict)
    return saw


def manual(lines: Sequence[str], still_named: set[str] = frozenset()) -> bool:
    """Some step did work by hand, citing a lemma the new side no longer names.

    `exact h` does not count -- closing a goal with a hypothesis is not relying
    on a theorem -- and neither does `rw [foo]` becoming `simp [foo]`: the proof
    still rests on `foo`, only the tactic applying it changed."""
    return any(step_is_automation(step) is False and citations(step) - still_named
               for step in _steps(lines))


def automation_verdict(old: Sequence[list[str]], new: Sequence[list[str]],
                       cited_before: set[str],
                       definitions: frozenset[str] | set[str] = frozenset(),
                       lemmas: frozenset[str] | set[str] | None = None) -> str | None:
    """Whether the steps `new` replaced `old` by automation.

    `replace`: the old steps did work by hand and the new ones are automation
    and routing only.  `upgrade`: the old steps were automation already and
    collapsed into fewer calls, or into a different, stronger tactic, with
    fewer tokens.  None otherwise -- including when the new steps hand
    automation a lemma the old proof never used (a new argument), and when no
    automation came in at all (`field_simp [..]; ring` -> `field_simp [..]`
    is pruning).  Unfolding a definition is not a new lemma: neither one whose
    equation lemmas the old proof cited (`rw [houseM_zero]` -> `simp_all
    [houseM]`) nor any of the repository's own definitions.

    `lemmas`, the short names of the repository's theorems, narrows "new lemma"
    to a new repository dependency -- the thing that makes a change structural.
    A library hint (`simp_all [ne_comm]`) adds no dependency and does not
    count.  Without it every unknown name counts."""
    def known(name: str) -> bool:
        if lemmas is not None and name.rsplit(".", 1)[-1] not in lemmas:
            return True
        return (name in cited_before or name.rsplit(".", 1)[-1] in definitions or any(
            c.startswith(name + "_") or c.startswith(name + ".") for c in cited_before))

    labels = [step_is_automation(step) for step in new]
    if not new or False in labels or True not in labels:
        return None
    # only what is handed to automation can be a new lemma; a stated subgoal
    # names functions and types, not lemmas
    def named(step: Sequence[str]) -> set[str]:
        found = citations(step)
        if lemmas is not None:  # a repository theorem is a lemma whatever its name
            found |= {t for t in step if is_ident(t) and t.rsplit(".", 1)[-1] in lemmas}
        return found

    if any(not known(c) for step, label in zip(new, labels) if label for c in named(step)):
        return None
    kept = {_bare(step) for step in old}
    if all(_bare(step) in kept for step, label in zip(new, labels) if label):
        return None
    if any(step_is_automation(step) is False for step in old):
        return "replace"
    old_calls = [step for step in old if step_is_automation(step)]
    new_calls = [step for step, label in zip(new, labels) if label]
    fewer = len(new_calls) < len(old_calls)
    different = bool({x[0] for x in new_calls} - {x[0] for x in old_calls})
    if old_calls and (fewer and len(old_calls) >= 2 or different) \
            and scored_steps(old) > scored_steps(new):
        return "upgrade"
    return None


def proof_steps(hunk: Hunk) -> tuple[list[list[str]], list[list[str]]]:
    """A hunk's proof steps, old side after the syntax rules; statements and
    separators are not steps, so nothing charged from them can overlap syntax."""
    return (list(_steps(_proof_lines(hunk.rewritten or hunk.before, hunk.header))),
            list(_steps(_proof_lines(hunk.after, hunk.header))))


def classify_automation(hunk: Hunk, definitions: frozenset[str] | set[str] = frozenset(),
                        known: frozenset[str] | set[str] = frozenset(),
                        lemmas: frozenset[str] | set[str] | None = None,
                        still_named: set[str] | None = None,
                        neighbours: tuple[list[str] | None, list[str] | None] = (None, None)) -> None:
    """Charge to automation the proof steps done by hand -- lemma applications,
    rewriting chains, calc blocks, case splits closed step by step -- that
    collapsed into automation.

    Only proof steps are ever charged: the old steps' tokens minus the new
    steps' tokens, plus the helper claims that went with them
    (`charged_steps`).  The declaration's statement, and whatever notation
    shortened it, stays with syntax and the other categories.

    The whole hunk is tried first; otherwise its steps are aligned (edited but
    similar steps anchor) and each replaced run is tested on its own.  A run
    deleted outright is `absorbed` when the new step next to the gap is
    automation that now does its work -- unless its lemmas are still named in
    the new proof (`still_named`, the new declaration's identifiers), which
    means they were inlined, not automated."""
    if hunk.kind == "syntax":
        return
    old_steps, new_steps = proof_steps(hunk)
    if not old_steps:
        return
    cited_before = {t for t in token_list(hunk.before) if is_ident(t)} | set(known)
    whole = automation_verdict(old_steps, new_steps, cited_before, definitions, lemmas) if new_steps else None
    runs = [(old_steps, new_steps, whole)] if whole else []
    named = still_named or set()
    for old, new, start, end in ([] if whole else _aligned_runs_at(old_steps, new_steps)):
        kind = automation_verdict(old, new, cited_before, definitions, lemmas)
        adjacent = _absorber(new_steps, start, end, neighbours) is not None
        # dropping `<;> ring` after `field_simp` is pruning: only hand work or
        # a helper claim can be absorbed
        handwork = any(step_is_automation(x) is False or _claim_key(x) is not None for x in old)
        if kind or not handwork:
            pass
        elif not new:
            # deleted outright next to automation
            if adjacent and not _inlined(old, named) and charged_steps(old, []) > 0:
                kind = "absorbed"
        elif all(_inlining_target(x, old, cited_before) for x in new
                 if step_is_automation(x) is False):
            # partly inlined into the new hand steps; the old hand steps
            # whose lemmas the new proof no longer names went into the
            # automation beside or inside the run, net of any call it added
            calls = [x for x in new if step_is_automation(x) is True]
            seen = {_bare(x) for x in old}
            fresh = [x for x in calls if _bare(x) not in seen]
            gone = _dropped(old, named)
            if gone and (adjacent or calls) and charged_steps(gone, fresh) > 0:
                old, new, kind = gone, fresh, "absorbed"
        if kind:
            runs.append((old, new, kind))
    room = hunk.delta - hunk.syntax
    for old, new, kind in runs:
        value = charged_steps(old, new)
        if kind == "absorbed":
            # an estimate from steps, so never more than the hunk has left
            value = max(0, min(value, room - hunk.automation))
        hunk.automation += value
        if kind == "upgrade":
            hunk.upgrade += value
        hunk.automated_runs.append((list(old), list(new), kind))
    if runs:
        hunk.kind = "automation" if whole else "partial_automation"


def pure_automation(before_text: str, after_text: str, found: list[Hunk],
                    definitions: frozenset[str] | set[str] = frozenset(),
                    lemmas: frozenset[str] | set[str] | None = None) -> bool:
    """The declaration's proof as a whole was replaced by automation.

    Judged on the entire proof, not hunk by hunk: before, it did some work by
    hand; after, it is automation and routing only, naming no new lemma.  Then
    every hunk's proof-step difference is automation, however the line diff
    happened to cut the proof up."""
    old = list(_steps(_proof_lines(clean_lines(before_text), True)))
    new = list(_steps(_proof_lines(clean_lines(after_text), True)))
    cited = {t for t in token_list(clean_lines(before_text)) if is_ident(t)}
    kind = automation_verdict(old, new, cited, definitions, lemmas)
    if not kind:
        return False
    for hunk in found:
        old_steps, new_steps = proof_steps(hunk)
        value = charged_steps(old_steps, new_steps)
        hunk.automation = value
        hunk.upgrade = value if kind == "upgrade" else 0
        hunk.automated_runs = [(old_steps, new_steps, kind)] if old_steps or new_steps else []
        hunk.kind = "automation"
        hunk.pure = True
    return True


def _bare(step: Sequence[str]) -> str:
    """A step without the stray closers a step split can leave on its end."""
    toks = list(step)
    while toks and toks[-1] in CLOSE and toks.count(toks[-1]) > toks.count(CLOSE[toks[-1]]):
        toks.pop()
    return " ".join(toks)


def scored_steps(steps: Sequence[list[str]]) -> int:
    """Tokens of the steps that do work -- manual and automation.  Routing
    (a stated subgoal, `intro`, `constructor`) is not charged: a `have`
    statement that got shorter through a new definition is not automation."""
    return sum(len(step) for step in steps if step_is_automation(step) is not None)


def _claim_key(step: Sequence[str]) -> str | None:
    """The name a stated claim binds (`have h : T :=` -> `h`, `obtain ⟨a, b⟩ :
    T` -> `⟨ a , b ⟩`, `have : T` -> `this`); None for any other step."""
    if not step or step[0] not in _CLAIM or not _states_a_goal(step):
        return None
    head = _until(step, 1, {":", ":="})
    return " ".join(step[1:head]) or "this"


def charged_steps(old: Sequence[list[str]], new: Sequence[list[str]]) -> int:
    """What collapsing `old` into `new` saved: the steps that do work, plus the
    helper claims the old steps stated and the new ones no longer do.  A claim
    stated on both sides (a `have hdiff : T := by` whose proof became `by
    nlinarith`) is not charged, so a statement re-spelled through a new
    definition stays out of it; one that is gone was part of the work the
    automation took over."""
    old_claims = {k: len(s) for s in old if (k := _claim_key(s)) is not None}
    new_claims = {k: len(s) for s in new if (k := _claim_key(s)) is not None}
    return (scored_steps(old) - scored_steps(new)
            + sum(v for k, v in old_claims.items() if k not in new_claims)
            - sum(v for k, v in new_claims.items() if k not in old_claims))


def _similar(a: Sequence[str], b: Sequence[str], threshold: float = 0.6) -> bool:
    """Same tactic, mostly the same arguments: `exact baz h` ~ `exact baz' h`."""
    return (bool(a) and bool(b) and a[0] == b[0]
            and difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= threshold)


def _aligned_runs(old: list[list[str]], new: list[list[str]]) -> Iterator[tuple[list, list]]:
    """Runs of old steps and the new steps that replaced them.

    Exact step matches split the two sequences first; inside a replaced
    stretch, a step that survived with small edits -- an inlined `have`, a
    `rw` that lost one lemma -- is an anchor too, so the automation next to it
    is judged on its own instead of failing with the edited step."""
    for a, b, _, _ in _aligned_runs_at(old, new):
        yield a, b


def _aligned_runs_at(old: list[list[str]], new: list[list[str]]) -> Iterator[tuple[list, list, int, int]]:
    """`_aligned_runs`, with each run's slice [start, end) of `new`."""
    matcher = difflib.SequenceMatcher(None, [" ".join(x) for x in old],
                                      [" ".join(x) for x in new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        a, b = old[i1:i2], new[j1:j2]
        # greedy in-order pairing of near-identical steps
        anchors: list[tuple[int, int]] = []
        k = 0
        for i, step in enumerate(a):
            for j in range(k, len(b)):
                if _similar(step, b[j]):
                    anchors.append((i, j))
                    k = j + 1
                    break
        pi = pj = 0
        for i, j in anchors + [(len(a), len(b))]:
            if a[pi:i] or b[pj:j]:
                yield a[pi:i], b[pj:j], j1 + pj, j1 + j
            pi, pj = i + 1, j + 1


def _absorber(new: Sequence[list[str]], start: int, end: int,
              neighbours: tuple[list[str] | None, list[str] | None] = (None, None)) -> list[str] | None:
    """The automation step that took over a run of old steps deleted outright:
    the new step right after the gap (a deleted `refine le_trans ?_ ..` before
    the `nlinarith` that now does it), or, at the end, the one right before it
    (`simp; exact foo` -> `simp`).  Past the hunk's own steps, `neighbours`
    are the unchanged steps just before and after it in the same block."""
    after = new[end] if end < len(new) else neighbours[1]
    before = new[start - 1] if start > 0 else neighbours[0]
    for step in (after, before):
        if step:
            return step if step_is_automation(step) else None
    return None


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _neighbours(hunk: Hunk, after_lines: Sequence[str]) -> tuple[list[str] | None, list[str] | None]:
    """The unchanged steps just before and after a hunk in the new text, when
    they sit at the indentation the hunk's old steps did (the same block)."""
    body = [line for line in hunk.before if line.strip()]
    if not body or hunk.header:
        return None, None
    indent = _indent(body[0])
    prev = next((line for line in reversed(after_lines[:hunk.after_start]) if line.strip()), None)
    nxt = next((line for line in after_lines[hunk.after_end:] if line.strip()), None)
    def step(line: str | None, last: bool) -> list[str] | None:
        if line is None or _indent(line) != indent:
            return None
        steps = list(_steps([line]))
        return (steps[-1] if last else steps[0]) if steps else None
    return step(prev, True), step(nxt, False)


def _proof_part(step: Sequence[str]) -> Sequence[str]:
    """What a step proves with: for `have h : T := e`, `e` -- the stated type
    names functions, not the lemmas the step relies on."""
    if step and step[0] in _STATES:
        cut = _until(step, 1, {":="})
        return step[cut + 1:] if cut is not None else []
    return step


def _cites(step: Sequence[str]) -> set[str]:
    return citations(_proof_part(step))


def _inlined(old: Sequence[list[str]], still_named: set[str]) -> bool:
    """The old run's hand steps cite lemmas the new proof still names: they
    moved into another step (`have hd := foo hne` inlined as `.. (foo hne)`)
    rather than into automation."""
    cited = set().union(*(_cites(s) for s in old if step_is_automation(s) is False))
    return bool(cited) and cited <= still_named


def _inlining_target(step: Sequence[str], old: Sequence[list[str]], cited_before: set[str]) -> bool:
    """A new hand step beside absorbed ones is where old steps were inlined
    (it names a lemma they named, and none the old proof never did) or a
    local abbreviation.  Anything else -- `exact foo` -> `exact bar` -- is a
    hand step replacing a hand step, not automation."""
    if step and step[0] in ("let", "set"):
        return True
    cited = _cites(step)
    return bool(cited & set().union(*(_cites(x) for x in old))) and not cited - cited_before


def _dropped(old: Sequence[list[str]], still_named: set[str]) -> list[list[str]]:
    """The old hand steps whose every lemma is gone from the new proof: not
    moved elsewhere, so whatever now does their work does it unnamed."""
    return [s for s in old if step_is_automation(s) is False
            and (cited := _cites(s)) and not cited & still_named]


_VARIABLE = re.compile(r"^\s*variable\b(.*)$")


def variable_binders(file_text: str) -> set[str]:
    """Names bound by `variable` commands in a file, and its instance binders."""
    found: set[str] = set()
    lines = remove_lean_comments(file_text).splitlines()
    for index, line in enumerate(lines):
        match = _VARIABLE.match(line)
        if not match:
            continue
        chunk = match.group(1)
        k = index + 1
        while k < len(lines) and lines[k].startswith((" ", "\t")) and lines[k].strip():
            chunk += " " + lines[k]
            k += 1
        toks = lean_refactor_lexer(chunk.strip()).split()
        i = 0
        while i < len(toks):
            if toks[i] in ("(", "{", "[", "⦃"):
                close = _match(toks, i)
                if close is None:
                    break
                inner = toks[i + 1:close]
                if toks[i] == "[":
                    found.add(" ".join(inner))
                elif ":" in inner:
                    found.update(n for n in inner[:inner.index(":")] if is_ident(n))
                i = close + 1
            else:
                i += 1
    return found


def analyse(before_text: str, after_text: str, *, after_file: str = "",
            definitions: frozenset[str] | set[str] = frozenset(),
            lemmas: frozenset[str] | set[str] | None = None) -> list[Hunk]:
    found = hunks(before_text, after_text)
    binders = variable_binders(after_file) if after_file else set()
    idents: set[str] = set()
    for tok in token_list(clean_lines(after_text)):
        if is_ident(tok):
            parts = tok.split(".")
            idents.update(".".join(parts[:k]) for k in range(1, len(parts) + 1))
    # Names the old declaration used anywhere, or the new statement mentions,
    # are known objects: a function in the goal is not a new lemma.
    after_lines = clean_lines(after_text)
    statement = after_lines[:header_end(after_lines) + 1]
    known = {t for t in token_list(clean_lines(before_text)) + token_list(statement) if is_ident(t)}
    for hunk in found:
        apply_syntax(hunk, binders, idents)
        classify_automation(hunk, definitions, known, lemmas, still_named=idents,
                            neighbours=_neighbours(hunk, after_lines))
    absorbed_at_use(found, idents, known, lemmas)
    pure_automation(before_text, after_text, found, definitions, lemmas)
    return found


def _claim_blocks(lines: Sequence[str]) -> Iterator[tuple[str, list[str]]]:
    """Each named claim (`have hpos : T := ..`) with the deeper-indented lines
    that prove it."""
    lines = [line for line in lines if line.strip()]
    index = 0
    while index < len(lines):
        toks = lean_refactor_lexer(lines[index].strip()).split()
        if len(toks) > 1 and toks[0] in _CLAIM and is_ident(toks[1]):
            end = index + 1
            while end < len(lines) and _indent(lines[end]) > _indent(lines[index]):
                end += 1
            yield toks[1], lines[index:end]
            index = end
        else:
            index += 1


def _nested_calls(lines: Sequence[str]) -> collections.Counter:
    """`by <automation>` inside a term, `(by positivity)`: never a step of its own."""
    toks = token_list(lines)
    return collections.Counter(b for a, b in zip(toks, toks[1:]) if a == "by" and b in AUTOMATION)


def _new_automation(before: Sequence[str], after: Sequence[str], known: set[str],
                    lemmas: frozenset[str] | set[str] | None) -> list[list[str]]:
    """Automation the new lines call that the old ones did not: steps first,
    then `by <tactic>` nested in a term.  A call handed a repository lemma the
    old proof never named is a new argument, not automation."""
    old = {_bare(step) for step in _steps(before)}
    came = [step for step in _steps(after)
            if step_is_automation(step) and _bare(step) not in old
            and not (lemmas is not None and any(
                is_ident(t) and t.rsplit(".", 1)[-1] in lemmas and t not in known for t in step))]
    nested = _nested_calls(after) - _nested_calls(before)
    return came + [["by", tactic] for tactic in nested]


def absorbed_at_use(found: list[Hunk], still_named: set[str], known: set[str],
                    lemmas: frozenset[str] | set[str] | None = None) -> None:
    """A helper claim deleted in one hunk whose use, in another hunk, became
    automation: `have hhpos : 0 < h := by ..` gone, and `log_sinh_div_le hhpos`
    now `log_sinh_div_le (by positivity)`.  The line diff cuts this collapse in
    two, so neither hunk alone shows it.  The deleted block is charged to its
    own hunk, net of the automation step that replaced it, and never beyond
    what that hunk has left after syntax."""
    for hunk in found:
        if hunk.automation or hunk.kind == "syntax":
            continue
        for name, block in _claim_blocks(hunk.before):
            if name in still_named:
                continue
            block_steps = list(_steps(block))
            if _inlined(block_steps, still_named):
                continue
            for use in found:
                if use is hunk or name not in token_list(use.before) or name in token_list(use.after):
                    continue
                came = _new_automation(use.before, use.after, known, lemmas)
                if not came:
                    continue
                cost = 0 if use.automation else len(came[0])
                charge = min(charged_steps(block_steps, []) - cost,
                             hunk.delta - hunk.syntax - hunk.automation)
                if charge > 0:
                    hunk.automation += charge
                    hunk.automated_runs.append((block_steps, [came[0]], "absorbed_at_use"))
                    hunk.kind = "partial_automation"
                break


# --- automation on aligned proof steps ------------------------------------------
# The same collapse rules, applied to the steps of the whole proof instead of
# hunk by hunk: the line hunks are used for syntax only.  Aligning steps over
# the whole proof makes the hunk patches unnecessary -- a collapse the line diff
# cut in two is one run here, the step beside a gap is simply the next step of
# the new proof, and the cap is the declaration's.


@dataclass
class StepRun:
    old: list[list[str]]
    new: list[list[str]]
    kind: str        # replace | upgrade | absorbed | absorbed_at_use
    charge: int
    old_idx: list[int] = field(default_factory=list)   # the steps of OLD / NEW it covers
    new_idx: list[int] = field(default_factory=list)
    moved: int = 0   # tokens of automation inside a new step it does not cover (`(by positivity)`)


@dataclass
class StepAnalysis:
    old_lines: list[str]          # the old declaration, syntax rules applied
    new_lines: list[str]
    old_proof: list[str]
    new_proof: list[str]
    OLD: list[list[str]]
    NEW: list[list[str]]
    runs: list[StepRun]
    pairs: dict[int, int] = field(default_factory=dict)   # old step -> the new step it became
    named: set[str] = field(default_factory=set)           # identifiers of the new declaration
    inlined: set[int] = field(default_factory=set)         # old steps of helpers inlined at their use
    inline_saving: int = 0


def _positions(seq: Sequence[str], sub: Sequence[str]) -> list[int]:
    k = len(sub)
    return [i for i in range(len(seq) - k + 1) if list(seq[i:i + k]) == list(sub)] if k else []


def _inlined_helpers(old_proof: list[str], OLD: list[list[str]], NEW: list[list[str]],
                     named: set[str]) -> tuple[set[int], int]:
    """Helper claims deleted whose proof now sits inside the steps that used them
    (`have h0 : T := foo` .. `h0.mul_const` -> `(foo).mul_const`; `have hk : .. :=
    by omega` .. `hk` -> `by omega`): their old steps, and the tokens saved --
    the helper minus the copies (and parentheses) that replaced its name.  A copy
    counts only beyond the times that proof already occurred in the old proof."""
    steps_of = set()
    saving = 0
    for name, block in _claim_blocks(old_proof):
        if name in named:
            continue
        steps = list(_steps(block))
        start = next((i for i in range(len(OLD)) if OLD[i:i + len(steps)] == steps), None)
        if start is None:
            continue
        toks = [t for t in token_list(block) if t != NL]
        cut = _until(toks, 1, {":="})
        proof = toks[cut + 1:] if cut is not None else []
        if not proof:
            continue
        outside = [st for i, st in enumerate(OLD) if not start <= i < start + len(steps)]
        bare = sum(t == name for st in outside for t in st)
        dotted = sum(t.startswith(name + ".") for st in outside for t in st)
        if not bare and not dotted:
            continue
        copies = [(st, k) for st in NEW for k in _positions(st, proof)]
        extra = len(copies) - sum(len(_positions(st, proof)) for st in outside)
        if extra <= 0:
            continue
        copies = copies[:extra]
        parens = sum(2 for st, k in copies if 0 < k and k + len(proof) < len(st)
                     and st[k - 1] == "(" and st[k + len(proof)] == ")")
        inserted = len(copies) * len(proof) + parens - min(bare, len(copies))
        steps_of.update(range(start, start + len(steps)))
        saving += len(toks) - inserted
    return steps_of, saving


def old_after_syntax(before_text: str, found: list[Hunk]) -> list[str]:
    """The old declaration with every hunk's syntax rewrites applied."""
    old = clean_lines(before_text)
    for hunk in sorted(found, key=lambda h: -h.before_start):
        if hunk.rewritten and hunk.rewritten != hunk.before:
            old[hunk.before_start:hunk.before_start + len(hunk.before)] = hunk.rewritten
    return old


def _alignment(old: list[list[str]], new: list[list[str]]):
    """Replaced runs (old slice, new slice) and the old -> new pairs of steps
    kept identical or with small edits (the anchors), as in `_aligned_runs`."""
    runs: list[tuple[int, int, int, int]] = []
    pairs: dict[int, int] = {}
    matcher = difflib.SequenceMatcher(None, [" ".join(x) for x in old],
                                      [" ".join(x) for x in new], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            pairs.update({i1 + k: j1 + k for k in range(i2 - i1)})
            continue
        anchors: list[tuple[int, int]] = []
        k = j1
        for i in range(i1, i2):
            for j in range(k, j2):
                if _similar(old[i], new[j]):
                    anchors.append((i, j))
                    k = j + 1
                    break
        pi, pj = i1, j1
        for i, j in anchors + [(i2, j2)]:
            if pi < i or pj < j:
                runs.append((pi, i, pj, j))
            pi, pj = i + 1, j + 1
        pairs.update(dict(anchors))
    return runs, pairs


def _nested(step: Sequence[str]) -> collections.Counter:
    return collections.Counter(b for a, b in zip(step, step[1:]) if a == "by" and b in AUTOMATION)


def step_automation(before_text: str, after_text: str, found: list[Hunk],
                    definitions: frozenset[str] | set[str] = frozenset(),
                    lemmas: frozenset[str] | set[str] | None = None) -> tuple[int, list[StepRun]]:
    """Tokens of the declaration that collapsed into automation, judged on the
    aligned steps of the whole proof (`found`: its hunks, syntax applied)."""
    analysis = step_analysis(before_text, after_text, found, definitions, lemmas)
    saving = sum(h.delta for h in found) - sum(h.syntax for h in found)
    return max(0, min(sum(r.charge for r in analysis.runs), saving)), analysis.runs


def step_analysis(before_text: str, after_text: str, found: list[Hunk],
                  definitions: frozenset[str] | set[str] = frozenset(),
                  lemmas: frozenset[str] | set[str] | None = None) -> StepAnalysis:
    """The aligned steps of a declaration's old and new proof, and the runs of
    them that collapsed into automation."""
    old_lines, new_lines = old_after_syntax(before_text, found), clean_lines(after_text)
    old_proof, new_proof = _proof_lines(old_lines, True), _proof_lines(new_lines, True)
    OLD, NEW = list(_steps(old_proof)), list(_steps(new_proof))
    if not OLD:
        return StepAnalysis(old_lines, new_lines, old_proof, new_proof, OLD, NEW, [])
    named: set[str] = set()
    for tok in token_list(new_lines):
        if is_ident(tok):
            parts = tok.split(".")
            named.update(".".join(parts[:k]) for k in range(1, len(parts) + 1))
    known = {t for t in token_list(clean_lines(before_text))
             + token_list(new_lines[:header_end(new_lines) + 1]) if is_ident(t)}

    # a repository lemma the old declaration never named: deleted hand work may
    # have moved into it, so nothing is inferred to be absorbed by automation
    new_lemma = lemmas is not None and any(
        is_ident(t) and t not in known and (last := t.rsplit(".", 1)[-1]) in lemmas
        and not _LOCAL.match(last) and last not in _TACTIC_WORDS for t in token_list(new_proof))
    # inlining is syntax, and is judged first: an inlined helper is no hand work
    # that automation took over
    inlined, inline_saving = _inlined_helpers(old_proof, OLD, NEW, named)
    runs: list[StepRun] = []
    charged_old: set[int] = set(inlined)   # old step indices already accounted for
    kept_old = [x for i, x in enumerate(OLD) if i not in inlined]
    whole = automation_verdict(kept_old, NEW, known, definitions, lemmas) if NEW and kept_old else None
    if whole:
        keep = [i for i in range(len(OLD)) if i not in inlined]
        runs.append(StepRun(kept_old, NEW, whole, charged_steps(kept_old, NEW), keep, list(range(len(NEW)))))
        charged_old.update(range(len(OLD)))
        pairs: dict[int, int] = {}
    else:
        spans, pairs = _alignment(OLD, NEW)
        for i1, i2, j1, j2 in spans:
            idx = [i for i in range(i1, i2) if i not in inlined]
            old, new = [OLD[i] for i in idx], NEW[j1:j2]
            if not old:
                continue
            kind = automation_verdict(old, new, known, definitions, lemmas)
            if kind:
                runs.append(StepRun(old, new, kind, charged_steps(old, new), idx, list(range(j1, j2))))
                charged_old.update(idx)
                continue
            if new_lemma or not any(step_is_automation(x) is False or _claim_key(x) is not None
                                    for x in old):
                continue      # hand work gone into a new lemma; or `<;> ring` dropped, pruning
            adjacent = _absorber(NEW, j1, j2) is not None
            if not new:
                if adjacent and not _inlined(old, named) and charged_steps(old, []) > 0:
                    runs.append(StepRun(old, [], "absorbed", charged_steps(old, []), idx))
                    charged_old.update(idx)
            elif all(_inlining_target(x, old, known) for x in new if step_is_automation(x) is False):
                calls = [x for x in new if step_is_automation(x) is True]
                seen = {_bare(x) for x in old}
                fresh = [x for x in calls if _bare(x) not in seen]
                gone = _dropped(old, named)
                if gone and (adjacent or calls) and charged_steps(gone, fresh) > 0:
                    gone_idx = [idx[k] for k, x in enumerate(old) if any(x is g for g in gone)]
                    fresh_idx = [j1 + k for k, x in enumerate(new) if any(x is f for f in fresh)]
                    runs.append(StepRun(gone, fresh, "absorbed", charged_steps(gone, fresh),
                                        gone_idx, fresh_idx))
                    charged_old.update(gone_idx)

    # a helper claim deleted where it stood, whose use gained automation: nested
    # (`bar_le (by positivity)`) or as the new step right after it (`rw [.., hval]`
    # then `grind` for `linear_combination`)
    old_bare = {_bare(x) for x in OLD}
    for name, block in ([] if new_lemma else _claim_blocks(old_proof)):
        if name in named:
            continue
        steps = list(_steps(block))
        start = next((i for i in range(len(OLD)) if OLD[i:i + len(steps)] == steps), None)
        if start is None or charged_old & set(range(start, start + len(steps))) or _inlined(steps, named):
            continue
        for i, step in enumerate(OLD):
            if start <= i < start + len(steps) or name not in step or i not in pairs:
                continue
            j = pairs[i]
            # automation the helper already ran and that merely moved to its use
            # (`have h : .. := by omega` -> `.., by omega, ..`) is inlining
            moved = _nested([t for x in steps for t in x]) + collections.Counter(
                x[0] for x in steps if step_is_automation(x) is True)
            gained = _nested(NEW[j]) - _nested(step) - moved
            after = NEW[j + 1] if j + 1 < len(NEW) else None
            if gained:
                call, cost = ["by", next(iter(gained))], 2 * sum(gained.values())
            elif after and step_is_automation(after) is True and _bare(after) not in old_bare:
                call, cost = after, 0      # the call is charged, if at all, by its own run
            else:
                break
            # only the helper's hand work collapsed: its statement is now the
            # goal the use site infers
            hand = [start + k for k, x in enumerate(steps) if step_is_automation(x) is False]
            charge = sum(len(OLD[i]) for i in hand) - cost
            if charge > 0:
                runs.append(StepRun(steps, [call], "absorbed_at_use", charge, hand, [], cost))
                charged_old.update(range(start, start + len(steps)))
            break
    return StepAnalysis(old_lines, new_lines, old_proof, new_proof, OLD, NEW, runs, pairs, named,
                        inlined, inline_saving)


# --- proof skeletons ----------------------------------------------------------
# A proof is a tree of claims: the declaration's own goal at the root, and one
# node per `have`/`obtain`/`suffices` with a stated type.  Two proofs of the
# same declaration are compared claim by claim: a claim stated identically on
# both sides is proved twice, once by the old proof and once by the new, and
# those two sub-proofs can be compared step by step without any line diff.

_CLAIM = frozenset({"have", "haveI", "obtain", "suffices", "replace"})


@dataclass
class Claim:
    key: str                     # the stated type, token-normalised ("" for the root)
    own: list[str]               # lines of this claim's proof, children excluded
    children: list["Claim"] = field(default_factory=list)


def _claim_head(line: str) -> tuple[str, str] | None:
    """For `have h : T := <rest>` return (T, rest); rest may start with `by`."""
    toks = lean_refactor_lexer(line.strip()).split()
    if not toks or toks[0] not in _CLAIM:
        return None
    depth, colon, assign = 0, None, None
    for index, tok in enumerate(toks):
        if tok in OPEN:
            depth += 1
        elif tok in CLOSE:
            depth -= 1
        elif depth == 0 and tok == ":" and colon is None:
            colon = index
        elif depth == 0 and tok == ":=":
            assign = index
            break
    if colon is None or assign is None or assign <= colon + 1:
        return None
    return " ".join(toks[colon + 1:assign]), " ".join(toks[assign + 1:])


def claim_tree(lines: Sequence[str], key: str = "") -> Claim:
    """Split a proof into its claims by indentation."""
    node = Claim(key=key, own=[])
    index = 0
    lines = [line for line in lines if line.strip()]
    while index < len(lines):
        line = lines[index]
        head = _claim_head(line)
        indent = len(line) - len(line.lstrip())
        if head is None:
            node.own.append(line)
            index += 1
            continue
        stated, rest = head
        end = index + 1
        # the claim's proof: the rest of its line, then every deeper line
        while end < len(lines) and len(lines[end]) - len(lines[end].lstrip()) > indent:
            end += 1
        body = lines[index + 1:end]
        rest_toks = rest.split()
        if rest_toks[:1] == ["by"]:
            rest = " ".join(rest_toks[1:])
        inner = ([" " * (indent + 2) + rest] if rest else []) + list(body)
        # the statement itself stays in the parent as routing
        node.own.append(line.split(":=")[0])
        node.children.append(claim_tree(inner, stated))
        index = end
    return node


def _claims(node: Claim) -> Iterator[Claim]:
    yield node
    for child in node.children:
        yield from _claims(child)


def _own_with_unmatched(node: Claim, matched: set[int]) -> list[str]:
    """A claim's own lines plus the lines of descendants that have no partner:
    a `have` the other side does not state is part of this claim's work."""
    lines = list(node.own)
    for child in node.children:
        if id(child) not in matched:
            lines.extend(_own_with_unmatched(child, matched))
    return lines


def match_claims(old: Claim, new: Claim) -> list[tuple[Claim, Claim]]:
    """Pairs of claims stated identically on both sides; the roots always pair."""
    pairs = [(old, new)]
    remaining: dict[str, list[Claim]] = collections.defaultdict(list)
    for claim in list(_claims(new))[1:]:
        remaining[claim.key].append(claim)
    for claim in list(_claims(old))[1:]:
        if remaining.get(claim.key):
            pairs.append((claim, remaining[claim.key].pop(0)))
    return pairs


@dataclass
class ClaimComparison:
    key: str
    old: list[str]
    new: list[str]
    labels_old: collections.Counter
    labels_new: collections.Counter
    calls_old: int
    calls_new: int
    strict: int = 0     # new sub-proof is automation-only, old did manual work
    runs: int = 0       # otherwise: aligned runs inside the claim that became automation
    shift: int = 0      # manual down, automation calls up, net of automation added


def _labels(lines: Sequence[str]) -> tuple[collections.Counter, int, list[list[str]]]:
    counts: collections.Counter = collections.Counter()
    steps = list(_steps(lines))
    calls = 0
    for step in steps:
        verdict = step_is_automation(step)
        counts["automation" if verdict else "routing" if verdict is None else "manual"] += len(step)
        calls += bool(verdict)
    return counts, calls, steps


def compare_skeletons(old_lines: Sequence[str], new_lines: Sequence[str],
                      cited_before: set[str],
                      definitions: frozenset[str] | set[str] = frozenset(),
                      lemmas: frozenset[str] | set[str] | None = None) -> list[ClaimComparison]:
    """Compare two proofs claim by claim (both given as proof lines)."""
    return _compare_trees(claim_tree(old_lines), claim_tree(new_lines), cited_before,
                          definitions, lemmas)


def _compare_trees(old_tree: Claim, new_tree: Claim, cited_before: set[str],
                   definitions: frozenset[str] | set[str] = frozenset(),
                   lemmas: frozenset[str] | set[str] | None = None) -> list[ClaimComparison]:
    pairs = match_claims(old_tree, new_tree)
    matched_old = {id(a) for a, _ in pairs}
    matched_new = {id(b) for _, b in pairs}
    out: list[ClaimComparison] = []
    for a, b in pairs:
        old_own, new_own = _own_with_unmatched(a, matched_old), _own_with_unmatched(b, matched_new)
        lo, co, so = _labels(old_own)
        ln, cn, sn = _labels(new_own)
        cmp = ClaimComparison(a.key, old_own, new_own, lo, ln, co, cn)
        work_old = lo["manual"] + lo["automation"]
        work_new = ln["manual"] + ln["automation"]
        if lo["manual"] and automation_verdict(so, sn, cited_before, definitions, lemmas) == "replace":
            cmp.strict = work_old - work_new
        else:
            for old_run, new_run in _aligned_runs(so, sn):
                if automation_verdict(old_run, new_run, cited_before, definitions, lemmas):
                    cmp.runs += scored_steps(old_run) - scored_steps(new_run)
        dm, da = lo["manual"] - ln["manual"], ln["automation"] - lo["automation"]
        saving = sum(lo.values()) - sum(ln.values())
        if cn > co and dm > 0:
            cmp.shift = max(0, min(dm - da, saving))
        out.append(cmp)
    return out


def skeleton_automation(before_text: str, after_text: str, found: list[Hunk],
                        definitions: frozenset[str] | set[str] = frozenset(),
                        lemmas: frozenset[str] | set[str] | None = None) -> list[ClaimComparison]:
    """Claim-by-claim comparison of a declaration's old proof -- after the syntax
    rules, rebuilt from its hunks -- with its new proof."""
    old = clean_lines(before_text)
    for hunk in sorted(found, key=lambda h: -h.before_start):
        size = len(hunk.before)
        if hunk.rewritten != hunk.before:
            old[hunk.before_start:hunk.before_start + size] = hunk.rewritten
    old_proof = _proof_lines(old, True)
    new_proof = _proof_lines(clean_lines(after_text), True)
    cited = {t for t in token_list(clean_lines(before_text)) if is_ident(t)}
    after_lines = clean_lines(after_text)
    cited |= {t for t in token_list(after_lines[:header_end(after_lines) + 1]) if is_ident(t)}
    return compare_skeletons(old_proof, new_proof, cited, definitions, lemmas)


def _numbered_proof(text: str, start_line: int) -> list[tuple[int, str]]:
    """The proof lines of a declaration with their line numbers in the file."""
    lines = clean_lines(text)
    head = header_end(lines)
    proof = _proof_lines(lines[head:head + 1], True)
    numbered = [(start_line + head, line) for line in proof]
    numbered += [(start_line + index, line) for index, line in enumerate(lines) if index > head]
    return [(n, line) for n, line in numbered if line.strip()]


def keyed_claim_tree(numbered: list[tuple[int, str]], keys: Mapping[int, str], key: str = "") -> Claim:
    """`claim_tree`, but a claim is any claim-introducing line Lean elaborated,
    keyed by its elaborated type (`keys`: file line -> normalised type), so a
    claim re-spelled through notation, a definition or a local `set` still
    meets its twin, and `have h := e` is a claim too."""
    node = Claim(key=key, own=[])
    index = 0
    while index < len(numbered):
        number, line = numbered[index]
        toks = lean_refactor_lexer(line.strip()).split()
        indent = len(line) - len(line.lstrip())
        if not toks or toks[0] not in _CLAIM or not keys.get(number) or ":=" not in toks:
            node.own.append(line)
            index += 1
            continue
        end = index + 1
        while end < len(numbered) and len(numbered[end][1]) - len(numbered[end][1].lstrip()) > indent:
            end += 1
        cut = toks.index(":=")
        rest = toks[cut + 1:]
        if rest[:1] == ["by"]:
            rest = rest[1:]
        inner = ([(number, " " * (indent + 2) + " ".join(rest))] if rest else []) + numbered[index + 1:end]
        node.own.append(" " * indent + " ".join(toks[:cut]))
        node.children.append(keyed_claim_tree(inner, keys, keys[number]))
        index = end
    return node


def keyed_skeleton_automation(before_text: str, after_text: str, before_start: int, after_start: int,
                              keys_before: Mapping[int, str], keys_after: Mapping[int, str],
                              definitions: frozenset[str] | set[str] = frozenset(),
                              lemmas: frozenset[str] | set[str] | None = None) -> list[ClaimComparison]:
    """Claim-by-claim comparison with claims matched on their elaborated types."""
    old_tree = keyed_claim_tree(_numbered_proof(before_text, before_start), keys_before)
    new_tree = keyed_claim_tree(_numbered_proof(after_text, after_start), keys_after)
    cited = {t for t in token_list(clean_lines(before_text)) if is_ident(t)}
    after_lines = clean_lines(after_text)
    cited |= {t for t in token_list(after_lines[:header_end(after_lines) + 1]) if is_ident(t)}
    return _compare_trees(old_tree, new_tree, cited, definitions, lemmas)
