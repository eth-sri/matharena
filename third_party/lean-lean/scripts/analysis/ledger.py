"""One ledger per kept declaration: where every token of its change went.

The old and new proof are split into steps and aligned (proof_hunks.step_analysis);
each step is automation or not.  Syntax is booked on the side, never by
rewriting text:

  spelling   tokens the syntax rules took off the old text (`simp only` ->
             `simp`, dropped `have` types, binders hoisted into `variable` ...)
  inlining   a helper claim deleted whose proof now sits inside the step that
             used it (`have h0 : T := foo` .. `h0.mul_const` -> `(foo).mul_const`,
             `have hk : .. := by omega` .. `hk` -> `by omega`): the helper's
             tokens minus what the inlined copies added
  uses       tokens the new abbreviations saved, priced per use in each step
             and in the statement (abbreviations.py), plus the call sites of
             programmatic macros, each measured by Lean (program_sites.py)

and the rest of the change is split by step:

  automation   old steps of runs that collapsed into automation, minus the new
               steps they became, minus the abbreviation uses inside those
  rest         everything else -- the other steps, the separators (`by`, `;`,
               `·` ...) and the statement -- minus its abbreviation uses; it
               goes to the declaration's own class (proof simplification or
               structural diff)

    delta = spelling + inlining + uses + automation + rest       exactly
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from scripts.analysis import proof_hunks as ph

Price = Callable[[Sequence[Sequence[str]]], int]


@dataclass
class Ledger:
    delta: int                 # tokens(old) - tokens(new)
    spelling: int
    uses: int                  # abbreviation uses, all of the declaration
    automation_uses: int       # the part of `uses` inside automation steps
    automation: int
    rest: int
    inlining: int = 0
    runs: list = field(default_factory=list)

    @property
    def syntax(self) -> int:
        return self.spelling + self.inlining + self.uses


def _count(lines: Sequence[str]) -> int:
    return ph.scored(ph.token_list(lines))


def declaration_ledger(before_text: str, after_text: str, found: list[ph.Hunk],
                       definitions=frozenset(), lemmas=None,
                       price: Price = lambda steps: 0,
                       sites: Sequence[tuple[Sequence[str], int]] = ()) -> Ledger:
    """`sites`: the programmatic macro calls in this declaration, in text order,
    as (the call's leading tokens, tokens it saved)."""
    a = ph.step_analysis(before_text, after_text, found, definitions, lemmas)
    old_clean = ph.clean_lines(before_text)
    delta = _count(old_clean) - _count(a.new_lines)
    spelling = _count(old_clean) - _count(a.old_lines)

    auto_old: set[int] = set()
    auto_new: set[int] = set()
    moved = 0
    for run in a.runs:
        auto_old.update(run.old_idx)
        auto_new.update(run.new_idx)
        moved += run.moved
    automation_raw = (sum(len(a.OLD[i]) for i in auto_old)
                      - sum(len(a.NEW[j]) for j in auto_new) - moved)

    # abbreviation uses: per new step, and in the statement (the tokens before
    # the proof: the new token stream is statement + `:=` + proof)
    step_uses = [price([step]) for step in a.NEW]
    # a measured call site sits in the step holding its leading tokens: the
    # calls and the occurrences are matched in order
    occurrences = [(j, k) for j, step in enumerate(a.NEW) for k in range(len(step))]
    cursor = 0
    unplaced = 0
    for lead, saving in sites:
        lead = list(lead)
        while cursor < len(occurrences):
            j, k = occurrences[cursor]
            cursor += 1
            if a.NEW[j][k:k + len(lead)] == lead:
                step_uses[j] += saving
                break
        else:
            unplaced += saving     # in the statement, or not found: not automation
    step_uses_total = sum(step_uses)
    stream = [t for t in ph.token_list(a.new_lines) if t != ph.NL]
    statement = stream[:len(stream) - _count(a.new_proof)]
    uses = step_uses_total + price([statement]) + unplaced
    automation_uses = sum(step_uses[j] for j in auto_new)

    automation = automation_raw - automation_uses
    inlined = a.inline_saving
    rest = delta - spelling - inlined - uses - automation
    return Ledger(delta, spelling, uses, automation_uses, automation, rest, inlined, a.runs)
