"""Recover simple command-generated syntax from source and Lean's graph names.

Only a one-input command elaborator that quotes a notation or macro and splices
that input as its body is accepted. Arbitrary elaborators remain unsupported.
The literal actually generated is read from graph evidence, not guessed from a
counter. Prefix arguments have :max precedence; their token-template expansion
is priced symbolically, including repeated/dropped parameters and nested calls.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from leanlean.metrics.tokens import remove_lean_comments


@dataclass(frozen=True)
class Factory:
    literal: str
    category: str
    params: tuple[str, ...]


def factory(source: str) -> Factory | None:
    code = remove_lean_comments(source).strip()
    # Require a single, direct command quotation and an unmodified body splice.
    match = re.fullmatch(
        r'elab\s+"(?P<literal>[^"\\]+)"\s+(?P<body>\w+):(?P<input>term|tactic|command)'
        r'\s*:\s*command\s*=>\s*do\s+elabCommand\s*\(←\s*`\(command\|\s*'
        r'(?P<output>notation:max|macro)\s+\$\(←\s*\w+\):str\s*(?P<tail>.*?)\)\)',
        code, re.S)
    if not match:
        return None
    body, category, tail = match['body'], match['input'], match['tail']
    if match['output'] == 'notation:max':
        if category != 'term' or not tail.endswith('=> $' + body):
            return None
        signature = tail[:-len('=> $' + body)].strip()
        param_pattern = r'(\w+):max'
    else:
        ending = re.fullmatch(r'(.*?)\s*:\s*(tactic|command)\s*=>\s*`\((tactic|command)\|\s*\$(\w+)\)', tail, re.S)
        if not ending or ending[2] != category or ending[3] != category or ending[4] != body:
            return None
        signature = ending[1].strip()
        param_pattern = r'(\w+):(?:term:max|ident)'
    params = []
    for word in signature.split():
        p = re.fullmatch(param_pattern, word)
        if not p:
            return None
        params.append(p[1])
    if len(set(params)) != len(params):
        return None
    return Factory(match['literal'], category, tuple(params))


def symbol(names: Sequence[str], category: str) -> str | None:
    """Keep internal underscores; strip parser-hole and implementation suffixes."""
    values = set()
    for name in names:
        if '___macroRules_' in name:
            candidate = name.split('___macroRules_', 1)[1]
            candidate = re.sub(r'_\d+$', '', candidate)
        else:
            candidate = name.rsplit('.', 1)[-1].lstrip('«').rstrip('»')
        if candidate.startswith(category):
            value = candidate[len(category):].rstrip('_')
            if value:
                values.add(value)
    return next(iter(values)) if len(values) == 1 else None


def template(source: str, spec: Factory, tokenize) -> list[str] | None:
    code = remove_lean_comments(source).strip()
    match = re.match(re.escape(spec.literal) + r'(?=\s|\()', code)
    if not match:
        return None
    body = code[match.end():].strip()
    # Macro quotations antiquote their typed parameters; notation bodies use
    # the parameter identifiers directly. Do not rewrite arbitrary splices.
    if spec.category != 'term':
        body = re.sub(r'\$(\w+)(?::(?:term|ident))?', lambda m: m[1] if m[1] in spec.params else m[0], body)
        if '$' in body:
            return None
    return tokenize(body)


OPEN = {'(': ')', '[': ']', '{': '}', '⟨': '⟩', '⦃': '⦄'}
CLOSE = set(OPEN.values())


def atom_end(seq: Sequence[str], start: int, lookup=None) -> int:
    """End of an atomic :max argument; fail closed on incomplete groups."""
    if start >= len(seq) or seq[start] in CLOSE or seq[start] in (':=', '=>', ',', ';', '·'):
        raise ValueError('missing generated-syntax argument')
    t = seq[start]
    if lookup and t in lookup:
        end = start + 1
        for _ in lookup[t].params:
            end = atom_end(seq, end, lookup)
        return end
    if t in OPEN:
        stack = [OPEN[t]]
        i = start + 1
        while i < len(seq):
            if seq[i] in OPEN:
                stack.append(OPEN[seq[i]])
            elif seq[i] in CLOSE:
                if seq[i] != stack.pop():
                    raise ValueError('unbalanced generated-syntax argument')
                if not stack:
                    return i + 1
            i += 1
        raise ValueError('incomplete generated-syntax argument')
    if t in ('↑', '⇑', '↥', '@', '-', '+', '!'):
        return atom_end(seq, start + 1, lookup)
    return start + 1


# A linear token-count expression: constant + sum(weight[param] * tokens(arg)).
@dataclass
class Form:
    constant: int
    weights: dict[str, int]

    def plus(self, other: Form, times: int = 1) -> None:
        self.constant += times * other.constant
        for key, value in other.weights.items():
            self.weights[key] = self.weights.get(key, 0) + times * value


def compile_forms(abbreviations) -> None:
    lookup = {a.keys[0][0]: a for a in abbreviations if a.generated_category}
    active = set()

    def form(a):
        if a.generated_constant is not None:
            return Form(a.generated_constant, dict(zip(a.params, a.generated_weights)))
        key = a.keys[0][0]
        if key in active:
            raise ValueError(f'cyclic generated syntax: {key}')
        active.add(key)
        result = walk(a.template, set(a.params))
        active.remove(key)
        a.generated_constant = result.constant
        a.generated_weights = [result.weights.get(p, 0) for p in a.params]
        a.saving = result.constant + sum(a.generated_weights) - 1 - len(a.params)
        return result

    def walk(seq, params):
        result = Form(0, {})
        i = 0
        while i < len(seq):
            a = lookup.get(seq[i])
            if a is None:
                result.plus(Form(0, {seq[i]: 1}) if seq[i] in params else Form(1, {}))
                i += 1
                continue
            i += 1
            arguments = []
            for _ in a.params:
                end = atom_end(seq, i, lookup)
                arguments.append(walk(seq[i:end], params))
                i = end
            f = form(a)
            result.plus(Form(f.constant, {}))
            for p, argument in zip(a.params, arguments):
                result.plus(argument, f.weights.get(p, 0))
        return result

    for a in lookup.values():
        form(a)


def price(seq: Sequence[str], lookup) -> int:
    """Fully nested generated-template token count minus literal source count.

    Arguments are consumed once. Their expansions are multiplied by their use
    count in the template, so a duplicated argument is neither dropped nor
    separately double-charged. Unknown syntax remains literal.
    """
    def size(seq):
        total, i = 0, 0
        while i < len(seq):
            a = lookup.get(seq[i])
            if a is None:
                total += 1
                i += 1
                continue
            i += 1
            total += a.generated_constant
            for weight in a.generated_weights:
                end = atom_end(seq, i, lookup)
                total += weight * size(seq[i:end])
                i = end
        return total
    return size(seq) - len(seq)
