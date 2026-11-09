"""Adapt the frozen macro measurement to real command and parser contexts."""
from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import re


def rule_head(pattern: str) -> str | None:
    """Find the first literal after leading Lean quotation antiquotations."""
    text = pattern.lstrip()
    while text.startswith('$'):
        if text.startswith('$['):
            depth, end = 1, 2
            while end < len(text) and depth:
                depth += (text[end] == '[') - (text[end] == ']')
                end += 1
            if depth:
                return None
            text = text[end:].lstrip('*+?, ').lstrip()
        else:
            match = re.match(r"\$[\w']+(?::[\w'.]+)?", text)
            if not match:
                return None
            text = text[match.end():].lstrip()
    match = re.match(r"[^\s$()]+", text)
    return match.group(0) if match else None


def preceding_doc(body: str, start: int) -> int | None:
    prefix = body[:start].rstrip()
    if not prefix.endswith('-/'):
        return None
    for match in reversed(list(re.finditer(r'/--', prefix))):
        depth = 0
        for token in re.finditer(r'/-|-/', prefix[match.start():]):
            depth += 1 if token.group() == '/-' else -1
            if depth == 0:
                if match.start() + token.end() == len(prefix):
                    return match.start()
                break
    return None


def command_sites(sites, programs, files):
    commands = {p.lead: p for p in programs if p.category == 'command'}
    result = []
    for site in sites:
        program = commands.get(site.lead) if site.category == 'command' else None
        if program is None:
            result.append(site)
            continue
        body = files[site.path]
        if body[body.rfind('\n', 0, site.start) + 1:site.start].strip():
            continue  # A variable or notation parameter is not a command invocation.
        if re.search(r'\bmacro\s+\w+:docComment\b', program.source):
            start = preceding_doc(body, site.start)
            if start is None:
                continue
            site = replace(site, start=start, text=body[start:site.end])
        result.append(site)
    return result


def install(macros, program_sites, files: dict[str, str], job: dict, audit: dict):
    """Retain the reference ledger; repair only its measurement inputs."""
    collect, find, build, measure = (macros.collect_programs, macros.find_sites,
                                     macros.build_driver, program_sites.measure)
    mathlib = False

    def collect_programs(sources, after, added, **kwargs):
        programs, uncovered, context = collect(sources, after, added, **kwargs)
        leads = {'elab_rules': set(), 'macro_rules': set()}
        for representative, _ in macros.written_declaration_groups(after, added):
            node = after.nodes[representative]
            source = sources.slice(node) or ''
            keyword = macros.leading_command(source)
            if keyword not in leads:
                continue
            path = sources.path_of_module(str(node.get('module', ''))) or ''
            code = macros.remove_lean_comments(source, mask_strings=True)
            for pattern in re.findall(r'\|\s*`\((.*?)\)\s*=>', code, re.S):
                head = rule_head(pattern)
                if head:
                    leads[keyword].add((path, head))
        elaborators = leads['elab_rules'] - leads['macro_rules']
        kept = []
        for program in programs:
            if program.keyword == 'syntax' and (program.path or '', program.lead) in elaborators:
                uncovered.append(dict(keyword='syntax', reason='elab_rules: not a macro expansion', tokens=program.tokens))
                audit['elaborators_excluded'] = audit.get('elaborators_excluded', 0) + 1
            else:
                kept.append(program)
        return kept, uncovered, context

    def find_sites(source_files, programs, enclosing=None):
        return command_sites(find(source_files, programs, enclosing), programs, source_files)

    def build_driver(context, sites, *args, **kwargs):
        prepared = []
        for site in sites:
            body = files.get(site.path, '')
            column = site.start - body.rfind('\n', 0, site.start) - 1
            prepared.append(replace(site, text=' ' * max(0, column) + site.text))
        driver = build(context, prepared, *args, **kwargs)
        driver = driver.replace('open Lean Elab Command PrettyPrinter\n',
                                'open Lean Elab Command PrettyPrinter\nopen Lean.Parser.Tactic\n', 1)
        # Prefix indentation restores the real layout boundary. Remove only that
        # synthetic prefix from the consumed-byte receipt used for source slicing.
        driver = driver.replace('| .some (stx, consumed) => do',
                                "| .some (stx, consumed) => do\n    let consumed := consumed - (input.toList.takeWhile (· == ' ')).length", 1)
        if mathlib:
            driver = driver.replace('import Lean\n', 'import Mathlib\n', 1)
        return driver

    def measure_with_context(*args, **kwargs):
        nonlocal mathlib
        result = measure(*args, **kwargs)
        if not result.get('error'):
            return result
        packages = Path('/opt/leanlean/packages') / job['key']
        if not packages.is_dir():
            return result
        paths = sorted(p / '.lake/build/lib/lean' for p in packages.iterdir() if p.is_dir())
        paths = [p for p in paths if p.is_dir()]
        if not any((p / 'Mathlib.olean').is_file() for p in paths):
            return result
        audit['core_driver_error'] = result['error']
        audit['parser_context'] = 'pinned Mathlib and Lean.Parser.Tactic'
        previous = os.environ.get('LEAN_PATH')
        try:
            os.environ['LEAN_PATH'] = os.pathsep.join(map(str, paths))
            mathlib = True
            return measure(*args, **{**kwargs, 'refresh': True})
        finally:
            mathlib = False
            if previous is None:
                os.environ.pop('LEAN_PATH', None)
            else:
                os.environ['LEAN_PATH'] = previous

    macros.collect_programs = collect_programs
    macros.find_sites = find_sites
    macros.build_driver = build_driver
    program_sites.measure = measure_with_context
