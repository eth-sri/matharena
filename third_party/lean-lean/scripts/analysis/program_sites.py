"""Call sites of the macros that compute their own expansion, measured by Lean.

A `syntax` answered by a programmatic `macro_rules`, or a `macro … => do …`,
has no template to price a use from (abbreviations.py lists it as machinery).
expand_macro_programs.py runs those macros in a standalone core-Lean file and
takes one macro step at each call site; this keeps what one use saved, per site:

    saving = tokens(expansion) - tokens(call)       (no parentheses added)

with the site's place (file, character offset, line), so the ledger can book it
on the step it sits in.  Results are cached per repository, since Lean is slow.

Detection comes with it: every site carries Lean's status, and a repository
whose programmatic sites Lean could not expand is reported, not read as zero.
"""

from __future__ import annotations

import collections
import json
import re
from pathlib import Path
from typing import Any, Mapping

import scripts.analysis.expand_macro_programs as lean_pass

CACHE = Path("output/analysis/compression_categories/program_sites")


def measure(model: str, repo: str, repo_dir: Path, record: Mapping[str, Any], *,
            timeout: int = 2400, refresh: bool = False, cache: Path = CACHE,
            lean_project: Path | None = None) -> dict[str, Any]:
    path = cache / model / f"{repo}.json"
    if path.is_file() and not refresh:
        return json.loads(path.read_text(encoding="utf-8"))
    options = {'lean_project': lean_project} if lean_project is not None else {}
    result = lean_pass.analyse_repository(model, repo, repo_dir, record,
                                          list(lean_pass.PREFERRED_TOOLCHAINS), timeout, **options)
    out: dict[str, Any] = {"programs": [], "sites": [], "uncovered": [], "toolchain": "", "error": ""}
    if result is not None:
        files = {}
        out["programs"] = [{"keyword": p.keyword, "lead": p.lead, "path": p.path, "line": p.start_line,
                            "tokens": p.tokens} for p in result.programs]
        out["uncovered"] = result.uncovered
        out["toolchain"], out["error"] = result.toolchain, (result.driver_error or "")[-2000:]
        for site in result.sites:
            text = files.setdefault(site.path, "")
            out["sites"].append({
                "path": site.path, "start": site.start, "lead": site.lead, "status": site.status,
                "before": site.before_tokens if site.expansion else None,
                "after": site.after_tokens if site.expansion else None,
                "call": site.site_text if site.expansion else site.text[:80],
            })
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".partial")
    tmp.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.rename(path)
    return out


def line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


_PARSER_ALIAS = re.compile(r"^\s*(?:@\[[^\]]*\]\s*)*(?:(?:local|scoped)\s+)?syntax\s+[\w.'«»]+\s*:=")


def effective_sites(data: Mapping[str, Any], after, after_src, before_vocab: set[str],
                    tokens) -> tuple[list[dict[str, Any]], collections.Counter]:
    """The call sites that count, with what each saved.

    Lean was handed each candidate on its own, starting at the macro's leading
    literal, so a candidate is kept only when it is a call in the real file:

      * a `syntax name := ..` is a parser fragment used inside other syntax,
        not a macro: its "sites" (every `[` of a file) are dropped;
      * a command macro is called where a command starts: first on its line;
      * a candidate Lean expanded counts what it measured; one it could not
        parse (notation the core-Lean driver lacks) counts the macro's usual
        saving -- the most common one among its measured sites -- but only when
        its literal is distinctive, so the candidate is surely a call.
    """
    starts: dict[tuple[str, int], tuple[int, int]] = {}
    for node in after.nodes.values():
        path = after_src.path_of_module(str(node.get("module", "")))
        if path and node.get("start_line"):
            key = (path, int(node["start_line"]))
            starts[key] = (int(node["start_line"]), max(int(node["end_line"]), starts.get(key, (0, 0))[1]))
    kind: dict[str, dict[str, Any]] = {}
    for program in data["programs"]:
        text = after_src._files.get(program["path"] or "", "")  # noqa: SLF001
        start, end = starts.get((program["path"], program["line"]), (program["line"], program["line"]))
        source = "".join(text.splitlines(keepends=True)[start - 1:end])
        masked = lean_pass.remove_lean_comments(source, mask_strings=True)
        info = kind.setdefault(program["lead"], {"alias": True, "command": False})
        info["alias"] &= bool(_PARSER_ALIAS.match(masked))
        info["command"] |= lean_pass.declared_category(source, program["keyword"]) == "command"
    usual: dict[str, int] = {}
    for lead in kind:
        measured = [s["after"] - s["before"] for s in data["sites"] if s["lead"] == lead and s["before"] is not None]
        if measured:
            usual[lead] = collections.Counter(measured).most_common(1)[0][0]
    kept: list[dict[str, Any]] = []
    tally: collections.Counter = collections.Counter()
    for site in data["sites"]:
        info = kind.get(site["lead"], {"alias": False, "command": False})
        text = after_src._files.get(site["path"], "")  # noqa: SLF001
        line_start = text.rfind("\n", 0, site["start"]) + 1
        if info["alias"]:
            tally["parser fragment"] += 1
            continue
        if info["command"] and text[line_start:site["start"]].strip():
            tally["not at a command position"] += 1
            continue
        if site["before"] is not None:
            kept.append({**site, "saving": site["after"] - site["before"], "how": "lean"})
            tally["measured by Lean"] += 1
        elif site["status"] == "PARSE_ERROR" and site["lead"] in usual \
                and not all(t in before_vocab for t in tokens(site["lead"])):
            kept.append({**site, "saving": usual[site["lead"]], "how": "usual saving"})
            tally["unparsed call, usual saving"] += 1
        else:
            tally[f"dropped: {site['status']}"] += 1
    return kept, tally
