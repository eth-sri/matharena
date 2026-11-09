"""Track declaration identities across files and account for directed deletions.

Moved declarations retain identity; physical copies keep separate token costs.
"""
from __future__ import annotations
import collections
import hashlib
from dataclasses import replace
from leanlean.metrics.tokens import remove_lean_comments
from scripts.analysis.source_accounting import _COMMAND, _NAME
from scripts.analysis import source_accounting, classify_compression as classifier
from scripts.analysis.quantify_compression_categories import (
    EDGE_KIND_BITS, IDENTITY_GRAPH_SCHEMA, MASKED_GRAPH_SCHEMAS,
)

KINDS = {"theorem", "lemma", "def", "abbrev", "opaque", "axiom", "constant",
         "class", "structure", "inductive", "instance"}


def declarations(path, text):
    """Conservative conventional command scanner for source identity.

    Handles namespace/section nesting; skips anonymous declarations and
    indented commands inside proofs. Does not expand macros or elaborate types.
    """
    clean = remove_lean_comments(text)
    lines = clean.splitlines()
    masked = remove_lean_comments(text, mask_strings=True).splitlines()
    commands = [(i, m) for i, line in enumerate(masked)
                if (m := _COMMAND.match(line)) and not (m.group("indent") or m.group("hash_indent"))]
    scopes, result = [], []
    for index, (start, match) in enumerate(commands):
        kind = match.group("kind")
        rest = lines[start][match.end():]
        named = _NAME.match(rest)
        label = named.group(1) if named else None
        if kind in {"namespace", "section"}:
            scopes.append(label if kind == "namespace" and label else "")
        elif kind == "end":
            if scopes:
                scopes.pop()
        elif kind in KINDS and label:
            namespace = ".".join(s for s in scopes if s)
            name = label.removeprefix("_root_.") if label.startswith("_root_.") else ".".join(
                x for x in (namespace, label) if x)
            end = commands[index + 1][0] if index + 1 < len(commands) else len(lines)
            normalized = " ".join("\n".join(lines[start:end]).split()).removeprefix("private ")
            result.append(dict(name=name, path=path, line=start + 1, kind=kind,
                digest=hashlib.sha256(normalized.encode()).hexdigest()))
    return result


def reachable(graph, roots):
    dependencies = collections.defaultdict(set)
    for a, b in graph.edges:
        dependencies[a].add(b)
    seen, pending = set(), list(roots)
    while pending:
        name = pending.pop()
        if name not in seen:
            seen.add(name)
            pending.extend(dependencies[name] - seen)
    return seen


def user_name(row):
    prefix = "_private." + row["module"] + "."
    name = row["name"]
    if name.startswith(prefix):
        counter, separator, suffix = name[len(prefix):].partition(".")
        if separator and counter.isdigit():
            return suffix
    return name


REMOVAL_KEYS = {"dead": "dead_code", "deleted": "deleted"}


def baseline_dead_nodes(original, protected):
    """Deadness can only originate in the original directed graph.

    Source-only additions to the declaration index are not evidence of original
    unreachability. Missing roots must fail closed, rather than label everything
    unreachable. Later reachability is deliberately not an input.
    """
    if not protected or set(protected)-original.nodes.keys():
        raise ValueError('Original protected declarations are missing')
    return {n for n,node in original.nodes.items() if not node.get('source_only')} - reachable(original,protected)


def directed_removal_group(baseline, after):
    """Original nodes absent and unreachable from all surviving original nodes.

    Edges point from a declaration to its dependencies. A deleted caller may
    share a retained helper; that does not make the caller used by the helper.
    A retained caller, including a moved node or source copy, blocks removal
    of every dependency it could reach in the original graph.
    """
    present = set(after.nodes) | {node.get("logical_name", name) for name, node in after.nodes.items()}
    surviving = baseline.nodes.keys() & present
    return baseline.nodes.keys() - present - reachable(baseline, surviving)


def removal_status(name, baseline_dead, removed_group):
    """Baseline deadness plus directed removal of its original users."""
    if name in baseline_dead:
        return "dead" if name in removed_group else "deleted"
    return "deleted"


KEYS = ["dead_code", "syntax_optimization", "automation", "proof_simplification",
        "structural_diff", "deleted", "added"]


class Identities:
    """One logical ID across files; separate physical copies point to that ID."""
    def __init__(self):
        self.file_cache = {}
        self.primary_paths = {}

    def align(self, graph, source):
        augmented = source_accounting.with_source_only_declarations(graph, source)
        source_names = {}
        for path in source.paths():
            text = source._files[path]
            key = path, hashlib.sha256(text.encode()).hexdigest()
            if key not in self.file_cache:
                self.file_cache[key] = declarations(path, text)
            for row in self.file_cache[key]:
                source_names[path, row["line"]] = row["name"]
        groups = collections.defaultdict(list)
        for raw, node in augmented.nodes.items():
            if node.get("source_only"):
                path = source.path_of_module(node["module"])
                logical = source_names.get((path, node["start_line"]), raw)
            else:
                logical = user_name(dict(name=raw, module=node["module"]))
            groups[logical].append(raw)
        mapping, nodes, ambiguous = {}, {}, []
        for logical, members in sorted(groups.items()):
            compiled = [n for n in members if not augmented.nodes[n].get("source_only")]
            if len(compiled) > 1:
                # Distinct private declarations can legitimately have the same user name.
                ambiguous.append(dict(logical_name=logical, raw_names=sorted(members)))
                for raw in members:
                    mapping[raw] = raw
                    nodes[raw] = dict(augmented.nodes[raw], name=raw, original_name=raw,
                                     logical_name=raw, identity_status="ambiguous")
                continue
            previous = [n for n in members
                        if augmented.nodes[n]["module"] == self.primary_paths.get(logical)]
            primary = compiled[0] if compiled else previous[0] if len(previous) == 1 else sorted(members)[0]
            for raw in members:
                node = augmented.nodes[raw]
                copy = raw != primary
                stable = f"__copy__:{logical}:{raw}" if copy else logical
                assert stable not in nodes
                mapping[raw] = stable
                nodes[stable] = dict(node, name=stable, original_name=raw, logical_name=logical,
                    identity_status="copy" if copy else "detached" if node.get("source_only") else "compiled")
            self.primary_paths[logical] = augmented.nodes[primary]["module"]
        def edges(values):
            return {(mapping[a], mapping[b]) for a, b in values}
        aligned = replace(augmented, nodes=nodes,
                          static_edges=edges(augmented.static_edges),
                          grind_edges=edges(augmented.grind_edges),
                          layers={k: edges(v) for k, v in augmented.layers.items()})
        assert len(aligned.nodes) == len(augmented.nodes)
        assert len(aligned.edges) == len(augmented.edges)
        return aligned, mapping, ambiguous


def graph_payload(graph):
    production = dict(grind_edges=sorted(graph.grind_edges))
    if graph.schema not in MASKED_GRAPH_SCHEMAS:
        return dict(schema="sendov_identity_graph_v1", nodes=list(graph.nodes.values()),
                    edges=sorted(graph.static_edges), policies=dict(production=production))
    # keep each edge's evidence layers: the structural rule reads kernel/ilean
    masks = collections.Counter()
    for kind, bit in EDGE_KIND_BITS.items():
        for pair in graph.layers.get(kind, ()):
            masks[pair] |= bit
    assert set(masks) == graph.static_edges
    return dict(schema=IDENTITY_GRAPH_SCHEMA, nodes=list(graph.nodes.values()),
                edges=[[a, b, masks[a, b]] for a, b in sorted(graph.static_edges)],
                policies=dict(production=production))


def corrected_categories(result, before, after, dead_ids, copy_delta, removed_group):
    """Keep the existing proof ledger; correct deletion provenance and copies."""
    parts = classifier.parts(result)
    counts = {k: parts.get(k, 0) for k in KEYS}
    changes = []
    copy_contribution = 0
    for name, row in result["declarations"].items():
        status = row["status"]
        node = after.nodes.get(name, before.nodes.get(name))
        is_copy = node["identity_status"] == "copy"
        if status in {"dead", "deleted"} and not is_copy:
            new_status = removal_status(name, dead_ids, removed_group)
            row["original_status"] = status
            if new_status != status:
                tokens = row["tokens"]
                counts[REMOVAL_KEYS[status]] -= tokens
                counts[REMOVAL_KEYS[new_status]] += tokens
                row["status"] = new_status
                changes.append(dict(name=name, old_status=status, new_status=new_status, tokens=tokens))
        if not is_copy:
            continue
        if status in {"dead", "deleted"}:
            counts["dead_code" if status == "dead" else "deleted"] -= row["tokens"]
            copy_contribution += row["tokens"]
        elif status == "added":
            syntax = row.get("syntax", 0)
            counts["added"] += row["tokens"] + syntax
            counts["syntax_optimization"] -= syntax
            copy_contribution -= row["tokens"]
        elif status == "syntax":
            counts["syntax_optimization"] += row["cost"]
            copy_contribution -= row["cost"]
        else:
            counts["syntax_optimization"] -= row["syntax"]
            counts["automation"] -= row["automation"]
            counts["structural_diff" if status == "structural" else "proof_simplification"] -= row["rest"]
            copy_contribution += row["delta"]
        row["status"] = "added" if name in after.nodes else "deleted"
    assert copy_contribution == copy_delta, (copy_contribution, copy_delta)
    counts["added"] += copy_contribution
    assert sum(counts.values()) == result["scored_tokens_saved"]
    return counts, changes
