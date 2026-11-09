"""Adapt lean-strip v3 physical declarations to the reference analysis graph.

Names remain unchanged when unique. Duplicate providers keep separate keys;
only an unambiguous, unchanged source body can move a provider across modules.
Ambiguous dependency targets contribute all candidates to conservative
reachability and remain explicitly recorded in the audit.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict


def checksum(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps({k: v for k, v in payload.items() if k != "sha256"}, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate(payload: dict) -> None:
    if payload.get("schema") != "preprocessing_dependency_graph_v3" or payload.get("sha256") != checksum(payload):
        raise ValueError("Invalid v3 graph schema or checksum")
    identities = [(node["module"], node["name"]) for node in payload["nodes"]]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate physical declaration identity")


def provider_keys(payloads: dict, source_text) -> tuple[dict, dict]:
    """Give duplicate providers stable keys across the two measured source states."""
    duplicate_names = set()
    for graph in payloads.values():
        validate(graph)
        duplicate_names.update(name for name, n in Counter(node["name"] for node in graph["nodes"]).items() if n > 1)
    keys = {}
    for phase, graph in payloads.items():
        keys[phase] = {
            (n["module"], n["name"]): (
                "__provider__:" + json.dumps([n["module"], n["name"]], separators=(",", ":"))
                if n["name"] in duplicate_names
                else n["name"]
            )
            for n in graph["nodes"]
        }
        if len(set(keys[phase].values())) != len(keys[phase]):
            raise ValueError("Synthetic provider key collides with a declaration name")
    moves, unresolved = [], []
    before, after = ({(n["module"], n["name"]): n for n in payloads[p]["nodes"]} for p in ("before", "after"))
    for name in sorted(duplicate_names):
        removed = sorted(i for i in before.keys() - after.keys() if i[1] == name)
        added = sorted(i for i in after.keys() - before.keys() if i[1] == name)
        if not removed or not added:
            continue
        old, new = defaultdict(list), defaultdict(list)
        for phase, candidates, nodes, grouped in (("before", removed, before, old), ("after", added, after, new)):
            for identity in candidates:
                text = source_text(phase, nodes[identity])
                if text is not None and text.strip():
                    # Do not normalize whitespace inside Lean string literals.
                    grouped[text.strip()].append(identity)
        matched = set()
        for body in sorted(old.keys() & new.keys()):
            if len(old[body]) == len(new[body]) == 1:
                source, target = old[body][0], new[body][0]
                keys["after"][target] = keys["before"][source]
                moves.append(
                    dict(name=name, before=list(source), after=list(target), evidence="unique unchanged source")
                )
                matched.add(target)
        if any(i not in matched for i in added):
            unresolved.append(
                dict(name=name, removed=[list(i) for i in removed], added=[list(i) for i in added if i not in matched])
            )
    return keys, dict(duplicate_names=sorted(duplicate_names), moves=moves, unresolved_moves=unresolved)


def adapt(payload: dict, keys: dict) -> tuple[dict, dict]:
    validate(payload)
    raw = payload["nodes"]
    names = [keys[(node["module"], node["name"])] for node in raw]
    if len(set(names)) != len(names):
        raise ValueError("Provider mapping merged separate declarations")
    nodes = {name: {**node, "name": name, "lean_name": node["name"]} for name, node in zip(names, raw)}
    layers = {name: set() for name in ("source_text", "kernel", "ilean", "grind")}
    bits = {"source_text": 1, "kernel": 2, "ilean": 4}

    def at(index):
        if type(index) is not int or not 0 <= index < len(names):
            raise ValueError(f"Invalid graph endpoint: {index}")
        return names[index]

    def edge(source, target, mask):
        if type(mask) is not int or not 0 < mask <= 7:
            raise ValueError(f"Invalid edge mask: {mask}")
        pair = at(source), at(target)
        for layer, bit in bits.items():
            if mask & bit:
                layers[layer].add(pair)

    for source, target, mask in payload["edges"]:
        edge(source, target, mask)
    ambiguous = payload.get("ambiguous_edges", [])
    for row in ambiguous:
        if not row["source_candidates"] or not row["target_candidates"]:
            raise ValueError("Ambiguous edge has no candidate endpoints")
        for source in row["source_candidates"]:
            for target in row["target_candidates"]:
                edge(source, target, row["kind_mask"])
    policy = payload.get("policies", {}).get("production", {})
    layers["grind"].update((at(a), at(b)) for a, b in policy.get("grind_edges", []))
    grind_ambiguities = policy.get("grind_recovery", {}).get("ambiguous_targets", [])
    for row in grind_ambiguities:
        source = keys[tuple(row["source"])]
        if not row["candidates"]:
            raise ValueError("Ambiguous Grind edge has no targets")
        layers["grind"].update((source, keys[tuple(target)]) for target in row["candidates"])
    protected = [at(i) for i in policy.get("protected", [])]
    return dict(
        nodes=nodes,
        layers=layers,
        static_edges=set.union(*(layers[k] for k in bits)),
        grind_edges=layers["grind"],
        schema=payload["schema"],
        sha256=payload["sha256"],
    ), dict(
        protected=protected,
        ambiguous_edges=ambiguous,
        grind_ambiguities=grind_ambiguities,
        ambiguity_policy="all candidate edges retained conservatively; no single provider inferred",
    )
