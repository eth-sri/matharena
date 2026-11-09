"""Persist and project declaration dependency graphs for preprocessing review."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


GRAPH_SCHEMA = "preprocessing_dependency_graph_v2"
# v1 artifacts carry one priority-ordered kind string per edge; v2 carries a
# multi-label bitmask. Readers accept both so existing datasets stay valid.
GRAPH_SCHEMAS = (GRAPH_SCHEMA, "preprocessing_dependency_graph_v1")
# Source text first: it is the layer a reader can check against the file in
# front of them, so it leads the presentation. The order no longer decides
# attribution - every edge records *every* layer that found it (EDGE_KIND_BITS).
EDGE_KINDS = ("source_text", "kernel", "ilean")
EDGE_KIND_BITS = {"source_text": 1, "kernel": 2, "ilean": 4}



def graph_sha256(payload: Mapping[str, Any]) -> str:
    """Hash the complete graph envelope, excluding only its hash field."""

    canonical_payload = dict(payload)
    canonical_payload.pop("sha256", None)
    canonical = json.dumps(
        canonical_payload, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()



def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _name_set(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    names: set[str] = set()
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("name")
        if isinstance(item, str) and item:
            names.add(item)
    return names


def policy_state(
    report: Mapping[str, Any],
    node_names: list[str],
    node_modules: Mapping[str, str] | None = None,
) -> dict[str, list[int]]:
    """Map one preprocessing report onto exact graph-node status indexes."""

    live = _name_set(_mapping(report.get("module_pruning")).get("live_declarations"))
    dropped = _name_set(report.get("dropped_decls"))
    protected = _name_set(report.get("protected_declarations"))
    recorded_kept = report.get("kept_declarations")
    if isinstance(recorded_kept, list):
        kept = _name_set(recorded_kept)
    else:
        kept_modules = _name_set(
            _mapping(report.get("module_pruning")).get("kept_modules")
        )
        if node_modules is not None and kept_modules:
            kept = {
                name
                for name in node_names
                if node_modules.get(name) in kept_modules
            } - dropped
        else:
            kept = live - dropped
    by_name = {name: index for index, name in enumerate(node_names)}
    kept_indexes = sorted(by_name[name] for name in kept if name in by_name)
    protected_indexes = sorted(
        by_name[name] for name in protected if name in by_name
    )
    kept_set = set(kept_indexes)
    return {
        "kept": kept_indexes,
        "removed": [
            index for index in range(len(node_names)) if index not in kept_set
        ],
        "protected": protected_indexes,
        "status_source": str(
            report.get("_graph_status_source") or "preprocessing_report"
        ),
        "grind_evidence_source": str(
            report.get("_grind_evidence_source") or "preprocessing_report"
        ),
    }


def _line_at_offset(source: str, offset: Any) -> int | None:
    if not isinstance(offset, int) or offset < 0 or offset > len(source):
        return None
    return source.count("\n", 0, offset) + 1


def _owner_at_line(
    declarations: list[Mapping[str, Any]], line: int
) -> str | None:
    candidates = [
        declaration
        for declaration in declarations
        if isinstance(declaration.get("start_line"), int)
        and isinstance(declaration.get("end_line"), int)
        and int(declaration["start_line"]) <= line <= int(declaration["end_line"])
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda declaration: (
            int(declaration["end_line"]) - int(declaration["start_line"]),
            str(declaration.get("name") or ""),
        )
    )
    name = candidates[0].get("name")
    return str(name) if isinstance(name, str) and name else None


def _resolve_rule(name: str, node_names: set[str]) -> str | None:
    if name in node_names:
        return name
    candidates = sorted(
        candidate
        for candidate in node_names
        if candidate.endswith("." + name)
    )
    return candidates[0] if len(candidates) == 1 else None


def grind_edges(
    report: Mapping[str, Any],
    *,
    declarations: list[Mapping[str, Any]],
    originals: Mapping[str, str],
    module_paths: Mapping[str, str],
) -> tuple[list[tuple[str, str]], dict[str, int]]:
    """Recover explicit Grind-rule edges from recorded query/replay decisions."""

    by_path: dict[str, list[Mapping[str, Any]]] = {}
    for declaration in declarations:
        module = declaration.get("module")
        path = module_paths.get(str(module))
        if path:
            by_path.setdefault(path, []).append(declaration)
    node_names = {
        str(declaration.get("name"))
        for declaration in declarations
        if declaration.get("name")
    }
    edges: set[tuple[str, str]] = set()
    stats = {"calls": 0, "resolved": 0, "owner_unresolved": 0, "rule_unresolved": 0}
    files = _mapping(report.get("grind_normalization")).get("files")
    if not isinstance(files, list):
        return [], stats
    for file_row in files:
        if not isinstance(file_row, Mapping):
            continue
        path = str(file_row.get("path") or "")
        source = originals.get(path)
        declarations_in_file = by_path.get(path, [])
        calls = file_row.get("calls")
        if not isinstance(source, str) or not isinstance(calls, list):
            continue
        for call in calls:
            if not isinstance(call, Mapping):
                continue
            normalization = _mapping(call.get("normalization"))
            retained = normalization.get("retained_parameters")
            if not isinstance(retained, list) or not retained:
                continue
            stats["calls"] += 1
            line = _line_at_offset(source, call.get("tactic_offset"))
            owner = (
                _owner_at_line(declarations_in_file, line)
                if line is not None
                else None
            )
            if owner is None:
                stats["owner_unresolved"] += 1
                continue
            call_resolved = False
            for value in retained:
                if not isinstance(value, str) or not value:
                    continue
                target = _resolve_rule(value, node_names)
                if target is None:
                    stats["rule_unresolved"] += 1
                    continue
                edges.add((owner, target))
                call_resolved = True
            if call_resolved:
                stats["resolved"] += 1
    return sorted(edges), stats


def build_graph_artifact(
    *,
    repository: str,
    source_image: str,
    source_image_id: str,
    declarations: list[Mapping[str, Any]],
    kernel_edges: set[tuple[str, str]],
    ilean_edges: set[tuple[str, str]],
    source_text_edges: set[tuple[str, str]],
    reports: Mapping[str, Mapping[str, Any]],
    originals: Mapping[str, str],
    module_paths: Mapping[str, str],
) -> dict[str, Any]:
    """Build a compact, typed graph shared by all signature policies."""

    ordered = sorted(
        declarations,
        key=lambda declaration: (
            str(declaration.get("module") or ""),
            int(declaration.get("start_line") or 0),
            str(declaration.get("name") or ""),
        ),
    )
    nodes = [
        {
            "name": str(declaration.get("name") or ""),
            "module": str(declaration.get("module") or ""),
            "kind": str(declaration.get("kind") or "unknown"),
            "start_line": declaration.get("start_line"),
            "end_line": declaration.get("end_line"),
        }
        for declaration in ordered
        if declaration.get("name")
    ]
    node_names = [node["name"] for node in nodes]
    node_modules = {node["name"]: node["module"] for node in nodes}
    by_name = {name: index for index, name in enumerate(node_names)}

    # Each layer is recorded independently: an edge that three methods agree on
    # carries all three bits, so per-layer coverage is readable straight off the
    # artifact instead of being an artefact of evaluation order.
    masks: dict[tuple[int, int], int] = {}
    for kind, edges in (
        ("source_text", source_text_edges),
        ("kernel", kernel_edges),
        ("ilean", ilean_edges),
    ):
        bit = EDGE_KIND_BITS[kind]
        for source, target in edges:
            if source not in by_name or target not in by_name:
                continue
            pair = (by_name[source], by_name[target])
            masks[pair] = masks.get(pair, 0) | bit
    typed_edges: list[list[Any]] = [
        [pair[0], pair[1], mask] for pair, mask in sorted(masks.items())
    ]

    policies: dict[str, Any] = {}
    declaration_rows = [dict(value) for value in ordered if value.get("name")]
    for policy, report in reports.items():
        state = policy_state(report, node_names, node_modules)
        recovered, recovery = grind_edges(
            report,
            declarations=declaration_rows,
            originals=originals,
            module_paths=module_paths,
        )
        state["grind_edges"] = [
            [by_name[source], by_name[target]]
            for source, target in recovered
            if source in by_name and target in by_name
        ]
        state["grind_recovery"] = recovery
        policies[str(policy)] = state

    payload = {
        "schema": GRAPH_SCHEMA,
        "repository": repository,
        "source_image": source_image,
        "source_image_id": source_image_id,
        "nodes": nodes,
        "edges": typed_edges,
        "policies": policies,
        "counts": {
            "nodes": len(nodes),
            "edges": len(typed_edges),
            # Coverage: how many union edges each layer found, independently.
            "source_text_edges": sum(
                edge[2] & EDGE_KIND_BITS["source_text"] != 0
                for edge in typed_edges
            ),
            "kernel_edges": sum(
                edge[2] & EDGE_KIND_BITS["kernel"] != 0 for edge in typed_edges
            ),
            "ilean_edges": sum(
                edge[2] & EDGE_KIND_BITS["ilean"] != 0 for edge in typed_edges
            ),
            # Exclusive: edges no other layer supplies - the necessity figure.
            "source_text_only_edges": sum(
                edge[2] == EDGE_KIND_BITS["source_text"] for edge in typed_edges
            ),
            "kernel_only_edges": sum(
                edge[2] == EDGE_KIND_BITS["kernel"] for edge in typed_edges
            ),
            "ilean_only_edges": sum(
                edge[2] == EDGE_KIND_BITS["ilean"] for edge in typed_edges
            ),
            "all_layers_edges": sum(edge[2] == 7 for edge in typed_edges),
        },
        "edge_kind_bits": dict(EDGE_KIND_BITS),
    }
    payload["sha256"] = graph_sha256(payload)
    return payload


def write_graph_artifact(path: Path, payload: Mapping[str, Any]) -> None:
    complete = dict(payload)
    complete["sha256"] = graph_sha256(complete)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(complete, separators=(",", ":")) + "\n")
    temporary.replace(path)
