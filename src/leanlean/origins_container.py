"""Apply lean-lean's frozen compression ledger to one verified graph pair.

The reference analysis is mounted separately. Lean macro measurements run in
this offline container using the project's own toolchain.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from graph_identity import adapt, provider_keys
from definition_savings import split_definition_savings


def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_sequence_driver(driver: str) -> str:
    """Print every command returned by a macro's mkNullNode sequence.

    ppCategory expects one command. Passing a sequence makes Lean try to
    resolve a constant named `null`, losing all of the macro's expansion.
    Keep the frozen reference's parsing and single expansion step intact.
    """
    original = '''      let fmt ← liftCoreM (do
        try ppCategory printCat out catch _ => ppCategory cat out)
      IO.println s!"@@SITE {ident} OK {consumed} {chosen}"
      IO.println "@@BEGIN"
      IO.println (toString fmt)
      IO.println "@@END"'''
    replacement = '''      let outputs := if cat == `command && out.getKind == nullKind then out.getArgs else #[out]
      IO.println s!"@@SITE {ident} OK {consumed} {chosen}"
      IO.println "@@BEGIN"
      for node in outputs do
        let fmt ← liftCoreM (do
          try ppCategory printCat node catch _ => ppCategory cat node)
        IO.println (toString fmt)
      IO.println "@@END"'''
    if driver.count(original) != 1:
        raise ValueError("Unsupported reference macro driver")
    return driver.replace(original, replacement)


def apply_renames(graph, before, renames: dict) -> tuple[object, dict, dict]:
    """Give after-graph declarations their checkpoint-traced baseline names.

    A rename applies only when the new name did not exist before, the baseline
    name exists in the before graph, and nothing else in the after graph uses it,
    so a renamed declaration is matched as retained instead of deleted and added.
    """
    from dataclasses import replace

    applied, skipped = {}, []
    taken = set(graph.nodes)
    for new, old in sorted(renames.items()):
        node = graph.nodes.get(new)
        reason = (
            "not in after graph" if node is None else
            "existed before" if new in before.nodes else
            "baseline name absent" if old not in before.nodes else
            "baseline name in use" if old in taken or old in applied.values() else
            None
        )
        if reason:
            skipped.append({"name": new, "baseline_name": old, "reason": reason})
        else:
            applied[new] = old
    nodes = {}
    for name, node in graph.nodes.items():
        target = applied.get(name, name)
        nodes[target] = dict(node, name=target, logical_name=target, renamed_from=name) if target != name else node

    def edges(values):
        return {(applied.get(a, a), applied.get(b, b)) for a, b in values}

    renamed = replace(graph, nodes=nodes, static_edges=edges(graph.static_edges), grind_edges=edges(graph.grind_edges),
                      layers={k: edges(v) for k, v in graph.layers.items()})
    assert len(renamed.nodes) == len(graph.nodes)
    return renamed, applied, {"applied": applied, "skipped": skipped}


def main() -> None:
    from scripts.analysis import checkpoint_identity as identity
    from scripts.analysis import classify_compression as classifier
    from scripts.analysis import expand_macro_programs, program_sites, source_accounting
    from scripts.analysis.quantify_compression_categories import GraphData, SourceIndex, load_graph
    from leanlean.metrics.tokens import remove_lean_comments

    inputs, output = Path("/inputs"), Path("/analysis")
    job = json.loads((inputs / "result.json").read_text())
    if job["status"] != "verified":
        raise ValueError("Compression attribution requires a verified submission")
    entry = job["entry"]
    toolchain = entry["lean_toolchain"].replace("/", "--").replace(":", "---")
    expand_macro_programs.TOOLCHAIN_ROOT = Path("/opt/elan/toolchains")
    expand_macro_programs.PREFERRED_TOOLCHAINS = (toolchain,)
    original_driver = expand_macro_programs.build_driver
    expand_macro_programs.build_driver = lambda *args, **kwargs: command_sequence_driver(
        original_driver(*args, **kwargs)
    )
    project_prefix = entry["project_path"]
    included = {
        str(Path(project_prefix) / name) for phase in ("before", "after") for name in job["metrics"][phase]["files"]
    }
    sources = {
        phase: SourceIndex.from_archive(inputs / phase / "original-sources.tar.gz") for phase in ("before", "after")
    }
    from macro_driver import install
    macro_audit = {"version": 1, "parser_context": "core Lean and Lean.Parser.Tactic"}
    install(expand_macro_programs, program_sites, sources["after"]._files, job, macro_audit)
    excluded = sorted({path for source in sources.values() for path in source.paths()} - included)
    scope = {"excluded_files": excluded, "exclude_dirs": [], "include_prefix": project_prefix}
    tracker = identity.Identities()
    renames_path = output / "renames.json"
    renames = json.loads(renames_path.read_text())["renames"] if renames_path.exists() else {}
    rename_audit = {"applied": {}, "skipped": []}
    payloads = {
        phase: json.loads((inputs / phase / "dependency-graph.json").read_text()) for phase in ("before", "after")
    }
    uses_v3 = any(p.get("schema") == "preprocessing_dependency_graph_v3" for p in payloads.values())
    physical_keys, physical_audit = {}, {}
    if uses_v3:

        def text(phase, node):
            value = sources[phase].slice(node)
            return remove_lean_comments(value) if value is not None else None

        physical_keys, physical_audit = provider_keys(payloads, text)
    graph_audits = {}
    graphs, mappings, ambiguities, copies, provenance = {}, {}, {}, {}, {}
    for phase in ("before", "after"):
        source = sources[phase]
        source.restrict(exclude_files=excluded, include_prefix=project_prefix)
        expected = job["metrics"][phase]["lean_tokens"]
        if source.total_tokens() != expected:
            raise ValueError(f"{phase}: archive token count differs from verification metrics")
        path = inputs / phase / "dependency-graph.json"
        if uses_v3:
            data, graph_audits[phase] = adapt(payloads[phase], physical_keys[phase])
            raw = GraphData(**data)
        else:
            raw = load_graph(path)  # validates the graph's embedded checksum
        graph, mapping, ambiguous = tracker.align(raw, source)
        if phase == "after" and renames:
            graph, applied, rename_audit = apply_renames(graph, graphs["before"], renames)
            mapping = {raw_name: applied.get(stable, stable) for raw_name, stable in mapping.items()}
        graphs[phase], mappings[phase], ambiguities[phase] = graph, mapping, ambiguous
        partition = source_accounting.partition(graph, source)
        copies[phase] = sum(
            partition.tokens.get(name, 0) for name, node in graph.nodes.items() if node.get("identity_status") == "copy"
        )
        write(output / f"{phase}.json", identity.graph_payload(graph))
        (output / phase).mkdir(exist_ok=True)
        archive = output / phase / "original-sources.tar.gz"
        if not archive.exists():
            archive.symlink_to(inputs / phase / "original-sources.tar.gz")
        provenance[phase] = {
            "graph_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "nodes": len(raw.nodes),
            "edges": len(raw.edges),
        }
    protected = entry.get("theorem_names", []) + entry.get("definition_names", [])
    if uses_v3:
        missing = set(protected) - {n["name"] for n in payloads["before"]["nodes"]}
        if missing:
            raise ValueError(f"Protected declarations missing: {sorted(missing)}")
        roots = [
            mappings["before"][physical_keys["before"][(n["module"], n["name"])]]
            for n in payloads["before"]["nodes"]
            if n["name"] in protected
        ]
    else:
        roots = [mappings["before"].get(name, name) for name in protected]
    dead = identity.baseline_dead_nodes(graphs["before"], roots)
    metrics = job["metrics"]
    record = {
        "metric_scope": scope,
        # a deleted protected result keeps its old dependency closure live
        "protected_declarations": roots,
        "reconciliation": {
            "scored_baseline_tokens": metrics["before"]["lean_tokens"],
            "scored_post_tokens": metrics["after"]["lean_tokens"],
            "scored_tokens_saved": metrics["tokens_saved"],
        },
    }
    write(output / "record.json", record)
    model, repository = os.environ.get("LEANLEAN_ANALYSIS_MODEL", "leanlean"), job["key"]
    cache = output / "program-sites"
    observation = program_sites.measure(model, repository, output, record, cache=cache)
    write(output / "macro-driver-audit.json", macro_audit)
    if observation.get("error"):
        raise RuntimeError(f'Macro measurement failed: {observation["error"]}')
    result = classifier.classify(repository, model, output, record, {}, set(), program_cache=cache)
    if result is None:
        raise ValueError("Reference classifier could not load the graph/source pair")
    counts, corrections = identity.corrected_categories(
        result,
        graphs["before"],
        graphs["after"],
        dead,
        copies["before"] - copies["after"],
        identity.directed_removal_group(graphs["before"], graphs["after"]),
    )
    counts, definition_audit = split_definition_savings(
        result, counts, graphs["before"], graphs["after"], sources["before"], sources["after"]
    )
    if sum(counts.values()) != metrics["tokens_saved"]:
        raise ValueError("Origin categories do not reconcile to the measured token reduction")
    write(output / "ledger.json", result)
    write(
        output / "identity-audit.json",
        {
            "ambiguities": ambiguities,
            "corrections": corrections,
            "new_definition_savings": definition_audit,
            "copy_tokens": copies,
            "protected_roots": roots,
            "physical_identities": physical_audit,
            "graph_adaptation": graph_audits,
            "checkpoint_renames": rename_audit,
        },
    )
    write(
        output / "origins.json",
        {
            "status": "passed",
            "model": model,
            "problem_idx": job["problem_idx"],
            "key": repository,
            "baseline_tokens": metrics["before"]["lean_tokens"],
            "final_tokens": metrics["after"]["lean_tokens"],
            "tokens_saved": metrics["tokens_saved"],
            "categories": counts,
            "graphs": provenance,
            "program_site_status": result["detail"].get("program_site_status", {}),
            "toolchain": toolchain,
            "dependency_ambiguity_policy": "all candidates retained conservatively" if uses_v3 else "legacy",
            "unresolved_provider_moves": physical_audit.get("unresolved_moves", []),
        },
    )


if __name__ == "__main__":
    main()
