"""Separate new-definition expansion savings from the frozen syntax ledger.

Uses follow the reference pricer's dependency, scope and unfolding-hint rules.
Definition calls carry their full expansion credit; notation and macro calls
carry only the definition credit nested inside their templates. Declaration
costs follow the uses, so the split preserves signed token reconciliation.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path


def origin_categories(path: Path, origin: dict) -> dict:
    """Read an optional revised attribution without changing frozen artifacts."""
    sidecar = path.with_name("new-definition-savings.json")
    if "new_definition_savings" in origin["categories"] or not sidecar.exists():
        return origin["categories"]
    revised = json.loads(sidecar.read_text())
    if revised["origins_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("Definition savings belong to a different origin ledger")
    categories = revised["categories"]
    original = origin["categories"]
    if (set(categories) != set(original) | {"new_definition_savings"}
            or sum(categories.values()) != sum(original.values())
            or any(categories[k] != v for k, v in original.items() if k != "syntax_optimization")
            or categories["new_definition_savings"] != revised["audit"]["uses"] - revised["audit"]["cost"]):
        raise ValueError("Definition savings do not reconcile with the frozen origin ledger")
    return categories


def definition_prices(abbreviations, pricer):
    """Copy templates with only the portion attributable to new definitions."""
    memo = {}

    def saving(abbreviation, stack=frozenset()):
        key = id(abbreviation)
        if key in stack or not abbreviation.priced:
            return 0
        if key in memo:
            return memo[key]
        if abbreviation.kind == "definition":
            value = abbreviation.saving
        elif abbreviation.generated_category:
            # Generated and measured macros are priced one expansion step deep.
            value = 0
        else:
            visible = {id(other): other for name in abbreviation.names for other in pricer.visible(name)}
            value = sum(
                abbreviation.nested.get(" ".join(other.keys[0]), 0) * saving(other, stack | {key})
                for other in visible.values() if other.keys and other is not abbreviation
            )
        memo[key] = value
        return value

    return {id(a): replace(a, saving=saving(a)) for a in abbreviations}


def split_definition_savings(result, categories, before, after, before_source, after_source):
    """Split an identity-corrected ledger using its exact archived source pair.

    Imports stay local because this module also ships into an offline container
    alongside the vendored lean-lean classifier (third_party/lean-lean).
    """
    from scripts.analysis import abbreviations, proof_hunks, source_accounting
    from scripts.analysis.ledger import declaration_ledger

    if "new_definition_savings" in categories:
        return dict(categories), result["detail"]["new_definition_savings"]
    before = source_accounting.with_source_only_declarations(before, before_source)
    after = source_accounting.with_source_only_declarations(after, after_source)
    retained = set(before.nodes) & set(after.nodes)
    old_partition = source_accounting.partition(before, before_source, retained)
    new_partition = source_accounting.partition(after, after_source, retained)
    templates, _ = abbreviations.build(before, after, before_source, after_source, new_partition.tokens)
    definition_names = {name for a in templates if a.command in {"def", "abbrev"} for name in a.names}
    # Respect template prices superseded by actual one-step Lean measurements.
    for template in templates:
        if any(result["declarations"].get(n, {}).get("note") == "priced per site by Lean" for n in template.names):
            template.priced = False
    pricer = abbreviations.Pricer(templates, after, after_source)
    prices = definition_prices(templates, pricer)
    definitions = {n.rsplit(".", 1)[-1] for graph in (before, after)
                   for n, node in graph.nodes.items() if str(node.get("kind")) != "theorem"}
    lemmas = {n.rsplit(".", 1)[-1] for graph in (before, after)
              for n, node in graph.nodes.items() if str(node.get("kind")) == "theorem"}
    uses, costs = {}, {}
    previous_automation = proof_hunks.EXTRA_AUTOMATION
    proof_hunks.EXTRA_AUTOMATION = abbreviations.automation_macros(templates)
    try:
        for name, row in result["declarations"].items():
            node = after.nodes.get(name, before.nodes.get(name, {}))
            if node.get("identity_status") == "copy":
                continue
            if row["status"] == "syntax" and name in definition_names:
                costs[name] = row["cost"]
                continue
            if row["status"] not in {"added", "structural", "simplification"} or row.get("shared"):
                continue
            visible = [prices[id(a)] for a in pricer.visible(name) if prices[id(a)].saving]
            if not visible:
                continue
            new = new_partition.texts.get(name)
            if not new or (row["status"] == "added" and not new_partition.tokens.get(name, 0)):
                continue
            price = lambda steps: abbreviations.Pricer.price_steps(steps, visible)
            if row["status"] == "added":
                credit = price([abbreviations.tokens(new)])
            else:
                old = old_partition.texts[name]
                path = after_source.path_of_module(str(node.get("module", "")))
                found = proof_hunks.analyse(old, new, definitions=definitions, lemmas=lemmas,
                                            after_file=after_source._files.get(path or "", ""))
                credit = declaration_ledger(old, new, found, definitions, lemmas, price=price).uses
            if credit:
                uses[name] = credit
    finally:
        proof_hunks.EXTRA_AUTOMATION = previous_automation
    audit = dict(version=1, uses=sum(uses.values()), cost=sum(costs.values()),
                 uses_by_declaration=uses, costs_by_declaration=costs)
    net = audit["uses"] - audit["cost"]
    split = {**categories, "syntax_optimization": categories.get("syntax_optimization", 0) - net,
             "new_definition_savings": net}
    if sum(split.values()) != result["scored_tokens_saved"]:
        raise ValueError("Definition savings do not reconcile to the measured token reduction")
    result["detail"]["new_definition_savings"] = audit
    return split, audit


def main():
    """Backfill saved ledgers; invoke this file directly to load the reference."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[2] / "third_party" / "lean-lean")
    parser.add_argument("directories", type=Path, nargs="+", help="directories containing origins.json and ledger.json")
    args = parser.parse_args()
    sys.path[:0] = [str(args.reference.resolve()), str(args.reference.resolve() / "src")]
    from scripts.analysis.quantify_compression_categories import SourceIndex, load_graph

    def read(path):
        return json.loads(path.read_text())

    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    for directory in args.directories:
        path = directory / "origins.json"
        origin, status = read(path), read(directory / "status.json")
        if status["status"] != "passed" or digest(path) != status["origins_sha256"]:
            raise ValueError(f"Unverified or changed origin artifact: {directory}")
        sources = {}
        for phase in ("before", "after"):
            archive = directory / phase / "original-sources.tar.gz"
            if not archive.is_file():
                # Saved ledgers link to the container's /inputs mount. Resolve
                # it through the recorded command, then verify the archive hash.
                mounts = [value.removesuffix(":/inputs:ro") for value in status["command"]
                          if value.endswith(":/inputs:ro")]
                if len(mounts) != 1:
                    raise ValueError(f"Cannot resolve recorded source mount: {directory}")
                archive = Path(mounts[0]) / phase / "original-sources.tar.gz"
            if digest(archive) != status["inputs"][f"{phase}/original-sources.tar.gz"]:
                raise ValueError(f"Changed source archive: {archive}")
            sources[phase] = SourceIndex.from_archive(archive)
            scope = read(directory / "record.json")["metric_scope"]
            sources[phase].restrict(exclude_files=scope.get("excluded_files", []),
                                   exclude_dirs=scope.get("exclude_dirs", []),
                                   include_prefix=scope.get("include_prefix", ""))
        if (sources["before"].total_tokens(), sources["after"].total_tokens()) != (
                origin["baseline_tokens"], origin["final_tokens"]):
            raise ValueError(f"Source tokens differ from verified metrics: {directory}")
        categories, audit = split_definition_savings(
            read(directory / "ledger.json"), origin["categories"],
            load_graph(directory / "before.json"), load_graph(directory / "after.json"),
            sources["before"], sources["after"],
        )
        report = dict(origins_sha256=digest(path), adapter_sha256=digest(Path(__file__)),
                      graphs_sha256={phase: digest(directory / f"{phase}.json") for phase in sources},
                      categories=categories, audit=audit)
        destination = directory / "new-definition-savings.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        temporary.replace(destination)
        print(f'{origin["model"]} P{origin["problem_idx"]}: {categories["new_definition_savings"]:+,} definition tokens', flush=True)


if __name__ == "__main__":
    main()
