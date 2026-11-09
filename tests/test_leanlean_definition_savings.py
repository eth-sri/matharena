"""Definition savings must partition syntax while preserving signed totals."""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from leanlean.definition_savings import definition_prices, origin_categories, split_definition_savings


@dataclass
class Template:
    kind: str
    saving: int
    names: list[str]
    keys: list[list[str]]
    nested: dict[str, int] = field(default_factory=dict)
    priced: bool = True
    generated_category: str = ""


def test_definition_credit_follows_nested_templates_and_visibility():
    definition = Template("definition", 8, ["helper"], [["helper"]])
    notation = Template("notation", 12, ["notation"], [["value"]], {"helper": 1})
    macro = Template("macro", 26, ["macro"], [["useValue"]], {"value": 2})
    unrelated = Template("definition", 1000, ["other.helper"], [["helper"]])
    unpriced = Template("definition", 20, ["recursive"], [["recursive"]], priced=False)
    templates = [definition, notation, macro, unrelated, unpriced]
    visible = {"helper": [], "notation": [definition], "macro": [notation],
               "other.helper": [], "recursive": []}
    prices = definition_prices(templates, SimpleNamespace(visible=lambda name: visible[name]))
    assert [prices[id(a)].saving for a in templates] == [8, 8, 16, 1000, 0]
    assert [a.saving for a in templates] == [8, 12, 26, 1000, 20]


def test_generated_macro_measurements_do_not_invent_deep_expansion_credit():
    generated = Template("generated", 50, ["generated"], [["macro"]], {"helper": 1},
                         generated_category="term")
    prices = definition_prices([generated], SimpleNamespace(visible=lambda name: []))
    assert prices[id(generated)].saving == 0


def test_sidecar_is_bound_to_original_ledger_and_only_splits_syntax(tmp_path):
    path = tmp_path / "origins.json"
    origin = dict(categories=dict(syntax_optimization=20, added=-4))
    path.write_text(json.dumps(origin))
    assert origin_categories(path, origin) == origin["categories"]
    sidecar = path.with_name("new-definition-savings.json")
    report = dict(origins_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  categories=dict(syntax_optimization=13, new_definition_savings=7, added=-4),
                  audit=dict(uses=10, cost=3))
    sidecar.write_text(json.dumps(report))
    assert origin_categories(path, origin) == report["categories"]
    report["categories"]["added"] = -3
    report["categories"]["syntax_optimization"] = 12
    sidecar.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="do not reconcile"):
        origin_categories(path, origin)
    path.write_text("changed ledger")
    with pytest.raises(ValueError, match="different origin ledger"):
        origin_categories(path, origin)


@pytest.fixture
def reference(monkeypatch):
    """Exercise the vendored classifier, or LEANLEAN_REFERENCE when set."""
    import leanlean
    from leanlean.compression_origins import REFERENCE

    root = Path(os.environ.get("LEANLEAN_REFERENCE") or REFERENCE)
    monkeypatch.setattr(leanlean, "__path__", [*leanlean.__path__, str(root / "src/leanlean")])
    monkeypatch.syspath_prepend(str(root))
    from scripts.analysis import classify_compression as classifier
    from scripts.analysis.quantify_compression_categories import SourceIndex

    # Fixtures contain no programmatic macros, so there are no Lean sites to measure.
    monkeypatch.setattr(classifier.program_sites, "measure", lambda *a, **kw: dict(programs=[], sites=[]))
    return classifier, SourceIndex


@pytest.mark.parametrize("command,proof,expected_uses", [
    ("def", "LongValue = LongValue := by rfl", 2),
    ("abbrev", "LongValue = LongValue := by rfl", 2),
    ("def", "True := by simp only [LongValue]", 0),
    ("def", "True := by trivial", 0),
])
def test_frozen_ledger_separates_uses_and_costs(reference, tmp_path, command, proof, expected_uses):
    import io
    import tarfile
    from scripts.analysis.quantify_compression_categories import load_graph

    classifier, SourceIndex = reference
    body = "Nat.succ (Nat.succ (Nat.succ 0))"
    old = f"theorem target : {body} = {body} := by rfl\n"
    new = f"{command} LongValue : Nat := {body}\ntheorem target : {proof}\n"
    sources = dict(before=SourceIndex({"Main.lean": old}), after=SourceIndex({"Main.lean": new}))
    for phase, text, names in [("before", old, ["target"]), ("after", new, ["LongValue", "target"])]:
        nodes = [dict(name=name, module="Main", kind="theorem" if name == "target" else "def",
                      start_line=i, end_line=i, identity_status="compiled") for i, name in enumerate(names, 1)]
        edges = [["target", "LongValue"]] if phase == "after" and expected_uses else []
        if phase == "after" and "[LongValue]" in proof:
            edges = [["target", "LongValue"]]
        (tmp_path / f"{phase}.json").write_text(json.dumps(dict(nodes=nodes, edges=edges)))
        directory = tmp_path / phase
        directory.mkdir()
        with tarfile.open(directory / "original-sources.tar.gz", "w:gz") as archive:
            data = text.encode()
            info = tarfile.TarInfo("Main.lean")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    baseline, final = (sources[phase].total_tokens() for phase in ("before", "after"))
    record = dict(reconciliation=dict(scored_baseline_tokens=baseline, scored_post_tokens=final,
                                      scored_tokens_saved=baseline - final))
    ledger = classifier.classify("fixture", "test", tmp_path, record, {}, set())
    categories = {key: value for key, value in classifier.parts(ledger).items()
                  if not key.startswith("automation_")}
    before, after = (load_graph(tmp_path / f"{phase}.json") for phase in ("before", "after"))
    split, audit = split_definition_savings(ledger, categories, before, after, sources["before"], sources["after"])
    expected_credit = expected_uses * (len(classifier.abbreviations.tokens(body)) - 1)
    assert audit["uses"] == expected_credit
    assert audit["cost"] == len(classifier.abbreviations.tokens(new.splitlines()[0]))
    assert split["new_definition_savings"] == expected_credit - audit["cost"]
    assert split["syntax_optimization"] == categories["syntax_optimization"] - split["new_definition_savings"]
    assert sum(split.values()) == baseline - final
    assert split_definition_savings(ledger, split, before, after, sources["before"], sources["after"])[0] == split
