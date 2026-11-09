from copy import deepcopy

import pytest

from leanlean.graph_identity import adapt, checksum, provider_keys


def graph(modules=("A", "B")):
    payload = dict(
        schema="preprocessing_dependency_graph_v3",
        nodes=[dict(name="shared", module=m, kind="theorem", start_line=1, end_line=2) for m in modules]
        + [dict(name="root", module="Main", kind="theorem", start_line=1, end_line=2)],
        edges=[[2, 0, 2]],
        policies={"production": {"kept": [0, 1, 2], "protected": [2]}},
        ambiguous_edges=[dict(source_candidates=[2], target_candidates=[0, 1], kind_mask=1)],
    )
    payload["sha256"] = checksum(payload)
    return payload


def test_duplicate_providers_and_candidate_edges_are_preserved():
    payload = graph()
    keys, _ = provider_keys({"before": payload, "after": payload}, lambda phase, node: node["module"])
    data, audit = adapt(payload, keys["before"])
    a, b = keys["before"][("A", "shared")], keys["before"][("B", "shared")]
    assert len(data["nodes"]) == 3
    assert data["layers"]["kernel"] == {("root", a)}
    assert data["layers"]["source_text"] == {("root", a), ("root", b)}
    assert audit["ambiguous_edges"] == payload["ambiguous_edges"]
    assert data["nodes"][a]["lean_name"] == "shared"


def test_unchanged_provider_move_keeps_identity():
    before, after = graph(), graph(("C", "B"))
    keys, audit = provider_keys(
        {"before": before, "after": after}, lambda phase, node: "body-a" if node["module"] in {"A", "C"} else "body-b"
    )
    assert keys["before"][("A", "shared")] == keys["after"][("C", "shared")]
    assert len(audit["moves"]) == 1
    assert audit["unresolved_moves"] == []


def test_ambiguous_copy_moves_are_not_guessed():
    keys, audit = provider_keys({"before": graph(), "after": graph(("C", "D"))}, lambda phase, node: "same")
    assert audit["moves"] == []
    assert len(audit["unresolved_moves"]) == 1
    assert keys["before"][("A", "shared")] != keys["after"][("C", "shared")]


@pytest.mark.parametrize("corruption", ["checksum", "duplicate", "endpoint", "empty_candidates"])
def test_invalid_graph_fails_closed(corruption):
    payload = graph()
    keys, _ = provider_keys({"before": payload, "after": payload}, lambda phase, node: "body")
    broken = deepcopy(payload)
    if corruption == "checksum":
        broken["nodes"][0]["name"] = "changed"
    elif corruption == "duplicate":
        broken["nodes"][1] = deepcopy(broken["nodes"][0])
    elif corruption == "endpoint":
        broken["edges"][0][0] = True
    else:
        broken["ambiguous_edges"][0]["target_candidates"] = []
    if corruption != "checksum":
        broken["sha256"] = checksum(broken)
    with pytest.raises(ValueError):
        adapt(broken, keys["before"])
