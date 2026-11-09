import io
import json
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path

from leanlean.checkpoint_renames import rename_map

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "leanlean"))
from origins_container import apply_renames  # noqa: E402


def checkpoint(attempt: Path, index: int, text: str) -> dict:
    archive = attempt / "sources" / f"s{index}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    data = text.encode()
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo("repository/Basic.lean")
        info.size = len(data)
        stream.addfile(info, io.BytesIO(data))
    return {"archive": f"sources/s{index}.tar.gz"}


def attempt_with(tmp: Path, name: str, states: list[str], previous: Path | None = None) -> Path:
    attempt = tmp / name
    rows = [checkpoint(attempt, i, text) for i, text in enumerate(states)]
    (attempt / "checkpoints.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    if previous is not None:
        (attempt / "continues.json").write_text(json.dumps({"previous_attempt": str(previous)}))
    return attempt


NESTED = "namespace Foo\nnamespace Data\ntheorem a : True := trivial\ntheorem b : True := trivial\nend Data\nend Foo\n"
FLAT = "namespace Foo\ntheorem a : True := trivial\ntheorem b : True := by trivial\ntheorem c : True := trivial\nend Foo\n"


def test_namespace_flattening_is_traced_and_ambiguity_is_not_linked(tmp_path):
    attempt = attempt_with(tmp_path, "a", [NESTED, FLAT])
    result = rename_map(attempt)
    assert result["renames"] == {"Foo.a": "Foo.Data.a", "Foo.b": "Foo.Data.b"}
    assert result["checkpoints"] == 2 and result["ambiguous_short_names"] == 0
    # Two removed `a`s and two added `a`s in one step: no link.
    ambiguous = attempt_with(tmp_path, "b", [
        "namespace X\ntheorem a : True := trivial\nend X\nnamespace Y\ntheorem a : True := trivial\nend Y\n",
        "namespace Z\ntheorem a : True := trivial\nend Z\nnamespace W\ntheorem a : True := trivial\nend W\n",
    ])
    assert rename_map(ambiguous)["renames"] == {} and rename_map(ambiguous)["ambiguous_short_names"] == 1


def test_renames_chain_through_steps_and_continued_attempts(tmp_path):
    first = attempt_with(tmp_path, "first", [NESTED, FLAT])
    renamed_again = FLAT.replace("namespace Foo", "namespace Bar").replace("end Foo", "end Bar")
    grace = attempt_with(tmp_path, "grace", [FLAT, renamed_again], previous=first)
    # `c` first appears mid-attempt, so it has no baseline name.
    assert rename_map(grace)["renames"] == {"Bar.a": "Foo.Data.a", "Bar.b": "Foo.Data.b"}
    # A declaration deleted at one step and re-created at a later one starts a new identity.
    gone = "namespace Foo\ntheorem b : True := trivial\nend Foo\n"
    recreated = attempt_with(tmp_path, "c", [NESTED, gone, FLAT])
    assert "Foo.a" not in rename_map(recreated)["renames"]


@dataclass(frozen=True)
class Graph:
    nodes: dict
    layers: dict
    static_edges: set
    grind_edges: set


def test_apply_renames_keeps_edges_and_skips_unsafe_links():
    before = Graph({"Foo.Data.a": {}, "Foo.Data.b": {}, "Foo.taken": {}, "Foo.c": {}}, {}, set(), set())
    after = Graph(
        {"Foo.a": {"name": "Foo.a"}, "Foo.b": {"name": "Foo.b"}, "Foo.taken": {}, "Foo.x": {}, "Foo.c": {}},
        {"kernel": {("Foo.b", "Foo.a")}}, {("Foo.b", "Foo.a")}, {("Foo.x", "Foo.b")},
    )
    renames = {"Foo.a": "Foo.Data.a", "Foo.b": "Foo.Data.b", "Foo.x": "Foo.taken", "Foo.c": "Foo.Old",
               "Foo.missing": "Foo.Data.m"}
    graph, applied, audit = apply_renames(after, before, renames)
    assert applied == {"Foo.a": "Foo.Data.a", "Foo.b": "Foo.Data.b"}
    assert set(graph.nodes) == {"Foo.Data.a", "Foo.Data.b", "Foo.taken", "Foo.x", "Foo.c"}
    assert graph.nodes["Foo.Data.a"]["renamed_from"] == "Foo.a"
    assert graph.static_edges == {("Foo.Data.b", "Foo.Data.a")} and graph.grind_edges == {("Foo.x", "Foo.Data.b")}
    assert graph.layers["kernel"] == {("Foo.Data.b", "Foo.Data.a")}
    reasons = {row["name"]: row["reason"] for row in audit["skipped"]}
    assert reasons == {"Foo.x": "baseline name in use", "Foo.c": "existed before", "Foo.missing": "not in after graph"}
