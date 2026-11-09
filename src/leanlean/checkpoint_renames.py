"""Trace namespace-only declaration renames through an attempt's checkpoints.

Compression origins match declarations by qualified name, so moving declarations
into another namespace (`Foo.Data.lemma` -> `Foo.lemma`) reads as one deletion and
one addition. Checkpoints show where that happened: between two consecutive
states a declaration disappears and one with the same unqualified name appears.
Such a pair is linked when that short name is unique among the step's removed and
added names; ambiguous steps link nothing. Links are carried forward, so a
declaration renamed several times maps back to its baseline name, while one that
is deleted and later re-created at another step starts a new identity. Attempts
continued by a post-hoc grace turn (`continues.json`) are followed from their start.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from pathlib import Path

REFERENCE = Path(__file__).resolve().parents[2] / "third_party" / "lean-lean"


def _scanner():
    # Only importable without matharena's own `leanlean` package on the path.
    from scripts.analysis.checkpoint_identity import declarations

    return declarations


def attempt_chain(attempt: Path) -> list[Path]:
    """The attempt and those it continues, oldest first."""
    chain = [attempt]
    while (link := chain[0] / "continues.json").is_file():
        chain.insert(0, Path(json.loads(link.read_text())["previous_attempt"]))
    return chain


def checkpoint_archives(chain: list[Path]) -> list[Path]:
    archives = []
    for attempt in chain:
        for line in (attempt / "checkpoints.jsonl").read_text().splitlines():
            row = json.loads(line)
            if row.get("archive"):
                archives.append(attempt / row["archive"])
    return archives


class _Names:
    """Declared names per checkpoint, scanning each distinct file once."""

    def __init__(self, prefix: str):
        self.prefix = prefix.strip("/")
        self.declarations = _scanner()
        self.cache: dict[tuple[str, str], list[str]] = {}

    def __call__(self, archive: Path) -> set[str]:
        names = set()
        with tarfile.open(archive) as stream:
            for member in stream:
                if not member.isfile() or not member.name.endswith(".lean"):
                    continue
                parts = member.name.split("/")
                if any(part.startswith(".") for part in parts) or parts[-1] == "lakefile.lean":
                    continue
                data = stream.extractfile(member).read()
                key = (member.name, hashlib.sha256(data).hexdigest())
                if key not in self.cache:
                    text = data.decode("utf-8", "replace")
                    self.cache[key] = [row["name"] for row in self.declarations(member.name, text)]
                names.update(self.cache[key])
        return names


def rename_map(attempt: Path, project_prefix: str = "") -> dict:
    """Map final declaration names to their baseline names, with an audit.

    The vendored lean-lean scanner's `leanlean` package collides with
    matharena's, so the scan runs in a child interpreter that sees only lean-lean.
    """
    env = {**os.environ, "PYTHONPATH": f"{REFERENCE}{os.pathsep}{REFERENCE / 'src'}", "PYTHONDONTWRITEBYTECODE": "1"}
    done = subprocess.run([sys.executable, __file__, str(attempt), project_prefix], env=env,
                          capture_output=True, text=True, check=True)
    return json.loads(done.stdout)


def _rename_map(attempt: Path, project_prefix: str = "") -> dict:
    chain = attempt_chain(attempt)
    archives = checkpoint_archives(chain)
    if not archives:
        return {"renames": {}, "checkpoints": 0, "steps": [], "ambiguous_short_names": 0, "attempts": [str(a) for a in chain]}
    names = _Names(project_prefix)
    states = (names(archive) for archive in archives)
    previous = next(states)
    origin = {name: name for name in previous}
    steps, ambiguous = [], 0
    for index, current in enumerate(states, 1):
        removed, added = previous - current, current - previous
        by_short_removed, by_short_added = collections.defaultdict(list), collections.defaultdict(list)
        for name in removed:
            by_short_removed[name.rsplit(".", 1)[-1]].append(name)
        for name in added:
            by_short_added[name.rsplit(".", 1)[-1]].append(name)
        links = {}
        for short, gone in by_short_removed.items():
            new = by_short_added.get(short, [])
            if len(gone) == 1 and len(new) == 1:
                links[new[0]] = gone[0]
            elif gone and new:
                ambiguous += 1
        origin = {
            name: origin.get(name) if name in previous else origin.get(links[name]) if name in links else None
            for name in current
        }
        if links:
            steps.append({"checkpoint": index, "links": len(links)})
        previous = current
    renames = {name: base for name, base in sorted(origin.items()) if base is not None and base != name}
    return {
        "renames": renames,
        "checkpoints": len(archives),
        "steps": steps,
        "ambiguous_short_names": ambiguous,
        "attempts": [str(a) for a in chain],
    }


if __name__ == "__main__":
    print(json.dumps(_rename_map(Path(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "")))
