"""Offline model-facing tools. Feedback only; final scoring is separate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from leanlean.common import read_json, sha256

# Palomar verifier revisions used by lean-strip 1.1.
COMPARATOR_COMMIT = "575674928e239f5bc452aab72d1dd7b0f1326494"
NANODA_COMMIT = "68d5ca9db226849b41a6fff59d796ff19d0a8840"


def context() -> tuple[dict, Path]:
    # Metadata comes from the immutable image, never from the model's workspace.
    request = read_json(Path("/work/leanlean-project.json"))
    entries = read_json(Path("/opt/leanlean/manifest.json"))["projects"]
    entry = next(item for item in entries if item["key"] == request["key"])
    return entry, Path("/work/repository") / entry["project_path"]


def proof_length() -> None:
    from lean_strip._pipeline.metrics.tokens import count_lean_tokens_in_source

    entry, project = context()
    challenge = Path("/work/repository") / entry["challenge_path"]
    counts = {}
    for directory, dirs, files in os.walk(project, followlinks=False):
        dirs[:] = sorted(name for name in dirs if not name.startswith("."))
        for name in sorted(files):
            path = Path(directory) / name
            if path.suffix != ".lean" or name == "lakefile.lean" or path == challenge:
                continue
            if path.is_symlink():
                raise ValueError(f"Source symlinks are not supported by proof-length: {path}")
            counts[str(path.relative_to(project))] = count_lean_tokens_in_source(path.read_text())
    print(json.dumps({"lean_tokens": sum(counts.values()), "files": counts}, indent=2))


def check() -> None:
    entry, project = context()
    baseline = Path("/opt/leanlean/sources") / entry["key"]
    for field in ("challenge_path", "comparator_path", "toolchain_path", "lakefile_path"):
        relative = entry[field]
        if sha256(Path("/work/repository") / relative) != sha256(baseline / relative):
            raise ValueError(f"Protected input changed: {relative}")
    manifest = Path(entry["project_path"]) / "lake-manifest.json"
    if sha256(Path("/work/repository") / manifest) != sha256(baseline / manifest):
        raise ValueError("Dependency lockfile changed")
    config = Path("/work/repository") / entry["comparator_path"]
    solution_module = read_json(config)["solution_module"]
    subprocess.run(["lake", "build", f"+{solution_module}"], cwd=project, check=True)
    help_result = subprocess.run(
        ["lake", "comparator", "--help"], cwd=project, capture_output=True, text=True, check=False
    )
    if "Judge a solution" in help_result.stdout + help_result.stderr:
        command = ["lake", "comparator", "--config", str(config)]
        environment = os.environ.copy()
        # Native Comparator locates bundled external kernels on PATH. Elan's
        # proxy directory alone does not expose the toolchain's nanoda_bin.
        lean = subprocess.check_output(["elan", "which", "lean"], cwd=project, text=True).strip()
        # Resolve Elan aliases before Comparator bind-mounts the external
        # kernel: bubblewrap cannot recreate a destination through /elan's symlink.
        environment["PATH"] = str(Path(lean).resolve().parent) + os.pathsep + environment.get("PATH", "")
    else:
        tools = Path("/opt/leanlean/tools")
        comparator = tools / f"comparator-{COMPARATOR_COMMIT}"
        version = entry["lean_toolchain"].split(":", 1)[1]
        environment = {
            **os.environ,
            "COMPARATOR_LEAN4EXPORT": str(tools / f"lean4export-{version}/.lake/build/bin/lean4export"),
            "COMPARATOR_LANDRUN": str(comparator / "scripts/fake-landrun.sh"),
            "COMPARATOR_NANODA": str(tools / f"nanoda-{NANODA_COMMIT}/target/release/nanoda_bin"),
        }
        command = ["lake", "env", str(comparator / ".lake/build/bin/comparator"), str(config)]
    # Bound Comparator separately from the initial cache-warming build. GNU
    # timeout signals the process group and escalates if shutdown gets stuck.
    subprocess.run(
        ["timeout", "--kill-after=10s", "3600s", *command],
        cwd=project, env=environment, check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tool", choices=["check", "proof-length"])
    args = parser.parse_args()
    if args.tool == "check":
        check()
    else:
        proof_length()


if __name__ == "__main__":
    main()
