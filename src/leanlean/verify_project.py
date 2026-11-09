"""Check the applied output with lean-strip's verifier, without stripping again.

The native CLI judges its scratch tree before applying changes. Keep this final
check because an applied tree can still fail even when that scratch build passed.
Comparator/toolchain setup belongs entirely to the pinned lean-strip package.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from leanlean.common import write_json


def verify(project: Path, comparator: Path, jobs: int, timeout: int) -> dict:
    from lean_strip.verify import run_comparator
    from lean_strip.workspace import LocalWorkspace

    subprocess.run(
        ["lake", "build"], cwd=project, check=True, timeout=timeout, env={**os.environ, "LEAN_NUM_THREADS": str(jobs)}
    )
    workspace = LocalWorkspace(project, project / ".lean-strip/final-check", default_timeout=timeout, lean_threads=jobs)
    return run_comparator(workspace, comparator, timeout, lambda line: print(line, flush=True))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--comparator", type=Path, required=True)
    parser.add_argument("--jobs", type=int, required=True)
    parser.add_argument("--timeout", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verdict = verify(args.project, args.comparator, args.jobs, args.timeout)
    write_json(args.output, verdict)
    return int(verdict["status"] != "accepted")


if __name__ == "__main__":
    raise SystemExit(main())
