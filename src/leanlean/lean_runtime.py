"""Set a Lean stack default compatible with finite virtual address space."""

from __future__ import annotations

import sys
from pathlib import Path

# Same first fallback as lean-strip: Lean's 1 GiB reservation per thread can
# exhaust RLIMIT_AS during parallel elaboration while physical RAM remains free.
DEFAULT_STACK_KIB = 256 * 1024


def command(real: str, arguments: list[str]) -> list[str]:
    explicit = any(
        arg in {"-s", "--tstack"} or arg.startswith("--tstack=")
        or (arg.startswith("-s") and arg[2:].isdigit())
        for arg in arguments
    )
    defaults = [] if explicit else [f"--tstack={DEFAULT_STACK_KIB}"]
    return [real, *defaults, *arguments]


def install() -> None:
    for path in sorted(Path("/opt/elan/toolchains").glob("*/bin/lean")):
        real = path.with_name(".lean-real")
        if real.exists():
            raise RuntimeError(f"Lean stack default already installed: {path}")
        path.rename(real)
        path.write_text(
            "#!/usr/local/bin/python3 -I\nimport os, sys\nsys.path.insert(0, '/pipeline')\n"
            "from leanlean.lean_runtime import command\n"
            f"argv = command({str(real)!r}, sys.argv[1:])\nos.execv(argv[0], argv)\n"
        )
        path.chmod(0o755)


if __name__ == "__main__":
    if sys.argv[1:] != ["install"]:
        raise SystemExit("Usage: python -m leanlean.lean_runtime install")
    install()
