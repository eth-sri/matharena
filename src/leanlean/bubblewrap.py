"""Preserve the Lean thread default inside native Comparator's clean environment."""

from __future__ import annotations

import os
import sys

DEFAULT_LEAN_THREADS = "8"


def command(arguments: list[str]) -> list[str]:
    # Lake launches `env -i bwrap ... --clearenv ... -- COMMAND`. A Docker ENV
    # alone therefore cannot reach its build/export children. Inject the default
    # after --clearenv; retain any explicit setting supplied by the caller.
    if "--" not in arguments:
        return ["/usr/bin/bwrap", *arguments]
    boundary = arguments.index("--")
    options = arguments[:boundary]
    explicit = any(
        options[i:i + 2] == ["--setenv", "LEAN_NUM_THREADS"]
        for i in range(len(options) - 2)
    )
    if not explicit:
        options = [*options, "--setenv", "LEAN_NUM_THREADS", os.environ.get("LEAN_NUM_THREADS", DEFAULT_LEAN_THREADS)]
    return ["/usr/bin/bwrap", *options, *arguments[boundary:]]


def main() -> None:
    argv = command(sys.argv[1:])
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
