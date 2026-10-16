#!/usr/bin/env python3
"""Build an isolated, pinned Lean runtime without altering historical editions."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

LEAN_VERSION = "4.34.1"
TOOLCHAIN = f"leanprover/lean4:v{LEAN_VERSION}"
MATHLIB_REV = "d13f23b723b8a846827a245b89c10fc7d3f11612"
DEPENDENCIES = {
    "comparator": "d03acab154d269c06e60e4de7e4cc85deebff94b",
    "lean4export": "076e8e57707e813375e8f9da8bf989799ace9680",
}


def run(args, *, cwd=None):
    print("+ " + " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def sha256(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def elf_interpreter(executable):
    headers = subprocess.check_output(["readelf", "-l", executable], text=True)
    match = re.search(r"\[Requesting program interpreter: ([^\]\n]+)\]", headers)
    if not match or not Path(match[1]).is_absolute() or not Path(match[1]).is_file():
        raise RuntimeError("Cannot identify the Lean ELF loader for the sandbox allowlist")
    return Path(match[1]).resolve()


def allow_elf_interpreter(main_file, interpreter):
    # Landrun 0.1.13's ldd parser omits the ELF interpreter on this toolchain.
    # Permit only that loader, not /lib or unrestricted executable access.
    original = 'let args := #["--best-effort", "--ro", "/", "--rw", "/dev", "-ldd", "-add-exec"]'
    patched = original[:-1] + ', "--rox", ' + json.dumps(str(interpreter)) + ']'
    source = main_file.read_text()
    if patched in source:
        return
    if source.count(original) != 1:
        raise RuntimeError("Unexpected upstream comparator sandbox configuration")
    main_file.write_text(source.replace(original, patched))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--elan", type=Path, required=True)
    parser.add_argument("--landrun-bin", type=Path, required=True,
                        help="Existing trusted Landrun binary to copy (not a symlink)")
    args = parser.parse_args()
    root = args.runtime_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    external = root / "external"
    external.mkdir(exist_ok=True)
    os.environ.update(ELAN_HOME=str(root / "elan"), XDG_CACHE_HOME=str(root / "cache"))
    os.environ["PATH"] = str(args.elan.resolve().parent) + os.pathsep + os.environ["PATH"]
    run([args.elan, "toolchain", "install", TOOLCHAIN])
    lean_bin = root / "elan/toolchains" / f"leanprover--lean4---v{LEAN_VERSION}" / "bin"
    os.environ["PATH"] = str(lean_bin) + os.pathsep + os.environ["PATH"]
    run([lean_bin / "lean", "--version"])
    interpreter = elf_interpreter(lean_bin / "lake")

    for name, revision in DEPENDENCIES.items():
        directory = external / name
        if not directory.exists():
            run(["git", "clone", "--depth", "1", "--branch", "v4.34.0",
                 f"https://github.com/leanprover/{name}", directory])
        actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()
        if actual != revision:
            raise RuntimeError(f"Refusing to replace unexpected {name} revision: {actual}")
        (directory / "lean-toolchain").write_text(TOOLCHAIN + "\n")
        if name == "comparator":
            allow_elf_interpreter(directory / "Main.lean", interpreter)
        run(["lake", "build"], cwd=directory)

    landrun = external / "landrun/bin/landrun"
    landrun.parent.mkdir(parents=True, exist_ok=True)
    if not args.landrun_bin.is_file() or not os.access(args.landrun_bin, os.X_OK):
        raise RuntimeError("Landrun must be an executable file")
    shutil.copy2(args.landrun_bin, landrun)
    project = external / "comparator_project"
    project.mkdir(exist_ok=True)
    (project / "lean-toolchain").write_text(TOOLCHAIN + "\n")
    (project / "lakefile.lean").write_text(
        'import Lake\nopen Lake DSL\n\npackage comparatorcheck where\n\n'
        'require mathlib from git\n'
        f'  "https://github.com/leanprover-community/mathlib4" @ "{MATHLIB_REV}"\n\n'
        'lean_lib Challenge where\n  roots := #[`Challenge]\n\n'
        'lean_lib Solution where\n  roots := #[`Solution]\n'
    )
    for name in ("Challenge", "Solution"):
        (project / f"{name}.lean").write_text(
            "import Mathlib\n\ntheorem comparator_template_challenge : True := by trivial\n"
        )
    run(["lake", "update"], cwd=project)
    run(["lake", "exe", "cache", "get"], cwd=project)
    run(["lake", "build", "Challenge", "Solution"], cwd=project)
    assets = [landrun, project / "lean-toolchain", project / "lakefile.lean",
              project / "lake-manifest.json", lean_bin / "lean", interpreter,
              external / "comparator/Main.lean"]
    assets += [external / name / ".lake/build/bin" / name for name in DEPENDENCIES]
    receipt = {"toolchain": TOOLCHAIN, "mathlib_revision": MATHLIB_REV,
               "dependencies": DEPENDENCIES, "sandbox_elf_interpreter": str(interpreter),
               "runtime_sha256": {str(p): sha256(p) for p in assets}}
    (root / "runtime.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
