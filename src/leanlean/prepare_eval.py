"""Package a verified strip run into an offline MathArena harness image."""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
from pathlib import Path

from matharena.monitoring.commands import TRACE_VERSION
from leanlean.common import (
    exclusive_lock,
    python_package_sha256,
    read_json,
    sha256,
    write_json,
)
from leanlean.eval_tools import COMPARATOR_COMMIT, NANODA_COMMIT
from leanlean.evaluation import load_problems


def copy_directory(source: Path, destination: Path) -> None:
    """Preserve compiled caches; reflinks avoid duplicating bytes when supported."""
    print(f"Preparing {destination.name} from {source}", flush=True)
    destination.mkdir(parents=True, exist_ok=True)
    subprocess.run(["cp", "-a", "--reflink=auto", str(source) + "/.", str(destination)], check=True)


def prepare(run: Path, bundle: Path, output: Path, storage_root: Path | None, *, command_monitoring="host") -> dict:
    if command_monitoring not in {"host", "wrapper"}:
        raise ValueError("command_monitoring must be host or wrapper")
    metadata = read_json(run / "run.json")
    native = metadata.get("schema_version") == 2
    if native and python_package_sha256(Path(metadata["lean_strip_package"])) != metadata["lean_strip_sha256"]:
        raise ValueError("Installed lean-strip sources changed since preprocessing; restore the recorded version")
    problems = load_problems(str(bundle))
    output.mkdir(parents=True, exist_ok=True)
    storage_root = storage_root or (Path(metadata["storage_root"]) if metadata.get("storage_root") else None)
    prefix = ["podman"]
    if storage_root:
        prefix += ["--root", str(storage_root / "storage"), "--runroot", str(storage_root / "runtime")]
    info = subprocess.check_output(prefix + ["info", "--format", "{{.Host.Security.Rootless}}"], text=True).strip()
    if info != "true":
        raise ValueError("This preparation command requires rootless Podman")
    context = output / "context"
    context.mkdir(exist_ok=True)
    fingerprint = {"bundle_sha256": sha256(bundle / "manifest.json"), "strip_image": metadata["image"]}
    if native:
        fingerprint["lean_strip_sha256"] = metadata["lean_strip_sha256"]
    receipt = output / "inputs.json"
    if receipt.exists() and read_json(receipt) != fingerprint:
        raise ValueError("Preparation output belongs to different inputs; choose a new output directory")
    write_json(receipt, fingerprint)
    for name in ("toolchains", "packages", "tools", "sources", "bin"):
        (context / name).mkdir(exist_ok=True)
    versions = set()
    for problem in problems:
        entry = problem["leanlean_project"]
        key = entry["key"]
        result = read_json(run / "results" / key / "status.json")
        if result["status"] != "verified":
            raise ValueError(f"Unverified run result: {key}")
        original = read_json(run / "bundle/provenance" / key / "entry.json")
        if original["project_tree_sha256"] != entry["project_tree_sha256"]:
            raise ValueError(f"Bundle does not match strip run: {key}")
        copy_directory(Path(entry["source_directory"]), context / "sources" / key)
        packages = run / "work" / key / entry["project_path"] / ".lake/packages"
        lockfile = Path(entry["source_directory"]) / entry["project_path"] / "lake-manifest.json"
        if packages.is_dir():
            copy_directory(packages, context / "packages" / key)
        elif read_json(lockfile).get("packages") == []:
            (context / "packages" / key).mkdir(exist_ok=True)
        else:
            raise ValueError(f"Missing prepared dependencies: {packages}")
        versions.add(entry["lean_toolchain"].replace("/", "--").replace(":", "---"))
        verification = result["verification"]
        if verification["judge"] == "standalone comparator" or verification["judge"].startswith("comparator "):
            commit = metadata["comparator_commit"] if native else verification["comparator_commit"]
            if commit != COMPARATOR_COMMIT:
                raise ValueError("Standalone Comparator version does not match this adapter")
            version = entry["lean_toolchain"].split(":")[1]
            paths = [
                f"comparator-{COMPARATOR_COMMIT}/.lake/build/bin/comparator",
                f"comparator-{COMPARATOR_COMMIT}/scripts/fake-landrun.sh",
                f"lean4export-{version}/.lake/build/bin/lean4export",
            ]
            comparator_config = read_json(Path(entry["source_directory"]) / entry["comparator_path"])
            needs_nanoda = comparator_config.get("enable_nanoda") or comparator_config.get("external_kernels")
            if needs_nanoda:
                if native and metadata["nanoda_commit"] != NANODA_COMMIT:
                    raise ValueError("NanoDa version does not match this adapter")
                paths.append(f"nanoda-{NANODA_COMMIT}/target/release/nanoda_bin")
            for relative in paths:
                target = context / "tools" / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                source = run / "cache/tools" / relative
                if native:
                    upstream_path = relative.replace(COMPARATOR_COMMIT, COMPARATOR_COMMIT[:12]).replace(
                        NANODA_COMMIT, NANODA_COMMIT[:12]
                    )
                    source = run / "cache" / key / "lean-strip/comparator-tools" / upstream_path
                shutil.copy2(source, target)
    for version in sorted(versions):
        # Strip caches may link into the host's elan installation; materialize
        # only the toolchains required by these projects, never host credentials.
        source = (
            (Path(metadata["elan_home"]) / "toolchains" / version)
            if native
            else run / "cache/elan/toolchains" / version
        )
        if source.is_symlink() and str(source.readlink()).startswith("/host-toolchains/"):
            source = Path.home() / ".elan/toolchains" / version
        if not source.is_dir():
            raise ValueError(f"Missing project toolchain: {source}")
        copy_directory(source.resolve(), context / "toolchains" / version)
    shutil.copy2(bundle / "manifest.json", context / "manifest.json")
    package = Path(__file__).resolve().parent
    shutil.copytree(
        package, context / "pipeline/leanlean", dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__")
    )
    shared_package = package.parent / "matharena"
    (context / "pipeline/matharena").mkdir(parents=True, exist_ok=True)
    shutil.copy2(shared_package / "__init__.py", context / "pipeline/matharena/__init__.py")
    shutil.copytree(
        shared_package / "monitoring", context / "pipeline/matharena/monitoring",
        dirs_exist_ok=True, ignore=shutil.ignore_patterns("__pycache__"),
    )
    if native:
        shutil.copytree(
            metadata["lean_strip_package"],
            context / "pipeline/lean_strip",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    repository = Path.cwd()
    shutil.copy2(repository / "docker/harness-arxivlean/requirements.toml", context / "requirements.toml")
    shutil.copy2(repository / "docker/leanlean-eval/Dockerfile", context / "Dockerfile")
    for name, argument in [("proof-length", "proof-length"), ("leanlean-check", "check")]:
        executable = context / "bin" / name
        executable.write_text(f'#!/bin/sh\nexec /usr/local/bin/python -m leanlean.eval_tools {argument} "$@"\n')
        executable.chmod(0o755)
    # Native Comparator clears the environment before launching bubblewrap.
    # Put this adapter on PATH so bare `lake comparator` retains our thread default.
    bubblewrap = context / "bin/bwrap"
    bubblewrap.write_text(
        "#!/usr/local/bin/python3 -I\nimport sys\nsys.path.insert(0, '/pipeline')\n"
        "from leanlean.bubblewrap import main\nmain()\n"
    )
    bubblewrap.chmod(0o755)
    # A wrapper keeps all normal harness commands (run/network/info/rm/unshare)
    # in the same Podman store without changing the shared runner's CLI.
    engine = output / "podman-engine"
    engine.write_text("#!/bin/sh\nexec " + shlex.join(prefix) + ' "$@"\n')
    engine.chmod(0o755)
    tag = "localhost/matharena-leanlean-eval:" + sha256(bundle / "manifest.json")[:12]
    subprocess.run(
        prefix + ["build", "--build-arg", f"STRIP_IMAGE={metadata['image']}",
                  "--build-arg", f"COMMAND_MONITORING={command_monitoring}", "-t", tag, str(context)], check=True
    )
    image = json.loads(subprocess.check_output(prefix + ["image", "inspect", tag], text=True))[0]["Id"]
    runtime = {
        "schema_version": 1,
        "bundle_sha256": sha256(bundle / "manifest.json"),
        "docker_image": image,
        "container_engine": str(engine),
        "container_user": "0:0",
        # Nested bubblewrap maps UID 0 in its user namespace and mounts proc.
        # These settings are scoped to the rootless Podman container.
        "docker_extra_args": ["--cap-add", "SETFCAP", "--security-opt", "unmask=/proc/*"],
        "command_monitoring_backend": command_monitoring,
        "command_trace_version": TRACE_VERSION if command_monitoring == "wrapper" else None,
    }
    write_json(bundle / "eval-runtime.json", runtime)
    return runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path)
    parser.add_argument("--command-monitoring", choices=["host", "wrapper"], default="host")
    args = parser.parse_args()
    with exclusive_lock(args.output / ".prepare.lock"):
        result = prepare(
            args.run.resolve(),
            args.bundle.resolve(),
            args.output.resolve(),
            args.storage_root.resolve() if args.storage_root else None,
            command_monitoring=args.command_monitoring,
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
