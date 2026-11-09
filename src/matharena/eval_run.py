"""Freeze evaluation inputs for the standard runner, independently of a benchmark."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import yaml


def _config_path(directory, name):
    relative = Path(name + ".yaml")
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Config IDs must be relative paths: {name!r}")
    return Path(directory) / relative


def _copy(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Lean source bundles can be large. Reflinks preserve an independent snapshot
    # without duplicating blocks on filesystems that support them.
    if shutil.which("cp"):
        subprocess.run(["cp", "-a", "--reflink=auto", "--", str(source), str(destination)], check=True)
    elif source.is_dir():
        shutil.copytree(source, destination, symlinks=True)
    else:
        shutil.copy2(source, destination)


def prepare_run_directory(
    root, *, competitions, models, competition_configs, model_configs,
    workers=None, shuffle_seed=None,
):
    """Copy configs and local datasets into a fresh run directory; never call a model.

    Remote dataset IDs remain references to their configured provider. Runtime
    manifests are copied too; images and dependency caches stay on the host.
    """
    root = Path(root).resolve()
    configs = []
    for kind, names, directory in (
        ("competitions", competitions, competition_configs),
        ("models", models, model_configs),
    ):
        for name in dict.fromkeys(names):
            path = _config_path(directory, name).resolve()
            config = yaml.safe_load(path.read_text())
            if not isinstance(config, dict):
                raise ValueError(f"Expected a YAML mapping in {path}")
            configs.append((kind, name, config))
    # A snapshot cannot contain itself, even if the caller chooses a run folder
    # under the original dataset directory.
    for kind, _, config in configs:
        source = Path(config.get("dataset_path", ""))
        if kind == "competitions" and source.is_dir() and root.is_relative_to(source.resolve()):
            raise ValueError("Run directory must be outside the dataset directory")
        helpers = (config.get("harness_config") or {}).get("helper_files")
        if helpers and root.is_relative_to(Path(helpers).resolve()):
            raise ValueError("Run directory must be outside the helper files directory")
    root.mkdir(parents=True, exist_ok=False)
    digests = {}

    def save_config(target, config):
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(yaml.safe_dump(config, sort_keys=False))
        digests[str(target.relative_to(root))] = hashlib.sha256(target.read_bytes()).hexdigest()

    for kind, name, config in configs:
        if kind == "models":
            if config.get("type") == "agent":
                # Scaffold agents reference another model and scaffold via paths
                # relative to configs/. Freeze those dependencies as well.
                for field in ("model_config", "scaffold_config"):
                    dependency = _config_path("configs", config[field])
                    values = yaml.safe_load(dependency.read_text())
                    if field == "model_config" and workers is not None:
                        values["concurrent_requests"] = workers
                    frozen = root / "configs/dependencies" / name / f"{field}.yaml"
                    save_config(frozen, values)
                    config[field] = str(frozen.with_suffix(""))
            elif workers is not None:
                config["concurrent_requests"] = workers
        else:
            if shuffle_seed is not None:
                config["shuffle_seed"] = shuffle_seed
            source = Path(config["dataset_path"]).resolve()
            destination = root / "datasets" / name
            if source.exists():
                _copy(source, destination)
                config["dataset_path"] = str(destination)
            harness = config.get("harness_config")
            if harness is None and config.get("allow_harness"):
                harness = config.setdefault("harness_config", {})
            if harness is not None:
                helpers = harness.get("helper_files")
                if helpers:
                    helpers = Path(helpers).resolve()
                    if source.is_dir() and helpers.is_relative_to(source):
                        frozen_helpers = destination / helpers.relative_to(source)
                    else:
                        frozen_helpers = root / "helpers" / name
                        _copy(helpers, frozen_helpers)
                    harness["helper_files"] = str(frozen_helpers)
                manifest = harness.get("runtime_manifest")
                if manifest:
                    manifest = Path(manifest).resolve()
                    if source.is_dir() and manifest.is_relative_to(source):
                        frozen_manifest = destination / manifest.relative_to(source)
                    else:
                        frozen_manifest = root / "runtime" / name / manifest.name
                        _copy(manifest, frozen_manifest)
                    harness["runtime_manifest"] = str(frozen_manifest)
                harness["workspace_root"] = str(
                    root / "workspaces" / name / "{solver_name}" / "p{problem_idx}_r{run_idx}"
                )
                harness["cli_cache_root"] = str(root / "cli-cache")
        target = _config_path(root / "configs" / kind, name)
        save_config(target, config)
    (root / "run.json").write_text(json.dumps({
        "competitions": competitions, "models": models,
        "workers": workers, "shuffle_seed": shuffle_seed, "config_sha256": digests,
    }, indent=2) + "\n")
    return root
