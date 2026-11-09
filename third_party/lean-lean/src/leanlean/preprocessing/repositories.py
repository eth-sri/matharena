"""Canonical repository database used by production preprocessing.

The database normalizes monorepo members and ordinary repositories into one
input shape. A LeanPool member has one extra preparation step which produces a
physically isolated raw image; after that step the preprocessing engine does
not branch on repository family.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DATABASE_KIND = "leanlean_repository_database"
DATABASE_SCHEMA_VERSION = 2
REPOSITORY_FAMILIES = {"standalone", "leanpool_member"}
CURATED_PROTECTED_PROVENANCE_KINDS = frozenset(
    {
        "leanpool_main_declarations_and_results",
        "manual_headline_results",
        "palomar_challenge_comparator_results",
        "palomar_registry_main_results",
        "tau_ceti_public_api",
    }
)


@dataclass(frozen=True)
class RepositorySpec:
    """One fully normalized preprocessing input."""

    instance_id: str
    family: str
    repository_url: str
    commit: str
    toolchain: str
    raw_image_tag: str
    raw_image_id: str
    preprocessing_image_tag: str
    entry_module: str
    target_dir: str
    build_target: str
    exclude_dirs: tuple[str, ...]
    protected_declarations: frozenset[str]
    protected_provenance: Mapping[str, Any]
    variants: Mapping[str, Mapping[str, str]]

    @property
    def needs_raw_preparation(self) -> bool:
        return self.family == "leanpool_member"


@dataclass(frozen=True)
class RepositoryDatabase:
    path: Path
    sha256: str
    definition_sha256: str
    dataset_id: str
    dataset_version: str
    repositories: tuple[RepositorySpec, ...]

    def by_id(self) -> dict[str, RepositorySpec]:
        return {
            repository.instance_id: repository
            for repository in self.repositories
        }



def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_definition_sha256(payload: Mapping[str, Any]) -> str:
    """Hash source/scopes/protected policy while allowing artifact-pin refresh."""

    rows = []
    for row in payload.get("repositories", []):
        if not isinstance(row, Mapping):
            continue
        rows.append(
            {key: value for key, value in row.items() if key != "variants"}
        )
    definition = {
        "kind": payload.get("kind"),
        "schema_version": payload.get("schema_version"),
        "dataset_id": payload.get("dataset_id"),
        "dataset_version": payload.get("dataset_version"),
        "source_presets": payload.get("source_presets"),
        "repositories": rows,
    }
    if "curation" in payload:
        definition["curation"] = payload.get("curation")
    canonical = json.dumps(
        definition, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _require_string(mapping: Mapping[str, Any], key: str, context: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context}.{key} must be a non-empty string")
    return value


def _validate_provenance_file(
    root: Path,
    record: Mapping[str, Any],
    *,
    path_key: str,
    sha_key: str,
    context: str,
) -> None:
    relative = record.get(path_key)
    expected = record.get(sha_key)
    if relative is None and expected is None:
        return
    if (
        not isinstance(relative, str)
        or not relative
        or not isinstance(expected, str)
        or not expected
    ):
        raise ValueError(
            f"{context} must pin both {path_key} and {sha_key}"
        )
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"{context}.{path_key} is missing or outside the repo")
    actual = file_sha256(resolved)
    if actual != expected:
        raise ValueError(
            f"{context}.{path_key} drift: expected {expected}, found {actual}"
        )


def _load_protected_names(
    root: Path,
    protected: Mapping[str, Any],
    *,
    context: str,
) -> tuple[frozenset[str], Mapping[str, Any]]:
    inline = protected.get("declarations")
    file_record = protected.get("file")
    if (inline is None) == (file_record is None):
        raise ValueError(
            f"{context}.protected must define exactly one of declarations or file"
        )

    provenance = protected.get("provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise ValueError(f"{context}.protected.provenance must be a mapping")
    _validate_provenance_file(
        root,
        provenance,
        path_key="registry",
        sha_key="registry_sha256",
        context=f"{context}.protected.provenance",
    )
    _validate_provenance_file(
        root,
        provenance,
        path_key="report",
        sha_key="report_sha256",
        context=f"{context}.protected.provenance",
    )
    resolution_evidence = provenance.get("resolution_evidence", {})
    if not isinstance(resolution_evidence, Mapping):
        raise ValueError(f"{context}.protected.provenance.resolution_evidence")
    _validate_provenance_file(
        root, resolution_evidence, path_key="report", sha_key="sha256",
        context=f"{context}.protected.provenance.resolution_evidence",
    )

    if inline is not None:
        names = inline
    else:
        if not isinstance(file_record, Mapping):
            raise ValueError(f"{context}.protected.file must be a mapping")
        relative = _require_string(
            file_record, "path", f"{context}.protected.file"
        )
        expected_sha = _require_string(
            file_record, "sha256", f"{context}.protected.file"
        )
        path = root / relative
        if not path.is_file():
            raise ValueError(
                f"{context}: protected declaration file is missing: {path}"
            )
        actual_sha = file_sha256(path)
        if actual_sha != expected_sha:
            raise ValueError(
                f"{context}: protected declaration file drift: "
                f"expected {expected_sha}, found {actual_sha}"
            )
        payload = json.loads(path.read_text())
        names = [
            *payload.get("signature_theorems", []),
            *payload.get("protected_definitions", []),
        ]
        recorded_count = file_record.get("count")
        if recorded_count is not None and int(recorded_count) != len(set(names)):
            raise ValueError(
                f"{context}: protected declaration count does not match {path}"
            )

    if (
        not isinstance(names, list)
        or not names
        or not all(isinstance(name, str) and name for name in names)
    ):
        raise ValueError(
            f"{context}.protected declarations must be non-empty names"
        )
    if len(names) != len(set(names)):
        raise ValueError(f"{context}.protected declarations contain duplicates")
    return frozenset(names), provenance


def _parse_repository(
    root: Path,
    row: Mapping[str, Any],
    index: int,
    source_presets: Mapping[str, Any],
) -> RepositorySpec:
    context = f"repositories[{index}]"
    instance_id = _require_string(row, "instance_id", context)
    family = _require_string(row, "family", context)
    if family not in REPOSITORY_FAMILIES:
        raise ValueError(
            f"{context}.family must be one of {sorted(REPOSITORY_FAMILIES)}"
        )

    source_row = row.get("source")
    if not isinstance(source_row, Mapping):
        raise ValueError(f"{context}.source must be a mapping")
    preset_name = source_row.get("preset")
    if preset_name is None:
        source = dict(source_row)
    else:
        if (
            not isinstance(preset_name, str)
            or not isinstance(source_presets.get(preset_name), Mapping)
        ):
            raise ValueError(
                f"{context}.source.preset does not name a source preset"
            )
        source = {
            **source_presets[preset_name],
            **{
                key: value for key, value in source_row.items() if key != "preset"
            },
        }
    raw_image = source.get("raw_image")
    if not isinstance(raw_image, Mapping):
        raise ValueError(f"{context}.source.raw_image must be a mapping")
    raw_tag = _require_string(
        raw_image, "tag", f"{context}.source.raw_image"
    )
    raw_id = _require_string(
        raw_image, "image_id", f"{context}.source.raw_image"
    )
    if not raw_id.startswith("sha256:"):
        raise ValueError(
            f"{context}.source.raw_image.image_id must be immutable"
        )

    prepared = source.get("prepared_image")
    if family == "leanpool_member":
        if not isinstance(prepared, Mapping):
            raise ValueError(
                f"{context}.source.prepared_image must be a mapping"
            )
        preprocessing_tag = _require_string(
            prepared, "tag", f"{context}.source.prepared_image"
        )
        if prepared.get("source_isolation") != "standalone_v2":
            raise ValueError(
                f"{context}.source.prepared_image.source_isolation "
                "must be standalone_v2"
            )
    else:
        if prepared is not None:
            raise ValueError(
                f"{context}: standalone repositories need no prepared image"
            )
        preprocessing_tag = raw_tag

    scopes = row.get("scopes", {})
    if not isinstance(scopes, Mapping):
        raise ValueError(f"{context}.scopes must be a mapping")
    exclude_dirs = scopes.get("exclude_dirs", [])
    if not isinstance(exclude_dirs, list) or not all(
        isinstance(value, str) for value in exclude_dirs
    ):
        raise ValueError(
            f"{context}.scopes.exclude_dirs must be a string list"
        )

    protected = row.get("protected")
    if not isinstance(protected, Mapping):
        raise ValueError(f"{context}.protected must be a mapping")
    names, provenance = _load_protected_names(
        root, protected, context=context
    )

    variants = row.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ValueError(f"{context}.variants must be a mapping")
    for variant, image in variants.items():
        if (
            variant not in {"stripped", "optimized"}
            or not isinstance(image, Mapping)
        ):
            raise ValueError(
                f"{context}.variants contains an invalid entry"
            )
        _require_string(image, "tag", f"{context}.variants.{variant}")
        image_id = _require_string(
            image, "image_id", f"{context}.variants.{variant}"
        )
        if not image_id.startswith("sha256:"):
            raise ValueError(
                f"{context}.variants.{variant}.image_id must be immutable"
            )
        provenance_keys = {
            "preprocessing_version", "run_id", "grind_source_form"
        }
        present = provenance_keys & set(image)
        if present and present != provenance_keys:
            raise ValueError(
                f"{context}.variants.{variant} has partial provenance"
            )
        if present:
            for key in provenance_keys:
                _require_string(
                    image, key, f"{context}.variants.{variant}"
                )
            if image["grind_source_form"] != "original_grind_calls":
                raise ValueError(
                    f"{context}.variants.{variant}.grind_source_form "
                    "must be original_grind_calls"
                )

    return RepositorySpec(
        instance_id=instance_id,
        family=family,
        repository_url=_require_string(
            source, "repository_url", f"{context}.source"
        ),
        commit=_require_string(source, "commit", f"{context}.source"),
        toolchain=_require_string(source, "toolchain", f"{context}.source"),
        raw_image_tag=raw_tag,
        raw_image_id=raw_id,
        preprocessing_image_tag=preprocessing_tag,
        entry_module=str(scopes.get("entry_module", "")),
        target_dir=str(scopes.get("target_dir", "")),
        build_target=str(scopes.get("build_target", "")),
        exclude_dirs=tuple(exclude_dirs),
        protected_declarations=names,
        protected_provenance=provenance,
        variants=variants,
    )


def load_repository_database(
    path: Path, *, repo_root: Path
) -> RepositoryDatabase:
    resolved = path if path.is_absolute() else repo_root / path
    payload = json.loads(resolved.read_text())
    if not isinstance(payload, Mapping):
        raise ValueError(f"{resolved}: expected a JSON object")
    if payload.get("kind") != DATABASE_KIND:
        raise ValueError(f"{resolved}: unexpected database kind")
    if payload.get("schema_version") != DATABASE_SCHEMA_VERSION:
        raise ValueError(
            f"{resolved}: schema_version must be {DATABASE_SCHEMA_VERSION}"
        )
    dataset_id = _require_string(payload, "dataset_id", str(resolved))
    dataset_version = _require_string(
        payload, "dataset_version", str(resolved)
    )
    rows = payload.get("repositories")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            f"{resolved}: repositories must be a non-empty list"
        )
    source_presets = payload.get("source_presets", {})
    if not isinstance(source_presets, Mapping):
        raise ValueError(f"{resolved}: source_presets must be a mapping")
    repositories = tuple(
        _parse_repository(repo_root, row, index, source_presets)
        for index, row in enumerate(rows)
        if isinstance(row, Mapping)
    )
    if len(repositories) != len(rows):
        raise ValueError(
            f"{resolved}: every repository row must be an object"
        )
    ids = [repository.instance_id for repository in repositories]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{resolved}: duplicate instance_id")
    return RepositoryDatabase(
        path=resolved,
        sha256=file_sha256(resolved),
        definition_sha256=repository_definition_sha256(payload),
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        repositories=repositories,
    )
