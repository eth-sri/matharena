"""Separate local telemetry from portable graph and analysis artifacts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile

from leanlean.common import read_json, write_json
from leanlean.publication import Bundle, bundle_checksum, bundle_parts, open_bundle


def split_telemetry(folder: Path) -> dict:
    """Retain every original member while publishing a separate local log bundle."""
    folder = Path(folder)
    staged = folder.with_name(".telemetry-split-" + folder.name)
    backup = folder.with_name(".telemetry-backup-" + folder.name)
    if backup.exists():
        if folder.exists():
            shutil.rmtree(backup)
        else:
            backup.rename(folder)
    record = read_json(folder / "manifest.json")
    if record.get("telemetry", {}).get("local_only"):
        return record
    if staged.exists():
        shutil.rmtree(staged)
    shutil.copytree(folder, staged, copy_function=os.link)
    updated = _split(staged, record)
    folder.rename(backup)
    staged.rename(folder)
    shutil.rmtree(backup)
    return updated


def _split(folder: Path, record: dict) -> dict:
    original = folder / "analysis.tar.zst"
    if bundle_checksum(original) != record["analysis"]["sha256"]:
        raise ValueError("Analysis archive changed before telemetry split")
    with tempfile.TemporaryDirectory(prefix=".telemetry-work-", dir=folder) as temporary:
        temporary = Path(temporary)
        analysis = Bundle(temporary / "analysis.tar.zst")
        telemetry = Bundle(temporary / "telemetry.tar.zst")
        expected = None
        with open_bundle(original) as archive:
            for member in archive:
                if not member.isfile():
                    raise ValueError("Unexpected analysis archive entry")
                if member.name == "bundle-members.json":
                    expected = json.load(archive.extractfile(member))
                    continue
                output = telemetry if member.name.startswith("telemetry/") else analysis
                output.stream(member.name, archive.extractfile(member), member.size, mode=member.mode)
        if expected is None or {**analysis.members, **telemetry.members} != expected:
            raise ValueError("Telemetry split changed original member bytes")
        kept, local = analysis.finish(), telemetry.finish()
        from leanlean.release_audit import check_analysis

        check_analysis(temporary / "analysis.tar.zst")
        check_analysis(temporary / "telemetry.tar.zst")
        for part in bundle_parts(original):
            part.unlink()
        for filename in ("analysis.tar.zst", "telemetry.tar.zst"):
            for part in bundle_parts(temporary / filename):
                part.rename(folder / part.name)
        record["analysis"] = kept
        record["telemetry"] = dict(local, local_only=True)
        write_json(folder / "manifest.json", record)
    return record
