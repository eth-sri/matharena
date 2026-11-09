"""Fetch an immutable, reviewable set of Palomar source projects.

Usage: leanlean-fetch --month 2026-10 --output /path/to/october/snapshot

The month refers to first registration, not the publication of a later version.
We retain the latest active version of each matching entry at snapshot time.
Fetching never executes repository code. Stripping is a separate command.
"""

from __future__ import annotations

import argparse
import calendar
import concurrent.futures
import re
import shutil
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from leanlean.common import (
    exclusive_lock,
    read_json,
    relative_path,
    sha256,
    tree_sha256,
    utc_now,
    within,
    write_index,
    write_json,
)

REGISTRY = "https://data.palomar-registry.org/"
USER_AGENT = "MathArena-LeanLean/1.0 (https://github.com/eth-sri/matharena)"
MAX_ARCHIVE_BYTES = 1024**3
MAX_EXTRACTED_BYTES = 4 * 1024**3


class ModifiedSourceError(ValueError):
    """A previously frozen artifact changed; do not overwrite it on retry."""


def download(url: str, destination: Path, *, max_bytes: int = 32 * 1024**2) -> None:
    """Bounded downloads with retries and atomic installation."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        temporary = None
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with (
                urllib.request.urlopen(request, timeout=90) as response,
                tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream,
            ):
                temporary = Path(stream.name)
                size = 0
                deadline = time.monotonic() + 600
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > max_bytes or time.monotonic() > deadline:
                        raise ValueError(f"Download exceeds size/time limit: {url}")
                    stream.write(block)
            temporary.replace(destination)
            return
        except urllib.error.HTTPError as error:
            if error.code not in {408, 429, 500, 502, 503, 504} or attempt == 3:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 3:
                raise
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        time.sleep(2**attempt)


def month_bounds(month: str) -> tuple[str, str]:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError("Month must use YYYY-MM")
    year, number = map(int, month.split("-"))
    last_day = calendar.monthrange(year, number)[1]
    return f"{month}-01", f"{month}-{last_day:02}"


def entry_key(entry: dict) -> str:
    identifier, version = entry["id"], entry["version"]
    if not re.fullmatch(r"PALOMAR-\d{4}-\d{2}-\d{2}-\d{6}", identifier):
        raise ValueError(f"Invalid Palomar ID: {identifier!r}")
    if type(version) is not int or version < 1:
        raise ValueError(f"Invalid Palomar version: {version!r}")
    return f"{identifier}-v{version}"


def snapshot_registry(root: Path, month: str) -> dict:
    """Freeze all API pages from one revision; never silently refresh a snapshot."""
    start, end = month_bounds(month)
    manifest_path = root / "snapshot.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("month") != month:
            raise ValueError("Output already contains a different month; choose a new output directory")
        for page in manifest["pages"]:
            if sha256(within(root, page["path"])) != page["sha256"]:
                raise ValueError("Cached registry page was modified; use a fresh snapshot directory")
        return manifest

    entries, pages, seen_cursors, seen_ids = [], [], set(), set()
    cursor, revision, release = None, None, None
    while True:
        query = {"from": start, "to": end, "order": "registered"}
        if cursor:
            query["cursor"] = cursor
        url = REGISTRY + "api/v1/results?" + urllib.parse.urlencode(query)
        path = root / "registry" / f"page-{len(pages) + 1:04}.json"
        download(url, path)
        page = read_json(path)
        if page.get("schema_version") != 1 or not isinstance(page.get("entries"), list):
            raise ValueError("Unsupported registry listing schema")
        if page.get("dropped"):
            raise ValueError("Registry omitted invalid rows; refusing an incomplete snapshot")
        if not pages:
            revision, release = page["revision"], page["release"]
        if (page["revision"], page["release"]) != (revision, release):
            raise ValueError("Registry changed while fetching; rerun to start a consistent snapshot")
        for entry in page["entries"]:
            entry_key(entry)
            if entry["id"] in seen_ids or not start <= entry["id"][8:18] <= end:
                raise ValueError(f"Duplicate or out-of-month registry entry: {entry['id']}")
            if entry.get("status") != "registered":
                raise ValueError(f"Unexpected entry status: {entry['id']}")
            seen_ids.add(entry["id"])
            entries.append(entry)
        pages.append({"path": str(path.relative_to(root)), "url": url, "sha256": sha256(path)})
        cursor = page["next"]
        if cursor is None:
            break
        if not isinstance(cursor, str) or not cursor or cursor in seen_cursors:
            raise ValueError("Invalid or repeated registry cursor")
        seen_cursors.add(cursor)
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "month": month,
        "selection": "latest active version per ID, filtered by first registration date",
        "registry_revision": revision,
        "registry_release": release,
        "pages": pages,
        "entries": sorted(entries, key=lambda entry: entry["id"]),
    }
    write_json(manifest_path, manifest)
    return manifest


def snapshot_selection(root: Path, selection: Path) -> dict:
    """Freeze an explicit set of entry versions without a moving registry query."""
    selected = read_json(selection)
    if selected.get("schema_version") != 1 or not selected.get("entries"):
        raise ValueError("Selection must contain schema_version: 1 and nonempty entries")
    keys = []
    for entry in selected["entries"]:
        key = entry_key(entry)
        source_identity(entry)
        if entry.get("path") != f"entries/{key}.json":
            raise ValueError(f"Selection has an unexpected record path: {key}")
        keys.append(key)
    if len(set(keys)) != len(keys):
        raise ValueError("Selection contains duplicate entry versions")
    order = selected.get("problem_order", sorted(keys))
    if (not isinstance(order, list) or not all(isinstance(key, str) for key in order)
            or sorted(order) != sorted(keys)):
        raise ValueError("Selection problem_order must list each entry key once")
    entries = dict(zip(keys, selected["entries"], strict=True))
    manifest_path = root / "snapshot.json"
    digest = sha256(selection)
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if manifest.get("selection_sha256") != digest:
            raise ValueError("Output already contains another selection; choose a new output directory")
        return manifest
    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "selection": "explicit pinned entry versions",
        "selection_sha256": digest,
        "entries": [entries[key] for key in order],
        "pages": [],
    }
    write_json(root / "selection.json", selected)
    write_json(manifest_path, manifest)
    return manifest


def source_identity(record: dict) -> tuple[str, str]:
    repository, commit = record["source"]["repository"], record["source"]["commit"]
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError(f"Invalid GitHub repository: {repository!r}")
    if not re.fullmatch(r"[a-f0-9]{40}", commit):
        raise ValueError("Repository commit must be a full immutable SHA")
    relative_path(repository)
    return repository, commit


def source_archive(record: dict, cache: Path) -> tuple[Path, dict]:
    repository, commit = source_identity(record)
    path = cache / repository / f"{commit}.tar.gz"
    receipt_path = path.with_suffix(".json")
    with exclusive_lock(path.with_suffix(".lock"), blocking=True):
        if path.exists() and receipt_path.exists():
            receipt = read_json(receipt_path)
            if sha256(path) == receipt["sha256"]:
                return path, receipt
        mirrors = [
            row["fork_repository"]
            for row in record.get("preservation", {}).get("repositories", [])
            if row.get("source_repository") == repository and row.get("commit") == commit
        ]
        for candidate in [repository, *mirrors]:
            source_identity({"source": {"repository": candidate, "commit": commit}})
            url = f"https://codeload.github.com/{candidate}/tar.gz/{commit}"
            try:
                download(url, path, max_bytes=MAX_ARCHIVE_BYTES)
                break
            except urllib.error.HTTPError as error:
                if error.code not in {404, 410} or candidate == [repository, *mirrors][-1]:
                    raise
        receipt = {"url": url, "sha256": sha256(path), "repository": repository, "commit": commit}
        write_json(receipt_path, receipt)
        return path, receipt


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract a GitHub tarball without allowing links or paths outside its tree."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as source:
            members, total, prefix = [], 0, None
            for member in source:
                name = relative_path(member.name.rstrip("/"))
                first = name.split("/", 1)[0]
                prefix = prefix or first
                if first != prefix or not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                total += member.size
                if total > MAX_EXTRACTED_BYTES or len(members) >= 100_000:
                    raise ValueError("Source archive exceeds extraction limits")
                members.append(member)
            if not members:
                raise ValueError("Empty source archive")
            source.extractall(staging, members=members, filter="data")
        tree = staging / prefix
        if not tree.is_dir() or tree.is_symlink():
            raise ValueError("Expected a single source directory")
        # No .git directory is accepted, including through a symlink.
        for path in tree.rglob("*"):
            if path.name == ".git" or not path.resolve().is_relative_to(tree.resolve()):
                raise ValueError(f"Unsafe source path: {path.relative_to(tree)}")
        tree.rename(destination)


def normalize_record(record: dict, source_root: Path) -> dict:
    """Resolve registered entry points, check source hashes, and expose useful links."""
    key = entry_key(record)
    repository, commit = source_identity(record)
    formalization = record["formalization"]
    project = record["source"].get("project_path") or ""
    project_root = within(source_root, project) if project else source_root
    paths = {
        "challenge_path": formalization["challenge_path"],
        "solution_path": formalization["solution_path"],
        "comparator_path": formalization["comparator_config_path"],
        "metadata_path": formalization["formalization_metadata_path"],
        "lakefile_path": formalization["lakefile_path"],
        "toolchain_path": f"{project}/lean-toolchain" if project else "lean-toolchain",
    }
    hashes = {}
    for name, relative in paths.items():
        path = within(source_root, relative)
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(project_root.resolve()):
            raise ValueError(f"Missing or out-of-project {name}: {relative}")
        hashes[relative] = sha256(path)
        expected = record.get("verification", {}).get(name.replace("_path", "_sha256"))
        if expected and hashes[relative] != expected:
            raise ValueError(f"Registry source hash mismatch: {relative}")
    comparator = read_json(source_root / paths["comparator_path"])
    for field in ("theorem_names", "definition_names", "permitted_axioms"):
        if comparator.get(field, []) != formalization.get(field, []):
            raise ValueError(f"Registry/Comparator mismatch: {field}")
    toolchain = (source_root / paths["toolchain_path"]).read_text().strip()
    if toolchain != formalization["lean_toolchain"]:
        raise ValueError("Registry/source Lean toolchain mismatch")
    github = f"https://github.com/{repository}"
    blob = f"{github}/blob/{commit}/"
    return {
        "key": key,
        "id": record["id"],
        "version": record["version"],
        "title": record["title"],
        "abstract": record.get("abstract", ""),
        "authors": record.get("authors", []),
        "repository": repository,
        "commit": commit,
        "github_url": github,
        "project_url": f"{github}/tree/{commit}" + ("/" + urllib.parse.quote(project) if project else ""),
        "project_path": project,
        **paths,
        **{name.replace("_path", "_url"): blob + urllib.parse.quote(value) for name, value in paths.items()},
        "challenge_module": comparator["challenge_module"],
        "solution_module": comparator["solution_module"],
        "theorem_names": comparator["theorem_names"],
        "definition_names": comparator.get("definition_names", []),
        "lean_toolchain": toolchain,
        "file_hashes": hashes,
        "provenance": record.get("provenance", {}),
        "source": record["source"],
        "formalization": formalization,
        "fetch_status": "ready",
    }


def fetch_entry(projection: dict, root: Path) -> dict:
    key = entry_key(projection)
    output = root / "entries" / f"{key}.json"
    record_path = root / "records" / f"{key}.json"
    source_root = root / "sources" / key
    previous = {}
    try:
        if output.exists():
            previous = read_json(output)
            if previous.get("fetch_status") == "modified":
                raise ModifiedSourceError(previous["error"])
            if previous.get("fetch_status") == "ready":
                if not record_path.is_file() or sha256(record_path) != previous["record_sha256"]:
                    raise ModifiedSourceError("Cached registry record changed; use a fresh snapshot directory")
                if source_root.is_dir() and tree_sha256(source_root) == previous["source_tree_sha256"]:
                    return previous
                raise ModifiedSourceError("Cached source was modified; use a fresh snapshot directory")
        if not record_path.exists():
            record_relative = relative_path(projection["path"])
            if record_relative != f"entries/{key}.json":
                raise ValueError("Unexpected registry record path")
            download(REGISTRY + record_relative, record_path)
        record = read_json(record_path)
        if entry_key(record) != key or record.get("status") != "registered":
            raise ValueError("Record does not match the selected active entry")
        if source_identity(record) != source_identity(projection):
            raise ValueError("Record and listing disagree about the source commit")
        archive, receipt = source_archive(record, root / "archives")
        if source_root.exists():
            shutil.rmtree(source_root)
        extract_archive(archive, source_root)
        entry = normalize_record(record, source_root)
        entry.update(
            record_sha256=sha256(record_path),
            source_tree_sha256=tree_sha256(source_root),
            archive=receipt,
            fetched_at=utc_now(),
        )
    except Exception as error:
        entry = {
            **previous,
            "key": key,
            "id": projection["id"],
            "version": projection["version"],
            "title": projection.get("title", ""),
            "fetch_status": "modified" if isinstance(error, ModifiedSourceError) else "failed",
            "error": str(error),
            "github_url": f"https://github.com/{projection.get('source', {}).get('repository', '')}",
        }
    write_json(output, entry)
    return entry


def fetch(root: Path, month: str | None, workers: int, *, selection: Path | None = None) -> list[dict]:
    with exclusive_lock(root / ".fetch.lock"):
        manifest = snapshot_selection(root, selection) if selection else snapshot_registry(root, month)
        print(f"Frozen {len(manifest['entries'])} entries ({manifest.get('selection', 'registry')})", flush=True)
        entries = []
        write_index(root, entries)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch_entry, entry, root) for entry in manifest["entries"]]
            for future in concurrent.futures.as_completed(futures):
                entry = future.result()
                entries.append(entry)
                print(
                    f"[{len(entries)}/{len(futures)}] {entry['key']}: {entry['fetch_status']} {entry.get('error', '')}",
                    flush=True,
                )
                write_index(root, sorted(entries, key=lambda row: row["key"]))
        failures = sum(entry["fetch_status"] != "ready" for entry in entries)
        write_json(root / "fetch-summary.json", {"finished_at": utc_now(), "total": len(entries), "failed": failures})
        return entries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--month", help="First registration month, YYYY-MM")
    inputs.add_argument("--selection", type=Path, help="JSON file pinning explicit entry versions and source commits")
    parser.add_argument("--output", required=True, type=Path, help="Frozen source snapshot directory")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent source downloads")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        entries = fetch(args.output.resolve(), args.month, args.workers, selection=args.selection)
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(1, f"Fetch failed: {error}\n")
    return int(any(entry["fetch_status"] != "ready" for entry in entries))


if __name__ == "__main__":
    raise SystemExit(main())
