"""Small, disposable score summaries for the output viewer.

Full traces can be several GB per competition. Keep those out of the overview
and reuse summaries across refreshes, competition switches, and server restarts.
"""

import json
import logging
import sqlite3
from functools import lru_cache
from pathlib import Path

from matharena.json_zst import load_json_zst


_SUMMARY_FIELDS = (
    "correct", "warnings", "llm_annotation", "manual_overwrite", "cost", "detailed_costs",
)


def _signature(path):
    stat = path.stat()
    return f"{stat.st_mtime_ns}:{stat.st_ctime_ns}:{stat.st_size}:{stat.st_ino}"


def _summary(path):
    record = load_json_zst(path)
    return {key: record[key] for key in _SUMMARY_FIELDS if key in record}


class ViewerResultCache:
    def __init__(self, directory):
        self.database = Path(directory) / "results.sqlite3"

    def summaries(self, paths):
        paths = [Path(path).resolve() for path in paths]
        try:
            self.database.parent.mkdir(parents=True, exist_ok=True)
            connection = sqlite3.connect(self.database)
        except (OSError, sqlite3.Error) as error:
            logging.getLogger(__name__).warning("Viewer cache unavailable: %s", error)
            return {str(path): _summary(path) for path in paths}

        try:
            with connection:
                # Version the table when the summary format changes.
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS summaries_v1 "
                    "(path TEXT PRIMARY KEY, signature TEXT NOT NULL, summary TEXT NOT NULL)"
                )
                summaries = {}
                updates = []
                for path in paths:
                    key = str(path)
                    signature = _signature(path)
                    row = connection.execute(
                        "SELECT summary FROM summaries_v1 WHERE path = ? AND signature = ?",
                        (key, signature),
                    ).fetchone()
                    if row is not None:
                        try:
                            summaries[key] = json.loads(row[0])
                            continue
                        except ValueError:
                            pass  # A damaged cache entry can be rebuilt from its source.
                    summaries[key] = _summary(path)
                    updates.append((key, signature, json.dumps(summaries[key])))
                # Do not hold a database write lock while decompressing results.
                connection.executemany(
                    "INSERT OR REPLACE INTO summaries_v1 VALUES (?, ?, ?)", updates
                )
                return summaries
        except sqlite3.Error as error:
            logging.getLogger(__name__).warning("Viewer cache unavailable: %s", error)
            return {str(path): _summary(path) for path in paths}
        finally:
            connection.close()

    def detail(self, path):
        path = Path(path).resolve()
        return self._detail(path, _signature(path))

    @lru_cache(maxsize=4)
    def _detail(self, path, signature):
        # The signature invalidates modified files, including manual overrides.
        # Bound this cache: even a single problem can contain a very large trace.
        return load_json_zst(path)
