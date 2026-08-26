"""Per-user private GUI state: private tags, saved views, project locations.

Private state lives outside the project: one SQLite database per project under
``<home>/.qphase/gui/<project_id>.sqlite``. It never feeds the shared catalog
read model — :class:`~qphase.service.catalog.CatalogService` overlays it at
query time. The file is created lazily on the first write; pure reads and
deletes on a missing database are no-ops so read-only usage never touches the
user's home directory.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = ["UserPrivateStore"]

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS private_tags (
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    created_at REAL NOT NULL,
    PRIMARY KEY (object_kind, object_id, tag)
);
CREATE TABLE IF NOT EXISTS saved_views (
    name TEXT PRIMARY KEY,
    query_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS project_locations (
    project_id TEXT PRIMARY KEY,
    root TEXT NOT NULL,
    last_seen REAL NOT NULL
);
"""


class UserPrivateStore:
    """SQLite-backed store of one user's private state for one project."""

    def __init__(self, project_id: str, home: Path | None = None) -> None:
        self.project_id = project_id
        self.home = Path.home() if home is None else Path(home)

    @property
    def path(self) -> Path:
        return self.home / ".qphase" / "gui" / f"{self.project_id}.sqlite"

    def add_private_tag(self, object_kind: str, object_id: str, tag: str) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR IGNORE INTO private_tags VALUES (?, ?, ?, ?)",
                    (object_kind, object_id, tag, time.time()),
                )
        finally:
            connection.close()

    def remove_private_tag(self, object_kind: str, object_id: str, tag: str) -> None:
        if not self.path.exists():
            return
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "DELETE FROM private_tags"
                    " WHERE object_kind = ? AND object_id = ? AND tag = ?",
                    (object_kind, object_id, tag),
                )
        finally:
            connection.close()

    def list_private_tags(
        self, object_kind: str, object_id: str | None = None
    ) -> list[tuple[str, str]]:
        """Return ``(object_id, tag)`` pairs, optionally for one object."""
        if not self.path.exists():
            return []
        if object_id is None:
            sql = (
                "SELECT object_id, tag FROM private_tags"
                " WHERE object_kind = ? ORDER BY created_at, rowid"
            )
            params: tuple[str, ...] = (object_kind,)
        else:
            sql = (
                "SELECT object_id, tag FROM private_tags"
                " WHERE object_kind = ? AND object_id = ? ORDER BY created_at, rowid"
            )
            params = (object_kind, object_id)
        connection = self._connect()
        try:
            rows = connection.execute(sql, params).fetchall()
        finally:
            connection.close()
        return [(str(row[0]), str(row[1])) for row in rows]

    def save_view(self, name: str, query: Mapping[str, Any]) -> None:
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO saved_views VALUES (?, ?)",
                    (name, json.dumps(dict(query), allow_nan=False)),
                )
        finally:
            connection.close()

    def list_views(self) -> list[tuple[str, dict[str, Any]]]:
        """Return ``(name, query payload)`` pairs ordered by name."""
        if not self.path.exists():
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT name, query_json FROM saved_views ORDER BY name"
            ).fetchall()
        finally:
            connection.close()
        return [(str(row[0]), dict(json.loads(row[1]))) for row in rows]

    def delete_view(self, name: str) -> None:
        if not self.path.exists():
            return
        connection = self._connect()
        try:
            with connection:
                connection.execute("DELETE FROM saved_views WHERE name = ?", (name,))
        finally:
            connection.close()

    def record_location(self, root: Path) -> None:
        """Remember the project root and refresh its last-seen timestamp."""
        connection = self._connect()
        try:
            with connection:
                connection.execute(
                    "INSERT OR REPLACE INTO project_locations VALUES (?, ?, ?)",
                    (self.project_id, str(root), time.time()),
                )
        finally:
            connection.close()

    def list_locations(self) -> list[tuple[str, str, float]]:
        """Return ``(project_id, root, last_seen)`` most recent first."""
        if not self.path.exists():
            return []
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT project_id, root, last_seen FROM project_locations"
                " ORDER BY last_seen DESC"
            ).fetchall()
        finally:
            connection.close()
        return [(str(row[0]), str(row[1]), float(row[2])) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.executescript(_SCHEMA_SQL)
        return connection
