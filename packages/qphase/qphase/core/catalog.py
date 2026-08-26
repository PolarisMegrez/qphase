"""Project object catalog: a rebuildable SQLite read model over disk truth.

The catalog at ``<project>/.qphase/object_catalog.sqlite`` indexes workflows,
executions, sessions, artifacts and artifact occurrences together with their
facets and effective tags. It is a *read model*: the session directories,
workflow files, execution records and annotation documents remain the only
truth, and :meth:`ProjectObjectCatalog.reindex` must rebuild the identical
content from disk at any time.

Effective tags are materialized at index time with full provenance (source,
assignment id, inherited and shadowed flags) so queries never recompute the
inheritance chain:

```text
workflow declared -> execution submission -> session annotation
                 -> job declared -> occurrence annotation
```

Cardinality-one namespaces shadow farther assignments when a nearer object
sets the same namespace; ``inherit = false`` namespaces never flow downward.
Lifecycle never inherits; retention inherits from session to occurrence when
the policy allows it.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotations import ARTIFACT_ANNOTATIONS_FILENAME
from .persistence import ProjectStateStore
from .project import ProjectContext
from .tags import TagPolicy, canonicalize_tag_syntax, load_tag_policy
from .utils import load_yaml
from .workflow import WorkflowCatalog

__all__ = [
    "CATALOG_FILENAME",
    "CATALOG_SCHEMA",
    "ArtifactOccurrence",
    "CatalogQuery",
    "CatalogStats",
    "EffectiveTag",
    "ProjectObjectCatalog",
    "compute_effective_tags",
]

#: Catalog file name inside the project ``.qphase`` directory.
CATALOG_FILENAME = "object_catalog.sqlite"

#: Read-model schema version; a mismatch forces a rebuild.
CATALOG_SCHEMA = "qphase.catalog/1"

_NAMESPACE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

OBJECT_KINDS = ("workflow", "execution", "session", "artifact", "occurrence")

_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE workflows (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    collection TEXT,
    relative_path TEXT NOT NULL,
    revision TEXT NOT NULL,
    job_count INTEGER NOT NULL
);
CREATE TABLE executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    state TEXT NOT NULL,
    submitted_at TEXT NOT NULL
);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT,
    status TEXT NOT NULL,
    start_time TEXT,
    path TEXT NOT NULL,
    lifecycle TEXT,
    retention TEXT,
    alias TEXT,
    note TEXT
);
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    bundle_type TEXT,
    product_schemas_json TEXT NOT NULL,
    parents_json TEXT NOT NULL,
    path TEXT NOT NULL,
    health TEXT NOT NULL,
    lifecycle TEXT,
    retention TEXT
);
CREATE TABLE occurrences (
    id TEXT PRIMARY KEY,
    artifact_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    job_name TEXT NOT NULL,
    path TEXT NOT NULL,
    retention TEXT,
    effective_retention TEXT
);
CREATE TABLE effective_tags (
    object_kind TEXT NOT NULL,
    object_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    source TEXT NOT NULL,
    assignment_id TEXT,
    policy_revision TEXT,
    inherited INTEGER NOT NULL,
    shadowed INTEGER NOT NULL
);
CREATE INDEX idx_effective_tags_tag ON effective_tags (tag, object_kind);
CREATE INDEX idx_effective_tags_object ON effective_tags (object_kind, object_id);
CREATE INDEX idx_occurrences_artifact ON occurrences (artifact_id);
CREATE INDEX idx_occurrences_session ON occurrences (session_id);
"""

#: Facet columns queries may filter on, per object kind.
_FACETS: dict[str, tuple[str, ...]] = {
    "workflow": ("id", "title", "collection", "relative_path", "revision"),
    "execution": ("id", "workflow_id", "state", "submitted_at"),
    "session": (
        "id",
        "workflow_id",
        "status",
        "start_time",
        "lifecycle",
        "retention",
    ),
    "artifact": (
        "id",
        "created_at",
        "bundle_type",
        "health",
        "lifecycle",
        "retention",
    ),
    "occurrence": (
        "id",
        "artifact_id",
        "session_id",
        "job_name",
        "retention",
        "effective_retention",
    ),
}

#: Stable sort column pairs per object kind.
_SORT: dict[str, tuple[str, str]] = {
    "workflow": ("id", "id"),
    "execution": ("submitted_at", "id"),
    "session": ("start_time", "id"),
    "artifact": ("created_at", "id"),
    "occurrence": ("id", "id"),
}


@dataclass(frozen=True)
class ArtifactOccurrence:
    """One producing occurrence of an artifact inside a session job."""

    artifact_id: str
    session_id: str
    job_name: str

    @property
    def object_id(self) -> str:
        """Deterministic occurrence identity (survives catalog rebuilds)."""
        return f"{self.artifact_id}:{self.session_id}:{self.job_name}"


@dataclass
class EffectiveTag:
    """One materialized effective tag with provenance."""

    tag: str
    source: str
    assignment_id: str | None
    policy_revision: str | None
    inherited: bool
    shadowed: bool = False


@dataclass(frozen=True)
class CatalogQuery:
    """Filter, pagination and stable sort for catalog object listing.

    Tag predicates match canonical tags; ``effective=False`` restricts the
    match to tags assigned directly on the object (not inherited).
    """

    object_kind: str
    facets: Mapping[str, str] = field(default_factory=dict)
    ranges: Mapping[str, tuple[str | None, str | None]] = field(default_factory=dict)
    tags_all: tuple[str, ...] = ()
    tags_any: tuple[str, ...] = ()
    tags_without: tuple[str, ...] = ()
    tag_descendant_of: str | None = None
    tag_namespace: str | None = None
    lifecycle: str | None = None
    retention: str | None = None
    effective: bool = True
    limit: int = 100
    offset: int = 0

    def __post_init__(self) -> None:
        """Canonicalize tag filters and reject unknown object kinds."""
        if self.object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown catalog object kind {self.object_kind!r}")
        object.__setattr__(self, "tags_all", _canonical_tags(self.tags_all))
        object.__setattr__(self, "tags_any", _canonical_tags(self.tags_any))
        object.__setattr__(self, "tags_without", _canonical_tags(self.tags_without))
        if self.tag_descendant_of is not None:
            object.__setattr__(
                self,
                "tag_descendant_of",
                canonicalize_tag_syntax(self.tag_descendant_of),
            )
        if self.tag_namespace is not None:
            namespace = self.tag_namespace.strip().lower()
            if not _NAMESPACE_PATTERN.fullmatch(namespace):
                raise ValueError(
                    f"invalid tag namespace {self.tag_namespace!r}"
                )
            object.__setattr__(self, "tag_namespace", namespace)


def _canonical_tags(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonicalize_tag_syntax(value) for value in values)


@dataclass(frozen=True)
class CatalogStats:
    """Row counts of one catalog rebuild."""

    workflows: int
    executions: int
    sessions: int
    artifacts: int
    occurrences: int
    effective_tags: int
    duration_seconds: float


class ProjectObjectCatalog:
    """Rebuildable SQLite index of all addressable objects in one project."""

    def __init__(self, project: ProjectContext) -> None:
        self.project = project

    @property
    def path(self) -> Path:
        return self.project.root / ".qphase" / CATALOG_FILENAME

    def reindex(self) -> CatalogStats:
        """Rebuild the whole read model from disk truth in one transaction."""
        started = time.monotonic()
        policy = load_tag_policy(self.project)
        policy_revision = policy.revision if policy is not None else None
        scan = _Scanner(self.project, policy, policy_revision).collect()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = self._connect(fresh=True)
        try:
            with connection:
                connection.executemany(
                    "INSERT INTO workflows VALUES (?, ?, ?, ?, ?, ?)",
                    scan["workflows"],
                )
                connection.executemany(
                    "INSERT INTO executions VALUES (?, ?, ?, ?)",
                    scan["executions"],
                )
                connection.executemany(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    scan["sessions"],
                )
                connection.executemany(
                    "INSERT INTO artifacts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    scan["artifacts"],
                )
                connection.executemany(
                    "INSERT INTO occurrences VALUES (?, ?, ?, ?, ?, ?, ?)",
                    scan["occurrences"],
                )
                connection.executemany(
                    "INSERT INTO effective_tags VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    scan["effective_tags"],
                )
        finally:
            connection.close()
        return CatalogStats(
            workflows=len(scan["workflows"]),
            executions=len(scan["executions"]),
            sessions=len(scan["sessions"]),
            artifacts=len(scan["artifacts"]),
            occurrences=len(scan["occurrences"]),
            effective_tags=len(scan["effective_tags"]),
            duration_seconds=time.monotonic() - started,
        )

    def query(self, query: CatalogQuery) -> list[dict[str, Any]]:
        """List objects of one kind matching the query, stably sorted."""
        table = f"{query.object_kind}s"
        where: list[str] = []
        params: list[Any] = []
        for facet, value in query.facets.items():
            self._require_column(query, facet)
            where.append(f"o.{facet} = ?")
            params.append(value)
        for facet, (low, high) in query.ranges.items():
            self._require_column(query, facet)
            if low is not None:
                where.append(f"o.{facet} >= ?")
                params.append(low)
            if high is not None:
                where.append(f"o.{facet} <= ?")
                params.append(high)
        if query.lifecycle is not None:
            self._require_column(query, "lifecycle")
            where.append("o.lifecycle = ?")
            params.append(query.lifecycle)
        if query.retention is not None:
            column = (
                "effective_retention"
                if query.object_kind == "occurrence"
                else "retention"
            )
            self._require_column(query, column)
            where.append(f"o.{column} = ?")
            params.append(query.retention)
        self._tag_predicates(query, where, params)
        sort_column, tiebreaker = _SORT[query.object_kind]
        sql = f"SELECT o.* FROM {table} o "  # built from whitelisted names
        if where:
            sql += "WHERE " + " AND ".join(where) + " "
        sql += f"ORDER BY o.{sort_column}, o.{tiebreaker} LIMIT ? OFFSET ?"
        params.extend([query.limit, query.offset])
        connection = self._connect()
        try:
            cursor = connection.execute(sql, params)
            names = [description[0] for description in cursor.description or []]
            rows = cursor.fetchall()
        finally:
            connection.close()
        return [dict(zip(names, row, strict=True)) for row in rows]

    def effective_tags(self, object_kind: str, object_id: str) -> list[EffectiveTag]:
        """Return the materialized effective tags of one object."""
        if object_kind not in OBJECT_KINDS:
            raise ValueError(f"unknown catalog object kind {object_kind!r}")
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT tag, source, assignment_id, policy_revision, inherited,"
                " shadowed FROM effective_tags"
                " WHERE object_kind = ? AND object_id = ? ORDER BY rowid",
                (object_kind, object_id),
            ).fetchall()
        finally:
            connection.close()
        return [
            EffectiveTag(
                tag=row[0],
                source=row[1],
                assignment_id=row[2],
                policy_revision=row[3],
                inherited=bool(row[4]),
                shadowed=bool(row[5]),
            )
            for row in rows
        ]

    def locate_artifact(self, artifact_id: str) -> str | None:
        """Return the indexed session-relative artifact path, if present."""
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT path FROM artifacts WHERE id = ?", (artifact_id,)
            ).fetchone()
        finally:
            connection.close()
        return row[0] if row else None

    @staticmethod
    def _require_column(query: CatalogQuery, facet: str) -> None:
        if facet not in _FACETS[query.object_kind]:
            raise ValueError(
                f"unknown {query.object_kind} facet {facet!r};"
                f" allowed: {sorted(_FACETS[query.object_kind])}"
            )

    @staticmethod
    def _tag_predicates(
        query: CatalogQuery, where: list[str], params: list[Any]
    ) -> None:
        base = "et.object_kind = ? AND et.object_id = o.id AND et.shadowed = 0"
        if not query.effective:
            base += " AND et.inherited = 0"
        for tag in query.tags_all:
            where.append(
                f"EXISTS (SELECT 1 FROM effective_tags et WHERE {base} AND et.tag = ?)"
            )
            params.extend([query.object_kind, tag])
        if query.tags_any:
            placeholders = ", ".join("?" for _ in query.tags_any)
            where.append(
                f"EXISTS (SELECT 1 FROM effective_tags et WHERE {base}"
                f" AND et.tag IN ({placeholders}))"
            )
            params.append(query.object_kind)
            params.extend(query.tags_any)
        for tag in query.tags_without:
            where.append(
                f"NOT EXISTS (SELECT 1 FROM effective_tags et WHERE {base}"
                " AND et.tag = ?)"
            )
            params.extend([query.object_kind, tag])
        if query.tag_descendant_of is not None:
            where.append(
                f"EXISTS (SELECT 1 FROM effective_tags et WHERE {base}"
                " AND (et.tag = ? OR et.tag LIKE ?))"
            )
            params.extend(
                [
                    query.object_kind,
                    query.tag_descendant_of,
                    query.tag_descendant_of + "/%",
                ]
            )
        if query.tag_namespace is not None:
            # tag_namespace is validated against the namespace pattern in
            # __post_init__, so the LIKE pattern carries no user metacharacters
            # beyond the (harmless) underscore wildcard.
            where.append(
                f"EXISTS (SELECT 1 FROM effective_tags et WHERE {base}"
                " AND et.tag LIKE ?)"
            )
            params.extend([query.object_kind, query.tag_namespace + ":%"])

    def _connect(self, *, fresh: bool = False) -> sqlite3.Connection:
        path = self.path
        if fresh:
            path.unlink(missing_ok=True)
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA_SQL)
            self._stamp_meta(connection)
            return connection
        try:
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=WAL")
            version = connection.execute(
                "SELECT value FROM meta WHERE key = 'schema'"
            ).fetchone()
            if version is None or version[0] != CATALOG_SCHEMA:
                raise _CatalogStaleError
            return connection
        except (sqlite3.DatabaseError, _CatalogStaleError):
            # Corrupt or outdated read model: rebuild the empty schema; the
            # next reindex repopulates it from disk truth.
            connection.close()
            path.unlink(missing_ok=True)
            connection = sqlite3.connect(path)
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA_SQL)
            self._stamp_meta(connection)
            return connection

    def _stamp_meta(self, connection: sqlite3.Connection) -> None:
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema", CATALOG_SCHEMA), ("project_id", self.project.project_id)],
        )
        connection.commit()


class _CatalogStaleError(Exception):
    """The on-disk catalog predates the current read-model schema."""


def compute_effective_tags(
    levels: list[tuple[str, list[tuple[str, str | None]], bool]],
    policy_revision: str | None,
    policy: TagPolicy | None,
) -> list[EffectiveTag]:
    """Merge far-to-near tag levels into effective tags with provenance.

    ``levels`` is an ordered list of ``(source, [(tag, assignment_id)],
    is_self)`` triples. Tags from levels other than the object's own are
    marked inherited and skipped when their namespace disables inheritance;
    cardinality-one namespaces shadow all farther assignments.
    """
    effective: list[EffectiveTag] = []
    for source, items, is_self in levels:
        for tag, assignment_id in items:
            namespace = tag.split(":", 1)[0]
            rule = policy.namespaces.get(namespace) if policy is not None else None
            if not is_self and rule is not None and not rule.inherit:
                continue
            if rule is not None and rule.cardinality == "one":
                for prior in effective:
                    if prior.tag.split(":", 1)[0] == namespace:
                        prior.shadowed = True
            effective.append(
                EffectiveTag(
                    tag=tag,
                    source=source,
                    assignment_id=assignment_id,
                    policy_revision=policy_revision,
                    inherited=not is_self,
                )
            )
    return effective


class _Scanner:
    """Collect catalog rows from workflow files, records and sessions."""

    def __init__(
        self,
        project: ProjectContext,
        policy: TagPolicy | None,
        policy_revision: str | None,
    ) -> None:
        self.project = project
        self.policy = policy
        self.policy_revision = policy_revision
        self.store = ProjectStateStore(project)
        self._seen_artifacts: set[str] = set()
        self.rows: dict[str, list[tuple[Any, ...]]] = {
            "workflows": [],
            "executions": [],
            "sessions": [],
            "artifacts": [],
            "occurrences": [],
            "effective_tags": [],
        }

    def collect(self) -> dict[str, list[tuple[Any, ...]]]:
        self._scan_workflows()
        self._scan_executions()
        self._scan_sessions()
        return self.rows

    def _tags(self, object_kind: str, object_id: str, tags: list[EffectiveTag]) -> None:
        for tag in tags:
            self.rows["effective_tags"].append(
                (
                    object_kind,
                    object_id,
                    tag.tag,
                    tag.source,
                    tag.assignment_id,
                    tag.policy_revision,
                    int(tag.inherited),
                    int(tag.shadowed),
                )
            )

    def _scan_workflows(self) -> None:
        for reference in WorkflowCatalog(self.project).list():
            raw = reference.path.read_bytes()
            self.rows["workflows"].append(
                (
                    reference.id,
                    reference.title,
                    reference.collection,
                    reference.relative_path,
                    hashlib.sha256(raw).hexdigest(),
                    reference.job_count,
                )
            )
            self._tags(
                "workflow",
                reference.id,
                compute_effective_tags(
                    [
                        (
                            "workflow_declared",
                            [(tag, None) for tag in reference.tags],
                            True,
                        )
                    ],
                    self.policy_revision,
                    self.policy,
                ),
            )

    def _scan_executions(self) -> None:
        for payload in self.store.load_executions():
            workflow = payload.get("workflow") or {}
            tags = [str(tag) for tag in payload.get("submission_tags", [])]
            self.rows["executions"].append(
                (
                    str(payload["execution_id"]),
                    str(workflow.get("id", "")),
                    str(payload.get("state", "unknown")),
                    str(payload.get("submitted_at", "")),
                )
            )
            self._tags(
                "execution",
                str(payload["execution_id"]),
                compute_effective_tags(
                    [("execution_submission", [(tag, None) for tag in tags], True)],
                    self.policy_revision,
                    self.policy,
                ),
            )

    def _scan_sessions(self) -> None:
        root = self.project.session_root
        if not root.exists():
            return
        for manifest_path in sorted(root.rglob("session_manifest.json")):
            if ".trash" in manifest_path.parts:
                continue
            self._scan_session(manifest_path.parent)

    def _scan_session(self, session_dir: Path) -> None:
        manifest = self.store.load_session_manifest(session_dir)
        session_id = str(manifest.get("session_id") or session_dir.name)
        annotations = self.store.load_session_annotations(session_dir) or {}
        submission_tags = [str(tag) for tag in manifest.get("submission_tags", [])]
        snapshot_tags, job_tags = self._snapshot_tags(session_dir)

        self.rows["sessions"].append(
            (
                session_id,
                manifest.get("workflow_id"),
                str(manifest.get("status", "unknown")),
                manifest.get("start_time"),
                session_dir.relative_to(self.project.session_root).as_posix(),
                annotations.get("lifecycle"),
                annotations.get("retention"),
                annotations.get("alias"),
                annotations.get("note"),
            )
        )
        # The occurrence chain shares the session's far levels; the session's
        # own annotation level is the occurrence's nearest inherited level.
        inherited_chain: list[tuple[str, list[tuple[str, str | None]], bool]] = [
            ("workflow_declared", [(tag, None) for tag in snapshot_tags], False),
            ("execution_submission", [(tag, None) for tag in submission_tags], False),
            (
                "session_annotation",
                _assignment_pairs(annotations.get("assignments")),
                False,
            ),
        ]
        session_chain: list[tuple[str, list[tuple[str, str | None]], bool]] = [
            (source, items, source == "session_annotation")
            for source, items, _ in inherited_chain
        ]
        self._tags(
            "session",
            session_id,
            compute_effective_tags(session_chain, self.policy_revision, self.policy),
        )

        retention = annotations.get("retention")
        inherit_retention = (
            self.policy.retention_inherits_to_occurrences
            if self.policy is not None
            else True
        )
        occurrence_annotations = annotations.get("occurrences") or {}
        for manifest_file in sorted(session_dir.rglob("artifact_manifest.json")):
            self._scan_artifact(
                manifest_file.parent,
                session_dir,
                session_id,
                inherited_chain,
                retention if inherit_retention else None,
                job_tags,
                occurrence_annotations,
            )

    @staticmethod
    def _snapshot_tags(session_dir: Path) -> tuple[list[str], dict[str, list[str]]]:
        """Return declared tags frozen in the session's workflow snapshot."""
        path = session_dir / "workflow_snapshot.yaml"
        if not path.exists():
            return [], {}
        payload = load_yaml(path)
        if not isinstance(payload, dict):
            return [], {}
        workflow_tags = [
            canonicalize_tag_syntax(str(tag)) for tag in payload.get("tags", [])
        ]
        job_tags: dict[str, list[str]] = {}
        for job in payload.get("jobs", []):
            if isinstance(job, dict) and job.get("name"):
                job_tags[str(job["name"])] = [
                    canonicalize_tag_syntax(str(tag)) for tag in job.get("tags", [])
                ]
        return workflow_tags, job_tags

    def _scan_artifact(
        self,
        artifact_dir: Path,
        session_dir: Path,
        session_id: str,
        inherited_chain: list[tuple[str, list[tuple[str, str | None]], bool]],
        session_retention: str | None,
        job_tags: dict[str, list[str]],
        occurrence_annotations: Mapping[str, Any],
    ) -> None:
        try:
            payload = json.loads(
                (artifact_dir / "artifact_manifest.json").read_text(encoding="utf-8")
            )
            artifact_id = str(payload["artifact_id"])
            health = "ok"
        except (OSError, ValueError, KeyError):
            # A corrupt manifest still indexes the occurrence so the damage is
            # visible in queries; the id falls back to the directory name.
            artifact_id = artifact_dir.name
            payload = {}
            health = "corrupt"
        job_name = artifact_dir.relative_to(session_dir).as_posix()
        annotations = self._load_artifact_annotations(artifact_dir)
        occurrence = ArtifactOccurrence(
            artifact_id=artifact_id, session_id=session_id, job_name=job_name
        )
        occurrence_annotation = occurrence_annotations.get(artifact_id) or {}

        bundle = payload.get("bundle")
        product_schemas = sorted(
            {
                str(entry.get("product_schema"))
                for entry in payload.get("products", [])
                if isinstance(entry, dict)
            }
        )
        # The artifact row is identity-scoped: the first occurrence wins and
        # later occurrences of the same artifact add only occurrence rows.
        if artifact_id not in self._seen_artifacts:
            self._seen_artifacts.add(artifact_id)
            self.rows["artifacts"].append(
                (
                    artifact_id,
                    payload.get("created_at"),
                    bundle.get("type_id") if isinstance(bundle, dict) else None,
                    json.dumps(product_schemas),
                    json.dumps(payload.get("parents", [])),
                    artifact_dir.relative_to(self.project.session_root).as_posix(),
                    health,
                    annotations.get("lifecycle"),
                    annotations.get("retention"),
                )
            )
            self._tags(
                "artifact",
                artifact_id,
                compute_effective_tags(
                    [
                        (
                            "artifact_annotation",
                            _assignment_pairs(annotations.get("assignments")),
                            True,
                        )
                    ],
                    self.policy_revision,
                    self.policy,
                ),
            )
        own_retention = occurrence_annotation.get("retention")
        self.rows["occurrences"].append(
            (
                occurrence.object_id,
                artifact_id,
                session_id,
                job_name,
                artifact_dir.relative_to(self.project.session_root).as_posix(),
                own_retention,
                own_retention or session_retention,
            )
        )
        occurrence_chain: list[tuple[str, list[tuple[str, str | None]], bool]] = [
            *inherited_chain,
            (
                "job_declared",
                [(tag, None) for tag in job_tags.get(job_name, [])],
                False,
            ),
            (
                "occurrence_annotation",
                _assignment_pairs(occurrence_annotation.get("assignments")),
                True,
            ),
        ]
        self._tags(
            "occurrence",
            occurrence.object_id,
            compute_effective_tags(occurrence_chain, self.policy_revision, self.policy),
        )

    @staticmethod
    def _load_artifact_annotations(artifact_dir: Path) -> dict[str, Any]:
        path = artifact_dir / ARTIFACT_ANNOTATIONS_FILENAME
        if not path.exists():
            return {}
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload) if isinstance(payload, dict) else {}


def _assignment_pairs(raw: Any) -> list[tuple[str, str | None]]:
    """Flatten annotation assignments into ``(tag, assignment_id)`` pairs."""
    if not isinstance(raw, list):
        return []
    return [
        (str(item["tag"]), str(item["id"]) if item.get("id") else None)
        for item in raw
        if isinstance(item, dict) and item.get("tag")
    ]
