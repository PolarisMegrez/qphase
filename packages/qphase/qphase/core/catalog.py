"""Project object catalog: a rebuildable SQLite read model over disk truth.

The catalog at ``<project>/.qphase/object_catalog.sqlite`` indexes the
project itself, workflow revisions, jobs, executions, sessions, artifacts and
artifact occurrences together with their facets and effective tags. It is a
*read model*: the session directories, workflow files, execution records and
annotation documents remain the only truth, and
:meth:`ProjectObjectCatalog.reindex` must rebuild the identical content from
disk at any time.

Object identity:

- ``project``: the single project row, keyed by project id;
- ``workflow``: one *workflow revision*, keyed by ``workflow_id@revision``
  where the revision is a content hash of the normalized workflow document —
  the current file and every frozen session snapshot of the same workflow
  yield their own rows without overwriting each other;
- ``job``: one logical job of one revision, keyed by ``revision_id:job_name``;
- ``occurrence``: one producing context of an artifact, keyed by
  ``artifact_id:session_id:job_name``.

Effective tags are materialized at index time with full provenance (source,
assignment id, policy revision, inherited and shadowed flags) so queries
never recompute the inheritance chain:

```text
project annotation                                     (project only)
workflow declared -> workflow annotation               (workflow revision)
workflow declared -> job declared -> job annotation    (job)
execution submission -> execution annotation           (execution)
workflow declared -> execution submission -> session annotation    (session)
                 -> job declared -> occurrence annotation          (occurrence)
```

Annotation assignments on the project, workflow revisions, jobs and
executions live in the project annotation document
(``.qphase/project_annotations.json``) and never flow downward: they are the
organization layer of those objects themselves, not new inheritance levels.

Declared levels take their policy revision from the frozen tag snapshot of
the session (or from the current policy for the live workflow file);
annotation assignments carry the revision of the policy that validated them
at write time. Cardinality-one namespaces shadow farther assignments when a
nearer object sets the same namespace; ``inherit = false`` namespaces never
flow downward. Lifecycle never inherits; retention inherits from session to
occurrence when the policy allows it.

Artifact scan contract (artifact schema ``qphase.artifact/4``):

- every artifact manifest is read and fully validated through
  :meth:`qphase.data.store.ArtifactManifest.read`;
- an unreadable location (missing/corrupt manifest) and a manifest of an
  unsupported schema version index **no** object rows; each is recorded in
  the ``location_issues`` table (``corrupt`` / ``unsupported``) instead;
- when two occurrences of one artifact identity disagree on identity facets
  (created_at, bundle type, product schemas, quantities, parents), the
  first occurrence wins the artifact row and the later location is recorded
  as a ``conflict`` location issue;
- annotation documents are validated against their typed schemas
  (``SessionAnnotationDocument`` / ``ArtifactAnnotationDocument`` /
  ``ProjectAnnotationDocument``) including location identity; a corrupt
  document or identity mismatch indexes no annotation layer at all and is
  recorded as an ``annotation`` location issue;
- ``product_schemas_json`` maps product name to the stable schema
  fingerprint; ``quantities_json`` is the sorted set of non-empty physical
  quantities across all products.

Stale detection and reindexing:

1. **Startup recovery** — a catalog file that is corrupt, has an
   unexpected ``schema`` version, or belongs to a different ``project_id``
   is rebuilt from disk truth on first use, never silently trusted.
2. **Fingerprint probe** — read entry points compare a cheap filesystem
   fingerprint (project root; session/artifact manifest and execution
   record counts with newest mtimes; annotation document counts with newest
   mtime; workflow file count with newest mtime; tag policy mtime) against
   the value stored at index time; a mismatch triggers one automatic
   reindex.
3. **Explicit lifecycle triggers** — the scheduler reindexes after session
   initialization and job finalization, the execution service reindexes
   after execution records are persisted, and every annotation write
   through the service layer reindexes.

Trigger boundary (documented contract): state flips of a *running* job do
not themselves reindex eagerly; their manifest mtime flips the fingerprint
probe, so the *next* catalog query rebuilds — the accepted cost of the
catalog being a derived read model, not a real-time status source.
Artifact saves need no per-save hook: a published artifact manifest is
immutable, so every new artifact adds a manifest file and flips the count
probe.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .annotations import (
    ARTIFACT_ANNOTATIONS_FILENAME,
    PROJECT_ANNOTATIONS_FILENAME,
    SESSION_ANNOTATIONS_FILENAME,
    ArtifactAnnotationDocument,
    ObjectAnnotations,
    OccurrenceAnnotations,
    ProjectAnnotationDocument,
    SessionAnnotationDocument,
    TagAssignment,
)
from .errors import QPhaseConfigError
from .locking import file_lock
from .persistence import ProjectStateStore
from .project import ProjectContext
from .tags import (
    TAG_POLICY_FILENAME,
    FrozenNamespaceRule,
    ObjectKind,
    TagPolicy,
    canonicalize_tag_syntax,
    job_tag_assignment_id,
    load_tag_policy,
    workflow_tag_assignment_id,
)
from .utils import load_yaml
from .workflow import WorkflowCatalog, load_workflow, workflow_revision

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
CATALOG_SCHEMA = "qphase.catalog/3"

_NAMESPACE_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

OBJECT_KINDS = (
    "project",
    "workflow",
    "job",
    "execution",
    "session",
    "artifact",
    "occurrence",
)

_SCHEMA_SQL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE projects (
    id TEXT PRIMARY KEY,
    name TEXT,
    root TEXT NOT NULL
);
CREATE TABLE workflows (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    title TEXT NOT NULL,
    collection TEXT,
    relative_path TEXT,
    revision TEXT NOT NULL,
    job_count INTEGER NOT NULL,
    sources_json TEXT NOT NULL
);
CREATE TABLE jobs (
    id TEXT PRIMARY KEY,
    workflow_revision_id TEXT NOT NULL,
    workflow_id TEXT NOT NULL,
    name TEXT NOT NULL,
    engine TEXT,
    model TEXT,
    plugins_json TEXT NOT NULL
);
CREATE TABLE executions (
    id TEXT PRIMARY KEY,
    workflow_id TEXT NOT NULL,
    workflow_revision_id TEXT,
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
    note TEXT,
    submission_tag_policy_revision TEXT
);
CREATE TABLE artifacts (
    id TEXT PRIMARY KEY,
    created_at TEXT,
    bundle_type TEXT,
    product_schemas_json TEXT NOT NULL,
    quantities_json TEXT NOT NULL,
    parents_json TEXT NOT NULL,
    path TEXT NOT NULL,
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
CREATE TABLE location_issues (
    path TEXT NOT NULL,
    kind TEXT NOT NULL,
    message TEXT NOT NULL
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
CREATE INDEX idx_workflows_workflow_id ON workflows (workflow_id);
CREATE INDEX idx_jobs_revision ON jobs (workflow_revision_id);
"""

#: Facet columns queries may filter on, per object kind.
_FACETS: dict[str, tuple[str, ...]] = {
    "project": ("id", "name", "root"),
    "workflow": (
        "id",
        "workflow_id",
        "title",
        "collection",
        "relative_path",
        "revision",
    ),
    "job": (
        "id",
        "workflow_revision_id",
        "workflow_id",
        "name",
        "engine",
        "model",
    ),
    "execution": ("id", "workflow_id", "workflow_revision_id", "state", "submitted_at"),
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
    "project": ("id", "id"),
    "workflow": ("workflow_id", "id"),
    "job": ("id", "id"),
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
        if self.offset < 0:
            raise ValueError(f"offset must be non-negative, got {self.offset}")
        if not 1 <= self.limit <= 10000:
            raise ValueError(f"limit must be within 1..10000, got {self.limit}")
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
                raise ValueError(f"invalid tag namespace {self.tag_namespace!r}")
            object.__setattr__(self, "tag_namespace", namespace)


def _canonical_tags(values: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(canonicalize_tag_syntax(value) for value in values)


def _like_escape(value: str) -> str:
    """Escape LIKE metacharacters so the pattern matches literally."""
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _try_canonical_tag(raw: str) -> str | None:
    """Canonicalize one declared tag, dropping syntax-invalid declarations.

    Invalid declared tags in legacy snapshots are a data-quality issue the
    migration dry-run reports; they must never abort a catalog rebuild.
    """
    try:
        return canonicalize_tag_syntax(raw)
    except QPhaseConfigError:
        return None


@dataclass(frozen=True)
class CatalogStats:
    """Row counts of one catalog rebuild."""

    projects: int
    workflows: int
    jobs: int
    executions: int
    sessions: int
    artifacts: int
    occurrences: int
    effective_tags: int
    location_issues: int
    duration_seconds: float


class ProjectObjectCatalog:
    """Rebuildable SQLite index of all addressable objects in one project.

    ``db_path`` overrides the database location; the migration dry-run uses
    it to rebuild a throwaway copy in a temporary directory without touching
    the project's own catalog.
    """

    def __init__(self, project: ProjectContext, *, db_path: Path | None = None) -> None:
        self.project = project
        self._db_path = db_path

    @property
    def path(self) -> Path:
        if self._db_path is not None:
            return self._db_path
        return self.project.root / ".qphase" / CATALOG_FILENAME

    def reindex(self) -> CatalogStats:
        """Rebuild the whole read model from disk truth in one transaction.

        The rebuild holds a cross-process sibling-file lock and populates a
        temporary database that atomically replaces the live one, so a
        failed rebuild never deletes the previous read model. A state flip
        of a running job (manifest/record mtime) makes the next catalog
        query trigger a rebuild — the accepted cost of keeping the catalog
        a derived read model instead of a live state source.
        """
        with _catalog_lock(self.path), file_lock(self.path):
            return self._reindex_locked()

    def _reindex_locked(self) -> CatalogStats:
        started = time.monotonic()
        fingerprint = self._fingerprint()
        policy = load_tag_policy(self.project)
        policy_revision = policy.revision if policy is not None else None
        scan = _Scanner(self.project, policy, policy_revision).collect()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        build_path = self.path.with_name(self.path.name + ".build")
        build_path.unlink(missing_ok=True)
        try:
            # Default (delete) journal mode: a WAL build would leave live
            # rows in the -wal sidecar that os.replace would not carry over.
            connection = sqlite3.connect(build_path)
            try:
                connection.executescript(_SCHEMA_SQL)
                self._stamp_meta(connection)
                with connection:
                    connection.executemany(
                        "INSERT INTO projects VALUES (?, ?, ?)",
                        scan["projects"],
                    )
                    connection.executemany(
                        "INSERT INTO workflows VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        scan["workflows"],
                    )
                    connection.executemany(
                        "INSERT INTO jobs VALUES (?, ?, ?, ?, ?, ?, ?)",
                        scan["jobs"],
                    )
                    connection.executemany(
                        "INSERT INTO executions VALUES (?, ?, ?, ?, ?)",
                        scan["executions"],
                    )
                    connection.executemany(
                        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                    connection.executemany(
                        "INSERT INTO location_issues VALUES (?, ?, ?)",
                        scan["location_issues"],
                    )
                    connection.execute(
                        "INSERT OR REPLACE INTO meta (key, value) VALUES"
                        " ('fingerprint', ?)",
                        (json.dumps(fingerprint, sort_keys=True),),
                    )
            finally:
                connection.close()
            os.replace(build_path, self.path)
        except BaseException:
            build_path.unlink(missing_ok=True)
            raise
        self._verify_meta()
        return CatalogStats(
            projects=len(scan["projects"]),
            workflows=len(scan["workflows"]),
            jobs=len(scan["jobs"]),
            executions=len(scan["executions"]),
            sessions=len(scan["sessions"]),
            artifacts=len(scan["artifacts"]),
            occurrences=len(scan["occurrences"]),
            effective_tags=len(scan["effective_tags"]),
            location_issues=len(scan["location_issues"]),
            duration_seconds=time.monotonic() - started,
        )

    def _verify_meta(self) -> None:
        """Re-open the replaced database and check its identity stamps."""
        connection = sqlite3.connect(self.path)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
        finally:
            connection.close()
        if (
            meta.get("schema") != CATALOG_SCHEMA
            or meta.get("project_id") != self.project.project_id
        ):
            raise RuntimeError(f"rebuilt catalog failed meta verification: {self.path}")

    def query(self, query: CatalogQuery) -> list[dict[str, Any]]:
        """List objects of one kind matching the query, stably sorted."""
        self._ensure_fresh()
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
        self._ensure_fresh()
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

    def locate_artifact_paths(self, artifact_id: str) -> list[str]:
        """Return every indexed session-relative occurrence path of one artifact.

        An artifact id denotes the immutable artifact, not one location: the
        same artifact may occur in several session job directories. The
        returned list is sorted and may be empty.
        """
        self._ensure_fresh()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT DISTINCT path FROM occurrences WHERE artifact_id = ?"
                " ORDER BY path",
                (artifact_id,),
            ).fetchall()
        finally:
            connection.close()
        return [str(row[0]) for row in rows]

    def location_issues(self) -> list[dict[str, str]]:
        """List artifact locations the last scan could not index.

        Each entry carries the session-relative ``path``, a ``kind``
        (``unsupported`` schema, ``corrupt`` manifest, ``conflict``
        between occurrences of one artifact identity, or ``annotation``
        for a corrupt/foreign annotation document) and a human-readable
        ``message``. The list is sorted by ``(path, kind)``.
        """
        self._ensure_fresh()
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT path, kind, message FROM location_issues ORDER BY path, kind"
            ).fetchall()
        finally:
            connection.close()
        return [
            {"path": str(path), "kind": str(kind), "message": str(message)}
            for path, kind, message in rows
        ]

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
                " AND (et.tag = ? OR et.tag LIKE ? ESCAPE '\\'))"
            )
            params.extend(
                [
                    query.object_kind,
                    query.tag_descendant_of,
                    _like_escape(query.tag_descendant_of) + "/%",
                ]
            )
        if query.tag_namespace is not None:
            # Namespaces legally contain underscores, which are LIKE
            # wildcards: escape them so ``foo_bar`` never matches ``fooxbar``.
            where.append(
                f"EXISTS (SELECT 1 FROM effective_tags et WHERE {base}"
                " AND et.tag LIKE ? ESCAPE '\\')"
            )
            params.extend([query.object_kind, _like_escape(query.tag_namespace) + ":%"])

    def _connect(self) -> sqlite3.Connection:
        connection: sqlite3.Connection | None = None
        try:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA journal_mode=WAL")
            meta = dict(connection.execute("SELECT key, value FROM meta").fetchall())
            if (
                meta.get("schema") != CATALOG_SCHEMA
                or meta.get("project_id") != self.project.project_id
            ):
                raise _CatalogStaleError
            return connection
        except (sqlite3.DatabaseError, _CatalogStaleError):
            if connection is not None:
                connection.close()
            # Corrupt, outdated or foreign read model: rebuild from disk
            # truth instead of serving empty results.
            self.reindex()
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA journal_mode=WAL")
            return connection

    def _stamp_meta(self, connection: sqlite3.Connection) -> None:
        connection.executemany(
            "INSERT INTO meta (key, value) VALUES (?, ?)",
            [("schema", CATALOG_SCHEMA), ("project_id", self.project.project_id)],
        )
        connection.commit()

    def _fingerprint(self) -> dict[str, Any]:
        """Cheap disk-state probe: counts and newest mtimes of every input.

        Covers the project root (a moved project must refresh its root
        facet), session manifests, artifact manifests, execution records,
        all three annotation document kinds, workflow files and the tag
        policy. Session manifests and execution records are rewritten on
        lifecycle transitions, so a state flip is caught by the mtime and
        makes the next query rebuild the read model.
        """
        sessions = 0
        sessions_mtime_ns = 0
        artifacts = 0
        annotation_docs = 0
        annotations_mtime_ns = 0
        session_root = self.project.session_root
        if session_root.exists():
            for manifest in session_root.rglob("session_manifest.json"):
                if ".trash" not in manifest.parts:
                    sessions += 1
                    sessions_mtime_ns = max(
                        sessions_mtime_ns, manifest.stat().st_mtime_ns
                    )
            for manifest in session_root.rglob("artifact_manifest.json"):
                if ".trash" not in manifest.parts:
                    artifacts += 1
            for name in (SESSION_ANNOTATIONS_FILENAME, ARTIFACT_ANNOTATIONS_FILENAME):
                for document in session_root.rglob(name):
                    if ".trash" not in document.parts:
                        annotation_docs += 1
                        annotations_mtime_ns = max(
                            annotations_mtime_ns, document.stat().st_mtime_ns
                        )
        executions = 0
        executions_mtime_ns = 0
        executions_dir = self.project.root / ".qphase" / "executions"
        if executions_dir.exists():
            for record in executions_dir.glob("*.json"):
                executions += 1
                executions_mtime_ns = max(
                    executions_mtime_ns, record.stat().st_mtime_ns
                )
        project_annotations = (
            self.project.root / ".qphase" / PROJECT_ANNOTATIONS_FILENAME
        )
        if project_annotations.exists():
            annotation_docs += 1
            annotations_mtime_ns = max(
                annotations_mtime_ns, project_annotations.stat().st_mtime_ns
            )
        workflows = 0
        workflows_mtime_ns = 0
        workflow_root = self.project.workflow_root
        if workflow_root.exists():
            for pattern in ("*.yaml", "*.yml"):
                for workflow in workflow_root.rglob(pattern):
                    workflows += 1
                    workflows_mtime_ns = max(
                        workflows_mtime_ns, workflow.stat().st_mtime_ns
                    )
        tag_policy = self.project.defaults_path.parent / TAG_POLICY_FILENAME
        tag_policy_mtime_ns = (
            tag_policy.stat().st_mtime_ns if tag_policy.exists() else 0
        )
        return {
            "project_root": str(self.project.root),
            "sessions": sessions,
            "sessions_mtime_ns": sessions_mtime_ns,
            "artifacts": artifacts,
            "executions": executions,
            "executions_mtime_ns": executions_mtime_ns,
            "annotation_docs": annotation_docs,
            "annotations_mtime_ns": annotations_mtime_ns,
            "workflows": workflows,
            "workflows_mtime_ns": workflows_mtime_ns,
            "tag_policy_mtime_ns": tag_policy_mtime_ns,
        }

    def _ensure_fresh(self) -> None:
        """Reindex once when the stored fingerprint no longer matches disk."""
        if not self.path.exists():
            return  # _connect rebuilds from scratch on first use
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT value FROM meta WHERE key = 'fingerprint'"
            ).fetchone()
        finally:
            connection.close()
        current = json.dumps(self._fingerprint(), sort_keys=True)
        if row is None or row[0] != current:
            self.reindex()


class _CatalogStaleError(Exception):
    """The on-disk catalog predates the current read-model schema."""


#: Reindexing replaces the database file, so concurrent rebuilds of the
#: same project (e.g. scheduler thread plus execution service) must be
#: serialized per catalog path in-process; cross-process writers are
#: serialized by the sibling lock file (see ``core.locking.file_lock``).
_CATALOG_LOCKS: dict[Path, threading.RLock] = {}
_CATALOG_LOCKS_GUARD = threading.Lock()


def _catalog_lock(path: Path) -> threading.RLock:
    with _CATALOG_LOCKS_GUARD:
        return _CATALOG_LOCKS.setdefault(path, threading.RLock())


#: One tag level in the inheritance chain: (tag, assignment_id,
#: policy_revision, frozen_rule) quadruples per assignment.
TagLevel = tuple[
    str,
    list[tuple[str, str | None, str | None, FrozenNamespaceRule | None]],
    bool,
]

#: Declared-tag pairs as read from snapshots: (tag, assignment_id, frozen_rule).
DeclaredPairs = list[tuple[str, str | None, FrozenNamespaceRule | None]]


def compute_effective_tags(
    levels: list[TagLevel],
    policy: TagPolicy | None,
    target_kind: ObjectKind,
) -> list[EffectiveTag]:
    """Merge far-to-near tag levels into effective tags with provenance.

    ``levels`` is an ordered list of ``(source, [(tag, assignment_id,
    policy_revision, frozen_rule)], is_self)`` triples. Each assignment
    carries the revision of the policy that validated it (frozen at compile
    or write time) plus the minimal namespace rule frozen alongside it;
    ``None`` marks assignments that predate provenance/rule tracking, which
    fall back to the policy current at read time. Tags from levels other
    than the object's own are marked inherited and skipped when their
    namespace disables inheritance or does not apply to ``target_kind``;
    cardinality-one namespaces shadow all farther assignments. The object's
    own level is never filtered: it was validated at write time.
    """
    effective: list[EffectiveTag] = []
    for source, items, is_self in levels:
        for tag, assignment_id, policy_revision, frozen_rule in items:
            namespace = tag.split(":", 1)[0]
            rule = policy.namespaces.get(namespace) if policy is not None else None
            inherit = (
                frozen_rule.inherit
                if frozen_rule is not None
                else (rule.inherit if rule is not None else True)
            )
            cardinality = (
                frozen_rule.cardinality
                if frozen_rule is not None
                else (rule.cardinality if rule is not None else "many")
            )
            if not is_self:
                if not inherit:
                    continue
                if frozen_rule is not None:
                    if frozen_rule.objects and target_kind not in frozen_rule.objects:
                        continue
                elif policy is not None and not policy.tag_applies_to(tag, target_kind):
                    continue
            if cardinality == "one":
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


def _job_facets(
    job_payload: Mapping[str, Any],
) -> tuple[str | None, str | None, list[str]]:
    """Extract (engine, model, plugins) facets from one job payload."""
    engine = job_payload.get("engine")
    engine_name = (
        str(next(iter(engine))) if isinstance(engine, Mapping) and engine else None
    )
    plugins = job_payload.get("plugins")
    model_name: str | None = None
    plugin_names: list[str] = []
    if isinstance(plugins, Mapping):
        for namespace in sorted(plugins):
            entries = plugins[namespace]
            if isinstance(entries, Mapping):
                plugin_names.extend(f"{namespace}:{name}" for name in sorted(entries))
        model_entries = plugins.get("model")
        if isinstance(model_entries, Mapping) and model_entries:
            model_name = str(sorted(model_entries)[0])
    return engine_name, model_name, plugin_names


def _frozen_assignment_pairs(raw_assignments: Any, tags: Any) -> DeclaredPairs:
    """Merge frozen assignment entries with their tag list.

    The ``assignments`` section of a frozen tag snapshot supplies the stable
    ids and the frozen namespace rule; tags listed without an assignment
    entry (older snapshots) keep a ``None`` assignment id and rule.
    """
    pairs: DeclaredPairs = []
    seen: set[str] = set()
    if isinstance(raw_assignments, list):
        for item in raw_assignments:
            if isinstance(item, Mapping) and item.get("tag"):
                tag = str(item["tag"])
                assignment_id = item.get("assignment_id")
                pairs.append(
                    (
                        tag,
                        str(assignment_id) if assignment_id else None,
                        _parse_frozen_rule(item),
                    )
                )
                seen.add(tag)
    if isinstance(tags, list):
        for raw_tag in tags:
            tag = str(raw_tag)
            if tag not in seen:
                pairs.append((tag, None, None))
    return pairs


def _parse_frozen_rule(item: Mapping[str, Any]) -> FrozenNamespaceRule | None:
    """Parse the frozen namespace rule of one snapshot assignment entry."""
    inherit = item.get("inherit")
    if inherit is None:
        return None
    cardinality = item.get("cardinality")
    objects = item.get("objects")
    return FrozenNamespaceRule(
        inherit=bool(inherit),
        cardinality="one" if cardinality == "one" else "many",
        objects=tuple(str(value) for value in objects) if objects else (),
    )


def _assignment_triples(
    assignments: Iterable[TagAssignment],
) -> list[tuple[str, str | None, str | None, FrozenNamespaceRule | None]]:
    """Flatten annotation assignments into ``(tag, id, revision, rule)``."""
    return [
        (
            assignment.tag,
            assignment.id,
            assignment.policy_revision,
            _frozen_rule_of(assignment),
        )
        for assignment in assignments
    ]


def _frozen_rule_of(assignment: TagAssignment) -> FrozenNamespaceRule | None:
    """Rebuild the namespace rule frozen on one annotation assignment."""
    if assignment.inherit is None:
        return None
    return FrozenNamespaceRule(
        inherit=assignment.inherit,
        cardinality=assignment.cardinality or "many",
        objects=tuple(assignment.objects or ()),
    )


def _declared_triples(
    pairs: DeclaredPairs,
    policy_revision: str | None,
) -> list[tuple[str, str | None, str | None, FrozenNamespaceRule | None]]:
    """Attach one frozen policy revision to ``(tag, assignment_id, rule)``."""
    return [
        (tag, assignment_id, policy_revision, rule)
        for tag, assignment_id, rule in pairs
    ]


@dataclass
class _RevisionEntry:
    """One workflow revision under construction during a scan."""

    workflow_id: str
    revision: str
    title: str
    collection: str | None
    relative_path: str | None
    sources: list[str]
    job_rows: list[tuple[Any, ...]]

    @property
    def revision_id(self) -> str:
        return f"{self.workflow_id}@{self.revision}"


@dataclass
class _SessionSnapshot:
    """Declared-tag truth frozen in one session's snapshot files."""

    workflow_tags: DeclaredPairs
    job_tags: dict[str, DeclaredPairs]
    policy_revision: str | None
    workflow_payload: dict[str, Any] | None


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
        self._artifact_facets: dict[str, tuple[tuple[Any, ...], str]] = {}
        self._revisions: dict[str, _RevisionEntry] = {}
        self._project_annotations: ProjectAnnotationDocument | None = None
        self.rows: dict[str, list[tuple[Any, ...]]] = {
            "projects": [],
            "workflows": [],
            "jobs": [],
            "executions": [],
            "sessions": [],
            "artifacts": [],
            "occurrences": [],
            "effective_tags": [],
            "location_issues": [],
        }

    def collect(self) -> dict[str, list[tuple[Any, ...]]]:
        # The project annotation document is read once per scan so the
        # project, workflow, job and execution levels stay consistent.
        self._project_annotations = self._load_project_annotations()
        self._scan_project()
        self._scan_workflows()
        self._scan_executions()
        self._scan_sessions()
        for revision_id in sorted(self._revisions):
            entry = self._revisions[revision_id]
            self.rows["workflows"].append(
                (
                    revision_id,
                    entry.workflow_id,
                    entry.title,
                    entry.collection,
                    entry.relative_path,
                    entry.revision,
                    len(entry.job_rows),
                    json.dumps(entry.sources),
                )
            )
            self.rows["jobs"].extend(entry.job_rows)
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

    def _location_issue(self, path: str, kind: str, message: str) -> None:
        self.rows["location_issues"].append((path, kind, message))

    def _scan_project(self) -> None:
        # The project object is a single addressable row; its annotations
        # (tags, alias, note) live in the project annotation document.
        self.rows["projects"].append(
            (
                self.project.project_id,
                self.project.manifest.name,
                str(self.project.root),
            )
        )
        document = self._project_annotations
        self._tags(
            "project",
            self.project.project_id,
            compute_effective_tags(
                [
                    (
                        "project_annotation",
                        _assignment_triples(
                            document.assignments if document is not None else []
                        ),
                        True,
                    )
                ],
                self.policy,
                "project",
            ),
        )

    def _load_project_annotations(self) -> ProjectAnnotationDocument | None:
        """Typed-load the project annotation document, or report an issue.

        Corruption or a foreign ``project_id`` drops the whole annotation
        layer of the project document and records an ``annotation``
        location issue.
        """
        path = self.project.root / ".qphase" / PROJECT_ANNOTATIONS_FILENAME
        if not path.exists():
            return None
        issue_path = f".qphase/{PROJECT_ANNOTATIONS_FILENAME}"
        try:
            document = ProjectAnnotationDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            self._location_issue(
                issue_path, "annotation", f"invalid annotation document: {exc}"
            )
            return None
        if document.project_id != self.project.project_id:
            self._location_issue(
                issue_path,
                "annotation",
                f"annotation identity mismatch: project_id={document.project_id!r}",
            )
            return None
        return document

    def _object_annotations(self, object_id: str) -> ObjectAnnotations:
        """Return the per-object annotation entry of the project document."""
        document = self._project_annotations
        if document is None:
            return ObjectAnnotations()
        return document.objects.get(object_id, ObjectAnnotations())

    def _register_revision(
        self,
        payload: Mapping[str, Any],
        *,
        source: str,
        relative_path: str | None,
        policy_revision: str | None,
    ) -> str | None:
        """Index one workflow revision; first writer wins for its own tags.

        The current workflow file is scanned first, so a revision's own
        declared tags carry the current policy revision; a revision seen only
        in a frozen session snapshot keeps the snapshot's frozen revision.
        Later sources of the same revision only extend its source list.
        """
        workflow_id = str(payload.get("id") or "")
        if not workflow_id:
            return None
        revision = workflow_revision(payload)
        entry = self._revisions.get(f"{workflow_id}@{revision}")
        if entry is not None:
            if source not in entry.sources:
                entry.sources.append(source)
            return entry.revision_id
        workflow_pairs: list[tuple[str, str | None, FrozenNamespaceRule | None]] = []
        for raw_tag in payload.get("tags") or []:
            tag = _try_canonical_tag(str(raw_tag))
            if tag is not None:
                workflow_pairs.append(
                    (tag, workflow_tag_assignment_id(workflow_id, revision, tag), None)
                )
        job_rows: list[tuple[Any, ...]] = []
        job_tags: dict[str, DeclaredPairs] = {}
        jobs = payload.get("jobs")
        for job in jobs if isinstance(jobs, list) else []:
            if not isinstance(job, Mapping) or not job.get("name"):
                continue
            name = str(job["name"])
            engine, model, plugins = _job_facets(job)
            job_rows.append(
                (
                    f"{workflow_id}@{revision}:{name}",
                    f"{workflow_id}@{revision}",
                    workflow_id,
                    name,
                    engine,
                    model,
                    json.dumps(plugins),
                )
            )
            pairs: DeclaredPairs = []
            for raw_tag in job.get("tags") or []:
                tag = _try_canonical_tag(str(raw_tag))
                if tag is not None:
                    pairs.append(
                        (
                            tag,
                            job_tag_assignment_id(workflow_id, revision, name, tag),
                            None,
                        )
                    )
            job_tags[name] = pairs
        self._revisions[f"{workflow_id}@{revision}"] = _RevisionEntry(
            workflow_id=workflow_id,
            revision=revision,
            title=str(payload.get("title") or workflow_id),
            collection=(
                str(payload["collection"])
                if payload.get("collection") is not None
                else None
            ),
            relative_path=relative_path,
            sources=[source],
            job_rows=job_rows,
        )
        revision_id = f"{workflow_id}@{revision}"
        self._tags(
            "workflow",
            revision_id,
            compute_effective_tags(
                [
                    (
                        "workflow_declared",
                        _declared_triples(workflow_pairs, policy_revision),
                        True,
                    ),
                    (
                        "workflow_annotation",
                        _assignment_triples(
                            self._object_annotations(revision_id).assignments
                        ),
                        True,
                    ),
                ],
                self.policy,
                "workflow",
            ),
        )
        for name, pairs in job_tags.items():
            job_id = f"{revision_id}:{name}"
            self._tags(
                "job",
                job_id,
                compute_effective_tags(
                    [
                        (
                            "workflow_declared",
                            _declared_triples(workflow_pairs, policy_revision),
                            False,
                        ),
                        (
                            "job_declared",
                            _declared_triples(pairs, policy_revision),
                            True,
                        ),
                        (
                            "job_annotation",
                            _assignment_triples(
                                self._object_annotations(job_id).assignments
                            ),
                            True,
                        ),
                    ],
                    self.policy,
                    "job",
                ),
            )
        return revision_id

    def _scan_workflows(self) -> None:
        catalog = WorkflowCatalog(self.project)
        for reference in catalog.list():
            try:
                payload: Mapping[str, Any] = load_workflow(reference.path).model_dump(
                    mode="json", by_alias=True
                )
            except Exception:  # noqa: BLE001 - index facets even when invalid
                payload = load_yaml(reference.path)
            if not isinstance(payload, Mapping):
                continue
            self._register_revision(
                payload,
                source=f"file:{reference.relative_path}",
                relative_path=reference.relative_path,
                policy_revision=self.policy_revision,
            )

    def _scan_executions(self) -> None:
        for payload in self.store.load_executions():
            workflow = payload.get("workflow") or {}
            workflow_id = str(workflow.get("id", ""))
            tags = [str(tag) for tag in payload.get("submission_tags", [])]
            tag_revision = payload.get("tag_policy_revision")
            compiled = payload.get("compiled_workflow")
            revision_id: str | None = None
            if isinstance(compiled, Mapping) and isinstance(
                compiled.get("workflow"), Mapping
            ):
                # Declared-tag provenance comes from the compiled tag
                # snapshot frozen at compile time, not from the submission
                # tag policy revision.
                frozen = compiled.get("tag_snapshot")
                frozen_revision = (
                    frozen.get("policy_revision")
                    if isinstance(frozen, Mapping)
                    else None
                )
                revision_id = self._register_revision(
                    compiled["workflow"],
                    source=f"execution:{payload.get('execution_id', '')}",
                    relative_path=None,
                    policy_revision=(
                        str(frozen_revision) if frozen_revision is not None else None
                    ),
                )
            self.rows["executions"].append(
                (
                    str(payload["execution_id"]),
                    workflow_id,
                    revision_id,
                    str(payload.get("state", "unknown")),
                    str(payload.get("submitted_at", "")),
                )
            )
            execution_id = str(payload["execution_id"])
            self._tags(
                "execution",
                execution_id,
                compute_effective_tags(
                    [
                        (
                            "execution_submission",
                            [
                                (
                                    tag,
                                    None,
                                    str(tag_revision) if tag_revision else None,
                                    None,
                                )
                                for tag in tags
                            ],
                            True,
                        ),
                        (
                            "execution_annotation",
                            _assignment_triples(
                                self._object_annotations(execution_id).assignments
                            ),
                            True,
                        ),
                    ],
                    self.policy,
                    "execution",
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
        document = self._load_session_annotations(session_dir, session_id)
        submission_revision = manifest.get("submission_tag_policy_revision")
        submission_tags: list[
            tuple[str, str | None, str | None, FrozenNamespaceRule | None]
        ] = [
            (
                str(tag),
                None,
                str(submission_revision) if submission_revision else None,
                None,
            )
            for tag in manifest.get("submission_tags", [])
        ]
        snapshot = self._snapshot_truth(session_dir)
        if snapshot.workflow_payload is not None:
            session_path = session_dir.relative_to(self.project.session_root).as_posix()
            self._register_revision(
                snapshot.workflow_payload,
                source=f"session:{session_path}",
                relative_path=None,
                policy_revision=snapshot.policy_revision,
            )

        self.rows["sessions"].append(
            (
                session_id,
                manifest.get("workflow_id"),
                str(manifest.get("status", "unknown")),
                manifest.get("start_time"),
                session_dir.relative_to(self.project.session_root).as_posix(),
                document.lifecycle if document is not None else None,
                document.retention if document is not None else None,
                document.alias if document is not None else None,
                document.note if document is not None else None,
                str(submission_revision) if submission_revision else None,
            )
        )
        # The session's declared levels read its own frozen snapshot, so a
        # later policy edit never rewrites this session's provenance.
        session_annotation = _assignment_triples(
            document.assignments if document is not None else []
        )
        inherited_chain: list[TagLevel] = [
            (
                "workflow_declared",
                _declared_triples(snapshot.workflow_tags, snapshot.policy_revision),
                False,
            ),
            ("execution_submission", submission_tags, False),
            ("session_annotation", session_annotation, False),
        ]
        session_chain: list[TagLevel] = [
            (source, items, source == "session_annotation")
            for source, items, _ in inherited_chain
        ]
        self._tags(
            "session",
            session_id,
            compute_effective_tags(session_chain, self.policy, "session"),
        )

        retention = document.retention if document is not None else None
        inherit_retention = (
            self.policy.retention_inherits_to_occurrences
            if self.policy is not None
            else True
        )
        occurrence_annotations = document.occurrences if document is not None else {}
        for manifest_file in sorted(session_dir.rglob("artifact_manifest.json")):
            self._scan_artifact(
                manifest_file.parent,
                session_dir,
                session_id,
                inherited_chain,
                retention if inherit_retention else None,
                snapshot,
                occurrence_annotations,
            )

    def _load_session_annotations(
        self, session_dir: Path, session_id: str
    ) -> SessionAnnotationDocument | None:
        """Typed-load the session annotation document, or report an issue.

        A corrupt document or one whose identity does not match its
        location indexes no annotation layer at all (session facets fall
        back to ``None``) and is recorded as an ``annotation`` location
        issue; everything else about the session still indexes.
        """
        path = session_dir / SESSION_ANNOTATIONS_FILENAME
        if not path.exists():
            return None
        issue_path = (
            f"{session_dir.relative_to(self.project.session_root).as_posix()}"
            f"/{SESSION_ANNOTATIONS_FILENAME}"
        )
        try:
            document = SessionAnnotationDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            self._location_issue(
                issue_path, "annotation", f"invalid annotation document: {exc}"
            )
            return None
        if (
            document.project_id != self.project.project_id
            or document.session_id != session_id
        ):
            self._location_issue(
                issue_path,
                "annotation",
                f"annotation identity mismatch: project_id="
                f"{document.project_id!r}, session_id={document.session_id!r}",
            )
            return None
        return document

    @staticmethod
    def _snapshot_truth(session_dir: Path) -> _SessionSnapshot:
        """Return the declared-tag truth frozen in the session.

        Current sessions store the compiled tag snapshot in
        ``tag_snapshot.yaml`` (canonical tags, frozen policy revision and
        stable assignment ids) next to the ``workflow_snapshot.yaml``
        document. Older sessions have only the raw workflow snapshot; their
        tags are syntax-canonicalized with ``None`` provenance.
        """
        workflow_path = session_dir / "workflow_snapshot.yaml"
        if not workflow_path.exists():
            return _SessionSnapshot([], {}, None, None)
        payload = load_yaml(workflow_path)
        if not isinstance(payload, dict):
            return _SessionSnapshot([], {}, None, None)
        tag_path = session_dir / "tag_snapshot.yaml"
        if tag_path.exists():
            frozen = load_yaml(tag_path)
            if isinstance(frozen, dict):
                assignments = frozen.get("assignments")
                assignments = assignments if isinstance(assignments, Mapping) else {}
                jobs_assignments = assignments.get("jobs")
                jobs_assignments = (
                    jobs_assignments if isinstance(jobs_assignments, Mapping) else {}
                )
                frozen_job_tags: dict[str, DeclaredPairs] = {}
                raw_job_tags = frozen.get("job_tags")
                if isinstance(raw_job_tags, Mapping):
                    for name, tags in raw_job_tags.items():
                        frozen_job_tags[str(name)] = _frozen_assignment_pairs(
                            jobs_assignments.get(name), tags
                        )
                return _SessionSnapshot(
                    _frozen_assignment_pairs(
                        assignments.get("workflow"), frozen.get("canonical_tags")
                    ),
                    frozen_job_tags,
                    (
                        str(frozen["policy_revision"])
                        if frozen.get("policy_revision") is not None
                        else None
                    ),
                    payload,
                )
        workflow_tags: DeclaredPairs = [
            (tag, None, None)
            for raw_tag in payload.get("tags", [])
            if (tag := _try_canonical_tag(str(raw_tag))) is not None
        ]
        job_tags: dict[str, DeclaredPairs] = {}
        for job in payload.get("jobs", []):
            if isinstance(job, Mapping) and job.get("name"):
                job_tags[str(job["name"])] = [
                    (tag, None, None)
                    for raw_tag in job.get("tags", [])
                    if (tag := _try_canonical_tag(str(raw_tag))) is not None
                ]
        return _SessionSnapshot(workflow_tags, job_tags, None, payload)

    def _scan_artifact(
        self,
        artifact_dir: Path,
        session_dir: Path,
        session_id: str,
        inherited_chain: list[TagLevel],
        session_retention: str | None,
        snapshot: _SessionSnapshot,
        occurrence_annotations: Mapping[str, OccurrenceAnnotations],
    ) -> None:
        # Late import: qphase.data imports qphase.core at module level.
        from ..data.errors import ArtifactError, ArtifactUnsupportedError
        from ..data.store import ArtifactManifest

        relative_path = artifact_dir.relative_to(self.project.session_root).as_posix()
        try:
            manifest = ArtifactManifest.read(artifact_dir)
        except ArtifactUnsupportedError as exc:
            self._location_issue(relative_path, "unsupported", str(exc))
            return
        except ArtifactError as exc:
            # A missing/corrupt manifest or invalid adapter payload indexes no
            # rows at all; the damage is surfaced as a location issue instead
            # of a fabricated identity.
            self._location_issue(relative_path, "corrupt", str(exc))
            return
        artifact_id = manifest.artifact_id
        job_name = artifact_dir.relative_to(session_dir).as_posix()
        document = self._load_artifact_annotations(
            artifact_dir, relative_path, artifact_id
        )
        occurrence = ArtifactOccurrence(
            artifact_id=artifact_id, session_id=session_id, job_name=job_name
        )
        # Occurrence annotations are keyed by "job_name:artifact_id" so two
        # occurrences of one artifact inside a session never collide.
        occurrence_annotation = occurrence_annotations.get(f"{job_name}:{artifact_id}")

        product_schemas = {
            entry.name: entry.product_schema.fingerprint()
            for entry in manifest.products
        }
        quantities = sorted(
            {
                variable.quantity
                for entry in manifest.products
                for variable in entry.product_schema.variables
                if variable.quantity
            }
        )
        facets: tuple[Any, ...] = (
            manifest.created_at,
            manifest.bundle.type_id,
            json.dumps(product_schemas, sort_keys=True),
            json.dumps(quantities),
            json.dumps(manifest.parents),
        )
        # The artifact row is identity-scoped: the first occurrence wins and
        # later occurrences of the same artifact add only occurrence rows. A
        # later occurrence whose identity facets disagree is a conflict: the
        # row stays first-seen and the divergent location is reported.
        known = self._artifact_facets.get(artifact_id)
        if known is None:
            self._artifact_facets[artifact_id] = (facets, relative_path)
            self.rows["artifacts"].append(
                (
                    artifact_id,
                    *facets,
                    relative_path,
                    document.lifecycle if document is not None else None,
                    document.retention if document is not None else None,
                )
            )
            self._tags(
                "artifact",
                artifact_id,
                compute_effective_tags(
                    [
                        (
                            "artifact_annotation",
                            _assignment_triples(
                                document.assignments if document is not None else []
                            ),
                            True,
                        )
                    ],
                    self.policy,
                    "artifact",
                ),
            )
        elif known[0] != facets:
            self._location_issue(
                relative_path,
                "conflict",
                f"artifact {artifact_id!r} facets disagree with the first "
                f"occurrence at {known[1]}",
            )
        own_retention = (
            occurrence_annotation.retention
            if occurrence_annotation is not None
            else None
        )
        self.rows["occurrences"].append(
            (
                occurrence.object_id,
                artifact_id,
                session_id,
                job_name,
                relative_path,
                own_retention,
                own_retention or session_retention,
            )
        )
        occurrence_chain: list[TagLevel] = [
            *inherited_chain,
            (
                "job_declared",
                _declared_triples(
                    snapshot.job_tags.get(job_name, []), snapshot.policy_revision
                ),
                False,
            ),
            (
                "occurrence_annotation",
                _assignment_triples(
                    occurrence_annotation.assignments
                    if occurrence_annotation is not None
                    else []
                ),
                True,
            ),
        ]
        self._tags(
            "occurrence",
            occurrence.object_id,
            compute_effective_tags(occurrence_chain, self.policy, "occurrence"),
        )

    def _load_artifact_annotations(
        self, artifact_dir: Path, relative_path: str, artifact_id: str
    ) -> ArtifactAnnotationDocument | None:
        """Typed-load the artifact annotation document, or report an issue.

        Same contract as the session document: corruption or an identity
        mismatch drops the whole annotation layer and records an
        ``annotation`` location issue.
        """
        path = artifact_dir / ARTIFACT_ANNOTATIONS_FILENAME
        if not path.exists():
            return None
        issue_path = f"{relative_path}/{ARTIFACT_ANNOTATIONS_FILENAME}"
        try:
            document = ArtifactAnnotationDocument.model_validate(
                json.loads(path.read_text(encoding="utf-8"))
            )
        except (OSError, ValueError) as exc:
            self._location_issue(
                issue_path, "annotation", f"invalid annotation document: {exc}"
            )
            return None
        if (
            document.project_id != self.project.project_id
            or document.artifact_id != artifact_id
        ):
            self._location_issue(
                issue_path,
                "annotation",
                f"annotation identity mismatch: project_id="
                f"{document.project_id!r}, artifact_id={document.artifact_id!r}",
            )
            return None
        return document
