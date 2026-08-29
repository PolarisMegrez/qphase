"""Project object catalog service: query facade and annotation mutations.

``CatalogService`` is the single entry point used by the CLI and the GUI for
catalog queries and annotation writes. Queries lazily build the read model on
first use; every shared mutation validates tags against the project policy,
applies an optimistic-locked annotation document write and then reindexes the
catalog so the change is immediately visible in queries.

User-private state (private tags, saved views) lives in a per-user
:class:`~qphase.service.private.UserPrivateStore` outside the project. Private
tags never enter the catalog: they are overlaid onto query results at read
time with source ``"user_private"``, they take part in tag predicates with
the same semantics as shared tags, and in cardinality-one namespaces they
shadow the shared assignments of the same namespace (near shadows far, and
private is the nearest level).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import tempfile
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from qphase.core.annotations import (
    ARTIFACT_ANNOTATIONS_FILENAME,
    PROJECT_ANNOTATIONS_FILENAME,
    SESSION_ANNOTATIONS_FILENAME,
    ArtifactAnnotationDocument,
    Lifecycle,
    ObjectAnnotations,
    OccurrenceAnnotations,
    ProjectAnnotationDocument,
    RetentionPolicy,
    SessionAnnotationDocument,
    TagAssignment,
)
from qphase.core.catalog import (
    CATALOG_FILENAME,
    OBJECT_KINDS,
    CatalogQuery,
    CatalogStats,
    EffectiveTag,
    ProjectObjectCatalog,
)
from qphase.core.errors import QPhaseConfigError
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext
from qphase.core.tags import (
    TAG_POLICY_FILENAME,
    FrozenNamespaceRule,
    ObjectKind,
    TagPolicy,
    canonicalize_tag_syntax,
    freeze_namespace_rule,
    load_tag_policy,
    validate_declared_tags,
)
from qphase.core.utils import load_yaml
from qphase.data.errors import ArtifactAmbiguousError, ArtifactNotFoundError

from .models import (
    AmbiguousOccurrenceKey,
    CatalogObject,
    DuplicateArtifact,
    EffectiveTagInfo,
    IdSeparatorViolation,
    InvalidAnnotation,
    InvalidSnapshotTag,
    LegacyMetadataImport,
    MigrationReport,
    OccurrenceKeyConversion,
    SessionSummary,
    TagPolicyInfo,
)
from .private import UserPrivateStore
from .project import ProjectService

__all__ = [
    "CatalogService",
    "VIRTUAL_FOLDERS",
    "parse_facet_filters",
    "parse_range_filters",
]

#: Names of the built-in virtual folders, in display order.
VIRTUAL_FOLDERS = (
    "by-model",
    "paper-evidence",
    "diagnostics",
    "superseded",
    "cold-storage",
)

#: Candidate page size of private-tag queries (batch tag load per page).
_PRIVATE_QUERY_PAGE_SIZE = 1000


class CatalogService:
    """Query the project object catalog and mutate object annotations."""

    def __init__(self, project: ProjectContext, *, home: Path | None = None) -> None:
        self.project = project
        self.catalog = ProjectObjectCatalog(project)
        self.state_store = ProjectStateStore(project)
        self.project_service = ProjectService(project)
        self.private = UserPrivateStore(project.project_id, home=home)

    def reindex(self) -> CatalogStats:
        """Rebuild the catalog read model from disk truth."""
        return self.catalog.reindex()

    def location_issues(self) -> list[dict[str, str]]:
        """List artifact locations the catalog could not index."""
        self._ensure_index()
        return self.catalog.location_issues()

    def query(self, query: CatalogQuery) -> list[CatalogObject]:
        """List objects of one kind with their non-shadowed effective tags.

        Private tags participate in tag predicates with the same semantics as
        shared tags without ever entering the shared catalog: when the query
        carries tag predicates and the user has private tags on this object
        kind, filtering runs over merged shared+private tags in memory.
        """
        self._ensure_index()
        if self._has_tag_predicates(query) and self.private.list_private_tags(
            query.object_kind
        ):
            return self._query_with_private(query)
        private = self._private_annotations(query.object_kind)
        return [
            self._catalog_object(query.object_kind, row, private)
            for row in self.catalog.query(query)
        ]

    @staticmethod
    def _has_tag_predicates(query: CatalogQuery) -> bool:
        return bool(
            query.tags_all
            or query.tags_any
            or query.tags_without
            or query.tag_descendant_of is not None
            or query.tag_namespace is not None
        )

    def _query_with_private(self, query: CatalogQuery) -> list[CatalogObject]:
        """Evaluate tag predicates over merged shared+private tags.

        Candidates stream through the shared query in pages; each page's
        effective tags load in one batch query, so no fixed candidate cap
        and no per-object round trips remain. The caller's offset/limit
        apply after the merged filtering.
        """
        shared_only = replace(
            query,
            tags_all=(),
            tags_any=(),
            tags_without=(),
            tag_descendant_of=None,
            tag_namespace=None,
            limit=_PRIVATE_QUERY_PAGE_SIZE,
            offset=0,
        )
        private_by_object: dict[str, list[str]] = {}
        for object_id, tag in self.private.list_private_tags(query.object_kind):
            private_by_object.setdefault(object_id, []).append(tag)
        policy = load_tag_policy(self.project)
        matched: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            page = self.catalog.query(replace(shared_only, offset=page_offset))
            if not page:
                break
            tags_by_object = self.catalog.effective_tags_for_objects(
                query.object_kind, [str(row["id"]) for row in page]
            )
            for row in page:
                object_id = str(row["id"])
                merged = _merge_private(
                    [
                        _tag_info(tag)
                        for tag in tags_by_object.get(object_id, [])
                    ],
                    [
                        EffectiveTagInfo(tag=tag, source="user_private")
                        for tag in private_by_object.get(object_id, [])
                    ],
                    policy,
                )
                visible = [tag for tag in merged if not tag.shadowed]
                if not query.effective:
                    visible = [tag for tag in visible if not tag.inherited]
                tags = {tag.tag for tag in visible}
                if not all(tag in tags for tag in query.tags_all):
                    continue
                if query.tags_any and not any(tag in tags for tag in query.tags_any):
                    continue
                if any(tag in tags for tag in query.tags_without):
                    continue
                if query.tag_descendant_of is not None and not any(
                    tag == query.tag_descendant_of
                    or tag.startswith(query.tag_descendant_of + "/")
                    for tag in tags
                ):
                    continue
                if query.tag_namespace is not None and not any(
                    tag.split(":", 1)[0] == query.tag_namespace for tag in tags
                ):
                    continue
                matched.append(row)
            if len(page) < _PRIVATE_QUERY_PAGE_SIZE:
                break
            page_offset += len(page)
        private = self._private_annotations(query.object_kind)
        return [
            self._catalog_object(query.object_kind, row, private)
            for row in matched[query.offset : query.offset + query.limit]
        ]

    def effective_tags(
        self, object_kind: str, object_id: str
    ) -> list[EffectiveTagInfo]:
        """Return the effective tags of one object, shared plus private."""
        self._ensure_index()
        shared = [
            _tag_info(tag)
            for tag in self.catalog.effective_tags(object_kind, object_id)
        ]
        private = [
            EffectiveTagInfo(tag=tag, source="user_private")
            for _object_id, tag in self.private.list_private_tags(
                object_kind, object_id
            )
        ]
        return _merge_private(shared, private, load_tag_policy(self.project))

    def tag_policy(self) -> TagPolicyInfo:
        """Return the resolved project tag policy (empty when unconfigured)."""
        path = self.project.defaults_path.parent / TAG_POLICY_FILENAME
        policy = load_tag_policy(self.project)
        return TagPolicyInfo(
            path=str(path) if policy is not None else None,
            revision=policy.revision if policy is not None else None,
            namespaces=(
                {
                    name: rule.model_dump(mode="json")
                    for name, rule in policy.namespaces.items()
                }
                if policy is not None
                else {}
            ),
        )

    def save_view(self, name: str, query: CatalogQuery) -> None:
        """Save or replace one named catalog view in the private store."""
        self.private.save_view(name, asdict(query))

    def list_views(self) -> list[tuple[str, CatalogQuery]]:
        """Return saved views as ``(name, query)`` pairs ordered by name."""
        return [
            (name, CatalogQuery(**payload))
            for name, payload in self.private.list_views()
        ]

    def delete_view(self, name: str) -> None:
        """Delete one saved view from the private store."""
        self.private.delete_view(name)

    def migration_dry_run(self) -> MigrationReport:
        """Preview the Phase 4 history migration; pure read, writes nothing.

        Reports sessions whose legacy ``session_metadata.json`` would seed a
        new annotation document, counts sessions needing no action, and lists
        declared tags in session workflow snapshots that fail validation
        against the current tag policy. On top of that it previews the
        workflow revisions and jobs a rebuild can recover, classifies legacy
        bare-artifact occurrence keys, lists duplicate artifact identities,
        lists existing job names and artifact ids containing the reserved
        ``:`` separator, counts annotation assignments frozen without policy
        provenance, counts sessions whose retention predates the frozen
        inheritance flag, rebuilds the catalog into a temporary database
        outside the project to check reindex parity (which also surfaces the
        annotation documents it could not load), summarizes the user's
        private store, and collects the per-kind object totals and location
        issues the Phase 4 action manifest needs. Never reindexes the
        project's own catalog and never writes into the project directory.
        """
        policy = load_tag_policy(self.project)
        report = MigrationReport()
        root = self.project.session_root
        if root.exists():
            for manifest_path in sorted(root.rglob("session_manifest.json")):
                if ".trash" in manifest_path.parts:
                    continue
                report.sessions_total += 1
                session_dir = manifest_path.parent
                manifest = self.state_store.load_session_manifest(session_dir)
                session_id = str(manifest.get("session_id") or session_dir.name)
                annotations_path = session_dir / SESSION_ANNOTATIONS_FILENAME
                if not annotations_path.exists():
                    # Seed the document exactly the way the migration will, so
                    # the preview matches the future import by construction.
                    seeded = self.project_service.new_session_annotations(
                        session_dir, session_id
                    )
                    if seeded.alias is not None or seeded.note is not None:
                        report.legacy_metadata_imports.append(
                            LegacyMetadataImport(
                                session_id=session_id,
                                path=session_dir.relative_to(root).as_posix(),
                                alias=seeded.alias,
                                note=seeded.note,
                            )
                        )
                    else:
                        report.untouched_sessions += 1
                else:
                    payload = _read_json_document(annotations_path)
                    if (
                        payload is not None
                        and payload.get("retention") is not None
                        and payload.get("retention_inherits_to_occurrences") is None
                    ):
                        report.sessions_missing_retention_inheritance += 1
                report.invalid_snapshot_tags.extend(
                    self._invalid_snapshot_tags(session_dir, session_id, policy)
                )
                report.id_separator_violations.extend(
                    self._id_separator_violations(session_dir, root)
                )
                self._occurrence_key_preview(session_dir, session_id, report)
        self._annotation_provenance(report)
        self._rebuild_parity(report)
        self._private_summary(report)
        return report

    @staticmethod
    def _id_separator_violations(
        session_dir: Path, session_root: Path
    ) -> list[IdSeparatorViolation]:
        """List job names and artifact ids containing the ``:`` separator.

        ``:`` joins catalog object ids (``artifact_id:session_id:job_name``)
        and occurrence annotation keys (``job_name:artifact_id``), so new
        writes reject it; existing data is surfaced here for the migration.
        """
        violations: list[IdSeparatorViolation] = []
        relative = session_dir.relative_to(session_root).as_posix()
        snapshot_path = session_dir / "workflow_snapshot.yaml"
        if snapshot_path.exists():
            payload = load_yaml(snapshot_path)
            if isinstance(payload, dict):
                for job in payload.get("jobs", []):
                    if not isinstance(job, dict) or not job.get("name"):
                        continue
                    name = str(job["name"])
                    if ":" in name:
                        violations.append(
                            IdSeparatorViolation(
                                object_kind="job",
                                value=name,
                                path=f"{relative}/workflow_snapshot.yaml",
                            )
                        )
        for manifest_file in sorted(session_dir.rglob("artifact_manifest.json")):
            artifact_id = _manifest_artifact_id(manifest_file)
            if artifact_id is not None and ":" in artifact_id:
                violations.append(
                    IdSeparatorViolation(
                        object_kind="artifact",
                        value=artifact_id,
                        path=manifest_file.parent.relative_to(
                            session_root
                        ).as_posix(),
                    )
                )
        return violations

    def _occurrence_key_preview(
        self, session_dir: Path, session_id: str, report: MigrationReport
    ) -> None:
        """Classify legacy bare-artifact occurrence keys of one session.

        Keys written before the ``job_name:artifact_id`` convention are bare
        artifact ids. A key converts unambiguously when the artifact occurs
        in exactly one job of the session; otherwise the migration needs a
        human decision and the key is reported as ambiguous.
        """
        path = session_dir / SESSION_ANNOTATIONS_FILENAME
        if not path.exists():
            return
        payload = _read_json_document(path)
        if payload is None:
            return
        occurrences = payload.get("occurrences")
        if not isinstance(occurrences, dict):
            return
        legacy = sorted(str(key) for key in occurrences if ":" not in str(key))
        if not legacy:
            return
        jobs_by_artifact: dict[str, list[str]] = {}
        for manifest_file in sorted(session_dir.rglob("artifact_manifest.json")):
            artifact_id = _manifest_artifact_id(manifest_file)
            if artifact_id is None:
                continue
            job_name = manifest_file.parent.relative_to(session_dir).as_posix()
            jobs_by_artifact.setdefault(artifact_id, []).append(job_name)
        for artifact_id in legacy:
            jobs = jobs_by_artifact.get(artifact_id, [])
            if len(jobs) == 1:
                report.convertible_occurrence_keys.append(
                    OccurrenceKeyConversion(
                        session_id=session_id,
                        old_key=artifact_id,
                        new_key=f"{jobs[0]}:{artifact_id}",
                    )
                )
            else:
                report.ambiguous_occurrence_keys.append(
                    AmbiguousOccurrenceKey(
                        session_id=session_id, old_key=artifact_id, locations=jobs
                    )
                )

    def _annotation_provenance(self, report: MigrationReport) -> None:
        """Count assignments frozen without a policy revision, per scope."""
        counts: dict[str, int] = {}

        def add(scope: str, assignments: Any) -> None:
            if not isinstance(assignments, list):
                return
            missing = sum(
                1
                for item in assignments
                if isinstance(item, dict) and item.get("policy_revision") is None
            )
            if missing:
                counts[scope] = counts.get(scope, 0) + missing

        root = self.project.session_root
        if root.exists():
            for path in sorted(root.rglob(SESSION_ANNOTATIONS_FILENAME)):
                if ".trash" in path.parts:
                    continue
                payload = _read_json_document(path)
                if payload is None:
                    continue
                add("session", payload.get("assignments"))
                occurrences = payload.get("occurrences")
                if isinstance(occurrences, dict):
                    for entry in occurrences.values():
                        if isinstance(entry, dict):
                            add("occurrence", entry.get("assignments"))
            for path in sorted(root.rglob(ARTIFACT_ANNOTATIONS_FILENAME)):
                if ".trash" in path.parts:
                    continue
                payload = _read_json_document(path)
                if payload is not None:
                    add("artifact", payload.get("assignments"))
        payload = _read_json_document(
            self.project.root / ".qphase" / PROJECT_ANNOTATIONS_FILENAME
        )
        if payload is not None:
            add("project", payload.get("assignments"))
            objects = payload.get("objects")
            if isinstance(objects, dict):
                for object_id, entry in objects.items():
                    if isinstance(entry, dict):
                        add(
                            _project_object_scope(str(object_id)),
                            entry.get("assignments"),
                        )
        report.assignments_without_policy_revision = counts

    def _rebuild_parity(self, report: MigrationReport) -> None:
        """Rebuild the catalog into a temp database and compare with on-disk.

        The throwaway rebuild also yields the rebuildable workflow/job
        counts, duplicate artifact identities, per-kind object totals and
        location issues. Parity compares the full row content of every
        catalog table (bidirectional EXCEPT); an unreadable on-disk catalog
        counts as drift.
        """
        with tempfile.TemporaryDirectory(prefix="qphase-migration-") as tmp:
            rebuilt = ProjectObjectCatalog(
                self.project, db_path=Path(tmp) / CATALOG_FILENAME
            )
            stats = rebuilt.reindex()
            report.rebuildable_workflow_revisions = stats.workflows
            report.rebuildable_jobs = stats.jobs
            rebuilt_counts = {
                "projects": stats.projects,
                "workflows": stats.workflows,
                "jobs": stats.jobs,
                "executions": stats.executions,
                "sessions": stats.sessions,
                "artifacts": stats.artifacts,
                "occurrences": stats.occurrences,
                "effective_tags": stats.effective_tags,
            }
            report.object_counts = {
                kind: rebuilt_counts[f"{kind}s"] for kind in OBJECT_KINDS
            }
            issues = rebuilt.location_issues()
            by_kind: dict[str, int] = {}
            for issue in issues:
                by_kind[issue["kind"]] = by_kind.get(issue["kind"], 0) + 1
            report.location_issues_by_kind = by_kind
            report.invalid_annotations = [
                InvalidAnnotation(path=issue["path"], error=issue["message"])
                for issue in issues
                if issue["kind"] == "annotation"
            ]
            report.duplicate_artifacts = _duplicate_artifacts(rebuilt.path, issues)
            current = self.catalog.path
            if not current.exists():
                report.catalog_drift = None
                return
            try:
                drift = _catalog_drift(current, rebuilt.path, Path(tmp))
            except sqlite3.DatabaseError:
                report.catalog_drift = True
                return
            report.catalog_drift_tables = drift
            report.catalog_drift = bool(drift)

    def _private_summary(self, report: MigrationReport) -> None:
        """Record informational counts of the current user's private store."""
        report.private_tag_count = sum(
            len(self.private.list_private_tags(kind)) for kind in OBJECT_KINDS
        )
        report.saved_view_count = len(self.private.list_views())
        report.private_annotation_count = sum(
            len(self.private.list_private_annotations(kind)) for kind in OBJECT_KINDS
        )

    @staticmethod
    def _invalid_snapshot_tags(
        session_dir: Path, session_id: str, policy: TagPolicy | None
    ) -> list[InvalidSnapshotTag]:
        """List declared snapshot tags failing the current validation rules."""
        path = session_dir / "workflow_snapshot.yaml"
        if not path.exists():
            return []
        payload = load_yaml(path)
        if not isinstance(payload, dict):
            return []
        declared: list[tuple[str, str, ObjectKind]] = [
            (str(tag), "workflow", "workflow") for tag in payload.get("tags", [])
        ]
        for job in payload.get("jobs", []):
            if isinstance(job, dict) and job.get("name"):
                declared.extend(
                    (str(tag), str(job["name"]), "job") for tag in job.get("tags", [])
                )
        invalid = []
        for tag, source, kind in declared:
            try:
                validate_declared_tags([tag], kind, policy)
            except QPhaseConfigError as exc:
                invalid.append(
                    InvalidSnapshotTag(
                        session_id=session_id, tag=tag, source=source, error=str(exc)
                    )
                )
        return invalid

    def virtual_folders(self) -> list[tuple[str, int]]:
        """Return ``(name, object count)`` for every built-in folder."""
        return [(name, len(self.virtual_folder(name))) for name in VIRTUAL_FOLDERS]

    def virtual_folder(self, name: str) -> list[CatalogObject]:
        """Return the session objects of one built-in virtual folder."""
        if name == "by-model":
            # Sessions whose workflow revision declares any model plugin;
            # filter by a concrete model with the ``model`` query filter.
            return self.query(CatalogQuery(object_kind="session", has_model=True))
        if name == "paper-evidence":
            return self.query(
                CatalogQuery(object_kind="session", retention="evidence")
            ) + self.query(CatalogQuery(object_kind="session", retention="pinned"))
        if name == "diagnostics":
            return self.query(
                CatalogQuery(object_kind="session", tags_all=("task:diagnostics",))
            )
        if name == "superseded":
            return self.query(
                CatalogQuery(object_kind="session", lifecycle="superseded")
            )
        if name == "cold-storage":
            return self.query(CatalogQuery(object_kind="session", lifecycle="archived"))
        raise KeyError(f"unknown virtual folder: {name!r}")

    def tag_session(
        self,
        session_id: str,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> SessionSummary:
        """Add/remove session tag assignments; returns the session summary."""
        added, policy_revision, rules = self._validate_tags(add, "session")
        removed = _canonical_set(remove)
        root = self.project_service.session_dir(session_id)
        if private:
            self._edit_private_tags("session", session_id, added, removed)
            return self.project_service.get_session(session_id)
        document, expected = self._session_document(root, session_id)
        _apply_tag_edits(document.assignments, added, removed, policy_revision, rules)
        self._save_session_document(root, document, expected)
        return self.project_service.get_session(session_id)

    def set_session_lifecycle(
        self, session_id: str, lifecycle: Lifecycle | None
    ) -> SessionSummary:
        """Set or clear the session lifecycle."""
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        document.lifecycle = lifecycle
        self._save_session_document(root, document, expected)
        return self.project_service.get_session(session_id)

    def set_session_retention(
        self, session_id: str, retention: RetentionPolicy | None
    ) -> SessionSummary:
        """Set or clear the session retention policy.

        Setting a retention also freezes whether it inherits to occurrences
        (from the current tag policy, defaulting to ``True``), so a later
        policy edit never rewrites historical sessions. Clearing the
        retention clears the frozen flag as well.
        """
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        document.retention = retention
        if retention is None:
            document.retention_inherits_to_occurrences = None
        else:
            policy = load_tag_policy(self.project)
            document.retention_inherits_to_occurrences = (
                policy.retention_inherits_to_occurrences
                if policy is not None
                else True
            )
        self._save_session_document(root, document, expected)
        return self.project_service.get_session(session_id)

    def tag_artifact(
        self,
        artifact_id: str,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove artifact tag assignments; returns its effective tags."""
        added, policy_revision, rules = self._validate_tags(add, "artifact")
        removed = _canonical_set(remove)
        if private:
            self._require_artifact(artifact_id)
            self._edit_private_tags("artifact", artifact_id, added, removed)
            return self.effective_tags("artifact", artifact_id)
        artifact_dir = self._artifact_dir(artifact_id)
        document, expected = self._artifact_document(artifact_dir, artifact_id)
        _apply_tag_edits(document.assignments, added, removed, policy_revision, rules)
        self._save_artifact_document(artifact_dir, document, expected)
        return self.effective_tags("artifact", artifact_id)

    def set_artifact_lifecycle(
        self, artifact_id: str, lifecycle: Lifecycle | None
    ) -> CatalogObject:
        """Set or clear the artifact lifecycle."""
        artifact_dir = self._artifact_dir(artifact_id)
        document, expected = self._artifact_document(artifact_dir, artifact_id)
        document.lifecycle = lifecycle
        self._save_artifact_document(artifact_dir, document, expected)
        return self._single_object("artifact", artifact_id)

    def set_artifact_retention(
        self, artifact_id: str, retention: RetentionPolicy | None
    ) -> CatalogObject:
        """Set or clear the artifact retention policy."""
        artifact_dir = self._artifact_dir(artifact_id)
        document, expected = self._artifact_document(artifact_dir, artifact_id)
        document.retention = retention
        self._save_artifact_document(artifact_dir, document, expected)
        return self._single_object("artifact", artifact_id)

    def tag_occurrence(
        self,
        session_id: str,
        artifact_id: str,
        *,
        job_name: str | None = None,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove occurrence tag assignments; returns its effective tags.

        ``job_name`` disambiguates artifacts occurring in several jobs of the
        session; when omitted, exactly one occurrence must exist.
        """
        row = self._resolve_occurrence(session_id, artifact_id, job_name)
        occurrence_id = str(row["id"])
        added, policy_revision, rules = self._validate_tags(add, "occurrence")
        removed = _canonical_set(remove)
        if private:
            self._edit_private_tags("occurrence", occurrence_id, added, removed)
            return self.effective_tags("occurrence", occurrence_id)
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        key = f"{row['job_name']}:{artifact_id}"
        occurrence = document.occurrences.setdefault(key, OccurrenceAnnotations())
        _apply_tag_edits(occurrence.assignments, added, removed, policy_revision, rules)
        self._save_session_document(root, document, expected)
        return self.effective_tags("occurrence", occurrence_id)

    def set_occurrence_retention(
        self,
        session_id: str,
        artifact_id: str,
        retention: RetentionPolicy | None,
        *,
        job_name: str | None = None,
    ) -> CatalogObject:
        """Set or clear one occurrence's retention policy."""
        row = self._resolve_occurrence(session_id, artifact_id, job_name)
        occurrence_id = str(row["id"])
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        key = f"{row['job_name']}:{artifact_id}"
        occurrence = document.occurrences.setdefault(key, OccurrenceAnnotations())
        occurrence.retention = retention
        self._save_session_document(root, document, expected)
        return self._single_object("occurrence", occurrence_id)

    def tag_project(
        self,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove project tag assignments; returns its effective tags."""
        added, policy_revision, rules = self._validate_tags(add, "project")
        removed = _canonical_set(remove)
        object_id = self.project.project_id
        if private:
            self._edit_private_tags("project", object_id, added, removed)
            return self.effective_tags("project", object_id)
        document, expected = self._project_document()
        _apply_tag_edits(document.assignments, added, removed, policy_revision, rules)
        self._save_project_document(document, expected)
        return self.effective_tags("project", object_id)

    def set_project_alias(self, alias: str | None) -> CatalogObject:
        """Set or clear the shared project alias."""
        document, expected = self._project_document()
        document.alias = alias
        self._save_project_document(document, expected)
        return self._single_object("project", self.project.project_id)

    def set_project_note(self, note: str | None) -> CatalogObject:
        """Set or clear the shared project note."""
        document, expected = self._project_document()
        document.note = note
        self._save_project_document(document, expected)
        return self._single_object("project", self.project.project_id)

    def project_annotations(self) -> ProjectAnnotationDocument:
        """Return the project annotation document (empty default when absent)."""
        return self._project_document()[0]

    def tag_workflow(
        self,
        revision_id: str,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove tag assignments on one workflow revision.

        ``revision_id`` is the catalog workflow object id
        (``workflow_id@revision``); the annotation lives in the project
        annotation document and never flows down to jobs or sessions.
        """
        self._require_object("workflow", revision_id)
        return self._tag_project_object(
            "workflow", revision_id, add=add, remove=remove, private=private
        )

    def tag_job(
        self,
        job_id: str,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove tag assignments on one job (``workflow_id@revision:name``)."""
        self._require_object("job", job_id)
        return self._tag_project_object(
            "job", job_id, add=add, remove=remove, private=private
        )

    def tag_execution(
        self,
        execution_id: str,
        *,
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove tag assignments on one execution.

        Execution submission tags (``submission_tags`` on the execution
        record) are the frozen submit-time layer and can only change while
        the execution is queued; these annotations are the after-the-fact
        organization layer and stay editable for the execution's lifetime.
        The two layers coexist as separate effective-tag levels.
        """
        self._require_object("execution", execution_id)
        return self._tag_project_object(
            "execution", execution_id, add=add, remove=remove, private=private
        )

    def _tag_project_object(
        self,
        object_kind: ObjectKind,
        object_id: str,
        *,
        add: list[str] | tuple[str, ...],
        remove: list[str] | tuple[str, ...],
        private: bool,
    ) -> list[EffectiveTagInfo]:
        """Edit the tag assignments of one project-document-scoped object."""
        added, policy_revision, rules = self._validate_tags(add, object_kind)
        removed = _canonical_set(remove)
        if private:
            self._edit_private_tags(object_kind, object_id, added, removed)
            return self.effective_tags(object_kind, object_id)
        document, expected = self._project_document()
        entry = document.objects.setdefault(object_id, ObjectAnnotations())
        _apply_tag_edits(entry.assignments, added, removed, policy_revision, rules)
        self._save_project_document(document, expected)
        return self.effective_tags(object_kind, object_id)

    def _require_object(self, object_kind: str, object_id: str) -> None:
        self._ensure_index()
        rows = self.catalog.query(
            CatalogQuery(object_kind=object_kind, facets={"id": object_id})
        )
        if not rows:
            raise ValueError(f"unknown {object_kind}: {object_id}")

    def promote_tag(
        self, object_kind: str, object_id: str, tag: str
    ) -> list[EffectiveTagInfo]:
        """Move one private tag into the shared annotation document."""
        canonical = canonicalize_tag_syntax(tag)
        if object_kind == "session":
            self.tag_session(object_id, add=[canonical])
        elif object_kind == "artifact":
            self.tag_artifact(object_id, add=[canonical])
        elif object_kind == "occurrence":
            artifact_id, session_id, job_name = object_id.split(":", 2)
            self.tag_occurrence(
                session_id, artifact_id, job_name=job_name, add=[canonical]
            )
        elif object_kind == "project":
            self.tag_project(add=[canonical])
        elif object_kind == "workflow":
            self.tag_workflow(object_id, add=[canonical])
        elif object_kind == "job":
            self.tag_job(object_id, add=[canonical])
        elif object_kind == "execution":
            self.tag_execution(object_id, add=[canonical])
        else:
            raise ValueError(f"cannot promote tags on {object_kind!r} objects")
        self.private.remove_private_tag(object_kind, object_id, canonical)
        return self.effective_tags(object_kind, object_id)

    def _edit_private_tags(
        self,
        object_kind: str,
        object_id: str,
        added: list[str],
        removed: set[str],
    ) -> None:
        for tag in removed:
            self.private.remove_private_tag(object_kind, object_id, tag)
        for tag in added:
            self.private.add_private_tag(object_kind, object_id, tag)

    def _ensure_index(self) -> None:
        if not self.catalog.path.exists():
            self.catalog.reindex()

    def _validate_tags(
        self, values: list[str] | tuple[str, ...], object_kind: ObjectKind
    ) -> tuple[list[str], str | None, dict[str, FrozenNamespaceRule]]:
        """Validate tags against the current policy; freeze its provenance.

        Returns the canonical tags, the policy revision, and the minimal
        namespace rule frozen per tag (default rules when no policy governs
        the namespace, so history stays stable if a policy appears later).
        """
        policy = load_tag_policy(self.project)
        tags = validate_declared_tags(list(values), object_kind, policy)
        rules = {tag: freeze_namespace_rule(policy, tag) for tag in tags}
        return tags, (policy.revision if policy is not None else None), rules

    def _session_document(
        self, root: Path, session_id: str
    ) -> tuple[SessionAnnotationDocument, int | None]:
        current = self.state_store.load_session_annotations(root)
        if current is not None:
            document = SessionAnnotationDocument.model_validate(current)
            return document, document.revision
        # Creation goes through ProjectService so the legacy alias/note import
        # stays in exactly one place.
        return self.project_service.new_session_annotations(root, session_id), None

    def _artifact_document(
        self, artifact_dir: Path, artifact_id: str
    ) -> tuple[ArtifactAnnotationDocument, int | None]:
        current = self.state_store.load_artifact_annotations(artifact_dir)
        if current is not None:
            document = ArtifactAnnotationDocument.model_validate(current)
            return document, document.revision
        return (
            ArtifactAnnotationDocument(
                project_id=self.project.project_id, artifact_id=artifact_id
            ),
            None,
        )

    def _project_document(self) -> tuple[ProjectAnnotationDocument, int | None]:
        current = self.state_store.load_project_annotations()
        if current is not None:
            document = ProjectAnnotationDocument.model_validate(current)
            return document, document.revision
        return (
            ProjectAnnotationDocument(project_id=self.project.project_id),
            None,
        )

    def _save_project_document(
        self,
        document: ProjectAnnotationDocument,
        expected_revision: int | None,
    ) -> None:
        self.state_store.save_project_annotations(
            document.model_dump(mode="json", by_alias=True),
            expected_revision=expected_revision,
        )
        self.catalog.reindex()

    def _save_session_document(
        self,
        root: Path,
        document: SessionAnnotationDocument,
        expected_revision: int | None,
    ) -> None:
        self.state_store.save_session_annotations(
            root,
            document.model_dump(mode="json", by_alias=True),
            expected_revision=expected_revision,
        )
        self.catalog.reindex()

    def _save_artifact_document(
        self,
        artifact_dir: Path,
        document: ArtifactAnnotationDocument,
        expected_revision: int | None,
    ) -> None:
        self.state_store.save_artifact_annotations(
            artifact_dir,
            document.model_dump(mode="json", by_alias=True),
            expected_revision=expected_revision,
        )
        self.catalog.reindex()

    def _require_artifact(self, artifact_id: str) -> None:
        self._ensure_index()
        if not self.catalog.locate_artifact_paths(artifact_id):
            raise ArtifactNotFoundError(f"unknown artifact: {artifact_id}")

    def _artifact_dir(self, artifact_id: str) -> Path:
        """Return the single occurrence location of one artifact identity.

        An artifact id denotes the immutable artifact, not a location. When
        several occurrences exist the identity is ambiguous: artifact-level
        annotations need one canonical location, so callers must annotate the
        occurrence (session/job) instead.
        """
        self._ensure_index()
        paths = self.catalog.locate_artifact_paths(artifact_id)
        if not paths:
            raise ArtifactNotFoundError(f"unknown artifact: {artifact_id}")
        if len(paths) > 1:
            raise ArtifactAmbiguousError(
                f"artifact {artifact_id!r} occurs in {len(paths)} locations "
                f"{paths}; annotate a specific occurrence (session/job) "
                "instead of the artifact identity"
            )
        return (self.project.session_root / paths[0]).resolve()

    def _resolve_occurrence(
        self, session_id: str, artifact_id: str, job_name: str | None = None
    ) -> dict[str, Any]:
        """Resolve one occurrence row, never silently picking the first."""
        self._ensure_index()
        facets = {"session_id": session_id, "artifact_id": artifact_id}
        if job_name is not None:
            facets["job_name"] = job_name
        rows = self.catalog.query(
            CatalogQuery(object_kind="occurrence", facets=facets)
        )
        if not rows:
            raise ArtifactNotFoundError(
                f"unknown occurrence: artifact {artifact_id} in session {session_id}"
            )
        if len(rows) > 1:
            candidates = sorted(str(row["job_name"]) for row in rows)
            raise ValueError(
                f"ambiguous occurrence: artifact {artifact_id} occurs in jobs "
                f"{candidates} of session {session_id}; pass job_name"
            )
        return rows[0]

    def _single_object(self, object_kind: str, object_id: str) -> CatalogObject:
        rows = self.catalog.query(
            CatalogQuery(object_kind=object_kind, facets={"id": object_id})
        )
        return self._catalog_object(
            object_kind, rows[0], self._private_annotations(object_kind)
        )

    def _private_annotations(
        self, object_kind: str
    ) -> dict[str, tuple[str | None, str | None]]:
        """Prefetch the private alias/note overlay of one object kind."""
        return {
            object_id: (alias, note)
            for object_id, alias, note in self.private.list_private_annotations(
                object_kind
            )
        }

    def _catalog_object(
        self,
        object_kind: str,
        row: dict[str, Any],
        private_annotations: dict[str, tuple[str | None, str | None]] | None = None,
    ) -> CatalogObject:
        object_id = str(row["id"])
        alias, note = (
            private_annotations.get(object_id, (None, None))
            if private_annotations is not None
            else self.private.get_private_annotation(object_kind, object_id)
        )
        return CatalogObject(
            kind=object_kind,
            id=object_id,
            facets=row,
            effective_tags=[
                tag
                for tag in self.effective_tags(object_kind, object_id)
                if not tag.shadowed
            ],
            private_alias=alias,
            private_note=note,
        )


#: Tables compared between the on-disk catalog and a throwaway rebuild
#: (every content table; the ``meta`` bookkeeping table is excluded).
_PARITY_TABLES = (
    "projects",
    "workflows",
    "jobs",
    "executions",
    "sessions",
    "artifacts",
    "occurrences",
    "effective_tags",
    "location_issues",
    "job_plugins",
    "artifact_quantities",
)


def _read_json_document(path: Path) -> dict[str, Any] | None:
    """Read one annotation document as raw JSON, tolerating corruption."""
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def _manifest_artifact_id(manifest_path: Path) -> str | None:
    """Read the artifact id of one manifest, tolerating corrupt files."""
    payload = _read_json_document(manifest_path)
    if payload is None or payload.get("artifact_id") is None:
        return None
    return str(payload["artifact_id"])


def _project_object_scope(object_id: str) -> str:
    """Classify one project-document object key by id shape."""
    if ":" in object_id:
        return "job"
    if "@" in object_id:
        return "workflow"
    return "execution"


def _duplicate_artifacts(
    catalog_path: Path, issues: list[dict[str, str]]
) -> list[DuplicateArtifact]:
    """List artifact identities materialized at more than one location."""
    connection = sqlite3.connect(catalog_path)
    try:
        artifact_ids = [
            str(row[0])
            for row in connection.execute(
                "SELECT artifact_id FROM occurrences GROUP BY artifact_id"
                " HAVING COUNT(DISTINCT path) > 1 ORDER BY artifact_id"
            ).fetchall()
        ]
        duplicates = []
        for artifact_id in artifact_ids:
            locations = [
                str(row[0])
                for row in connection.execute(
                    "SELECT DISTINCT path FROM occurrences"
                    " WHERE artifact_id = ? ORDER BY path",
                    (artifact_id,),
                ).fetchall()
            ]
            conflict = any(
                issue["kind"] == "conflict" and f"'{artifact_id}'" in issue["message"]
                for issue in issues
            )
            duplicates.append(
                DuplicateArtifact(
                    artifact_id=artifact_id, locations=locations, conflict=conflict
                )
            )
    finally:
        connection.close()
    return duplicates


def _catalog_drift(
    catalog_path: Path, rebuilt_path: Path, work_dir: Path
) -> dict[str, int]:
    """Diff two catalog databases table by table as multisets of full rows.

    Returns per-table differing row counts (multiset symmetric difference,
    so a duplicated row counts as drift even when the row set is identical);
    an empty dict means no drift. The on-disk catalog is WAL-mode, so even
    opening it could create ``-shm`` sidecar files next to it: it is copied
    (with WAL/SHM sidecars) into ``work_dir`` first, keeping the project
    directory untouched.
    """
    current = work_dir / "current_catalog.sqlite"
    shutil.copy2(catalog_path, current)
    for suffix in ("-wal", "-shm"):
        sidecar = catalog_path.with_name(catalog_path.name + suffix)
        if sidecar.exists():
            shutil.copy2(sidecar, current.with_name(current.name + suffix))
    connection = sqlite3.connect(":memory:")
    try:
        connection.execute("ATTACH DATABASE ? AS current", (str(current),))
        connection.execute("ATTACH DATABASE ? AS rebuilt", (str(rebuilt_path),))
        drift: dict[str, int] = {}
        for table in _PARITY_TABLES:
            # Full rows are str/int/None tuples, safe to count as multisets.
            current_rows = Counter(
                connection.execute(
                    f"SELECT * FROM current.{table}"  # noqa: S608
                ).fetchall()
            )
            rebuilt_rows = Counter(
                connection.execute(
                    f"SELECT * FROM rebuilt.{table}"  # noqa: S608
                ).fetchall()
            )
            differing = sum((current_rows - rebuilt_rows).values()) + sum(
                (rebuilt_rows - current_rows).values()
            )
            if differing:
                drift[table] = differing
        return drift
    finally:
        connection.close()


def _tag_info(tag: EffectiveTag) -> EffectiveTagInfo:
    return EffectiveTagInfo(
        tag=tag.tag,
        source=tag.source,
        assignment_id=tag.assignment_id,
        policy_revision=tag.policy_revision,
        inherited=tag.inherited,
        shadowed=tag.shadowed,
    )


def _merge_private(
    shared: list[EffectiveTagInfo],
    private: list[EffectiveTagInfo],
    policy: TagPolicy | None,
) -> list[EffectiveTagInfo]:
    """Overlay private tags: cardinality-one namespaces shadow shared tags."""
    for tag in private:
        namespace = tag.tag.split(":", 1)[0]
        rule = policy.namespaces.get(namespace) if policy is not None else None
        if rule is not None and rule.cardinality == "one":
            for other in shared:
                if other.tag.split(":", 1)[0] == namespace:
                    other.shadowed = True
    return [*shared, *private]


def _canonical_set(values: list[str] | tuple[str, ...]) -> set[str]:
    return {canonicalize_tag_syntax(value) for value in values}


def parse_facet_filters(values: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Parse repeated ``key=value`` facet filter options into a mapping."""
    facets: dict[str, str] = {}
    for item in values:
        key, separator, value = item.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"facet filter must be key=value, got {item!r}")
        facets[key.strip()] = value
    return facets


def parse_range_filters(
    values: list[str] | tuple[str, ...],
) -> dict[str, tuple[str | None, str | None]]:
    """Parse repeated ``key=low..high`` range filters; either bound may be empty."""
    ranges: dict[str, tuple[str | None, str | None]] = {}
    for item in values:
        key, separator, body = item.partition("=")
        low, dots, high = body.partition("..")
        if not separator or not key.strip() or not dots:
            raise ValueError(f"range filter must be key=low..high, got {item!r}")
        ranges[key.strip()] = (low or None, high or None)
    return ranges


def _apply_tag_edits(
    assignments: list[TagAssignment],
    added: list[str],
    removed: set[str],
    policy_revision: str | None = None,
    rules: dict[str, FrozenNamespaceRule] | None = None,
) -> None:
    """Apply immutable-assignment edits in place on an assignment list.

    New assignments freeze the revision of the policy that validated them
    plus the minimal namespace rule governing their resolution.
    """
    kept = [assignment for assignment in assignments if assignment.tag not in removed]
    existing = {assignment.tag for assignment in kept}
    kept.extend(
        _new_assignment(tag, policy_revision, (rules or {}).get(tag))
        for tag in added
        if tag not in existing
    )
    assignments[:] = kept


def _new_assignment(
    tag: str,
    policy_revision: str | None,
    rule: FrozenNamespaceRule | None,
) -> TagAssignment:
    return TagAssignment(
        tag=tag,
        policy_revision=policy_revision,
        inherit=rule.inherit if rule is not None else None,
        cardinality=rule.cardinality if rule is not None else None,
        objects=rule.objects if rule is not None else None,
    )
