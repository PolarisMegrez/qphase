"""Project object catalog service: query facade and annotation mutations.

``CatalogService`` is the single entry point used by the CLI and the GUI for
catalog queries and annotation writes. Queries lazily build the read model on
first use; every shared mutation validates tags against the project policy,
applies an optimistic-locked annotation document write and then reindexes the
catalog so the change is immediately visible in queries.

User-private state (private tags, saved views) lives in a per-user
:class:`~qphase.service.private.UserPrivateStore` outside the project. Private
tags never enter the catalog: they are overlaid onto query results at read
time with source ``"user_private"``, and in cardinality-one namespaces they
shadow the shared assignments of the same namespace (near shadows far, and
private is the nearest level).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from qphase.core.annotations import (
    ArtifactAnnotationDocument,
    Lifecycle,
    OccurrenceAnnotations,
    RetentionPolicy,
    SessionAnnotationDocument,
    TagAssignment,
)
from qphase.core.catalog import (
    CatalogQuery,
    CatalogStats,
    EffectiveTag,
    ProjectObjectCatalog,
)
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext
from qphase.core.tags import (
    TAG_POLICY_FILENAME,
    ObjectKind,
    TagPolicy,
    canonicalize_tag_syntax,
    load_tag_policy,
    validate_declared_tags,
)
from qphase.data.errors import ArtifactNotFoundError

from .models import (
    CatalogObject,
    EffectiveTagInfo,
    SessionSummary,
    TagPolicyInfo,
)
from .private import UserPrivateStore
from .project import ProjectService

__all__ = ["CatalogService", "VIRTUAL_FOLDERS"]

#: Names of the built-in virtual folders, in display order.
VIRTUAL_FOLDERS = (
    "by-model",
    "paper-evidence",
    "diagnostics",
    "superseded",
    "cold-storage",
)


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

    def query(self, query: CatalogQuery) -> list[CatalogObject]:
        """List objects of one kind with their non-shadowed effective tags."""
        self._ensure_index()
        return [
            self._catalog_object(query.object_kind, row)
            for row in self.catalog.query(query)
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

    def virtual_folders(self) -> list[tuple[str, int]]:
        """Return ``(name, object count)`` for every built-in folder."""
        return [(name, len(self.virtual_folder(name))) for name in VIRTUAL_FOLDERS]

    def virtual_folder(self, name: str) -> list[CatalogObject]:
        """Return the session objects of one built-in virtual folder."""
        if name == "by-model":
            return self.query(
                CatalogQuery(object_kind="session", tag_namespace="model")
            )
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
        added = self._validate_tags(add, "session")
        removed = _canonical_set(remove)
        root = self.project_service.session_dir(session_id)
        if private:
            self._edit_private_tags("session", session_id, added, removed)
            return self.project_service.get_session(session_id)
        document, expected = self._session_document(root, session_id)
        _apply_tag_edits(document.assignments, added, removed)
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
        """Set or clear the session retention policy."""
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        document.retention = retention
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
        added = self._validate_tags(add, "artifact")
        removed = _canonical_set(remove)
        if private:
            self._require_artifact(artifact_id)
            self._edit_private_tags("artifact", artifact_id, added, removed)
            return self.effective_tags("artifact", artifact_id)
        artifact_dir = self._artifact_dir(artifact_id)
        document, expected = self._artifact_document(artifact_dir, artifact_id)
        _apply_tag_edits(document.assignments, added, removed)
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
        add: list[str] | tuple[str, ...] = (),
        remove: list[str] | tuple[str, ...] = (),
        private: bool = False,
    ) -> list[EffectiveTagInfo]:
        """Add/remove occurrence tag assignments; returns its effective tags."""
        occurrence_id = self._occurrence_id(session_id, artifact_id)
        added = self._validate_tags(add, "occurrence")
        removed = _canonical_set(remove)
        if private:
            self._edit_private_tags("occurrence", occurrence_id, added, removed)
            return self.effective_tags("occurrence", occurrence_id)
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        occurrence = document.occurrences.setdefault(
            artifact_id, OccurrenceAnnotations()
        )
        _apply_tag_edits(occurrence.assignments, added, removed)
        self._save_session_document(root, document, expected)
        return self.effective_tags("occurrence", occurrence_id)

    def set_occurrence_retention(
        self,
        session_id: str,
        artifact_id: str,
        retention: RetentionPolicy | None,
    ) -> CatalogObject:
        """Set or clear one occurrence's retention policy."""
        occurrence_id = self._occurrence_id(session_id, artifact_id)
        root = self.project_service.session_dir(session_id)
        document, expected = self._session_document(root, session_id)
        occurrence = document.occurrences.setdefault(
            artifact_id, OccurrenceAnnotations()
        )
        occurrence.retention = retention
        self._save_session_document(root, document, expected)
        return self._single_object("occurrence", occurrence_id)

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
            artifact_id, session_id, _job_name = object_id.split(":", 2)
            self.tag_occurrence(session_id, artifact_id, add=[canonical])
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
    ) -> list[str]:
        policy = load_tag_policy(self.project)
        return validate_declared_tags(list(values), object_kind, policy)

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
        if self.catalog.locate_artifact(artifact_id) is None:
            raise ArtifactNotFoundError(f"unknown artifact: {artifact_id}")

    def _artifact_dir(self, artifact_id: str) -> Path:
        self._require_artifact(artifact_id)
        relative = self.catalog.locate_artifact(artifact_id)
        assert relative is not None
        return (self.project.session_root / relative).resolve()

    def _occurrence_id(self, session_id: str, artifact_id: str) -> str:
        self._ensure_index()
        rows = self.catalog.query(
            CatalogQuery(
                object_kind="occurrence",
                facets={"session_id": session_id, "artifact_id": artifact_id},
            )
        )
        if not rows:
            raise ArtifactNotFoundError(
                f"unknown occurrence: artifact {artifact_id} in session {session_id}"
            )
        return str(rows[0]["id"])

    def _single_object(self, object_kind: str, object_id: str) -> CatalogObject:
        rows = self.catalog.query(
            CatalogQuery(object_kind=object_kind, facets={"id": object_id})
        )
        return self._catalog_object(object_kind, rows[0])

    def _catalog_object(self, object_kind: str, row: dict[str, Any]) -> CatalogObject:
        object_id = str(row["id"])
        return CatalogObject(
            kind=object_kind,
            id=object_id,
            facets=row,
            effective_tags=[
                tag
                for tag in self.effective_tags(object_kind, object_id)
                if not tag.shadowed
            ],
        )


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


def _apply_tag_edits(
    assignments: list[TagAssignment], added: list[str], removed: set[str]
) -> None:
    """Apply immutable-assignment edits in place on an assignment list."""
    kept = [assignment for assignment in assignments if assignment.tag not in removed]
    existing = {assignment.tag for assignment in kept}
    kept.extend(TagAssignment(tag=tag) for tag in added if tag not in existing)
    assignments[:] = kept
