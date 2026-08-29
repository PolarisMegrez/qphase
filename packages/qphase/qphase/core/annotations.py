"""Typed annotation documents for sessions, artifacts and occurrences.

Annotations are the mutable organization layer: tags, lifecycle, retention,
alias and note. They are *not* part of the immutable artifact manifest — the
manifest never changes after publication, annotations live in sidecar
documents with optimistic-concurrency revisions:

- ``SessionAnnotationDocument`` (``qphase.session-annotations/1``) lives in
  the session directory (``session_annotations.json``) and also carries the
  per-occurrence annotations keyed by ``job_name:artifact_id``, because an
  occurrence's producing context is session-scoped truth;
- ``ArtifactAnnotationDocument`` (``qphase.artifact-annotations/1``) lives in
  the artifact directory (``artifact_annotations.json``) so identity-scoped
  annotations travel with the artifact;
- ``ProjectAnnotationDocument`` (``qphase.project-annotations/1``) lives in
  the project ``.qphase`` directory (``project_annotations.json``) and
  carries the annotations of the project itself plus the per-object
  annotations of workflow revisions (``workflow_id@revision``), jobs
  (``workflow_id@revision:job_name``) and executions, which have no
  directory of their own.

Lifecycle and retention are typed fields, never plain tags. Tag assignments
are immutable once created — editing a tag removes one assignment and adds a
new one, so every effective tag can cite a stable assignment id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .tags import canonicalize_tag_syntax

__all__ = [
    "ARTIFACT_ANNOTATIONS_FILENAME",
    "PROJECT_ANNOTATIONS_FILENAME",
    "SESSION_ANNOTATIONS_FILENAME",
    "ArtifactAnnotationDocument",
    "Lifecycle",
    "ObjectAnnotations",
    "OccurrenceAnnotations",
    "ProjectAnnotationDocument",
    "RetentionPolicy",
    "SessionAnnotationDocument",
    "TagAssignment",
]

#: File name of the session annotation document inside a session directory.
SESSION_ANNOTATIONS_FILENAME = "session_annotations.json"

#: File name of the artifact annotation document inside an artifact directory.
ARTIFACT_ANNOTATIONS_FILENAME = "artifact_annotations.json"

#: File name of the project annotation document inside the project
#: ``.qphase`` directory.
PROJECT_ANNOTATIONS_FILENAME = "project_annotations.json"

Lifecycle = Literal["active", "reference", "superseded", "archived"]
RetentionPolicy = Literal["transient", "preserve", "evidence", "pinned"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TagAssignment(BaseModel):
    """One immutable tag assignment with a stable id.

    ``policy_revision`` freezes the tag policy revision that validated the
    assignment at write time, so historical provenance survives later policy
    edits. ``inherit``/``cardinality``/``objects`` freeze the minimal
    namespace rule governing effective-tag resolution at write time;
    assignments written before rule freezing (or without a governing
    namespace rule) carry ``None`` and fall back to the current policy.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tag: str
    created_at: str = Field(default_factory=_utc_now)
    policy_revision: str | None = None
    inherit: bool | None = None
    cardinality: Literal["one", "many"] | None = None
    objects: tuple[str, ...] | None = None

    @field_validator("tag")
    @classmethod
    def _check_tag(cls, value: str) -> str:
        return canonicalize_tag_syntax(value)


class OccurrenceAnnotations(BaseModel):
    """Annotations of one producing occurrence (keyed by job + artifact id)."""

    model_config = ConfigDict(extra="forbid")

    assignments: list[TagAssignment] = Field(default_factory=list)
    retention: RetentionPolicy | None = None
    note: str | None = None


class SessionAnnotationDocument(BaseModel):
    """Durable annotation state of one session and its occurrences."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["qphase.session-annotations/1"] = Field(
        default="qphase.session-annotations/1", alias="schema"
    )
    project_id: str
    session_id: str
    revision: int = 0
    updated_at: str = Field(default_factory=_utc_now)
    assignments: list[TagAssignment] = Field(default_factory=list)
    lifecycle: Lifecycle | None = None
    retention: RetentionPolicy | None = None
    # Frozen at retention-set time; ``None`` on legacy documents falls back
    # to the current policy (and to ``True`` without a policy).
    retention_inherits_to_occurrences: bool | None = None
    alias: str | None = None
    note: str | None = None
    occurrences: dict[str, OccurrenceAnnotations] = Field(default_factory=dict)


class ArtifactAnnotationDocument(BaseModel):
    """Durable identity-scoped annotations of one artifact."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["qphase.artifact-annotations/1"] = Field(
        default="qphase.artifact-annotations/1", alias="schema"
    )
    project_id: str
    artifact_id: str
    revision: int = 0
    updated_at: str = Field(default_factory=_utc_now)
    assignments: list[TagAssignment] = Field(default_factory=list)
    lifecycle: Lifecycle | None = None
    retention: RetentionPolicy | None = None
    note: str | None = None


class ObjectAnnotations(BaseModel):
    """Annotations of one catalog object inside the project document.

    Keyed by object id: ``workflow_id@revision`` for workflow revisions,
    ``workflow_id@revision:job_name`` for jobs and the execution id for
    executions.
    """

    model_config = ConfigDict(extra="forbid")

    assignments: list[TagAssignment] = Field(default_factory=list)
    note: str | None = None


class ProjectAnnotationDocument(BaseModel):
    """Durable annotations of the project and its workflow/job/execution objects."""

    model_config = ConfigDict(extra="forbid")

    schema_: Literal["qphase.project-annotations/1"] = Field(
        default="qphase.project-annotations/1", alias="schema"
    )
    project_id: str
    revision: int = 0
    updated_at: str = Field(default_factory=_utc_now)
    assignments: list[TagAssignment] = Field(default_factory=list)
    alias: str | None = None
    note: str | None = None
    objects: dict[str, ObjectAnnotations] = Field(default_factory=dict)
