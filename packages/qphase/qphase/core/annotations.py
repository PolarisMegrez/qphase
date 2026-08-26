"""Typed annotation documents for sessions, artifacts and occurrences.

Annotations are the mutable organization layer: tags, lifecycle, retention,
alias and note. They are *not* part of the immutable artifact manifest — the
manifest never changes after publication, annotations live in sidecar
documents with optimistic-concurrency revisions:

- ``SessionAnnotationDocument`` (``qphase.session-annotations/1``) lives in
  the session directory (``session_annotations.json``) and also carries the
  per-occurrence annotations keyed by artifact id, because an occurrence's
  producing context is session-scoped truth;
- ``ArtifactAnnotationDocument`` (``qphase.artifact-annotations/1``) lives in
  the artifact directory (``artifact_annotations.json``) so identity-scoped
  annotations travel with the artifact.

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
    "SESSION_ANNOTATIONS_FILENAME",
    "ArtifactAnnotationDocument",
    "Lifecycle",
    "OccurrenceAnnotations",
    "RetentionPolicy",
    "SessionAnnotationDocument",
    "TagAssignment",
]

#: File name of the session annotation document inside a session directory.
SESSION_ANNOTATIONS_FILENAME = "session_annotations.json"

#: File name of the artifact annotation document inside an artifact directory.
ARTIFACT_ANNOTATIONS_FILENAME = "artifact_annotations.json"

Lifecycle = Literal["active", "reference", "superseded", "archived"]
RetentionPolicy = Literal["transient", "preserve", "evidence", "pinned"]


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class TagAssignment(BaseModel):
    """One immutable tag assignment with a stable id."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tag: str
    created_at: str = Field(default_factory=_utc_now)

    @field_validator("tag")
    @classmethod
    def _check_tag(cls, value: str) -> str:
        return canonicalize_tag_syntax(value)


class OccurrenceAnnotations(BaseModel):
    """Annotations of one producing occurrence (keyed by artifact id)."""

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
