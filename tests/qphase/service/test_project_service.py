"""Session annotation writes through ProjectService."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qphase.core.project import ProjectContext
from qphase.service.project import ProjectService


def _make_session(project: ProjectContext, session_id: str = "session-1") -> Path:
    root = project.session_root / "2026" / "08" / session_id
    root.mkdir(parents=True)
    manifest = {
        "schema": "qphase.session/2",
        "session_id": session_id,
        "project_id": project.project_id,
        "workflow_id": "wf",
        "status": "completed",
        "jobs": {},
    }
    (root / "session_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root


def test_update_session_writes_annotation_document(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _make_session(project)
    service = ProjectService(project)

    service.update_session("session-1", note="keep this")
    summary = service.update_session("session-1", alias="reference")

    assert summary.alias == "reference"
    assert summary.note == "keep this"
    # The legacy sidecar is no longer written.
    assert not (root / "session_metadata.json").exists()
    document = json.loads(
        (root / "session_annotations.json").read_text(encoding="utf-8")
    )
    assert document["schema"] == "qphase.session-annotations/1"
    assert document["revision"] == 1


def test_legacy_metadata_sidecar_is_ignored(tmp_path):
    """The removed session_metadata.json fallback must not leak alias/note."""
    project = ProjectContext.create(tmp_path / "project")
    root = _make_session(project)
    (root / "session_metadata.json").write_text(
        json.dumps({"alias": "legacy-alias", "note": "legacy-note"}),
        encoding="utf-8",
    )
    service = ProjectService(project)

    # Without an annotation document the legacy sidecar is not read.
    summary = service.get_session("session-1")
    assert summary.alias is None
    assert summary.note is None

    # Creating the annotation document does not import legacy values.
    summary = service.update_session("session-1", note="new note")
    assert summary.alias is None
    assert summary.note == "new note"
    document = json.loads(
        (root / "session_annotations.json").read_text(encoding="utf-8")
    )
    assert document["alias"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
