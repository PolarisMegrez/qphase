"""Long-lived tests for the project Session/Event persistence ports."""

from __future__ import annotations

import threading

import pytest
from qphase.core.errors import QPhaseIOError
from qphase.core.persistence import ProjectStateStore
from qphase.core.project import ProjectContext

pytestmark = pytest.mark.integration


def test_project_state_store_round_trips_manifest_and_event_cursor(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    session_dir = project.session_root / "session"
    session_dir.mkdir()

    store.save_session_manifest(
        session_dir,
        {"schema": "qphase.session/2", "session_id": "session", "jobs": {}},
    )
    store.append_events(
        session_dir,
        [
            {"sequence": 1, "kind": "queued"},
            {"sequence": 2, "kind": "completed"},
        ],
    )

    assert store.load_session_manifest(session_dir)["session_id"] == "session"
    assert [event["sequence"] for event in store.read_events(session_dir)] == [1, 2]
    assert [event["kind"] for event in store.read_events(session_dir, after=1)] == [
        "completed"
    ]


def test_project_state_store_round_trips_execution_record(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    payload = {
        "schema": "qphase.execution/1",
        "execution_id": "execution-1",
        "submitted_at": "2026-08-26T10:00:00+08:00",
        "state": "queued",
    }

    store.save_execution(payload)

    assert store.load_executions() == [payload]
    store.delete_execution("execution-1")
    assert store.load_executions() == []


def test_project_state_store_rejects_paths_outside_project(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)

    with pytest.raises(QPhaseIOError, match="escapes"):
        store.load_session_manifest(tmp_path / "outside")


def _annotation_document(project: ProjectContext, **extra):
    document = {
        "schema": "qphase.session-annotations/1",
        "project_id": project.project_id,
        "session_id": "session-1",
        "assignments": [],
        "alias": "first",
    }
    document.update(extra)
    return document


def test_annotation_store_roundtrip_and_revision_conflict(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    session_dir = project.session_root / "2026" / "08" / "session-1"
    session_dir.mkdir(parents=True)

    stored = store.save_session_annotations(
        session_dir, _annotation_document(project), expected_revision=None
    )
    assert stored["revision"] == 0
    assert store.load_session_annotations(session_dir)["alias"] == "first"

    stored = store.save_session_annotations(
        session_dir, {**stored, "alias": "second"}, expected_revision=0
    )
    assert stored["revision"] == 1
    assert store.load_session_annotations(session_dir)["alias"] == "second"

    with pytest.raises(RuntimeError, match="annotation revision conflict"):
        store.save_session_annotations(session_dir, stored, expected_revision=0)

    events = store.read_events(session_dir)
    assert [event["payload"]["kind"] for event in events] == [
        "annotations_updated",
        "annotations_updated",
    ]
    assert [event["sequence"] for event in events] == [1, 2]


def test_annotation_store_rejects_stale_create_and_ignores_tmp(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    session_dir = project.session_root / "session-1"
    session_dir.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="annotation revision conflict"):
        store.save_session_annotations(
            session_dir, _annotation_document(project), expected_revision=3
        )
    assert store.load_session_annotations(session_dir) is None

    # A stale temporary file from an interrupted write never shadows the
    # committed document.
    store.save_session_annotations(
        session_dir, _annotation_document(project), expected_revision=None
    )
    (session_dir / "session_annotations.tmp").write_text("garbage")
    assert store.load_session_annotations(session_dir)["revision"] == 0


def test_artifact_annotations_stay_inside_session_root(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    session_dir = project.session_root / "session-1"
    artifact_dir = session_dir / "job"
    artifact_dir.mkdir(parents=True)

    with pytest.raises(QPhaseIOError, match="escapes"):
        store.save_artifact_annotations(
            tmp_path / "outside",
            _annotation_document(project),
            expected_revision=None,
        )

    store.save_artifact_annotations(
        artifact_dir, _annotation_document(project), expected_revision=None
    )
    assert store.load_artifact_annotations(artifact_dir)["alias"] == "first"
    # Artifact annotation writes journal to the owning session directory.
    assert store.read_events(session_dir)[0]["payload"]["kind"] == (
        "annotations_updated"
    )


def test_annotation_writes_serialize_concurrent_writers(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    store = ProjectStateStore(project)
    session_dir = project.session_root / "2026" / "08" / "session-1"
    session_dir.mkdir(parents=True)
    store.save_session_annotations(
        session_dir, _annotation_document(project), expected_revision=None
    )

    barrier = threading.Barrier(2)
    outcomes: list[str] = []

    def write(alias: str) -> None:
        # Both writers base their edit on revision 0, then race to commit.
        document = store.load_session_annotations(session_dir)
        barrier.wait(timeout=10)
        try:
            store.save_session_annotations(
                session_dir, {**document, "alias": alias}, expected_revision=0
            )
        except RuntimeError:
            outcomes.append("conflict")
        else:
            outcomes.append("ok")

    threads = [
        threading.Thread(target=write, args=(alias,)) for alias in ("a", "b")
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)
        assert not thread.is_alive()

    # Exactly one writer wins; the loser gets a stable revision conflict and
    # no update is lost silently.
    assert sorted(outcomes) == ["conflict", "ok"]
    assert store.load_session_annotations(session_dir)["revision"] == 1
