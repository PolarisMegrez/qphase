"""Long-lived tests for the project Session/Event persistence ports."""

from __future__ import annotations

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
