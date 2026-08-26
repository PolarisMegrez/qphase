"""CatalogService mutation roundtrips and policy enforcement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qphase.core.catalog import CatalogQuery
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.service import CatalogService
from qphase.service.project import ProjectService

pytestmark = pytest.mark.integration


def _session(
    project: ProjectContext,
    session_id: str,
    *,
    artifacts: tuple[tuple[str, str], ...] = (),
    legacy_metadata: dict | None = None,
) -> Path:
    root = project.session_root / "2026" / "08" / session_id
    root.mkdir(parents=True)
    manifest = {
        "schema": "qphase.session/2",
        "session_id": session_id,
        "project_id": project.project_id,
        "workflow_id": "example",
        "status": "completed",
        "start_time": "2026-08-26T10:00:00+08:00",
        "jobs": {},
    }
    (root / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    for job_name, artifact_id in artifacts:
        job_dir = root / job_name
        job_dir.mkdir()
        (job_dir / "artifact_manifest.json").write_text(
            json.dumps(
                {
                    "artifact_id": artifact_id,
                    "created_at": "2026-08-26T10:01:00+08:00",
                    "bundle": {"type_id": "generic.dataset_bundle/1"},
                    "products": [{"product_schema": "qphase.dataset/1"}],
                    "parents": [],
                }
            ),
            encoding="utf-8",
        )
    if legacy_metadata is not None:
        (root / "session_metadata.json").write_text(
            json.dumps(legacy_metadata), encoding="utf-8"
        )
    return root


def _write_policy(project: ProjectContext, body: str) -> None:
    (project.defaults_path.parent / "tags.yaml").write_text(body, encoding="utf-8")


def test_tag_session_roundtrip_visible_in_query(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project)

    service.tag_session("session-1", add=["Task:Scan"])

    rows = service.query(CatalogQuery(object_kind="session", tags_all=("task:scan",)))
    assert [row.id for row in rows] == ["session-1"]
    tags = {tag.tag: tag for tag in service.effective_tags("session", "session-1")}
    assert tags["task:scan"].source == "session_annotation"
    assert tags["task:scan"].assignment_id

    service.tag_session("session-1", remove=["task:scan"])

    assert service.query(
        CatalogQuery(object_kind="session", tags_all=("task:scan",))
    ) == []
    assert service.effective_tags("session", "session-1") == []


def test_session_lifecycle_and_retention_roundtrip(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project)

    service.set_session_lifecycle("session-1", "reference")
    service.set_session_retention("session-1", "evidence")

    rows = service.query(CatalogQuery(object_kind="session", lifecycle="reference"))
    assert [row.id for row in rows] == ["session-1"]
    assert rows[0].facets["retention"] == "evidence"

    service.set_session_lifecycle("session-1", None)

    assert service.query(
        CatalogQuery(object_kind="session", lifecycle="reference")
    ) == []


def test_policy_rejects_illegal_tag_value(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  stage:\n"
        "    values: [q1, q2]\n",
    )
    _session(project, "session-1")
    service = CatalogService(project)

    with pytest.raises(QPhaseConfigError, match="not an allowed value"):
        service.tag_session("session-1", add=["stage:q3"])

    service.tag_session("session-1", add=["stage:q1"])
    info = service.tag_policy()
    assert info.path is not None
    assert info.revision is not None
    assert set(info.namespaces) == {"stage"}


def test_occurrence_annotations_are_isolated(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    _session(project, "session-2", artifacts=(("sim", "art-1"),))
    service = CatalogService(project)

    service.tag_occurrence("session-1", "art-1", add=["purpose:paper/fig3"])
    service.set_occurrence_retention("session-2", "art-1", "pinned")

    first = {tag.tag for tag in service.effective_tags("occurrence", "art-1:session-1:sim")}
    second = {tag.tag for tag in service.effective_tags("occurrence", "art-1:session-2:sim")}
    assert "purpose:paper/fig3" in first
    assert "purpose:paper/fig3" not in second
    rows = {
        row.id: row
        for row in service.query(CatalogQuery(object_kind="occurrence"))
    }
    assert rows["art-1:session-1:sim"].facets["retention"] is None
    assert rows["art-1:session-2:sim"].facets["retention"] == "pinned"


def test_artifact_tag_and_lifecycle_roundtrip(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    service = CatalogService(project)

    tags = service.tag_artifact("art-1", add=["method:cam"])
    assert [tag.tag for tag in tags] == ["method:cam"]

    updated = service.set_artifact_lifecycle("art-1", "archived")
    assert updated.facets["lifecycle"] == "archived"
    rows = service.query(
        CatalogQuery(object_kind="artifact", lifecycle="archived")
    )
    assert [row.id for row in rows] == ["art-1"]


def test_tag_session_imports_legacy_alias_on_first_document(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(
        project,
        "session-1",
        legacy_metadata={"alias": "legacy-alias", "note": "legacy-note"},
    )
    service = CatalogService(project)

    service.tag_session("session-1", add=["task:scan"])

    summary = ProjectService(project).get_session("session-1")
    assert summary.alias == "legacy-alias"
    assert summary.note == "legacy-note"


def test_revision_conflict_surfaces_as_runtime_error(tmp_path, monkeypatch):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project)
    service.tag_session("session-1", add=["task:a"])

    original = service.state_store.save_session_annotations

    def interfering(session_dir, document, *, expected_revision):
        # A concurrent writer commits between the service's load and save.
        original(session_dir, document, expected_revision=expected_revision)
        return original(session_dir, document, expected_revision=expected_revision)

    monkeypatch.setattr(
        service.state_store, "save_session_annotations", interfering
    )
    with pytest.raises(RuntimeError, match="annotation revision conflict"):
        service.tag_session("session-1", add=["task:b"])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
