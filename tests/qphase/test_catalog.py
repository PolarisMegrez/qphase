"""Tests for the project object catalog read model."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from qphase.core.catalog import CatalogQuery, ProjectObjectCatalog
from qphase.core.project import ProjectContext

pytestmark = pytest.mark.integration


def _workflow_file(project: ProjectContext, tags: tuple[str, ...] = ()) -> None:
    path = project.workflow_root / "example.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "schema: qphase.workflow/2",
        "id: example",
        "title: Example",
    ]
    if tags:
        lines.append("tags:")
        lines.extend(f"  - {tag}" for tag in tags)
    lines.extend(["jobs:", "  - name: sim", "    engine:", "      dummy: {}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_policy(project: ProjectContext, body: str) -> None:
    (project.defaults_path.parent / "tags.yaml").write_text(body, encoding="utf-8")


def _session(
    project: ProjectContext,
    session_id: str,
    *,
    submission_tags: tuple[str, ...] = (),
    snapshot_tags: tuple[str, ...] = (),
    job_tags: dict[str, tuple[str, ...]] | None = None,
    artifacts: tuple[tuple[str, str], ...] = (),
    annotations: dict | None = None,
    start_time: str = "2026-08-26T10:00:00+08:00",
) -> Path:
    root = project.session_root / "2026" / "08" / session_id
    root.mkdir(parents=True)
    manifest = {
        "schema": "qphase.session/2",
        "session_id": session_id,
        "project_id": project.project_id,
        "workflow_id": "example",
        "status": "completed",
        "start_time": start_time,
        "submission_tags": list(submission_tags),
        "jobs": {},
    }
    (root / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    snapshot = {
        "schema": "qphase.workflow/2",
        "id": "example",
        "title": "Example",
        "tags": list(snapshot_tags),
        "jobs": [
            {"name": name, "tags": list(tags)}
            for name, tags in (job_tags or {}).items()
        ],
    }
    (root / "workflow_snapshot.yaml").write_text(
        json.dumps(snapshot), encoding="utf-8"
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
    if annotations is not None:
        document = {
            "schema": "qphase.session-annotations/1",
            "project_id": project.project_id,
            "session_id": session_id,
            "revision": 0,
            **annotations,
        }
        (root / "session_annotations.json").write_text(
            json.dumps(document), encoding="utf-8"
        )
    return root


def test_reindex_counts_and_rebuild_parity(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project, tags=("task:scan",))
    _session(
        project,
        "session-1",
        artifacts=(("sim", "art-1"), ("fit", "art-2")),
    )
    catalog = ProjectObjectCatalog(project)

    stats = catalog.reindex()

    assert stats.workflows == 1
    assert stats.sessions == 1
    assert stats.artifacts == 2
    assert stats.occurrences == 2
    again = catalog.reindex()
    assert again == stats.__class__(**{**stats.__dict__, "duration_seconds": again.duration_seconds})
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]


def test_occurrence_effective_tags_carry_provenance(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(
        project,
        "session-1",
        submission_tags=("task:urgent",),
        snapshot_tags=("task:scan",),
        job_tags={"sim": ("method:cam",)},
        artifacts=(("sim", "art-1"),),
        annotations={
            "assignments": [{"id": "s1", "tag": "task:review"}],
            "occurrences": {
                "art-1": {
                    "assignments": [{"id": "o1", "tag": "purpose:paper/fig3"}]
                }
            },
        },
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    occurrence_tags = catalog.effective_tags("occurrence", "art-1:session-1:sim")
    by_tag = {tag.tag: tag for tag in occurrence_tags}
    assert set(by_tag) == {
        "task:scan",
        "task:urgent",
        "task:review",
        "method:cam",
        "purpose:paper/fig3",
    }
    assert by_tag["task:scan"].source == "workflow_declared"
    assert by_tag["task:scan"].inherited
    assert by_tag["task:urgent"].source == "execution_submission"
    assert by_tag["task:review"].source == "session_annotation"
    assert by_tag["task:review"].assignment_id == "s1"
    assert by_tag["method:cam"].source == "job_declared"
    own = by_tag["purpose:paper/fig3"]
    assert own.source == "occurrence_annotation"
    assert own.assignment_id == "o1"
    assert not own.inherited

    session_tags = catalog.effective_tags("session", "session-1")
    session_by_tag = {tag.tag: tag for tag in session_tags}
    assert session_by_tag["task:review"].inherited is False
    assert session_by_tag["task:scan"].inherited is True
    # Occurrence-only tags never leak back onto the session.
    assert "purpose:paper/fig3" not in session_by_tag
    assert "method:cam" not in session_by_tag


def test_cardinality_one_shadows_farther_assignment(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  stage:\n"
        "    cardinality: one\n"
        "    open: true\n",
    )
    _workflow_file(project)
    _session(
        project,
        "session-1",
        snapshot_tags=("stage:q1",),
        artifacts=(("sim", "art-1"),),
        annotations={"assignments": [{"id": "s1", "tag": "stage:q2"}]},
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    tags = catalog.effective_tags("occurrence", "art-1:session-1:sim")
    by_tag = {tag.tag: tag for tag in tags}
    assert by_tag["stage:q1"].shadowed
    assert not by_tag["stage:q2"].shadowed
    # Shadowed tags do not match queries.
    assert catalog.query(
        CatalogQuery(object_kind="occurrence", tags_all=("stage:q1",))
    ) == []
    assert len(
        catalog.query(
            CatalogQuery(object_kind="occurrence", tags_all=("stage:q2",))
        )
    ) == 1


def test_inherit_false_namespace_does_not_flow_down(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  local:\n"
        "    inherit: false\n"
        "    open: true\n",
    )
    _workflow_file(project, tags=("local:wip",))
    _session(project, "session-1", snapshot_tags=("local:wip",))
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    workflow_tags = catalog.effective_tags("workflow", "example")
    assert [tag.tag for tag in workflow_tags] == ["local:wip"]
    session_tags = catalog.effective_tags("session", "session-1")
    assert session_tags == []


def test_same_artifact_in_two_sessions_keeps_contexts_isolated(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(
        project,
        "session-1",
        submission_tags=("task:first",),
        artifacts=(("sim", "art-1"),),
    )
    _session(
        project,
        "session-2",
        submission_tags=("task:second",),
        artifacts=(("sim", "art-1"),),
    )
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()

    assert stats.artifacts == 1
    assert stats.occurrences == 2
    first = {
        tag.tag for tag in catalog.effective_tags("occurrence", "art-1:session-1:sim")
    }
    second = {
        tag.tag for tag in catalog.effective_tags("occurrence", "art-1:session-2:sim")
    }
    assert first == {"task:first"}
    assert second == {"task:second"}
    assert catalog.locate_artifact("art-1") is not None


def test_query_filters_and_stable_pagination(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(
        project,
        "session-1",
        start_time="2026-08-26T10:00:00+08:00",
        annotations={
            "lifecycle": "active",
            "retention": "evidence",
            "assignments": [{"id": "s1", "tag": "purpose:paper/fig3"}],
        },
    )
    _session(
        project,
        "session-2",
        start_time="2026-08-27T10:00:00+08:00",
        annotations={"lifecycle": "superseded", "retention": "transient"},
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    assert [
        row["id"]
        for row in catalog.query(
            CatalogQuery(object_kind="session", lifecycle="active")
        )
    ] == ["session-1"]
    assert [
        row["id"]
        for row in catalog.query(
            CatalogQuery(object_kind="session", retention="transient")
        )
    ] == ["session-2"]
    assert [
        row["id"]
        for row in catalog.query(
            CatalogQuery(object_kind="session", tag_descendant_of="purpose:paper")
        )
    ] == ["session-1"]
    assert [
        row["id"]
        for row in catalog.query(
            CatalogQuery(object_kind="session", tags_without=("purpose:paper/fig3",))
        )
    ] == ["session-2"]
    assert [
        row["id"]
        for row in catalog.query(
            CatalogQuery(
                object_kind="session",
                ranges={"start_time": (None, "2026-08-26T23:59:59+08:00")},
            )
        )
    ] == ["session-1"]
    # Stable sort: ascending start_time, then id; pagination slices it.
    page = catalog.query(CatalogQuery(object_kind="session", limit=1, offset=1))
    assert [row["id"] for row in page] == ["session-2"]
    with pytest.raises(ValueError, match="unknown session facet"):
        catalog.query(CatalogQuery(object_kind="session", facets={"nope": "x"}))


def test_retention_inherits_to_occurrences(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(
        project,
        "session-1",
        artifacts=(("sim", "art-1"), ("fit", "art-2"),),
        annotations={
            "retention": "evidence",
            "occurrences": {"art-2": {"retention": "pinned"}},
        },
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    rows = {
        row["artifact_id"]: row
        for row in catalog.query(CatalogQuery(object_kind="occurrence", limit=10))
    }
    assert rows["art-1"]["retention"] is None
    assert rows["art-1"]["effective_retention"] == "evidence"
    assert rows["art-2"]["retention"] == "pinned"
    assert rows["art-2"]["effective_retention"] == "pinned"
    assert [
        row["artifact_id"]
        for row in catalog.query(
            CatalogQuery(object_kind="occurrence", retention="evidence")
        )
    ] == ["art-1"]


def test_corrupt_catalog_rebuilds_from_disk_truth(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()
    assert stats.sessions == 1

    catalog.path.write_bytes(b"corrupt sqlite payload")

    # The read model detects corruption, resets to an empty schema and the
    # next reindex restores identical content from disk truth.
    assert catalog.query(CatalogQuery(object_kind="session")) == []
    rebuilt = catalog.reindex()
    assert rebuilt.sessions == 1
    assert rebuilt.artifacts == 1
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]
