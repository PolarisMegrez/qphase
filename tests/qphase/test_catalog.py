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


def _v4_artifact_manifest(artifact_id: str, quantities: tuple[str, ...] = ()) -> dict:
    """Minimal valid ``qphase.artifact/4`` manifest for scan fixtures."""
    products = [
        {
            "name": f"product_{index}",
            "product_schema": {
                "schema_version": "qphase.product/1",
                "kind": "time_series",
                "axes": [{"name": "time", "role": "coordinate", "size": 4}],
                "variables": [
                    {
                        "name": "x",
                        "dtype": "float64",
                        "value_domain": "real",
                        "dims": ["time"],
                        "quantity": quantity,
                    }
                ],
            },
            "storage": {
                # An unregistered adapter skips descriptor validation while
                # the generic summary is still cross-checked.
                "adapter": "none/1",
                "descriptor_schema": "none/1",
                "summary": {
                    "x": {
                        "full_shape": [4],
                        "dtype": "<f8",
                        "nbytes": 32,
                        "chunk_count": 1,
                    }
                },
                "descriptor": {},
            },
        }
        for index, quantity in enumerate(quantities)
    ]
    return {
        "schema_version": "qphase.artifact/4",
        "artifact_id": artifact_id,
        "created_at": "2026-08-26T10:01:00+08:00",
        "bundle": {
            "type_id": "generic.dataset_bundle/1",
            "adapter_id": "test-unregistered/1",
            "descriptor_schema": "test.bundle/1",
            "descriptor": {},
            "product_roles": {},
        },
        "products": products,
        "provenance": {},
        "parents": [],
    }


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
    frozen: dict | None = None,
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
    # The snapshot mirrors the current workflow file (job "sim", dummy
    # engine) so an unchanged session rebuilds the same workflow revision.
    snapshot = {
        "schema": "qphase.workflow/2",
        "id": "example",
        "title": "Example",
        "tags": list(snapshot_tags),
        "jobs": [
            {
                "name": "sim",
                "engine": {"dummy": {}},
                "tags": list((job_tags or {}).get("sim", ())),
            }
        ],
    }
    (root / "workflow_snapshot.yaml").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    if frozen is not None:
        (root / "tag_snapshot.yaml").write_text(
            json.dumps(frozen), encoding="utf-8"
        )
    for job_name, artifact_id in artifacts:
        job_dir = root / job_name
        job_dir.mkdir()
        (job_dir / "artifact_manifest.json").write_text(
            json.dumps(_v4_artifact_manifest(artifact_id)),
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
        snapshot_tags=("task:scan",),
        artifacts=(("sim", "art-1"), ("fit", "art-2")),
    )
    catalog = ProjectObjectCatalog(project)

    stats = catalog.reindex()

    assert stats.projects == 1
    assert stats.workflows == 1
    assert stats.jobs == 1
    assert stats.sessions == 1
    assert stats.artifacts == 2
    assert stats.occurrences == 2
    again = catalog.reindex()
    assert again == stats.__class__(**{**stats.__dict__, "duration_seconds": again.duration_seconds})
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]
    projects = catalog.query(CatalogQuery(object_kind="project"))
    assert [row["id"] for row in projects] == [project.project_id]


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
                "sim:art-1": {
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

    (workflow_row,) = catalog.query(CatalogQuery(object_kind="workflow"))
    workflow_tags = catalog.effective_tags("workflow", workflow_row["id"])
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
    # Both occurrence locations stay indexed under one artifact identity.
    assert len(catalog.locate_artifact_paths("art-1")) == 2


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
            "occurrences": {"fit:art-2": {"retention": "pinned"}},
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

    # The read model detects corruption and rebuilds from disk truth on the
    # next read instead of serving empty results.
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]
    rebuilt = catalog.reindex()
    assert rebuilt.sessions == 1
    assert rebuilt.artifacts == 1
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]


def test_query_tag_namespace_filter(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(
        project,
        "session-1",
        annotations={"assignments": [{"id": "s1", "tag": "model:cam"}]},
    )
    _session(
        project,
        "session-2",
        start_time="2026-08-27T10:00:00+08:00",
        annotations={"assignments": [{"id": "s2", "tag": "task:scan"}]},
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    rows = catalog.query(CatalogQuery(object_kind="session", tag_namespace="model"))
    assert [row["id"] for row in rows] == ["session-1"]
    with pytest.raises(ValueError, match="invalid tag namespace"):
        CatalogQuery(object_kind="session", tag_namespace="model:cam")


def test_workflow_revisions_accumulate_without_overwrite(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project, tags=("task:scan",))
    # A historical session froze an older revision of the same workflow id.
    _session(project, "session-1", snapshot_tags=("task:legacy",))
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()

    assert stats.workflows == 2
    rows = catalog.query(
        CatalogQuery(object_kind="workflow", facets={"workflow_id": "example"})
    )
    assert len(rows) == 2
    ids = {row["id"] for row in rows}
    assert all(revision.startswith("example@") for revision in ids)
    sources = {row["id"]: json.loads(row["sources_json"]) for row in rows}
    file_rows = [row for row in rows if row["relative_path"] == "example.yaml"]
    snapshot_rows = [row for row in rows if row["relative_path"] is None]
    assert len(file_rows) == 1 and len(snapshot_rows) == 1
    assert sources[file_rows[0]["id"]] == ["file:example.yaml"]
    assert sources[snapshot_rows[0]["id"]][0].startswith("session:2026/08/session-1")
    # Editing the file replaces the file-backed revision (the old content is
    # no longer on disk); the snapshot-frozen revision survives.
    old_file_id = file_rows[0]["id"]
    _workflow_file(project, tags=("task:revised",))
    assert catalog.reindex().workflows == 2
    rows = catalog.query(
        CatalogQuery(object_kind="workflow", facets={"workflow_id": "example"})
    )
    new_ids = {row["id"] for row in rows}
    assert old_file_id not in new_ids
    assert snapshot_rows[0]["id"] in new_ids


def test_job_objects_indexed_with_facets(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1")
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()

    # File and session snapshot freeze the same document, so one revision
    # with two sources and one job object.
    assert stats.workflows == 1
    assert stats.jobs == 1
    (workflow_row,) = catalog.query(CatalogQuery(object_kind="workflow"))
    assert sorted(json.loads(workflow_row["sources_json"])) == [
        "file:example.yaml",
        "session:2026/08/session-1",
    ]
    (job_row,) = catalog.query(
        CatalogQuery(
            object_kind="job",
            facets={"workflow_revision_id": workflow_row["id"]},
        )
    )
    assert job_row["id"] == f"{workflow_row['id']}:sim"
    assert job_row["workflow_id"] == "example"
    assert job_row["name"] == "sim"
    assert job_row["engine"] == "dummy"
    assert job_row["model"] is None


def test_frozen_snapshot_provenance_survives_policy_change(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n  stage:\n    open: true\n",
    )
    from qphase.core.tags import load_tag_policy

    revision_v1 = load_tag_policy(project).revision
    _workflow_file(project)
    _session(
        project,
        "session-1",
        frozen={
            "raw_tags": ["stage:q1"],
            "canonical_tags": ["stage:q1"],
            "job_tags": {},
            "policy_revision": revision_v1,
            "assignments": {
                "workflow": [{"tag": "stage:q1", "assignment_id": "wf-a1"}],
                "jobs": {},
            },
        },
        artifacts=(("sim", "art-1"),),
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    session_tags = {
        tag.tag: tag for tag in catalog.effective_tags("session", "session-1")
    }
    assert session_tags["stage:q1"].policy_revision == revision_v1
    assert session_tags["stage:q1"].assignment_id == "wf-a1"
    occurrence_tags = {
        tag.tag: tag
        for tag in catalog.effective_tags("occurrence", "art-1:session-1:sim")
    }
    assert occurrence_tags["stage:q1"].assignment_id == "wf-a1"

    # A policy edit must not rewrite the frozen provenance on reindex.
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n"
        "  stage:\n    open: true\n  task:\n    open: true\n",
    )
    assert load_tag_policy(project).revision != revision_v1
    catalog.reindex()

    again = {
        tag.tag: tag for tag in catalog.effective_tags("session", "session-1")
    }
    assert again["stage:q1"].policy_revision == revision_v1
    assert again["stage:q1"].assignment_id == "wf-a1"


def test_execution_links_workflow_revision_and_freezes_provenance(tmp_path):
    from qphase.core.persistence import ProjectStateStore
    from qphase.core.workflow import load_workflow

    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    workflow = load_workflow(project.workflow_root / "example.yaml")
    workflow_payload = workflow.model_dump(mode="json", by_alias=True)
    ProjectStateStore(project).save_execution(
        {
            "schema": "qphase.execution/1",
            "execution_id": "exec-1",
            "source_workflow": "example",
            "submission_tags": ["task:urgent"],
            "tag_policy_revision": "rev-at-submit",
            "workflow": workflow_payload,
            "compiled_workflow": {
                "schema": "qphase.compiled_workflow/1",
                "workflow": workflow_payload,
            },
            "submitted_at": "2026-08-26T09:00:00+08:00",
            "state": "completed",
        }
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    (execution,) = catalog.query(CatalogQuery(object_kind="execution"))
    (workflow_row,) = catalog.query(CatalogQuery(object_kind="workflow"))
    assert execution["workflow_revision_id"] == workflow_row["id"]
    tags = catalog.effective_tags("execution", "exec-1")
    assert [(tag.tag, tag.policy_revision) for tag in tags] == [
        ("task:urgent", "rev-at-submit")
    ]


def test_artifact_facets_extracted_from_v4_manifest(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    (root / "sim" / "artifact_manifest.json").write_text(
        json.dumps(
            _v4_artifact_manifest(
                "art-1", quantities=("power_spectral_density", "amplitude")
            )
        ),
        encoding="utf-8",
    )
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    (row,) = catalog.query(CatalogQuery(object_kind="artifact"))
    assert row["id"] == "art-1"
    assert row["created_at"] == "2026-08-26T10:01:00+08:00"
    assert row["bundle_type"] == "generic.dataset_bundle/1"
    schemas = json.loads(row["product_schemas_json"])
    assert sorted(schemas) == ["product_0", "product_1"]
    assert all(len(fingerprint) == 64 for fingerprint in schemas.values())
    # Quantities are the sorted set of non-empty variable quantities.
    assert json.loads(row["quantities_json"]) == [
        "amplitude",
        "power_spectral_density",
    ]
    assert json.loads(row["parents_json"]) == []
    assert catalog.location_issues() == []


def test_unreadable_artifact_locations_become_issues(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    corrupt = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    (corrupt / "sim" / "artifact_manifest.json").write_text(
        "{not json", encoding="utf-8"
    )
    legacy = _session(project, "session-2", artifacts=(("sim", "art-2"),))
    manifest = _v4_artifact_manifest("art-2")
    manifest["schema_version"] = "qphase.artifact/3"
    (legacy / "sim" / "artifact_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()

    # Unreadable locations index no artifact/occurrence rows at all.
    assert stats.artifacts == 0
    assert stats.occurrences == 0
    assert stats.location_issues == 2
    issues = {(issue["path"], issue["kind"]) for issue in catalog.location_issues()}
    assert issues == {
        ("2026/08/session-1/sim", "corrupt"),
        ("2026/08/session-2/sim", "unsupported"),
    }


def test_conflicting_occurrences_keep_first_row_and_report(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    divergent = _session(project, "session-2", artifacts=(("sim", "art-1"),))
    (divergent / "sim" / "artifact_manifest.json").write_text(
        json.dumps(_v4_artifact_manifest("art-1", quantities=("amplitude",))),
        encoding="utf-8",
    )
    catalog = ProjectObjectCatalog(project)
    stats = catalog.reindex()

    # Both occurrences stay indexed; the artifact row keeps the first
    # location's facets and the divergent location is a conflict issue.
    assert stats.artifacts == 1
    assert stats.occurrences == 2
    (row,) = catalog.query(CatalogQuery(object_kind="artifact"))
    assert row["path"] == "2026/08/session-1/sim"
    assert json.loads(row["quantities_json"]) == []
    (issue,) = catalog.location_issues()
    assert issue["kind"] == "conflict"
    assert issue["path"] == "2026/08/session-2/sim"
    assert "session-1/sim" in issue["message"]


def test_foreign_project_id_catalog_rebuilds(tmp_path):
    import sqlite3

    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    connection = sqlite3.connect(catalog.path)
    with connection:
        connection.execute(
            "UPDATE meta SET value = 'qp_foreign' WHERE key = 'project_id'"
        )
    connection.close()

    # A catalog stamped with another project id is never trusted: the next
    # read rebuilds it from disk truth instead of serving stale rows.
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1"]


def test_query_reindexes_when_disk_changes(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1")
    catalog = ProjectObjectCatalog(project)
    catalog.reindex()

    # A new session flips the fingerprint probe; the next query reindexes.
    _session(project, "session-2")
    rows = catalog.query(CatalogQuery(object_kind="session"))
    assert [row["id"] for row in rows] == ["session-1", "session-2"]
