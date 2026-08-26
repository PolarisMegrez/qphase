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


def test_private_tags_overlay_shared_without_reindex(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["task:scan"])
    service.tag_session("session-1", add=["task:wip"], private=True)

    tags = {tag.tag: tag for tag in service.effective_tags("session", "session-1")}
    assert tags["task:scan"].source == "session_annotation"
    assert tags["task:wip"].source == "user_private"
    # Private writes never create or touch the shared annotation document.
    document = json.loads(
        (root / "session_annotations.json").read_text(encoding="utf-8")
    )
    assert [item["tag"] for item in document["assignments"]] == ["task:scan"]

    service.tag_session("session-1", remove=["task:wip"], private=True)
    assert {
        tag.tag for tag in service.effective_tags("session", "session-1")
    } == {"task:scan"}


def test_private_tag_shadows_shared_in_cardinality_one_namespace(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  stage:\n"
        "    cardinality: one\n"
        "    open: true\n",
    )
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["stage:q1"])
    service.tag_session("session-1", add=["stage:q2"], private=True)

    tags = {tag.tag: tag for tag in service.effective_tags("session", "session-1")}
    assert tags["stage:q1"].shadowed
    assert not tags["stage:q2"].shadowed
    assert tags["stage:q2"].source == "user_private"
    # CatalogObject.effective_tags hides shadowed entries.
    row = service.query(
        CatalogQuery(object_kind="session", facets={"id": "session-1"})
    )[0]
    assert [tag.tag for tag in row.effective_tags] == ["stage:q2"]


def test_promote_tag_moves_private_tag_into_shared_annotations(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["task:wip"], private=True)
    promoted = service.promote_tag("session", "session-1", "task:wip")

    tags = {tag.tag: tag for tag in promoted}
    assert tags["task:wip"].source == "session_annotation"
    assert service.private.list_private_tags("session", "session-1") == []


def test_promote_occurrence_tag(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_occurrence("session-1", "art-1", add=["purpose:draft"], private=True)
    promoted = service.promote_tag(
        "occurrence", "art-1:session-1:sim", "purpose:draft"
    )

    tags = {tag.tag: tag for tag in promoted}
    assert tags["purpose:draft"].source == "occurrence_annotation"
    assert service.private.list_private_tags("occurrence") == []


def test_saved_views_roundtrip(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    service = CatalogService(project, home=tmp_path / "home")

    service.save_view(
        "review", CatalogQuery(object_kind="session", tags_all=("task:scan",))
    )

    views = service.list_views()
    assert [name for name, _ in views] == ["review"]
    assert views[0][1] == CatalogQuery(
        object_kind="session", tags_all=("task:scan",)
    )

    service.delete_view("review")
    assert service.list_views() == []


def test_virtual_folders(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "s-model")
    _session(project, "s-evidence")
    _session(project, "s-pinned")
    _session(project, "s-diag")
    _session(project, "s-superseded")
    _session(project, "s-archived")
    _session(project, "s-plain")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("s-model", add=["model:cam"])
    service.set_session_retention("s-evidence", "evidence")
    service.set_session_retention("s-pinned", "pinned")
    service.tag_session("s-diag", add=["task:diagnostics"])
    service.set_session_lifecycle("s-superseded", "superseded")
    service.set_session_lifecycle("s-archived", "archived")

    folders = dict(service.virtual_folders())
    assert folders == {
        "by-model": 1,
        "paper-evidence": 2,
        "diagnostics": 1,
        "superseded": 1,
        "cold-storage": 1,
    }
    assert [row.id for row in service.virtual_folder("by-model")] == ["s-model"]
    assert [row.id for row in service.virtual_folder("paper-evidence")] == [
        "s-evidence",
        "s-pinned",
    ]
    assert [row.id for row in service.virtual_folder("diagnostics")] == ["s-diag"]
    assert [row.id for row in service.virtual_folder("superseded")] == [
        "s-superseded"
    ]
    assert [row.id for row in service.virtual_folder("cold-storage")] == [
        "s-archived"
    ]
    with pytest.raises(KeyError, match="unknown virtual folder"):
        service.virtual_folder("nope")


def _snapshot_files(root: Path) -> dict[Path, bytes]:
    return {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_migration_dry_run_writes_nothing(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(
        project,
        "session-legacy",
        legacy_metadata={"alias": "old-run", "note": "from v1"},
    )
    _session(project, "session-plain")
    _session(project, "session-annotated")
    (project.session_root / "2026" / "08" / "session-annotated"
     / "session_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": project.project_id,
                "session_id": "session-annotated",
                "revision": 0,
            }
        ),
        encoding="utf-8",
    )
    before = _snapshot_files(project.root)

    report = CatalogService(project, home=tmp_path / "home").migration_dry_run()

    assert _snapshot_files(project.root) == before
    assert not (root / "session_annotations.json").exists()
    assert not (project.root / ".qphase").exists()
    assert report.sessions_total == 3
    assert [item.session_id for item in report.legacy_metadata_imports] == [
        "session-legacy"
    ]
    assert report.untouched_sessions == 1


def test_migration_dry_run_preview_matches_real_seed(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(
        project,
        "session-legacy",
        legacy_metadata={"alias": "old-run", "note": "from v1"},
    )
    service = CatalogService(project, home=tmp_path / "home")

    (item,) = service.migration_dry_run().legacy_metadata_imports

    seeded = ProjectService(project).new_session_annotations(root, "session-legacy")
    assert item.alias == seeded.alias == "old-run"
    assert item.note == seeded.note == "from v1"
    assert item.path == "2026/08/session-legacy"


def test_migration_dry_run_lists_invalid_snapshot_tags(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    (root / "workflow_snapshot.yaml").write_text(
        "schema: qphase.workflow/2\n"
        "id: example\n"
        "tags:\n"
        "  - vdp_2mode\n"
        "  - task:scan\n"
        "jobs:\n"
        "  - name: sim\n"
        "    tags:\n"
        "      - Cam\n"
        "      - method:cam\n",
        encoding="utf-8",
    )
    service = CatalogService(project, home=tmp_path / "home")

    report = service.migration_dry_run()

    invalid = {(item.tag, item.source) for item in report.invalid_snapshot_tags}
    assert invalid == {("vdp_2mode", "workflow"), ("Cam", "sim")}
    assert all(
        item.session_id == "session-1" for item in report.invalid_snapshot_tags
    )


def test_tag_artifact_never_touches_manifest_or_payload(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    artifact_dir = root / "sim"
    (artifact_dir / "payload.bin").write_bytes(b"\x00\x01payload")
    manifest_before = (artifact_dir / "artifact_manifest.json").read_bytes()
    payload_before = (artifact_dir / "payload.bin").read_bytes()
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_artifact("art-1", add=["method:cam"])
    service.set_artifact_lifecycle("art-1", "reference")

    assert (artifact_dir / "artifact_manifest.json").read_bytes() == manifest_before
    assert (artifact_dir / "payload.bin").read_bytes() == payload_before
    assert (artifact_dir / "artifact_annotations.json").exists()
