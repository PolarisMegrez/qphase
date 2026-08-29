"""CatalogService mutation roundtrips and policy enforcement."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml
from qphase.core.catalog import CatalogQuery
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.service import CatalogService
from qphase.service.project import ProjectService

from tests.qphase.test_catalog import _v4_artifact_manifest, _workflow_file

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
            json.dumps(_v4_artifact_manifest(artifact_id)),
            encoding="utf-8",
        )
    if legacy_metadata is not None:
        (root / "session_metadata.json").write_text(
            json.dumps(legacy_metadata), encoding="utf-8"
        )
    return root


def _write_policy(project: ProjectContext, body: str) -> None:
    (project.defaults_path.parent / "tags.yaml").write_text(body, encoding="utf-8")


def _legacy_workflow_snapshot(root: Path) -> Path:
    path = root / "workflow_snapshot.yaml"
    path.write_text(
        "schema: qphase.workflow/2\n"
        "id: example\n"
        "title: Example\n"
        "tags: [legacy_model]\n"
        "jobs:\n"
        "  - name: sim\n"
        "    engine: {dummy: {}}\n"
        "    tags: [temporary]\n",
        encoding="utf-8",
    )
    return path


def test_metadata_migration_freezes_tags_and_session_policy(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    source = _legacy_workflow_snapshot(root)
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  model: {open: true}\n",
    )
    service = CatalogService(project, home=tmp_path / "home")
    manifest = {
        "schema": "qphase.phase4a-metadata-actions/1",
        "project_id": project.project_id,
        "external_snapshot": "snapshot-1",
        "actions": [
            {
                "action": "freeze_legacy_tag_snapshot",
                "session_id": "session-1",
                "session_path": root.relative_to(project.root).as_posix(),
                "expected_source_sha256": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
                "replacements": [
                    {"raw": "legacy_model", "canonical_tag": "model:legacy"},
                    {"raw": "temporary", "canonical_tag": None},
                ],
            },
            {
                "action": "set_session_policy",
                "session_id": "session-1",
                "session_path": root.relative_to(project.root).as_posix(),
                "lifecycle": "superseded",
                "retention": "transient",
            },
        ],
    }

    counts = service.apply_metadata_migration(manifest)

    assert counts == {"freeze_legacy_tag_snapshot": 1, "set_session_policy": 1}
    frozen = yaml.safe_load((root / "tag_snapshot.yaml").read_text(encoding="utf-8"))
    assert frozen["canonical_tags"] == ["model:legacy"]
    assert frozen["job_tags"] == {"sim": []}
    annotations = service.state_store.load_session_annotations(root)
    assert annotations["lifecycle"] == "superseded"
    assert annotations["retention"] == "transient"
    assert service.migration_dry_run().invalid_snapshot_tags == []


def test_metadata_migration_checks_all_actions_before_writing(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    _legacy_workflow_snapshot(root)
    service = CatalogService(project, home=tmp_path / "home")
    manifest = {
        "schema": "qphase.phase4a-metadata-actions/1",
        "project_id": project.project_id,
        "external_snapshot": "snapshot-1",
        "actions": [
            {
                "action": "set_session_policy",
                "session_id": "session-1",
                "session_path": root.relative_to(project.root).as_posix(),
                "lifecycle": "reference",
                "retention": "pinned",
            },
            {
                "action": "freeze_legacy_tag_snapshot",
                "session_id": "session-1",
                "session_path": root.relative_to(project.root).as_posix(),
                "expected_source_sha256": "wrong",
                "replacements": [],
            },
        ],
    }

    with pytest.raises(ValueError, match="workflow snapshot changed"):
        service.apply_metadata_migration(manifest)

    assert not (root / "tag_snapshot.yaml").exists()
    assert service.state_store.load_session_annotations(root) is None


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


def test_session_retention_freezes_inheritance_flag(tmp_path):
    from qphase.core.catalog import ProjectObjectCatalog

    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    service = CatalogService(project, home=tmp_path / "home")

    service.set_session_retention("session-1", "evidence")

    # Frozen at write time with the default (no policy yet: inherit).
    (row,) = service.query(CatalogQuery(object_kind="occurrence"))
    assert row.facets["effective_retention"] == "evidence"

    # A policy introduced later disabling inheritance must not rewrite the
    # historical session's frozen flag.
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nretention_inherits_to_occurrences: false\n",
    )
    ProjectObjectCatalog(project).reindex()
    (row,) = service.query(CatalogQuery(object_kind="occurrence"))
    assert row.facets["effective_retention"] == "evidence"

    # Clearing the retention clears the frozen flag with it.
    service.set_session_retention("session-1", None)
    (row,) = service.query(CatalogQuery(object_kind="occurrence"))
    assert row.facets["effective_retention"] is None


def test_legacy_retention_document_falls_back_to_current_policy(tmp_path):
    from qphase.core.catalog import ProjectObjectCatalog

    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    # Pre-freeze document: retention set, no inheritance flag recorded.
    (root / "session_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": project.project_id,
                "session_id": "session-1",
                "revision": 0,
                "retention": "evidence",
            }
        ),
        encoding="utf-8",
    )
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nretention_inherits_to_occurrences: false\n",
    )
    service = CatalogService(project, home=tmp_path / "home")
    ProjectObjectCatalog(project).reindex()

    (row,) = service.query(CatalogQuery(object_kind="occurrence"))
    assert row.facets["effective_retention"] is None


def test_migration_dry_run_counts_missing_retention_inheritance(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    (root / "session_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": project.project_id,
                "session_id": "session-1",
                "revision": 0,
                "retention": "evidence",
            }
        ),
        encoding="utf-8",
    )
    service = CatalogService(project, home=tmp_path / "home")

    report = service.migration_dry_run()
    assert report.sessions_missing_retention_inheritance == 1

    # Rewriting the retention through the service freezes the flag.
    service.set_session_retention("session-1", "evidence")
    report = service.migration_dry_run()
    assert report.sessions_missing_retention_inheritance == 0


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

    first = {
        tag.tag
        for tag in service.effective_tags("occurrence", "art-1:session-1:sim")
    }
    second = {
        tag.tag
        for tag in service.effective_tags("occurrence", "art-1:session-2:sim")
    }
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
    model_root = _session(project, "s-model")
    _session(project, "s-evidence")
    _session(project, "s-pinned")
    _session(project, "s-diag")
    _session(project, "s-superseded")
    _session(project, "s-archived")
    _session(project, "s-plain")
    # by-model follows the workflow revision's model plugin, not tags.
    snapshot = {
        "schema": "qphase.workflow/2",
        "id": "example",
        "title": "Example",
        "jobs": [
            {
                "name": "sim",
                "engine": {"dummy": {}},
                "plugins": {"model": {"cam": {}}},
            }
        ],
    }
    (model_root / "workflow_snapshot.yaml").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    service = CatalogService(project, home=tmp_path / "home")

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
    # A concrete model resolves through the ``model`` query filter.
    rows = service.query(CatalogQuery(object_kind="session", model="cam"))
    assert [row.id for row in rows] == ["s-model"]
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


def test_assignments_freeze_the_current_policy_revision(tmp_path):
    from qphase.core.tags import load_tag_policy

    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n  stage:\n    open: true\n",
    )
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["stage:q1"])
    revision_v1 = load_tag_policy(project).revision
    tags = {tag.tag: tag for tag in service.effective_tags("session", "session-1")}
    assert tags["stage:q1"].policy_revision == revision_v1

    # After a policy edit, the historical assignment keeps its provenance and
    # only new mutations freeze the new revision.
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n"
        "  stage:\n    open: true\n  task:\n    open: true\n",
    )
    revision_v2 = load_tag_policy(project).revision
    assert revision_v2 != revision_v1
    service.tag_session("session-1", add=["task:scan"])

    tags = {tag.tag: tag for tag in service.effective_tags("session", "session-1")}
    assert tags["stage:q1"].policy_revision == revision_v1
    assert tags["task:scan"].policy_revision == revision_v2


def test_occurrence_mutation_requires_job_name_when_ambiguous(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(
        project, "session-1", artifacts=(("sim", "art-1"), ("fit", "art-1"))
    )
    service = CatalogService(project, home=tmp_path / "home")

    with pytest.raises(ValueError, match="ambiguous occurrence"):
        service.tag_occurrence("session-1", "art-1", add=["purpose:draft"])
    with pytest.raises(ValueError, match="ambiguous occurrence"):
        service.set_occurrence_retention("session-1", "art-1", "pinned")

    service.tag_occurrence(
        "session-1", "art-1", job_name="fit", add=["purpose:draft"]
    )
    service.set_occurrence_retention("session-1", "art-1", "pinned", job_name="fit")

    sim_tags = {
        tag.tag
        for tag in service.effective_tags("occurrence", "art-1:session-1:sim")
    }
    fit_tags = {
        tag.tag
        for tag in service.effective_tags("occurrence", "art-1:session-1:fit")
    }
    assert "purpose:draft" not in sim_tags
    assert "purpose:draft" in fit_tags
    # The sidecar keys occurrences by job_name:artifact_id.
    document = json.loads((root / "session_annotations.json").read_text("utf-8"))
    assert set(document["occurrences"]) == {"fit:art-1"}


def test_artifact_mutation_rejects_multiple_locations(tmp_path):
    from qphase.data.errors import ArtifactAmbiguousError

    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    _session(project, "session-2", artifacts=(("sim", "art-1"),))
    service = CatalogService(project, home=tmp_path / "home")

    with pytest.raises(ArtifactAmbiguousError, match="2 locations"):
        service.tag_artifact("art-1", add=["method:cam"])
    with pytest.raises(ArtifactAmbiguousError):
        service.set_artifact_lifecycle("art-1", "archived")

    # Removing the second location restores a unique annotation target.
    service.project_service.trash_session("session-2")
    service.reindex()
    tags = service.tag_artifact("art-1", add=["method:cam"])
    assert [tag.tag for tag in tags] == ["method:cam"]


def test_location_issues_passthrough(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    (root / "sim" / "artifact_manifest.json").write_text(
        "{not json", encoding="utf-8"
    )
    service = CatalogService(project, home=tmp_path / "home")

    (issue,) = service.location_issues()
    assert issue["kind"] == "corrupt"
    assert issue["path"] == "2026/08/session-1/sim"
    assert issue["message"]


def test_private_tags_participate_in_tag_queries(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    _session(project, "session-2")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["task:wip"], private=True)

    rows = service.query(CatalogQuery(object_kind="session", tags_all=("task:wip",)))
    assert [row.id for row in rows] == ["session-1"]
    rows = service.query(
        CatalogQuery(object_kind="session", tags_without=("task:wip",))
    )
    assert [row.id for row in rows] == ["session-2"]
    # The private tag never entered the shared catalog: another user (another
    # home) with the same query sees nothing.
    other = CatalogService(project, home=tmp_path / "other-home")
    query = CatalogQuery(object_kind="session", tags_all=("task:wip",))
    assert other.query(query) == []


def test_private_query_respects_cardinality_one_shadowing(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  stage:\n"
        "    open: true\n"
        "    cardinality: one\n",
    )
    _session(project, "session-1")
    _session(project, "session-2")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["stage:q1"])
    service.tag_session("session-1", add=["stage:q2"], private=True)

    # The private assignment shadows the shared one in this user's view.
    assert service.query(
        CatalogQuery(object_kind="session", tags_all=("stage:q1",))
    ) == []
    rows = service.query(CatalogQuery(object_kind="session", tags_all=("stage:q2",)))
    assert [row.id for row in rows] == ["session-1"]


def test_saved_view_applies_private_tag_predicates(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    _session(project, "session-2")
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["task:wip"], private=True)
    service.save_view(
        "wip", CatalogQuery(object_kind="session", tags_all=("task:wip",))
    )

    views = dict(service.list_views())
    rows = service.query(views["wip"])
    assert [row.id for row in rows] == ["session-1"]


def _execution_record(project: ProjectContext, execution_id: str) -> None:
    from qphase.core.persistence import ProjectStateStore

    ProjectStateStore(project).save_execution(
        {
            "schema": "qphase.execution/1",
            "execution_id": execution_id,
            "source_workflow": "example",
            "workflow": {"id": "example"},
            "submission_tags": [],
            "submitted_at": "2026-08-26T09:00:00+08:00",
            "state": "completed",
        }
    )


def _revision_and_job(service: CatalogService) -> tuple[str, str]:
    service.reindex()
    (workflow_row,) = service.catalog.query(CatalogQuery(object_kind="workflow"))
    revision_id = str(workflow_row["id"])
    return revision_id, f"{revision_id}:sim"


def test_project_annotation_roundtrip(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    service = CatalogService(project, home=tmp_path / "home")

    tags = service.tag_project(add=["task:paper"])
    assert [tag.tag for tag in tags] == ["task:paper"]
    service.set_project_alias("paper project")
    service.set_project_note("results for the paper")

    document = service.project_annotations()
    assert document.alias == "paper project"
    assert document.note == "results for the paper"
    assert [assignment.tag for assignment in document.assignments] == ["task:paper"]
    rows = service.query(CatalogQuery(object_kind="project", tags_all=("task:paper",)))
    assert [row.id for row in rows] == [project.project_id]


def test_workflow_job_execution_annotation_roundtrip(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _execution_record(project, "exec-1")
    service = CatalogService(project, home=tmp_path / "home")
    revision_id, job_id = _revision_and_job(service)

    workflow_tags = service.tag_workflow(revision_id, add=["task:reviewed"])
    assert {tag.tag: tag.source for tag in workflow_tags} == {
        "task:reviewed": "workflow_annotation"
    }
    job_tags = service.tag_job(job_id, add=["method:cam"])
    assert {tag.tag: tag.source for tag in job_tags} == {"method:cam": "job_annotation"}
    execution_tags = service.tag_execution("exec-1", add=["task:rerun"])
    assert {tag.tag: tag.source for tag in execution_tags} == {
        "task:rerun": "execution_annotation"
    }

    # The annotations live in the project document with provenance.
    document = service.project_annotations()
    objects = document.objects
    assert [a.tag for a in objects[revision_id].assignments] == ["task:reviewed"]
    assert [a.tag for a in objects[job_id].assignments] == ["method:cam"]
    assert [a.tag for a in objects["exec-1"].assignments] == ["task:rerun"]

    # Removal roundtrip.
    assert service.tag_workflow(revision_id, remove=["task:reviewed"]) == []


def test_tag_workflow_job_execution_reject_unknown_objects(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    service = CatalogService(project, home=tmp_path / "home")
    with pytest.raises(ValueError, match="unknown workflow"):
        service.tag_workflow("example@deadbeef", add=["task:x"])
    with pytest.raises(ValueError, match="unknown job"):
        service.tag_job("example@deadbeef:sim", add=["task:x"])
    with pytest.raises(ValueError, match="unknown execution"):
        service.tag_execution("exec-nope", add=["task:x"])


def test_private_tags_overlay_and_promote_on_new_kinds(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _execution_record(project, "exec-1")
    service = CatalogService(project, home=tmp_path / "home")
    revision_id, job_id = _revision_and_job(service)

    service.tag_project(add=["task:mine"], private=True)
    service.tag_workflow(revision_id, add=["task:mine"], private=True)
    service.tag_job(job_id, add=["task:mine"], private=True)
    service.tag_execution("exec-1", add=["task:mine"], private=True)

    targets = (
        ("project", project.project_id),
        ("workflow", revision_id),
        ("job", job_id),
        ("execution", "exec-1"),
    )
    for kind, object_id in targets:
        mine = [
            tag
            for tag in service.effective_tags(kind, object_id)
            if tag.tag == "task:mine"
        ]
        assert [tag.source for tag in mine] == ["user_private"]
    # Private tags never enter the shared project document.
    assert service.project_annotations().objects == {}
    assert service.project_annotations().assignments == []

    expected_sources = {
        "project": "project_annotation",
        "workflow": "workflow_annotation",
        "job": "job_annotation",
        "execution": "execution_annotation",
    }
    for kind, object_id in targets:
        promoted = service.promote_tag(kind, object_id, "task:mine")
        mine = [tag for tag in promoted if tag.tag == "task:mine"]
        assert [tag.source for tag in mine] == [expected_sources[kind]]
        assert service.private.list_private_tags(kind, object_id) == []


def test_migration_dry_run_occurrence_key_preview(tmp_path):
    """Legacy bare-artifact occurrence keys are classified for conversion."""
    project = ProjectContext.create(tmp_path / "project")
    root = _session(
        project,
        "session-1",
        artifacts=(("sim", "art-1"), ("fit", "art-2"), ("fit2", "art-2")),
    )
    (root / "session_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": project.project_id,
                "session_id": "session-1",
                "revision": 0,
                "occurrences": {
                    "art-1": {"assignments": []},
                    "art-2": {"assignments": []},
                    "sim:art-1": {"assignments": []},
                },
            }
        ),
        encoding="utf-8",
    )

    report = CatalogService(project, home=tmp_path / "home").migration_dry_run()

    assert [
        (item.old_key, item.new_key) for item in report.convertible_occurrence_keys
    ] == [("art-1", "sim:art-1")]
    (ambiguous,) = report.ambiguous_occurrence_keys
    assert ambiguous.session_id == "session-1"
    assert ambiguous.old_key == "art-2"
    assert ambiguous.locations == ["fit", "fit2"]


def test_migration_dry_run_duplicate_artifacts(tmp_path):
    """One artifact identity at two locations is listed; facet drift conflicts."""
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1", artifacts=(("sim", "art-dup"),))
    second = _session(project, "session-2", artifacts=(("sim", "art-dup"),))
    # Same id but divergent identity facets -> conflict location issue.
    manifest_path = second / "sim" / "artifact_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["created_at"] = "2026-08-27T00:00:00+08:00"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = CatalogService(project, home=tmp_path / "home").migration_dry_run()

    (duplicate,) = report.duplicate_artifacts
    assert duplicate.artifact_id == "art-dup"
    assert duplicate.locations == [
        "2026/08/session-1/sim",
        "2026/08/session-2/sim",
    ]
    assert duplicate.conflict
    assert report.location_issues_by_kind == {"conflict": 1}


def test_migration_dry_run_provenance_counts(tmp_path):
    """Assignments frozen without a policy revision are counted per scope."""
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    (root / "session_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.session-annotations/1",
                "project_id": project.project_id,
                "session_id": "session-1",
                "revision": 0,
                "assignments": [
                    {"tag": "task:a", "policy_revision": "rev-1"},
                    {"tag": "task:b"},
                ],
                "occurrences": {"sim:art-1": {"assignments": [{"tag": "task:c"}]}},
            }
        ),
        encoding="utf-8",
    )
    (root / "sim" / "artifact_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.artifact-annotations/1",
                "project_id": project.project_id,
                "artifact_id": "art-1",
                "revision": 0,
                "assignments": [{"tag": "method:cam"}],
            }
        ),
        encoding="utf-8",
    )
    qphase_dir = project.root / ".qphase"
    qphase_dir.mkdir(exist_ok=True)
    (qphase_dir / "project_annotations.json").write_text(
        json.dumps(
            {
                "schema": "qphase.project-annotations/1",
                "project_id": project.project_id,
                "revision": 0,
                "assignments": [{"tag": "task:p"}],
                "objects": {
                    "example@deadbeef": {"assignments": [{"tag": "task:w"}]},
                    "example@deadbeef:sim": {"assignments": [{"tag": "task:j"}]},
                    "exec-1": {"assignments": [{"tag": "task:e"}]},
                },
            }
        ),
        encoding="utf-8",
    )

    report = CatalogService(project, home=tmp_path / "home").migration_dry_run()

    assert report.assignments_without_policy_revision == {
        "session": 1,
        "occurrence": 1,
        "artifact": 1,
        "project": 1,
        "workflow": 1,
        "job": 1,
        "execution": 1,
    }


def test_migration_dry_run_reindex_parity_and_zero_writes(tmp_path):
    """Parity reports absent, then in sync; the project gains no files."""
    project = ProjectContext.create(tmp_path / "project")
    _workflow_file(project)
    _session(project, "session-1", artifacts=(("sim", "art-1"),))
    service = CatalogService(project, home=tmp_path / "home")

    report = service.migration_dry_run()

    assert report.catalog_drift is None
    assert report.rebuildable_workflow_revisions == 1
    assert report.rebuildable_jobs == 1
    assert report.object_counts == {
        "project": 1,
        "workflow": 1,
        "job": 1,
        "execution": 0,
        "session": 1,
        "artifact": 1,
        "occurrence": 1,
    }

    service.reindex()
    before = _snapshot_files(project.root)
    report = service.migration_dry_run()
    assert report.catalog_drift is False
    assert report.catalog_drift_tables == {}
    assert _snapshot_files(project.root) == before


def test_migration_dry_run_detects_catalog_drift(tmp_path):
    """Disk truth newer than the on-disk catalog is reported as drift."""
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")
    service.reindex()
    _session(project, "session-2")  # disk truth moved; catalog not rebuilt

    report = service.migration_dry_run()

    assert report.catalog_drift is True


def test_catalog_fresh_after_direct_session_alias_write(tmp_path):
    """A ProjectService alias write (no service reindex) shows in the next query."""
    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")
    (row,) = service.query(CatalogQuery(object_kind="session"))
    assert row.facets["alias"] is None

    ProjectService(project).update_session("session-1", alias="direct-alias")

    (row,) = service.query(CatalogQuery(object_kind="session"))
    assert row.facets["alias"] == "direct-alias"


def test_catalog_fresh_after_project_move(tmp_path):
    """A moved project (same project_id) refreshes the root facet."""
    import shutil

    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")
    (row,) = service.query(CatalogQuery(object_kind="project"))
    assert row.facets["root"] == str(project.root)

    moved_root = tmp_path / "moved"
    shutil.copytree(project.root, moved_root)
    moved = ProjectContext.load(moved_root)
    moved_service = CatalogService(moved, home=tmp_path / "home")

    (row,) = moved_service.query(CatalogQuery(object_kind="project"))

    assert row.id == project.project_id
    assert row.facets["root"] == str(moved.root)


def test_migration_dry_run_locates_drift_table(tmp_path):
    """A tampered catalog row is drift, attributed to its table."""
    import sqlite3

    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")
    service.reindex()
    connection = sqlite3.connect(service.catalog.path)
    with connection:
        connection.execute(
            "UPDATE sessions SET alias = 'tampered' WHERE id = 'session-1'"
        )
    connection.close()

    report = service.migration_dry_run()

    assert report.catalog_drift is True
    assert report.catalog_drift_tables == {"sessions": 2}


def test_migration_dry_run_detects_duplicate_rows(tmp_path):
    """A duplicated catalog row is drift even under set semantics."""
    import sqlite3

    project = ProjectContext.create(tmp_path / "project")
    _session(project, "session-1")
    service = CatalogService(project, home=tmp_path / "home")
    service.tag_session("session-1", add=["task:scan"])
    connection = sqlite3.connect(service.catalog.path)
    with connection:
        row = connection.execute(
            "SELECT * FROM effective_tags LIMIT 1"
        ).fetchone()
        connection.execute(
            "INSERT INTO effective_tags VALUES (?, ?, ?, ?, ?, ?, ?, ?)", row
        )
    connection.close()

    report = service.migration_dry_run()

    assert report.catalog_drift is True
    assert report.catalog_drift_tables == {"effective_tags": 1}



def test_assignments_freeze_the_namespace_rule(tmp_path):
    from qphase.core.catalog import ProjectObjectCatalog

    project = ProjectContext.create(tmp_path / "project")
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n  stage:\n    open: true\n",
    )
    root = _session(project, "session-1", artifacts=(("sim", "art-1"),))
    service = CatalogService(project, home=tmp_path / "home")

    service.tag_session("session-1", add=["stage:q1"])

    document = json.loads(
        (root / "session_annotations.json").read_text(encoding="utf-8")
    )
    (assignment,) = document["assignments"]
    assert assignment["inherit"] is True
    assert assignment["cardinality"] == "many"
    assert assignment["objects"] == []

    catalog = ProjectObjectCatalog(project)
    catalog.reindex()
    occurrence_tags = catalog.effective_tags("occurrence", "art-1:session-1:sim")
    assert [tag.tag for tag in occurrence_tags] == ["stage:q1"]

    # Disabling inheritance in the policy does not rewrite the frozen rule:
    # the occurrence still inherits the historical assignment.
    _write_policy(
        project,
        "schema: qphase.tag-policy/1\nnamespaces:\n"
        "  stage:\n    open: true\n    inherit: false\n",
    )
    catalog.reindex()
    occurrence_tags = catalog.effective_tags("occurrence", "art-1:session-1:sim")
    assert [tag.tag for tag in occurrence_tags] == ["stage:q1"]

    # A legacy assignment without frozen rule fields falls back to the
    # current policy and stops inheriting.
    for entry in document["assignments"]:
        for key in ("inherit", "cardinality", "objects"):
            entry.pop(key, None)
    (root / "session_annotations.json").write_text(
        json.dumps(document), encoding="utf-8"
    )
    catalog.reindex()
    assert catalog.effective_tags("occurrence", "art-1:session-1:sim") == []



def test_migration_dry_run_lists_id_separator_violations(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1", artifacts=(("sim", "art:1"),))
    snapshot = {
        "schema": "qphase.workflow/2",
        "id": "example",
        "title": "Example",
        "jobs": [{"name": "sim:bad", "engine": {"dummy": {}}}],
    }
    (root / "workflow_snapshot.yaml").write_text(
        json.dumps(snapshot), encoding="utf-8"
    )
    service = CatalogService(project, home=tmp_path / "home")

    report = service.migration_dry_run()

    violations = {
        (item.object_kind, item.value) for item in report.id_separator_violations
    }
    assert ("job", "sim:bad") in violations
    assert ("artifact", "art:1") in violations



def test_private_query_paginates_candidates_in_batches(tmp_path, monkeypatch):
    project = ProjectContext.create(tmp_path / "project")
    for index in range(5):
        _session(project, f"session-{index}")
    service = CatalogService(project, home=tmp_path / "home")
    service.tag_session("session-3", add=["task:scan"], private=True)
    service.tag_session("session-4", add=["task:scan"], private=True)

    from qphase.service import catalog as catalog_module

    # Three candidate pages of two, each with one batched tag load.
    monkeypatch.setattr(catalog_module, "_PRIVATE_QUERY_PAGE_SIZE", 2)
    batches: list[int] = []
    original = service.catalog.effective_tags_for_objects

    def spy(kind, ids):
        ids = list(ids)
        batches.append(len(ids))
        return original(kind, ids)

    monkeypatch.setattr(service.catalog, "effective_tags_for_objects", spy)

    rows = service.query(CatalogQuery(object_kind="session", tags_all=("task:scan",)))
    assert [row.id for row in rows] == ["session-3", "session-4"]
    assert batches == [2, 2, 1]
    # The caller's offset/limit apply after the merged filtering.
    rows = service.query(
        CatalogQuery(
            object_kind="session", tags_all=("task:scan",), offset=1, limit=1
        )
    )
    assert [row.id for row in rows] == ["session-4"]



def test_migration_dry_run_lists_invalid_annotations(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    root = _session(project, "session-1")
    (root / "session_annotations.json").write_text("{not json", encoding="utf-8")
    service = CatalogService(project, home=tmp_path / "home")

    report = service.migration_dry_run()

    assert [item.path for item in report.invalid_annotations] == [
        "2026/08/session-1/session_annotations.json"
    ]
    assert "invalid annotation document" in report.invalid_annotations[0].error
