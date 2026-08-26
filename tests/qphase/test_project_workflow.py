from pathlib import Path

import pytest
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.core.workflow import WorkflowCatalog, load_workflow


def _workflow(path: Path, workflow_id: str = "example") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "schema: qphase.workflow/2\n"
        f"id: {workflow_id}\n"
        "title: Example Workflow\n"
        "jobs:\n"
        "  - name: example\n"
        "    engine:\n"
        "      dummy: {}\n",
        encoding="utf-8",
    )


def test_project_discovery_walks_parents_and_resolves_portable_paths(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project", name="Example")
    nested = project.root / "notebooks" / "analysis"
    nested.mkdir(parents=True)

    discovered = ProjectContext.discover(nested, environ={})

    assert discovered.project_id == project.project_id
    assert discovered.workflow_root == project.root / "configs" / "workflows"
    assert discovered.session_root == project.root / "runs"


def test_explicit_project_environment_takes_precedence(tmp_path: Path):
    first = ProjectContext.create(tmp_path / "first")
    second = ProjectContext.create(tmp_path / "second")

    discovered = ProjectContext.discover(
        first.root, environ={"QPHASE_PROJECT": str(second.root)}
    )

    assert discovered.project_id == second.project_id


def test_project_manifest_rejects_paths_that_escape_root(tmp_path: Path):
    path = tmp_path / "qphase.toml"
    path.write_text(
        'schema = "qphase.project/2"\n'
        'project_id = "unsafe"\n'
        'name = "Unsafe"\n\n'
        "[paths]\n"
        'workflows = "../outside"\n'
        'defaults = "configs/defaults.yaml"\n'
        'plugins = ["models"]\n'
        'sessions = "runs"\n',
        encoding="utf-8",
    )

    with pytest.raises(QPhaseConfigError, match="relative paths"):
        ProjectContext.load(path)


def test_workflow_catalog_is_recursive_and_uses_stable_ids(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "models" / "example.yaml"
    _workflow(path, "stable-example")

    item = WorkflowCatalog(project).list()[0]

    assert item.id == "stable-example"
    assert item.relative_path == "models/example.yaml"
    assert WorkflowCatalog(project).resolve("stable-example").path == path


def test_workflow_catalog_rejects_duplicate_stable_ids(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project")
    _workflow(project.workflow_root / "a.yaml", "duplicate")
    _workflow(project.workflow_root / "group" / "b.yaml", "duplicate")

    with pytest.raises(QPhaseConfigError, match="Duplicate workflow id"):
        WorkflowCatalog(project).list()


def test_workflow_catalog_filters_collection_tag_and_query(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "models" / "example.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "schema: qphase.workflow/2\n"
        "id: stable-example\n"
        "title: Example Workflow\n"
        "collection: models\n"
        "tags:\n"
        "  - topic:quick\n"
        "  - engine:sde\n"
        "jobs:\n  - name: example\n    engine:\n      dummy: {}\n",
        encoding="utf-8",
    )
    catalog = WorkflowCatalog(project)

    assert [item.id for item in catalog.search(collection="models")] == [
        "stable-example"
    ]
    assert [item.id for item in catalog.search(tag="engine:sde")] == [
        "stable-example"
    ]
    assert [item.id for item in catalog.search(query="EXAMPLE")] == [
        "stable-example"
    ]
    assert catalog.search(tag="engine:cam") == []


def test_legacy_top_level_job_document_is_rejected(tmp_path: Path):
    path = tmp_path / "legacy.yaml"
    path.write_text("name: legacy\nengine:\n  dummy: {}\n", encoding="utf-8")

    with pytest.raises(QPhaseConfigError, match="qphase.workflow/2"):
        load_workflow(path)


def test_session_persists_workflow_snapshot_and_hash(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "example.yaml"
    _workflow(path)
    workflow = load_workflow(path)
    scheduler = Scheduler(system_config=SystemConfig(), project=project)

    scheduler.run(workflow)

    assert scheduler.session_dir is not None
    snapshot = scheduler.session_dir / "workflow_snapshot.yaml"
    manifest = scheduler.session_dir / "session_manifest.json"
    assert load_workflow(snapshot) == workflow
    assert '"workflow_hash"' in manifest.read_text(encoding="utf-8")


def test_session_persists_frozen_tag_snapshot(tmp_path: Path):
    """Sessions freeze the compiled tag snapshot as a sidecar."""
    import json

    from qphase.core.utils import load_yaml

    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "example.yaml"
    _workflow(path)
    workflow = load_workflow(path)
    scheduler = Scheduler(system_config=SystemConfig(), project=project)

    scheduler.run(workflow)

    assert scheduler.session_dir is not None
    frozen = load_yaml(scheduler.session_dir / "tag_snapshot.yaml")
    assert frozen["canonical_tags"] == []
    assert frozen["job_tags"] == {"example": []}
    assert frozen["policy_revision"] is None
    assert frozen["assignments"] == {"workflow": [], "jobs": {"example": []}}
    manifest = json.loads(
        (scheduler.session_dir / "session_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert "submission_tag_policy_revision" in manifest


def test_load_workflow_rejects_malformed_tags(tmp_path: Path):
    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "bad-tags.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "schema: qphase.workflow/2\n"
        "id: bad-tags\n"
        "title: Bad Tags\n"
        "tags: ['no-namespace']\n"
        "jobs:\n"
        "  - name: example\n"
        "    engine:\n"
        "      dummy: {}\n",
        encoding="utf-8",
    )

    with pytest.raises(QPhaseConfigError, match="namespace:value"):
        load_workflow(path)

    with pytest.raises(QPhaseConfigError, match="invalid tags"):
        WorkflowCatalog(project).list()


def test_session_manifest_records_submission_tags(tmp_path: Path):
    import json

    project = ProjectContext.create(tmp_path / "project")
    path = project.workflow_root / "example.yaml"
    _workflow(path)
    workflow = load_workflow(path)
    scheduler = Scheduler(system_config=SystemConfig(), project=project)

    scheduler.run(workflow, submission_tags=["task:urgent"])

    assert scheduler.session_dir is not None
    manifest = json.loads(
        (scheduler.session_dir / "session_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["submission_tags"] == ["task:urgent"]
