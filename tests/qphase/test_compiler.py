"""Long-lived tests for the Phase 2 logical workflow compiler."""

from __future__ import annotations

from typing import ClassVar

import pytest
from pydantic import BaseModel
from qphase.core.compiler import CompiledWorkflow, WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseConfigError
from qphase.core.project import ProjectContext
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig

pytestmark = pytest.mark.integration


def _workflow(*jobs: JobConfig) -> WorkflowSpec:
    return WorkflowSpec(
        schema_="qphase.workflow/2",
        id="compiler-test",
        title="Compiler test",
        jobs=list(jobs),
    )


def test_compiler_resolves_scan_and_topological_order(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    workflow = _workflow(
        JobConfig(
            name="source",
            engine={"dummy": {"param": 1.0}},
            model={"dummy": {"param": 1.0}},
            scan={
                "axes": {
                    "rate": {
                        "target": "model.dummy.param",
                        "values": [1.0, 2.0],
                    }
                }
            },
        ),
        JobConfig(
            name="sink",
            engine={"dummy": {"param": 2.0}},
            input={"from": "source", "mode": "dataset"},
        ),
    )

    compiled = WorkflowCompiler(project, SystemConfig()).compile(workflow)

    assert compiled.project_id == project.project_id
    assert compiled.topological_order == ("source", "sink")
    assert compiled.job("source").parameter_grid is not None
    assert compiled.job("source").parameter_grid.shape == (2,)
    assert compiled.job("sink").input_source == "source"


def test_compiler_rejects_duplicate_and_cyclic_jobs(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    with pytest.raises(QPhaseConfigError, match="duplicate job names"):
        compiler.compile(
            _workflow(
                JobConfig(name="same", engine={"dummy": {}}),
                JobConfig(name="same", engine={"dummy": {}}),
            )
        )

    with pytest.raises(QPhaseConfigError, match="cycle"):
        compiler.compile(
            _workflow(
                JobConfig(
                    name="left",
                    engine={"dummy": {}},
                    depends_on=["right"],
                ),
                JobConfig(
                    name="right",
                    engine={"dummy": {}},
                    depends_on=["left"],
                ),
            )
        )


def test_compiler_rejects_missing_external_input(tmp_path):
    project = ProjectContext.create(tmp_path / "project")

    with pytest.raises(QPhaseConfigError, match="existing external path"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(
                JobConfig(
                    name="job",
                    engine={"dummy": {}},
                    input={"from": "missing.data", "mode": "dataset"},
                )
            )
        )


def test_compiler_does_not_instantiate_engine(tmp_path):
    class Config(BaseModel):
        name: str = "counted"

    class CountedEngine:
        name: ClassVar[str] = "counted"
        description: ClassVar[str] = "compiler test engine"
        config_schema: ClassVar[type[Config]] = Config
        manifest: ClassVar[EngineManifest] = EngineManifest()
        constructed = 0

        def __init__(self, config=None, plugins=None, **kwargs):
            del config, plugins, kwargs
            type(self).constructed += 1

    registry.register("engine", "counted", CountedEngine, overwrite=True)
    try:
        project = ProjectContext.create(tmp_path / "project")
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(JobConfig(name="job", engine={"counted": {}}))
        )
        assert CountedEngine.constructed == 0
    finally:
        registry._tables.get("engine", {}).pop("counted", None)


def test_compiled_workflow_round_trips_without_registry_access(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    workflow = _workflow(
        JobConfig(
            name="job",
            engine={"dummy": {"param": 2.0}},
            model={"dummy": {"param": 3.0}},
            scan={
                "axes": {
                    "value": {
                        "target": "model.dummy.param",
                        "values": [1, 2],
                    }
                }
            },
        )
    )
    compiled = WorkflowCompiler(project, SystemConfig()).compile(workflow)

    restored = CompiledWorkflow.from_payload(compiled.to_payload())

    assert restored.project_id == compiled.project_id
    assert restored.topological_order == ("job",)
    assert restored.job("job").parameter_grid is not None
    assert restored.job("job").parameter_grid.shape == (2,)


def _write_tag_policy(project: ProjectContext) -> None:
    path = project.defaults_path.parent / "tags.yaml"
    path.write_text(
        "schema: qphase.tag-policy/1\n"
        "namespaces:\n"
        "  task:\n"
        "    open: true\n"
        "  model:\n"
        "    cardinality: one\n"
        "    values: [vdp, kerr]\n",
        encoding="utf-8",
    )


def _tagged_workflow(
    workflow_tags: list[str], job_tags: list[str]
) -> WorkflowSpec:
    return WorkflowSpec(
        schema_="qphase.workflow/2",
        id="tagged",
        title="Tagged",
        tags=workflow_tags,
        jobs=[JobConfig(name="sim", engine={"dummy": {}}, tags=job_tags)],
    )


def test_compiler_freezes_declared_tags_without_policy(tmp_path):
    project = ProjectContext.create(tmp_path / "project")

    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _tagged_workflow(["Task:Scan"], ["Model:VDP"])
    )

    snapshot = compiled.tag_snapshot
    assert snapshot is not None
    assert snapshot["raw_tags"] == ["Task:Scan"]
    assert snapshot["canonical_tags"] == ["task:scan"]
    assert snapshot["job_tags"] == {"sim": ["model:vdp"]}
    assert snapshot["policy_revision"] is None

    restored = CompiledWorkflow.from_payload(compiled.to_payload())
    assert restored.tag_snapshot is not None
    assert dict(restored.tag_snapshot) == dict(compiled.tag_snapshot)


def test_compiler_validates_declared_tags_against_policy(tmp_path):
    project = ProjectContext.create(tmp_path / "project")
    _write_tag_policy(project)
    compiler = WorkflowCompiler(project, SystemConfig())

    compiled = compiler.compile(_tagged_workflow(["task:scan"], ["model:vdp"]))

    assert compiled.tag_snapshot is not None
    assert compiled.tag_snapshot["policy_revision"] is not None
    assert compiled.tag_snapshot["canonical_tags"] == ["task:scan"]

    with pytest.raises(QPhaseConfigError, match="unknown tag namespace"):
        compiler.compile(_tagged_workflow(["unknown:value"], []))

    with pytest.raises(QPhaseConfigError, match="one tag per object"):
        compiler.compile(_tagged_workflow([], ["model:vdp", "model:kerr"]))

    with pytest.raises(QPhaseConfigError, match="not an allowed value"):
        compiler.compile(_tagged_workflow([], ["model:other"]))
