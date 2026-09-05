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


@pytest.fixture
def sde_profile_plugins():
    """Register the real SDE engine plus dummy plugins for its namespaces."""
    from qphase_sde.engine import Engine

    from tests.plugins.dummy_plugin import DummyPlugin

    engine_table = registry._tables.setdefault("engine", {})
    previous_sde = engine_table.get("sde")
    registry.register("engine", "sde", Engine, overwrite=True)
    registry.register("integrator", "dummy", DummyPlugin, overwrite=True)
    registry.register("analyser", "dummy", DummyPlugin, overwrite=True)
    try:
        yield
    finally:
        if previous_sde is None:
            engine_table.pop("sde", None)
        else:
            engine_table["sde"] = previous_sde
        registry._tables.get("integrator", {}).pop("dummy", None)
        registry._tables.get("analyser", {}).pop("dummy", None)


@pytest.fixture
def profiled_engine():
    """Register the fixture engine declaring conditional task profiles."""
    from tests.plugins.dummy_plugin import DummyPlugin
    from tests.plugins.profiled_plugins import ProfiledEngine

    registry.register("engine", "profiled", ProfiledEngine, overwrite=True)
    registry.register("observer", "dummy", DummyPlugin, overwrite=True)
    try:
        yield
    finally:
        registry._tables.get("engine", {}).pop("profiled", None)
        registry._tables.get("observer", {}).pop("dummy", None)


def _sde_job(name: str, engine_config: dict, **sections) -> JobConfig:
    return JobConfig(name=name, engine={"sde": engine_config}, **sections)


def test_compiler_selects_sde_task_profiles(tmp_path, sde_profile_plugins):
    """The SDE engine selects simulate/simulate_analyze/analyze from config."""
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    simulate = compiler.compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
            )
        )
    ).job("sim")
    assert simulate.required_plugins == ("backend", "integrator", "model")
    assert simulate.optional_plugins == ("analyser", "observer")

    simulate_analyze = compiler.compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
                analyser={"dummy": {}},
            )
        )
    ).job("sim")
    assert simulate_analyze.required_plugins == (
        "analyser",
        "backend",
        "integrator",
        "model",
    )
    assert simulate_analyze.optional_plugins == ("observer",)

    analyze = compiler.compile(
        _workflow(
            _sde_job("ana", {"mode": "analyze"}, analyser={"dummy": {}}),
        )
    ).job("ana")
    assert analyze.required_plugins == ("analyser",)
    assert analyze.optional_plugins == ("backend",)
    assert set(analyze.plugin_config) == {"analyser"}


def test_compiler_sde_analyze_profile_forbids_model(tmp_path, sde_profile_plugins):
    """The analyze profile turns configured model/integrator into errors."""
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError) as excinfo:
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(
                _sde_job(
                    "ana",
                    {"mode": "analyze"},
                    analyser={"dummy": {}},
                    model={"dummy": {}},
                )
            )
        )
    message = str(excinfo.value)
    assert "'analyze'" in message
    assert "forbidden plugins" in message
    assert "model" in message


def test_compiler_sde_profile_reports_missing_required(tmp_path, sde_profile_plugins):
    """Missing required plugins name the job, engine and resolved profile."""
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    with pytest.raises(QPhaseConfigError, match="missing required plugins"):
        compiler.compile(_workflow(_sde_job("ana", {"mode": "analyze"})))
    with pytest.raises(QPhaseConfigError) as excinfo:
        compiler.compile(
            _workflow(
                _sde_job(
                    "sim",
                    {"t1": 1.0, "dt": 0.1},
                    backend={"dummy": {}},
                    integrator={"dummy": {}},
                )
            )
        )
    message = str(excinfo.value)
    assert "'simulate'" in message
    assert "model" in message


def test_compiler_profile_resolver_replaces_defaults(tmp_path, profiled_engine):
    """A profile resolver refines requirements through the dotted-path hook."""
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    compiled = compiler.compile(
        _workflow(
            JobConfig(
                name="job",
                engine={"profiled": {}},
                backend={"dummy": {}},
                model={"dummy": {}},
            )
        )
    )
    # The 'base' profile defaults require only backend; its resolver adds
    # model and forbids observer.
    assert compiled.job("job").required_plugins == ("backend", "model")

    with pytest.raises(QPhaseConfigError, match="task profile 'base'"):
        compiler.compile(
            _workflow(
                JobConfig(name="job", engine={"profiled": {}}, backend={"dummy": {}})
            )
        )

    with pytest.raises(QPhaseConfigError) as excinfo:
        compiler.compile(
            _workflow(
                JobConfig(
                    name="job",
                    engine={"profiled": {}},
                    backend={"dummy": {}},
                    model={"dummy": {}},
                    observer={"dummy": {}},
                )
            )
        )
    assert "forbidden plugins" in str(excinfo.value)
    assert "observer" in str(excinfo.value)


def test_compiler_profile_selector_switches_profiles(tmp_path, profiled_engine):
    """The selector maps the normalized job config onto the profile id."""
    project = ProjectContext.create(tmp_path / "project")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            JobConfig(
                name="job",
                engine={"profiled": {"mode": "resolved"}},
                backend={"dummy": {}},
                observer={"dummy": {}},
            )
        )
    )
    job = compiled.job("job")
    assert job.required_plugins == ("backend",)
    assert job.optional_plugins == ("observer",)
    assert "observer" in job.plugin_config


def test_compiler_legacy_manifest_engines_keep_legacy_validation(tmp_path):
    """Engines without task profiles keep the legacy manifest checks."""

    class Config(BaseModel):
        name: str = "legacy"

    class LegacyEngine:
        name: ClassVar[str] = "legacy"
        description: ClassVar[str] = "legacy manifest engine"
        config_schema: ClassVar[type[Config]] = Config
        manifest: ClassVar[EngineManifest] = EngineManifest(required_plugins={"model"})

        def __init__(self, config=None, plugins=None, **kwargs):
            del config, plugins, kwargs

    registry.register("engine", "legacy", LegacyEngine, overwrite=True)
    try:
        project = ProjectContext.create(tmp_path / "project")
        compiler = WorkflowCompiler(project, SystemConfig())

        with pytest.raises(QPhaseConfigError) as excinfo:
            compiler.compile(_workflow(JobConfig(name="job", engine={"legacy": {}})))
        assert "missing required plugins" in str(excinfo.value)
        assert "task profile" not in str(excinfo.value)

        compiled = compiler.compile(
            _workflow(
                JobConfig(
                    name="job",
                    engine={"legacy": {}},
                    model={"dummy": {"param": 1.0}},
                )
            )
        )
        assert compiled.job("job").required_plugins == ("model",)
    finally:
        registry._tables.get("engine", {}).pop("legacy", None)


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


def _tagged_workflow(workflow_tags: list[str], job_tags: list[str]) -> WorkflowSpec:
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


def test_compiler_freezes_revision_scoped_ids_and_rules(tmp_path):
    from qphase.core.tags import (
        job_tag_assignment_id,
        workflow_tag_assignment_id,
    )
    from qphase.core.workflow import workflow_revision

    project = ProjectContext.create(tmp_path / "project")
    _write_tag_policy(project)
    workflow = _tagged_workflow(["task:scan"], ["model:vdp"])

    compiled = WorkflowCompiler(project, SystemConfig()).compile(workflow)

    snapshot = compiled.tag_snapshot
    assert snapshot is not None
    revision = workflow_revision(workflow.model_dump(mode="json", by_alias=True))
    assert snapshot["workflow_revision"] == revision
    (workflow_entry,) = snapshot["assignments"]["workflow"]
    assert workflow_entry["assignment_id"] == workflow_tag_assignment_id(
        "tagged", revision, "task:scan"
    )
    # An open namespace with default governance freezes the default rule.
    assert workflow_entry["inherit"] is True
    assert workflow_entry["cardinality"] == "many"
    assert workflow_entry["objects"] == []
    (job_entry,) = snapshot["assignments"]["jobs"]["sim"]
    assert job_entry["assignment_id"] == job_tag_assignment_id(
        "tagged", revision, "sim", "model:vdp"
    )
    assert job_entry["cardinality"] == "one"

    # A workflow edit derives a new revision and fresh assignment ids.
    edited = workflow.model_copy(update={"title": "Tagged v2"})
    recompiled = WorkflowCompiler(project, SystemConfig()).compile(edited)
    assert recompiled.tag_snapshot is not None
    assert recompiled.tag_snapshot["workflow_revision"] != revision
    assert (
        recompiled.tag_snapshot["assignments"]["workflow"][0]["assignment_id"]
        != workflow_entry["assignment_id"]
    )
