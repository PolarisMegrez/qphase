"""Long-lived tests for the Phase 2 logical workflow compiler."""

from __future__ import annotations

import json
from typing import ClassVar

import pytest
from pydantic import BaseModel
from qphase.core.compiler import CompiledWorkflow, WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseConfigError, QPhasePluginError
from qphase.core.project import ProjectContext
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig
from qphase.data import DataKind

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


@pytest.fixture
def recording_profile_engine():
    """Register the fixture engine whose resolver records its context."""
    from tests.plugins.profiled_plugins import RECORDED_CONTEXTS, RecordingProfileEngine

    RECORDED_CONTEXTS.clear()
    registry.register(
        "engine", "profiled_recording", RecordingProfileEngine, overwrite=True
    )
    try:
        yield
    finally:
        registry._tables.get("engine", {}).pop("profiled_recording", None)
        RECORDED_CONTEXTS.clear()


@pytest.fixture
def duplicate_profile_engine():
    """Register the fixture engine declaring a duplicate profile id."""
    from tests.plugins.profiled_plugins import DuplicateProfileEngine

    registry.register(
        "engine", "profiled_duplicate", DuplicateProfileEngine, overwrite=True
    )
    try:
        yield
    finally:
        registry._tables.get("engine", {}).pop("profiled_duplicate", None)


@pytest.fixture
def multi_input_profile_engine():
    """Register the fixture engine whose profile requires two input slots."""
    from tests.plugins.profiled_plugins import MultiInputProfileEngine

    registry.register(
        "engine", "profiled_multi_input", MultiInputProfileEngine, overwrite=True
    )
    try:
        yield
    finally:
        registry._tables.get("engine", {}).pop("profiled_multi_input", None)


@pytest.fixture
def typed_input_profile_engine():
    """Register the fixture engine whose profile requires a typed input."""
    from tests.plugins.profiled_plugins import TypedInputProfileEngine

    registry.register(
        "engine", "profiled_typed_input", TypedInputProfileEngine, overwrite=True
    )
    try:
        yield
    finally:
        registry._tables.get("engine", {}).pop("profiled_typed_input", None)


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
            _sde_job(
                "ana",
                {"mode": "analyze"},
                analyser={"dummy": {}},
                input={"from": str(tmp_path), "mode": "dataset"},
            ),
        )
    ).job("ana")
    assert analyze.required_plugins == ("analyser",)
    assert analyze.optional_plugins == ("backend",)
    assert set(analyze.plugin_config) == {"analyser"}


def test_compiler_normalizes_analyser_instance_labels(tmp_path, sde_profile_plugins):
    from qphase.core.job_runner import build_plugins
    from qphase.core.registry import discovery
    from qphase_sde.analyser.psd import PsdAnalyzer

    discovery.discover_plugins()
    project = ProjectContext.create(tmp_path / "project")
    project.defaults_path.write_text(
        "analyser:\n  psd:\n    convention: unitary\n",
        encoding="utf-8",
    )
    compiled_workflow = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
                analyser={
                    "raw_psd": {
                        "plugin": "psd",
                        "config": {"kind": "complex", "modes": [0]},
                    },
                    "other_psd": {
                        "plugin": "psd",
                        "config": {"kind": "complex", "modes": [0]},
                    },
                },
            )
        )
    )
    compiled = compiled_workflow.job("sim")

    expected = {
        "plugin": "psd",
        "config": {"kind": "complex", "modes": [0], "convention": "unitary"},
    }
    assert dict(compiled.plugin_config["analyser"]["raw_psd"]) == expected
    assert dict(compiled.plugin_config["analyser"]["other_psd"]) == expected
    assert dict(compiled.merged_config["analyser"]["raw_psd"]) == expected
    restored = CompiledWorkflow.from_payload(compiled_workflow.to_payload()).job("sim")
    assert dict(restored.plugin_config["analyser"]["raw_psd"]) == expected

    instances = build_plugins(registry, dict(compiled.plugin_config))["analyser"]
    assert set(instances) == {"raw_psd", "other_psd"}
    assert all(isinstance(instance, PsdAnalyzer) for instance in instances.values())
    assert compiled.product_graph is not None
    assert {node.node_id for node in compiled.product_graph.nodes} == {
        "analyser.raw_psd",
        "analyser.other_psd",
    }
    assert {node.producer for node in compiled.product_graph.nodes} == {"analyser:psd"}


def test_compiler_normalizes_analyser_shorthand(tmp_path, sde_profile_plugins):
    project = ProjectContext.create(tmp_path / "project")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
                analyser={"dummy": {"param": 2.0}},
            )
        )
    ).job("sim")

    assert dict(compiled.plugin_config["analyser"]["dummy"]) == {
        "plugin": "dummy",
        "config": {"param": 2.0},
    }


def test_compiler_rejects_unknown_analyser_implementation(
    tmp_path, sde_profile_plugins
):
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhasePluginError, match="missing_implementation"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(
                _sde_job(
                    "sim",
                    {"t1": 1.0, "dt": 0.1},
                    backend={"dummy": {}},
                    integrator={"dummy": {}},
                    model={"dummy": {}},
                    analyser={
                        "probe": {
                            "plugin": "missing_implementation",
                            "config": {},
                        }
                    },
                )
            )
        )


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


def test_compiler_yaml_job_input_bindings_reach_product_graph(
    tmp_path, sde_profile_plugins
):
    """A YAML job's input_bindings compile into explicit graph edges."""
    from qphase.core.workflow import load_workflow
    from qphase.data import DataKind, ProductDeclaration, ProductRequirement

    class StubConfig(BaseModel):
        requirement: str | None = None

    class StubAnalyser:
        config_schema = StubConfig

        def __init__(self, config):
            self.config = config

        def input_requirements(self):
            if self.config.requirement is None:
                return []
            return [
                ProductRequirement(
                    name=self.config.requirement, kind=DataKind.STATISTICS
                )
            ]

        def output_specs(self, *, label):
            return (ProductDeclaration(name=label, kind=DataKind.STATISTICS),)

    for name in ("binding_stub_source", "binding_stub_sink"):
        registry.register("analyser", name, StubAnalyser, overwrite=True)
    try:
        workflow_path = tmp_path / "workflow.yaml"
        workflow_path.write_text(
            "schema: qphase.workflow/2\n"
            "id: input-binding-test\n"
            "title: Input binding test\n"
            "jobs:\n"
            "- name: ana\n"
            "  engine: {sde: {mode: analyze}}\n"
            "  analyser:\n"
            "    binding_stub_source: {}\n"
            "    binding_stub_sink: {requirement: spectrum}\n"
            f"  input: {{from: {tmp_path.as_posix()!r}, mode: dataset}}\n"
            "  input_bindings:\n"
            "    binding_stub_sink:\n"
            "      spectrum: binding_stub_source.binding_stub_source\n"
        )
        project = ProjectContext.create(tmp_path / "project")
        compiled = WorkflowCompiler(project, SystemConfig()).compile(
            load_workflow(workflow_path)
        )
    finally:
        table = registry._tables.get("analyser", {})
        table.pop("binding_stub_source", None)
        table.pop("binding_stub_sink", None)

    graph = compiled.job("ana").product_graph
    assert graph is not None
    assert [
        (edge.producer, edge.product, edge.consumer, edge.requirement)
        for edge in graph.edges
    ] == [
        (
            "analyser.binding_stub_source",
            "binding_stub_source",
            "analyser.binding_stub_sink",
            "spectrum",
        )
    ]


def test_compiler_rejects_input_bindings_for_unknown_analyser(
    tmp_path, sde_profile_plugins
):
    """Invalid binding entries are config errors naming the job and entry."""
    from qphase.data import DataKind, ProductDeclaration, ProductRequirement

    class StubConfig(BaseModel):
        requirement: str | None = None

    class StubAnalyser:
        config_schema = StubConfig

        def __init__(self, config):
            self.config = config

        def input_requirements(self):
            if self.config.requirement is None:
                return []
            return [
                ProductRequirement(
                    name=self.config.requirement, kind=DataKind.STATISTICS
                )
            ]

        def output_specs(self, *, label):
            return (ProductDeclaration(name=label, kind=DataKind.STATISTICS),)

    registry.register("analyser", "binding_stub_sink", StubAnalyser, overwrite=True)
    try:
        project = ProjectContext.create(tmp_path / "project")
        with pytest.raises(QPhasePluginError) as excinfo:
            WorkflowCompiler(project, SystemConfig()).compile(
                _workflow(
                    _sde_job(
                        "ana",
                        {"mode": "analyze"},
                        analyser={"binding_stub_sink": {"requirement": "spectrum"}},
                        input={"from": str(tmp_path), "mode": "dataset"},
                        input_bindings={"missing": {"spectrum": "input.spectrum"}},
                    )
                )
            )
    finally:
        registry._tables.get("analyser", {}).pop("binding_stub_sink", None)

    message = str(excinfo.value)
    assert "'ana'" in message
    assert "unknown analyser 'missing'" in message


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


def test_compiled_workflow_roundtrip_preserves_resolved_task_profile(
    tmp_path, sde_profile_plugins
):
    """The payload preserves the resolved profile identity and typed I/O."""
    project = ProjectContext.create(tmp_path / "project")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
            ),
            _sde_job(
                "ana",
                {"mode": "analyze"},
                analyser={"dummy": {}},
                input={"from": str(tmp_path), "mode": "dataset"},
            ),
            JobConfig(name="legacy", engine={"dummy": {}}),
        )
    )
    sim = compiled.job("sim")
    assert sim.task_profile is not None
    assert sim.task_profile.profile_id == "simulate"
    assert compiled.job("ana").task_profile is not None
    assert compiled.job("legacy").task_profile is None

    payload = json.loads(json.dumps(compiled.to_payload()))
    restored = CompiledWorkflow.from_payload(payload)

    # The restored snapshot is exactly equal, including typed inputs/outputs.
    assert restored.job("sim").task_profile == sim.task_profile
    profile = restored.job("sim").task_profile
    assert profile is not None
    assert profile.requirements.required == ["backend", "integrator", "model"]
    assert profile.inputs == []
    assert [output.name for output in profile.outputs] == ["trajectories"]
    assert profile.outputs[0].declaration.kind is DataKind.TIME_SERIES
    analyze = restored.job("ana").task_profile
    assert analyze is not None
    assert analyze.profile_id == "analyze"
    assert [slot.name for slot in analyze.inputs] == ["input"]
    assert analyze.requirements.forbidden == ["integrator", "model", "observer"]
    # Legacy-manifest engines carry no snapshot in either direction.
    assert restored.job("legacy").task_profile is None


def test_compiler_analyze_profile_requires_a_job_input(tmp_path, sde_profile_plugins):
    """The analyze profile's required input slot must map to a job input."""
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError) as excinfo:
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(_sde_job("ana", {"mode": "analyze"}, analyser={"dummy": {}}))
        )
    message = str(excinfo.value)
    assert "'analyze'" in message
    assert "input slot" in message


def test_compiler_rejects_profiles_with_multiple_required_inputs(
    tmp_path, multi_input_profile_engine
):
    """Profiles declaring more required input slots than a job input can
    express are rejected explicitly instead of faking support.
    """
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhasePluginError, match="required input slots"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(
                JobConfig(
                    name="job",
                    engine={"profiled_multi_input": {}},
                    backend={"dummy": {}},
                    input={"from": str(tmp_path), "mode": "dataset"},
                )
            )
        )


def test_compiler_profile_selection_ignores_defaulted_optional_plugins(
    tmp_path, sde_profile_plugins
):
    """An analyser living only in project defaults never upgrades the profile."""
    project = ProjectContext.create(tmp_path / "project")
    project.defaults_path.write_text("analyser:\n  dummy: {}\n", encoding="utf-8")

    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
            )
        )
    )

    job = compiled.job("sim")
    assert job.task_profile is not None
    assert job.task_profile.profile_id == "simulate"
    assert job.required_plugins == ("backend", "integrator", "model")
    assert "analyser" not in job.plugin_config


def test_compiler_forbidden_plugins_ignore_defaulted_namespaces(
    tmp_path, sde_profile_plugins
):
    """A forbidden plugin injected solely by project defaults fails nothing."""
    project = ProjectContext.create(tmp_path / "project")
    project.defaults_path.write_text("model:\n  dummy: {}\n", encoding="utf-8")

    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "ana",
                {"mode": "analyze"},
                analyser={"dummy": {}},
                input={"from": str(tmp_path), "mode": "dataset"},
            )
        )
    )

    job = compiled.job("ana")
    assert job.task_profile is not None
    assert job.task_profile.profile_id == "analyze"
    assert "model" not in job.plugin_config


def test_compiler_rejects_duplicate_task_profile_ids(
    tmp_path, duplicate_profile_engine
):
    """Duplicate profile ids in one collection fail at the compile boundary."""
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhasePluginError, match="duplicate task profile id 'base'"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(JobConfig(name="job", engine={"profiled_duplicate": {}}))
        )


def test_compiler_populates_input_product_schemas_metadata_only(
    tmp_path, recording_profile_engine
):
    """Existing external artifact directories expose their product schemas."""
    import numpy as np
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
    )
    from qphase.data.store import save_products

    from tests.plugins.profiled_plugins import RECORDED_CONTEXTS

    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=4,
                coordinate="regular",
                start=0.0,
                step=0.1,
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
            )
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(
        schema,
        {"x": np.zeros((2, 4), dtype=np.complex128)},
        owner="engine.fake",
    )
    artifact_dir = tmp_path / "upstream_artifact"
    save_products(artifact_dir, {"trajectories": dataset})

    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())
    compiler.compile(
        _workflow(
            JobConfig(
                name="job",
                engine={"profiled_recording": {}},
                backend={"dummy": {}},
                input={"from": str(artifact_dir), "mode": "dataset"},
            )
        )
    )

    assert len(RECORDED_CONTEXTS) == 1
    schemas = RECORDED_CONTEXTS[0].named_input_product_schemas
    assert set(schemas) == {"trajectories"}
    assert schemas["trajectories"].kind is DataKind.TIME_SERIES

    # Upstream-job inputs have no products at compile time: empty by design.
    compiler.compile(
        _workflow(
            JobConfig(name="source", engine={"dummy": {}}),
            JobConfig(
                name="sink",
                engine={"profiled_recording": {}},
                backend={"dummy": {}},
                input={"from": "source", "mode": "dataset"},
            ),
        )
    )
    assert RECORDED_CONTEXTS[-1].named_input_product_schemas == {}


def _time_series_artifact(path):
    """Write a current-format artifact holding one time-series product."""
    import numpy as np
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
    )
    from qphase.data.store import save_products

    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=4,
                coordinate="regular",
                start=0.0,
                step=0.1,
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
            )
        ],
    )
    dataset = TimeSeriesDataset.from_arrays(
        schema, {"x": np.zeros((2, 4), dtype=np.complex128)}, owner="engine.fake"
    )
    save_products(path, {"trajectories": dataset})


def _statistics_artifact(path):
    """Write a current-format artifact holding one statistics product."""
    import numpy as np
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        StatisticsDataset,
        VariableSchema,
    )
    from qphase.data.store import save_products

    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="candidate", role=AxisRole.INDEX, size=2)],
        variables=[
            VariableSchema(
                name="rank", dtype="int64", value_domain="real", dims=("candidate",)
            )
        ],
    )
    dataset = StatisticsDataset.from_arrays(
        schema, {"rank": np.array([1, 2])}, owner="engine.fake"
    )
    save_products(path, {"table": dataset})


def _typed_input_job(artifact_dir) -> JobConfig:
    return JobConfig(
        name="job",
        engine={"profiled_typed_input": {}},
        backend={"dummy": {}},
        input={"from": str(artifact_dir), "mode": "dataset"},
    )


def test_compiler_matches_profile_input_against_artifact_products(
    tmp_path, typed_input_profile_engine
):
    """A required input slot is satisfied by a matching artifact product."""
    artifact_dir = tmp_path / "upstream_artifact"
    _time_series_artifact(artifact_dir)

    project = ProjectContext.create(tmp_path / "project")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(_typed_input_job(artifact_dir))
    )

    profile = compiled.job("job").task_profile
    assert profile is not None
    assert [slot.name for slot in profile.inputs] == ["input"]


def test_compiler_rejects_profile_input_kind_mismatch(
    tmp_path, typed_input_profile_engine
):
    """An artifact whose products violate the slot's kind fails the compile."""
    artifact_dir = tmp_path / "upstream_artifact"
    _statistics_artifact(artifact_dir)

    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError) as excinfo:
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(_typed_input_job(artifact_dir))
        )
    message = str(excinfo.value)
    assert "'job'" in message
    assert "'base'" in message
    assert "input slot 'input'" in message
    assert str(artifact_dir) in message


def test_compiler_rejects_corrupt_input_manifest(tmp_path, typed_input_profile_engine):
    """A corrupt input manifest fails fast instead of reading as no schema."""
    artifact_dir = tmp_path / "upstream_artifact"
    artifact_dir.mkdir()
    (artifact_dir / "artifact_manifest.json").write_text("{not json", encoding="utf-8")

    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError, match="unreadable artifact manifest"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(_typed_input_job(artifact_dir))
        )


def test_compiler_rejects_unsupported_input_manifest_version(
    tmp_path, typed_input_profile_engine
):
    """An unsupported manifest version fails fast instead of a silent skip."""
    artifact_dir = tmp_path / "upstream_artifact"
    artifact_dir.mkdir()
    (artifact_dir / "artifact_manifest.json").write_text(
        json.dumps({"schema_version": "qphase.artifact/3"}), encoding="utf-8"
    )

    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError, match="unreadable artifact manifest"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(_typed_input_job(artifact_dir))
        )


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
