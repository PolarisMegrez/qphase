"""Long-lived tests for the Phase 2 logical workflow compiler."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, ClassVar

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
def dummy_engine_plugins():
    """Register a core-owned engine plus dummy plugins for its namespaces."""
    from tests.plugins.dummy_engine import DummyEngine
    from tests.plugins.dummy_plugin import DummyPlugin

    engine_table = registry._tables.setdefault("engine", {})
    previous_sde = engine_table.get("sde")
    registry.register("engine", "sde", DummyEngine, overwrite=True)
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


def test_compiler_normalizes_analyser_shorthand(tmp_path, dummy_engine_plugins):
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
    tmp_path, dummy_engine_plugins
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


@pytest.mark.parametrize("source", ["top_level", "plugins"])
def test_analyser_label_does_not_merge_same_named_implementation_default(
    tmp_path, dummy_engine_plugins, source
):
    from qphase.core.registry import discovery

    discovery.discover_plugins()
    project = ProjectContext.create(tmp_path / "project")
    project.defaults_path.write_text(
        "analyser:\n  psd:\n    convention: unitary\n",
        encoding="utf-8",
    )
    instance = {
        "psd": {
            "plugin": "allan_variance",
            "config": {
                "measurement": {
                    "phase_frequency": {
                        "channels": [0],
                        "min_independent_windows": 1,
                    }
                },
                "points": 5,
                "min_windows": 2,
            },
        }
    }
    sections = (
        {"analyser": instance}
        if source == "top_level"
        else {"plugins": {"analyser": instance}}
    )
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
                **sections,
            )
        )
    ).job("sim")

    normalized = dict(compiled.plugin_config["analyser"]["psd"])
    assert normalized["plugin"] == "allan_variance"
    assert "convention" not in normalized["config"]


@pytest.mark.parametrize("label", ["with.dot", "input", 7])
def test_compiler_rejects_invalid_analyser_instance_label(
    tmp_path, dummy_engine_plugins, label
):
    project = ProjectContext.create(tmp_path / "project")
    with pytest.raises(QPhaseConfigError, match="instance label|labels must"):
        WorkflowCompiler(project, SystemConfig()).compile(
            _workflow(
                _sde_job(
                    "sim",
                    {"t1": 1.0, "dt": 0.1},
                    backend={"dummy": {}},
                    integrator={"dummy": {}},
                    model={"dummy": {}},
                    analyser={label: {"plugin": "dummy", "config": {}}},
                )
            )
        )


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


_EXPANDER_CALLS: list[dict] = []


def _probe_analyser_expander(analysers: Mapping[str, Any]) -> dict[str, Any]:
    """Test expander: records its input and adds a sibling synthetic entry."""
    import copy

    _EXPANDER_CALLS.append(copy.deepcopy(dict(analysers)))
    expanded = {label: dict(entry) for label, entry in analysers.items()}
    if "probe" in expanded:
        return expanded
    expanded["probe"] = {"plugin": "dummy", "config": {}}
    return expanded


def test_analyser_config_expander_runs_before_compile_and_snapshot(
    tmp_path, dummy_engine_plugins, monkeypatch
):
    """The engine's expander rewrites the analyser section compile sees."""
    from tests.plugins.dummy_engine import DummyEngine

    _EXPANDER_CALLS.clear()
    monkeypatch.setattr(
        DummyEngine.manifest,
        "analyser_config_expander",
        "tests.qphase.test_compiler:_probe_analyser_expander",
        raising=False,
    )
    project = ProjectContext.create(tmp_path / "project")
    compiler = WorkflowCompiler(project, SystemConfig())

    compiled = compiler.compile(
        _workflow(
            _sde_job(
                "sim",
                {"t1": 1.0, "dt": 0.1},
                backend={"dummy": {}},
                integrator={"dummy": {}},
                model={"dummy": {}},
                analyser={"dummy": {"plugin": "dummy", "config": {}}},
            )
        )
    ).job("sim")

    assert len(_EXPANDER_CALLS) == 1
    # The expander received the normalized instance shape.
    assert _EXPANDER_CALLS[0]["dummy"] == {"plugin": "dummy", "config": {}}
    # Validation, graph compilation and the snapshot all see the same expanded
    # analyser section.
    assert compiled.plugin_config["analyser"]["probe"] == {
        "plugin": "dummy",
        "config": {},
    }
    assert compiled.merged_config["analyser"] == compiled.plugin_config["analyser"]


def test_analyser_config_expander_is_idempotent_and_optional():
    """A missing hook is a no-op; a second expansion changes nothing."""
    from types import SimpleNamespace

    from qphase.core.compiler import WorkflowCompiler

    compiler = object.__new__(WorkflowCompiler)
    selected = {"analyser": {"dummy": {"plugin": "dummy", "config": {}}}}

    assert compiler._expand_analyser_config(SimpleNamespace(), selected) is selected
    once = compiler._expand_analyser_config(
        SimpleNamespace(
            analyser_config_expander="tests.qphase.test_compiler:_probe_analyser_expander"
        ),
        selected,
    )
    twice = compiler._expand_analyser_config(
        SimpleNamespace(
            analyser_config_expander="tests.qphase.test_compiler:_probe_analyser_expander"
        ),
        once,
    )
    assert once["analyser"] == twice["analyser"]
    # The caller's mapping is never mutated in place.
    assert "probe" not in selected["analyser"]
