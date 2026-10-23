"""SDE analyser instance labels and product graphs through the core compiler.

Normalising ``analyser`` instances, merging project defaults and compiling the
engine's product graph are core compiler steps whose inputs are SDE plugin
declarations, so these tests live with the SDE suite.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from qphase.core.compiler import CompiledWorkflow, WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhasePluginError
from qphase.core.project import ProjectContext
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


def _sde_job(name: str, engine_config: dict, **sections) -> JobConfig:
    return JobConfig(name=name, engine={"sde": engine_config}, **sections)


@pytest.fixture(autouse=True)
def register_dummy_plugins():
    """Register the dummy engine/backend/model the jobs name."""
    from tests.plugins.dummy_plugin import DummyPlugin

    for namespace in ("engine", "backend", "model"):
        registry.register(
            namespace=namespace, name="dummy", builder=DummyPlugin, overwrite=True
        )
    yield
    for namespace in ("engine", "backend", "model"):
        registry._tables.get(namespace, {}).pop("dummy", None)


@pytest.fixture
def sde_profile_plugins():
    """Register the real SDE engine plus dummy plugins for its namespaces."""
    from qphase.core.registry import discovery
    from qphase_sde.engine import Engine

    from tests.plugins.dummy_plugin import DummyPlugin

    discovery.discover_plugins()
    engine_table = registry._tables.setdefault("engine", {})
    previous_sde = engine_table.get("sde")
    registry.register("engine", "sde", Engine, overwrite=True)
    registry.register("integrator", "dummy", DummyPlugin, overwrite=True)
    registry.register("analyser", "dummy", DummyPlugin, overwrite=True)
    try:
        yield
    finally:
        if previous_sde is not None:
            engine_table["sde"] = previous_sde
        registry._tables.get("integrator", {}).pop("dummy", None)
        registry._tables.get("analyser", {}).pop("dummy", None)


def test_compiler_normalizes_analyser_instance_labels(tmp_path, sde_profile_plugins):
    from qphase.core.job_runner import build_plugins
    from qphase.core.registry import discovery
    from qphase_sde.analyser.psd import PsdAnalyser

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
                        "config": {
                            "measurement": {
                                "complex_field": {
                                    "channels": [0],
                                    "representation": "complex",
                                }
                            }
                        },
                    },
                    "other_psd": {
                        "plugin": "psd",
                        "config": {
                            "measurement": {
                                "complex_field": {
                                    "channels": [0],
                                    "representation": "complex",
                                }
                            }
                        },
                    },
                },
            )
        )
    )
    compiled = compiled_workflow.job("sim")

    expected = {
        "plugin": "psd",
        "config": {
            "measurement": {
                "complex_field": {"channels": [0], "representation": "complex"}
            },
            "convention": "unitary",
        },
    }
    assert dict(compiled.plugin_config["analyser"]["raw_psd"]) == expected
    assert dict(compiled.plugin_config["analyser"]["other_psd"]) == expected
    assert dict(compiled.merged_config["analyser"]["raw_psd"]) == expected
    restored = CompiledWorkflow.from_payload(compiled_workflow.to_payload()).job("sim")
    assert dict(restored.plugin_config["analyser"]["raw_psd"]) == expected

    instances = build_plugins(registry, dict(compiled.plugin_config))["analyser"]
    assert set(instances) == {"raw_psd", "other_psd"}
    assert all(isinstance(instance, PsdAnalyser) for instance in instances.values())
    assert compiled.product_graph is not None
    assert {node.node_id for node in compiled.product_graph.nodes} == {
        "analyser.raw_psd",
        "analyser.other_psd",
    }
    assert {node.producer for node in compiled.product_graph.nodes} == {"analyser:psd"}

def test_compiler_yaml_job_input_bindings_reach_product_graph(
    tmp_path, sde_profile_plugins
):
    """A YAML job's input_bindings compile into explicit graph edges."""
    from qphase.core.workflow import load_workflow
    from qphase.data import ProductDeclaration, ProductRequirement

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
    from qphase.data import ProductDeclaration, ProductRequirement

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
