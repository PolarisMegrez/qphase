"""SDE engine task profiles enforced through the core workflow compiler.

The SDE engine declares ``simulate``/``simulate_analyze``/``analyze`` profiles
in ``qphase_sde.contracts.tasks``; these tests drive them through
``qphase.core.compiler`` and therefore live with the SDE suite.
"""

from __future__ import annotations

import json

import pytest
from qphase.core.compiler import CompiledWorkflow, WorkflowCompiler
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseConfigError
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
