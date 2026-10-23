"""Single analyser instances keep their label -> instance mapping everywhere.

The core scheduler never folds the ``analyser`` namespace down to a bare
instance: graph node identity (``analyser.<label>``) must survive plugin
construction, scheduler execution, resume, and checkpoint fingerprinting.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import BaseModel
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.job_runner import build_plugins
from qphase.core.registry import registry
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig

from tests.plugins.dummy_plugin import DummyPlugin

pytestmark = pytest.mark.integration


@pytest.fixture
def sde_profile_plugins():
    """Register the real SDE engine plus dummy plugins for its namespaces."""
    from qphase.core.registry import discovery
    from qphase_sde.engine import Engine

    # Discover first so engine.sde enters the registry through the normal
    # entry-point path.  Discovery marks entry points as consumed, so the
    # teardown must never remove "sde": a popped entry can never be
    # rediscovered, which leaks broken state into later test modules.
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


class _Result:
    data = None
    metadata: dict = {}
    label = None

    def save(self, path):
        del path


@pytest.fixture
def capturing_engine():
    """Register an engine recording the plugins mapping it receives."""
    captured: list[dict[str, Any]] = []

    class CapturingEngine:
        class Config(BaseModel):
            pass

        config_schema = Config
        name = "capturing"

        def __init__(self, config=None, plugins=None, **kwargs):
            del config, kwargs
            captured.append(dict(plugins or {}))

        def run(self, data=None, **kwargs):
            del data, kwargs
            return _Result()

    registry.register("engine", "capturing", CapturingEngine, overwrite=True)
    registry.register("analyser", "dummy", DummyPlugin, overwrite=True)
    try:
        yield captured
    finally:
        registry._tables.get("engine", {}).pop("capturing", None)
        registry._tables.get("analyser", {}).pop("dummy", None)


def _analyser_workflow() -> WorkflowSpec:
    return WorkflowSpec(
        schema_="qphase.workflow/2",
        id="analyser-instance-labels",
        title="Analyser instance labels",
        jobs=[
            JobConfig(
                name="sim",
                engine={"capturing": {}},
                analyser={"dummy": {"param": 2.0}},
                save=False,
            )
        ],
    )


def _assert_label_mapping(plugins: dict[str, Any], label: str) -> None:
    analysers = plugins["analyser"]
    assert isinstance(analysers, dict)
    assert set(analysers) == {label}
    assert isinstance(analysers[label], DummyPlugin)
    assert plugins[f"analyser.{label}"] is analysers[label]


def test_build_plugins_keeps_single_analyser_instance_label(
    tmp_path, sde_profile_plugins
):
    """A single analyser instance still builds as a label -> instance dict."""
    from qphase.core.compiler import WorkflowCompiler
    from qphase.core.project import ProjectContext
    from qphase.core.registry import discovery
    from qphase_sde.analyser.psd import PsdAnalyser

    discovery.discover_plugins()
    project = ProjectContext.create(tmp_path / "project")
    compiled = WorkflowCompiler(project, SystemConfig()).compile(
        WorkflowSpec(
            schema_="qphase.workflow/2",
            id="analyser-instance-labels",
            title="Analyser instance labels",
            jobs=[
                JobConfig(
                    name="sim",
                    engine={"sde": {"t1": 1.0, "dt": 0.1}},
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
                    },
                )
            ],
        )
    ).job("sim")

    plugins = build_plugins(registry, dict(compiled.plugin_config))

    analysers = plugins["analyser"]
    assert isinstance(analysers, dict)
    assert set(analysers) == {"raw_psd"}
    assert isinstance(analysers["raw_psd"], PsdAnalyser)
    assert plugins["analyser.raw_psd"] is analysers["raw_psd"]


def test_scheduler_passes_single_analyser_as_label_mapping(
    temp_project, capturing_engine
):
    """The engine receives ``plugins["analyser"]`` as a label mapping."""
    scheduler = Scheduler(system_config=SystemConfig(), project=temp_project)
    results = scheduler.run(_analyser_workflow())

    assert [result.success for result in results] == [True]
    assert len(capturing_engine) == 1
    _assert_label_mapping(capturing_engine[0], "dummy")
    assert (results[0].job_dir / "config_snapshot.json").exists()


def test_resume_rebuilds_single_analyser_as_label_mapping(
    temp_project, capturing_engine
):
    """A resumed Session rebuilds the analyser label mapping from scratch."""
    workflow = _analyser_workflow()
    session_dir = temp_project.session_root / "old_session"
    session_dir.mkdir(parents=True)
    manifest = {
        "schema": "qphase.session/2",
        "session_id": "old_session",
        "project_id": temp_project.project_id,
        "workflow_id": workflow.id,
        "workflow_hash": Scheduler._workflow_hash(
            workflow.model_dump(mode="json", by_alias=True)
        ),
        "start_time": "2025-01-01T00:00:00",
        "status": "failed",
        "jobs": {"sim": {"status": "failed"}},
    }
    (session_dir / "session_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    scheduler = Scheduler(system_config=SystemConfig(), project=temp_project)
    results = scheduler.run(workflow, resume_from=session_dir)

    assert [result.job_name for result in results] == ["sim"]
    assert results[0].success
    assert len(capturing_engine) == 1
    _assert_label_mapping(capturing_engine[0], "dummy")


def test_checkpoint_fingerprint_records_single_analyser_instance(
    temp_project, capturing_engine
):
    """Checkpoint plugin fingerprints use the real class per dotted label."""
    system_config = SystemConfig.model_validate(
        {"scan_runtime": {"checkpoint": {"enabled": True, "keep_on_success": True}}}
    )
    scheduler = Scheduler(system_config=system_config, project=temp_project)
    results = scheduler.run(_analyser_workflow())

    assert [result.success for result in results] == [True]
    manifest_path = results[0].job_dir / ".checkpoints" / "checkpoint_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    plugin_ids = manifest["fingerprint"]["plugins"]
    assert set(plugin_ids) == {"analyser.dummy"}
    assert plugin_ids["analyser.dummy"]["class"].endswith(":DummyPlugin")
