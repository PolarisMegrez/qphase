from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydantic import BaseModel
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.dataset import DatasetSaveReport
from qphase.core.errors import QPhaseConfigError
from qphase.core.execution import CheckpointStore, plugin_fingerprint
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.scan import ScanSpec
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import SystemConfig
from qphase.service.scheduler import SchedulerService

from tests.plugins.dummy_plugin import DummyResult


class EmptyConfig(BaseModel):
    pass


class TinyDataset:
    axes = {"x": np.array([1.0, 2.0, 3.0])}
    shape = (3,)
    metadata: dict[str, Any] = {}
    label = None
    data = None

    def point_view(self, index):
        return DummyResult(data=float(self.axes["x"][index[0]]))

    def save(self, path):
        raise NotImplementedError

    def save_dataset(self, path, *, layout, shard_target_bytes):
        del path, shard_target_bytes
        return DatasetSaveReport(layout, ())


class DatasetSourceEngine:
    config_schema = EmptyConfig
    manifest = EngineManifest(required_plugins=set())

    def __init__(self, config=None, plugins=None):
        del config, plugins

    def run(self, data=None, context=None, progress_cb=None):
        del data, context, progress_cb
        return TinyDataset()


class CountingMapEngine:
    config_schema = EmptyConfig
    manifest = EngineManifest(required_plugins=set(), input_plugins=set())
    calls = 0

    def __init__(self, config=None, plugins=None):
        del config, plugins

    def run(self, data=None, context=None, progress_cb=None):
        del context, progress_cb
        type(self).calls += 1
        return DummyResult(data=data.data)


def test_scan_spec_preserves_axis_order_and_cartesian_shape():
    spec = ScanSpec.model_validate(
        {
            "combine": "cartesian",
            "axes": {
                "omega": {
                    "target": "model.dummy.param",
                    "linspace": {"start": 0.0, "stop": 1.0, "num": 3},
                },
                "gain": {
                    "target": "engine.dummy.param",
                    "values": [2.0, 4.0],
                },
            },
        }
    )

    grid = spec.compile()

    assert grid.axis_names == ("omega", "gain")
    assert grid.shape == (3, 2)
    assert grid.size == 6
    assert grid.point((1, 0)).values == {"omega": 0.5, "gain": 2.0}


def test_scan_spec_supports_zipped_logspace():
    spec = ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "x": {
                    "target": "model.dummy.param",
                    "logspace": {"start": -2, "stop": 0, "num": 3},
                },
                "y": {
                    "target": "engine.dummy.param",
                    "values": [1, 2, 3],
                },
            },
        }
    )

    grid = spec.compile()

    assert grid.shape == (3,)
    assert grid.point((1,)).values == {"x": pytest.approx(0.1), "y": 2}


def test_zipped_scan_rejects_mismatched_lengths():
    spec = ScanSpec.model_validate(
        {
            "combine": "zipped",
            "axes": {
                "x": {"target": "model.dummy.param", "values": [1, 2]},
                "y": {"target": "engine.dummy.param", "values": [1]},
            },
        }
    )

    with pytest.raises(QPhaseConfigError, match="equal lengths"):
        spec.compile()


def test_job_rejects_legacy_list_scan_syntax():
    with pytest.raises(Exception, match="list-as-scan syntax"):
        JobConfig(
            name="legacy",
            engine={"dummy": {}},
            plugins={"model": {"dummy": {"param": [1.0, 2.0]}}},
        )


def test_job_rejects_removed_string_input_syntax():
    with pytest.raises(Exception, match="string input syntax was removed"):
        JobConfig(name="sink", engine={"dummy": {}}, input="source")


@pytest.mark.parametrize(
    "field", ["storage", "storage_layout", "resources", "checkpoint", "scan_runtime"]
)
def test_job_rejects_scan_runtime_shortcuts(field: str):
    with pytest.raises(QPhaseConfigError, match="job.system.scan_runtime"):
        JobConfig(name="shortcut", engine={"dummy": {}}, **{field: {}})


def test_job_rejects_removed_parameter_scan_field():
    with pytest.raises(QPhaseConfigError, match="job.scan"):
        JobConfig(name="legacy", engine={"dummy": {}}, parameter_scan={})


def test_checkpoint_store_honors_interval_and_flushes_retained_tail(tmp_path: Path):
    config = type(
        "CheckpointConfig",
        (),
        {"enabled": True, "interval_chunks": 2, "keep_on_success": True},
    )()
    store = CheckpointStore(tmp_path, config, {"config_sha256": "test"})

    store.save_chunk("chunk-0", {"value": 0})
    assert not (store.root / "chunk-0.pkl").exists()

    store.save_chunk("chunk-1", {"value": 1})
    assert (store.root / "chunk-0.pkl").exists()
    assert (store.root / "chunk-1.pkl").exists()

    store.save_chunk("chunk-2", {"value": 2})
    store.complete()
    assert (store.root / "chunk-2.pkl").exists()


def test_job_system_deep_merges_scan_runtime_overrides():
    global_system = SystemConfig.model_validate(
        {
            "scan_runtime": {
                "auto_shard_threshold_mib": 900,
                "shard_target_mib": 222,
                "checkpoint": {"interval_chunks": 3},
            }
        }
    )
    job = JobConfig(
        name="override",
        engine={"dummy": {}},
        system={"scan_runtime": {"checkpoint": {"enabled": True}}},
    )

    merged = job.merge_with_system_config(global_system)

    assert merged.scan_runtime.auto_shard_threshold_mib == 900
    assert merged.scan_runtime.shard_target_mib == 222
    assert merged.scan_runtime.checkpoint.enabled is True
    assert merged.scan_runtime.checkpoint.interval_chunks == 3


def test_plugin_fingerprint_records_code_identity():
    fingerprint = plugin_fingerprint(DatasetSourceEngine())

    assert fingerprint["class"].endswith(":DatasetSourceEngine")
    assert "source_sha256" not in fingerprint


def test_execution_plan_keeps_scan_as_one_logical_job(tmp_path: Path, temp_project):
    config = _system_config(tmp_path)
    job = _scan_job()

    plan = SchedulerService(config, project=temp_project).build_plan(
        WorkflowSpec(
            schema="qphase.workflow/2",
            id="test-workflow",
            title="Test Workflow",
            jobs=[job],
        )
    )

    assert len(plan.jobs) == 1
    assert plan.jobs[0].name == "scan"
    assert plan.jobs[0].scan_summary["shape"] == [3]
    assert plan.jobs[0].scan_summary["size"] == 3


def test_scheduler_creates_one_manifest_entry_and_job_directory(
    tmp_path: Path, temp_project
):
    config = _system_config(tmp_path)
    scheduler = Scheduler(system_config=config, project=temp_project)

    results = scheduler.run(
        WorkflowSpec(
            schema="qphase.workflow/2",
            id="test-workflow",
            title="Test Workflow",
            jobs=[_scan_job()],
        )
    )

    assert len(results) == 1
    assert results[0].success
    assert scheduler.manifest is not None
    assert list(scheduler.manifest["jobs"]) == ["scan"]
    assert scheduler.session_dir is not None
    assert [path.name for path in scheduler.session_dir.iterdir() if path.is_dir()] == [
        "scan"
    ]
    assert (scheduler.session_dir / "scan" / "artifact_manifest.json").exists()


def test_map_input_runs_views_inside_one_logical_job(tmp_path: Path, temp_project):
    registry.register("engine", "dataset_source", DatasetSourceEngine, overwrite=True)
    registry.register("engine", "counting_map", CountingMapEngine, overwrite=True)
    CountingMapEngine.calls = 0
    scheduler = Scheduler(system_config=_system_config(tmp_path), project=temp_project)
    jobs = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[
            JobConfig(
                name="source",
                engine={"dataset_source": {}},
                save=False,
            ),
            JobConfig(
                name="mapped",
                engine={"counting_map": {}},
                input={"from": "source", "mode": "map"},
                save=False,
            ),
        ],
    )

    results = scheduler.run(jobs)

    assert all(result.success for result in results)
    assert CountingMapEngine.calls == 3
    assert scheduler.session_dir is not None
    assert sorted(
        path.name for path in scheduler.session_dir.iterdir() if path.is_dir()
    ) == ["mapped", "source"]


def _scan_job() -> JobConfig:
    return JobConfig(
        name="scan",
        engine={"dummy": {"param": 1.0}},
        scan={
            "axes": {
                "parameter": {
                    "target": "engine.dummy.param",
                    "values": [1.0, 2.0, 3.0],
                }
            }
        },
        save=True,
    )


def _system_config(tmp_path: Path) -> SystemConfig:
    return SystemConfig()
