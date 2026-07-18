from pathlib import Path

import pytest
from qphase.core.config import JobConfig, JobList
from qphase.core.errors import QPhaseConfigError
from qphase.core.scan import ScanSpec
from qphase.core.scheduler import Scheduler
from qphase.core.system_config import PathsConfig, SystemConfig
from qphase.service.scheduler import SchedulerService


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


def test_execution_plan_keeps_scan_as_one_logical_job(tmp_path: Path):
    config = _system_config(tmp_path)
    job = _scan_job()

    plan = SchedulerService(config).build_plan(JobList(jobs=[job]))

    assert len(plan.jobs) == 1
    assert plan.jobs[0].name == "scan"
    assert plan.jobs[0].scan_summary["shape"] == [3]
    assert plan.jobs[0].scan_summary["size"] == 3


def test_scheduler_creates_one_manifest_entry_and_job_directory(tmp_path: Path):
    config = _system_config(tmp_path)
    scheduler = Scheduler(system_config=config)

    results = scheduler.run(JobList(jobs=[_scan_job()]))

    assert len(results) == 1
    assert results[0].success
    assert scheduler.manifest is not None
    assert list(scheduler.manifest["jobs"]) == ["scan"]
    assert scheduler.session_dir is not None
    assert [path.name for path in scheduler.session_dir.iterdir() if path.is_dir()] == [
        "scan"
    ]
    assert (scheduler.session_dir / "scan" / "artifact_manifest.json").exists()


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
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "global.yaml").write_text("{}\n", encoding="utf-8")
    return SystemConfig(
        paths=PathsConfig(
            output_dir=str(tmp_path / "runs"),
            global_file=str(config_dir / "global.yaml"),
            plugin_dirs=[str(tmp_path / "plugins")],
            config_dirs=[str(config_dir)],
        )
    )
