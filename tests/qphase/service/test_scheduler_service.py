from unittest.mock import MagicMock, patch

from pydantic import BaseModel
from qphase.core.config import JobConfig, JobList
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig
from qphase.service import SchedulerService


class ManifestEngineConfig(BaseModel):
    param: float = 1.0


class OptionalAnalyserEngine:
    config_schema = ManifestEngineConfig
    manifest = EngineManifest(required_plugins=set(), optional_plugins={"analyser"})


def test_scheduler_service_builds_plan_without_creating_session(tmp_path):
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )
    job_list = JobList(
        jobs=[
            JobConfig(name="source", engine={"dummy": {"param": 1.0}}),
            JobConfig(name="sink", engine={"dummy": {}}, input="source"),
        ]
    )

    plan = SchedulerService(system_config).build_plan(job_list)

    assert [job.name for job in plan.original_jobs] == ["source", "sink"]
    assert [job.name for job in plan.expanded_jobs] == ["source", "sink"]
    assert [(edge.source, edge.target, edge.kind) for edge in plan.edges] == [
        ("source", "sink", "input")
    ]
    assert not (tmp_path / "runs").exists()


def test_scheduler_service_run_wraps_core_scheduler(tmp_path):
    system_config = MagicMock(spec=SystemConfig)
    system_config.paths = MagicMock()
    system_config.paths.output_dir = str(tmp_path / "runs")
    job_list = JobList(jobs=[JobConfig(name="job", engine={"dummy": {}})])

    with patch("qphase.service.scheduler.Scheduler") as scheduler_cls:
        scheduler = scheduler_cls.return_value
        scheduler.run.return_value = []

        results = SchedulerService(system_config).run(job_list)

    assert results == []
    scheduler_cls.assert_called_once()
    scheduler.run.assert_called_once_with(job_list, resume_from=None)


def test_scheduler_service_plan_includes_cartesian_scan_metadata(tmp_path):
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )
    job_list = JobList(
        jobs=[
            JobConfig(
                name="scan",
                engine={"dummy": {"param": [1.0, 2.0]}},
                plugins={"model": {"dummy": {"param": [10.0, 20.0]}}},
            )
        ]
    )

    plan = SchedulerService(system_config).build_plan(job_list)

    assert [job.name for job in plan.expanded_jobs] == [
        "scan_001",
        "scan_002",
        "scan_003",
        "scan_004",
    ]
    assert plan.scan_groups == {
        "scan": ["scan_001", "scan_002", "scan_003", "scan_004"]
    }
    assert plan.expanded_jobs[0].base_name == "scan"
    assert plan.expanded_jobs[0].index == 1
    assert plan.expanded_jobs[0].scan_params == {
        "engine.dummy.param": 1.0,
        "model.dummy.param": 10.0,
    }


def test_scheduler_service_plan_includes_zipped_scan_metadata(tmp_path):
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        },
        parameter_scan={"enabled": True, "method": "zipped"},
    )
    job_list = JobList(
        jobs=[
            JobConfig(
                name="zip_scan",
                engine={"dummy": {"param": [1.0, 2.0]}},
                plugins={"model": {"dummy": {"param": [10.0, 20.0]}}},
            )
        ]
    )

    plan = SchedulerService(system_config).build_plan(job_list)

    assert [job.scan_params for job in plan.expanded_jobs] == [
        {"engine.dummy.param": 1.0, "model.dummy.param": 10.0},
        {"engine.dummy.param": 2.0, "model.dummy.param": 20.0},
    ]


def test_scheduler_service_plan_marks_aggregate_input_edge(tmp_path):
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )
    job_list = JobList(
        jobs=[
            JobConfig(name="scan", engine={"dummy": {"param": [1.0, 2.0]}}),
            JobConfig(
                name="aggregate",
                engine={"dummy": {}},
                input="scan",
                aggregate_input={"on": "engine.dummy.param"},
            ),
        ]
    )

    plan = SchedulerService(system_config).build_plan(job_list)

    assert any(
        edge.source == "scan"
        and edge.target == "aggregate"
        and edge.kind == "aggregate"
        for edge in plan.edges
    )


def test_scheduler_service_plan_does_not_enable_optional_global_default(tmp_path):
    registry.register(
        "engine",
        "optional_analyser",
        OptionalAnalyserEngine,
        overwrite=True,
    )
    registry.register(
        "analyser",
        "dummy",
        OptionalAnalyserEngine,
        overwrite=True,
    )
    global_file = tmp_path / "global.yaml"
    global_file.write_text("analyser:\n  dummy:\n    param: 3.0\n", encoding="utf-8")
    system_config = SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(global_file),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )
    job_list = JobList(jobs=[JobConfig(name="job", engine={"optional_analyser": {}})])

    plan = SchedulerService(system_config).build_plan(job_list)

    assert plan.expanded_jobs[0].optional_plugins == ["analyser"]
    assert plan.expanded_jobs[0].optional_plugins_enabled == []
    assert plan.expanded_jobs[0].inherited_global_defaults == {}
