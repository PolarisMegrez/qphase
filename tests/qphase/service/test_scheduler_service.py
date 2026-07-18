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


def _system_config(tmp_path):
    return SystemConfig(
        paths={
            "output_dir": str(tmp_path / "runs"),
            "global_file": str(tmp_path / "global.yaml"),
            "config_dirs": [str(tmp_path / "configs")],
            "plugin_dirs": [str(tmp_path / "plugins")],
        }
    )


def test_scheduler_service_builds_logical_plan_without_creating_session(tmp_path):
    job_list = JobList(
        jobs=[
            JobConfig(name="source", engine={"dummy": {"param": 1.0}}),
            JobConfig(
                name="sink",
                engine={"dummy": {}},
                input={"from": "source", "mode": "dataset"},
            ),
        ]
    )

    plan = SchedulerService(_system_config(tmp_path)).build_plan(job_list)

    assert [job.name for job in plan.jobs] == ["source", "sink"]
    assert [(edge.source, edge.target, edge.kind) for edge in plan.edges] == [
        ("source", "sink", "input")
    ]
    assert plan.edges[0].input_mode == "dataset"
    assert not (tmp_path / "runs").exists()


def test_scheduler_service_run_wraps_core_scheduler(tmp_path):
    system_config = MagicMock(spec=SystemConfig)
    system_config.paths = MagicMock()
    system_config.paths.output_dir = str(tmp_path / "runs")
    job_list = JobList(jobs=[JobConfig(name="job", engine={"dummy": {}})])

    with patch("qphase.service.scheduler.Scheduler") as scheduler_cls:
        scheduler = scheduler_cls.return_value
        scheduler.run.return_value = []
        scheduler.session_id = "test-session"
        scheduler.session_dir = tmp_path / "runs" / "test-session"

        results = SchedulerService(system_config).run(job_list)

    assert results == []
    scheduler_cls.assert_called_once()
    scheduler.run.assert_called_once_with(job_list, resume_from=None)


def test_scheduler_service_reports_cartesian_and_zipped_scan_shapes(tmp_path):
    cartesian = JobConfig(
        name="cartesian",
        engine={"dummy": {"param": 1.0}},
        scan={
            "combine": "cartesian",
            "axes": {
                "x": {"target": "engine.dummy.param", "values": [1, 2]},
                "y": {"target": "model.dummy.param", "values": [10, 20, 30]},
            },
        },
    )
    zipped = JobConfig(
        name="zipped",
        engine={"dummy": {"param": 1.0}},
        scan={
            "combine": "zipped",
            "axes": {
                "x": {"target": "engine.dummy.param", "values": [1, 2]},
                "y": {"target": "model.dummy.param", "values": [10, 20]},
            },
        },
    )

    plan = SchedulerService(_system_config(tmp_path)).build_plan(
        JobList(jobs=[cartesian, zipped])
    )

    assert plan.jobs[0].scan_summary["shape"] == [2, 3]
    assert plan.jobs[0].scan_summary["size"] == 6
    assert plan.jobs[1].scan_summary["shape"] == [2]


def test_scheduler_service_marks_map_input_edge(tmp_path):
    jobs = JobList(
        jobs=[
            JobConfig(name="scan", engine={"dummy": {}}),
            JobConfig(
                name="mapped",
                engine={"dummy": {}},
                input={"from": "scan", "mode": "map", "group_by": ["omega"]},
            ),
        ]
    )

    plan = SchedulerService(_system_config(tmp_path)).build_plan(jobs)

    assert plan.edges[0].input_mode == "map"


def test_scheduler_service_does_not_enable_optional_global_default(tmp_path):
    registry.register(
        "engine", "optional_analyser", OptionalAnalyserEngine, overwrite=True
    )
    registry.register("analyser", "dummy", OptionalAnalyserEngine, overwrite=True)
    global_file = tmp_path / "global.yaml"
    global_file.write_text("analyser:\n  dummy:\n    param: 3.0\n", encoding="utf-8")
    system_config = _system_config(tmp_path)
    system_config.paths.global_file = str(global_file)

    plan = SchedulerService(system_config).build_plan(
        JobList(
            jobs=[JobConfig(name="job", engine={"optional_analyser": {}})]
        )
    )

    assert plan.jobs[0].optional_plugins == ["analyser"]
    assert plan.jobs[0].optional_plugins_enabled == []
    assert plan.jobs[0].inherited_global_defaults == {}
