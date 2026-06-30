from unittest.mock import MagicMock, patch

from qphase.core.config import JobConfig, JobList
from qphase.core.system_config import SystemConfig
from qphase.service import SchedulerService


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
