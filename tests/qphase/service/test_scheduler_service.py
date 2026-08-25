from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    save_products,
)
from qphase.service import SchedulerService


class ManifestEngineConfig(BaseModel):
    param: float = 1.0


class OptionalAnalyserEngine:
    config_schema = ManifestEngineConfig
    manifest = EngineManifest(required_plugins=set(), optional_plugins={"analyser"})


def _system_config(tmp_path):
    return SystemConfig()


def test_scheduler_service_builds_logical_plan_without_creating_session(tmp_path):
    job_list = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[
            JobConfig(name="source", engine={"dummy": {"param": 1.0}}),
            JobConfig(
                name="sink",
                engine={"dummy": {}},
                input={"from": "source", "mode": "dataset"},
            ),
        ],
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
    job_list = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[JobConfig(name="job", engine={"dummy": {}})],
    )

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
        WorkflowSpec(
            schema="qphase.workflow/2",
            id="test-workflow",
            title="Test Workflow",
            jobs=[cartesian, zipped],
        )
    )

    assert plan.jobs[0].scan_summary["shape"] == [2, 3]
    assert plan.jobs[0].scan_summary["size"] == 6
    assert plan.jobs[1].scan_summary["shape"] == [2]


def test_scheduler_service_marks_map_input_edge(tmp_path):
    jobs = WorkflowSpec(
        schema="qphase.workflow/2",
        id="test-workflow",
        title="Test Workflow",
        jobs=[
            JobConfig(name="scan", engine={"dummy": {}}),
            JobConfig(
                name="mapped",
                engine={"dummy": {}},
                input={"from": "scan", "mode": "map", "group_by": ["omega"]},
            ),
        ],
    )

    plan = SchedulerService(_system_config(tmp_path)).build_plan(jobs)

    assert plan.edges[0].input_mode == "map"


def test_scheduler_service_does_not_enable_optional_global_default(tmp_path):
    registry.register(
        "engine", "optional_analyser", OptionalAnalyserEngine, overwrite=True
    )
    registry.register("analyser", "dummy", OptionalAnalyserEngine, overwrite=True)
    global_file = tmp_path / "configs" / "defaults.yaml"
    global_file.parent.mkdir()
    global_file.write_text("analyser:\n  dummy:\n    param: 3.0\n", encoding="utf-8")
    system_config = _system_config(tmp_path)

    plan = SchedulerService(system_config).build_plan(
        WorkflowSpec(
            schema="qphase.workflow/2",
            id="test-workflow",
            title="Test Workflow",
            jobs=[JobConfig(name="job", engine={"optional_analyser": {}})],
        )
    )

    assert plan.jobs[0].optional_plugins == ["analyser"]
    assert plan.jobs[0].optional_plugins_enabled == []
    assert plan.jobs[0].inherited_project_defaults == {}


def _products_dataset(rows: int = 4) -> TimeSeriesDataset:
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=rows),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=8,
                coordinate="regular",
                start=0.0,
                step=0.1,
                units="s",
            ),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time"),
                quantity="amplitude",
            ),
        ],
    )
    rng = np.random.default_rng(3)
    x = rng.normal(size=(rows, 8)) + 1j * rng.normal(size=(rows, 8))
    return TimeSeriesDataset.from_arrays(schema, {"x": x}, owner="engine.fake")


def test_scheduler_service_describes_products_without_materializing(tmp_path):
    session_root = tmp_path / "session"
    manifest = save_products(
        session_root / "job1",
        {"trajectories": _products_dataset()},
        provenance={"engine": "test"},
    )

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    assert catalog.artifact_id == manifest.artifact_id
    assert catalog.content_hash == manifest.content_hash
    assert catalog.loader == manifest.loader
    assert catalog.size > 0
    assert len(catalog.products) == 1
    product = catalog.products[0]
    assert product.name == "trajectories"
    assert product.kind == "time_series"
    assert product.backing == "artifact"
    assert product.devices == ["cpu"]
    assert product.materializable is True
    assert product.nbytes == 4 * 8 * 16
    assert product.chunk_count == 1
    assert product.sha256 == manifest.products[0].sha256
    axes = {axis.name: axis for axis in product.axes}
    assert axes["time"].coordinate == "regular"
    assert axes["time"].start == 0.0
    assert axes["time"].step == 0.1
    assert axes["time"].units == "s"
    assert axes["trajectory"].role == "realization"
    variable = product.variables[0]
    assert variable.name == "x"
    assert np.dtype(variable.dtype) == np.dtype("complex128")
    assert variable.dims == ["trajectory", "time"]
    # JSON round-trip: the DTO must be serializable for the GUI route.
    payload = catalog.model_dump(mode="json")
    assert payload["products"][0]["axes"][1]["step"] == 0.1


def test_scheduler_service_describe_products_rejects_non_artifact(tmp_path):
    session_root = tmp_path / "session"
    (session_root / "job1").mkdir(parents=True)
    service = SchedulerService(_system_config(tmp_path))

    with pytest.raises(FileNotFoundError):
        service.describe_products("job1", session_dir=session_root)
    with pytest.raises(FileNotFoundError):
        service.describe_products("missing", session_dir=session_root)
