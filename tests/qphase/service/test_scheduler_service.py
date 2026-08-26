import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from pydantic import BaseModel
from qphase.core.config import JobConfig, WorkflowSpec
from qphase.core.errors import QPhaseIOError
from qphase.core.protocols import EngineManifest
from qphase.core.registry import registry
from qphase.core.system_config import SystemConfig
from qphase.data import (
    AxisRole,
    AxisSchema,
    BundleDescriptor,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
    save_products,
)
from qphase.data.errors import ArtifactCorruptError
from qphase.data.resolver import default_artifact_resolver
from qphase.data.store import storage_referenced_files
from qphase.service import SchedulerService
from qphase.service.project import ProjectService


class ManifestEngineConfig(BaseModel):
    param: float = 1.0


class OptionalAnalyserEngine:
    config_schema = ManifestEngineConfig
    manifest = EngineManifest(required_plugins=set(), optional_plugins={"analyser"})


def _system_config(tmp_path):
    return SystemConfig()


def test_project_json_corruption_fails_fast(tmp_path):
    path = tmp_path / "project.json"
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(QPhaseIOError, match="failed to read project JSON"):
        ProjectService._read_json(path)

    path.write_bytes(b"\xff\xfe")
    with pytest.raises(QPhaseIOError, match="failed to read project JSON"):
        ProjectService._read_json(path)


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


def test_scheduler_service_reports_cancelled_session(tmp_path):
    job_list = WorkflowSpec(
        schema="qphase.workflow/2",
        id="cancelled-workflow",
        title="Cancelled Workflow",
        jobs=[JobConfig(name="job", engine={"dummy": {}})],
    )

    with patch("qphase.service.scheduler.Scheduler") as scheduler_cls:
        scheduler = scheduler_cls.return_value
        scheduler.run.return_value = [MagicMock(status="cancelled")]
        scheduler.session_id = "cancelled-session"
        scheduler.session_dir = tmp_path / "runs" / "cancelled-session"

        service = SchedulerService(_system_config(tmp_path))
        service.run(job_list)

    assert service.last_session_handle is not None
    assert service.last_session_handle.status == "cancelled"


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
    assert not hasattr(catalog, "content_hash")
    assert catalog.loader == "npz/3"
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


def test_scheduler_service_lists_manifest_artifact_as_one_item(tmp_path):
    session_root = tmp_path / "session"
    job_dir = session_root / "job1"
    manifest = save_products(
        job_dir,
        {"trajectories": _products_dataset()},
        provenance={"engine": "test"},
    )
    service = SchedulerService(_system_config(tmp_path))

    items = service.list_artifacts(session_root)

    assert len(items) == 1
    assert items[0].artifact_id == manifest.artifact_id
    assert items[0].file_ref is None
    assert items[0].path == job_dir
    assert items[0].job_name == "job1"
    assert service.describe_artifact_by_id(
        manifest.artifact_id, session_dir=session_root
    ).artifact_id == manifest.artifact_id


def test_scheduler_service_reports_duplicate_artifact_identity(tmp_path):
    session_root = tmp_path / "session"
    for job_name in ("job1", "job2"):
        save_products(
            session_root / job_name,
            {"trajectories": _products_dataset()},
            artifact_id="duplicate-id",
        )

    with pytest.raises(ArtifactCorruptError, match="identity conflict"):
        SchedulerService(_system_config(tmp_path)).describe_artifact_by_id(
            "duplicate-id", session_dir=session_root
        )


def test_scheduler_service_rejects_corrupt_manifest_during_listing(tmp_path):
    session_root = tmp_path / "session"
    job_dir = session_root / "job1"
    job_dir.mkdir(parents=True)
    (job_dir / "artifact_manifest.json").write_text("{bad", encoding="utf-8")

    with pytest.raises(ArtifactCorruptError, match="failed to parse"):
        SchedulerService(_system_config(tmp_path)).list_artifacts(session_root)


def test_scheduler_service_describe_products_rejects_non_artifact(tmp_path):
    session_root = tmp_path / "session"
    (session_root / "job1").mkdir(parents=True)
    service = SchedulerService(_system_config(tmp_path))

    with pytest.raises(FileNotFoundError):
        service.describe_products("job1", session_dir=session_root)
    with pytest.raises(FileNotFoundError):
        service.describe_products("missing", session_dir=session_root)


def test_scheduler_service_describe_products_exposes_schema_and_storage_details(
    tmp_path,
):
    session_root = tmp_path / "session"
    manifest = save_products(
        session_root / "job1",
        {"trajectories": _products_dataset()},
        provenance={"engine": "test"},
    )

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    entry = manifest.products[0]
    product = catalog.products[0]
    assert product.schema_version == entry.product_schema.schema_version
    assert product.schema_fingerprint == entry.product_schema.fingerprint()
    assert product.storage_adapter == "npz/3"
    assert product.storage_descriptor_schema == entry.storage.descriptor_schema
    assert product.physical_nbytes > 0
    assert product.missing_reason is None
    assert product.coordinates == []
    assert product.sampling_bases == []
    assert product.uncertainties == []
    constraints = product.variables[0].constraints
    assert constraints["nonnegative"] is False
    assert constraints["layout"] == "dense"
    json.dumps(catalog.model_dump(mode="json"))


def test_scheduler_service_describe_products_reports_generic_bundle_summary(
    tmp_path,
):
    session_root = tmp_path / "session"
    save_products(session_root / "job1", {"trajectories": _products_dataset()})

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    bundle = catalog.bundle
    assert bundle is not None
    assert bundle.type_id == "generic.dataset_bundle/1"
    assert bundle.adapter_id == "generic/1"
    assert bundle.scan_shape is None
    assert bundle.scan_combine is None
    assert bundle.scan_axes is None
    assert bundle.n_traj_per_point is None


def test_scheduler_service_describe_products_unpacks_scan_bundle_summary(tmp_path):
    session_root = tmp_path / "session"
    descriptor = BundleDescriptor(
        type_id="test.bundle/1",
        adapter_id="test/1",
        descriptor_schema="test.bundle_descriptor/1",
        descriptor={
            "scan": {
                "shape": [2, 3],
                "dimension_order": ["alpha", "beta"],
                "axes": {"alpha": [1.0, 2.0], "beta": [3.0, 4.0, 5.0]},
                "n_traj_per_point": 4,
                "combine": "cartesian",
            }
        },
        product_roles={"primary": "trajectories"},
    )
    save_products(
        session_root / "job1",
        {"trajectories": _products_dataset()},
        bundle=descriptor,
    )

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    bundle = catalog.bundle
    assert bundle is not None
    assert bundle.type_id == "test.bundle/1"
    assert bundle.adapter_id == "test/1"
    assert bundle.descriptor_schema == "test.bundle_descriptor/1"
    assert bundle.product_roles == {"primary": "trajectories"}
    assert bundle.scan_shape == [2, 3]
    assert bundle.scan_combine == "cartesian"
    assert bundle.scan_axes == {"alpha": [1.0, 2.0], "beta": [3.0, 4.0, 5.0]}
    assert bundle.n_traj_per_point == 4


def test_scheduler_service_describe_products_size_counts_referenced_files_only(
    tmp_path,
):
    session_root = tmp_path / "session"
    save_products(session_root / "job1", {"trajectories": _products_dataset()})
    service = SchedulerService(_system_config(tmp_path))
    size_before = service.describe_products("job1", session_dir=session_root).size

    (session_root / "job1" / "junk.bin").write_bytes(b"x" * 1024)

    catalog = service.describe_products("job1", session_dir=session_root)
    assert catalog.size == size_before
    assert catalog.size == sum(product.physical_nbytes for product in catalog.products)


def test_scheduler_service_describe_products_marks_missing_payload(tmp_path):
    session_root = tmp_path / "session"
    manifest = save_products(
        session_root / "job1", {"trajectories": _products_dataset()}
    )
    payload = next(iter(storage_referenced_files(manifest.products[0])))
    (session_root / "job1" / payload).unlink()

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    product = catalog.products[0]
    assert product.materializable is False
    assert payload in product.missing_reason
    assert product.physical_nbytes == 0
    assert catalog.size == 0


def test_scheduler_service_describe_products_has_no_resolver_side_effects(tmp_path):
    session_root = tmp_path / "session"
    save_products(session_root / "job1", {"trajectories": _products_dataset()})
    resolver = default_artifact_resolver()
    bindings_before = dict(resolver._bindings)

    SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    assert dict(resolver._bindings) == bindings_before


def test_scheduler_service_describe_products_rejects_removed_hash_field(tmp_path):
    session_root = tmp_path / "session"
    save_products(session_root / "job1", {"trajectories": _products_dataset()})
    manifest_path = session_root / "job1" / "artifact_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["content_hash"] = "removed"
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArtifactCorruptError):
        SchedulerService(_system_config(tmp_path)).describe_products(
            "job1", session_dir=session_root
        )


def test_scheduler_service_rejects_cross_product_payload_aliasing(tmp_path):
    session_root = tmp_path / "session"
    job_dir = session_root / "job1"
    save_products(
        job_dir,
        {
            "first": _products_dataset(),
            "second": _products_dataset(),
        },
        layout="single",
    )
    manifest_path = job_dir / "artifact_manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw["products"][1]["storage"] = raw["products"][0]["storage"]
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(ArtifactCorruptError, match="across products"):
        SchedulerService(_system_config(tmp_path)).describe_products(
            "job1", session_dir=session_root
        )


def test_scheduler_service_describe_products_marks_unknown_storage_adapter(tmp_path):
    session_root = tmp_path / "session"
    manifest = save_products(
        session_root / "job1", {"trajectories": _products_dataset()}
    )
    entry = manifest.products[0]
    storage = entry.storage.model_copy(update={"adapter": "unknown/9"})
    forged = entry.model_copy(update={"storage": storage})
    forged_manifest = manifest.model_copy(update={"products": [forged]})
    forged_manifest.write(session_root / "job1")

    catalog = SchedulerService(_system_config(tmp_path)).describe_products(
        "job1", session_dir=session_root
    )

    product = catalog.products[0]
    assert product.materializable is False
    assert "unknown/9" in product.missing_reason
    assert product.physical_nbytes == 0
    assert catalog.loader == "unknown/9"
    assert catalog.size == 0
