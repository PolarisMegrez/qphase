"""Tests for the 2.0 SDE data bundle (Phase 1 typed result products)."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.core.artifacts import ArtifactStore
from qphase.core.dataset import DatasetResultProtocol, ResultProtocol
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    SpectralDataset,
    VariableSchema,
    load_products,
)
from qphase_sde.contracts.bundle import SDEDataBundleProtocol, SDEProvenance
from qphase_sde.result import (
    SDEDataBundle,
    trajectory_product,
)

pytestmark = pytest.mark.integration


def _provenance(**overrides) -> SDEProvenance:
    payload = {"t0": 0.0, "dt": 0.1, "master_seed": 7}
    payload.update(overrides)
    return SDEProvenance(**payload)


class _Trajectory:
    """Minimal TrajectorySet stand-in for product construction tests."""

    def __init__(self, data, t0=0.0, dt=0.1, meta=None):
        self.data = np.asarray(data)
        self.t0 = t0
        self.dt = dt
        self.meta = meta or {}


def _traj_bundle(scan_size=1, n_traj=2, n_time=5, n_channel=1):
    fused = scan_size * n_traj
    data = np.arange(fused * n_time * n_channel, dtype=np.complex128)
    data = data.reshape(fused, n_time, n_channel)
    product = trajectory_product(
        _Trajectory(data),
        scan_size=scan_size,
        n_traj_per_point=n_traj,
    )
    return SDEDataBundle(
        {"trajectories": product},
        _provenance(),
        meta={"label": "test"},
        n_traj_per_point=n_traj,
    )


def test_trajectory_product_structure():
    bundle = _traj_bundle(scan_size=1, n_traj=2, n_time=5, n_channel=2)
    traj = bundle.products["trajectories"]
    assert traj.kind is DataKind.TIME_SERIES
    alpha = traj.handle("alpha").materialize()
    assert alpha.shape == (1, 2, 5, 2)
    assert alpha.dtype == np.complex128
    time_axis = traj.axis("time")
    assert time_axis.start == pytest.approx(0.0)
    assert time_axis.step == pytest.approx(0.1)
    valid = traj.handle("valid_length").materialize()
    np.testing.assert_array_equal(valid, np.full((1, 2), 5))


def test_trajectory_product_records_valid_lengths_and_meta():
    meta = {"valid_length": [3, 5], "warmup_steps": 2, "note": "ok"}
    data = np.zeros((2, 5, 1), dtype=np.complex128)
    product = trajectory_product(
        _Trajectory(data, meta=meta), scan_size=None, n_traj_per_point=None
    )
    bundle = SDEDataBundle({"trajectories": product}, _provenance())
    traj = bundle.products["trajectories"]
    valid = traj.handle("valid_length").materialize()
    np.testing.assert_array_equal(valid, [[3, 5]])
    assert traj.attributes["warmup_steps"] == 2
    assert traj.attributes["note"] == "ok"


def test_bundle_satisfies_all_protocols():
    bundle = _traj_bundle()
    assert isinstance(bundle, SDEDataBundleProtocol)
    assert isinstance(bundle, ResultProtocol)
    assert isinstance(bundle, DatasetResultProtocol)
    assert bundle.metadata["label"] == "test"
    assert "provenance" in bundle.metadata


def test_analyser_output_inherits_source_scan_identity(tmp_path):
    """A scan-free analyze job preserves the upstream dataset scan."""
    from qphase.core.registry import discovery, registry
    from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
    from qphase_sde.engine import Engine, EngineConfig
    from qphase_sde.graph import compile_product_graph

    axis = np.linspace(-1.0, 1.0, 101)
    payload = [
        {
            "axis": axis,
            "psd": np.exp(-(((axis - 0.1 * index) / 0.08) ** 2))[:, None],
            "modes": [0],
            "orientation": "phase_decreasing",
        }
        for index in range(4)
    ]
    source_schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=4),
            AxisSchema(name="frequency", role=AxisRole.COORDINATE, size=101),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=1),
        ],
        variables=[
            VariableSchema(
                name="axis",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency"),
            ),
            VariableSchema(
                name="psd",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency", "channel"),
            ),
        ],
        attributes={
            "frequency_units": "inverse_time",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "density",
            "window": "none",
            "estimator": "periodogram",
        },
    )
    source_product = SpectralDataset.from_arrays(
        source_schema,
        {
            "axis": np.stack([item["axis"] for item in payload]),
            "psd": np.stack([item["psd"] for item in payload]),
        },
        owner="test",
    )
    source = SDEDataBundle(
        {"psd": source_product},
        _provenance(),
        scan_axes={"x": np.asarray([0.0, 1.0]), "y": np.asarray([10.0, 20.0])},
        scan_shape=(2, 2),
        meta={"scan_combine": "cartesian"},
    )
    from qphase_sde.peak_finder import ScipyPeakFinder

    analyzer = PeaksAnalyzer(
        PeaksAnalyzerConfig(scan_param="x"),
        subplugins={"finder": ScipyPeakFinder()},
    )
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={
            "analyser": {"peaks": {"scan_param": "epsilon", "finder": {"scipy": {}}}}
        },
        registry=registry,
    )
    engine = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"peaks": analyzer}},
        product_graph=graph,
    )

    result = engine.run(source)

    assert result.shape == (2, 2)
    product = result.products["peaks"]
    assert product.axis("scan").size == 4
    np.testing.assert_array_equal(product.coordinate("x"), [0.0, 0.0, 1.0, 1.0])
    np.testing.assert_array_equal(product.coordinate("y"), [10.0, 20.0, 10.0, 20.0])
    assert result.bundle_descriptor.descriptor["scan"]["shape"] == [2, 2]

    source.save(tmp_path / "source")
def test_bundle_require_filters_by_kind_quantity_fields():
    bundle = _traj_bundle()
    by_kind = bundle.require(kind=DataKind.TIME_SERIES)
    assert sorted(by_kind) == ["trajectories"]
    by_quantity = bundle.require(quantity="field_amplitude")
    assert sorted(by_quantity) == ["trajectories"]
    by_fields = bundle.require(fields=("alpha", "valid_length"))
    assert sorted(by_fields) == ["trajectories"]
    assert bundle.require(fields=("nonexistent",)) == {}


def test_scan_bundle_axes_shape_and_point_view():
    product = trajectory_product(
        _Trajectory(np.zeros((6, 5, 1), dtype=np.complex128)),
        scan_size=3,
        n_traj_per_point=2,
    )
    bundle = SDEDataBundle(
        {"trajectories": product},
        _provenance(),
        scan_axes={"kappa": np.asarray([1.0, 2.0, 3.0])},
        scan_shape=(3,),
        n_traj_per_point=2,
    )
    assert bundle.shape == (3,)
    np.testing.assert_array_equal(bundle.axes["kappa"], [1.0, 2.0, 3.0])
    point = bundle.point_view((1,))
    assert point.metadata["scan_index"] == 1
    assert point.metadata["scan_point"]["kappa"] == pytest.approx(2.0)
    alpha = point.products["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (2, 5, 1)


def test_bundle_save_and_load_roundtrip(tmp_path):
    bundle = _traj_bundle()
    bundle.save(tmp_path)
    loaded = load_products(tmp_path)
    assert sorted(loaded) == ["trajectories"]
    alpha = loaded["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (1, 2, 5, 1)


def test_bundle_save_dataset_report(tmp_path):
    bundle = _traj_bundle()
    report = bundle.save_dataset(tmp_path, layout="single", shard_target_bytes=1 << 20)
    assert report.layout == "single"
    assert any(path.name == "artifact_manifest.json" for path in report.files)


def test_artifact_store_save_result_writes_v3_manifest(tmp_path):
    from types import SimpleNamespace

    bundle = _traj_bundle()
    store = ArtifactStore(tmp_path, SimpleNamespace(shard_target_mib=1))
    manifest_path = store.save_result(bundle, "job")
    assert manifest_path.name == "artifact_manifest.json"
    assert manifest_path.exists()
    loaded = load_products(tmp_path)
    assert "trajectories" in loaded


def test_engine_run_returns_bundle():
    from qphase.backend.numpy_backend import NumpyBackend
    from qphase_sde.engine import Engine, EngineConfig
    from qphase_sde.integrator.euler_maruyama import EulerMaruyama

    class _Model:
        name = "bundle_test_model"
        n_modes = 1
        noise_basis = "real"
        noise_dim = 1
        params: dict = {}

        def drift(self, y, t, p):
            return -y

        def diffusion(self, y, t, p):
            return np.ones((y.shape[0], 1, 1))

    engine = Engine(
        config=EngineConfig(dt=0.01, t0=0.0, t1=0.05, n_traj=2, seed=3, ic=[[0.0]]),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": _Model(),
        },
    )
    bundle = engine.run()
    assert isinstance(bundle, SDEDataBundle)
    traj = bundle.products["trajectories"]
    alpha = traj.handle("alpha").materialize()
    assert alpha.shape[0] == 1  # single scan point
    assert alpha.shape[1] == 2  # n_traj
    assert bundle.provenance.master_seed == 3
