"""Tests for the 2.0 SDE data bundle (Phase 1 typed result products)."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.core.aggregation import DirectoryInputResult
from qphase.core.artifacts import ArtifactStore
from qphase.core.dataset import DatasetResultProtocol, ResultProtocol
from qphase.core.scan import ParameterGrid
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
from qphase_sde.errors import SDEExecutionError
from qphase_sde.result import (
    SDEDataBundle,
    SDEResult,
    _analysis_product,
    bundle_from_result,
    legacy_view_from_products,
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


def _traj_bundle(scan_size=1, n_traj=2, n_time=5, n_channel=1, analysis=None):
    fused = scan_size * n_traj
    data = np.arange(fused * n_time * n_channel, dtype=np.complex128)
    data = data.reshape(fused, n_time, n_channel)
    result = SDEResult(
        trajectory=_Trajectory(data),
        analysis=analysis or {},
        meta={"label": "test"},
    )
    return bundle_from_result(result, provenance=_provenance(), n_traj_per_point=n_traj)


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
    result = SDEResult(trajectory=_Trajectory(data, meta=meta))
    bundle = bundle_from_result(result, provenance=_provenance())
    traj = bundle.products["trajectories"]
    valid = traj.handle("valid_length").materialize()
    np.testing.assert_array_equal(valid, [[3, 5]])
    assert traj.attributes["warmup_steps"] == 2
    assert traj.attributes["note"] == "ok"


def test_analysis_product_single_point_dict_payload():
    payload = {
        "axis": np.arange(4.0),
        "psd": np.ones(4),
        "sample_dt": np.float64(0.5),
        "orientation": "phase_decreasing",
        "uncertainty": {"field": "psd_sem"},
    }
    product = _analysis_product("psd", payload, scan_size=None)
    assert product is not None
    assert product.kind is DataKind.STATISTICS
    assert product.attributes["bridge"] == "legacy_analysis/1"
    assert product.attributes["payload_meta"]["orientation"] == "phase_decreasing"
    assert product.attributes["payload_meta"]["uncertainty.field"] == "psd_sem"
    np.testing.assert_array_equal(
        product.handle("sample_dt").materialize(), np.float64(0.5)
    )


def test_analysis_product_flattens_nested_dicts():
    payload = {
        "distributions": {
            0: {"hist": np.arange(4.0), "type": "1d_real"},
            1: {"hist": np.arange(4.0) + 1, "type": "1d_real"},
        },
        "bins": 16,
    }
    product = _analysis_product("dist", payload, scan_size=None)
    assert product is not None
    names = {variable.name for variable in product.variables}
    assert "distributions.0.hist" in names
    assert "distributions.1.hist" in names
    assert product.attributes["payload_meta"]["distributions.0.type"] == "1d_real"


def test_analysis_product_scan_stacks_per_point_payloads():
    payload = [{"axis": np.arange(3.0), "psd": np.full(3, index)} for index in range(3)]
    product = _analysis_product("psd", payload, scan_size=3)
    assert product is not None
    psd = product.handle("psd").materialize()
    assert psd.shape == (3, 3)
    np.testing.assert_array_equal(psd[1], np.full(3, 1))
    scan_axis = product.axis("scan")
    assert scan_axis.size == 3


def test_analysis_product_scan_demotes_ragged_leaves_to_per_point_meta():
    payload = [
        {"axis": np.arange(3.0), "peaks": np.arange(float(count))}
        for count in (1, 2, 3)
    ]
    product = _analysis_product("psd", payload, scan_size=3)
    assert product is not None
    assert "peaks" not in {variable.name for variable in product.variables}
    assert product.attributes["per_point_meta"] == ["peaks"]
    assert product.attributes["payload_meta"]["peaks"][1] == [0.0, 1.0]


def test_analysis_product_scan_rejects_inconsistent_keys():
    payload = [{"a": np.ones(2)}, {"b": np.ones(2)}]
    with pytest.raises(TypeError, match="keys differ"):
        _analysis_product("bad", payload, scan_size=2)


def test_analysis_product_skips_none_and_empty():
    assert _analysis_product("none", None, scan_size=None) is None
    assert _analysis_product("empty", {}, scan_size=None) is None


def test_analysis_product_singleton_scan_keeps_size_one_scan_axis():
    """Cardinality freeze: a genuine 1-point scan keeps a scan axis of
    exactly size 1 instead of being demoted to a non-scan product.
    """
    payload = {
        "axis": np.arange(4.0),
        "psd": np.ones(4),
        "orientation": "phase_decreasing",
    }
    product = _analysis_product("psd", payload, scan_size=1)
    assert product is not None
    assert product.axis("scan").size == 1
    assert product.schema.variable("psd").dims == ("scan", "psd.dim0")
    np.testing.assert_array_equal(product.handle("psd").materialize(), np.ones((1, 4)))
    # Scan products keep meta leaves as per-point lists, like larger scans.
    assert product.attributes["payload_meta"]["orientation"] == ["phase_decreasing"]
    assert "orientation" in product.attributes["per_point_meta"]
    # The per-point view drops the scan axis and rebuilds the bare payload.
    point = product.point_view(scan=0)
    restored = legacy_view_from_products({"psd": point}, meta={"scan_index": 0})
    assert restored.analysis["psd"]["orientation"] == "phase_decreasing"
    np.testing.assert_array_equal(restored.analysis["psd"]["psd"], np.ones(4))


def test_bundle_fail_fast_when_registered_analyser_builds_nothing():
    """A registered analyser payload that builds no product fails the bundle
    instead of being silently dropped.
    """

    class _NoneBuilder:
        def build_products(self, payload, *, scan_size, label):
            del payload, scan_size, label
            return None

    class _EmptyBuilder:
        def build_products(self, payload, *, scan_size, label):
            del payload, scan_size, label
            return {}

    result = SDEResult(analysis={"bad": {"value": np.ones(2)}})
    for analyser in (_NoneBuilder(), _EmptyBuilder()):
        with pytest.raises(SDEExecutionError, match="'bad'"):
            bundle_from_result(
                result, provenance=_provenance(), analysers={"bad": analyser}
            )

    # A builder-less analyser whose payload has no numeric leaves fails too.
    metadata_only = SDEResult(analysis={"bad": {"note": "no numbers"}})
    with pytest.raises(SDEExecutionError, match="'bad'"):
        bundle_from_result(
            metadata_only,
            provenance=_provenance(),
            analysers={"bad": object()},
        )

    # A registered analyser that produced no payload at all fails as well.
    empty_payload = SDEResult(analysis={"bad": None})
    with pytest.raises(SDEExecutionError, match="'bad'"):
        bundle_from_result(
            empty_payload,
            provenance=_provenance(),
            analysers={"bad": object()},
        )


def test_bundle_fail_fast_on_scan_cardinality_mismatch():
    """A built product whose scan axis contradicts the job scan size fails."""

    class _FixedProductBuilder:
        def __init__(self, product):
            self._product = product

        def build_products(self, payload, *, scan_size, label):
            del payload, scan_size
            return {label: self._product}

    two_point = _analysis_product("bad", [{"value": 1.0}, {"value": 2.0}], scan_size=2)
    assert two_point is not None
    # No parameter grid, yet the builder emitted a scan product.
    with pytest.raises(SDEExecutionError, match="scan axis"):
        bundle_from_result(
            SDEResult(analysis={"bad": {"value": 1.0}}),
            provenance=_provenance(),
            analysers={"bad": _FixedProductBuilder(two_point)},
        )
    # A 2-point scan whose builder emitted a 3-point scan product.
    three_point = _analysis_product(
        "bad", [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}], scan_size=3
    )
    assert three_point is not None
    grid = _grid(kappa=[1.0, 2.0])
    with pytest.raises(SDEExecutionError, match="scan axis"):
        bundle_from_result(
            _ScanResultLike(grid, SDEResult(analysis={"bad": [{"value": 1.0}] * 2}), 1),
            provenance=_provenance(),
            analysers={"bad": _FixedProductBuilder(three_point)},
        )


def test_observer_payload_without_numeric_leaves_lands_in_metadata():
    """Observer/metadata payloads (names outside the analysers mapping) are
    not products, but they are kept in the bundle metadata under their
    plugin name instead of being dropped.
    """
    result = SDEResult(
        analysis={
            "first_passage": {"orientation": "below", "rule": "state_norm"},
        }
    )
    bundle = bundle_from_result(result, provenance=_provenance(), analysers={})

    assert "first_passage" not in bundle.products
    assert "dropped_products" not in bundle.metadata
    routed = bundle.metadata["observer_payloads"]["first_passage"]
    assert routed == {"orientation": "below", "rule": "state_norm"}
    provenance_meta = bundle.manifest_provenance["meta"]
    assert provenance_meta["observer_payloads"]["first_passage"] == routed

    # Unregistered payloads also route when no analysers mapping is given.
    bundle = bundle_from_result(result, provenance=_provenance())
    assert "first_passage" not in bundle.products
    assert "dropped_products" not in bundle.metadata
    assert bundle.metadata["observer_payloads"]["first_passage"] == routed


def test_bundle_satisfies_all_protocols():
    bundle = _traj_bundle()
    assert isinstance(bundle, SDEDataBundleProtocol)
    assert isinstance(bundle, ResultProtocol)
    assert isinstance(bundle, DatasetResultProtocol)
    assert bundle.metadata["label"] == "test"
    assert "provenance" in bundle.metadata


def test_analyser_output_inherits_source_scan_identity(tmp_path):
    """A scan-free analyze job preserves the upstream dataset scan."""
    from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
    from qphase_sde.engine import Engine, EngineConfig

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
        scan_axes={"epsilon": np.arange(4.0)},
        scan_shape=(4,),
        meta={"scan_combine": "zipped"},
    )
    analyzer = PeaksAnalyzer(PeaksAnalyzerConfig(scan_param="epsilon", finder="scipy"))
    engine = Engine(
        config=EngineConfig(mode="analyze"),
        plugins={"analyser": {"peaks": analyzer}},
    )

    result = engine.run(source)

    assert result.shape == (4,)
    product = result.products["peaks"]
    assert product.axis("scan").size == 4
    assert product.coordinate("epsilon").shape == (4,)
    assert result.bundle_descriptor.descriptor["scan"]["shape"] == [4]

    source.save(tmp_path / "source")
    restored_result = engine.run(DirectoryInputResult(path=tmp_path / "source"))
    assert restored_result.shape == (4,)
    assert restored_result.products["peaks"].axis("scan").size == 4


def test_bundle_require_filters_by_kind_quantity_fields():
    analysis = {"psd": {"axis": np.arange(3.0), "psd": np.ones(3)}}
    bundle = _traj_bundle(analysis=analysis)
    by_kind = bundle.require(kind=DataKind.TIME_SERIES)
    assert sorted(by_kind) == ["trajectories"]
    by_quantity = bundle.require(quantity="field_amplitude")
    assert sorted(by_quantity) == ["trajectories"]
    by_fields = bundle.require(fields=("alpha", "valid_length"))
    assert sorted(by_fields) == ["trajectories"]
    assert bundle.require(fields=("nonexistent",)) == {}


def _grid(**axes) -> ParameterGrid:
    """Build a cartesian ParameterGrid from axis value lists."""
    arrays = {name: np.asarray(values, dtype=float) for name, values in axes.items()}
    return ParameterGrid(
        combine="cartesian",
        axes=arrays,
        targets={name: "model" for name in arrays},
        shape=tuple(values.size for values in arrays.values()),
    )


def test_scan_bundle_axes_shape_and_point_view():
    grid = _grid(kappa=[1.0, 2.0, 3.0])
    result = SDEResult(trajectory=_Trajectory(np.zeros((6, 5, 1), dtype=np.complex128)))
    bundle = bundle_from_result(
        _ScanResultLike(grid, result, n_traj_per_point=2),
        provenance=_provenance(),
    )
    assert bundle.shape == (3,)
    np.testing.assert_array_equal(bundle.axes["kappa"], [1.0, 2.0, 3.0])
    point = bundle.point_view((1,))
    assert point.metadata["scan_index"] == 1
    assert point.metadata["scan_point"]["kappa"] == pytest.approx(2.0)
    alpha = point.products["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (2, 5, 1)


class _ScanResultLike:
    """Minimal SDEScanResult stand-in (grid + combined result)."""

    def __init__(self, grid, combined, n_traj_per_point):
        self.grid = grid
        self.combined = combined
        self.n_traj_per_point = n_traj_per_point
        self.meta = {}


def test_bundle_save_and_load_roundtrip(tmp_path):
    analysis = {"psd": {"axis": np.arange(4.0), "sample_dt": np.float64(0.5)}}
    bundle = _traj_bundle(analysis=analysis)
    bundle.save(tmp_path)
    loaded = load_products(tmp_path)
    assert sorted(loaded) == ["psd", "trajectories"]
    alpha = loaded["trajectories"].handle("alpha").materialize()
    assert alpha.shape == (1, 2, 5, 1)
    assert float(loaded["psd"].handle("sample_dt").materialize()) == 0.5


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


def test_legacy_result_view_roundtrip():
    analysis = {
        "psd": {
            "axis": np.arange(3.0),
            "orientation": "phase_decreasing",
        },
        "dist": {
            "distributions": {0: {"hist": np.arange(4.0), "type": "1d_real"}},
            "bins": 16,
        },
    }
    bundle = _traj_bundle(analysis=analysis)
    legacy = bundle.legacy_result()
    assert isinstance(legacy, SDEResult)
    assert legacy.trajectory.data.shape == (2, 5, 1)
    assert legacy.meta["label"] == "test"
    np.testing.assert_array_equal(legacy.analysis["psd"]["axis"], np.arange(3.0))
    assert legacy.analysis["psd"]["orientation"] == "phase_decreasing"
    dist = legacy.analysis["dist"]["distributions"][0]
    np.testing.assert_array_equal(dist["hist"], np.arange(4.0))
    assert dist["type"] == "1d_real"
    assert legacy.analysis["dist"]["bins"] == 16


def test_legacy_result_view_scan_per_point_meta():
    payload = [
        {"axis": np.arange(2.0), "peaks": np.arange(float(count))}
        for count in (1, 2, 3)
    ]
    grid = _grid(kappa=[1.0, 2.0, 3.0])
    result = SDEResult(analysis={"psd": payload})
    bundle = bundle_from_result(
        _ScanResultLike(grid, result, n_traj_per_point=None),
        provenance=_provenance(),
    )
    point = bundle.point_view((2,))
    legacy = point.legacy_result()
    assert legacy.analysis["psd"]["peaks"] == [0.0, 1.0, 2.0]


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
