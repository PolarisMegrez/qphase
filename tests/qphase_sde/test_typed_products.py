"""Graph-ready typed analyser products (2.0 production output)."""

from types import SimpleNamespace

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.scan import ScanSpec
from qphase.data import (
    AxisRole,
    DataKind,
    ProductDeclaration,
    ProductSchema,
    SpectralDataset,
    StatisticsDataset,
    VariableSchema,
)
from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
from qphase_sde.analyser.coherence_carrier import CoherenceCarrierAnalyzer
from qphase_sde.analyser.coherence_matrix import CoherenceMatrixAnalyzer
from qphase_sde.analyser.moment_statistics import MomentStatisticsAnalyzer
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.analyser.quadratic_moments import QuadraticMomentAnalyzer
from qphase_sde.analyser.result import AnalysisResult
from qphase_sde.contracts.quantities import SDEQuantity
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama


class _OneModeModel:
    name = "typed_one_mode"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1

    def __init__(self):
        self._params = {"rate": 1.0}

    @property
    def params(self):
        return self._params

    def drift(self, y, t, params):
        del t
        return -np.asarray(params["rate"])[..., None] * y

    def diffusion(self, y, t, params):
        del t, params
        return np.ones(y.shape + (1,))


class _TwoModeModel(_OneModeModel):
    name = "typed_two_mode"
    n_modes = 2
    noise_dim = 2

    def diffusion(self, y, t, params):
        del t, params
        return np.ones(y.shape + (2,))


class _MeanAnalyzer:
    """Minimal analyser without a product builder (legacy bridge path)."""

    name = "mean"
    config = SimpleNamespace(expected_freq_max=None)

    def analyze(self, data, backend):
        del backend
        return AnalysisResult({"mean": float(np.mean(np.asarray(data.data)))})


class _ProductBuilderAnalyzer(_MeanAnalyzer):
    def __init__(
        self,
        product_name: str,
        *,
        graph_ready: bool = True,
        declared_kind: DataKind = DataKind.STATISTICS,
    ):
        self.product_name = product_name
        self.graph_ready = graph_ready
        self.declared_kind = declared_kind

    def output_spec(self):
        return ProductDeclaration(
            name=self.product_name,
            kind=self.declared_kind,
            quantity="mean",
            fields=["mean"],
        )

    def build_products(self, payload, *, scan_size, label):
        del payload, scan_size, label
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(
                    name="mean",
                    dtype="float64",
                    value_domain="real",
                    quantity="mean",
                )
            ],
            attributes={"graph_ready": self.graph_ready},
        )
        return {
            self.product_name: StatisticsDataset.from_arrays(
                schema, {"mean": np.asarray(1.0)}, owner="test"
            )
        }


def _grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.typed_one_mode.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()


def _two_mode_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.typed_two_mode.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()


def _run(
    model, analysers, *, scan=False, t1=0.2, n_traj=8, keep_traj=False
):
    n_modes = model.n_modes
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=t1,
            dt=0.01,
            n_traj=n_traj,
            seed=11,
            ic=[["1.0+0.0j"] * n_modes],
            keep_traj=keep_traj,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": analysers,
        },
    )
    grid = None
    if scan:
        grid = _two_mode_grid() if n_modes == 2 else _grid()
    return engine.run(
        context=SimpleNamespace(
            parameter_grid=grid, progress=None, cancellation=None
        )
    )


def test_psd_product_is_graph_ready_spectral():
    bundle = _run(_OneModeModel(), {"psd": PsdAnalyzer(kind="complex", modes=[0])})

    product = bundle.products["psd"]
    assert isinstance(product, SpectralDataset)
    assert product.kind is DataKind.SPECTRAL
    spectral = product.spectral_attributes
    assert spectral.sidedness == "two_sided"
    assert spectral.estimator == "periodogram"
    assert spectral.window == "rectangular"
    assert spectral.frequency_units == "inverse_time"

    roles = {axis.name: axis.role for axis in product.axes}
    assert roles["frequency"] is AxisRole.COORDINATE
    assert roles["channel"] is AxisRole.COMPONENT
    coordinates = {c.name: c for c in product.schema.coordinates}
    assert coordinates["frequency"].variable == "axis"
    assert coordinates["mode"].variable == "modes"

    psd = product.schema.variable("psd")
    assert psd.dims == ("frequency", "channel")
    assert psd.quantity == "power_spectral_density"
    assert psd.constraints.nonnegative
    uncertainties = {
        (u.target, u.kind): u for u in product.schema.uncertainties
    }
    assert uncertainties[("psd", "sample_std")].data_variable == "psd_std"
    assert uncertainties[("psd", "sem")].data_variable == "psd_sem"
    for uncertainty in uncertainties.values():
        assert uncertainty.sampling_basis == "trajectory"
        assert uncertainty.scope == "sampling"
    basis = product.schema.sampling_bases[0]
    assert basis.name == "trajectory"
    assert basis.count_variable == "uncertainty.n_independent"
    assert product.attributes["graph_ready"] is True
    assert bundle.bundle_descriptor.product_roles == {"primary_spectrum": "psd"}

    # The legacy view still rebuilds the original payload for downstream.
    legacy = bundle.legacy_result()
    payload = legacy.analysis["psd"]
    assert {"axis", "psd", "psd_std", "psd_sem", "modes"} <= set(payload)
    assert payload["axis"].shape == payload["psd"].shape[:1]


def test_psd_scan_product_stacks_over_scan_axis():
    bundle = _run(
        _OneModeModel(),
        {"psd": PsdAnalyzer(kind="complex", modes=[0])},
        scan=True,
    )

    product = bundle.products["psd"]
    assert isinstance(product, SpectralDataset)
    scan_axis = product.axis("scan")
    assert scan_axis.role is AxisRole.PARAMETER
    assert scan_axis.size == 2
    assert product.schema.variable("psd").dims == ("scan", "frequency", "channel")
    coordinates = {item.name: item for item in product.schema.coordinates}
    assert coordinates["frequency"].dims == ("frequency",)
    assert coordinates["mode"].dims == ("channel",)
    assert coordinates["rate"].dims == ("scan",)
    np.testing.assert_array_equal(product.coordinate("rate"), [1.0, 2.0])
    assert product.attributes["graph_ready"] is True

    point = bundle.point_view((1,))
    payload = point.legacy_result().analysis["psd"]
    assert payload["orientation"] in ("phase_decreasing", "phase_increasing")
    np.testing.assert_array_equal(payload["modes"], [0])


def test_psd_peak_metadata_isolated_from_graph_ready_spectrum():
    analyser = PsdAnalyzer(kind="complex", modes=[0])
    axis = np.linspace(-1.0, 1.0, 5)
    payload = {
        "axis": axis,
        "psd": np.ones((5, 1)),
        "psd_std": np.ones((5, 1)) * 0.1,
        "psd_sem": np.ones((5, 1)) * 0.05,
        "modes": [0],
        "orientation": "phase_decreasing",
        "uncertainty": {"n_independent": 4},
        "peaks": {0: {"center": 0.25, "linewidth": 0.1}},
    }
    products = analyser.build_products(payload, scan_size=1, label="psd")
    assert products is not None
    spectrum = products["psd"]
    assert "peaks" not in spectrum.attributes["payload_meta"]
    legacy = products["psd.legacy_peaks"]
    assert legacy.attributes["bridge"] == "legacy_peaks/1"
    assert legacy.attributes["graph_ready"] is False


def test_allan_product_is_graph_ready_statistics():
    bundle = _run(
        _OneModeModel(),
        {
            "allan_variance": AllanVarianceAnalyzer(
                modes=[0], points=5, min_windows=2, min_independent_windows=1
            )
        },
        t1=0.4,
    )

    product = bundle.products["allan_variance"]
    assert isinstance(product, StatisticsDataset)
    assert product.kind is DataKind.STATISTICS
    roles = {axis.name: axis.role for axis in product.axes}
    assert roles["tau"] is AxisRole.COORDINATE
    assert roles["trajectory"] is AxisRole.REALIZATION

    variance = product.schema.variable(
        "mode_results.0.allan.angular_frequency_variance"
    )
    assert variance.dims == ("tau",)
    assert variance.quantity == SDEQuantity.ALLAN_VARIANCE.value
    assert variance.constraints.nonnegative
    assert (
        product.schema.variable("mode_results.0.allan.per_trajectory").dims
        == ("trajectory", "tau")
    )
    uncertainties = {
        (u.target, u.kind): u for u in product.schema.uncertainties
    }
    sem = uncertainties[("mode_results.0.allan.angular_frequency_variance", "sem")]
    assert sem.data_variable == "mode_results.0.allan.angular_frequency_variance_sem"
    assert sem.sampling_basis == "trajectory"
    basis = product.schema.sampling_bases[0]
    assert basis.source_axis == "trajectory"
    coordinates = {c.name: c for c in product.schema.coordinates}
    assert coordinates["tau"].variable == "mode_results.0.allan.tau"
    assert product.attributes["estimator"] == "non_overlapping_windows"
    assert product.attributes["graph_ready"] is True


def test_moment_and_coherence_products_are_typed_over_scan():
    bundle = _run(
        _TwoModeModel(),
        {
            "quadratic_moments": QuadraticMomentAnalyzer(
                observables={
                    "n0": {"matrix": [[1.0, 0.0], [0.0, 0.0]]},
                    "n1": {"matrix": [[0.0, 0.0], [0.0, 1.0]]},
                },
                time_blocks=2,
                min_block_samples=2,
            ),
            "moment_statistics": MomentStatisticsAnalyzer(
                time_blocks=2, min_block_samples=2
            ),
            "coherence_matrix": CoherenceMatrixAnalyzer(
                time_blocks=2, min_block_samples=2
            ),
            "coherence_carrier": CoherenceCarrierAnalyzer(
                modes=[0], include_trace=True
            ),
        },
        scan=True,
        t1=0.4,
    )

    quadratic = bundle.products["quadratic_moments"]
    assert quadratic.attributes["graph_ready"] is True
    roles = {axis.name: axis.role for axis in quadratic.axes}
    assert roles["order"] is AxisRole.INDEX
    assert quadratic.axis("order").coordinate == "regular"
    assert (
        quadratic.schema.variable("raw_moments").quantity
        == SDEQuantity.MOMENTS.value
    )
    families = quadratic.attributes["moment_families"]
    assert families["raw_moments"]["moment_kind"] == "raw"
    assert families["cumulants"]["moment_kind"] == "cumulant"
    assert families["raw_moments"]["orders"] == [1, 2, 3, 4]

    statistics = bundle.products["moment_statistics"]
    assert statistics.attributes["graph_ready"] is True
    assert statistics.attributes["moment_family"]["orders"] == [1, 2, 4]
    assert statistics.attributes["normalized_variables"] == ["g2"]
    assert (
        statistics.schema.variable("g2").quantity == SDEQuantity.MOMENTS.value
    )
    assert statistics.schema.variable("g2").dims == (
        "scan",
        "channel",
        "channel_2",
    )

    matrix = bundle.products["coherence_matrix"]
    assert matrix.attributes["graph_ready"] is True
    matrix_variable = matrix.schema.variable("matrix")
    assert matrix_variable.quantity == "coherence_matrix"
    assert matrix_variable.constraints.symmetry == "hermitian"
    (uncertainty,) = matrix.schema.uncertainties
    assert uncertainty.target == "matrix"
    assert uncertainty.covariance == "real_imag"
    assert uncertainty.data_variable == "matrix_sem"

    carrier = bundle.products["coherence_carrier"]
    assert carrier.attributes["graph_ready"] is True
    assert (
        carrier.schema.variable("frequency").quantity
        == SDEQuantity.COHERENCE_FREQUENCY.value
    )
    carrier_roles = {axis.name: axis.role for axis in carrier.axes}
    assert carrier_roles["measurement"] is AxisRole.COMPONENT
    assert carrier_roles["lag"] is AxisRole.COORDINATE
    assert carrier.axis("lag").units == "time"

    for name, product in bundle.products.items():
        assert product.attributes["graph_ready"] is True, name
        coordinates = {item.name: item for item in product.schema.coordinates}
        assert coordinates["rate"].role == "parameter", name
        np.testing.assert_array_equal(product.coordinate("rate"), [1.0, 2.0])


def test_unmigrated_analyser_keeps_legacy_bridge():
    bundle = _run(_OneModeModel(), {"mean": _MeanAnalyzer()})

    product = bundle.products["mean"]
    assert isinstance(product, StatisticsDataset)
    assert product.attributes["bridge"] == "legacy_analysis/1"
    assert product.attributes["graph_ready"] is False


def test_analyser_products_cannot_collide_with_each_other_or_engine_products():
    with np.testing.assert_raises_regex(ValueError, "collides"):
        _run(
            _OneModeModel(),
            {
                "first": _ProductBuilderAnalyzer("shared"),
                "second": _ProductBuilderAnalyzer("shared"),
            },
        )
    with np.testing.assert_raises_regex(ValueError, "collides"):
        _run(
            _OneModeModel(),
            {"bad": _ProductBuilderAnalyzer("trajectories")},
            keep_traj=True,
        )


def test_analyser_products_must_be_graph_ready_and_match_declaration():
    with np.testing.assert_raises_regex(TypeError, "not graph-ready"):
        _run(
            _OneModeModel(),
            {"bad": _ProductBuilderAnalyzer("bad", graph_ready=False)},
        )
    with np.testing.assert_raises_regex(TypeError, "declared 'spectral'"):
        _run(
            _OneModeModel(),
            {
                "bad": _ProductBuilderAnalyzer(
                    "bad", declared_kind=DataKind.SPECTRAL
                )
            },
        )
