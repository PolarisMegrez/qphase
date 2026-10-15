"""Graph-ready typed analyser products (2.0 production output)."""

from types import SimpleNamespace

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.scan import ScanSpec
from qphase.data import (
    AxisRole,
    DataKind,
    ExternalInputBinding,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
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
from qphase_sde.contracts.quantities import SDEQuantity
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.errors import SDEExecutionError
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
    """Minimal typed analyser used to exercise engine product validation."""

    name = "mean"
    config = SimpleNamespace(expected_freq_max=None)

    def input_requirements(self):
        from qphase_sde.analyser.base import trajectory_requirement

        return [trajectory_requirement(value_domain=None)]

    def output_specs(self, *, label):
        return (
            ProductDeclaration(
                name=label, kind=DataKind.STATISTICS, quantity="mean", fields=["mean"]
            ),
        )

    def execute(self, inputs, *, backend, label, metadata=None, context=None):
        del inputs, backend, metadata, context
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(
                    name="mean", dtype="float64", value_domain="real", quantity="mean"
                )
            ],
            attributes={"graph_ready": True},
        )
        return {
            label: StatisticsDataset.from_arrays(
                schema, {"mean": np.asarray(1.0)}, owner="test"
            )
        }


class _ProductBuilderAnalyzer(_MeanAnalyzer):
    def __init__(
        self,
        product_name: str,
        *,
        graph_ready: bool = True,
        declared_kind: DataKind = DataKind.STATISTICS,
        declared_name: str | None = None,
        extra_declarations: tuple[ProductDeclaration, ...] = (),
    ):
        self.product_name = product_name
        self.graph_ready = graph_ready
        self.declared_kind = declared_kind
        self.declared_name = declared_name or product_name
        self.extra_declarations = extra_declarations

    def output_specs(self, *, label):
        return (
            ProductDeclaration(
                name=self.declared_name,
                kind=self.declared_kind,
                quantity="mean",
                fields=["mean"],
            ),
            *self.extra_declarations,
        )

    def execute(self, inputs, *, backend, label, metadata=None, context=None):
        del inputs, backend, metadata, context
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


def _psd_analyzer() -> PsdAnalyzer:
    return PsdAnalyzer(
        measurement={"complex_field": {"channels": [0], "representation": "complex"}}
    )


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


def _run(model, analysers, *, scan=False, t1=0.2, n_traj=8, keep_traj=False, grid=None):
    n_modes = model.n_modes
    nodes = [
        ProductNode(
            node_id=f"analyser.{name}",
            producer=f"analyser:{name}",
            requirements=list(analyser.input_requirements()),
            outputs=list(analyser.output_specs(label=name)),
        )
        for name, analyser in analysers.items()
    ]
    bindings = [
        ExternalInputBinding(
            consumer=node.node_id,
            requirement=requirement.name,
            source="input",
            product="trajectories",
        )
        for node in nodes
        for requirement in node.requirements
    ]
    declared_names = [output.name for node in nodes for output in node.outputs]
    if len(declared_names) != len(set(declared_names)) or (
        keep_traj and "trajectories" in declared_names
    ):
        raise ValueError("analyser product collides with another product")
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
        product_graph=ProductGraph(nodes=nodes, external_inputs=bindings),
    )
    if grid is None and scan:
        grid = _two_mode_grid() if n_modes == 2 else _grid()
    result = engine.run(
        context=SimpleNamespace(parameter_grid=grid, progress=None, cancellation=None)
    )
    for product in result.products.values():
        if product.attributes.get("graph_ready") is False:
            raise TypeError("analyser product is not graph-ready")
        if grid is not None and "scan" not in {axis.name for axis in product.axes}:
            raise SDEExecutionError("scan analyser product has no scan axis")
    return result


def _singleton_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.typed_one_mode.rate",
                    "values": [1.0],
                }
            }
        }
    ).compile()


def test_psd_product_is_graph_ready_spectral():
    bundle = _run(_OneModeModel(), {"psd": _psd_analyzer()})

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
    uncertainties = {(u.target, u.kind): u for u in product.schema.uncertainties}
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

    assert {"axis", "psd", "psd_std", "psd_sem", "modes"} <= {
        variable.name for variable in product.variables
    }


def test_psd_scan_product_stacks_over_scan_axis():
    bundle = _run(
        _OneModeModel(),
        {"psd": _psd_analyzer()},
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

    point = bundle.point_view((1,)).products["psd"]
    assert point.attributes["orientation"] in ("phase_decreasing", "phase_increasing")
    np.testing.assert_array_equal(point.handle("modes").materialize(), [0])


def test_psd_does_not_build_inline_peak_products():
    bundle = _run(_OneModeModel(), {"psd": _psd_analyzer()})
    spectrum = bundle.products["psd"]
    products = bundle.products
    assert set(products) == {"psd"}
    assert "peaks" not in spectrum.attributes.get("payload_meta", {})


def test_typed_reducer_merge_and_checkpoint_are_portable():
    analyser = _psd_analyzer()

    def product(offset: float):
        from qphase_sde.result import _trajectory_product
        from qphase_sde.state import TrajectorySet

        values = np.full((2, 16, 1), 1.0 + offset, dtype=complex)
        trajectory = _trajectory_product(
            TrajectorySet(values, t0=0.0, dt=0.1),
            scan_size=None,
            n_traj_per_point=None,
        )
        return analyser.execute(
            {"signal": trajectory}, backend=NumpyBackend(), label="psd"
        )["psd"]

    reducer = analyser.reducer()
    assert reducer is not None
    left = reducer.prepare(None, None)
    right = reducer.prepare(None, None)
    reducer.consume(product(0.0), left, None)
    reducer.consume(product(1.0), right, None)
    merged = reducer.merge(left, right)
    restored = reducer.restore(reducer.checkpoint(merged))
    output = reducer.finalize(restored)
    np.testing.assert_allclose(
        output["psd"].handle("psd").materialize(),
        (
            product(0.0).handle("psd").materialize()
            + product(1.0).handle("psd").materialize()
        )
        / 2,
    )


def test_allan_product_is_graph_ready_statistics():
    bundle = _run(
        _OneModeModel(),
        {
            "allan_variance": AllanVarianceAnalyzer(
                measurement={
                    "phase_frequency": {"channels": [0], "min_independent_windows": 1}
                },
                points=5,
                min_windows=2,
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
    assert product.schema.variable("mode_results.0.allan.per_trajectory").dims == (
        "trajectory",
        "tau",
    )
    uncertainties = {(u.target, u.kind): u for u in product.schema.uncertainties}
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
        quadratic.schema.variable("raw_moments").quantity == SDEQuantity.MOMENTS.value
    )
    families = quadratic.attributes["moment_families"]
    assert families["raw_moments"]["moment_kind"] == "raw"
    assert families["cumulants"]["moment_kind"] == "cumulant"
    assert families["raw_moments"]["orders"] == [1, 2, 3, 4]

    statistics = bundle.products["moment_statistics"]
    assert statistics.attributes["graph_ready"] is True
    assert statistics.attributes["moment_family"]["orders"] == [1, 2, 4]
    assert statistics.attributes["normalized_variables"] == ["g2"]
    assert statistics.schema.variable("g2").quantity == SDEQuantity.MOMENTS.value
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
    # The conditional uncertainty/confidence payload is typed alongside the
    # sampling SEM: per-(scan, measurement) variables closing on declared
    # uncertainty entries.
    for name in ("conditional_std", "confidence_lower", "confidence_upper"):
        assert carrier.schema.variable(name).dims == ("scan", "measurement"), name
    carrier_uncertainties = {
        (u.target, u.scope, u.data_variable) for u in carrier.schema.uncertainties
    }
    assert ("frequency", "conditional", "conditional_std") in carrier_uncertainties
    assert ("frequency", "sampling", "frequency_sem") in carrier_uncertainties
    lower = carrier.handle("confidence_lower").materialize()
    upper = carrier.handle("confidence_upper").materialize()
    finite = np.isfinite(lower) & np.isfinite(upper)
    assert np.all(lower[finite] <= upper[finite])
    center = carrier.handle("frequency").materialize()
    assert np.all(lower[finite] <= center[finite])
    assert np.all(center[finite] <= upper[finite])

    for name, product in bundle.products.items():
        assert product.attributes["graph_ready"] is True, name
        coordinates = {item.name: item for item in product.schema.coordinates}
        assert coordinates["rate"].role == "parameter", name
        np.testing.assert_array_equal(product.coordinate("rate"), [1.0, 2.0])


def test_minimal_typed_analyser_produces_graph_ready_product():
    bundle = _run(_OneModeModel(), {"mean": _MeanAnalyzer()})

    product = bundle.products["mean"]
    assert isinstance(product, StatisticsDataset)
    assert product.attributes["graph_ready"] is True


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
    with np.testing.assert_raises_regex(TypeError, "expected 'spectral'"):
        _run(
            _OneModeModel(),
            {"bad": _ProductBuilderAnalyzer("bad", declared_kind=DataKind.SPECTRAL)},
        )


def test_singleton_scan_product_carries_scan_axis():
    """A genuine 1-point scan closes the producer's size-1 scan axis."""
    result = _run(
        _OneModeModel(),
        {"bad": _ProductBuilderAnalyzer("bad")},
        grid=_singleton_grid(),
    )
    assert result.products["bad"].axis("scan").size == 1


def test_builder_products_must_be_declared():
    with np.testing.assert_raises_regex(ValueError, "expected"):
        _run(
            _OneModeModel(),
            {"bad": _ProductBuilderAnalyzer("bad", declared_name="other")},
        )


def test_builder_must_return_all_declared_outputs_atomically():
    analyser = _ProductBuilderAnalyzer(
        "bad",
        extra_declarations=(
            ProductDeclaration(
                name="bad.extra", kind=DataKind.STATISTICS, fields=["extra"]
            ),
        ),
    )
    with np.testing.assert_raises_regex(ValueError, "expected"):
        _run(_OneModeModel(), {"bad": analyser})
