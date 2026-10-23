"""Sampled-scalar PSD measurement: references, schema and binding boundaries.

The sampled-scalar measurement consumes the real ``value`` channel of a
sampled-observable product by channel name, detrends, applies the shared
spectral estimator, folds to the one-sided axis and publishes an unsigned
spectral product. Numerical fixtures mirror the frozen math baselines of
``test_sampled_scalar_math.py``; here they are exercised end-to-end through
the analyser, the typed product assembler and the result accumulator.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductSchema,
    TimeSeriesDataset,
    VariableSchema,
)
from qphase.data.graph import requirement_matches
from qphase_sde.analyser.frequency_orientation import (
    resolve_frequency_orientation,
    resolve_spectral_orientation,
)
from qphase_sde.analyser.psd import (
    PsdAnalyser,
    PsdAnalyserConfig,
    PsdResultAccumulator,
)
from qphase_sde.contracts.quantities import SDEQuantity

BACKEND = NumpyBackend()


def _observable_product(
    values: np.ndarray,
    *,
    dt: float,
    names: list[str] | None = None,
    units: str = "occupation",
    valid_length: np.ndarray | None = None,
):
    """Sampled-observable input product from a (traj, time, channel) array."""
    values = np.asarray(values, dtype=np.float64)
    n_traj, n_time, n_channel = values.shape
    if names is None:
        names = [f"n{index}" for index in range(n_channel)]
    axes = [
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=n_traj),
        AxisSchema(
            name="time",
            role=AxisRole.COORDINATE,
            size=n_time,
            coordinate="regular",
            start=0.0,
            step=dt,
            units="time",
        ),
        AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
    ]
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=axes,
        variables=[
            VariableSchema(
                name="value",
                dtype="<f8",
                value_domain="real",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
                units=units,
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
        attributes={"channel_names": list(names)},
    )
    return TimeSeriesDataset.from_arrays(
        schema,
        {
            "value": values,
            "valid_length": np.asarray(
                valid_length
                if valid_length is not None
                else np.full(n_traj, n_time),
                dtype=np.int64,
            ),
        },
        owner="test",
    )


def _scalar_analyzer(
    channels=("n0",), *, convention="pragmatic", detrend="none"
) -> PsdAnalyser:
    return PsdAnalyser(
        measurement={
            "sampled_scalar": {
                "channels": list(channels),
                "detrend": detrend,
            }
        },
        convention=convention,
    )


def _execute(analyzer: PsdAnalyser, product: TimeSeriesDataset):
    return analyzer.execute({"signal": product}, backend=BACKEND, label="psd")["psd"]


def test_scalar_psd_of_constant_folds_to_dc_bin():
    """A constant observable folds all power into DC: level^2 * N * dt."""
    n_traj, n_time, dt, level = 2, 256, 0.5, 1.7
    values = np.full((n_traj, n_time, 1), level)
    product = _execute(_scalar_analyzer(), _observable_product(values, dt=dt))

    axis = np.asarray(product.handle("axis").materialize())
    psd = np.asarray(product.handle("psd").materialize())[:, 0]
    assert psd[0] == pytest.approx(level**2 * n_time * dt, rel=1e-12)
    np.testing.assert_allclose(psd[1:], 0.0, atol=1e-24)
    # Parseval: the one-sided density integrates to the mean-square level.
    step = float(axis[1] - axis[0])
    assert float(psd.sum()) * step == pytest.approx(level**2, rel=1e-12)
    assert np.all(axis >= 0.0)


def test_scalar_psd_of_sinusoid_peaks_at_its_bin():
    n_time, dt = 512, 0.1
    times = np.arange(n_time) * dt
    omega = 2.0 * np.pi * 3.0 / (n_time * dt)  # exactly bin 3
    values = np.sin(omega * times)[None, :, None].repeat(4, axis=0)
    product = _execute(
        _scalar_analyzer(convention="symmetric"), _observable_product(values, dt=dt)
    )

    axis = np.asarray(product.handle("axis").materialize())
    psd = np.asarray(product.handle("psd").materialize())[:, 0]
    peak = int(np.argmax(psd))
    assert peak == 3
    assert axis[peak] == pytest.approx(omega)
    step = float(axis[1] - axis[0])
    assert float(psd.sum()) * step == pytest.approx(0.5, rel=1e-12)


def test_scalar_psd_of_white_noise_is_flat_at_twice_variance_dt():
    """Interior one-sided bins of white noise fold to ``2 * s^2 * dt``."""
    rng = np.random.default_rng(20261213)
    n_traj, n_time, dt, sigma = 256, 1024, 0.25, 0.7
    values = (sigma * rng.standard_normal((n_traj, n_time)))[:, :, None]
    product = _execute(_scalar_analyzer(), _observable_product(values, dt=dt))

    psd = np.asarray(product.handle("psd").materialize())[:, 0]
    sem = np.asarray(product.handle("psd_sem").materialize())[:, 0]
    interior = psd[1:-1]
    expected = 2.0 * sigma**2 * dt
    assert float(interior.mean()) == pytest.approx(expected, rel=0.05)
    # The ensemble interior average agrees within trajectory/bin scatter.
    scatter = float(psd[1:-1].std()) / np.sqrt(interior.size)
    assert abs(float(interior.mean()) - expected) < 5.0 * scatter
    assert np.all(np.isfinite(sem[1:-1]))
    assert np.all(sem[1:-1] > 0.0)


def test_scalar_psd_constant_detrend_empties_dc():
    n_traj, n_time, dt = 2, 256, 0.5
    values = np.full((n_traj, n_time, 1), 1.7)
    product = _execute(
        _scalar_analyzer(detrend="constant"), _observable_product(values, dt=dt)
    )
    psd = np.asarray(product.handle("psd").materialize())[:, 0]
    np.testing.assert_allclose(psd, 0.0, atol=1e-24)
    assert product.attributes["detrend"] == "constant"


def test_scalar_psd_selects_channels_by_name():
    rng = np.random.default_rng(20261213)
    values = rng.standard_normal((16, 256, 3))
    product = _observable_product(values, dt=0.2, names=["n0", "n1", "n2"])

    selected = _execute(_scalar_analyzer(("n1", "n2")), product)
    channels = np.asarray(selected.handle("channels").materialize())
    np.testing.assert_array_equal(channels, [0, 1])
    assert selected.attributes["channel_names"] == ["n1", "n2"]
    psd = np.asarray(selected.handle("psd").materialize())
    for position, column in enumerate((1, 2)):
        single = _execute(
            _scalar_analyzer((f"n{column}",)),
            _observable_product(values[:, :, [column]], dt=0.2, names=[f"n{column}"]),
        )
        np.testing.assert_allclose(
            psd[:, position],
            np.asarray(single.handle("psd").materialize())[:, 0],
            rtol=1e-12,
        )


def test_scalar_psd_product_schema_is_frozen():
    """The sampled-scalar PSD product keeps its exact concrete layout."""
    rng = np.random.default_rng(20261213)
    values = rng.standard_normal((8, 128, 1))
    product = _execute(_scalar_analyzer(), _observable_product(values, dt=0.1))

    assert {variable.name for variable in product.variables} == {
        "axis",
        "psd",
        "psd_std",
        "psd_sem",
        "channels",
        "sample_dt",
        "nyquist",
        "uncertainty.n_independent",
        "uncertainty.ddof",
        "uncertainty.available",
    }
    psd = product.schema.variable("psd")
    assert psd is not None
    assert psd.quantity == "power_spectral_density"
    assert psd.units == "occupation^2/inverse_time"
    assert psd.dims == ("frequency", "channel")
    # Estimate, sample STD and SEM share the physical dimension of the PSD.
    for uncertainty_variable in ("psd_std", "psd_sem"):
        leaf = product.schema.variable(uncertainty_variable)
        assert leaf is not None
        assert leaf.units == "occupation^2/inverse_time"
    assert product.axis("frequency").units == "inverse_time"
    assert product.attributes["orientation"] == "unsigned"
    assert product.attributes["sidedness"] == "one_sided"
    assert product.attributes["frequency_units"] == "inverse_time"
    assert product.attributes["normalization"] == "density"
    assert product.attributes["convention"] == "pragmatic"
    assert product.attributes["measurement_model"] == "sampled_scalar"
    assert product.attributes["channel_names"] == ["n0"]
    assert product.attributes["detrend"] == "none"
    assert product.attributes["estimator"] == "periodogram"


def test_scalar_psd_dimensionless_units():
    values = np.zeros((2, 64, 1))
    for dimensionless in ("", "1"):
        product = _execute(
            _scalar_analyzer(), _observable_product(values, dt=0.1, units=dimensionless)
        )
        for name in ("psd", "psd_std", "psd_sem"):
            variable = product.schema.variable(name)
            assert variable is not None
            assert variable.units == "1/inverse_time"


def test_scalar_psd_valid_length_censors_padding():
    """Padding beyond valid_length is truncated before the estimator."""
    rng = np.random.default_rng(20261213)
    n_traj, n_time, valid, dt = 8, 256, 200, 0.1
    clean = rng.standard_normal((n_traj, valid, 1))
    padded = np.concatenate(
        [clean, np.full((n_traj, n_time - valid, 1), np.nan)], axis=1
    )
    reference = _execute(
        _scalar_analyzer(), _observable_product(clean, dt=dt)
    )
    censored = _execute(
        _scalar_analyzer(),
        _observable_product(
            padded, dt=dt, valid_length=np.full(n_traj, valid)
        ),
    )
    np.testing.assert_allclose(
        np.asarray(censored.handle("psd").materialize()),
        np.asarray(reference.handle("psd").materialize()),
        rtol=1e-12,
    )


def test_scalar_psd_accumulator_matches_single_pass():
    rng = np.random.default_rng(20261213)
    values = rng.standard_normal((16, 128, 1))
    single = _execute(_scalar_analyzer(), _observable_product(values, dt=0.1))

    analyzer = _scalar_analyzer()
    accumulator = PsdResultAccumulator(analyzer)
    for half in (values[:8], values[8:]):
        batch = _execute(analyzer, _observable_product(half, dt=0.1))
        accumulator.consume_product(batch)
    merged_products = accumulator.finalize_product("psd")
    assert merged_products is not None
    merged = merged_products["psd"]

    for name in ("psd", "psd_std", "psd_sem"):
        np.testing.assert_allclose(
            np.asarray(merged.handle(name).materialize()),
            np.asarray(single.handle(name).materialize()),
            rtol=1e-12,
        )
    assert merged.attributes["channel_names"] == ["n0"]
    assert merged.attributes["convention"] == "pragmatic"
    for name in ("psd", "psd_std", "psd_sem"):
        merged_variable = merged.schema.variable(name)
        assert merged_variable is not None
        assert merged_variable.units == "occupation^2/inverse_time"


def test_scalar_psd_requires_signal_input():
    analyzer = _scalar_analyzer()
    product = _observable_product(np.zeros((2, 16, 1)), dt=0.1)
    with pytest.raises(TypeError, match="'signal'"):
        analyzer.execute({"trajectories": product}, backend=BACKEND, label="psd")


def test_scalar_requirement_matches_only_sampled_observable_products():
    requirement = _scalar_analyzer().input_requirements()[0]
    assert requirement.name == "signal"
    assert requirement.quantity == SDEQuantity.SAMPLED_OBSERVABLE.value
    assert requirement.value_domain == "real"
    assert set(requirement.required_fields) == {"value", "valid_length"}

    observable = _observable_product(np.zeros((2, 16, 1)), dt=0.1)
    assert requirement_matches(requirement, observable.schema)

    field_schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=observable.axes,
        variables=[
            VariableSchema(
                name="alpha",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
            )
        ],
    )
    assert not requirement_matches(requirement, field_schema)


def test_scalar_psd_rejects_field_product_at_materialization():
    analyzer = _scalar_analyzer()
    axes = [
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
        AxisSchema(
            name="time",
            role=AxisRole.COORDINATE,
            size=16,
            coordinate="regular",
            start=0.0,
            step=0.1,
            units="time",
        ),
        AxisSchema(name="channel", role=AxisRole.COMPONENT, size=1),
    ]
    field = TimeSeriesDataset.from_arrays(
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            variables=[
                VariableSchema(
                    name="alpha",
                    dtype="<c16",
                    value_domain="complex",
                    dims=("trajectory", "time", "channel"),
                    quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                )
            ],
        ),
        {"alpha": np.ones((2, 16, 1), dtype=np.complex128)},
        owner="test",
    )
    with pytest.raises(TypeError, match="missing the value variable"):
        analyzer.execute({"signal": field}, backend=BACKEND, label="psd")


def test_unsigned_orientation_is_rejected_by_phase_consumers():
    """Scalar products are unsigned; phase-specific resolvers must refuse."""
    product = _execute(
        _scalar_analyzer(), _observable_product(np.zeros((2, 16, 1)), dt=0.1)
    )
    assert resolve_spectral_orientation(product.attributes) == "unsigned"
    with pytest.raises(ValueError, match="orientation"):
        resolve_frequency_orientation(product.attributes)


def test_scalar_measurement_rejects_orientation_key():
    """The sampled-scalar config never accepts a phase orientation knob."""
    with pytest.raises(ValidationError):
        PsdAnalyserConfig(
            measurement={
                "sampled_scalar": {"channels": ["n0"], "orientation": "unsigned"}
            }
        )


def test_measurement_union_requires_exactly_one_variant():
    with pytest.raises(ValidationError):
        PsdAnalyserConfig(
            measurement={
                "complex_field": {"channels": [0], "representation": "complex"},
                "sampled_scalar": {"channels": ["n0"]},
            }
        )
    with pytest.raises(ValidationError):
        PsdAnalyserConfig(measurement={})
