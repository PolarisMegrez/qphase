"""Sampled-scalar Allan measurement: references, schema and binding boundaries.

The sampled-scalar Allan measurement consumes the real ``value`` channel of a
sampled-observable product by channel name and applies the frozen
non-overlapping block-mean-difference estimator (unsuffixed leaves) plus its
overlapping variant (``overlap_`` leaves). Numerical fixtures mirror the
frozen math baselines of ``test_sampled_scalar_math.py``; here they are
exercised end-to-end through the analyser, the typed product assembler and
the result accumulator.
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
from qphase_sde.analyser.allan_variance import (
    AllanResultAccumulator,
    AllanVarianceAnalyser,
    AllanVarianceConfig,
)
from qphase_sde.contracts.quantities import SDEQuantity

BACKEND = NumpyBackend()

_TAUS = [4.0, 16.0, 64.0]
_DT = 1.0


def _observable_product(
    values: np.ndarray,
    *,
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
            step=_DT,
            units="time",
        ),
        AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
    ]
    return TimeSeriesDataset.from_arrays(
        ProductSchema(
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
        ),
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


def _field_product(field: np.ndarray):
    """Complex field input product from a (traj, time, channel) array."""
    field = np.asarray(field, dtype=np.complex128)
    n_traj, n_time, n_channel = field.shape
    axes = [
        AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=n_traj),
        AxisSchema(
            name="time",
            role=AxisRole.COORDINATE,
            size=n_time,
            coordinate="regular",
            start=0.0,
            step=_DT,
            units="time",
        ),
        AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
    ]
    return TimeSeriesDataset.from_arrays(
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
                ),
                VariableSchema(
                    name="valid_length",
                    dtype="<i8",
                    value_domain="real",
                    dims=("trajectory",),
                    quantity="valid_length",
                ),
            ],
            attributes={"mode_indices": [0]},
        ),
        {
            "alpha": field,
            "valid_length": np.full(n_traj, n_time, dtype=np.int64),
        },
        owner="test",
    )


def _scalar_analyzer(channels=("n0",)) -> AllanVarianceAnalyser:
    return AllanVarianceAnalyser(
        measurement={
            "sampled_scalar": {"channels": list(channels), "min_nonoverlap_pairs": 4}
        },
        taus=list(_TAUS),
        min_windows=8,
    )


def _phase_analyzer() -> AllanVarianceAnalyser:
    return AllanVarianceAnalyser(
        measurement={
            "phase_frequency": {"channels": [0], "min_independent_windows": 4}
        },
        taus=list(_TAUS),
        min_windows=8,
    )


def _execute(analyzer: AllanVarianceAnalyser, product: TimeSeriesDataset):
    return analyzer.execute({"signal": product}, backend=BACKEND, label="allan")[
        "allan"
    ]


def _leaf(product, channel: int, leaf: str) -> np.ndarray:
    return np.asarray(
        product.handle(f"channel_results.{channel}.allan.{leaf}").materialize()
    )


def test_scalar_allan_white_noise_matches_analytic():
    """White noise of variance s^2 has Allan variance s^2 / m on both paths."""
    rng = np.random.default_rng(20261014)
    n_traj, n_time, sigma = 256, 8192, 0.7
    values = (sigma * rng.standard_normal((n_traj, n_time)))[:, :, None]

    product = _execute(_scalar_analyzer(), _observable_product(values))

    tau = _leaf(product, 0, "tau")
    np.testing.assert_array_equal(tau, _TAUS)
    for column, tau_value in enumerate(_TAUS):
        m = int(tau_value / _DT)
        expected = sigma**2 / m
        for mean_leaf, sem_leaf in (
            ("sampled_scalar_variance", "sampled_scalar_variance_sem"),
            ("overlap_sampled_scalar_variance", "overlap_sampled_scalar_variance_sem"),
        ):
            mean = float(_leaf(product, 0, mean_leaf)[column])
            sem = float(_leaf(product, 0, sem_leaf)[column])
            assert mean == pytest.approx(expected, rel=0.05)
            assert abs(mean - expected) < 5.0 * sem


def test_scalar_allan_overlap_matches_phase_frequency_path():
    """Overlapping scalar Allan of phase increments equals the phase Allan.

    For ``y_k = (phi_{k+1} - phi_k) / dt`` the overlapping block-mean
    difference is exactly the phase second difference over ``tau``, so the
    analyser-level overlap leaves reproduce the phase-frequency primary
    estimator of the same underlying process.
    """
    rng = np.random.default_rng(20261016)
    n_traj, n_time = 4, 2000
    increments = 0.3 * rng.standard_normal((n_traj, n_time))
    phase = np.concatenate(
        [np.zeros((n_traj, 1)), np.cumsum(increments * _DT, axis=1)], axis=1
    )
    field = np.exp(1j * phase)[:, :, None]

    scalar_product = _execute(
        _scalar_analyzer(), _observable_product(increments[:, :, None])
    )
    phase_product = _execute(_phase_analyzer(), _field_product(field))

    np.testing.assert_array_equal(
        _leaf(scalar_product, 0, "tau"),
        np.asarray(phase_product.handle("mode_results.0.allan.tau").materialize()),
    )
    np.testing.assert_allclose(
        _leaf(scalar_product, 0, "overlap_sampled_scalar_variance"),
        np.asarray(
            phase_product.handle(
                "mode_results.0.allan.angular_frequency_variance"
            ).materialize()
        ),
        rtol=1e-10,
    )


def test_scalar_allan_product_schema_is_frozen():
    """The sampled-scalar Allan product keeps its exact concrete layout."""
    rng = np.random.default_rng(20261213)
    values = rng.standard_normal((8, 512, 1))
    product = _execute(_scalar_analyzer(), _observable_product(values))

    leaves = {
        "tau",
        "sampled_scalar_variance",
        "sampled_scalar_variance_sem",
        "per_trajectory",
        "valid_window_pairs",
        "trajectory_sample_count",
        "nominal_windows_per_trajectory",
        "total_valid_window_pairs",
        "overlap_sampled_scalar_variance",
        "overlap_sampled_scalar_variance_sem",
        "overlap_per_trajectory",
        "overlap_valid_window_pairs",
        "overlap_trajectory_sample_count",
        "overlap_total_valid_window_pairs",
        "observation_duration",
        "min_nonoverlap_pairs",
    }
    assert {variable.name for variable in product.variables} == {
        "channels",
        "t0",
        "dt",
        "n_traj",
        "n_samples",
        *[f"channel_results.0.allan.{leaf}" for leaf in leaves],
    }
    variance = product.schema.variable(
        "channel_results.0.allan.sampled_scalar_variance"
    )
    assert variance is not None
    assert variance.quantity == SDEQuantity.ALLAN_VARIANCE.value
    assert variance.units == "occupation^2"
    assert variance.dims == ("tau",)
    # Target, SEM and per-trajectory estimates share the variance units.
    for leaf_name in (
        "sampled_scalar_variance_sem",
        "per_trajectory",
        "overlap_sampled_scalar_variance",
        "overlap_sampled_scalar_variance_sem",
        "overlap_per_trajectory",
    ):
        leaf = product.schema.variable(f"channel_results.0.allan.{leaf_name}")
        assert leaf is not None
        assert leaf.units == "occupation^2"
    per_trajectory = product.schema.variable(
        "channel_results.0.allan.per_trajectory"
    )
    assert per_trajectory is not None
    assert per_trajectory.dims == ("trajectory", "tau")
    overlap = product.schema.variable(
        "channel_results.0.allan.overlap_sampled_scalar_variance"
    )
    assert overlap is not None
    assert overlap.units == "occupation^2"
    assert product.axis("tau").units == "time"
    assert product.attributes["measurement_model"] == "sampled_scalar"
    assert product.attributes["estimator"] == "nonoverlapping_window_mean_difference"
    assert product.attributes["channel_names"] == ["n0"]
    assert "orientation" not in product.attributes


def test_scalar_allan_dimensionless_units():
    values = np.zeros((4, 512, 1))
    for dimensionless in ("", "1"):
        product = _execute(
            _scalar_analyzer(), _observable_product(values, units=dimensionless)
        )
        for leaf_name in (
            "sampled_scalar_variance",
            "sampled_scalar_variance_sem",
            "per_trajectory",
            "overlap_sampled_scalar_variance",
        ):
            variable = product.schema.variable(f"channel_results.0.allan.{leaf_name}")
            assert variable is not None
            assert variable.units == "1"


def test_scalar_allan_constant_series_vanishes():
    values = np.full((4, 512, 1), 2.5)
    product = _execute(_scalar_analyzer(), _observable_product(values))
    for leaf in ("sampled_scalar_variance", "overlap_sampled_scalar_variance"):
        np.testing.assert_array_equal(_leaf(product, 0, leaf), np.zeros(3))


def test_scalar_allan_valid_length_censors_padding():
    rng = np.random.default_rng(20261213)
    n_traj, n_time, valid = 8, 512, 400
    clean = rng.standard_normal((n_traj, valid, 1))
    padded = np.concatenate(
        [clean, np.full((n_traj, n_time - valid, 1), np.nan)], axis=1
    )
    reference = _execute(_scalar_analyzer(), _observable_product(clean))
    censored = _execute(
        _scalar_analyzer(),
        _observable_product(padded, valid_length=np.full(n_traj, valid)),
    )
    for leaf in (
        "sampled_scalar_variance",
        "sampled_scalar_variance_sem",
        "overlap_sampled_scalar_variance",
    ):
        np.testing.assert_allclose(
            _leaf(censored, 0, leaf),
            _leaf(reference, 0, leaf),
            rtol=1e-12,
        )


def test_scalar_allan_accumulator_matches_single_pass():
    rng = np.random.default_rng(20261213)
    values = rng.standard_normal((16, 512, 1))
    single = _execute(_scalar_analyzer(), _observable_product(values))

    accumulator = AllanResultAccumulator("sampled_scalar")
    analyzer = _scalar_analyzer()
    for half in (values[:8], values[8:]):
        batch = _execute(analyzer, _observable_product(half))
        accumulator.consume_product(batch)
    merged_products = accumulator.finalize_product("allan")
    assert merged_products is not None
    merged = merged_products["allan"]

    for leaf in (
        "sampled_scalar_variance",
        "sampled_scalar_variance_sem",
        "overlap_sampled_scalar_variance",
        "overlap_sampled_scalar_variance_sem",
    ):
        np.testing.assert_allclose(
            np.asarray(
                merged.handle(f"channel_results.0.allan.{leaf}").materialize()
            ),
            _leaf(single, 0, leaf),
            rtol=1e-12,
        )
    assert float(merged.handle("n_traj").materialize()) == 16
    assert merged.attributes["channel_names"] == ["n0"]
    merged_variance = merged.schema.variable(
        "channel_results.0.allan.sampled_scalar_variance"
    )
    assert merged_variance is not None
    assert merged_variance.units == "occupation^2"
    merged_sem = merged.schema.variable(
        "channel_results.0.allan.sampled_scalar_variance_sem"
    )
    assert merged_sem is not None
    assert merged_sem.units == "occupation^2"


def test_scalar_allan_requires_signal_input():
    analyzer = _scalar_analyzer()
    product = _observable_product(np.zeros((4, 64, 1)))
    with pytest.raises(TypeError, match="'signal'"):
        analyzer.execute({"trajectories": product}, backend=BACKEND, label="allan")


def test_scalar_requirement_matches_only_sampled_observable_products():
    requirement = _scalar_analyzer().input_requirements()[0]
    assert requirement.name == "signal"
    assert requirement.quantity == SDEQuantity.SAMPLED_OBSERVABLE.value
    assert requirement.value_domain == "real"
    assert set(requirement.required_fields) == {"value", "valid_length"}
    assert requirement_matches(
        requirement, _observable_product(np.zeros((4, 64, 1))).schema
    )
    assert not requirement_matches(
        requirement, _field_product(np.ones((4, 64, 1), dtype=np.complex128)).schema
    )


def test_scalar_allan_rejects_field_product_at_materialization():
    analyzer = _scalar_analyzer()
    field = _field_product(np.ones((4, 64, 1), dtype=np.complex128))
    with pytest.raises(TypeError, match="missing the value variable"):
        analyzer.execute({"signal": field}, backend=BACKEND, label="allan")


def test_scalar_measurement_keeps_disjoint_knob_namespace():
    """Scalar uses min_nonoverlap_pairs; the phase knob is rejected there."""
    with pytest.raises(ValidationError):
        AllanVarianceConfig(
            measurement={
                "sampled_scalar": {
                    "channels": ["n0"],
                    "min_independent_windows": 4,
                }
            }
        )
    # The phase variant keeps its independent-pair knob.
    config = AllanVarianceConfig(
        measurement={
            "phase_frequency": {"channels": [0], "min_independent_windows": 2}
        }
    )
    phase = config.measurement.phase_frequency
    assert phase is not None
    assert phase.min_independent_windows == 2


def test_scalar_measurement_rejects_blank_and_duplicate_channels():
    with pytest.raises(ValidationError):
        AllanVarianceConfig(
            measurement={"sampled_scalar": {"channels": ["n0", "n0"]}}
        )
    with pytest.raises(ValidationError):
        AllanVarianceConfig(measurement={"sampled_scalar": {"channels": ["  "]}})


def test_allan_measurement_union_requires_exactly_one_variant():
    with pytest.raises(ValidationError):
        AllanVarianceConfig(
            measurement={
                "phase_frequency": {"channels": [0]},
                "sampled_scalar": {"channels": ["n0"]},
            }
        )
    with pytest.raises(ValidationError):
        AllanVarianceConfig(measurement={})
