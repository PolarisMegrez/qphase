"""Tests for the ``method`` option of ``PsdAnalyser``.

The analyzer now supports three PSD estimation methods:

*   ``periodogram`` — classical averaged periodogram (default).
*   ``welch`` — averaged overlapping segment periodograms.
*   ``multitaper`` — DPSS (Slepian) multitaper estimate.

These tests use a synthetic complex sinusoid with known frequency so the
peak location can be verified regardless of scaling differences between
methods.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.registry import discovery, registry
from qphase.service import RegistryService
from qphase_sde.analyser.frequency_orientation import resolve_frequency_orientation
from qphase_sde.analyser.psd import PsdAnalyser, PsdAnalyserConfig
from qphase_sde.result import trajectory_product
from qphase_sde.spectral_estimator import create_builtin_estimator
from qphase_sde.state import TrajectorySet

BACKEND = NumpyBackend()


def _analyzer(method="periodogram", *, estimator_config=None, **kwargs):
    return PsdAnalyser(
        **kwargs,
        subplugins={
            "estimator": create_builtin_estimator(method, estimator_config or {})
        },
    )


def _field(channels=(0,), *, orientation=None, representation="complex"):
    """Complex-field measurement config dict for the given physical modes."""
    spec = {"channels": list(channels), "representation": representation}
    if orientation is not None:
        spec["orientation"] = orientation
    return {"complex_field": spec}


def _product(analyzer, data, backend=BACKEND):
    trajectory = trajectory_product(data, scan_size=None, n_traj_per_point=None)
    return analyzer.execute({"signal": trajectory}, backend=backend, label="psd")[
        "psd"
    ]


def _payload(analyzer, data, backend=BACKEND):
    product = _product(analyzer, data, backend)
    payload = {
        variable.name: np.asarray(product.handle(variable.name).materialize())
        for variable in product.variables
    }
    payload.update(product.attributes)
    payload.update(product.attributes.get("payload_meta", {}))
    payload["uncertainty"] = {
        "n_independent": int(payload["uncertainty.n_independent"]),
        "available": bool(np.isfinite(payload["psd_sem"]).any()),
        "independent_unit": "trajectory",
    }
    return payload


def _make_sine_data(
    n_traj: int = 16,
    n_time: int = 1024,
    dt: float = 0.1,
    omega: float = 1.0,
    noise_std: float = 0.05,
) -> np.ndarray:
    """Return complex-valued trajectories with a single angular-frequency peak."""
    rng = np.random.default_rng(42)
    t = np.arange(n_time) * dt
    # A complex exponential rotating at ``omega`` rad/s.
    signal = np.exp(1j * omega * t)
    noise = noise_std * (
        rng.standard_normal((n_traj, n_time))
        + 1j * rng.standard_normal((n_traj, n_time))
    )
    # Return a single-mode array of shape (n_traj, n_time, 1).
    return (signal + noise)[:, :, None]


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_method_peak_location(method):
    """Each method should recover the peak near the injected frequency."""
    dt = 0.1
    omega = 1.0
    data = TrajectorySet(
        data=_make_sine_data(n_traj=16, n_time=1024, dt=dt, omega=omega),
        t0=0.0,
        dt=dt,
    )

    analyzer = _analyzer(
        method,
        estimator_config={"nperseg": 256} if method == "welch" else {},
        measurement=_field(),
        convention="symmetric",
    )
    payload = _payload(analyzer, data)
    axis = payload["axis"]
    psd = payload["psd"]
    assert axis.ndim == 1
    assert psd.shape == (axis.size, 1)

    peak_idx = int(np.argmax(psd[:, 0]))
    peak_freq = float(axis[peak_idx])
    # The two-sided spectrum contains the peak at ``+omega`` (and a mirror at
    # ``-omega``).  Either is fine; just check the magnitude.
    assert abs(abs(peak_freq) - omega) < 0.05, (
        f"{method}: peak at {peak_freq}, expected near ±{omega}"
    )


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_method_unitary_convention(method):
    """Unitary convention should scale the frequency axis by 2*pi."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=512, dt=dt, omega=2.0),
        t0=0.0,
        dt=dt,
    )

    analyzer = _analyzer(
        method,
        estimator_config={"nperseg": 128} if method == "welch" else {},
        measurement=_field(),
        convention="unitary",
    )
    axis = _payload(analyzer, data)["axis"]
    # Unitary convention uses angular frequency, so the Nyquist limit is pi/dt.
    assert np.isclose(np.max(np.abs(axis)), np.pi / dt, rtol=1e-12)


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_default_maps_decreasing_phase_to_positive_frequency(method):
    dt = 0.1
    omega = 1.25
    time = np.arange(2048) * dt
    values = np.exp(-1j * omega * time)[None, :, None]
    data = TrajectorySet(data=np.repeat(values, 3, axis=0), t0=0.0, dt=dt)
    kwargs = {"nperseg": 1024} if method == "welch" else {}

    payload = _payload(
        _analyzer(
            method,
            estimator_config=kwargs,
            measurement=_field(),
            convention="symmetric",
        ),
        data,
    )

    peak = float(payload["axis"][np.argmax(payload["psd"][:, 0])])
    assert peak == pytest.approx(omega, abs=0.04)
    assert payload["orientation"] == "phase_decreasing"
    assert payload["positive_frequency_time_dependence"] == "exp(-i * omega * t)"


def test_psd_phase_increasing_preserves_forward_fft_axis():
    dt = 0.1
    omega = 0.75
    time = np.arange(2048) * dt
    values = np.exp(-1j * omega * time)[None, :, None]
    data = TrajectorySet(data=np.repeat(values, 2, axis=0), t0=0.0, dt=dt)

    payload = _payload(
        PsdAnalyser(
            measurement=_field(orientation="phase_increasing"),
            convention="symmetric",
        ),
        data,
    )

    peak = float(payload["axis"][np.argmax(payload["psd"][:, 0])])
    assert peak == pytest.approx(-omega, abs=0.04)
    assert payload["orientation"] == "phase_increasing"


def test_frequency_orientation_resolver_rejects_missing_metadata():
    with pytest.raises(ValueError, match="does not record"):
        resolve_frequency_orientation(None)
    with pytest.raises(ValueError, match="does not record"):
        resolve_frequency_orientation({})
    assert (
        resolve_frequency_orientation({"orientation": "phase_decreasing"})
        == "phase_decreasing"
    )


@pytest.mark.parametrize(
    ("configured", "canonical"),
    [("fft", "phase_increasing"), ("physical", "phase_decreasing")],
)
def test_psd_orientation_aliases_serialize_canonically(configured, canonical):
    config = PsdAnalyserConfig(measurement=_field(orientation=configured))

    field = config.measurement.complex_field
    assert field is not None
    assert field.orientation == canonical
    assert (
        config.model_dump()["measurement"]["complex_field"]["orientation"] == canonical
    )


def test_psd_orientation_aliases_are_discoverable_in_schema():
    schema = PsdAnalyserConfig.model_json_schema()
    orientation = schema["$defs"]["PsdComplexFieldMeasurement"]["properties"][
        "orientation"
    ]

    assert orientation["value_aliases"] == {
        "fft": "phase_increasing",
        "physical": "phase_decreasing",
    }


def test_welch_nperseg_default_reduces_variance():
    """Welch with a segment length shorter than the series produces more segments."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=1024, dt=dt, omega=1.0),
        t0=0.0,
        dt=dt,
    )

    analyzer = _analyzer(
        "welch",
        estimator_config={"nperseg": 256, "noverlap": 128},
        measurement=_field(),
        convention="symmetric",
    )
    axis = _payload(analyzer, data)["axis"]
    # 256-point FFT gives 256 frequency bins (two-sided).
    assert axis.size == 256


def test_multitaper_k_tapers_default():
    """Multitaper defaults should produce a stable spectrum."""
    dt = 0.1
    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=512, dt=dt, omega=1.5),
        t0=0.0,
        dt=dt,
    )

    analyzer = _analyzer(
        "multitaper",
        estimator_config={"nw": 2.5},
        measurement=_field(),
        convention="symmetric",
    )
    payload = _payload(analyzer, data)
    axis = payload["axis"]
    psd = payload["psd"]
    peak_idx = int(np.argmax(psd[:, 0]))
    assert abs(abs(axis[peak_idx]) - 1.5) < 0.1


def test_periodogram_reports_cross_trajectory_standard_error():
    """PSD uncertainty uses the sample variance of independent trajectories."""
    dt = 0.2
    n_time = 32
    t = np.arange(n_time) * dt
    amplitudes = np.array([1.0, 1.5, 2.0, 3.0])[:, None]
    values = amplitudes * np.exp(1j * 0.7 * t)[None, :]
    data = TrajectorySet(data=values[:, :, None], t0=0.0, dt=dt)

    payload = _payload(
        PsdAnalyser(
            measurement=_field(orientation="phase_increasing"),
            convention="pragmatic",
        ),
        data,
    )

    trajectory_psd = np.abs(np.fft.fft(values, axis=-1)) ** 2
    trajectory_psd *= dt / n_time
    trajectory_psd = np.fft.fftshift(trajectory_psd, axes=-1)
    expected_mean = np.mean(trajectory_psd, axis=0)
    expected_std = np.std(trajectory_psd, axis=0, ddof=1)

    np.testing.assert_allclose(payload["psd"][:, 0], expected_mean)
    np.testing.assert_allclose(payload["psd_std"][:, 0], expected_std)
    np.testing.assert_allclose(
        payload["psd_sem"][:, 0], expected_std / np.sqrt(amplitudes.shape[0])
    )
    assert payload["uncertainty"]["independent_unit"] == "trajectory"
    assert payload["uncertainty"]["n_independent"] == amplitudes.shape[0]


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_methods_report_sem_for_each_mode(method):
    """Every PSD estimator exposes arrays aligned with the mean PSD."""
    n_traj = 8
    data = TrajectorySet(
        data=_make_sine_data(n_traj=n_traj, n_time=256),
        t0=0.0,
        dt=0.1,
    )
    payload = _payload(
        _analyzer(
            method,
            estimator_config={"nperseg": 64} if method == "welch" else {},
            measurement=_field(),
        ),
        data,
    )

    assert payload["psd_std"].shape == payload["psd"].shape
    assert payload["psd_sem"].shape == payload["psd"].shape
    np.testing.assert_allclose(payload["psd_sem"], payload["psd_std"] / np.sqrt(n_traj))
    assert np.all(payload["psd_std"] >= 0.0)


def test_single_trajectory_marks_psd_uncertainty_unavailable():
    """One trajectory has no cross-trajectory sample variance."""
    data = TrajectorySet(
        data=_make_sine_data(n_traj=1, n_time=128),
        t0=0.0,
        dt=0.1,
    )
    payload = _payload(PsdAnalyser(measurement=_field()), data)

    assert np.all(np.isnan(payload["psd_std"]))
    assert np.all(np.isnan(payload["psd_sem"]))
    assert payload["uncertainty"]["available"] is False


@pytest.mark.parametrize("method", ["periodogram", "welch", "multitaper"])
def test_psd_accumulator_matches_full_trajectory_analysis(method):
    """Online aggregation preserves the estimator's trajectory statistics."""
    values = _make_sine_data(n_traj=11, n_time=256)
    analyzer = _analyzer(
        method,
        estimator_config={"nperseg": 64} if method == "welch" else {},
        measurement=_field(),
    )
    full_product = _product(analyzer, TrajectorySet(values, t0=0.0, dt=0.1))
    reducer = analyzer.reducer()
    assert reducer is not None
    accumulator = reducer.prepare(None, None)
    for start, stop in ((0, 3), (3, 8), (8, 11)):
        partial = _product(analyzer, TrajectorySet(values[start:stop], t0=0.0, dt=0.1))
        reducer.consume(partial, accumulator, None)
    merged_product = reducer.finalize(accumulator)["psd"]
    merged = {
        v.name: np.asarray(merged_product.handle(v.name).materialize())
        for v in merged_product.variables
    }
    full = {
        v.name: np.asarray(full_product.handle(v.name).materialize())
        for v in full_product.variables
    }

    np.testing.assert_array_equal(merged["axis"], full["axis"])
    np.testing.assert_allclose(merged["psd"], full["psd"], rtol=1e-12)
    np.testing.assert_allclose(merged["psd_std"], full["psd_std"], rtol=1e-12)
    np.testing.assert_allclose(merged["psd_sem"], full["psd_sem"], rtol=1e-12)
    assert int(merged["uncertainty.n_independent"]) == 11


def test_expected_freq_max_guards_against_aliasing():
    """The optional frequency-band hint rejects an undersampled PSD."""
    values = _make_sine_data(n_traj=2, n_time=128)
    undersampled = TrajectorySet(data=values, t0=0.0, dt=50.0)
    analyzer = PsdAnalyser(
        measurement=_field(),
        convention="symmetric",
        expected_freq_max=0.34,
    )

    with pytest.raises(ValueError, match="Nyquist limit"):
        _product(analyzer, undersampled)

    adequately_sampled = TrajectorySet(data=values, t0=0.0, dt=9.0)
    payload = _payload(analyzer, adequately_sampled)
    assert np.isclose(payload["nyquist"], np.pi / 9.0)
    assert payload["sample_dt"] == 9.0


def test_psd_method_invalid():
    """An unsupported method is rejected at configuration time."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="method"):
        PsdAnalyser(
            measurement=_field(),
            method="not_a_method",
        )


def test_psd_estimator_subplugins_are_discoverable_and_constructed():
    discovery.discover_plugins()
    service = RegistryService()

    options = service.get_subplugin_options("analyser.psd", "estimator")
    assert options.default == "periodogram"
    assert {option.name for option in options.options} >= {
        "periodogram",
        "welch",
        "multitaper",
    }

    analyzer = registry.create_plugin_instance(
        "analyser",
        {
            "name": "psd",
            "measurement": {
                "complex_field": {"channels": [0], "representation": "complex"}
            },
            "estimator": {"welch": {"nperseg": 64}},
        },
    )
    assert analyzer.estimator.name == "welch"
    assert analyzer.estimator.config.nperseg == 64

    template = service.build_template("analyser.psd", selections={"estimator": "welch"})
    assert template["estimator"]["welch"]["nperseg"] is None
    assert "method" not in template
    assert "nperseg" not in template


def test_psd_legacy_method_is_rejected_by_plugin_resolver():
    discovery.discover_plugins()
    with pytest.raises(Exception, match="method|extra"):
        registry.create_plugin_instance(
            "analyser",
            {
                "name": "psd",
                "measurement": {
                    "complex_field": {"channels": [0], "representation": "complex"}
                },
                "method": "multitaper",
                "nw": 3.5,
            },
        )


def test_estimator_capabilities_do_not_overclaim_time_streaming():
    """Built-in estimators must not declare time streaming they do not implement.

    Welch still materializes the full record before segmenting, so its
    declaration stays ``time_streaming=False`` until a true overlap-buffer
    streaming implementation is cross-checked against the non-streaming path.
    """
    from qphase_sde.spectral_estimator.builtin import (
        MultitaperEstimator,
        PeriodogramEstimator,
        WelchEstimator,
    )

    for estimator in (
        PeriodogramEstimator(),
        WelchEstimator(),
        MultitaperEstimator(),
    ):
        capabilities = estimator.capabilities()
        assert capabilities.time_streaming is False
        assert capabilities.requires_full_trajectory is True


def test_periodogram_fft_chunk_matches_full_batch_estimate():
    """Chunked trajectory FFT must reproduce the full-batch periodogram."""
    from qphase_sde.spectral_estimator.builtin import (
        PeriodogramEstimator,
        PeriodogramEstimatorConfig,
    )

    dt = 0.1
    data = _make_sine_data(n_traj=8, n_time=512, dt=dt)[:, :, 0]
    reference = PeriodogramEstimator().estimate(data, dt, "symmetric", BACKEND)
    for chunk in (1, 3, 8, 16):
        estimator = PeriodogramEstimator(
            PeriodogramEstimatorConfig(fft_chunk_trajectories=chunk)
        )
        estimate = estimator.estimate(data, dt, "symmetric", BACKEND)
        np.testing.assert_allclose(estimate.axis, reference.axis, rtol=0, atol=0)
        np.testing.assert_allclose(
            estimate.mean, reference.mean, rtol=1e-12, atol=1e-30
        )
        np.testing.assert_allclose(estimate.std, reference.std, rtol=1e-9, atol=1e-30)
        np.testing.assert_allclose(estimate.sem, reference.sem, rtol=1e-9, atol=1e-30)


def test_periodogram_fft_chunk_config_reaches_estimator():
    discovery.discover_plugins()
    analyzer = registry.create_plugin_instance(
        "analyser",
        {
            "name": "psd",
            "measurement": {
                "complex_field": {"channels": [0], "representation": "complex"}
            },
            "estimator": {"periodogram": {"fft_chunk_trajectories": 4}},
        },
    )
    assert analyzer.estimator.name == "periodogram"
    assert analyzer.estimator.config.fft_chunk_trajectories == 4


def test_raw_psd_product_fields_are_frozen():
    """The complex-field PSD product keeps its stable concrete fields.

    Wave 6B freezes the conditional product layouts: the raw-field PSD
    product keeps ``modes`` channel coordinates and a two-sided, phase-signed
    axis; only the future sampled-scalar variant uses ``channel_names``
    attributes and an unsigned one-sided axis.
    """
    data = TrajectorySet(
        data=_make_sine_data(n_traj=4, n_time=128), t0=0.0, dt=0.1
    )

    product = _product(PsdAnalyser(measurement=_field()), data)

    assert {variable.name for variable in product.variables} == {
        "axis",
        "psd",
        "psd_std",
        "psd_sem",
        "modes",
        "sample_dt",
        "nyquist",
        "uncertainty.n_independent",
        "uncertainty.ddof",
        "uncertainty.available",
    }
    assert product.variable("psd").quantity == "power_spectral_density"
    assert product.axis("frequency").units == "inverse_time"
    # The engine trajectory product declares no field units, so the estimate
    # and its uncertainty leaves carry the dimensionless density form.
    for name in ("psd", "psd_std", "psd_sem"):
        assert product.variable(name).units == "1/inverse_time"
    assert product.attributes["orientation"] == "phase_decreasing"
    assert product.attributes["sidedness"] == "two_sided"
    assert product.attributes["frequency_units"] == "inverse_time"
    assert product.attributes["convention"] == "symmetric"
    assert product.attributes["representation"] == "complex"


def test_psd_config_orientation_rejects_unsigned():
    """The phase-signed config orientation never accepts ``unsigned``."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        PsdAnalyserConfig(measurement=_field(orientation="unsigned"))


def test_psd_field_units_follow_source_variable_units():
    """A field with declared units propagates ``<units>^2/inverse_time``."""
    from qphase.data import (
        AxisRole,
        AxisSchema,
        DataKind,
        ProductSchema,
        TimeSeriesDataset,
        VariableSchema,
    )
    from qphase_sde.contracts.quantities import SDEQuantity

    values = _make_sine_data(n_traj=4, n_time=128)
    n_traj, n_time, n_channel = values.shape
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=n_traj),
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=n_time,
                coordinate="regular",
                start=0.0,
                step=0.1,
                units="time",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=n_channel),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype="<c16",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                units="sqrt(inverse_time)",
            ),
            VariableSchema(
                name="valid_length",
                dtype="<i8",
                value_domain="real",
                dims=("trajectory",),
                quantity="valid_length",
            ),
        ],
    )
    product = TimeSeriesDataset.from_arrays(
        schema,
        {"alpha": values, "valid_length": np.full(n_traj, n_time, dtype=np.int64)},
        owner="test",
    )

    analyzer = PsdAnalyser(
        measurement=_field(),
        convention="unitary",
        subplugins={"estimator": create_builtin_estimator("periodogram", {})},
    )
    result = analyzer.execute({"signal": product}, backend=BACKEND, label="psd")["psd"]
    for name in ("psd", "psd_std", "psd_sem"):
        assert result.variable(name).units == "sqrt(inverse_time)^2/inverse_time"
    assert result.attributes["convention"] == "unitary"
    assert result.attributes["representation"] == "complex"


def test_psd_field_accumulator_roundtrip_preserves_units_and_convention():
    """The reducer roundtrip keeps the declared units and spectral attributes."""
    from qphase_sde.analyser.psd import PsdResultAccumulator

    data = TrajectorySet(
        data=_make_sine_data(n_traj=8, n_time=128), t0=0.0, dt=0.1
    )
    analyzer = PsdAnalyser(measurement=_field(), convention="unitary")
    single = _product(analyzer, data)

    accumulator = PsdResultAccumulator(analyzer)
    values = _make_sine_data(n_traj=8, n_time=128)
    for batch_values in (values[:4], values[4:]):
        batch = TrajectorySet(data=batch_values, t0=0.0, dt=0.1)
        accumulator.consume_product(_product(analyzer, batch))
    merged = accumulator.finalize_product("psd")
    assert merged is not None
    merged_product = merged["psd"]

    for name in ("psd", "psd_std", "psd_sem"):
        np.testing.assert_allclose(
            np.asarray(merged_product.handle(name).materialize()),
            np.asarray(single.handle(name).materialize()),
            rtol=1e-12,
        )
        assert merged_product.variable(name).units == "1/inverse_time"
    assert merged_product.attributes["convention"] == "unitary"
    assert merged_product.attributes["representation"] == "complex"
