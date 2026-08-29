import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.trajectory_diagnostic import TrajectoryDiagnosticContext
from qphase_sde.analyser.trajectory_diagnostics import (
    TrajectoryDiagnostics,
    TrajectoryDiagnosticsConfig,
)
from qphase_sde.state import TrajectorySet


def _rotating_trajectory(omega=0.7, *, n_traj=3, n_time=512, dt=0.1):
    time = np.arange(n_time) * dt
    data = np.exp(-1j * omega * time)[None, :, None]
    data = np.repeat(data, n_traj, axis=0)
    return TrajectorySet(data=data, t0=20.0, dt=dt, meta={"mode_indices": [2]})


def test_trajectory_diagnostics_recovers_coherence_frequency_and_zero_allan():
    trajectory = _rotating_trajectory()
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[2],
            block_durations=[5.0],
            coherence_max_lag=2.0,
            allan_taus=[0.1, 0.5, 1.0],
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    mode = result["mode_results"][2]

    assert result["t0"] == pytest.approx(20.0)
    np.testing.assert_allclose(
        mode["block_statistics"][0]["mean_angular_frequency"],
        0.7,
        atol=1e-12,
    )
    expected_g1 = np.exp(-1j * 0.7 * mode["coherence"]["lag"])
    np.testing.assert_allclose(
        mode["coherence"]["g1_normalized"], expected_g1, atol=1e-12
    )
    np.testing.assert_allclose(
        mode["allan"]["angular_frequency_variance"], 0.0, atol=1e-24
    )
    assert result["orientation"] == "phase_decreasing"
    assert mode["phase_increment"]["orientation"] == "phase_decreasing"


def test_trajectory_diagnostics_validates_time_scales_against_saved_dt():
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(modes=[2], block_durations=[0.15])
    )

    with pytest.raises(ValueError, match="block duration=.*multiple of dt"):
        analyser.analyze(_rotating_trajectory(), NumpyBackend())


def test_trajectory_diagnostics_rejects_real_input():
    data = TrajectorySet(data=np.ones((2, 16, 1)), dt=0.1)
    analyser = TrajectoryDiagnostics(TrajectoryDiagnosticsConfig(modes=[0]))

    with pytest.raises(ValueError, match="complex mode amplitudes"):
        analyser.analyze(data, NumpyBackend())


def test_trajectory_diagnostic_context_builds_shared_coordinates_once():
    calls = 0

    def build(values):
        nonlocal calls
        calls += 1
        return np.abs(values) ** 2

    values = np.ones((2, 4, 1), dtype=np.complex128)
    context = TrajectoryDiagnosticContext(
        values=values,
        dt=0.5,
        t0=0.0,
        modes=(0,),
        mode_columns=(0,),
        coordinate_builder=build,
    )

    assert context.canonical_coordinates() is context.canonical_coordinates()
    assert calls == 1


def _ou_phase_diffusion_trajectory(
    d_phi=0.5, *, n_traj=256, n_time=1024, dt=0.05, seed=11
):
    """Unit-amplitude trajectories with Brownian phase: |g1| = exp(-d_phi*tau)."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(0.0, np.sqrt(2.0 * d_phi * dt), size=(n_traj, n_time - 1))
    phase = np.concatenate(
        (np.zeros((n_traj, 1)), np.cumsum(increments, axis=1)), axis=1
    )
    data = np.exp(1j * phase)[:, :, None]
    return TrajectorySet(data=data, dt=dt)


def _static_disorder_trajectory(sigma=0.8, *, n_traj=512, n_time=256, dt=0.05, seed=23):
    """Constant per-trajectory Gaussian frequency offsets: ensemble g1 ~ Gaussian."""
    rng = np.random.default_rng(seed)
    offsets = rng.normal(0.0, sigma, size=n_traj)
    time = np.arange(n_time) * dt
    data = np.exp(1j * offsets[:, None] * time[None, :])[:, :, None]
    return TrajectorySet(data=data, dt=dt)


def test_g1_model_fit_recovers_phase_diffusion_rate():
    d_phi = 0.5
    trajectory = _ou_phase_diffusion_trajectory(d_phi)
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(modes=[0], allan=False, coherence_max_lag=20.0)
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    fits = result["mode_results"][0]["coherence"]["model_fits"]

    assert fits["preferred_model"] == "exponential"
    exponential = fits["models"]["exponential"]
    assert exponential["status"] == "ok"
    assert exponential["params"]["gamma"] == pytest.approx(d_phi, rel=0.15)
    assert exponential["n_train"] + exponential["n_holdout"] == fits["n_finite"]


def test_g1_model_fit_prefers_gaussian_for_static_disorder():
    trajectory = _static_disorder_trajectory()
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(modes=[0], allan=False, coherence_max_lag=10.0)
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    fits = result["mode_results"][0]["coherence"]["model_fits"]
    models = fits["models"]

    assert (
        models["gaussian"]["holdout_rms_residual"]
        < models["exponential"]["holdout_rms_residual"]
    )
    assert fits["preferred_model"] == "gaussian"


def test_g1_model_fit_handles_degenerate_signals():
    zeros = TrajectorySet(data=np.zeros((4, 256, 1), dtype=complex), dt=0.1)
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(modes=[0], allan=False)
    )

    result = analyser.analyze(zeros, NumpyBackend()).data_dict
    fits = result["mode_results"][0]["coherence"]["model_fits"]

    assert "preferred_model" not in fits
    for model in fits["models"].values():
        assert model["status"] in ("fit_failed", "insufficient_data")
        assert "params" not in model


def test_block_spectrum_recovers_tone_features():
    dt = 0.1
    n_time = 512
    amplitude = 1.3
    resolution = 2.0 * np.pi / 6.4
    omega0 = 10.2 * resolution  # deliberately off-bin by 0.2 bins
    time = np.arange(n_time) * dt
    data = (amplitude * np.exp(-1j * omega0 * time))[None, :, None]
    data = np.repeat(data, 2, axis=0)
    trajectory = TrajectorySet(data=data, dt=dt)
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[0],
            block_durations=[6.4],
            block_spectrum=True,
            coherence=False,
            allan=False,
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    entry = result["mode_results"][0]["block_spectrum"]["entries"][0]

    assert entry["status"] == "ok"
    assert entry["resolution_angular"] == pytest.approx(resolution)
    assert entry["frequency_unit"] == "angular"
    assert entry["sidedness"] == "two-sided"
    assert entry["window"] == "rectangular"
    peaks = entry["peak_angular_frequency"]
    assert peaks.shape == (2, 8)
    assert np.all(np.abs(peaks - omega0) < resolution)
    hwhm = entry["local_hwhm_angular"]
    assert np.all(hwhm > 0.0)
    assert np.all(hwhm <= 1.5 * resolution)
    assert np.all(entry["resolution_limited"])
    np.testing.assert_allclose(entry["integrated_power"], amplitude**2, rtol=1e-12)
    wing = entry["wing_fraction"]
    assert np.all(wing >= 0.0)
    assert np.all(wing <= 1.0)
    assert np.all(wing < 0.5)


def test_block_spectrum_reports_configured_and_insufficient_states():
    trajectory = _rotating_trajectory(n_traj=2, n_time=64)
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[2], block_spectrum=True, coherence=False, allan=False
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    mode = result["mode_results"][2]
    assert mode["block_spectrum"] == {"status": "no_blocks_configured"}

    # Five-sample blocks and a single whole-trajectory block are both unusable.
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[2],
            block_durations=[0.5, 6.4],
            block_spectrum=True,
            coherence=False,
            allan=False,
        )
    )
    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    entries = result["mode_results"][2]["block_spectrum"]["entries"]
    assert [entry["status"] for entry in entries] == [
        "insufficient_data",
        "insufficient_data",
    ]


def test_outputs_carry_explicit_units_and_definitions():
    trajectory = _rotating_trajectory()
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[2],
            block_durations=[5.0],
            coherence_max_lag=2.0,
            allan_taus=[0.1, 0.5, 1.0],
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    mode = result["mode_results"][2]

    coherence = mode["coherence"]
    assert coherence["quantity"] == "first_order_coherence"
    assert coherence["normalization"] == "g1(0)"
    assert coherence["lag_unit"] == "seconds"
    assert set(coherence["model_fits"]["models"]) == {
        "exponential",
        "gaussian",
        "kubo",
    }
    allan = mode["allan"]
    assert allan["quantity"] == "allan_variance"
    assert allan["variable"] == "angular_frequency"
    assert allan["tau_unit"] == "seconds"
    assert mode["block_statistics"][0]["frequency_unit"] == "angular"


def _two_mode_trajectory(*, n_traj=2, n_time=512, dt=0.1, amp0=1.3, amp1=0.7):
    """Two-mode rotating amplitudes with analytically known R coordinates.

    R_01(t) = amp0 * amp1 * exp(1j * t), so the canonical layout
    [diag_0, diag_1, Re(R_01), Im(R_01)] equals
    [amp0**2, amp1**2, amp0*amp1*cos(t), amp0*amp1*sin(t)].
    """
    time = np.arange(n_time) * dt
    alpha0 = amp0 * np.exp(1j * 0.7 * time)
    alpha1 = amp1 * np.exp(-1j * 0.3 * time)
    data = np.stack([alpha0, alpha1], axis=-1)[None, :, :]
    data = np.repeat(data, n_traj, axis=0)
    return TrajectorySet(data=data, dt=dt)


def test_matrix_projection_recovers_mode_population_and_layout():
    trajectory = _two_mode_trajectory()
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[0],
            coherence=False,
            allan=False,
            matrix_projection=True,
            matrix_projection_left_vector=[1.0, 0.0, 0.0, 0.0],
            matrix_projection_keep_coordinates=True,
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    projection = result["matrix_projection"]

    assert projection["status"] == "ok"
    assert projection["quantity"] == "matrix_projection"
    assert projection["convention"] == "R[i,j] = alpha_i * conj(alpha_j)"
    assert projection["n_modes"] == 2
    assert projection["n_coordinates"] == 4
    assert projection["reference_norm"] == pytest.approx(0.0)
    assert projection["left_vector_norm"] == pytest.approx(1.0)

    series = projection["projection_per_trajectory"]
    assert series.shape == (2, 512)
    np.testing.assert_allclose(series, 1.3**2, rtol=1e-12)
    np.testing.assert_allclose(
        projection["projection_mean_per_trajectory"], 1.3**2, rtol=1e-12
    )
    np.testing.assert_allclose(
        projection["projection_std_per_trajectory"], 0.0, atol=1e-12
    )

    # Canonical layout: [diag_0, diag_1, Re(R_01), Im(R_01)].
    time = np.arange(512) * 0.1
    expected = np.stack(
        [
            np.full_like(time, 1.3**2),
            np.full_like(time, 0.7**2),
            1.3 * 0.7 * np.cos(time),
            1.3 * 0.7 * np.sin(time),
        ],
        axis=-1,
    )
    coordinates = projection["coordinates_per_trajectory"]
    assert coordinates.shape == (2, 512, 4)
    np.testing.assert_allclose(coordinates[0], expected, atol=1e-12)
    np.testing.assert_allclose(
        projection["mean_coordinates_per_trajectory"][0],
        np.mean(expected, axis=0),
        atol=1e-12,
    )


def test_matrix_projection_subtracts_reference():
    trajectory = _two_mode_trajectory()
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[0],
            coherence=False,
            allan=False,
            matrix_projection=True,
            matrix_projection_reference=[0.5, 0.0, 0.0, 0.0],
            matrix_projection_left_vector=[1.0, 0.0, 0.0, 0.0],
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    projection = result["matrix_projection"]

    assert projection["reference_norm"] == pytest.approx(0.5)
    np.testing.assert_allclose(
        projection["projection_per_trajectory"], 1.3**2 - 0.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        projection["projection_min_per_trajectory"], 1.3**2 - 0.5, rtol=1e-12
    )
    np.testing.assert_allclose(
        projection["projection_max_per_trajectory"], 1.3**2 - 0.5, rtol=1e-12
    )


def test_matrix_projection_config_validation():
    with pytest.raises(ValidationError, match="requires matrix_projection"):
        TrajectoryDiagnosticsConfig(
            modes=[0], matrix_projection_left_vector=[1.0, 0.0, 0.0, 0.0]
        )
    with pytest.raises(ValidationError, match="requires matrix_projection"):
        TrajectoryDiagnosticsConfig(
            modes=[0], matrix_projection_reference=[0.0, 0.0, 0.0, 0.0]
        )
    with pytest.raises(ValidationError, match="length n_modes"):
        TrajectoryDiagnosticsConfig(
            modes=[0],
            matrix_projection=True,
            matrix_projection_left_vector=[1.0, 0.0, 0.0],
        )
    with pytest.raises(ValidationError, match="matching lengths"):
        TrajectoryDiagnosticsConfig(
            modes=[0],
            matrix_projection=True,
            matrix_projection_reference=[0.0] * 4,
            matrix_projection_left_vector=[1.0] * 9,
        )

    # Length must match n_modes**2 of the analysed data (checked at analyze).
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[0],
            coherence=False,
            allan=False,
            matrix_projection=True,
            matrix_projection_left_vector=[1.0] * 9,
        )
    )
    with pytest.raises(ValueError, match=r"length n_modes\*\*2 = 4"):
        analyser.analyze(_two_mode_trajectory(), NumpyBackend())


def test_matrix_projection_block_spectrum_reuses_periodogram_features():
    # left_vector selects Re(R_01) = 0.91 * cos(1.0 * t), a zero-mean tone.
    trajectory = _two_mode_trajectory(n_traj=2, n_time=512, dt=0.1)
    analyser = TrajectoryDiagnostics(
        TrajectoryDiagnosticsConfig(
            modes=[0],
            block_durations=[6.4],
            block_spectrum=True,
            coherence=False,
            allan=False,
            matrix_projection=True,
            matrix_projection_left_vector=[0.0, 0.0, 1.0, 0.0],
        )
    )

    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    entry = result["matrix_projection"]["block_spectrum"]["entries"][0]

    assert entry["status"] == "ok"
    peaks = entry["peak_angular_frequency"]
    assert peaks.shape == (2, 8)
    assert np.all(np.abs(np.abs(peaks) - 1.0) < entry["resolution_angular"])
    np.testing.assert_allclose(
        entry["integrated_power"], 0.5 * (1.3 * 0.7) ** 2, rtol=0.05
    )


def _stationary_noise_trajectory(*, n_traj=3, n_time=1024, dt=0.1, seed=7):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(n_traj, n_time, 2)) + 1j * rng.normal(
        size=(n_traj, n_time, 2)
    )
    return TrajectorySet(data=data, dt=dt)


def _stationarity_entry(trajectory, **config_kwargs):
    config = {
        "modes": [0],
        "block_durations": [3.2],
        "coherence": False,
        "allan": False,
        "stationarity_details": True,
    }
    config.update(config_kwargs)
    analyser = TrajectoryDiagnostics(TrajectoryDiagnosticsConfig(**config))
    result = analyser.analyze(trajectory, NumpyBackend()).data_dict
    return result["stationarity"]


def test_stationarity_details_stationary_noise_scores_small():
    stationarity = _stationarity_entry(_stationary_noise_trajectory())

    assert stationarity["status"] == "ok"
    assert stationarity["feature"] == "block_mean_canonical_r"
    assert stationarity["convention"] == "R[i,j] = alpha_i * conj(alpha_j)"
    assert stationarity["n_features"] == 4
    entry = stationarity["entries"][0]
    assert entry["status"] == "ok"
    assert entry["n_blocks"] == 32

    # The trajectory axis is preserved for every per-trajectory output.
    assert entry["block_covariance"].shape == (3, 4, 4)
    quantiles = entry["radial_quantiles"]
    assert quantiles.shape == (3, 3)
    assert np.all(quantiles >= 0.0)
    assert np.all(np.diff(quantiles, axis=1) >= 0.0)
    assert set(entry["radial_distance_metric_per_trajectory"]) <= {
        "mahalanobis",
        "euclidean",
    }
    assert len(entry["radial_distance_metric_per_trajectory"]) == 3

    drift = entry["head_tail_drift"]
    assert drift["delta"].shape == (3, 4)
    assert drift["blocks_per_side"] == 8
    assert drift["significance_ratio"].shape == (3,)
    assert np.all(drift["significance_ratio"] < 10.0)

    change_point = entry["change_point"]
    assert change_point["score"].shape == (3,)
    assert change_point["block_index"].shape == (3,)
    assert np.all(change_point["score"] < 20.0)


def test_stationarity_details_preserves_single_feature_covariance_axis():
    source = _stationary_noise_trajectory(n_traj=2)
    single_mode = TrajectorySet(data=source.data[..., :1], dt=source.dt)

    stationarity = _stationarity_entry(single_mode)
    entry = stationarity["entries"][0]

    assert stationarity["n_features"] == 1
    assert entry["status"] == "ok"
    assert entry["block_covariance"].shape == (2, 1, 1)
    assert entry["head_tail_drift"]["delta"].shape == (2, 1)


def test_stationarity_details_detects_level_shift():
    base = _stationary_noise_trajectory(n_traj=2, n_time=1024)
    shifted = base.data.copy()
    shifted[:, 512:, 0] += 3.0  # level shift at the block-16 boundary
    stationary = _stationarity_entry(base)["entries"][0]
    detected = _stationarity_entry(TrajectorySet(data=shifted, dt=base.dt))["entries"][
        0
    ]

    np.testing.assert_array_equal(detected["change_point"]["block_index"], [16, 16])
    assert np.all(
        detected["change_point"]["score"] > 5.0 * stationary["change_point"]["score"]
    )
    assert np.all(detected["head_tail_drift"]["significance_ratio"] > 10.0)


def test_stationarity_details_reports_insufficient_blocks():
    trajectory = _stationary_noise_trajectory(n_traj=2, n_time=64)

    stationarity = _stationarity_entry(trajectory)
    entry = stationarity["entries"][0]
    assert entry["status"] == "insufficient_data"
    assert entry["n_blocks"] == 2
    assert entry["min_blocks"] == 4
    assert "block_covariance" not in entry

    no_blocks = _stationarity_entry(trajectory, block_durations=[])
    assert no_blocks == {"status": "no_blocks_configured"}
