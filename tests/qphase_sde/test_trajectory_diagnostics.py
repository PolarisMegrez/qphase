import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.trajectory_diagnostics import (
    TrajectoryDiagnostics,
    TrajectoryDiagnosticsConfig,
)
from qphase_sde.state import TrajectorySet


def _rotating_trajectory(omega=0.7, *, n_traj=3, n_time=512, dt=0.1):
    time = np.arange(n_time) * dt
    data = np.exp(1j * omega * time)[None, :, None]
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
    expected_g1 = np.exp(1j * 0.7 * mode["coherence"]["lag"])
    np.testing.assert_allclose(
        mode["coherence"]["g1_normalized"], expected_g1, atol=1e-12
    )
    np.testing.assert_allclose(
        mode["allan"]["angular_frequency_variance"], 0.0, atol=1e-24
    )


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


def _ou_phase_diffusion_trajectory(
    d_phi=0.5, *, n_traj=256, n_time=1024, dt=0.05, seed=11
):
    """Unit-amplitude trajectories with Brownian phase: |g1| = exp(-d_phi*tau)."""
    rng = np.random.default_rng(seed)
    increments = rng.normal(
        0.0, np.sqrt(2.0 * d_phi * dt), size=(n_traj, n_time - 1)
    )
    phase = np.concatenate(
        (np.zeros((n_traj, 1)), np.cumsum(increments, axis=1)), axis=1
    )
    data = np.exp(1j * phase)[:, :, None]
    return TrajectorySet(data=data, dt=dt)


def _static_disorder_trajectory(
    sigma=0.8, *, n_traj=512, n_time=256, dt=0.05, seed=23
):
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
    data = (amplitude * np.exp(1j * omega0 * time))[None, :, None]
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
    np.testing.assert_allclose(
        entry["integrated_power"], amplitude**2, rtol=1e-12
    )
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
