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
