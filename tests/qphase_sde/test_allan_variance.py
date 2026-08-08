import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.allan_variance import (
    AllanVarianceAnalyzer,
    AllanVarianceConfig,
)
from qphase_sde.state import TrajectorySet


def _phase_diffusion_trajectory(n_traj: int = 9) -> TrajectorySet:
    rng = np.random.default_rng(20260808)
    dt = 0.1
    increments = 0.04 * rng.standard_normal((n_traj, 1024))
    phase = np.concatenate(
        [np.zeros((n_traj, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    values = np.exp(1j * phase)[:, :, None]
    return TrajectorySet(values, t0=10.0, dt=dt, meta={"mode_indices": [2]})


def _analyzer() -> AllanVarianceAnalyzer:
    return AllanVarianceAnalyzer(
        AllanVarianceConfig(
            modes=[2],
            taus=[1.0, 2.0, 4.0],
            min_independent_windows=4,
        )
    )


def test_allan_variance_reports_nonoverlap_counts_and_frequency():
    payload = (
        _analyzer().analyze(_phase_diffusion_trajectory(), NumpyBackend()).data_dict
    )
    mode = payload["mode_results"][2]
    allan = mode["allan"]

    np.testing.assert_array_equal(allan["tau"], [1.0, 2.0, 4.0])
    np.testing.assert_array_equal(
        allan["nominal_independent_windows_per_trajectory"], [51, 25, 12]
    )
    np.testing.assert_array_equal(
        allan["total_independent_window_count"], [459, 225, 108]
    )
    assert allan["nonoverlap_per_trajectory"].shape == (9, 3)
    assert np.all(allan["angular_frequency_variance"] > 0.0)
    assert mode["phase_increment"]["mean_angular_frequency_per_trajectory"].shape == (
        9,
    )


def test_allan_accumulator_matches_full_trajectory_analysis():
    trajectory = _phase_diffusion_trajectory()
    analyzer = _analyzer()
    full = analyzer.analyze(trajectory, NumpyBackend()).data_dict
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 2), (2, 6), (6, 9)):
        partial = analyzer.analyze(
            TrajectorySet(
                trajectory.data[start:stop],
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            ),
            NumpyBackend(),
        ).data_dict
        accumulator.update(partial)
    merged = accumulator.finalize()

    assert merged["n_traj"] == 9
    full_mode = full["mode_results"][2]
    merged_mode = merged["mode_results"][2]
    for key in (
        "angular_frequency_variance",
        "angular_frequency_variance_sem",
        "per_trajectory",
        "nonoverlap_angular_frequency_variance",
        "nonoverlap_angular_frequency_variance_sem",
        "nonoverlap_per_trajectory",
        "total_independent_window_count",
    ):
        np.testing.assert_allclose(
            merged_mode["allan"][key], full_mode["allan"][key], rtol=1e-13
        )
    for key in (
        "mean_angular_frequency_per_trajectory",
        "max_abs_phase_step_per_trajectory",
        "near_nyquist_fraction_per_trajectory",
    ):
        np.testing.assert_allclose(
            merged_mode["phase_increment"][key],
            full_mode["phase_increment"][key],
            rtol=1e-13,
        )


def test_allan_analyzer_advertises_trajectory_batching():
    capabilities = _analyzer().capabilities()
    assert capabilities.execution_location == "host"
    assert capabilities.requires_full_trajectory is True
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False
