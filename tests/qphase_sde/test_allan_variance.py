import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.allan_variance import (
    AllanVarianceAnalyzer,
    AllanVarianceConfig,
)
from qphase_sde.analyser.base import AnalyzerWorkspaceRequest
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


def test_allan_frequency_uses_configured_phase_orientation():
    dt = 0.1
    omega = 0.6
    time = np.arange(257) * dt
    values = np.exp(-1j * omega * time)[None, :, None]
    trajectory = TrajectorySet(
        np.repeat(values, 2, axis=0), dt=dt, meta={"mode_indices": [0]}
    )

    phase_decreasing = (
        AllanVarianceAnalyzer(
            AllanVarianceConfig(modes=[0], points=4, min_independent_windows=2)
        )
        .analyze(trajectory, NumpyBackend())
        .data_dict
    )
    legacy = (
        AllanVarianceAnalyzer(
            AllanVarianceConfig(
                modes=[0],
                points=4,
                min_independent_windows=2,
                orientation="phase_increasing",
            )
        )
        .analyze(trajectory, NumpyBackend())
        .data_dict
    )

    decreasing_frequency = phase_decreasing["mode_results"][0]["phase_increment"][
        "mean_angular_frequency_per_trajectory"
    ]
    increasing_frequency = legacy["mode_results"][0]["phase_increment"][
        "mean_angular_frequency_per_trajectory"
    ]
    np.testing.assert_allclose(decreasing_frequency, omega, atol=1e-12)
    np.testing.assert_allclose(increasing_frequency, -omega, atol=1e-12)
    assert phase_decreasing["orientation"] == "phase_decreasing"


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


def test_allan_workspace_materializes_one_mode_at_a_time():
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(modes=[0, 1, 2], transfer_chunk_samples=8192)
    )
    trajectory_bytes = 60 * 1_000_001 * 3 * 16
    request = AnalyzerWorkspaceRequest(
        trajectory_bytes=trajectory_bytes,
        n_traj=60,
        saved_samples=1_000_001,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )

    estimate = analyzer.estimate_workspace(request)
    assert estimate.host_bytes == 5 * trajectory_bytes // 3
    assert estimate.device_bytes < 8 * 1024**2


def test_allan_device_transfer_is_time_chunked():
    source = _phase_diffusion_trajectory(n_traj=3)
    device = _DeviceLikeArray(source.data)
    trajectory = TrajectorySet(
        device,
        t0=source.t0,
        dt=source.dt,
        meta=dict(source.meta),
    )
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(
            modes=[2],
            taus=[1.0, 2.0],
            min_independent_windows=4,
            transfer_chunk_samples=127,
        )
    )

    payload = analyzer.analyze(trajectory, NumpyBackend()).data_dict
    reference = analyzer.analyze(source, NumpyBackend()).data_dict

    assert max(device.transfer_lengths) <= 127
    assert sum(device.transfer_lengths) == source.n_steps
    np.testing.assert_allclose(
        payload["mode_results"][2]["allan"]["angular_frequency_variance"],
        reference["mode_results"][2]["allan"]["angular_frequency_variance"],
    )


class _DeviceLikeSlice:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array

    def get(self) -> np.ndarray:
        return self.array.copy()


class _DeviceLikeArray:
    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype
        self.transfer_lengths: list[int] = []

    def __getitem__(self, key):
        selected = self.array[key]
        self.transfer_lengths.append(int(selected.shape[1]))
        return _DeviceLikeSlice(selected)
