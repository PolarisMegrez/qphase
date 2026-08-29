import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.base import AnalyzerWorkspaceRequest
from qphase_sde.analyser.moment_statistics import (
    MomentStatisticsAnalyzer,
    MomentStatisticsConfig,
)
from qphase_sde.state import TrajectorySet


def _analyzer(**kwargs):
    return MomentStatisticsAnalyzer(MomentStatisticsConfig(**kwargs))


def test_moment_statistics_matches_direct_reduction():
    rng = np.random.default_rng(20260810)
    data = rng.normal(size=(7, 31, 3)) + 1j * rng.normal(size=(7, 31, 3))
    payload = (
        _analyzer(
            modes=[0, 2],
            time_blocks=4,
            min_block_samples=4,
            time_chunk_samples=7,
        )
        .analyze(TrajectorySet(data, t0=4.0, dt=0.25), NumpyBackend())
        .data_dict
    )

    intensity = np.abs(data[:, :, [0, 2]]) ** 2
    per_trajectory = np.mean(intensity, axis=1)
    per_product = np.einsum("rti,rtj->rij", intensity, intensity) / data.shape[1]
    expected_occupation = np.mean(per_trajectory, axis=0)
    expected_product = np.mean(per_product, axis=0)

    np.testing.assert_allclose(payload["occupation"], expected_occupation)
    np.testing.assert_allclose(payload["occupation_product"], expected_product)
    np.testing.assert_allclose(payload["fourth_moment"], np.diag(expected_product))
    np.testing.assert_allclose(
        payload["occupation_covariance"],
        expected_product - np.outer(expected_occupation, expected_occupation),
    )
    np.testing.assert_allclose(
        payload["g2"],
        expected_product / np.outer(expected_occupation, expected_occupation),
    )
    np.testing.assert_allclose(payload["per_trajectory_occupation"], per_trajectory)
    assert payload["ordering_correction"] == "none"


def test_moment_statistics_constant_amplitudes_have_unit_g2():
    vector = np.asarray([1.0 + 2.0j, 3.0 - 1.0j])
    data = np.broadcast_to(vector, (5, 64, 2)).copy()
    payload = (
        _analyzer(time_blocks=4).analyze(TrajectorySet(data), NumpyBackend()).data_dict
    )

    np.testing.assert_allclose(payload["occupation"], np.abs(vector) ** 2)
    np.testing.assert_allclose(payload["g2"], np.ones((2, 2)))
    np.testing.assert_allclose(payload["time_blocks"]["g2"], 1.0)


def test_moment_statistics_marks_zero_occupation_ratios_unavailable():
    data = np.zeros((4, 32, 2), dtype=complex)
    payload = _analyzer().analyze(TrajectorySet(data), NumpyBackend()).data_dict

    np.testing.assert_allclose(payload["occupation"], 0.0)
    assert np.all(np.isnan(payload["g2"]))
    assert np.all(np.isnan(payload["g2_sem"]))


def test_moment_statistics_resolves_physical_recorded_modes():
    data = np.zeros((3, 16, 2), dtype=complex)
    data[:, :, 0] = 2.0
    data[:, :, 1] = 3.0j
    trajectory = TrajectorySet(data, meta={"mode_indices": [2, 5]})
    payload = _analyzer(modes=[5]).analyze(trajectory, NumpyBackend()).data_dict

    assert payload["modes"] == [5]
    np.testing.assert_allclose(payload["occupation"], [9.0])
    np.testing.assert_allclose(payload["fourth_moment"], [81.0])


def test_moment_statistics_accumulator_matches_full_ensemble():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=(11, 48, 3)) + 1j * rng.normal(size=(11, 48, 3))
    analyzer = _analyzer(time_blocks=6, min_block_samples=4, time_chunk_samples=11)
    full = analyzer.analyze(
        TrajectorySet(data, t0=5.0, dt=0.25), NumpyBackend()
    ).data_dict
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 3), (3, 7), (7, 11)):
        partial = analyzer.analyze(
            TrajectorySet(data[start:stop], t0=5.0, dt=0.25), NumpyBackend()
        ).data_dict
        accumulator.update(partial)
    merged = accumulator.finalize()

    for key in (
        "occupation",
        "occupation_sem",
        "fourth_moment",
        "occupation_product",
        "occupation_product_sem",
        "occupation_covariance",
        "g2",
        "g2_sem",
        "per_trajectory_occupation",
        "per_trajectory_occupation_product",
    ):
        np.testing.assert_allclose(merged[key], full[key], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        merged["time_blocks"]["occupation_product"],
        full["time_blocks"]["occupation_product"],
        rtol=1e-13,
        atol=1e-13,
    )
    assert merged["n_traj"] == 11


def test_moment_statistics_workspace_is_bounded_by_time_chunk():
    analyzer = _analyzer(modes=[0, 1, 2], time_chunk_samples=8192)
    short = AnalyzerWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=60,
        saved_samples=100_000,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )
    long = AnalyzerWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=60,
        saved_samples=1_000_001,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )

    short_estimate = analyzer.estimate_workspace(short)
    long_estimate = analyzer.estimate_workspace(long)
    assert short_estimate.device_bytes == long_estimate.device_bytes
    assert long_estimate.device_bytes < 46 * 1024**2
    assert long_estimate.host_bytes < 16 * 1024


def test_moment_statistics_advertises_trajectory_batching():
    capabilities = _analyzer().capabilities()
    assert capabilities.execution_location == "backend"
    assert capabilities.requires_full_trajectory is True
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False
