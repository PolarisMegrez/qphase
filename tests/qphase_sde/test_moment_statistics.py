import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.moment_statistics import (
    MomentStatisticsAnalyzer,
    MomentStatisticsConfig,
)
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.result import trajectory_product
from qphase_sde.state import TrajectorySet

from .conftest import product_payload


def _analyzer(**kwargs):
    return MomentStatisticsAnalyzer(MomentStatisticsConfig(**kwargs))


def _payload(analyzer, trajectory: TrajectorySet):
    product = analyzer.execute(
        {
            "trajectories": trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=NumpyBackend(),
        label="moment_statistics",
    )["moment_statistics"]
    return product, product_payload(product)


def test_moment_statistics_matches_direct_reduction():
    rng = np.random.default_rng(20260810)
    data = rng.normal(size=(7, 31, 3)) + 1j * rng.normal(size=(7, 31, 3))
    _, payload = _payload(
        _analyzer(
            modes=[0, 2],
            time_blocks=4,
            min_block_samples=4,
            time_chunk_samples=7,
        ),
        TrajectorySet(data, t0=4.0, dt=0.25),
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
    _, payload = _payload(_analyzer(time_blocks=4), TrajectorySet(data))

    np.testing.assert_allclose(payload["occupation"], np.abs(vector) ** 2)
    np.testing.assert_allclose(payload["g2"], np.ones((2, 2)))
    np.testing.assert_allclose(payload["time_blocks"]["g2"], 1.0)


def test_moment_statistics_marks_zero_occupation_ratios_unavailable():
    data = np.zeros((4, 32, 2), dtype=complex)
    _, payload = _payload(_analyzer(), TrajectorySet(data))

    np.testing.assert_allclose(payload["occupation"], 0.0)
    assert np.all(np.isnan(payload["g2"]))
    assert np.all(np.isnan(payload["g2_sem"]))


def test_moment_statistics_resolves_physical_recorded_modes():
    data = np.zeros((3, 16, 2), dtype=complex)
    data[:, :, 0] = 2.0
    data[:, :, 1] = 3.0j
    trajectory = TrajectorySet(data, meta={"mode_indices": [2, 5]})
    _, payload = _payload(_analyzer(modes=[5]), trajectory)

    assert payload["modes"] == [5]
    np.testing.assert_allclose(payload["occupation"], [9.0])
    np.testing.assert_allclose(payload["fourth_moment"], [81.0])


def test_moment_statistics_accumulator_matches_full_ensemble():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=(11, 48, 3)) + 1j * rng.normal(size=(11, 48, 3))
    analyzer = _analyzer(time_blocks=6, min_block_samples=4, time_chunk_samples=11)
    _, full = _payload(analyzer, TrajectorySet(data, t0=5.0, dt=0.25))
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 3), (3, 7), (7, 11)):
        partial, _ = _payload(
            analyzer, TrajectorySet(data[start:stop], t0=5.0, dt=0.25)
        )
        accumulator.consume_product(partial)
    products = accumulator.finalize_product("moment_statistics")
    assert products is not None
    merged = product_payload(products["moment_statistics"])

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
    short = AnalyserWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=60,
        saved_samples=100_000,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )
    long = AnalyserWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=60,
        saved_samples=1_000_001,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )

    short_estimate = analyzer.workspace(short)
    long_estimate = analyzer.workspace(long)
    assert short_estimate.device_bytes == long_estimate.device_bytes
    assert long_estimate.device_bytes < 46 * 1024**2
    assert long_estimate.host_bytes < 16 * 1024


def test_moment_statistics_invocation_workspace_tracks_physical_batch():
    analyzer = _analyzer(modes=[0, 1, 2], time_chunk_samples=128)
    common = dict(
        trajectory_bytes=1,
        saved_samples=512,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
        total_n_traj=128,
    )

    small = analyzer.workspace(AnalyserWorkspaceRequest(n_traj=32, **common))
    large = analyzer.workspace(AnalyserWorkspaceRequest(n_traj=64, **common))

    assert small.device_bytes * 2 == large.device_bytes
    assert small.host_bytes == large.host_bytes == 0


def test_moment_statistics_retained_workspace_covers_ensemble_and_batches():
    analyzer = _analyzer(modes=[0, 1, 2], time_blocks=5)
    estimate = analyzer.workspace(
        AnalyserWorkspaceRequest(
            trajectory_bytes=1,
            n_traj=32,
            total_n_traj=100,
            saved_samples=512,
            n_record_modes=3,
            real_itemsize=8,
            backend_name="numpy",
        )
    )
    occupation_payload = 100 * 3 * 8
    product_payload = 100 * 3 * 3 * 8
    payload_and_concatenate = 2 * (occupation_payload + product_payload)
    jackknife_workspace = occupation_payload + 5 * product_payload
    fixed_payload = 5 * (3 + 3 * 3) * 8 + 2 * 5 * 8 + 3 * 8

    assert estimate.retained_host_bytes == (
        payload_and_concatenate + jackknife_workspace + 4 * fixed_payload
    )
    assert estimate.retained_device_bytes == 0


def test_moment_statistics_advertises_trajectory_batching():
    capabilities = _analyzer().execution_capabilities()
    assert capabilities.preferred_location == "backend"
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False
