import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.quadratic_moments import (
    QuadraticMomentAnalyser,
    QuadraticMomentConfig,
)
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.result import trajectory_product
from qphase_sde.state import TrajectorySet

from .conftest import product_payload


def _analyzer(**kwargs):
    return QuadraticMomentAnalyser(QuadraticMomentConfig(**kwargs))


def _payload(analyzer, trajectory: TrajectorySet):
    product = analyzer.execute(
        {
            "trajectories": trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=NumpyBackend(),
        label="quadratic_moments",
    )["quadratic_moments"]
    return product, product_payload(product)


def _cumulants(raw):
    mean, second, third, fourth = np.moveaxis(raw, -1, 0)
    return np.stack(
        (
            mean,
            second - mean**2,
            third - 3.0 * second * mean + 2.0 * mean**3,
            fourth
            - 4.0 * third * mean
            - 3.0 * second**2
            + 12.0 * second * mean**2
            - 6.0 * mean**4,
        ),
        axis=-1,
    )


def test_quadratic_moments_match_direct_reduction():
    rng = np.random.default_rng(20260821)
    data = rng.normal(size=(7, 41, 2)) + 1j * rng.normal(size=(7, 41, 2))
    matrix = np.asarray([[1.0, 0.25 + 0.1j], [0.25 - 0.1j, -0.5]])
    reference = np.asarray([[2.0, 0.2j], [-0.2j, 1.0]])
    center = float(np.trace(matrix @ reference).real)
    _, payload = _payload(
        _analyzer(
            observables={
                "soft": {
                    "matrix": matrix.tolist(),
                    "reference_matrix": reference.tolist(),
                },
                "total": {"matrix": np.eye(2).tolist(), "center": 1.25},
            },
            max_order=4,
            time_blocks=4,
            min_block_samples=4,
            time_chunk_samples=9,
        ),
        TrajectorySet(data, t0=3.0, dt=0.25),
    )

    soft = np.einsum("rti,ij,rtj->rt", data.conj(), matrix, data).real - center
    total = np.sum(np.abs(data) ** 2, axis=-1) - 1.25
    values = np.stack((soft, total), axis=-1)
    per_trajectory = np.stack(
        [np.mean(values**order, axis=1) for order in range(1, 5)], axis=-1
    )
    raw = np.mean(per_trajectory, axis=0)

    assert payload["observable_names"] == ["soft", "total"]
    np.testing.assert_allclose(payload["centers"], [center, 1.25])
    np.testing.assert_allclose(payload["raw_moments"], raw)
    np.testing.assert_allclose(payload["cumulants"], _cumulants(raw))
    np.testing.assert_allclose(payload["per_trajectory_raw_moments"], per_trajectory)


def test_quadratic_moments_resolve_physical_recorded_modes():
    data = np.zeros((3, 16, 2), dtype=complex)
    data[:, :, 0] = 2.0
    data[:, :, 1] = 3.0j
    trajectory = TrajectorySet(data, meta={"mode_indices": [2, 5]})
    _, payload = _payload(
        _analyzer(
            modes=[5],
            observables={"occupation": {"matrix": [[1.0]]}},
        ),
        trajectory,
    )

    assert payload["modes"] == [5]
    np.testing.assert_allclose(payload["raw_moments"][0], [9.0, 81.0, 729.0, 6561.0])
    np.testing.assert_allclose(payload["cumulants"][0, 1:], 0.0, atol=1e-12)


def test_quadratic_moments_reject_nonhermitian_matrix():
    data = np.ones((2, 8, 2), dtype=complex)
    analyzer = _analyzer(observables={"bad": {"matrix": [[1.0, 1.0], [0.0, 1.0]]}})
    with pytest.raises(ValueError, match="not Hermitian"):
        _payload(analyzer, TrajectorySet(data))


def test_quadratic_moments_accumulator_matches_full_ensemble():
    rng = np.random.default_rng(12345)
    data = rng.normal(size=(11, 48, 3)) + 1j * rng.normal(size=(11, 48, 3))
    analyzer = _analyzer(
        observables={
            "difference": {
                "matrix": [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 0.0]],
                "center": 0.5,
            }
        },
        time_blocks=6,
        min_block_samples=4,
        time_chunk_samples=11,
    )
    _, full = _payload(analyzer, TrajectorySet(data, t0=5.0, dt=0.25))
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 3), (3, 7), (7, 11)):
        partial, _ = _payload(
            analyzer, TrajectorySet(data[start:stop], t0=5.0, dt=0.25)
        )
        accumulator.consume_product(partial)
    products = accumulator.finalize_product("quadratic_moments")
    assert products is not None
    merged = product_payload(products["quadratic_moments"])

    for key in (
        "raw_moments",
        "raw_moment_sem",
        "central_moments",
        "cumulants",
        "cumulant_sem",
        "per_trajectory_raw_moments",
    ):
        np.testing.assert_allclose(merged[key], full[key], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        merged["time_blocks"]["raw_moments"],
        full["time_blocks"]["raw_moments"],
        rtol=1e-12,
        atol=1e-12,
    )
    assert merged["n_traj"] == 11


def test_quadratic_moments_workspace_is_bounded_by_time_chunk():
    analyzer = _analyzer(
        modes=[0, 1, 2],
        observables={
            "one": {"matrix": np.eye(3).tolist()},
            "two": {"matrix": np.diag([1.0, -1.0, 0.0]).tolist()},
        },
        time_chunk_samples=4096,
    )
    short = AnalyserWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=64,
        saved_samples=100_000,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )
    long = AnalyserWorkspaceRequest(
        trajectory_bytes=1,
        n_traj=64,
        saved_samples=1_500_001,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="cupy",
    )

    short_estimate = analyzer.workspace(short)
    long_estimate = analyzer.workspace(long)
    assert short_estimate == long_estimate
    assert long_estimate.device_bytes < 16 * 1024**2
    assert long_estimate.host_bytes < 32 * 1024


def test_quadratic_moments_invocation_workspace_tracks_physical_batch():
    analyzer = _analyzer(
        modes=[0, 1, 2],
        observables={
            "one": {"matrix": np.eye(3).tolist()},
            "two": {"matrix": np.diag([1.0, -1.0, 0.0]).tolist()},
        },
        time_chunk_samples=128,
    )
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

    static_device = 2 * 3 * 3 * 2 * 8 + 2 * 8
    assert (small.device_bytes - static_device) * 2 == (
        large.device_bytes - static_device
    )
    assert small.host_bytes == large.host_bytes == 0


def test_quadratic_moments_retained_workspace_covers_ensemble_and_batches():
    analyzer = _analyzer(
        modes=[0, 1, 2],
        observables={
            "one": {"matrix": np.eye(3).tolist()},
            "two": {"matrix": np.diag([1.0, -1.0, 0.0]).tolist()},
        },
        max_order=4,
        time_blocks=5,
    )
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
    matrices = 2 * 3 * 3 * 2 * 8
    trajectory_payload = 100 * 2 * 4 * 8
    payload_and_concatenate = 2 * trajectory_payload
    leave_raw_and_construction = 2 * trajectory_payload
    cumulant_values = trajectory_payload
    variance_temporaries = 2 * trajectory_payload
    fixed_payload = (
        2 * matrices
        + 2 * 8
        + 5 * 2 * 4 * 8
        + 2 * 5 * 8
        + 3 * 8
        + 2 * 64
    )

    assert estimate.retained_host_bytes == (
        payload_and_concatenate
        + leave_raw_and_construction
        + cumulant_values
        + variance_temporaries
        + 4 * fixed_payload
    )
    assert estimate.retained_device_bytes == 0


def test_quadratic_moments_advertise_trajectory_batching():
    capabilities = _analyzer(
        observables={"occupation": {"matrix": [[1.0]]}}
    ).execution_capabilities()
    assert capabilities.preferred_location == "backend"
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False
