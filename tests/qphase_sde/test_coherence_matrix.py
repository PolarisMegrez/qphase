import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.base import AnalyzerWorkspaceRequest
from qphase_sde.analyser.coherence_matrix import (
    CoherenceMatrixAnalyzer,
    CoherenceMatrixConfig,
)
from qphase_sde.state import TrajectorySet


def _analyzer(**kwargs):
    return CoherenceMatrixAnalyzer(CoherenceMatrixConfig(**kwargs))


def test_coherence_matrix_reports_rank_one_modal_purity():
    vector = np.asarray([1.0 + 1.0j, 2.0 - 0.5j])
    data = np.broadcast_to(vector, (6, 64, 2)).copy()
    payload = (
        _analyzer(time_blocks=4)
        .analyze(TrajectorySet(data, t0=10.0, dt=0.5), NumpyBackend())
        .data_dict
    )

    expected = np.outer(vector, vector.conj())
    np.testing.assert_allclose(payload["matrix"], expected)
    assert np.isclose(payload["purity"], 1.0)
    assert np.isclose(payload["effective_rank"], 1.0)
    assert np.isclose(payload["principal_fraction"], 1.0)
    assert payload["uncertainty"]["independent_unit"] == "trajectory"
    assert payload["time_blocks"]["count"] == 2


def test_coherence_matrix_distinguishes_incoherent_modal_mixture():
    data = np.zeros((8, 32, 2), dtype=complex)
    data[:4, :, 0] = 1.0
    data[4:, :, 1] = 1.0
    payload = (
        _analyzer(time_blocks=4, min_block_samples=4)
        .analyze(TrajectorySet(data), NumpyBackend())
        .data_dict
    )

    np.testing.assert_allclose(payload["matrix"], 0.5 * np.eye(2))
    np.testing.assert_allclose(payload["normalized_eigenvalues"], [0.5, 0.5])
    assert np.isclose(payload["purity"], 0.5)
    assert np.isclose(payload["effective_rank"], 2.0)
    assert np.isclose(payload["spectral_entropy"], np.log(2.0))
    assert payload["minimum_eigenvalue"] >= 0.0


def test_coherence_matrix_resolves_physical_recorded_modes():
    data = np.zeros((3, 16, 2), dtype=complex)
    data[:, :, 0] = 2.0
    data[:, :, 1] = 3.0j
    trajectory = TrajectorySet(data, meta={"mode_indices": [2, 5]})
    payload = _analyzer(modes=[5]).analyze(trajectory, NumpyBackend()).data_dict

    assert payload["modes"] == [5]
    np.testing.assert_allclose(payload["matrix"], [[9.0]])
    assert np.isclose(payload["purity"], 1.0)


def test_coherence_matrix_accumulator_matches_full_ensemble():
    rng = np.random.default_rng(20260809)
    data = rng.normal(size=(11, 48, 3)) + 1j * rng.normal(size=(11, 48, 3))
    trajectory = TrajectorySet(data, t0=5.0, dt=0.25)
    analyzer = _analyzer(time_blocks=6, min_block_samples=4)
    full = analyzer.analyze(trajectory, NumpyBackend()).data_dict
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 3), (3, 7), (7, 11)):
        partial = analyzer.analyze(
            TrajectorySet(data[start:stop], t0=5.0, dt=0.25), NumpyBackend()
        ).data_dict
        accumulator.update(partial)
    merged = accumulator.finalize()

    for key in (
        "matrix",
        "matrix_sem",
        "normalized_matrix",
        "eigenvalues",
        "purity_ci",
        "per_trajectory_matrix",
    ):
        np.testing.assert_allclose(merged[key], full[key], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(
        merged["time_blocks"]["matrix"],
        full["time_blocks"]["matrix"],
        rtol=1e-13,
        atol=1e-13,
    )
    assert np.isclose(merged["purity"], full["purity"])
    assert np.isclose(merged["purity_sem"], full["purity_sem"])
    assert merged["n_traj"] == 11


def test_coherence_matrix_advertises_trajectory_batching():
    capabilities = _analyzer().capabilities()
    assert capabilities.execution_location == "backend"
    assert capabilities.requires_full_trajectory is True
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False


def test_coherence_matrix_workspace_is_bounded_by_time_chunk():
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
