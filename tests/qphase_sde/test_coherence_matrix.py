import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.coherence_matrix import (
    CoherenceMatrixAnalyzer,
    CoherenceMatrixConfig,
)
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.result import _trajectory_product
from qphase_sde.state import TrajectorySet

from .conftest import product_payload


def _analyzer(**kwargs):
    return CoherenceMatrixAnalyzer(CoherenceMatrixConfig(**kwargs))


def _payload(analyzer, trajectory: TrajectorySet):
    product = analyzer.execute(
        {
            "trajectories": _trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=NumpyBackend(),
        label="coherence_matrix",
    )["coherence_matrix"]
    return product, product_payload(product)


def test_coherence_matrix_reports_rank_one_modal_purity():
    vector = np.asarray([1.0 + 1.0j, 2.0 - 0.5j])
    data = np.broadcast_to(vector, (6, 64, 2)).copy()
    _, payload = _payload(
        _analyzer(time_blocks=4), TrajectorySet(data, t0=10.0, dt=0.5)
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
    _, payload = _payload(
        _analyzer(time_blocks=4, min_block_samples=4), TrajectorySet(data)
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
    _, payload = _payload(_analyzer(modes=[5]), trajectory)

    assert payload["modes"] == [5]
    np.testing.assert_allclose(payload["matrix"], [[9.0]])
    assert np.isclose(payload["purity"], 1.0)


def test_coherence_matrix_accumulator_matches_full_ensemble():
    rng = np.random.default_rng(20260809)
    data = rng.normal(size=(11, 48, 3)) + 1j * rng.normal(size=(11, 48, 3))
    trajectory = TrajectorySet(data, t0=5.0, dt=0.25)
    analyzer = _analyzer(time_blocks=6, min_block_samples=4)
    _, full = _payload(analyzer, trajectory)
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 3), (3, 7), (7, 11)):
        partial, _ = _payload(
            analyzer, TrajectorySet(data[start:stop], t0=5.0, dt=0.25)
        )
        accumulator.consume_product(partial)
    products = accumulator.finalize_product("coherence_matrix")
    assert products is not None
    merged = product_payload(products["coherence_matrix"])

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
    capabilities = _analyzer().execution_capabilities()
    assert capabilities.preferred_location == "backend"
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False


def test_coherence_matrix_workspace_is_bounded_by_time_chunk():
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


def test_coherence_matrix_invocation_workspace_tracks_physical_batch():
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


def test_coherence_matrix_retained_workspace_covers_ensemble_and_batches():
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
    matrix_payload = 100 * 3 * 3 * 2 * 8
    amplitude_payload = 100 * 3 * 2 * 8
    payload_and_concatenate = 2 * (matrix_payload + amplitude_payload)
    finalize_matrix_workspace = 3 * matrix_payload
    fixed_payload = 5 * 3 * 3 * 2 * 8 + 2 * 5 * 8 + 3 * 8

    assert estimate.retained_host_bytes == (
        payload_and_concatenate + finalize_matrix_workspace + 4 * fixed_payload
    )
    assert estimate.retained_device_bytes == 0
