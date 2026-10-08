import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.analyser.allan_variance import (
    AllanVarianceAnalyzer,
    AllanVarianceConfig,
)
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.result import _trajectory_product
from qphase_sde.state import TrajectorySet


def _cupy_available() -> bool:
    try:
        import cupy  # noqa: F401
    except Exception:
        return False
    return True


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


def _product_payload(product):
    payload = {}
    values = list(product.attributes.get("payload_meta", {}).items()) + [
        (variable.name, product.handle(variable.name).materialize())
        for variable in product.variables
    ]
    for name, value in values:
        if np.asarray(value).ndim == 0:
            value = np.asarray(value).item()
        target = payload
        parts = name.split(".")
        for part in parts[:-1]:
            key = int(part) if part.isdigit() else part
            target = target.setdefault(key, {})
        target[parts[-1]] = value
    return payload


def _execute(analyzer, trajectory: TrajectorySet, backend=None):
    product = analyzer.execute(
        {
            "trajectories": _trajectory_product(
                trajectory, scan_size=None, n_traj_per_point=None
            )
        },
        backend=backend or NumpyBackend(),
        label="allan_variance",
    )["allan_variance"]
    return product, _product_payload(product)


def test_allan_variance_reports_nonoverlap_counts_and_frequency():
    _, payload = _execute(_analyzer(), _phase_diffusion_trajectory())
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

    _, phase_decreasing = _execute(
        AllanVarianceAnalyzer(
            AllanVarianceConfig(modes=[0], points=4, min_independent_windows=2)
        ),
        trajectory,
    )
    _, legacy = _execute(
        AllanVarianceAnalyzer(
            AllanVarianceConfig(
                modes=[0],
                points=4,
                min_independent_windows=2,
                orientation="phase_increasing",
            )
        ),
        trajectory,
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
    _, full = _execute(analyzer, trajectory)
    accumulator = analyzer.create_result_accumulator()
    for start, stop in ((0, 2), (2, 6), (6, 9)):
        partial, _ = _execute(
            analyzer,
            TrajectorySet(
                trajectory.data[start:stop],
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            ),
        )
        accumulator.consume_product(partial)
    products = accumulator.finalize_product("allan_variance")
    assert products is not None
    merged = _product_payload(products["allan_variance"])

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


def test_typed_allan_reducer_accepts_multiple_batches():
    trajectory = _phase_diffusion_trajectory()
    analyzer = _analyzer()
    reducer = analyzer.reducer()
    assert reducer is not None
    state = reducer.prepare(None, None)
    for start, stop in ((0, 4), (4, 9)):
        product, _ = _execute(
            analyzer,
            TrajectorySet(
                trajectory.data[start:stop],
                t0=trajectory.t0,
                dt=trajectory.dt,
                meta=dict(trajectory.meta),
            ),
        )
        reducer.consume(product, state, None)

    merged = reducer.finalize(state)["allan_variance"]
    assert merged.handle("n_traj").materialize() == 9
    assert "mode_results.2.allan.angular_frequency_variance" in {
        variable.name for variable in merged.variables
    }


def test_allan_analyzer_advertises_trajectory_batching():
    capabilities = _analyzer().execution_capabilities()
    assert capabilities.preferred_location == "backend"
    assert capabilities.supports_trajectory_batching is True
    assert capabilities.supports_time_streaming is False


def test_allan_config_rejects_empty_taus():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        AllanVarianceConfig(modes=[0], taus=[])
    with pytest.raises(ValidationError):
        AllanVarianceConfig(modes=[0], taus=[0.0])


def test_allan_workspace_is_host_resident_for_numpy_backend():
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(modes=[0, 1, 2], transfer_chunk_samples=8192)
    )
    trajectory_bytes = 60 * 1_000_001 * 3 * 16
    request = AnalyserWorkspaceRequest(
        trajectory_bytes=trajectory_bytes,
        n_traj=60,
        saved_samples=1_000_001,
        n_record_modes=3,
        real_itemsize=8,
        backend_name="numpy",
    )

    estimate = analyzer.workspace(request)
    assert estimate.host_bytes == 5 * trajectory_bytes // 3
    assert estimate.device_bytes == 0


@pytest.mark.parametrize(
    ("real_itemsize", "bytes_per_sample"),
    [(4, 56), (8, 80)],
)
def test_allan_workspace_is_trajectory_chunked_on_device(
    real_itemsize, bytes_per_sample
):
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(modes=[0, 1, 2], device_chunk_trajectories=16)
    )
    request = AnalyserWorkspaceRequest(
        trajectory_bytes=60 * 1_000_001 * 3 * 16,
        n_traj=60,
        saved_samples=1_000_001,
        n_record_modes=3,
        real_itemsize=real_itemsize,
        backend_name="cupy",
    )

    estimate = analyzer.workspace(request)
    assert estimate.device_bytes == 16 * 1_000_001 * bytes_per_sample
    # Host memory no longer materializes full modes; only tau tables remain.
    assert estimate.host_bytes == 60 * 256 * 4 * 8


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

    _, payload = _execute(analyzer, trajectory)
    _, reference = _execute(analyzer, source)

    assert max(device.transfer_lengths) <= 127
    assert sum(device.transfer_lengths) == source.n_steps
    np.testing.assert_allclose(
        payload["mode_results"][2]["allan"]["angular_frequency_variance"],
        reference["mode_results"][2]["allan"]["angular_frequency_variance"],
    )


class _DeviceLikeSlice:
    def __init__(self, array: np.ndarray, transfer_lengths: list[int]) -> None:
        self.array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype
        self.transfer_lengths = transfer_lengths

    def __getitem__(self, key):
        selected = self.array[key]
        self.transfer_lengths.append(int(selected.shape[1]))
        return _DeviceLikeSlice(selected, self.transfer_lengths)

    def get(self) -> np.ndarray:
        return self.array.copy()


class _DeviceLikeArray:
    def __init__(
        self, array: np.ndarray, transfer_lengths: list[int] | None = None
    ) -> None:
        self.array = array
        self.shape = array.shape
        self.ndim = array.ndim
        self.dtype = array.dtype
        self.transfer_lengths = transfer_lengths if transfer_lengths is not None else []

    def reshape(self, *shape):
        return _DeviceLikeArray(self.array.reshape(*shape), self.transfer_lengths)

    def __getitem__(self, key):
        selected = self.array[key]
        if self.ndim == 3:
            self.transfer_lengths.append(int(selected.shape[1]))
        return _DeviceLikeSlice(selected, self.transfer_lengths)


_ALLAN_COMPARE_KEYS = (
    "tau",
    "angular_frequency_variance",
    "angular_frequency_variance_sem",
    "per_trajectory",
    "valid_second_differences",
    "nonoverlap_angular_frequency_variance",
    "nonoverlap_angular_frequency_variance_sem",
    "nonoverlap_per_trajectory",
    "nonoverlap_valid_second_differences",
    "nominal_independent_windows_per_trajectory",
    "total_independent_window_count",
)

_PHASE_COMPARE_KEYS = (
    "mean_angular_frequency_per_trajectory",
    "max_abs_phase_step_per_trajectory",
    "near_nyquist_fraction_per_trajectory",
)


def _assert_mode_results_equal(
    device_payload: dict, host_payload: dict, mode: int
) -> None:
    device_mode = device_payload["mode_results"][mode]
    host_mode = host_payload["mode_results"][mode]
    for key in _ALLAN_COMPARE_KEYS:
        np.testing.assert_allclose(
            np.asarray(device_mode["allan"][key], dtype=float),
            np.asarray(host_mode["allan"][key], dtype=float),
            rtol=1e-10,
            atol=1e-30,
            err_msg=f"allan[{key}]",
        )
    for key in _PHASE_COMPARE_KEYS:
        np.testing.assert_allclose(
            np.asarray(device_mode["phase_increment"][key], dtype=float),
            np.asarray(host_mode["phase_increment"][key], dtype=float),
            rtol=1e-10,
            atol=1e-30,
            err_msg=f"phase_increment[{key}]",
        )


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_allan_device_path_matches_host_complex64_long_record():
    """complex64 long records must accumulate phase in float64 on device.

    np.angle/cp.angle differ at the float32 ulp level, which sets a host/device
    noise floor near 5e-9 for this record length; accumulating the phase in
    float32 produces a substantially larger deviation. SEM keys amplify the
    floor by cancellation across trajectories, so they use a 2e-7 tolerance
    while the primary estimates retain a 2e-8 tolerance.

    The test first executes a complex128 analysis in the same process. This
    guards mixed-precision sequencing instead of relying on pytest ordering.
    """
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    complex128_source = _phase_diffusion_trajectory(n_traj=3)
    complex128_device = TrajectorySet(
        cp.asarray(complex128_source.data),
        t0=complex128_source.t0,
        dt=complex128_source.dt,
        meta=dict(complex128_source.meta),
    )
    _execute(_analyzer(), complex128_device, CuPyBackend())

    rng = np.random.default_rng(20260901)
    n_traj, n_time = 4, 500_001
    dt = 0.1
    increments = (0.1 * rng.standard_normal((n_traj, n_time - 1))).astype(np.float32)
    phase = np.concatenate(
        [
            np.zeros((n_traj, 1), dtype=np.float32),
            np.cumsum(increments, axis=1, dtype=np.float32),
        ],
        axis=1,
    )
    amplitude = np.ones((n_traj, n_time), dtype=np.float32)
    amplitude[1, 100_000:105_000] = 0.0
    amplitude[3, 400_000:] = 0.0
    values = (amplitude * np.exp(1j * phase.astype(np.float64))).astype(np.complex64)[
        :, :, None
    ]
    trajectory = TrajectorySet(values, t0=0.0, dt=dt, meta={"mode_indices": [0]})
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(
            modes=[0],
            taus=[0.5, 1.0, 4.0, 16.0, 64.0],
            min_independent_windows=2,
            amplitude_floor=0.5,
            device_chunk_trajectories=3,
        )
    )

    _, reference = _execute(analyzer, trajectory)
    device_trajectory = TrajectorySet(
        cp.asarray(values), t0=0.0, dt=dt, meta={"mode_indices": [0]}
    )
    _, payload = _execute(analyzer, device_trajectory, CuPyBackend())

    for key in _ALLAN_COMPARE_KEYS:
        rtol = 2e-7 if key.endswith("_sem") else 2e-8
        np.testing.assert_allclose(
            np.asarray(payload["mode_results"][0]["allan"][key], dtype=float),
            np.asarray(reference["mode_results"][0]["allan"][key], dtype=float),
            rtol=rtol,
            atol=1e-30,
            err_msg=f"allan[{key}]",
        )
    for key in _PHASE_COMPARE_KEYS:
        # Phase summaries are computed from increments (no accumulation), so
        # the dtype regression does not affect them; only the float32 angle
        # ulp noise matters, including for near-zero mean frequencies.
        np.testing.assert_allclose(
            np.asarray(payload["mode_results"][0]["phase_increment"][key], dtype=float),
            np.asarray(
                reference["mode_results"][0]["phase_increment"][key], dtype=float
            ),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"phase_increment[{key}]",
        )
    # Masked trajectories lose windows relative to unmasked ones.
    valid = payload["mode_results"][0]["allan"]["valid_second_differences"]
    assert np.any(valid[3] < valid[1])


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_allan_device_path_matches_host():
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    source = _phase_diffusion_trajectory()
    # Odd chunking exercises the trailing partial chunk.
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(
            modes=[2],
            taus=[1.0, 2.0, 4.0],
            min_independent_windows=4,
            device_chunk_trajectories=4,
        )
    )
    _, reference = _execute(analyzer, source)
    device_trajectory = TrajectorySet(
        cp.asarray(source.data),
        t0=source.t0,
        dt=source.dt,
        meta=dict(source.meta),
    )

    _, payload = _execute(analyzer, device_trajectory, CuPyBackend())

    assert payload["n_traj"] == reference["n_traj"]
    assert payload["n_samples"] == reference["n_samples"]
    _assert_mode_results_equal(payload, reference, 2)


@pytest.mark.gpu
@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_allan_device_path_matches_host_with_masked_windows():
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    rng = np.random.default_rng(20260831)
    n_traj, n_time = 5, 513
    increments = 0.05 * rng.standard_normal((n_traj, n_time - 1))
    phase = np.concatenate(
        [np.zeros((n_traj, 1)), np.cumsum(increments, axis=1)], axis=1
    )
    amplitude = np.ones((n_traj, n_time))
    amplitude[0, 100:140] = 0.0
    amplitude[3, 300:] = 0.0
    values = (amplitude * np.exp(1j * phase))[:, :, None]
    trajectory = TrajectorySet(values, t0=0.0, dt=0.1, meta={"mode_indices": [0]})
    analyzer = AllanVarianceAnalyzer(
        AllanVarianceConfig(
            modes=[0],
            taus=[0.5, 1.0, 2.0, 4.0],
            min_independent_windows=2,
            amplitude_floor=0.5,
            device_chunk_trajectories=3,
        )
    )

    _, reference = _execute(analyzer, trajectory)
    device_trajectory = TrajectorySet(
        cp.asarray(values), t0=0.0, dt=0.1, meta={"mode_indices": [0]}
    )
    _, payload = _execute(analyzer, device_trajectory, CuPyBackend())

    _assert_mode_results_equal(payload, reference, 0)
    # Masked trajectories lose windows relative to unmasked ones.
    valid = payload["mode_results"][0]["allan"]["valid_second_differences"]
    assert np.any(valid[0] < valid[1])
    assert np.any(valid[3] < valid[1])
