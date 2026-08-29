"""Tests for SDE Engine."""

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend, NumpyConfig
from qphase_sde.engine import (
    Engine,
    EngineConfig,
    TrajectoryDivergenceError,
    _GroupedRNG,
)
from qphase_sde.integrator.base import ChunkStepResult
from qphase_sde.integrator.euler_maruyama import EulerMaruyama

pytestmark = pytest.mark.integration


def _trajectory_alpha(bundle):
    """(n_traj, n_time, n_channel) array of the trajectory product."""
    dataset = bundle.products["trajectories"]
    return dataset.handle("alpha").materialize()[0]


def _trajectory_times(bundle):
    """Observation-window sample times of the trajectory product."""
    axis = bundle.products["trajectories"].axis("time")
    return axis.start + axis.step * np.arange(axis.size)


class DummySDEModel:
    """Dummy SDE model for engine tests."""

    name = "dummy_sde"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1
    params: dict[str, np.ndarray] = {}

    def drift(self, y, t, p):
        return -y

    def diffusion(self, y, t, p):
        n = y.shape[0]
        return np.ones((n, 1, 1))


class TwoModeModel:
    name = "two_mode"
    n_modes = 2
    noise_basis = "real"
    noise_dim = 2
    params = {}

    def drift(self, y, t, p):
        return np.zeros_like(y)

    def diffusion(self, y, t, p):
        return np.zeros(y.shape[:-1] + (2, 2), dtype=y.dtype)


class UnitDriftModel:
    name = "unit_drift"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1
    params = {}

    def drift(self, y, t, p):
        return np.ones_like(y)

    def diffusion(self, y, t, p):
        return np.zeros(y.shape[:-1] + (1, 1), dtype=y.dtype)


def test_engine_initialization():
    """Test engine initialization."""
    config = EngineConfig(dt=0.01, t0=0.0, t1=0.1, n_traj=2, seed=None, ic=None)
    engine = Engine(config)
    assert engine.config is not None
    assert engine.config.dt == 0.01
    assert engine.config.n_traj == 2


def test_engine_run():
    """Test engine run execution."""
    # Provide initial condition (n_traj, n_modes) = (2, 1)
    ic = np.zeros((2, 1))
    config = EngineConfig(dt=0.01, t0=0.0, t1=0.05, n_traj=2, seed=None, ic=ic)

    backend = NumpyBackend()
    integrator = EulerMaruyama()
    model = DummySDEModel()

    plugins = {"backend": backend, "integrator": integrator, "model": model}

    engine = Engine(config=config, plugins=plugins)

    # Run simulation
    bundle = engine.run()

    assert bundle is not None
    assert "trajectories" in bundle.products
    # t=0 to t=0.05 with dt=0.01 -> 0, 0.01, 0.02, 0.03, 0.04, 0.05 -> 6 points
    # Shape: (n_traj, n_steps, n_modes) or similar.
    # Check actual shape from result
    alpha = _trajectory_alpha(bundle)
    assert alpha.shape[0] == 2  # n_traj
    assert alpha.shape[1] >= 5  # n_steps


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_randn_into_preserves_legacy_seeded_stream(dtype):
    backend = NumpyBackend()
    expected = backend.randn(backend.rng(17), (7, 3), dtype=np.dtype(dtype))
    actual = np.empty((7, 3), dtype=dtype)

    returned = backend.randn_into(backend.rng(17), actual)

    assert returned is actual
    np.testing.assert_array_equal(actual, expected)


def test_engine_reuses_step_noise_buffer():
    class RecordingBackend(NumpyBackend):
        def __init__(self):
            super().__init__()
            self.noise_buffers = []

        def randn_into(self, rng, out):
            self.noise_buffers.append(id(out))
            return super().randn_into(rng, out)

    backend = RecordingBackend()
    engine = Engine(
        config=EngineConfig(
            dt=0.01,
            t0=0.0,
            t1=0.05,
            n_traj=2,
            seed=11,
            ic=[[0.0]],
        ),
        plugins={
            "backend": backend,
            "integrator": EulerMaruyama(),
            "model": DummySDEModel(),
        },
    )

    engine.run()

    assert len(backend.noise_buffers) == 5
    assert len(set(backend.noise_buffers)) == 1


def test_grouped_rng_fills_noncontiguous_chunk_slices():
    backend = NumpyBackend()
    actual = np.empty((3, 8, 2), dtype=np.float64)
    grouped = _GroupedRNG((backend.rng(31), backend.rng(37)), group_size=4)

    Engine._draw_standard_normal_into(backend, grouped, actual)

    expected = np.concatenate(
        (
            backend.randn(backend.rng(31), (3, 4, 2), dtype=np.float64),
            backend.randn(backend.rng(37), (3, 4, 2), dtype=np.float64),
        ),
        axis=1,
    )
    np.testing.assert_array_equal(actual, expected)


class DummyChunkIntegrator:
    class Config:
        """Minimal chunk configuration."""

        chunk_steps = 4

    config = Config()

    def supports_chunk_step(self, model, backend):
        return True

    def step_chunk(
        self,
        y,
        t,
        dt,
        model,
        noise,
        backend,
        *,
        n_steps,
        save_offsets,
        record_modes,
    ):
        del t, model, noise, backend, record_modes
        if save_offsets:
            saved = np.stack([y + offset * dt for offset in save_offsets], axis=1)
        else:
            saved = np.empty((y.shape[0], 0, y.shape[1]), dtype=y.dtype)
        return ChunkStepResult(final_state=y + n_steps * dt, saved_states=saved)


def test_engine_chunk_path_preserves_save_boundaries():
    config = EngineConfig(dt=0.1, t0=0.0, t1=1.0, n_traj=2, seed=7, ic=[[0.0]])
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(NumpyConfig(float_dtype="float32")),
            "integrator": DummyChunkIntegrator(),
            "model": DummySDEModel(),
        },
    )

    trajectory = engine.run_sde(
        model=DummySDEModel(),
        ic=[[0.0]],
        time={"t0": 0.0, "dt": 0.1, "steps": 10},
        n_traj=2,
        seed=7,
        return_stride=3,
    )

    assert trajectory.data.shape == (2, 4, 1)
    np.testing.assert_allclose(trajectory.data[0, :, 0], [0.0, 0.3, 0.6, 0.9])


def test_engine_integrates_warmup_but_only_returns_observation_window():
    config = EngineConfig(
        dt=0.1,
        t0=0.2,
        t1=1.0,
        n_traj=1,
        seed=7,
        ic=[[0.0]],
        save_stride=3,
    )
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(NumpyConfig(float_dtype="float32")),
            "integrator": DummyChunkIntegrator(),
            "model": DummySDEModel(),
        },
    )

    bundle = engine.run()
    trajectory = bundle.products["trajectories"]
    axis = trajectory.axis("time")

    assert axis.start == pytest.approx(0.2)
    assert axis.step == pytest.approx(0.3)
    assert trajectory.attributes["integration_t0"] == pytest.approx(0.0)
    assert trajectory.attributes["warmup_steps"] == 2
    np.testing.assert_allclose(_trajectory_times(bundle), [0.2, 0.5, 0.8])
    np.testing.assert_allclose(_trajectory_alpha(bundle)[0, :, 0], [0.2, 0.5, 0.8])


def test_engine_non_chunk_path_samples_from_observation_start():
    config = EngineConfig(
        dt=0.1,
        t0=0.2,
        t1=1.0,
        n_traj=1,
        seed=7,
        ic=[[0.0]],
        save_stride=3,
    )
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": UnitDriftModel(),
        },
    )

    bundle = engine.run()

    np.testing.assert_allclose(_trajectory_times(bundle), [0.2, 0.5, 0.8])
    np.testing.assert_allclose(_trajectory_alpha(bundle)[0, :, 0], [0.2, 0.5, 0.8])


def test_engine_state_norm_guard_rejects_escaped_chunk_trajectory():
    config = EngineConfig(
        dt=0.1,
        t0=0.0,
        t1=1.0,
        n_traj=1,
        seed=7,
        ic=[[0.0]],
        max_state_norm=0.25,
        state_check_interval_steps=1,
    )
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(NumpyConfig(float_dtype="float32")),
            "integrator": DummyChunkIntegrator(),
            "model": DummySDEModel(),
        },
    )

    with pytest.raises(TrajectoryDivergenceError, match="No PSD was produced"):
        engine.run_sde(
            model=DummySDEModel(),
            ic=[[0.0]],
            time={"t0": 0.0, "dt": 0.1, "steps": 10},
            n_traj=1,
            seed=7,
        )


def test_engine_records_selected_modes_in_state_dtype():
    ic = np.array([[1.0 + 2.0j, 3.0 + 4.0j]], dtype=np.complex64)
    config = EngineConfig(
        dt=0.1,
        t0=0.0,
        t1=0.2,
        n_traj=1,
        seed=8,
        ic=ic,
        record_modes=[1],
    )
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(NumpyConfig(float_dtype="float32")),
            "integrator": EulerMaruyama(),
            "model": TwoModeModel(),
        },
    )

    trajectory = engine.run_sde(
        model=TwoModeModel(),
        ic=ic,
        time={"t0": 0.0, "dt": 0.1, "steps": 2},
        n_traj=1,
        seed=8,
    )

    assert trajectory.data.shape == (1, 3, 1)
    assert trajectory.data.dtype == np.complex64
    assert trajectory.meta["mode_indices"] == [1]
    np.testing.assert_allclose(trajectory.data[0, :, 0], 3.0 + 4.0j)


@pytest.mark.parametrize("record_modes", [[0, 0], [2], [-1]])
def test_engine_rejects_invalid_record_modes(record_modes):
    config = EngineConfig(
        dt=0.1,
        t0=0.0,
        t1=0.1,
        n_traj=1,
        ic=[[0.0, 0.0]],
        record_modes=record_modes,
    )
    engine = Engine(config=config, plugins={"backend": NumpyBackend()})

    with pytest.raises(ValueError, match="record_modes"):
        engine.run_sde(
            model=TwoModeModel(),
            ic=[[0.0, 0.0]],
            time={"t0": 0.0, "dt": 0.1, "steps": 1},
            n_traj=1,
            solver=EulerMaruyama(),
            seed=9,
        )
