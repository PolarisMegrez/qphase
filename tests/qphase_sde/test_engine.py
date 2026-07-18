"""Tests for SDE Engine."""

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend, NumpyConfig
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.base import ChunkStepResult
from qphase_sde.integrator.euler_maruyama import EulerMaruyama

pytestmark = pytest.mark.integration


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
    result = engine.run()

    assert result is not None
    # assert result.success  # SDEResult doesn't have success attribute
    assert hasattr(result, "trajectory")
    # t=0 to t=0.05 with dt=0.01 -> 0, 0.01, 0.02, 0.03, 0.04, 0.05 -> 6 points
    # Shape: (n_traj, n_steps, n_modes) or similar.
    # Check actual shape from result
    assert result.trajectory.data.shape[0] == 2  # n_traj
    assert result.trajectory.data.shape[1] >= 5  # n_steps


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
        saved = np.stack([y + offset * dt for offset in save_offsets], axis=1)
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
