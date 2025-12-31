"""Tests for SDE Engine."""

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama


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
