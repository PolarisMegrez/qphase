"""Tests for SDE Integrators."""

from typing import Any

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.integrator.euler_maruyama import EulerMaruyama, EulerMaruyamaConfig
from qphase_sde.integrator.milstein import Milstein, MilsteinConfig
from qphase_sde.integrator.srk import GenericSRK, GenericSRKConfig


class DummyModel:
    """Dummy model for testing integrators."""

    name: str = "dummy_model"
    n_modes: int = 1
    noise_basis: str = "real"
    noise_dim: int = 1
    params: dict[str, np.ndarray] = {}

    def __init__(self, noise_basis: str = "real") -> None:
        self.noise_basis = noise_basis

    def drift(self, y: Any, t: float, p: dict[str, np.ndarray]) -> Any:
        return -y

    def diffusion(self, y: Any, t: float, p: dict[str, np.ndarray]) -> Any:
        # Simple additive noise
        n = y.shape[0]
        if self.noise_basis == "real":
            # (n_traj, n_modes, noise_dim)
            return np.ones((n, 1, 1))
        else:
            # (n_traj, n_modes, noise_dim_complex)
            return np.ones((n, 1, 1)) + 0j


@pytest.fixture
def backend():
    return NumpyBackend()


def test_euler_maruyama_step(backend):
    """Test Euler-Maruyama step."""
    config = EulerMaruyamaConfig()
    integrator = EulerMaruyama(config)

    model = DummyModel()
    y = np.array([[1.0]])
    t = 0.0
    dt = 0.01
    dW = np.array([[0.1]])  # (n_traj, noise_dim)

    # dy = -y*dt + 1*dW = -0.01 + 0.1 = 0.09
    # y_new = 1.0 + 0.09 = 1.09

    y_new = integrator.step(y, t, dt, model, dW, backend)

    # Note: step returns dy, not y_new in some implementations, or y_new?
    # Let's check the implementation.
    # EulerMaruyama.step returns: a * dt + self._contract_fn(backend, L, dW)
    # So it returns dy.

    assert np.allclose(y_new, 0.09)


def test_milstein_step(backend):
    """Test Milstein step."""
    config = MilsteinConfig()
    integrator = Milstein(config)

    model = DummyModel()
    y = np.array([[1.0]])
    t = 0.0
    dt = 0.01
    dW = np.array([[0.1]])

    # Milstein correction for constant diffusion is 0.
    # So result should be same as EM.

    dy = integrator.step(y, t, dt, model, dW, backend)
    assert np.allclose(dy, 0.09)


def test_srk_step(backend):
    """Test SRK step."""
    config = GenericSRKConfig(method="heun")
    integrator = GenericSRK(config)

    model = DummyModel()
    y = np.array([[1.0]])
    t = 0.0
    dt = 0.01
    dW = np.array([[0.1]])

    # SRK is higher order, but for linear drift and constant diffusion,
    # it might differ slightly or match depending on the scheme.
    # Just check it runs and returns correct shape.

    dy = integrator.step(y, t, dt, model, dW, backend)
    assert dy.shape == y.shape
