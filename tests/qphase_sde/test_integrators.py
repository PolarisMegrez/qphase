"""Tests for SDE Integrators."""

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.integrator.euler_maruyama import EulerMaruyama, EulerMaruyamaConfig
from qphase_sde.integrator.milstein import Milstein, MilsteinConfig
from qphase_sde.integrator.srk import GenericSRK, GenericSRKConfig
from qphase_sde.model import NoiseSpec


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


def test_srk_rejects_unsupported_method_during_config_validation():
    """Unsupported schemes must fail before an integration run starts."""
    with pytest.raises(ValidationError, match="Input should be 'euler' or 'heun'"):
        GenericSRKConfig(method="custom")  # type: ignore[arg-type]


class _NormalFacade:
    """Generator-style facade used to test the integrator RNG contract."""

    def __init__(self, seed):
        self._gen = np.random.default_rng(seed)
        self.calls = 0

    def normal(self, size=None):
        self.calls += 1
        return self._gen.normal(size=size)


def test_draw_standard_normal_routes_facade_and_native_rng(backend):
    """The integrator RNG contract accepts Generator-style facades."""
    from qphase_sde.integrator.base import draw_standard_normal

    facade = _NormalFacade(7)
    out = draw_standard_normal(facade, (4, 2), np.float32, backend)
    assert facade.calls == 1
    assert out.shape == (4, 2)
    assert out.dtype == np.float32

    # Native backend handles keep the legacy draw path bit-for-bit.
    native = draw_standard_normal(backend.rng(11), (8,), np.float64, backend)
    expected = backend.randn(backend.rng(11), (8,), dtype=np.float64)
    np.testing.assert_array_equal(native, expected)


def test_srk_adaptive_step_draws_through_rng_contract(backend):
    """GenericSRK adaptive stepping draws both half-steps via the contract."""
    integrator = GenericSRK(GenericSRKConfig(method="heun"))
    model = DummyModel()
    noise = NoiseSpec(kind="independent", dim=1)
    y = np.ones((4, 1))

    facade = _NormalFacade(13)
    y_next, t_next, dt_next, error = integrator.step_adaptive(
        y, 0.0, 0.01, 1e-3, model, noise, backend, facade
    )
    assert facade.calls == 2  # dW_a and dW_b both routed through the facade
    assert np.asarray(y_next).shape == y.shape
    assert t_next > 0.0

    # A bare numpy Generator (single-group case) takes the same code path.
    gen = np.random.default_rng(13)
    y_gen, t_gen, dt_gen, err_gen = integrator.step_adaptive(
        y, 0.0, 0.01, 1e-3, model, noise, backend, gen
    )
    np.testing.assert_array_equal(np.asarray(y_next), np.asarray(y_gen))
