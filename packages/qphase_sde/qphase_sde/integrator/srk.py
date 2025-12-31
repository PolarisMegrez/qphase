"""qphase_sde: Generic SRK Integrator
---------------------------------------------------------
Implements a generic SRK solver that can be configured to behave as
Euler-Maruyama, Heun, or higher-order schemes. Supports adaptive stepping.

Public API
----------
``GenericSRK`` : Generic Stochastic Runge-Kutta integrator implementation.
``GenericSRKConfig`` : Configuration for Generic SRK integrator.
"""

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, Field
from qphase.backend.base import BackendBase

from qphase_sde.integrator.base import Integrator
from qphase_sde.model import NoiseSpec, SDEModel


class GenericSRKConfig(BaseModel):
    """Configuration for Generic SRK integrator."""

    method: str = Field(
        "heun",
        description="Integration scheme (euler, heun, or custom)",
    )


def _expand_complex_noise_backend(Lc: Any, backend: BackendBase) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis."""
    a = backend.real(Lc)
    b = backend.imag(Lc)
    s = (2.0) ** 0.5
    Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
    Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
    return Lr_real + 1j * Lr_imag


class GenericSRK(Integrator):
    """Generic Stochastic Runge-Kutta Integrator.

    Supports various Butcher tableaus for SDEs.
    Default is Heun's method (weak order 2.0, strong order 1.0).
    """

    name: ClassVar[str] = "srk"
    description: ClassVar[str] = "Generic Stochastic Runge-Kutta solver"
    config_schema: ClassVar[type[GenericSRKConfig]] = GenericSRKConfig

    def __init__(self, config: GenericSRKConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = GenericSRKConfig(**kwargs)
        self.config = config
        self.method = config.method
        # self.tol removed, passed via step_adaptive
        self.adaptive_factor = 0.9
        self.adaptive_factor = 0.9
        self.min_dt = 1e-9
        self.max_dt = 1.0
        # TODO: Load coefficients based on method

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        model: SDEModel,
        noise: Any,
        backend: BackendBase,
    ) -> Any:
        """Perform one fixed step."""
        drift = model.drift(y, t, model.params)
        diffusion = model.diffusion(y, t, model.params)
        dW = noise

        if getattr(model, "noise_basis", "real") == "complex":
            diffusion = _expand_complex_noise_backend(diffusion, backend)

        if self.method == "euler":
            # Euler-Maruyama (Strong Order 0.5)
            # dy = a(y) dt + b(y) dW
            diff_term = backend.einsum("...ij,...j->...i", diffusion, dW)
            dy = drift * dt + diff_term
            return dy

        elif self.method == "heun":
            # Stochastic Heun (Stratonovich, Strong Order 1.0 approx)
            # Predictor
            diff_term = backend.einsum("...ij,...j->...i", diffusion, dW)
            y_bar = y + drift * dt + diff_term

            # Corrector
            drift_bar = model.drift(y_bar, t + dt, model.params)
            diffusion_bar = model.diffusion(y_bar, t + dt, model.params)

            if getattr(model, "noise_basis", "real") == "complex":
                diffusion_bar = _expand_complex_noise_backend(diffusion_bar, backend)

            diff_term_bar = backend.einsum("...ij,...j->...i", diffusion_bar, dW)

            dy = 0.5 * (drift + drift_bar) * dt + 0.5 * (diff_term + diff_term_bar)
            return dy

        else:
            raise ValueError(f"Unknown method: {self.method}")

    def step_adaptive(
        self,
        y: Any,
        t: float,
        dt: float,
        tol: float,
        model: SDEModel,
        noise: NoiseSpec,
        backend: BackendBase,
        rng: Any = None,
    ) -> tuple[Any, float, float, float]:
        """Perform one adaptive step using step doubling (Richardson Extrapolation)."""
        # 1. Generate noise for fine steps
        # dW_a, dW_b ~ N(0, dt/2)
        shape = y.shape[:-1] + (noise.dim,)
        sqrt_dt_2 = np.sqrt(dt / 2)

        # Ensure rng is provided
        if rng is None:
            # Fallback if rng not provided (should not happen in engine)
            # But backend.randn might fail or use global state
            dW_a = backend.randn(None, shape, dtype=float) * sqrt_dt_2
            dW_b = backend.randn(None, shape, dtype=float) * sqrt_dt_2
        else:
            dW_a = backend.randn(rng, shape, dtype=float) * sqrt_dt_2
            dW_b = backend.randn(rng, shape, dtype=float) * sqrt_dt_2

        dW = dW_a + dW_b

        # 2. Coarse step (dt)
        dy_coarse = self.step(y, t, dt, model, dW, backend)
        y_coarse = y + dy_coarse

        # 3. Fine steps (dt/2)
        dy_mid = self.step(y, t, dt / 2, model, dW_a, backend)
        y_mid = y + dy_mid

        dy_fine = self.step(y_mid, t + dt / 2, dt / 2, model, dW_b, backend)
        y_fine = y_mid + dy_fine

        # 4. Error estimate
        # error = max(|y_fine - y_coarse|)
        diff = abs(y_fine - y_coarse)
        if hasattr(diff, "max"):
            error = float(diff.max())
        else:
            error = float(np.max(diff))

        # 5. Adapt dt
        # Heuristic for SDEs (assuming p=1.0 for Heun)
        if error < 1e-16:
            factor = 2.0
        else:
            factor = 0.9 * (tol / error) ** 0.5

        factor = min(5.0, max(0.2, factor))

        next_dt = dt * factor
        next_dt = min(self.max_dt, max(self.min_dt, next_dt))

        if error <= tol or dt <= self.min_dt:
            # Accept: return fine solution (better accuracy)
            # If dt <= min_dt, we force acceptance to avoid infinite loops
            return y_fine, t + dt, next_dt, error
        else:
            # Reject: return old state
            return y, t, next_dt, error

    def supports_adaptive_step(self) -> bool:
        return True

    def reset(self) -> None:
        """Reset internal state (no-op for GenericSRK)."""
        pass

    def supports_strided_state(self) -> bool:
        """Strided state not supported by GenericSRK."""
        return False
