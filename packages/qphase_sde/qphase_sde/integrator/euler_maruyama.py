"""qphase_sde: Euler-Maruyama Integrator
---------------------------------------------------------
Reference It么 SDE solver with backend-optimized contractions, integrated with
the central registry for discovery and composition.

Behavior
--------
- Backend-agnostic step rule ``dy = a(y,t)路dt + L(y,t) @ dW``; contraction over
  noise channels is specialized per backend when possible. Complex noise bases
  are expanded to an equivalent real basis prior to contraction.

Public API
----------
``EulerMaruyama`` : Euler-Maruyama integrator implementation.
"""

from typing import Any, ClassVar

from pydantic import BaseModel
from qphase.backend.base import BackendBase as Backend

from .base import Integrator

__all__ = [
    "EulerMaruyama",
    "EulerMaruyamaConfig",
]


class EulerMaruyamaConfig(BaseModel):
    """Configuration for Euler-Maruyama integrator."""

    # No specific configuration needed for standard EM
    pass


class EulerMaruyama(Integrator):
    """Euler–Maruyama integrator for SDEs with backend-optimized contractions.

    This solver implements the classic Euler–Maruyama time stepping under the
    Itô interpretation. It is backend-agnostic: contractions over noise channels
    are delegated to the active backend (NumPy/Numba/CuPy/Torch). When the model
    declares a complex noise basis, the diffusion is expanded to an equivalent
    real basis internally to match the engine's real-valued noise increments.

    Attributes
    ----------
    _contract_fn : Optional[Callable[[Backend, Any, Any], Any]]
            An internal fast-path contraction function specialized on first use based
            on the backend. For Torch, a batched-matmul is used; otherwise falls back
            to ``backend.einsum('tnm,tm->tn', L, dW)``.

    Methods
    -------
    step(y, t, dt, model, dW, backend)
            Advance the state by one step according to
            ``y_{t+dt} = y_t + a(y_t,t)·dt + L(y_t,t) @ dW``.

    Examples
    --------
    >>> # 典型用法：直接 import 并实例化
    >>> from qphase_sde.integrators.euler_maruyama import EulerMaruyama
    >>> integrator = EulerMaruyama()
    >>> # integrator.step(y, t, dt, model, dW, backend)

    References
    ----------
    - Kloeden, P. E., & Platen, E. (1992). Numerical Solution of Stochastic
      Differential Equations. Springer. (Euler–Maruyama scheme)
      doi:10.1007/978-3-662-12616-5
    - Higham, D. J. (2001). An Algorithmic Introduction to Numerical Simulation
      of Stochastic Differential Equations. SIAM Review, 43(3), 525–546.
      doi:10.1137/S0036144500378302
    - Gardiner, C. W. (2009). Stochastic Methods: A Handbook for the Natural and
      Social Sciences (4th ed.). Springer.

    Attributes
    ----------
    name : str
        Unique identifier for this integrator.
    description : str
        Human-readable description of this integrator.
    config_schema : type
        Configuration schema for this integrator.

    """

    name: ClassVar[str] = "euler_maruyama"
    description: ClassVar[str] = (
        "Euler–Maruyama integrator for stochastic differential equations. "
        "A first-order explicit method suitable for general SDE systems."
    )
    config_schema: ClassVar[type[EulerMaruyamaConfig]] = EulerMaruyamaConfig

    def __init__(self, config: EulerMaruyamaConfig | None = None, **kwargs) -> None:
        """Initialize the integrator."""
        self.config = config or EulerMaruyamaConfig(**kwargs)

    def step(
        self, y: Any, t: float, dt: float, model: Any, noise: Any, backend: Backend
    ) -> Any:
        """Compute one-step increment ``dy`` using the Euler–Maruyama scheme.

        The update follows ``dy = a(y,t)·dt + L(y,t) @ dW``, where ``a`` is the
        drift and ``L`` the diffusion matrix. If the model declares a complex noise
        basis (``noise_basis == 'complex'``), the diffusion is expanded to a real
        basis before contracting with the real-valued increment ``dW``.

        Parameters
        ----------
        y : Any
                State array with shape ``(n_traj, n_modes)`` (complex).
        t : float
                Current simulation time.
        dt : float
                Time step size (positive).
        model : Any
                Object providing ``drift(y, t, params)`` and ``diffusion(y, t, params)``
                evaluated on ``y``; may define ``noise_basis`` in {'real','complex'}.
        noise : Any
                Noise increment array (dW) with shape ``(n_traj, M)`` (real).
                Note: In this version, dW is expected to be Gaussian noise
                scaled by sqrt(dt).
                The engine is responsible for generating this noise.
        backend : Backend
                Active backend implementing array operations and contractions.

        Returns
        -------
        Any
                Increment ``dy`` with the same shape as ``y`` (complex).

        Examples
        --------
        >>> # dy = em.step(y, t, dt, model, dW, backend)  # doctest: +SKIP

        """
        from qphase_sde import ops

        dW = noise
        if ops.supports_kernelized_terms(model, backend):
            a, L = model.kernelized_terms(y, t, model.params, backend)
        else:
            a = model.drift(y, t, model.params)  # (n_traj, n_modes)
            L = model.diffusion(y, t, model.params)  # (n_traj, n_modes, M_b)
        if getattr(model, "noise_basis", "real") == "complex":
            L = ops.expand_complex_noise(L, backend)
        # Contract noise channels: (tnm, tm) -> (tn)
        return a * dt + ops.contract_noise(L, dW, backend)

    def supports_adaptive_step(self) -> bool:
        return False

    def reset(self) -> None:
        """Reset internal state (no-op for Euler-Maruyama)."""
        pass

    def step_adaptive(
        self,
        y: Any,
        t: float,
        dt: float,
        tol: float,
        model: Any,
        noise: Any,
        backend: Backend,
        rng: Any = None,
    ) -> tuple[Any, float, float, float]:
        """Adaptive stepping not supported by Euler-Maruyama."""
        raise NotImplementedError("Euler-Maruyama does not support adaptive stepping")

    def supports_strided_state(self) -> bool:
        """Strided state not supported by Euler-Maruyama."""
        return False
