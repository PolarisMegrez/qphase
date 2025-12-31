"""qphase_sde: Euler-Maruyama Integrator
------------------------------------
Reference Itô SDE solver with backend-optimized contractions, integrated with
the central registry for discovery and composition.

Behavior
--------
- Backend-agnostic step rule ``dy = a(y,t)·dt + L(y,t) @ dW``; contraction over
  noise channels is specialized per backend when possible. Complex noise bases
  are expanded to an equivalent real basis prior to contraction.
"""

from collections.abc import Callable
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


def _expand_complex_noise_backend(Lc: Any, backend: Backend) -> Any:
    """Expand complex-basis diffusion matrix to an equivalent real basis.

    Transforms L_c ∈ C^{..., n_modes, M_c} into L_r ∈ C^{..., n_modes, 2·M_c}
    using only backend operations, preserving contraction with real noise.
    """
    a = backend.real(Lc)
    b = backend.imag(Lc)
    s = (2.0) ** 0.5
    Lr_real = backend.concatenate((a / s, -b / s), axis=-1)
    Lr_imag = backend.concatenate((b / s, a / s), axis=-1)
    return Lr_real + 1j * Lr_imag


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
        """Initialize the integrator and internal contraction cache.

        Lazily specializes a fast-path contraction function on first use based on
        the backend.
        """
        self.config = config or EulerMaruyamaConfig(**kwargs)
        # Lazily initialized fast-path functions based on backend
        self._contract_fn: Callable[[Backend, Any, Any], Any] | None = None

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
        dW = noise
        a = model.drift(y, t, model.params)  # (n_traj, n_modes)
        L = model.diffusion(y, t, model.params)  # (n_traj, n_modes, M_b)
        if getattr(model, "noise_basis", "real") == "complex":
            L = _expand_complex_noise_backend(L, backend)
        # Initialize fast-path at first use
        if self._contract_fn is None:
            try:
                be_name = str(backend.backend_name()).lower()
            except Exception:
                be_name = ""
            if be_name == "torch":
                try:
                    import torch as _th

                    def _contract(_backend: Backend, _L: Any, _dW: Any):
                        # (_T, N, M) bmm (_T, M, 1) -> (_T, N)
                        return _th.bmm(_L, _dW.unsqueeze(-1)).squeeze(-1)

                    self._contract_fn = _contract
                except Exception:
                    self._contract_fn = None
            # default fallback
            if self._contract_fn is None:

                def _fallback_contract(_backend: Backend, _L: Any, _dW: Any) -> Any:
                    return _backend.einsum("tnm,tm->tn", _L, _dW)

                self._contract_fn = _fallback_contract
        # Contract noise channels: (tnm, tm) -> (tn)
        return a * dt + self._contract_fn(backend, L, dW)

    def supports_adaptive_step(self) -> bool:
        return False

    def reset(self) -> None:
        """Reset internal caches (no-op for Euler-Maruyama)."""
        self._contract_fn = None

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
