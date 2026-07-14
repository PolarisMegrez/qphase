"""qphase_sde: Milstein Integrator
---------------------------------------------------------
Strong order-1.0 Milstein scheme for Itô SDEs under the commutative-noise
assumption (diagonal or commuting diffusion fields). Falls back to
Euler–Maruyama when a diffusion Jacobian is unavailable or the model uses a
complex noise basis that is not expanded with a matching Jacobian.

Behavior
--------
- Update rule (component-wise) under commutative noise:
- Update rule (component-wise) under commutative noise:
        y_{t+dt} = y_t + a(y_t,t)·dt + sum_k L_k(y_t,t)·dW_k
            + 1/2 · sum_k [
                (∑_j L_{j,k} ∂L_{:,k}/∂y_j) · (dW_k^2 - dt)
            ]
  where L_{:,k} is the k-th diffusion column, and ∂L_{:,k}/∂y_j denotes the
  Jacobian slice of L along the state dimension.

Public API
----------
``Milstein`` : Milstein integrator implementation.

Notes
-----
- Multi-dimensional, non-commutative noise generally requires Lévy area terms;
  those are not included here. For typical diagonal/commutative cases this
  implementation provides the classic Milstein correction.
- If ``model.noise_basis == 'complex'``, this implementation currently falls
  back to Euler–Maruyama unless the provided Jacobian matches a real-expanded
  diffusion; see implementation notes.

"""

from typing import Any, ClassVar

from pydantic import BaseModel
from qphase.backend.base import BackendBase as Backend

from .base import Integrator

__all__ = [
    "Milstein",
    "MilsteinConfig",
]


class MilsteinConfig(BaseModel):
    """Configuration for Milstein integrator."""

    # No specific configuration needed for standard Milstein
    pass


class Milstein(Integrator):
    """Milstein integrator (commutative-noise variant).

    Requires a diffusion Jacobian ``model.diffusion_jacobian`` with shape
    compatible with ``diffusion``. If unavailable or incompatible, falls back
    to Euler–Maruyama behavior (no correction term).

    Examples
    --------
    >>> # 典型用法：直接 import 并实例化
    >>> from qphase_sde.integrators.milstein import Milstein
    >>> integrator = Milstein()
    >>> # integrator.step(y, t, dt, model, dW, backend)

    Attributes
    ----------
    name : str
        Unique identifier for this integrator.
    description : str
        Human-readable description of this integrator.
    config_schema : type
        Configuration schema for this integrator.

    """

    name: ClassVar[str] = "milstein"
    description: ClassVar[str] = (
        "Milstein integrator (commutative-noise variant). "
        "Strong order 1.0 scheme for Itô SDEs."
    )
    config_schema: ClassVar[type[MilsteinConfig]] = MilsteinConfig

    def __init__(self, config: MilsteinConfig | None = None, **kwargs) -> None:
        self.config = config or MilsteinConfig(**kwargs)

    def step(
        self, y: Any, t: float, dt: float, model: Any, noise: Any, backend: Backend
    ) -> Any:
        """Compute one-step increment ``dy`` using the Milstein scheme.

        The update follows the Milstein rule for commutative noise:
        ``dy = a(y,t)·dt + L(y,t) @ dW + 0.5 * G(y,t) * (dW**2 - dt)``,
        where ``a`` is the drift, ``L`` the diffusion matrix, and ``G`` the
        Milstein correction.
        If the model declares a complex noise basis
        (``noise_basis == 'complex'``),
        the diffusion is expanded to a real basis and the Milstein correction
        is skipped.

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
        >>> # dy = milstein.step(y, t, dt, model, dW, backend)  # doctest: +SKIP

        """
        from qphase_sde import ops

        dW = noise
        a = model.drift(y, t, model.params)  # (T, N)
        L = model.diffusion(y, t, model.params)  # (T, N, M_b)

        # Expand complex-basis diffusion if declared; Jacobian handling for complex
        # basis is not implemented here, so we will skip Milstein correction when
        # noise basis is complex.
        noise_basis = getattr(model, "noise_basis", "real")
        if str(noise_basis) == "complex":
            L_eff = ops.expand_complex_noise(L, backend)
            # EM increment only (no Milstein correction under complex basis)
            return a * dt + ops.contract_noise(L_eff, dW, backend)

        # EM part (real basis)
        dy = a * dt + ops.contract_noise(L, dW, backend)

        # Milstein correction requires diffusion Jacobian.
        jac = getattr(model, "diffusion_jacobian", None)
        if jac is None:
            return dy

        try:
            J = jac(y, t, model.params)  # expected shape (T, N, M_b, N): ∂L_{i,k}/∂y_j
            # xi_k = dW_k^2 - dt, shape (T, M_b)
            xi = dW * dW - dt
            # G_{i,k} = sum_j L_{j,k} * J_{i,k,j} -> einsum('tjk,tikj->tik')
            G = backend.einsum("tjk,tikj->tik", L, J)
            corr = 0.5 * backend.einsum("tik,tk->ti", G, xi)
            return dy + corr
        except Exception:
            # Shape/capability mismatch — fall back to EM increment
            return dy

    def supports_adaptive_step(self) -> bool:
        return False

    def reset(self) -> None:
        """Reset internal state (no-op for Milstein)."""
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
        """Adaptive stepping not supported by Milstein."""
        raise NotImplementedError("Milstein does not support adaptive stepping")

    def supports_strided_state(self) -> bool:
        """Strided state not supported by Milstein."""
        return False
