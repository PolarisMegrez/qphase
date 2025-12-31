"""qphase_sde: Integrator Base Protocols
---------------------------------------------------------
Minimal contracts for single-step SDE integrators.
These protocols define the core integration interface for
stochastic differential equations.

This module is dependency-light and safe to import in any environment.

Public API
----------
``Integrator`` : Protocol for SDE integrators.
"""

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qphase.backend.base import BackendBase

    from qphase_sde.model import NoiseSpec, SDEModel

__all__ = [
    "Integrator",
]


@runtime_checkable
class Integrator(Protocol):
    """Protocol for single-step SDE integrators returning an increment ``dy``.

    Implementations advance the solution by producing an increment ``dy`` such
    that ``y_next = y + dy``. Integrators MUST NOT mutate the input ``y`` and
    MUST return an array with the same shape as ``y``. Advanced schemes may
    optionally expose capabilities like adaptive stepping or strided-state
    handling, which callers should feature-detect at runtime.

    Methods
    -------
    step(y, t, dt, model, noise, backend)
      Compute one-step increment ``dy`` without mutating inputs.

    Parameters
    ----------
        y : Any
          State array with shape ``(n_traj, n_modes)``.
        t : float
          Current simulation time.
        dt : float
          Time step size (positive).
        model : SDEModel
          SDE model providing drift and diffusion functions.
        noise : NoiseSpec
          Noise specification for the integration.
        backend : BackendBase
          Active backend providing array operations.

    Returns
    -------
        Any
          Increment ``dy`` with the same shape as ``y``.

    reset()
      Optional. Reset any internal caches or stateful buffers.

    supports_adaptive_step()
      Optional. Return True if the integrator can adapt ``dt`` internally.

    supports_strided_state()
      Optional. Return True if the integrator supports strided or partitioned
      state layouts.

    Examples
    --------
    >>> from qphase_sde.integrator import EulerMaruyama
    >>> integrator = EulerMaruyama()
    >>> dy = integrator.step(y, t, dt, model, noise, backend)

    """

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        model: "SDEModel",
        noise: Any,
        backend: "BackendBase",
    ) -> Any:
        """Compute one-step increment ``dy`` without mutating inputs.

        Parameters
        ----------
        y : Any
            State array with shape ``(n_traj, n_modes)``.
        t : float
            Current simulation time.
        dt : float
            Time step size (positive).
        model : SDEModel
            SDE model providing drift and diffusion.
        noise : Any
            Noise specification or increment (dW) for the integration.
        backend : BackendBase
            Active backend providing array operations.

        Returns
        -------
        Any
            Increment ``dy`` with the same shape as ``y``.

        """
        ...

    def step_adaptive(
        self,
        y: Any,
        t: float,
        dt: float,
        tol: float,
        model: "SDEModel",
        noise: "NoiseSpec",
        backend: "BackendBase",
        rng: Any = None,
    ) -> tuple[Any, float, float, float]:
        """Perform one adaptive step (optional).

        Parameters
        ----------
        y : Any
            State array.
        t : float
            Current time.
        dt : float
            Proposed step size.
        tol : float
            Error tolerance.
        model : SDEModel
            Model definition.
        noise : NoiseSpec
            Noise specification.
        backend : BackendBase
            Backend for array ops.
        rng : Any, optional
            Random number generator state/object.

        Returns
        -------
        tuple[Any, float, float, float]
            (y_next, t_next, dt_next, error)

        """
        ...

    # Optional capabilities
    def reset(self) -> None:
        """Reset any internal caches or stateful buffers (optional)."""
        ...

    def supports_adaptive_step(self) -> bool:
        """Return True if the integrator supports adaptive ``dt`` (optional)."""
        ...

    def supports_strided_state(self) -> bool:
        """Return True if strided/partitioned state layouts are supported (optional)."""
        ...
