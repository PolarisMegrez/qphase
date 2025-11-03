"""
QPhaseSDE: Integrator Protocols
-------------------------------
Contracts for single-step SDE integrators consumed by the simulation engine.
Implementations provide step(...) with fixed semantics and may optionally
expose capabilities for adaptive stepping or strided state handling.

Behavior
--------
- Integrators must not mutate input arrays and must return outputs with the
  same shape as the inputs.

Notes
-----
- Optional behaviors should be feature-detected at runtime.
"""

from typing import Any, Protocol
from ..core.protocols import BackendBase

__all__ = [
	"Integrator",
]

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
      y : Any
        State array with shape ``(n_traj, n_modes)``.
      t : float
        Current simulation time.
      dt : float
        Time step size (positive).
      model : Any
        Object exposing at least ``drift(y, t, params)`` and
        ``diffusion(y, t, params)`` evaluated on ``y``.
      noise : Any
        Noise increment for the step (shape and dtype depend on the
        model/noise basis; engine provides a real-valued increment).
      backend : BackendBase
        Active backend providing array ops and contractions.
    Returns
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
  >>> # Resolve an integrator via the registry (names may alias to 'euler')
  >>> from QPhaseSDE.core.registry import registry
  >>> em_cls = registry.create("integrator:euler_maruyama")  # doctest: +SKIP
  >>> em = em_cls()  # doctest: +SKIP
  >>> # dy = em.step(y, t, dt, model, dW, backend)  # doctest: +SKIP
  """

  def step(self, y: Any, t: float, dt: float, model: Any, noise: Any, backend: BackendBase) -> Any:
    """Compute one-step increment ``dy`` without mutating inputs.

    Parameters
    ----------
    y : Any
      State array with shape ``(n_traj, n_modes)``.
    t : float
      Current simulation time.
    dt : float
      Time step size (positive).
    model : Any
      Object providing at least ``drift(y, t, params)`` and
      ``diffusion(y, t, params)`` evaluated on ``y``.
    noise : Any
      Noise increment for the step (shape and dtype depend on the
      model/noise basis; the engine supplies a real-valued increment).
    backend : BackendBase
      Active backend providing array operations and contractions.

    Returns
    -------
    Any
      Increment ``dy`` with the same shape as ``y``.

    Examples
    --------
    >>> # dy = integrator.step(y, t, dt, model, dW, backend)  # doctest: +SKIP
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
