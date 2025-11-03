"""
QPhaseSDE: Noise Model Protocols
--------------------------------
Interfaces for noise specifications and models that produce per-step noise
increments for the engine. Implementations are backend-agnostic and interact
only via `BackendBase`.

Behavior
--------
- Deterministic seeding is handled by the engine-provided `RNG` object.

Notes
-----
- The shape and dtype of noise tensors are backend-defined but must remain
  consistent across steps within a run.
"""

from typing import Any, Protocol
from ..core.protocols import BackendBase

__all__ = [
  "NoiseSpecLike",
  "NoiseModel",
]

class NoiseSpecLike(Protocol):
  """Protocol for noise specification objects.

  Attributes
  ----------
  kind : str
    Noise type ('independent' or 'correlated').
  dim : int
    Number of noise channels.
  covariance : Any or None
    Covariance matrix for correlated noise, or None for independent noise.
  """
  kind: str  # 'independent' | 'correlated'
  dim: int
  covariance: Any | None

class NoiseModel(Protocol):
  """Protocol for noise models producing per-step increments for the engine.

  Implementations must provide a constructor accepting a noise specification
  and backend, and a sample method producing increments for each trajectory.

  Parameters
  ----------
  spec : NoiseSpecLike
    Noise specification object.
  backend : BackendBase
    Backend instance for array and RNG operations.

  Methods
  -------
  sample(rng, n_traj, dt)
    Produce per-step noise increments for the engine.

  Examples
  --------
  >>> model = NoiseModel(spec, backend)  # doctest: +SKIP
  >>> dW = model.sample(rng, n_traj=128, dt=0.01)  # doctest: +SKIP
  """
  def __init__(self, spec: NoiseSpecLike, backend: BackendBase) -> None:
    """Initialize the noise model with specification and backend."""
    ...

  def sample(self, rng: Any, n_traj: int, dt: float) -> Any:
    """Produce per-step noise increments for the engine."""
    ...
