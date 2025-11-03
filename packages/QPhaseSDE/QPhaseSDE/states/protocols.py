"""
QPhaseSDE: State Protocols
--------------------------
Contracts extending StateBase with optional higher-level conveniences used by
downstream analysis and visualization. Implementations must remain backend-
agnostic and delegate array logic to BackendBase.

Notes
-----
- Views may alias memory. Conversions like `to_numpy_if_supported()` are
  best-effort and should be guarded by callers.
"""

from typing import Any, Mapping, Optional, Protocol

__all__ = [
  "ExtendedState",
]

from ..core.protocols import BackendBase, StateBase


class ExtendedState(StateBase, Protocol):
  """
  Protocol for extended state operations supporting slicing, conversion, and persistence.

  Provides higher-level conveniences for downstream analysis and visualization, while remaining backend-agnostic.

  Parameters
  ----------
  Inherits all parameters from StateBase.

  Attributes
  ----------
  Inherits all attributes from StateBase.

  Methods
  -------
  slice
  view_complex_as_real
  persist
  to_numpy_if_supported

  Examples
  --------
  >>> class MyState(ExtendedState):
  ...     def slice(self, ...): ...
  ...     def view_complex_as_real(self): ...
  ...     def persist(self, path): ...
  ...     def to_numpy_if_supported(self): ...
  """

  def slice(self, time_index: Optional[int] = None, *, trajectories: Optional[Any] = None, modes: Optional[Any] = None) -> "ExtendedState":
    """Return a sliced view of the state along time, trajectories, or modes."""
    ...

  def view_complex_as_real(self) -> Any:
    """Return a real-valued view of the complex state data."""
    ...

  def persist(self, path: str) -> None:
    """Persist the state to disk at the given path."""
    ...

  def to_numpy_if_supported(self) -> Any:
    """Convert the state to a NumPy array if supported by the backend."""
    ...
