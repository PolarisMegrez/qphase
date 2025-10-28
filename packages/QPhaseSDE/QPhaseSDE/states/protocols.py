from __future__ import annotations

"""States domain protocol extensions.

Extends core StateBase with higher-level operations required by engines and
visualizers, while remaining backend-agnostic.
"""

from typing import Any, Mapping, Optional, Protocol

from ..core.protocols import BackendBase, StateBase


class ExtendedState(StateBase, Protocol):
    """Extended operations for states.

    Implementers should use backend primitives; avoid direct NumPy usage.
    """

    def slice(self, time_index: Optional[int] = None, *, trajectories: Optional[Any] = None, modes: Optional[Any] = None) -> "ExtendedState": ...
    def view_complex_as_real(self) -> Any: ...
    def persist(self, path: str) -> None: ...
    def to_numpy_if_supported(self) -> Any: ...
