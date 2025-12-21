"""qphase_sde: State Base Protocols
---------------------------------

Minimal contracts for state containers and trajectory sets.
These protocols define the core storage interface for simulation data.

This module is dependency-light and safe to import in any environment.
"""

from typing import Any, Protocol, runtime_checkable

from qphase_sde.core.protocols import SDEBackend

__all__ = [
    "StateBase",
    "ExtendedState",
    "TrajectorySetBase",
]


@runtime_checkable
class StateBase(Protocol):
    """Minimal state container for trajectories at a single time step.

    Encapsulates backend-defined storage of trajectory values and associated
    metadata. Callers must not assume specific array types.

    Attributes
    ----------
    y : Any
        Backend array of shape ``(n_traj, n_modes)``, complex-like.
    t : float
        Time stamp of the state.
    attrs : dict
        Lightweight metadata associated with the state.

    Methods
    -------
    n_traj -> int
        Number of trajectories.
    n_modes -> int
        Number of modes (state dimension).
    data_view() -> Any
        Return a non-copying view/alias of the underlying array when possible.
    view(modes=None, trajectories=None) -> StateBase
        Lightweight slicing/view into modes and/or trajectories.
    copy() -> StateBase
        Deep copy safe for mutation.
    select_modes(idx) -> StateBase
        Return a state with selected mode indices.
    slice_trajectories(idx) -> StateBase
        Return a state with selected trajectory indices.
    to_backend(target_backend, copy_if_needed=True) -> StateBase
        Migrate storage to a target backend when supported.

    """

    y: Any
    t: float
    attrs: dict[str, Any]

    # Shapes
    @property
    def n_traj(self) -> int: ...

    @property
    def n_modes(self) -> int: ...

    # Views and copies
    def data_view(self) -> Any:
        """Return a non-copying view/alias of underlying array when possible."""
        ...

    def view(
        self, *, modes: Any | None = None, trajectories: Any | None = None
    ) -> "StateBase": ...
    def copy(self) -> "StateBase": ...

    # Domain operations
    def select_modes(self, idx: Any) -> "StateBase": ...
    def slice_trajectories(self, idx: Any) -> "StateBase": ...

    # Backend migration
    def to_backend(
        self, target_backend: SDEBackend, *, copy_if_needed: bool = True
    ) -> "StateBase": ...


class ExtendedState(StateBase, Protocol):
    """Protocol for extended state operations.

    Supports slicing, conversion, and persistence. Provides convenience
    helpers for analysis and the visualizer while remaining backend-agnostic.

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

    def slice(
        self,
        time_index: int | None = None,
        *,
        trajectories: Any | None = None,
        modes: Any | None = None
    ) -> "ExtendedState":
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


@runtime_checkable
class TrajectorySetBase(Protocol):
    """Minimal time-series container for multiple trajectories.

    Represents a sampled trajectory set produced by the engine. Storage is
    backend-defined and must be treated as opaque by callers.

    Attributes
    ----------
    data : Any
        Backend array shaped ``(n_traj, n_steps, n_modes)``.
    t0 : float
        Initial time.
    dt : float
        Time step between consecutive samples.
    meta : dict
        Lightweight metadata (e.g., backend info, solver tags).

    Methods
    -------
    n_traj -> int
        Number of trajectories.
    n_steps -> int
        Number of stored steps.
    n_modes -> int
        State dimension per trajectory.
    times() -> Any
        Backend array of sample times derived from ``t0`` and ``dt``.

    """

    data: Any
    t0: float
    dt: float
    meta: dict[str, Any]

    @property
    def n_traj(self) -> int: ...

    @property
    def n_steps(self) -> int: ...

    @property
    def n_modes(self) -> int: ...

    def times(self) -> Any: ...
