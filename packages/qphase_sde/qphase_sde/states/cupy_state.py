"""qphase_sde: CuPy State Containers
--------------------------------
GPU-backed implementations of State and TrajectorySet using CuPy arrays.
Intended for experimental acceleration on NVIDIA GPUs.

Behavior
--------
- Provide CuPy-backed containers adhering to core protocols; slicing/view,
  copy, and backend conversion semantics are documented on the classes.

Notes
-----
- Requires CuPy; experimental status — APIs may evolve.

"""

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "State",
    "TrajectorySet",
]

try:
    import cupy as cp
except Exception as _e:  # pragma: no cover
    cp = None

from qphase_sde.states.base import StateBase as StateLike

from ..core.errors import QPSStateError
from .base import TrajectorySetBase as TrajectorySetLike


@dataclass
class State(StateLike):
    """GPU-backed quantum state container using CuPy arrays.

    Ensures 2D shape (n_traj, n_modes) and complex dtype. Supports view/slice operations
    without unnecessary copies. Conversion to NumPy is supported for interoperability.

    Parameters
    ----------
    y : cp.ndarray
        State array of shape (n_traj, n_modes), complex dtype.
    t : float
        Time associated with the state.
    attrs : dict, optional
        Additional metadata for the state.
    backend : Any, optional
        Backend instance (usually CuPy or NumPy).

    Attributes
    ----------
    y : cp.ndarray
        State data.
    t : float
        Time.
    attrs : dict
        Metadata.
    backend : Any
        Backend instance.

    Methods
    -------
    n_traj
    n_modes
    data_view
    view
    copy
    select_modes
    slice_trajectories
    to_backend
    to_numpy

    Examples
    --------
    >>> import cupy as cp
    >>> from qphase_sde.states.cupy_state import State
    >>> s = State(y=cp.ones((2, 3), dtype=cp.complex128), t=0.0)
    >>> s.n_traj
    2
    >>> s.n_modes
    3
    >>> s.to_numpy().shape
    (2, 3)

    """

    y: Any  # cp.ndarray
    t: float
    attrs: dict = field(default_factory=dict)
    backend: Any | None = None

    def __post_init__(self):
        """Ensure CuPy is available and state array is 2D and complex dtype."""
        if cp is None:
            raise QPSStateError("cupy is required for cupy_state.State")
        if getattr(self.y, "ndim", 1) == 1:
            self.y = self.y[None, ...]
        # Ensure complex dtype
        if self.y.dtype not in (cp.complex64, cp.complex128):
            self.y = self.y.astype(cp.complex128)

    @property
    def n_traj(self) -> int:
        """Number of trajectories."""
        return int(self.y.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return int(self.y.shape[1])

    def data_view(self) -> Any:
        """Return the underlying CuPy array view."""
        return self.y

    def view(
        self, *, modes: Any | None = None, trajectories: Any | None = None
    ) -> "State":
        """Return a view of the state with selected modes and/or trajectories.

        Parameters
        ----------
        modes : array-like, optional
            Indices of modes to select.
        trajectories : array-like, optional
            Indices of trajectories to select.

        Returns
        -------
        State
            New State instance with selected data.

        """
        y = self.y
        if trajectories is not None:
            y = y[trajectories, :]
        if modes is not None:
            y = y[:, modes]
        return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def copy(self) -> "State":
        """Return a deep copy of the state."""
        return State(
            y=self.y.copy(), t=self.t, attrs=self.attrs.copy(), backend=self.backend
        )

    def select_modes(self, idx) -> "State":
        """Return a new state with selected modes."""
        return State(
            y=self.y[:, idx], t=self.t, attrs=self.attrs.copy(), backend=self.backend
        )

    def slice_trajectories(self, idx) -> "State":
        """Return a new state with selected trajectories."""
        return State(
            y=self.y[idx, :], t=self.t, attrs=self.attrs.copy(), backend=self.backend
        )

    def to_backend(self, target_backend: Any, *, copy_if_needed: bool = True) -> Any:
        """Convert the state to a different backend (NumPy, CuPy, or custom).

        Parameters
        ----------
        target_backend : Any
            Target backend instance.
        copy_if_needed : bool, default True
            Whether to deep-copy the array when converting.

        Returns
        -------
        State or Any
            Converted state instance or backend-specific object.

        Raises
        ------
        QPSStateError
            - [704] Unsupported target backend or conversion failed.

        Examples
        --------
        >>> s = State(y=cp.ones((2, 3), dtype=cp.complex128), t=0.0)
        >>> s.to_backend(numpy_backend)

        """
        name = None
        try:
            name = str(target_backend.backend_name()).lower()
        except Exception:
            name = str(getattr(target_backend, "backend_name", lambda: "")()).lower()
        if name in ("numpy", "np"):  # to numpy
            import numpy as _np

            arr = cp.asnumpy(self.y)
            if copy_if_needed:
                arr = _np.array(arr, copy=True)
            from .numpy_state import State as _NpState

            return _NpState(
                y=arr, t=self.t, attrs=self.attrs.copy(), backend=target_backend
            )
        if name in ("cupy", "cp", ""):
            y = self.y.copy() if copy_if_needed else self.y
            return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=target_backend)
        # Fallback: convert via numpy then let target handle
        import numpy as _np

        arr = cp.asnumpy(self.y)
        if copy_if_needed:
            arr = _np.array(arr, copy=True)
        try:
            return target_backend.asarray(arr)
        except Exception as e:
            raise QPSStateError(
                f"[704] to_backend: unsupported target backend '{name}': {e}"
            ) from e

    # Convenience
    def to_numpy(self):
        """Convert the state to a NumPy array."""
        import numpy as _np

        return _np.asarray(cp.asnumpy(self.y))


@dataclass
class TrajectorySet(TrajectorySetLike):
    """GPU-backed trajectory set container using CuPy arrays.

    Stores (n_traj, n_steps, n_modes) simulation results on GPU. Provides times()
    for NumPy-based time axis extraction and to_numpy() for conversion.

    Parameters
    ----------
    data : cp.ndarray
        Trajectory data of shape (n_traj, n_steps, n_modes).
    t0 : float
        Initial time.
    dt : float
        Time step.
    meta : dict, optional
        Additional metadata for the trajectory set.

    Attributes
    ----------
    data : cp.ndarray
        Trajectory data.
    t0 : float
        Initial time.
    dt : float
        Time step.
    meta : dict
        Metadata.

    Methods
    -------
    n_traj
    n_steps
    n_modes
    times
    to_numpy

    Examples
    --------
    >>> import cupy as cp
    >>> from qphase_sde.states.cupy_state import TrajectorySet
    >>> ts = TrajectorySet(data=cp.ones((2, 10, 3)), t0=0.0, dt=0.1)
    >>> ts.n_traj
    2
    >>> ts.n_steps
    10
    >>> ts.n_modes
    3
    >>> ts.times().shape
    (10,)

    """

    data: Any  # cp.ndarray of shape (n_traj, n_steps, n_modes)
    t0: float
    dt: float
    meta: dict = field(default_factory=dict)

    @property
    def n_traj(self) -> int:
        """Number of trajectories."""
        return int(self.data.shape[0])

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return int(self.data.shape[1])

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return int(self.data.shape[2])

    def times(self) -> Any:
        """Return the time axis as a NumPy array for plotting or analysis."""
        import numpy as _np

        return self.t0 + self.dt * _np.arange(self.n_steps)

    def to_numpy(self):
        """Convert the trajectory set to a NumPy array."""
        import numpy as _np

        return _np.asarray(cp.asnumpy(self.data))
