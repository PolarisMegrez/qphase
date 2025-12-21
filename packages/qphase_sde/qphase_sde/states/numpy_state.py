"""qphase_sde: NumPy State Containers
---------------------------------
NumPy implementations of the State and TrajectorySet protocols for storing
multi-trajectory, complex-valued state vectors and time series.

Behavior
--------
- Provide NumPy-backed containers adhering to core protocols; view/copy and
  conversion semantics are documented on the classes and methods.

Notes
-----
- Callers should respect view semantics when using data_view().

"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "State",
    "TrajectorySet",
]

from qphase_sde.states.base import StateBase as StateLike

from ..core.errors import QPSStateError
from .base import TrajectorySetBase as TrajectorySetLike


def complex_to_real(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y.real, y.imag], axis=-1)


def real_to_complex(x: np.ndarray) -> np.ndarray:
    n2 = x.shape[-1]
    assert n2 % 2 == 0, "real_to_complex expects even last dimension"
    n = n2 // 2
    return x[..., :n] + 1j * x[..., n:]


@dataclass
class State(StateLike):
    """NumPy-backed state container for multi-trajectory complex states.

    Ensures a 2D shape ``(n_traj, n_modes)`` and complex dtype. Slicing
    returns views when possible. Conversion to real representation and
    backend migration are supported.

    Parameters
    ----------
    y : np.ndarray
        State array of shape (n_traj, n_modes), complex dtype.
    t : float
        Time associated with the state.
    attrs : dict, optional
        Additional metadata for the state.
    backend : Any, optional
        Backend instance (usually NumPy).

    Attributes
    ----------
    y : np.ndarray
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
    as_real
    from_real
    select_modes
    slice_trajectories
    to_backend

    Examples
    --------
    >>> import numpy as np
    >>> from qphase_sde.states.numpy_state import State
    >>> s = State(y=np.ones((2, 3), dtype=np.complex128), t=0.0)
    >>> s.n_traj
    2
    >>> s.n_modes
    3

    """

    y: np.ndarray
    t: float
    attrs: dict = field(default_factory=dict)
    backend: Any | None = None

    def __post_init__(self):
        """Ensure state array is 2D and complex dtype."""
        if self.y.ndim == 1:
            self.y = self.y[None, ...]
        if not np.iscomplexobj(self.y):
            self.y = self.y.astype(np.complex128)

    @property
    def n_traj(self) -> int:
        """Number of trajectories."""
        return self.y.shape[0]

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return self.y.shape[1]

    def data_view(self) -> np.ndarray:
        """Return the underlying NumPy array view."""
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
        # Views in NumPy are non-copying when slicing
        return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def copy(self) -> "State":
        """Return a deep copy of the state."""
        return State(
            y=self.y.copy(), t=self.t, attrs=self.attrs.copy(), backend=self.backend
        )

    def as_real(self) -> np.ndarray:
        """Convert the complex state to real representation."""
        return complex_to_real(self.y)

    @classmethod
    def from_real(cls, x: np.ndarray, t: float, attrs: dict | None = None) -> "State":
        """Construct a State from real-valued representation.

        Parameters
        ----------
        x : np.ndarray
            Real-valued array (last dimension must be even).
        t : float
            Time associated with the state.
        attrs : dict, optional
            Additional metadata.

        Returns
        -------
        State
            New State instance with complex-valued data.

        """
        return cls(y=real_to_complex(x), t=t, attrs=attrs or {})

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

    def to_backend(
        self, target_backend: Any, *, copy_if_needed: bool = True
    ) -> "State":
        """Convert the state to a different backend (NumPy only).

        Parameters
        ----------
        target_backend : Any
            Target backend instance.
        copy_if_needed : bool, default True
            Whether to deep-copy the array when converting.

        Returns
        -------
        State
            Converted state instance.

        Raises
        ------
        QPSStateError
            - [700] Only NumPy backend is supported in this version.

        Examples
        --------
        >>> s = State(y=np.ones((2, 3), dtype=np.complex128), t=0.0)
        >>> s.to_backend(numpy_backend)

        """
        name = str(getattr(target_backend, "backend_name", lambda: "")()).lower()
        if name in ("numpy", "np", ""):
            # Already numpy; optionally copy
            y = self.y.copy() if copy_if_needed else self.y
            return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=target_backend)
        raise QPSStateError(
            "[700] to_backend: numpy_state only supports numpy target in this version"
        )


@dataclass
class TrajectorySet(TrajectorySetLike):
    """NumPy-backed trajectory set container for multi-trajectory time series.

    Stores arrays of shape (n_traj, n_steps, n_modes). Provides times() for NumPy-based
    time axis extraction.

    Parameters
    ----------
    data : np.ndarray
        Trajectory data of shape (n_traj, n_steps, n_modes).
    t0 : float
        Initial time.
    dt : float
        Time step.
    meta : dict, optional
        Additional metadata for the trajectory set.

    Attributes
    ----------
    data : np.ndarray
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

    Examples
    --------
    >>> import numpy as np
    >>> from qphase_sde.states.numpy_state import TrajectorySet
    >>> ts = TrajectorySet(data=np.ones((2, 10, 3)), t0=0.0, dt=0.1)
    >>> ts.n_traj
    2
    >>> ts.n_steps
    10
    >>> ts.n_modes
    3
    >>> ts.times().shape
    (10,)

    """

    data: np.ndarray
    t0: float
    dt: float
    meta: dict = field(default_factory=dict)

    @property
    def n_traj(self) -> int:
        """Number of trajectories."""
        return self.data.shape[0]

    @property
    def n_steps(self) -> int:
        """Number of time steps."""
        return self.data.shape[1]

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return self.data.shape[2]

    def times(self) -> np.ndarray:
        """Return the time axis as a NumPy array for plotting or analysis."""
        return self.t0 + self.dt * np.arange(self.n_steps)

    @property
    def metadata(self) -> dict:
        """Alias for meta to satisfy ResultProtocol."""
        return self.meta

    def save(self, path: str | Any) -> None:
        """Save to disk (numpy format) to satisfy ResultProtocol."""
        import numpy as np

        p = str(path)
        if not p.endswith(".npz"):
            p += ".npz"
        # meta might contain non-serializable objects, but for now let's try
        np.savez(p, data=self.data, t0=self.t0, dt=self.dt, meta=self.meta)  # type: ignore[arg-type]
