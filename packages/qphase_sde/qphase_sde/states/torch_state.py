"""qphase_sde: Torch State Containers
---------------------------------
Experimental implementations of State and TrajectorySet backed by torch tensors
to keep computations on device when using a torch backend.

Behavior
--------
- Provide torch-backed containers adhering to core protocols; view/slice,
  copy, and backend conversion semantics are documented on the classes.

Notes
-----
- Requires PyTorch; experimental — behavior and performance may change.

"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

__all__ = [
    "State",
    "TrajectorySet",
]
from ..core.errors import QPSBackendError

if TYPE_CHECKING:
    import torch as _torch
else:
    try:
        import torch as _torch
    except Exception:  # pragma: no cover
        _torch = None

from qphase_sde.states.base import StateBase as StateLike

from .base import TrajectorySetBase as TrajectorySetLike


@dataclass
class State(StateLike):
    """Torch-backed quantum state container using torch tensors.

    Ensures 2D shape (n_traj, n_modes) and complex dtype. Supports view/slice operations
    without unnecessary copies. Conversion to NumPy is supported for interoperability.

    Parameters
    ----------
    y : torch.Tensor
        State tensor of shape (n_traj, n_modes), complex dtype.
    t : float
        Time associated with the state.
    attrs : dict, optional
        Additional metadata for the state.
    backend : Any, optional
        Backend instance (usually torch or NumPy).

    Attributes
    ----------
    y : torch.Tensor
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

    Examples
    --------
    >>> import torch
    >>> from qphase_sde.states.torch_state import State
    >>> s = State(y=torch.ones((2, 3), dtype=torch.complex128), t=0.0)
    >>> s.n_traj
    2
    >>> s.n_modes
    3

    """

    y: Any  # torch.Tensor
    t: float
    attrs: dict = field(default_factory=dict)
    backend: Any | None = None

    def __post_init__(self):
        """Ensure torch is available and state tensor is 2D and complex dtype."""
        if _torch is None:
            raise QPSBackendError("[202] torch is required for torch_state.State")
        if self.y.ndim == 1:
            self.y = self.y.unsqueeze(0)
        # Ensure complex dtype
        if self.y.dtype not in (_torch.complex64, _torch.complex128):
            self.y = self.y.to(dtype=_torch.complex128)

    @property
    def n_traj(self) -> int:
        """Number of trajectories."""
        return int(self.y.shape[0])

    @property
    def n_modes(self) -> int:
        """Number of modes."""
        return int(self.y.shape[1])

    def data_view(self) -> Any:
        """Return the underlying torch tensor view."""
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
            y=self.y.clone(), t=self.t, attrs=self.attrs.copy(), backend=self.backend
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
        """Convert the state to a different backend (NumPy or Torch).

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

        Examples
        --------
        >>> s = State(y=torch.ones((2, 3), dtype=torch.complex128), t=0.0)
        >>> s.to_backend(numpy_backend)

        """
        name = str(getattr(target_backend, "backend_name", lambda: "")()).lower()
        if name in ("numpy", "np", ""):
            arr = self.y.detach().cpu().numpy()
            from .numpy_state import State as _NpState

            return _NpState(
                y=arr.copy() if copy_if_needed else arr,
                t=self.t,
                attrs=self.attrs.copy(),
                backend=target_backend,
            )
        # Otherwise, already on torch backend
        return State(
            y=self.y.clone() if copy_if_needed else self.y,
            t=self.t,
            attrs=self.attrs.copy(),
            backend=target_backend,
        )


@dataclass
class TrajectorySet(TrajectorySetLike):
    """Torch-backed trajectory set container using torch tensors.

    Stores (n_traj, n_steps, n_modes) simulation results on device. Provides times()
    for NumPy-based time axis extraction.

    Parameters
    ----------
    data : torch.Tensor
        Trajectory data of shape (n_traj, n_steps, n_modes).
    t0 : float
        Initial time.
    dt : float
        Time step.
    meta : dict, optional
        Additional metadata for the trajectory set.

    Attributes
    ----------
    data : torch.Tensor
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
    >>> import torch
    >>> from qphase_sde.states.torch_state import TrajectorySet
    >>> ts = TrajectorySet(data=torch.ones((2, 10, 3)), t0=0.0, dt=0.1)
    >>> ts.n_traj
    2
    >>> ts.n_steps
    10
    >>> ts.n_modes
    3
    >>> ts.times().shape
    (10,)

    """

    data: Any  # torch.Tensor (n_traj, n_steps, n_modes)
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
