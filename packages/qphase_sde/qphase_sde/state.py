"""qphase_sde: State Containers
----------------------------
Unified, backend-agnostic state containers for SDE simulations.
Inherits from qphase.backend.state.ArrayBase.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from qphase.backend.base import ArrayBase

__all__ = ["State", "TrajectorySet"]


@dataclass
class State(ArrayBase):
    """Backend-agnostic quantum state container.

    Attributes
    ----------
    data : Any
        State array of shape (n_traj, n_modes).
    t : float
        Current time.
    meta : dict
        Metadata.

    """

    t: float = 0.0

    def __post_init__(self):
        """Post-initialization hook to ensure proper data shape."""
        # Ensure 2D shape (n_traj, n_modes)
        if hasattr(self.data, "ndim") and self.data.ndim == 1:
            try:
                self.data = self.data[None, ...]
            except Exception:
                pass

    @property
    def n_traj(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_modes(self) -> int:
        return int(self.data.shape[1])

    def view(self, *, modes=None, trajectories=None) -> "State":
        y = self.data
        if trajectories is not None:
            y = y[trajectories, :]
        if modes is not None:
            y = y[:, modes]
        return State(data=y, t=self.t, meta=self.meta.copy())

    # Alias for backward compatibility if needed, but prefer .data
    @property
    def y(self):
        return self.data

    @y.setter
    def y(self, value):
        self.data = value

    # Alias attrs to meta for backward compatibility
    @property
    def attrs(self):
        return self.meta

    @attrs.setter
    def attrs(self, value):
        self.meta = value


@dataclass
class TrajectorySet(ArrayBase):
    """Backend-agnostic trajectory set container.

    Attributes
    ----------
    data : Any
        Trajectory data of shape (n_traj, n_steps, n_modes).
    t0 : float
        Start time.
    dt : float
        Time step.
    meta : dict
        Metadata.

    """

    t0: float = 0.0
    dt: float = 1.0

    @property
    def n_traj(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_steps(self) -> int:
        return int(self.data.shape[1])

    @property
    def n_modes(self) -> int:
        return int(self.data.shape[2])

    def times(self) -> Any:
        """Return the time axis."""
        return self.t0 + self.dt * self.xp.arange(self.n_steps)

    def save(self, path: str | Any) -> None:
        """Save to disk (numpy format) to satisfy ResultProtocol."""
        p = str(path)
        if not p.endswith(".npz"):
            p += ".npz"

        # Always save as numpy for portability
        data_np = self.to_numpy()
        np.savez(p, data=data_np, t0=self.t0, dt=self.dt, meta=np.array(self.meta))
