from __future__ import annotations

"""NumPy implementation of State and TrajectorySet protocols."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np

from ..core.protocols import StateBase as StateLike, TrajectorySetBase as TrajectorySetLike
from ..core.errors import StateError


def complex_to_real(y: np.ndarray) -> np.ndarray:
    return np.concatenate([y.real, y.imag], axis=-1)


def real_to_complex(x: np.ndarray) -> np.ndarray:
    n2 = x.shape[-1]
    assert n2 % 2 == 0, "real_to_complex expects even last dimension"
    n = n2 // 2
    return x[..., :n] + 1j * x[..., n:]


@dataclass
class State(StateLike):
    y: np.ndarray
    t: float
    attrs: Dict = field(default_factory=dict)
    backend: Any | None = None

    def __post_init__(self):
        if self.y.ndim == 1:
            self.y = self.y[None, ...]
        if not np.iscomplexobj(self.y):
            self.y = self.y.astype(np.complex128)

    @property
    def n_traj(self) -> int:
        return self.y.shape[0]

    @property
    def n_modes(self) -> int:
        return self.y.shape[1]

    def data_view(self) -> np.ndarray:
        # For NumPy, returning the array itself is a view/alias semantics
        return self.y

    def view(self, *, modes: Optional[Any] = None, trajectories: Optional[Any] = None) -> "State":
        y = self.y
        if trajectories is not None:
            y = y[trajectories, :]
        if modes is not None:
            y = y[:, modes]
        # Views in NumPy are non-copying when slicing
        return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def copy(self) -> "State":
        return State(y=self.y.copy(), t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def as_real(self) -> np.ndarray:
        return complex_to_real(self.y)

    @classmethod
    def from_real(cls, x: np.ndarray, t: float, attrs: Optional[Dict] = None) -> "State":
        return cls(y=real_to_complex(x), t=t, attrs=attrs or {})

    def select_modes(self, idx) -> "State":
        return State(y=self.y[:, idx], t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def slice_trajectories(self, idx) -> "State":
        return State(y=self.y[idx, :], t=self.t, attrs=self.attrs.copy(), backend=self.backend)

    def to_backend(self, target_backend: Any, *, copy_if_needed: bool = True) -> "State":
        name = getattr(target_backend, 'backend_name', lambda: getattr(target_backend, 'name', ''))()
        if str(name).lower() in ("numpy", "np", ""):
            # Already numpy; optionally copy
            y = self.y.copy() if copy_if_needed else self.y
            return State(y=y, t=self.t, attrs=self.attrs.copy(), backend=target_backend)
        raise StateError("[700] to_backend: numpy_state only supports numpy target in this version")


@dataclass
class TrajectorySet(TrajectorySetLike):
    data: np.ndarray
    t0: float
    dt: float
    meta: Dict = field(default_factory=dict)

    @property
    def n_traj(self) -> int:
        return self.data.shape[0]

    @property
    def n_steps(self) -> int:
        return self.data.shape[1]

    @property
    def n_modes(self) -> int:
        return self.data.shape[2]

    def times(self) -> np.ndarray:
        return self.t0 + self.dt * np.arange(self.n_steps)

