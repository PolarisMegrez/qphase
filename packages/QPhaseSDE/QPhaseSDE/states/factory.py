from __future__ import annotations

"""State factory that returns concrete implementations based on backend."""

from typing import Any, Tuple, Type

from ..core.protocols import BackendBase as Backend
from .numpy_state import State as NpState, TrajectorySet as NpTrajectorySet


def get_state_classes(backend: Backend) -> Tuple[Type, Type]:
    # For now, select by backend name; can be extended for other backends
    name = getattr(backend, 'name', 'numpy').lower()
    if name in ("numpy", "np"):
        return (NpState, NpTrajectorySet)
    # Default fallback
    return (NpState, NpTrajectorySet)


def make_state(backend: Backend, y: Any, t: float, attrs: dict):
    StateCls, _ = get_state_classes(backend)
    return StateCls(y=y, t=t, attrs=attrs, backend=backend)


def make_trajectory_set(backend: Backend, data: Any, t0: float, dt: float, meta: dict):
    _, TSCls = get_state_classes(backend)
    return TSCls(data=data, t0=t0, dt=dt, meta=meta)
