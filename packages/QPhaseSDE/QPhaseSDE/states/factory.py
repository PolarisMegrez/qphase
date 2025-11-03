"""
QPhaseSDE: State Factory
------------------------
Select concrete State and TrajectorySet implementations based on the active
backend identity and capabilities, and provide convenience constructors.

Behavior
--------
- Resolve container classes by backend name with capability-based fallback.
    Construction helpers delegate specifics to the selected implementations.

Notes
-----
- If a requested backend implementation is unavailable, fall back using
    capability hints, ultimately defaulting to NumPy.
"""

from typing import Any, Tuple, Type

__all__ = [
    "get_state_classes",
    "make_state",
    "make_trajectory_set",
]

from ..core.protocols import BackendBase as Backend
from .numpy_state import State as NpState, TrajectorySet as NpTrajectorySet
try:
    from .torch_state import State as TorchState, TrajectorySet as TorchTrajectorySet  # type: ignore
except Exception:
    TorchState = None  # type: ignore
    TorchTrajectorySet = None  # type: ignore
try:
    from .cupy_state import State as CuPyState, TrajectorySet as CuPyTrajectorySet  # type: ignore
except Exception:
    CuPyState = None  # type: ignore
    CuPyTrajectorySet = None  # type: ignore


def get_state_classes(backend: Backend) -> Tuple[Type, Type]:
    """
    Select concrete State and TrajectorySet classes based on backend identity and capabilities.

    Prefers explicit backend_name (numpy/torch/cupy) and falls back to capability hints.
    Returns a tuple of (StateClass, TrajectorySetClass).

    Parameters
    ----------
    backend : BackendBase
        Backend instance providing array operations and metadata.

    Returns
    -------
    tuple of (Type, Type)
        Selected State and TrajectorySet classes for the backend.

    Examples
    --------
    >>> from QPhaseSDE.states.factory import get_state_classes
    >>> StateCls, TSCls = get_state_classes(numpy_backend)
    >>> s = StateCls(y, t)
    >>> ts = TSCls(data, t0, dt)
    """
    # Prefer explicit backend_name
    try:
        name = backend.backend_name().lower()
    except Exception:
        name = str(getattr(backend, 'backend_name', lambda: 'numpy')()).lower()

    if name in ("numpy", "np"):
        return (NpState, NpTrajectorySet)
    if name in ("torch", "pt") and (TorchState is not None):
        return (TorchState, TorchTrajectorySet)  # type: ignore
    if name in ("cupy", "cp") and (CuPyState is not None):
        return (CuPyState, CuPyTrajectorySet)  # type: ignore

    # Fallback by capabilities for resilience
    try:
        caps = backend.capabilities()
    except Exception:
        caps = {}
    if bool(caps.get('cupy', False)) and (CuPyState is not None):
        return (CuPyState, CuPyTrajectorySet)  # type: ignore
    if bool(caps.get('torch', False)) and (TorchState is not None):
        return (TorchState, TorchTrajectorySet)  # type: ignore

    # Default to numpy implementation
    return (NpState, NpTrajectorySet)


def make_state(backend: Backend, y: Any, t: float, attrs: dict):
    """
    Construct a state container with the selected implementation for the backend.

    Parameters
    ----------
    backend : BackendBase
        Backend instance.
    y : array-like
        State data.
    t : float
        Time associated with the state.
    attrs : dict
        Additional metadata.

    Returns
    -------
    State
        State instance for the selected backend.

    Examples
    --------
    >>> from QPhaseSDE.states.factory import make_state
    >>> s = make_state(numpy_backend, y, t, attrs)
    """
    StateCls, _ = get_state_classes(backend)
    return StateCls(y=y, t=t, attrs=attrs, backend=backend)


def make_trajectory_set(backend: Backend, data: Any, t0: float, dt: float, meta: dict):
    """
    Construct a trajectory set container with the selected implementation for the backend.

    Parameters
    ----------
    backend : BackendBase
        Backend instance.
    data : array-like
        Trajectory data.
    t0 : float
        Initial time.
    dt : float
        Time step.
    meta : dict
        Additional metadata.

    Returns
    -------
    TrajectorySet
        TrajectorySet instance for the selected backend.

    Examples
    --------
    >>> from QPhaseSDE.states.factory import make_trajectory_set
    >>> ts = make_trajectory_set(numpy_backend, data, t0, dt, meta)
    """
    _, TSCls = get_state_classes(backend)
    return TSCls(data=data, t0=t0, dt=dt, meta=meta)
