from __future__ import annotations

"""Integrator domain protocol.

Defines the Integrator step interface and optional capabilities.
"""

from typing import Any, Protocol

from ..core.protocols import BackendBase


class Integrator(Protocol):
    """Single-step SDE integrator.

    step(y, t, dt, model, noise, backend) -> dy
    - y: backend array (n_traj, n_modes)
    - t: float
    - dt: float
    - model: object providing drift/diffusion evaluated on y
    - noise: noise increment or model-dependent auxiliary
    - backend: BackendBase

    Contract:
    - MUST NOT mutate input y.
    - MUST return dy with same shape as y.
    - Adaptive/advanced schemes may expose extra kwargs; callers should feature-detect.
    """

    def step(self, y: Any, t: float, dt: float, model: Any, noise: Any, backend: BackendBase) -> Any: ...

    # Optional capabilities
    def reset(self) -> None: ...
    def supports_adaptive_step(self) -> bool: ...
    def supports_strided_state(self) -> bool: ...
