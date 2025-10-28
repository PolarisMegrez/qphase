from __future__ import annotations

"""Noise models domain protocols."""

from typing import Any, Protocol

from ..core.protocols import BackendBase


class NoiseSpecLike(Protocol):
    kind: str  # 'independent' | 'correlated'
    dim: int
    covariance: Any | None


class NoiseModel(Protocol):
    """Noise model that produces per-step increments for the engine."""

    def __init__(self, spec: NoiseSpecLike, backend: BackendBase) -> None: ...

    def sample(self, rng: Any, n_traj: int, dt: float) -> Any: ...
