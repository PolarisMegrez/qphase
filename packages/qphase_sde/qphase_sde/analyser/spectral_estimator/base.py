"""Protocols and result types shared by PSD spectral estimators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable


@dataclass(frozen=True)
class PsdEstimate:
    """PSD mean and cross-trajectory uncertainty for one mode."""

    axis: Any
    mean: Any
    std: Any
    sem: Any
    n_independent: int


@dataclass(frozen=True)
class SpectralEstimatorCapabilities:
    """Execution properties consumed by the SDE memory planner."""

    trajectory_batching: bool = True
    time_streaming: bool = False
    requires_full_trajectory: bool = True
    backend_native: bool = False


@runtime_checkable
class SpectralEstimator(Protocol):
    """Contract implemented by PSD estimator child plugins."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[Any]]

    def capabilities(self) -> SpectralEstimatorCapabilities: ...

    def estimate(
        self,
        x: Any,
        dt: float,
        convention: str,
        backend: Any,
    ) -> PsdEstimate: ...
