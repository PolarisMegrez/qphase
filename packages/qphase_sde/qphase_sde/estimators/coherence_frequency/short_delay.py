"""Short-delay first-order coherence frequency child plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field

from ...contracts.coherence import (
    CoherenceFrequencyEstimate,
    CoherenceFrequencyStatus,
)
from ...numerics.short_delay import estimate_coherence_carrier
from .base import BaseCoherenceFrequencyConfig, CoherenceFrequencyProtocol

__all__ = ["ShortDelayCoherenceFrequency", "ShortDelayCoherenceFrequencyConfig"]


class ShortDelayCoherenceFrequencyConfig(BaseCoherenceFrequencyConfig):
    orientation: Literal["phase_decreasing", "phase_increasing"] = "phase_decreasing"
    polynomial_order: int = Field(2, ge=1, le=4)
    minimum_lag_points: int = Field(4, ge=2, le=64)
    maximum_lag_points: int = Field(12, ge=2, le=256)
    consistency_sigma: float = Field(2.0, gt=0.0)
    intensity_floor: float = Field(1.0e-14, ge=0.0)
    confidence_level: float = Field(0.95, gt=0.0, lt=1.0)


class ShortDelayCoherenceFrequency(CoherenceFrequencyProtocol):
    """Estimate the zero-delay phase carrier from correlation samples."""

    name: ClassVar[str] = "short_delay"
    description: ClassVar[str] = "Short-delay first-order coherence carrier"
    config_schema: ClassVar[type[ShortDelayCoherenceFrequencyConfig]] = (
        ShortDelayCoherenceFrequencyConfig
    )
    required_inputs: ClassVar[tuple[str, ...]] = (
        "per_trajectory_correlation",
        "dt",
    )

    def estimate(self, inputs: Mapping[str, Any]) -> CoherenceFrequencyEstimate:
        config = cast(ShortDelayCoherenceFrequencyConfig, self.config)
        correlations = np.asarray(inputs["per_trajectory_correlation"], dtype=complex)
        result, _ = estimate_coherence_carrier(
            correlations,
            float(inputs["dt"]),
            orientation=config.orientation,
            polynomial_order=config.polynomial_order,
            minimum_lag_points=config.minimum_lag_points,
            maximum_lag_points=config.maximum_lag_points,
            consistency_sigma=config.consistency_sigma,
            intensity_floor=config.intensity_floor,
            confidence_level=config.confidence_level,
        )
        status = CoherenceFrequencyStatus.NONE
        if result.status != "failed":
            status |= CoherenceFrequencyStatus.COMPUTED
        if result.status == "ok":
            status |= (
                CoherenceFrequencyStatus.UNIQUE
                | CoherenceFrequencyStatus.INTERIOR_WINDOW
            )
        return CoherenceFrequencyEstimate(
            frequency=result.frequency,
            lag=result.selected_lag_time,
            estimator=self.name,
            conditional_std=result.conditional_std,
            frequency_sem=result.frequency_sem,
            confidence_lower=result.confidence_lower,
            confidence_upper=result.confidence_upper,
            independent_count=int(correlations.shape[0]),
            status=status,
            diagnostics={
                "selected_lag_points": result.selected_lag_points,
                "phase_fit_rms": result.phase_fit_rms,
                "error": result.error,
            },
        )
