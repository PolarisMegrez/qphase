"""Band-limited coherence frequency child plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field

from ..contracts.coherence import (
    CoherenceFrequencyEstimate,
    CoherenceFrequencyStatus,
)
from ..numerics.band_limited import estimate_band_limited_carrier
from .base import BaseCoherenceFrequencyConfig, CoherenceFrequencyProtocol

__all__ = ["BandLimitedCoherenceFrequency", "BandLimitedCoherenceFrequencyConfig"]


class BandLimitedCoherenceFrequencyConfig(BaseCoherenceFrequencyConfig):
    freq_min: float | None = None
    freq_max: float | None = None
    maximum_lag: float = Field(256.0, gt=0.0)
    bandwidth_min: float | None = Field(None, gt=0.0)
    bandwidth_max: float | None = Field(None, gt=0.0)
    fixed_center: float | None = None
    fixed_bandwidth_max: float | None = Field(None, gt=0.0)


class BandLimitedCoherenceFrequency(CoherenceFrequencyProtocol):
    """Estimate a carrier from a resolved spectral coherence platform."""

    name: ClassVar[str] = "band_limited"
    description: ClassVar[str] = "Adaptive band-limited coherence carrier"
    config_schema: ClassVar[type[BandLimitedCoherenceFrequencyConfig]] = (
        BandLimitedCoherenceFrequencyConfig
    )
    required_inputs: ClassVar[tuple[str, ...]] = ("axis", "spectrum")

    def estimate(self, inputs: Mapping[str, Any]) -> CoherenceFrequencyEstimate:
        config = cast(BandLimitedCoherenceFrequencyConfig, self.config)
        result, _ = estimate_band_limited_carrier(
            np.asarray(inputs["axis"], dtype=float),
            np.asarray(inputs["spectrum"], dtype=float),
            freq_min=config.freq_min,
            freq_max=config.freq_max,
            maximum_lag=config.maximum_lag,
            bandwidth_multipliers=(0.5, 0.75, 1.0, 1.5, 2.0),
            bandwidth_min=config.bandwidth_min,
            bandwidth_max=config.bandwidth_max,
            fixed_center=config.fixed_center,
            fixed_bandwidth_max=config.fixed_bandwidth_max,
        )
        status = CoherenceFrequencyStatus.NONE
        if result.status != "failed":
            status |= CoherenceFrequencyStatus.COMPUTED
        if result.status == "ok":
            status |= CoherenceFrequencyStatus.UNIQUE
            if not result.touched_maximum_lag:
                status |= CoherenceFrequencyStatus.INTERIOR_WINDOW
        return CoherenceFrequencyEstimate(
            frequency=result.frequency,
            bandwidth=result.selected_half_bandwidth,
            lag=result.lag_end,
            estimator=self.name,
            conditional_std=result.diagnostic_uncertainty,
            independent_count=None,
            status=status,
            diagnostics={
                "peak_center": result.peak_center,
                "spectral_width": result.spectral_width,
                "consensus_count": result.consensus_count,
                "error": result.error,
            },
        )
