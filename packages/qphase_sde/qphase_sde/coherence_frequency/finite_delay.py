"""Finite-delay coherence frequency child plugin."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field

from ..contracts.coherence import (
    CoherenceFrequencyEstimate,
    CoherenceFrequencyStatus,
)
from .base import BaseCoherenceFrequencyConfig, CoherenceFrequencyProtocol
from .math.finite_delay import finite_delay_carrier_from_spectrum

__all__ = ["FiniteDelayCoherenceFrequency", "FiniteDelayCoherenceFrequencyConfig"]


class FiniteDelayCoherenceFrequencyConfig(BaseCoherenceFrequencyConfig):
    detector_rates: list[float] = Field(default_factory=lambda: [0.1])
    rate_index: int = Field(0, ge=0)
    maximum_lag: float = Field(4096.0, gt=0.0)
    tail_time_constants: float = Field(12.0, gt=1.0)


class FiniteDelayCoherenceFrequency(CoherenceFrequencyProtocol):
    """Estimate a detector-weighted carrier from a PSD."""

    name: ClassVar[str] = "finite_delay"
    description: ClassVar[str] = "Finite-delay coherence-weighted carrier"
    config_schema: ClassVar[type[FiniteDelayCoherenceFrequencyConfig]] = (
        FiniteDelayCoherenceFrequencyConfig
    )
    required_inputs: ClassVar[tuple[str, ...]] = ("axis", "spectrum")

    def estimate(self, inputs: Mapping[str, Any]) -> CoherenceFrequencyEstimate:
        config = cast(FiniteDelayCoherenceFrequencyConfig, self.config)
        if config.rate_index >= len(config.detector_rates):
            raise ValueError("rate_index exceeds detector_rates")
        result = finite_delay_carrier_from_spectrum(
            np.asarray(inputs["axis"], dtype=float),
            np.asarray(inputs["spectrum"], dtype=float),
            np.asarray(config.detector_rates, dtype=float),
            maximum_lag=config.maximum_lag,
            tail_time_constants=config.tail_time_constants,
        )
        index = config.rate_index
        frequency = float(np.asarray(result["frequency"])[index])
        lag = float(np.asarray(result["lag_end"])[index])
        computed = np.isfinite(frequency)
        status = (
            CoherenceFrequencyStatus.COMPUTED
            if computed
            else CoherenceFrequencyStatus.NONE
        )
        return CoherenceFrequencyEstimate(
            frequency=frequency,
            lag=lag,
            estimator=self.name,
            status=status,
            diagnostics={
                "detector_rate": config.detector_rates[index],
                "coherent_weight": float(np.asarray(result["coherent_weight"])[index]),
                "sample_count": int(np.asarray(result["sample_count"])[index]),
            },
        )
