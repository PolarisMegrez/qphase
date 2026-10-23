"""Configuration of the band-limited carrier analyser."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.core.protocols import PluginConfigBase

from ..spectral_ridge import SpectralRidgeConfig

_CARRIER_RESULT_FIELDS = (
    "frequency",
    "regression_std",
    "bandwidth_std",
    "diagnostic_uncertainty",
    "peak_center",
    "peak_hwhm",
    "spectral_width",
    "selected_half_bandwidth",
    "consensus_count",
    "phase_fit_rms",
    "lag_points",
    "lag_start",
    "lag_end",
    "frequency_drift",
    "decay_rate",
    "touched_maximum_lag",
    "platform_count",
    "tracked_frequency",
    "tracked_diagnostic_uncertainty",
    "tracked_platform_index",
    "tracked_path_cost_gap",
    "row_valid",
)


def _unique_fields(*names: str) -> list[str]:
    return list(dict.fromkeys(names))

def _carrier_result_fields(scan_param: str) -> list[str]:
    return _unique_fields(scan_param, *_CARRIER_RESULT_FIELDS)

def _carrier_candidate_fields(scan_param: str) -> list[str]:
    return _unique_fields(
        scan_param,
        "half_bandwidth",
        "frequency",
        "regression_std",
        "phase_fit_rms",
        "frequency_drift",
        "decay_rate",
        "lag_points",
        "lag_start",
        "lag_end",
        "touched_maximum_lag",
        "row_valid",
    )

def _carrier_platform_fields(scan_param: str) -> list[str]:
    return _unique_fields(
        scan_param,
        "platform_index",
        "frequency",
        "regression_std",
        "bandwidth_std",
        "diagnostic_uncertainty",
        "score",
        "first_candidate",
        "last_candidate",
        "candidate_count",
        "minimum_half_bandwidth",
        "maximum_half_bandwidth",
        "log_bandwidth_span",
        "phase_fit_rms",
        "frequency_drift",
        "decay_rate",
        "lag_start",
        "lag_end",
        "touched_maximum_lag",
        "row_valid",
    )

class RidgeConditionedCenterConfig(PluginConfigBase):
    """Use retained spectral ridges as coarse carrier centers.

    A workflow authors the ridge configuration inline as ``spectral_ridge``;
    the analyser-config expander promotes it into a sibling ``spectral_ridge``
    analyser and replaces it here with that analyser's ``ridge_label``. The
    carrier then consumes the typed ridge products from the product graph
    instead of running a ridge analysis of its own.
    """

    spectral_ridge: SpectralRidgeConfig | None = Field(
        None, description="Inline spectral-ridge configuration (authored form)"
    )
    ridge_label: str | None = Field(
        None,
        description="Label of the expanded spectral_ridge analyser whose "
        "products this carrier consumes",
    )
    maximum_neighbor_fraction: float = Field(
        0.45,
        gt=0.0,
        le=0.5,
        description="Maximum half-band relative to the nearest retained ridge",
    )

    @model_validator(mode="after")
    def validate_ridge_source(self) -> RidgeConditionedCenterConfig:
        if (self.spectral_ridge is None) == (self.ridge_label is None):
            raise ValueError(
                "ridge-conditioned center needs exactly one of spectral_ridge "
                "(authored inline) or ridge_label (compiled sibling analyser)"
            )
        return self

class BandLimitedCarrierConfig(PluginConfigBase):
    """Configuration for adaptive, spectrally filtered carrier estimation."""

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD")
    readout: int | Literal["trace"] = Field(
        "trace",
        description="Physical mode index or incoherent trace over recorded modes",
    )
    center: RidgeConditionedCenterConfig | None = Field(
        None,
        description="Optional spectral-ridge-conditioned center estimator",
    )
    freq_min: float | None = Field(None, description="Lower carrier search bound")
    freq_max: float | None = Field(None, description="Upper carrier search bound")
    bandwidth_multipliers: list[float] = Field(
        default_factory=lambda: [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0],
        description=(
            "Nested half-bandwidths relative to the spectral concentration width"
        ),
    )
    bandwidth_min: float | None = Field(
        None, gt=0.0, description="Optional minimum candidate half-bandwidth"
    )
    bandwidth_max: float | None = Field(
        None, gt=0.0, description="Optional maximum candidate half-bandwidth"
    )
    baseline_quantile: float = Field(0.05, ge=0.0, lt=0.5)
    concentration_clip_sigma: float = Field(5.0, gt=1.0)
    concentration_iterations: int = Field(2, ge=0, le=8)
    peak_smoothing_bins: int = Field(5, ge=1, le=101)
    taper_fraction: float = Field(0.2, gt=0.0, lt=1.0)
    maximum_lag: float = Field(256.0, gt=0.0)
    coherence_floor: float = Field(0.05, gt=0.0, lt=1.0)
    minimum_lag_points: int = Field(8, ge=3)
    minimum_lag_span: float | None = Field(
        None,
        gt=0.0,
        description=("Minimum physical lag span; None uses 10% of maximum_lag"),
    )
    lag_window_trials: int = Field(
        16,
        ge=4,
        le=64,
        description="Maximum start/end grid density used to find lag plateaus",
    )
    max_phase_fit_rms: float = Field(
        0.05,
        gt=0.0,
        description="Maximum weighted phase residual in radians",
    )
    max_frequency_drift_fraction: float = Field(
        0.05,
        gt=0.0,
        description="Maximum phase-curvature frequency drift relative to bandwidth",
    )
    max_negative_decay_fraction: float = Field(
        0.02,
        ge=0.0,
        description="Allowed apparent correlation growth relative to bandwidth",
    )
    consensus_count: int = Field(3, ge=2, le=16)
    minimum_log_bandwidth_span: float = Field(
        0.2,
        gt=0.0,
        description="Minimum natural-log bandwidth span supporting one platform",
    )
    stability_fraction: float = Field(
        0.015,
        gt=0.0,
        description="Allowed carrier drift as a fraction of candidate bandwidth",
    )
    stability_sigma: float = Field(2.0, gt=0.0)
    platform_ambiguity_delta: float = Field(
        0.5,
        ge=0.0,
        description="Maximum score difference retaining a competing platform",
    )
    tracking_enabled: bool = Field(
        True,
        description="Track candidate platforms continuously across the scan",
    )
    tracking_frequency_scale: float | None = Field(
        None,
        gt=0.0,
        description="Optional frequency scale for scan-path curvature",
    )
    tracking_curvature_weight: float = Field(1.0, ge=0.0)
    tracking_max_normalized_curvature: float = Field(
        3.0,
        gt=0.0,
        description="Reject tracked points exceeding this normalized curvature",
    )
    tracking_ambiguity_delta: float = Field(
        1.0,
        ge=0.0,
        description="Maximum total-cost gap treated as a competing scan path",
    )

    @field_validator("bandwidth_multipliers")
    @classmethod
    def validate_bandwidth_multipliers(cls, values: list[float]) -> list[float]:
        if len(values) < 2 or any(
            not np.isfinite(value) or value <= 0 for value in values
        ):
            raise ValueError(
                "bandwidth_multipliers must contain at least two positives"
            )
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("bandwidth_multipliers must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_ranges(self) -> BandLimitedCarrierConfig:
        if (
            self.freq_min is not None
            and self.freq_max is not None
            and self.freq_min >= self.freq_max
        ):
            raise ValueError("freq_min must be smaller than freq_max")
        if (
            self.bandwidth_min is not None
            and self.bandwidth_max is not None
            and self.bandwidth_min > self.bandwidth_max
        ):
            raise ValueError("bandwidth_min cannot exceed bandwidth_max")
        if self.consensus_count > len(self.bandwidth_multipliers):
            raise ValueError("consensus_count cannot exceed the number of bandwidths")
        return self
