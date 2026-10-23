"""Configuration of the spectral-ridge analyser."""

from __future__ import annotations

from typing import Literal

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.core.protocols import PluginConfigBase

Readout = int | Literal["trace"]


def _unique_fields(*names: str) -> list[str]:
    return list(dict.fromkeys(names))

def _ridge_selected_fields(scan_param: str) -> list[str]:
    return _unique_fields(
        "point_index",
        scan_param,
        "frequency",
        "frequency_std",
        "local_frequency_std",
        "scale_frequency_std",
        "peak_value",
        "relative_height",
        "curvature",
        "curvature_std",
        "curvature_significance",
        "scale_support",
        "confidence_lower",
        "confidence_upper",
        "confidence_width",
        "ambiguity_lower",
        "ambiguity_upper",
        "ambiguity_width",
        "ambiguity_candidate_count",
        "plateau_lower",
        "plateau_upper",
        "plateau_width",
        "selected_candidate",
        "candidate_count",
        "retained_candidate_count",
        "accepted_tracking_path_count",
        "row_valid",
    )

def _ridge_candidate_fields(scan_param: str) -> list[str]:
    return _unique_fields(
        "point_index",
        scan_param,
        "candidate_index",
        "selected",
        "retained_for_association",
        "frequency",
        "frequency_std",
        "local_frequency_std",
        "scale_frequency_std",
        "peak_value",
        "relative_height",
        "curvature",
        "curvature_std",
        "curvature_significance",
        "scale_support",
        "confidence_lower",
        "confidence_upper",
        "plateau_lower",
        "plateau_upper",
        "score",
        "row_valid",
    )

def _default_readouts() -> list[Readout]:
    return ["trace"]

class SpectralRidgeConfig(PluginConfigBase):
    """Configuration for non-parametric PSD ridge extraction."""

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Analysis key containing the PSD")
    readouts: list[Readout] = Field(
        default_factory=_default_readouts,
        description="Recorded bare modes and/or the incoherent trace",
    )
    freq_min: float | None = Field(None, description="Lower ridge search bound")
    freq_max: float | None = Field(None, description="Upper ridge search bound")
    maximum_profile_bins: int = Field(
        4096,
        ge=128,
        description="Maximum bins retained for scale-space analysis",
    )
    smoothing_scale_bins: list[float] = Field(
        default_factory=lambda: [2.0, 4.0, 8.0, 16.0],
        description="Gaussian scale-space standard deviations in reduced bins",
    )
    local_window_scale: float = Field(
        4.0,
        ge=2.0,
        description="Local quadratic half-window in smoothing standard deviations",
    )
    maximum_candidates: int = Field(
        4,
        ge=1,
        le=16,
        description="Strongest local maxima retained at every scale",
    )
    minimum_scale_support: int = Field(
        2,
        ge=1,
        description="Distinct smoothing scales required for a ridge candidate",
    )
    minimum_prominence_fraction: float = Field(
        0.002,
        ge=0.0,
        lt=1.0,
        description="Minimum prominence relative to baseline-subtracted peak height",
    )
    cluster_scale_factor: float = Field(
        3.0,
        gt=0.0,
        description="Frequency clustering radius relative to the largest scale",
    )
    confidence_sigma: float = Field(
        2.0,
        gt=0.0,
        description="PSD-SEM threshold used for the peak confidence interval",
    )
    frequency_bin_covariance: Literal["diagonal", "conservative"] = Field(
        "diagonal",
        description=(
            "PSD-bin covariance approximation used when reducing and smoothing SEM"
        ),
    )
    plateau_fraction: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description="Relative peak height defining the descriptive plateau",
    )
    tracking_enabled: bool = Field(
        True,
        description="Select a ridge path using only scan continuity and peak evidence",
    )
    tracking_frequency_scale: float | None = Field(
        None,
        gt=0.0,
        description="Frequency scale for scan-path transition cost",
    )
    tracking_gap_factor: float | None = Field(
        None,
        gt=1.0,
        description=(
            "Split tracking when a scan-axis gap exceeds this times the median gap"
        ),
    )
    tracking_weight: float = Field(1.0, ge=0.0)
    tracking_huber_delta: float = Field(
        1.0,
        gt=0.0,
        description="Huber transition threshold in normalized frequency units",
    )
    tracking_path_count: int = Field(
        2,
        ge=1,
        le=8,
        description="Number of lowest-cost data-only ridge paths retained",
    )
    tracking_max_cost_delta: float = Field(
        2.0,
        ge=0.0,
        description="Maximum path cost above the optimum used for continuity rescue",
    )
    retention_relative_height_fraction: float = Field(
        0.95,
        gt=0.0,
        le=1.0,
        description="Minimum height relative to the strongest point candidate",
    )
    retention_min_scale_support: int = Field(
        3,
        ge=1,
        description="Scale support required for strict candidate retention",
    )
    retention_min_curvature_significance: float = Field(
        3.0,
        ge=0.0,
        description="Curvature significance required for strict retention",
    )
    continuity_rescue: bool = Field(
        True,
        description="Retain high candidates belonging to accepted data-only paths",
    )

    @field_validator("readouts")
    @classmethod
    def validate_readouts(cls, values: list[Readout]) -> list[Readout]:
        if not values:
            raise ValueError("readouts must not be empty")
        if any(isinstance(value, int) and value < 0 for value in values):
            raise ValueError("readout modes must be non-negative")
        keys = [str(value) for value in values]
        if len(keys) != len(set(keys)):
            raise ValueError("readouts must contain unique selections")
        return values

    @field_validator("smoothing_scale_bins")
    @classmethod
    def validate_scales(cls, values: list[float]) -> list[float]:
        if len(values) < 2 or any(
            not np.isfinite(value) or value <= 0.0 for value in values
        ):
            raise ValueError("smoothing_scale_bins must contain at least two positives")
        if any(right <= left for left, right in zip(values, values[1:], strict=False)):
            raise ValueError("smoothing_scale_bins must be strictly increasing")
        return values

    @model_validator(mode="after")
    def validate_ranges(self) -> SpectralRidgeConfig:
        if (
            self.freq_min is not None
            and self.freq_max is not None
            and self.freq_min >= self.freq_max
        ):
            raise ValueError("freq_min must be smaller than freq_max")
        if self.minimum_scale_support > len(self.smoothing_scale_bins):
            raise ValueError(
                "minimum_scale_support cannot exceed the number of smoothing scales"
            )
        return self
