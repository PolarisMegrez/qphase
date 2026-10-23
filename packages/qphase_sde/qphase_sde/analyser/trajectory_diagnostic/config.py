"""Configuration model and numeric constants of the diagnostics analyser."""

from __future__ import annotations

import math

from pydantic import Field, model_validator
from qphase.core.protocols import PluginConfigBase

from ...contracts.quantities import DEFAULT_FREQUENCY_ORIENTATION
from ..frequency_orientation import OrientationInput, orientation_schema_extra

# Minimum lag statistics required before any g1 line-shape model is fitted.
_G1_MIN_FINITE_LAGS = 8
_G1_MIN_SIGNAL_LAGS = 6
# A lag counts as signal while |g1| stays above this multiple of its SEM.
_G1_NOISE_FLOOR_FACTOR = 2.0
# Holdout RMS larger than factor * train RMS (plus an absolute guard) marks a
# fit as "holdout_mismatch"; the guard avoids flagging numerical noise.
_G1_HOLDOUT_MISMATCH_FACTOR = 3.0
_G1_HOLDOUT_MISMATCH_ATOL = 1.0e-6
# Holdout RMS values within this relative band of the best one are treated as
# ties and resolved toward the model with fewer parameters.
_G1_PREFERRED_RELATIVE_TOL = 0.02

# A block group must clear both thresholds to yield spectral features.
_BLOCK_SPECTRUM_MIN_SAMPLES = 8
_BLOCK_SPECTRUM_MIN_BLOCKS = 2

# Canonical R = alpha alpha^dagger coordinates (real, length n_modes**2) are
# shared with the observer subpackage via ``qphase_sde.numerics.coordinates``.

# Fewer blocks than this cannot support a meaningful covariance across blocks.
_STATIONARITY_MIN_BLOCKS = 4
_STATIONARITY_RADIAL_QUANTILES = (0.1, 0.5, 0.9)
# Above this covariance condition number the radial distance falls back to
# Euclidean (the Mahalanobis metric would be dominated by numerical noise).
_STATIONARITY_MAX_CONDITION = 1.0e6

# Numeric leaves of the phase-increment summary; orientation metadata strings
# are payload meta, not variables.
_PHASE_INCREMENT_FIELDS = (
    "mean_angular_frequency_per_trajectory",
    "max_abs_phase_step_per_trajectory",
    "near_nyquist_fraction_per_trajectory",
    "near_nyquist_threshold",
)
# Numeric leaves of ``allan_statistics._assemble_result`` (string quantity,
# unit, and definition keys are payload meta).
_ALLAN_TABLE_FIELDS = (
    "tau",
    "angular_frequency_variance",
    "angular_frequency_variance_sem",
    "per_trajectory",
    "valid_second_differences",
    "trajectory_sample_count",
    "nonoverlap_angular_frequency_variance",
    "nonoverlap_angular_frequency_variance_sem",
    "nonoverlap_per_trajectory",
    "nonoverlap_valid_second_differences",
    "nonoverlap_trajectory_sample_count",
    "nominal_independent_windows_per_trajectory",
    "total_independent_window_count",
    "observation_duration",
    "min_independent_windows",
)
# Always-present numeric leaves of the g1 model-fit summary; converged-fit
# parameters and residuals are data-dependent and stay undeclared.
_G1_MODEL_FIT_FIELDS = (
    "lag_step_seconds",
    "lag_window_seconds",
    "n_lags",
    "n_finite",
    "n_signal",
    "min_finite_lags",
    "min_signal_lags",
    "holdout_mismatch_factor",
)


class TrajectoryDiagnosticsConfig(PluginConfigBase):
    """Configuration for time-domain trajectory diagnostics."""

    modes: list[int] = Field(..., min_length=1, description="Physical mode indices")
    orientation: OrientationInput = Field(
        DEFAULT_FREQUENCY_ORIENTATION,
        description=(
            "Positive-frequency phase orientation: phase_decreasing maps "
            "exp(-i*omega*t) to +omega. Input aliases: physical and fft"
        ),
        json_schema_extra=orientation_schema_extra(),
    )
    block_durations: list[float] = Field(
        default_factory=list,
        description="Non-overlapping block durations used for stationarity summaries",
    )
    coherence: bool = Field(True, description="Compute complex first-order coherence")
    coherence_max_lag: float | None = Field(
        None,
        gt=0.0,
        description="Largest coherence lag; None uses at most 4096 saved samples",
    )
    allan: bool = Field(True, description="Compute angular-frequency Allan variance")
    allan_taus: list[float] | None = Field(
        None,
        description="Requested Allan averaging times; None selects logarithmic times",
    )
    allan_points: int = Field(24, ge=2, le=256)
    allan_min_windows: int = Field(8, ge=1)
    allan_min_independent_windows: int = Field(
        1,
        ge=1,
        description="Minimum nominal non-overlapping Allan windows per trajectory",
    )
    amplitude_floor: float = Field(
        0.0,
        ge=0.0,
        description="Exclude Allan phase differences touching lower amplitudes",
    )
    center_coherence: bool = Field(
        False,
        description="Subtract each trajectory mean before computing coherence",
    )
    keep_coherence_per_trajectory: bool = Field(
        False,
        description="Retain per-trajectory coherence arrays in addition to summaries",
    )
    block_spectrum: bool = Field(
        False,
        description=(
            "Compute per-trajectory per-block periodogram features (peak, "
            "local HWHM, integrated power, wing fraction) for block_durations"
        ),
    )
    block_spectrum_wing_factor: float = Field(
        3.0,
        gt=1.0,
        description=(
            "Wings start outside the peak core halfwidth defined as "
            "max(wing_factor * resolution_angular, local_hwhm_angular)"
        ),
    )
    stationarity_details: bool = Field(
        False,
        description=(
            "Adjudicate stationarity per trajectory from block-mean canonical "
            "R features over block_durations: block covariance, radial "
            "quantiles, head-tail drift, and a change-point score"
        ),
    )
    matrix_projection: bool = Field(
        False,
        description=(
            "Project canonical R = alpha alpha^dagger coordinates (built from "
            "all recorded modes) onto matrix_projection_left_vector against "
            "matrix_projection_reference"
        ),
    )
    matrix_projection_reference: list[float] | None = Field(
        None,
        description=(
            "Flat real canonical R coordinates of length n_modes**2 "
            "subtracted before projection; None uses the zero matrix"
        ),
    )
    matrix_projection_left_vector: list[float] | None = Field(
        None,
        description=(
            "Flat real weights of length n_modes**2 applied to the canonical "
            "coordinates; None emits coordinate summaries without a projection"
        ),
    )
    matrix_projection_keep_coordinates: bool = Field(
        False,
        description=(
            "Retain full per-trajectory canonical coordinate trajectories "
            "(n_traj, n_time, n_modes**2) in the matrix_projection output"
        ),
    )

    @model_validator(mode="after")
    def validate_durations(self) -> TrajectoryDiagnosticsConfig:
        if any(value <= 0.0 for value in self.block_durations):
            raise ValueError("block_durations must contain positive values")
        if self.allan_taus is not None and any(
            value <= 0.0 for value in self.allan_taus
        ):
            raise ValueError("allan_taus must contain positive values")
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        vectors = (
            ("matrix_projection_reference", self.matrix_projection_reference),
            ("matrix_projection_left_vector", self.matrix_projection_left_vector),
        )
        for label, vector in vectors:
            if vector is None:
                continue
            if not self.matrix_projection:
                raise ValueError(f"{label} requires matrix_projection=True")
            root = math.isqrt(len(vector))
            if root < 1 or root * root != len(vector):
                raise ValueError(f"{label} must have length n_modes**2")
        reference = self.matrix_projection_reference
        left_vector = self.matrix_projection_left_vector
        if (
            reference is not None
            and left_vector is not None
            and len(reference) != len(left_vector)
        ):
            raise ValueError(
                "matrix_projection_reference and matrix_projection_left_vector "
                "must have matching lengths"
            )
        return self
