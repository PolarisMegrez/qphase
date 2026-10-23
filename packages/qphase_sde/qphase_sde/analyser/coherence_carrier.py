"""Rayleigh-matched carrier frequencies from short-delay coherence."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, field_serializer, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    AxisRole,
    CoordinateSchema,
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SamplingBasisSchema,
    UncertaintySchema,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
)
from ..contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    SDEQuantity,
    orientation_sign,
)
from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .base import (
    Analyzer,
    check_execution_cancelled,
    map_trajectory_product,
    report_execution_progress,
    trajectory_arrays,
    trajectory_requirement,
)
from .frequency_orientation import (
    OrientationInput,
    orientation_metadata,
    orientation_schema_extra,
)

__all__ = [
    "CoherenceCarrierAnalyzer",
    "CoherenceCarrierConfig",
    "CoherenceCarrierEstimate",
    "estimate_coherence_carrier",
]


class CoherenceCarrierConfig(PluginConfigBase):
    """Configuration for a short-delay first-order-coherence carrier."""

    modes: list[int] = Field(
        default_factory=list,
        description="Bare physical-mode readouts to include",
    )
    channels: dict[str, list[complex]] = Field(
        default_factory=dict,
        description="Named fixed vectors l for coherent readouts c=l^dagger alpha",
    )
    include_trace: bool = Field(
        True,
        description="Include the incoherent equal-weight trace readout",
    )
    orientation: OrientationInput = Field(
        DEFAULT_FREQUENCY_ORIENTATION,
        description=(
            "Positive-frequency phase orientation: phase_decreasing maps "
            "exp(-i*omega*t) to +omega. Input aliases: physical and fft"
        ),
        json_schema_extra=orientation_schema_extra(),
    )
    polynomial_order: int = Field(
        2,
        ge=1,
        le=4,
        description="Local phase-polynomial order used at zero delay",
    )
    minimum_lag_points: int = Field(
        4,
        ge=2,
        le=64,
        description="Smallest nested short-delay fit window",
    )
    maximum_lag_points: int = Field(
        12,
        ge=2,
        le=256,
        description="Largest nested short-delay fit window",
    )
    consistency_sigma: float = Field(
        2.0,
        gt=0.0,
        description="Nested-window consistency threshold in combined SEM units",
    )
    confidence_level: float = Field(
        0.95,
        gt=0.0,
        lt=1.0,
        description=(
            "Normal-approximation level of the conditional confidence "
            "interval around the selected frequency"
        ),
    )
    time_chunk_samples: int = Field(
        8192,
        ge=1,
        description="Maximum saved time pairs reduced in one backend operation",
    )
    intensity_floor: float = Field(
        1.0e-14,
        ge=0.0,
        description="Minimum C(0) accepted for a readout",
    )

    @model_validator(mode="after")
    def validate_measurements(self) -> CoherenceCarrierConfig:
        names = [f"mode_{mode}" for mode in self.modes]
        if self.include_trace:
            names.append("trace")
        names.extend(self.channels)
        if not names:
            raise ValueError("configure at least one mode, channel, or trace")
        if len(names) != len(set(names)):
            raise ValueError("coherence-carrier measurement names must be unique")
        if len(self.modes) != len(set(self.modes)) or any(
            mode < 0 for mode in self.modes
        ):
            raise ValueError("modes must contain unique non-negative indices")
        if self.minimum_lag_points < self.polynomial_order + 1:
            raise ValueError("minimum_lag_points must exceed polynomial_order")
        if self.maximum_lag_points < self.minimum_lag_points:
            raise ValueError("maximum_lag_points must be at least minimum_lag_points")
        return self

    @field_serializer("channels")
    def serialize_channels(
        self, channels: dict[str, list[complex]]
    ) -> dict[str, list[str]]:
        return {
            name: [str(value) for value in values] for name, values in channels.items()
        }


@dataclass(frozen=True)
class CoherenceCarrierEstimate:
    """One zero-delay phase-derivative estimate and its diagnostics.

    ``conditional_std`` is the conditional uncertainty of the selected
    slope — the local phase-fit residual variance propagated through the
    normal equations, i.e. the error at the computed estimate given the
    ensemble correlation — while ``frequency_sem`` carries the sampling
    scope over independent trajectories. ``confidence_lower``/
    ``confidence_upper`` are the conditional normal-approximation interval
    at the configured confidence level.
    """

    frequency: float = math.nan
    frequency_sem: float = math.nan
    conditional_std: float = math.nan
    confidence_lower: float = math.nan
    confidence_upper: float = math.nan
    selected_lag_points: int = 0
    selected_lag_time: float = math.nan
    phase_fit_rms: float = math.nan
    first_lag_coherence: float = math.nan
    first_lag_phase: float = math.nan
    nyquist_fraction: float = math.nan
    status: str = "failed"
    error: str = ""


@dataclass(frozen=True)
class _CandidateFit:
    lag_points: int
    frequency: float
    frequency_sem: float
    conditional_std: float
    phase_fit_rms: float


def _phase_slope(
    correlation: np.ndarray,
    dt: float,
    lag_points: int,
    order: int,
) -> tuple[float, float, float]:
    c0 = complex(correlation[0])
    if not np.isfinite(c0.real) or abs(c0) <= np.finfo(float).tiny:
        return math.nan, math.nan, math.nan
    normalized = np.asarray(correlation[: lag_points + 1], dtype=complex) / c0
    phase = np.unwrap(np.angle(normalized))
    x = np.arange(1, lag_points + 1, dtype=float) / float(lag_points)
    design = np.column_stack([x**power for power in range(1, order + 1)])
    coefficients, *_ = np.linalg.lstsq(design, phase[1:], rcond=None)
    fitted = design @ coefficients
    slope = float(coefficients[0] / (lag_points * dt))
    residuals = phase[1:] - fitted
    rms = float(np.sqrt(np.mean(residuals**2)))
    # Conditional standard error of the slope: the residual variance of the
    # local phase fit propagated through the normal equations — the error at
    # the computed estimate given the ensemble correlation, not the scatter
    # over trajectories (that stays with the jackknife SEM).
    gram_inverse = np.linalg.inv(design.T @ design)
    conditional = float(
        math.sqrt(
            max(
                float(np.sum(residuals**2)) / (lag_points - order) * gram_inverse[0, 0],
                0.0,
            )
        )
        / (lag_points * dt)
    )
    return slope, rms, conditional


def _jackknife_sem(
    per_trajectory: np.ndarray,
    dt: float,
    lag_points: int,
    order: int,
    sign: float,
) -> float:
    n_traj = int(per_trajectory.shape[0])
    if n_traj < 2:
        return math.nan
    total = np.sum(per_trajectory, axis=0)
    estimates = np.empty(n_traj, dtype=float)
    for index in range(n_traj):
        leave_one = (total - per_trajectory[index]) / float(n_traj - 1)
        slope, _, _ = _phase_slope(leave_one, dt, lag_points, order)
        estimates[index] = sign * slope
    if not np.all(np.isfinite(estimates)):
        return math.nan
    center = float(np.mean(estimates))
    return float(np.sqrt((n_traj - 1.0) / n_traj * np.sum((estimates - center) ** 2)))


def _consistent(
    candidate: _CandidateFit,
    references: list[_CandidateFit],
    sigma: float,
) -> bool:
    for reference in references:
        combined = math.hypot(candidate.frequency_sem, reference.frequency_sem)
        if not np.isfinite(combined):
            return False
        numerical = (
            64.0
            * np.finfo(float).eps
            * max(1.0, abs(candidate.frequency), abs(reference.frequency))
        )
        if (
            abs(candidate.frequency - reference.frequency)
            > sigma * combined + numerical
        ):
            return False
    return True


def estimate_coherence_carrier(
    per_trajectory_correlation: np.ndarray,
    dt: float,
    *,
    orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
    polynomial_order: int = 2,
    minimum_lag_points: int = 4,
    maximum_lag_points: int | None = None,
    consistency_sigma: float = 2.0,
    intensity_floor: float = 1.0e-14,
    confidence_level: float = 0.95,
) -> tuple[CoherenceCarrierEstimate, list[_CandidateFit]]:
    """Estimate ``d arg C(tau)/d tau`` at ``tau=0+``.

    The input contains one independently time-averaged correlation row per
    trajectory. The point estimate is formed from the ensemble correlation;
    leave-one-trajectory-out estimates are used only for uncertainty and
    nested-window selection. ``confidence_level`` sets the level of the
    conditional normal-approximation interval around the selected
    frequency.
    """
    values = np.asarray(per_trajectory_correlation, dtype=complex)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 3:
        return (
            CoherenceCarrierEstimate(
                error="correlations must have shape (n_traj, n_lags>=3)"
            ),
            [],
        )
    if not np.isfinite(dt) or dt <= 0.0:
        return CoherenceCarrierEstimate(error="dt must be positive"), []
    correlation = np.mean(values, axis=0)
    intensity = float(np.real(correlation[0]))
    if not np.isfinite(intensity) or intensity <= intensity_floor:
        return CoherenceCarrierEstimate(error="readout has non-positive intensity"), []

    available = values.shape[1] - 1
    maximum = (
        available if maximum_lag_points is None else min(available, maximum_lag_points)
    )
    minimum = max(minimum_lag_points, polynomial_order + 1)
    if maximum < minimum:
        return CoherenceCarrierEstimate(error="too few lag points for local fit"), []

    sign = orientation_sign(orientation)
    candidates: list[_CandidateFit] = []
    for lag_points in range(minimum, maximum + 1):
        slope, residual, conditional = _phase_slope(
            correlation, dt, lag_points, polynomial_order
        )
        sem = _jackknife_sem(values, dt, lag_points, polynomial_order, sign)
        candidates.append(
            _CandidateFit(
                lag_points=lag_points,
                frequency=sign * slope,
                frequency_sem=sem,
                conditional_std=conditional,
                phase_fit_rms=residual,
            )
        )

    selected = candidates[0]
    if values.shape[0] > 1:
        for candidate in candidates[1:]:
            references = candidates[: candidate.lag_points - minimum]
            if _consistent(candidate, references, consistency_sigma):
                selected = candidate
    else:
        selected = min(candidates, key=lambda candidate: candidate.phase_fit_rms)

    normalized = correlation / correlation[0]
    first_phase = float(np.angle(normalized[1]))
    first_coherence = float(abs(normalized[1]))
    status = "ok" if selected.lag_points > minimum else "minimum_window"
    z = NormalDist().inv_cdf((1.0 + confidence_level) / 2.0)
    if np.isfinite(selected.conditional_std):
        confidence_lower = selected.frequency - z * selected.conditional_std
        confidence_upper = selected.frequency + z * selected.conditional_std
    else:
        confidence_lower = confidence_upper = math.nan
    return (
        CoherenceCarrierEstimate(
            frequency=selected.frequency,
            frequency_sem=selected.frequency_sem,
            conditional_std=selected.conditional_std,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            selected_lag_points=selected.lag_points,
            selected_lag_time=selected.lag_points * dt,
            phase_fit_rms=selected.phase_fit_rms,
            first_lag_coherence=first_coherence,
            first_lag_phase=first_phase,
            nyquist_fraction=abs(first_phase) / math.pi,
            status=status,
        ),
        candidates,
    )


def _measurement_matrices_for_recorded(
    recorded: list[int],
    config: CoherenceCarrierConfig,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    stored_modes = len(recorded)
    mapping = {mode: index for index, mode in enumerate(recorded)}
    missing_modes = [mode for mode in config.modes if mode not in mapping]
    if missing_modes:
        raise ValueError(f"requested modes were not recorded: {missing_modes}")
    names: list[str] = []
    kinds: list[str] = []
    matrices: list[np.ndarray] = []
    for mode in config.modes:
        column = mapping[mode]
        matrix = np.zeros((stored_modes, stored_modes), dtype=complex)
        matrix[column, column] = 1.0
        names.append(f"mode_{mode}")
        kinds.append("bare_mode")
        matrices.append(matrix)
    if config.include_trace:
        names.append("trace")
        kinds.append("incoherent_trace")
        matrices.append(np.eye(stored_modes, dtype=complex))
    recorded_set = set(recorded)
    for name, raw in config.channels.items():
        channel = np.asarray(raw, dtype=complex)
        if channel.ndim != 1 or channel.size <= max(recorded, default=-1):
            raise ValueError(
                f"channel {name!r} must index every recorded physical mode"
            )
        missing = [
            index
            for index, value in enumerate(channel)
            if index not in recorded_set and abs(value) > np.finfo(float).tiny
        ]
        if missing:
            raise ValueError(
                f"channel {name!r} uses unrecorded physical modes {missing}"
            )
        selected = channel[recorded]
        norm = float(np.linalg.norm(selected))
        if not np.isfinite(norm) or norm <= np.finfo(float).tiny:
            raise ValueError(f"channel {name!r} must have finite non-zero norm")
        selected /= norm
        names.append(name)
        kinds.append("coherent_channel")
        matrices.append(np.outer(selected, selected.conj()))
    return names, kinds, recorded, np.asarray(matrices)


class CoherenceCarrierAnalyzer(Analyzer):
    """Estimate an experimentally readable, Rayleigh-matched carrier."""

    name: ClassVar[str] = "coherence_carrier"
    description: ClassVar[str] = (
        "Short-delay first-order-coherence carrier matched to CAM Rayleigh quotients"
    )
    config_schema: ClassVar[type[CoherenceCarrierConfig]] = CoherenceCarrierConfig

    def input_requirements(self) -> list[ProductRequirement]:
        return [trajectory_requirement()]

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        del metadata
        trajectory = inputs.get("trajectories")
        if trajectory is None:
            raise TypeError("coherence_carrier requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            values, t0, dt, recorded = trajectory_arrays(point)
            payload = self._compute_payload(
                values,
                recorded_modes=list(recorded),
                t0=t0,
                dt=dt,
                backend=backend,
                context=context,
            )
            products = _build_coherence_carrier_products(
                payload, scan_size=None, label=label
            )
            if not products:
                raise ValueError("coherence_carrier produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        values: Any,
        *,
        recorded_modes: list[int],
        t0: float,
        dt: float,
        backend: BackendBase,
        context: Any | None = None,
    ) -> dict[str, Any]:
        config = cast(CoherenceCarrierConfig, self.config)
        if values.ndim != 3 or not np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError(
                "coherence_carrier expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, stored_modes = map(int, values.shape)
        if n_traj < 1 or n_samples <= config.maximum_lag_points:
            raise ValueError(
                "coherence_carrier requires trajectories longer than maximum_lag_points"
            )
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        if len(recorded_modes) != stored_modes:
            raise ValueError("recorded mode labels do not match trajectory columns")
        names, kinds, recorded, matrices = _measurement_matrices_for_recorded(
            recorded_modes, config
        )
        backend_matrices = backend.asarray(matrices, dtype=values.dtype)
        correlations = backend.zeros(
            (n_traj, len(names), config.maximum_lag_points + 1),
            dtype=values.dtype,
        )
        samples_done = 0
        for lag in range(config.maximum_lag_points + 1):
            check_execution_cancelled(context)
            pair_count = n_samples - lag
            for start in range(0, pair_count, config.time_chunk_samples):
                stop = min(pair_count, start + config.time_chunk_samples)
                left = values[:, start:stop, :]
                right = values[:, start + lag : stop + lag, :]
                correlations[:, :, lag] += backend.einsum(
                    "rti,mij,rtj->rm",
                    left.conj(),
                    backend_matrices,
                    right,
                )
            correlations[:, :, lag] /= float(pair_count)
            samples_done += pair_count
            report_execution_progress(
                context,
                completed=samples_done,
                unit="sample",
                message=(
                    f"coherence_carrier: correlated lag {lag + 1}/"
                    f"{config.maximum_lag_points + 1}"
                ),
            )
        return self._summarize(
            per_trajectory=np.asarray(convert_to_numpy(correlations)),
            names=names,
            kinds=kinds,
            recorded_modes=recorded,
            matrices=matrices,
            dt=dt,
            n_samples=n_samples,
            t0=t0,
            context=context,
        )

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Declare the single graph-ready coherence-carrier product."""
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                quantity=SDEQuantity.COHERENCE_FREQUENCY.value,
                fields=list(_COHERENCE_CARRIER_FIELDS),
            ),
        )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(
            preferred_location="backend",
            supports_trajectory_batching=True,
            supports_time_streaming=False,
            supports_scan_tiling=True,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        config = cast(CoherenceCarrierConfig, self.config)
        n_measurements = (
            len(config.modes) + len(config.channels) + int(config.include_trace)
        )
        chunk = min(config.time_chunk_samples, request.saved_samples)
        complex_itemsize = 2 * request.real_itemsize
        chunk_bytes = (
            2 * request.n_traj * chunk * request.n_record_modes * complex_itemsize
        )
        batch_correlations = (
            request.n_traj
            * n_measurements
            * (config.maximum_lag_points + 1)
            * complex_itemsize
        )
        total_n_traj = request.total_n_traj or request.n_traj
        batch_count = math.ceil(total_n_traj / request.n_traj)
        trajectory_payload = (
            total_n_traj
            * n_measurements
            * (config.maximum_lag_points + 1)
            * complex_itemsize
        )
        measurement_matrices = (
            n_measurements
            * request.n_record_modes
            * request.n_record_modes
            * complex_itemsize
        )
        fixed_payload = (
            measurement_matrices
            + request.n_record_modes * 8
            + n_measurements * 128
        )
        retained_host_bytes = 2 * trajectory_payload + batch_count * fixed_payload
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(
                device_bytes=chunk_bytes + batch_correlations + measurement_matrices,
                retained_host_bytes=retained_host_bytes,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=chunk_bytes,
            retained_host_bytes=retained_host_bytes,
        )

    def create_result_accumulator(self) -> CoherenceCarrierResultAccumulator:
        return CoherenceCarrierResultAccumulator(self)

    def _summarize(
        self,
        *,
        per_trajectory: np.ndarray,
        names: list[str],
        kinds: list[str],
        recorded_modes: list[int],
        matrices: np.ndarray,
        dt: float,
        n_samples: int,
        t0: float,
        context: Any | None = None,
    ) -> dict[str, Any]:
        config = cast(CoherenceCarrierConfig, self.config)
        estimates: list[CoherenceCarrierEstimate] = []
        candidate_frequencies = []
        candidate_sems = []
        candidate_residuals = []
        for index in range(len(names)):
            check_execution_cancelled(context)
            estimate, candidates = estimate_coherence_carrier(
                per_trajectory[:, index, :],
                dt,
                orientation=config.orientation,
                polynomial_order=config.polynomial_order,
                minimum_lag_points=config.minimum_lag_points,
                maximum_lag_points=config.maximum_lag_points,
                consistency_sigma=config.consistency_sigma,
                intensity_floor=config.intensity_floor,
                confidence_level=config.confidence_level,
            )
            estimates.append(estimate)
            candidate_frequencies.append([item.frequency for item in candidates])
            candidate_sems.append([item.frequency_sem for item in candidates])
            candidate_residuals.append([item.phase_fit_rms for item in candidates])

        correlation = np.mean(per_trajectory, axis=0)
        n_traj = int(per_trajectory.shape[0])
        if n_traj > 1:
            correlation_sem_real = np.std(
                per_trajectory.real, axis=0, ddof=1
            ) / math.sqrt(n_traj)
            correlation_sem_imag = np.std(
                per_trajectory.imag, axis=0, ddof=1
            ) / math.sqrt(n_traj)
        else:
            correlation_sem_real = np.full(correlation.shape, np.nan)
            correlation_sem_imag = np.full(correlation.shape, np.nan)
        metadata = orientation_metadata(config.orientation)
        return {
            "quantity": "short_delay_first_order_coherence_carrier",
            "definition": "omega = orientation_sign * Im[C_W'(0+) / C_W(0)]",
            "cam_correspondence": (
                "C_W'(0+)=-i*Tr[W*H(R)*R]; phase_decreasing gives "
                "Re Tr[W*H(R)*R]/Tr[W*R] under CAM closure"
            ),
            "method": "nested_local_phase_polynomial_with_trajectory_jackknife",
            "measurement_names": list(names),
            "measurement_kinds": list(kinds),
            "recorded_modes": list(recorded_modes),
            "measurement_matrices": np.asarray(matrices),
            "frequency": np.asarray([item.frequency for item in estimates]),
            "frequency_sem": np.asarray([item.frequency_sem for item in estimates]),
            "conditional_std": np.asarray([item.conditional_std for item in estimates]),
            "confidence_lower": np.asarray(
                [item.confidence_lower for item in estimates]
            ),
            "confidence_upper": np.asarray(
                [item.confidence_upper for item in estimates]
            ),
            "status": np.asarray([item.status for item in estimates]),
            "error": np.asarray([item.error for item in estimates]),
            "selected_lag_points": np.asarray(
                [item.selected_lag_points for item in estimates]
            ),
            "selected_lag_time": np.asarray(
                [item.selected_lag_time for item in estimates]
            ),
            "phase_fit_rms": np.asarray([item.phase_fit_rms for item in estimates]),
            "first_lag_coherence": np.asarray(
                [item.first_lag_coherence for item in estimates]
            ),
            "first_lag_phase": np.asarray([item.first_lag_phase for item in estimates]),
            "nyquist_fraction": np.asarray(
                [item.nyquist_fraction for item in estimates]
            ),
            "candidate_lag_points": np.arange(
                config.minimum_lag_points,
                config.maximum_lag_points + 1,
                dtype=int,
            ),
            "candidate_frequency": np.asarray(candidate_frequencies),
            "candidate_frequency_sem": np.asarray(candidate_sems),
            "candidate_phase_fit_rms": np.asarray(candidate_residuals),
            "lag": np.arange(config.maximum_lag_points + 1, dtype=float) * dt,
            "correlation": correlation,
            "correlation_sem_real": correlation_sem_real,
            "correlation_sem_imag": correlation_sem_imag,
            "per_trajectory_correlation": per_trajectory,
            "n_traj": n_traj,
            "n_samples": int(n_samples),
            "t0": float(t0),
            "dt": float(dt),
            "polynomial_order": config.polynomial_order,
            "consistency_sigma": config.consistency_sigma,
            "uncertainty": {
                "available": n_traj > 1,
                "independent_unit": "trajectory",
                "n_independent": n_traj,
                "frequency_method": "leave_one_trajectory_out_jackknife",
                "conditional_method": "local_phase_polynomial_residual",
                "confidence_level": float(config.confidence_level),
                "point_estimate_is_ratio_of_ensemble_correlations": True,
            },
            **metadata,
        }


#: Numeric payload leaves of ``analyze()`` (``_summarize``); string leaves
#: (``status``, ``error``, ``measurement_names``, ...) stay in payload metadata.
_COHERENCE_CARRIER_FIELDS = (
    "recorded_modes",
    "measurement_matrices",
    "frequency",
    "frequency_sem",
    "conditional_std",
    "confidence_lower",
    "confidence_upper",
    "selected_lag_points",
    "selected_lag_time",
    "phase_fit_rms",
    "first_lag_coherence",
    "first_lag_phase",
    "nyquist_fraction",
    "candidate_lag_points",
    "candidate_frequency",
    "candidate_frequency_sem",
    "candidate_phase_fit_rms",
    "lag",
    "correlation",
    "correlation_sem_real",
    "correlation_sem_imag",
    "per_trajectory_correlation",
    "n_traj",
    "n_samples",
    "t0",
    "dt",
    "polynomial_order",
    "consistency_sigma",
    "uncertainty.available",
    "uncertainty.n_independent",
    "uncertainty.confidence_level",
    "uncertainty.point_estimate_is_ratio_of_ensemble_correlations",
)


def _build_coherence_carrier_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset] | None:
    """Assemble the typed coherence-carrier product of one analyser payload."""
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    method = leaves.payload_meta.get("method", "short_delay")
    if isinstance(method, list):
        # Scan payloads keep per-point strings; the estimator is job-wide.
        method = method[0] if method else "short_delay"
    coordinates: list[CoordinateSchema] = []
    if "lag" in leaves.arrays:
        leading = ("scan",) if scan_size is not None else ()
        coordinates.append(
            CoordinateSchema(
                name="lag",
                variable="lag",
                dims=(*leading, "lag"),
                role="auxiliary" if leading else "dimension",
                units="time",
            )
        )
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims={
            "frequency": ("measurement",),
            "frequency_sem": ("measurement",),
            "conditional_std": ("measurement",),
            "confidence_lower": ("measurement",),
            "confidence_upper": ("measurement",),
            "selected_lag_points": ("measurement",),
            "selected_lag_time": ("measurement",),
            "phase_fit_rms": ("measurement",),
            "first_lag_coherence": ("measurement",),
            "first_lag_phase": ("measurement",),
            "nyquist_fraction": ("measurement",),
            "candidate_lag_points": ("candidate",),
            "candidate_frequency": ("measurement", "candidate"),
            "candidate_frequency_sem": ("measurement", "candidate"),
            "candidate_phase_fit_rms": ("measurement", "candidate"),
            "lag": ("lag",),
            "correlation": ("measurement", "lag"),
            "correlation_sem_real": ("measurement", "lag"),
            "correlation_sem_imag": ("measurement", "lag"),
            "per_trajectory_correlation": ("trajectory", "measurement", "lag"),
            "measurement_matrices": ("measurement", "mode", "mode_2"),
        },
        axis_specs={
            "measurement": TypedAxisSpec("measurement", AxisRole.COMPONENT),
            "candidate": TypedAxisSpec("candidate", AxisRole.INDEX),
            "lag": TypedAxisSpec("lag", AxisRole.COORDINATE, units="time"),
            "mode": TypedAxisSpec("mode", AxisRole.COMPONENT),
            "mode_2": TypedAxisSpec("mode_2", AxisRole.COMPONENT),
            "trajectory": TypedAxisSpec("trajectory", AxisRole.REALIZATION),
        },
        quantities={"frequency": SDEQuantity.COHERENCE_FREQUENCY.value},
        uncertainties=[
            UncertaintySchema(
                target="frequency",
                kind="sample_std",
                covariance="real",
                scope="conditional",
                data_variable="conditional_std",
            ),
            UncertaintySchema(
                target="frequency",
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable="frequency_sem",
            ),
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", source_axis="trajectory")
        ],
        coordinates=coordinates,
        attributes={
            "graph_ready": True,
            "estimator": str(method) if method else "short_delay",
        },
    )
    if dataset is None:
        return None
    return {label: dataset}


class CoherenceCarrierResultAccumulator:
    """Merge trajectory batches before estimating the correlation ratio."""

    def __init__(self, analyzer: CoherenceCarrierAnalyzer) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def _append_payload(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            for key in (
                "measurement_names",
                "measurement_kinds",
                "recorded_modes",
                "n_samples",
            ):
                if first[key] != payload[key]:
                    raise ValueError(
                        "coherence-carrier batches used incompatible measurements"
                    )
            if not math.isclose(float(first["dt"]), float(payload["dt"])):
                raise ValueError("coherence-carrier batches used different time grids")
            if not np.array_equal(
                first["measurement_matrices"], payload["measurement_matrices"]
            ):
                raise ValueError(
                    "coherence-carrier batches used different readout matrices"
                )
        self.payloads.append(payload)

    def consume_product(self, product: Dataset) -> None:
        def value(name: str) -> Any:
            return product.handle(name).materialize()

        metadata = product.attributes.get("payload_meta", {})
        if not isinstance(metadata, Mapping):
            raise TypeError("coherence carrier product metadata is invalid")
        self._append_payload(
            {
                "measurement_names": metadata["measurement_names"],
                "measurement_kinds": metadata["measurement_kinds"],
                "recorded_modes": [
                    int(mode) for mode in np.asarray(value("recorded_modes"))
                ],
                "measurement_matrices": value("measurement_matrices"),
                "n_traj": int(value("n_traj")),
                "n_samples": int(value("n_samples")),
                "t0": float(value("t0")),
                "dt": float(value("dt")),
                "per_trajectory_correlation": value("per_trajectory_correlation"),
            }
        )

    def merge(self, other: CoherenceCarrierResultAccumulator) -> None:
        for payload in other.payloads:
            self._append_payload(payload)

    def finalize_product(self, label: str) -> Mapping[str, Dataset] | None:
        return _build_coherence_carrier_products(
            self._finalized_payload(), scan_size=None, label=label
        )

    def checkpoint(self) -> dict[str, Any]:
        return {"payloads": list(self.payloads)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        payloads = checkpoint.get("payloads")
        if not isinstance(payloads, list):
            raise TypeError("coherence carrier checkpoint is missing payloads")
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("coherence carrier checkpoint payload is invalid")
            self._append_payload(payload)

    def _finalized_payload(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty coherence-carrier accumulator")
        first = self.payloads[0]
        return self.analyzer._summarize(
            per_trajectory=np.concatenate(
                [
                    np.asarray(item["per_trajectory_correlation"])
                    for item in self.payloads
                ],
                axis=0,
            ),
            names=list(first["measurement_names"]),
            kinds=list(first["measurement_kinds"]),
            recorded_modes=list(first["recorded_modes"]),
            matrices=np.asarray(first["measurement_matrices"]),
            dt=float(first["dt"]),
            n_samples=int(first["n_samples"]),
            t0=float(first["t0"]),
        )
