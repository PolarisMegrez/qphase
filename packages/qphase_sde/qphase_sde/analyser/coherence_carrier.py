"""Rayleigh-matched carrier frequencies from short-delay coherence."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, field_serializer, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase

from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
    resolve_mode_columns,
)
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    OrientationInput,
    orientation_metadata,
    orientation_schema_extra,
    orientation_sign,
)
from .result import AnalysisResult

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
            raise ValueError(
                "minimum_lag_points must exceed polynomial_order"
            )
        if self.maximum_lag_points < self.minimum_lag_points:
            raise ValueError(
                "maximum_lag_points must be at least minimum_lag_points"
            )
        return self

    @field_serializer("channels")
    def serialize_channels(
        self, channels: dict[str, list[complex]]
    ) -> dict[str, list[str]]:
        return {
            name: [str(value) for value in values]
            for name, values in channels.items()
        }


@dataclass(frozen=True)
class CoherenceCarrierEstimate:
    """One zero-delay phase-derivative estimate and its diagnostics."""

    frequency: float = math.nan
    frequency_sem: float = math.nan
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
    phase_fit_rms: float


def _phase_slope(
    correlation: np.ndarray,
    dt: float,
    lag_points: int,
    order: int,
) -> tuple[float, float]:
    c0 = complex(correlation[0])
    if not np.isfinite(c0.real) or abs(c0) <= np.finfo(float).tiny:
        return math.nan, math.nan
    normalized = np.asarray(correlation[: lag_points + 1], dtype=complex) / c0
    phase = np.unwrap(np.angle(normalized))
    x = np.arange(1, lag_points + 1, dtype=float) / float(lag_points)
    design = np.column_stack([x**power for power in range(1, order + 1)])
    coefficients, *_ = np.linalg.lstsq(design, phase[1:], rcond=None)
    fitted = design @ coefficients
    slope = float(coefficients[0] / (lag_points * dt))
    rms = float(np.sqrt(np.mean((phase[1:] - fitted) ** 2)))
    return slope, rms


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
        slope, _ = _phase_slope(leave_one, dt, lag_points, order)
        estimates[index] = sign * slope
    if not np.all(np.isfinite(estimates)):
        return math.nan
    center = float(np.mean(estimates))
    return float(
        np.sqrt((n_traj - 1.0) / n_traj * np.sum((estimates - center) ** 2))
    )


def _consistent(
    candidate: _CandidateFit,
    references: list[_CandidateFit],
    sigma: float,
) -> bool:
    for reference in references:
        combined = math.hypot(candidate.frequency_sem, reference.frequency_sem)
        if not np.isfinite(combined):
            return False
        numerical = 64.0 * np.finfo(float).eps * max(
            1.0, abs(candidate.frequency), abs(reference.frequency)
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
) -> tuple[CoherenceCarrierEstimate, list[_CandidateFit]]:
    """Estimate ``d arg C(tau)/d tau`` at ``tau=0+``.

    The input contains one independently time-averaged correlation row per
    trajectory. The point estimate is formed from the ensemble correlation;
    leave-one-trajectory-out estimates are used only for uncertainty and
    nested-window selection.
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
        available
        if maximum_lag_points is None
        else min(available, maximum_lag_points)
    )
    minimum = max(minimum_lag_points, polynomial_order + 1)
    if maximum < minimum:
        return CoherenceCarrierEstimate(error="too few lag points for local fit"), []

    sign = orientation_sign(orientation)
    candidates: list[_CandidateFit] = []
    for lag_points in range(minimum, maximum + 1):
        slope, residual = _phase_slope(correlation, dt, lag_points, polynomial_order)
        sem = _jackknife_sem(
            values, dt, lag_points, polynomial_order, sign
        )
        candidates.append(
            _CandidateFit(
                lag_points=lag_points,
                frequency=sign * slope,
                frequency_sem=sem,
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
    return (
        CoherenceCarrierEstimate(
            frequency=selected.frequency,
            frequency_sem=selected.frequency_sem,
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


def _recorded_modes(data: Any, stored: int) -> list[int]:
    meta = getattr(data, "meta", None)
    configured = meta.get("mode_indices") if isinstance(meta, dict) else None
    if configured is None:
        return list(range(stored))
    modes = [int(mode) for mode in configured]
    if len(modes) != stored:
        raise ValueError("trajectory mode_indices do not match stored columns")
    return modes


def _measurement_matrices(
    data: Any,
    stored_modes: int,
    config: CoherenceCarrierConfig,
) -> tuple[list[str], list[str], list[int], np.ndarray]:
    recorded = _recorded_modes(data, stored_modes)
    names: list[str] = []
    kinds: list[str] = []
    matrices: list[np.ndarray] = []
    for mode, column in zip(
        config.modes,
        resolve_mode_columns(data, config.modes),
        strict=True,
    ):
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

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="backend",
            requires_full_trajectory=True,
            supports_trajectory_batching=True,
            supports_time_streaming=False,
        )

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        config = cast(CoherenceCarrierConfig, self.config)
        n_measurements = len(config.modes) + len(config.channels) + int(
            config.include_trace
        )
        chunk = min(config.time_chunk_samples, request.saved_samples)
        complex_itemsize = 2 * request.real_itemsize
        chunk_bytes = (
            2
            * request.n_traj
            * chunk
            * request.n_record_modes
            * complex_itemsize
        )
        retained_bytes = (
            request.n_traj
            * n_measurements
            * (config.maximum_lag_points + 1)
            * complex_itemsize
        )
        if request.backend_name == "cupy":
            return AnalyzerWorkspaceEstimate(
                device_bytes=chunk_bytes + retained_bytes,
                host_bytes=retained_bytes,
            )
        return AnalyzerWorkspaceEstimate(host_bytes=chunk_bytes + retained_bytes)

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(CoherenceCarrierConfig, self.config)
        values = getattr(data, "data", data)
        if (
            not hasattr(values, "ndim")
            or values.ndim != 3
            or not np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ValueError(
                "coherence_carrier expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, stored_modes = map(int, values.shape)
        if n_traj < 1 or n_samples <= config.maximum_lag_points:
            raise ValueError(
                "coherence_carrier requires trajectories longer than maximum_lag_points"
            )
        dt = float(getattr(data, "dt", 1.0))
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        names, kinds, recorded_modes, matrices = _measurement_matrices(
            data, stored_modes, config
        )
        backend_matrices = backend.asarray(matrices, dtype=values.dtype)
        correlations = backend.zeros(
            (n_traj, len(names), config.maximum_lag_points + 1),
            dtype=values.dtype,
        )

        for lag in range(config.maximum_lag_points + 1):
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

        return AnalysisResult(
            data_dict=self._summarize(
                per_trajectory=np.asarray(convert_to_numpy(correlations)),
                names=names,
                kinds=kinds,
                recorded_modes=recorded_modes,
                matrices=matrices,
                dt=dt,
                n_samples=n_samples,
                t0=float(getattr(data, "t0", 0.0)),
            )
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
    ) -> dict[str, Any]:
        config = cast(CoherenceCarrierConfig, self.config)
        estimates: list[CoherenceCarrierEstimate] = []
        candidate_frequencies = []
        candidate_sems = []
        candidate_residuals = []
        for index in range(len(names)):
            estimate, candidates = estimate_coherence_carrier(
                per_trajectory[:, index, :],
                dt,
                orientation=config.orientation,
                polynomial_order=config.polynomial_order,
                minimum_lag_points=config.minimum_lag_points,
                maximum_lag_points=config.maximum_lag_points,
                consistency_sigma=config.consistency_sigma,
                intensity_floor=config.intensity_floor,
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
            "status": np.asarray([item.status for item in estimates]),
            "error": np.asarray([item.error for item in estimates]),
            "selected_lag_points": np.asarray(
                [item.selected_lag_points for item in estimates]
            ),
            "selected_lag_time": np.asarray(
                [item.selected_lag_time for item in estimates]
            ),
            "phase_fit_rms": np.asarray(
                [item.phase_fit_rms for item in estimates]
            ),
            "first_lag_coherence": np.asarray(
                [item.first_lag_coherence for item in estimates]
            ),
            "first_lag_phase": np.asarray(
                [item.first_lag_phase for item in estimates]
            ),
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
                "point_estimate_is_ratio_of_ensemble_correlations": True,
            },
            **metadata,
        }


class CoherenceCarrierResultAccumulator:
    """Merge trajectory batches before estimating the correlation ratio."""

    def __init__(self, analyzer: CoherenceCarrierAnalyzer) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any]) -> None:
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
                raise ValueError(
                    "coherence-carrier batches used different time grids"
                )
            if not np.array_equal(
                first["measurement_matrices"], payload["measurement_matrices"]
            ):
                raise ValueError(
                    "coherence-carrier batches used different readout matrices"
                )
        self.payloads.append(payload)

    def finalize(self) -> dict[str, Any]:
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
