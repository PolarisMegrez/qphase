"""Focused, trajectory-batchable Allan variance analyser."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    AxisRole,
    CoordinateSchema,
    DataKind,
    Dataset,
    SamplingBasisSchema,
    UncertaintySchema,
    VariableConstraints,
)

from ..contracts.quantities import SDEQuantity
from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .allan_statistics import (
    calculate_allan_variance,
    calculate_allan_variance_device,
    summarize_trajectories,
)
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

__all__ = ["AllanVarianceAnalyzer", "AllanVarianceConfig"]


class AllanVarianceConfig(PluginConfigBase):
    """Configuration for focused angular-frequency Allan statistics."""

    modes: list[int] = Field(..., min_length=1)
    orientation: OrientationInput = Field(
        DEFAULT_FREQUENCY_ORIENTATION,
        description=(
            "Positive-frequency phase orientation: phase_decreasing maps "
            "exp(-i*omega*t) to +omega. Input aliases: physical and fft"
        ),
        json_schema_extra=orientation_schema_extra(),
    )
    taus: list[float] | None = Field(
        None, description="Requested averaging times; None selects a logarithmic grid"
    )
    points: int = Field(40, ge=2, le=256)
    min_windows: int = Field(8, ge=1)
    min_independent_windows: int = Field(
        4,
        ge=1,
        description="Minimum nominal non-overlapping windows per trajectory",
    )
    amplitude_floor: float = Field(0.0, ge=0.0)
    transfer_chunk_samples: int = Field(
        8192,
        ge=1,
        description="Maximum saved samples copied from a device at once",
    )
    device_chunk_trajectories: int = Field(
        16,
        ge=1,
        description=(
            "Trajectories analyzed per device chunk when the input is "
            "device-resident; bounds the on-device analysis workspace"
        ),
    )

    @model_validator(mode="after")
    def validate_values(self) -> AllanVarianceConfig:
        if len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        if self.taus is not None and any(value <= 0.0 for value in self.taus):
            raise ValueError("taus must contain positive values")
        return self


class AllanVarianceAnalyzer(Analyzer):
    """Compute compact Allan statistics without retaining complex trajectories."""

    name: ClassVar[str] = "allan_variance"
    description: ClassVar[str] = (
        "Trajectory-batchable angular-frequency Allan variance statistics"
    )
    config_schema: ClassVar[type[AllanVarianceConfig]] = AllanVarianceConfig

    def capabilities(self) -> AnalyzerExecutionCapabilities:
        return AnalyzerExecutionCapabilities(
            execution_location="host",
            requires_full_trajectory=True,
            supports_trajectory_batching=True,
            supports_time_streaming=False,
        )

    def estimate_workspace(
        self, request: AnalyzerWorkspaceRequest
    ) -> AnalyzerWorkspaceEstimate:
        config = cast(AllanVarianceConfig, self.config)
        mode_bytes = request.trajectory_bytes // max(request.n_record_modes, 1)
        if request.backend_name == "cupy":
            chunk_rows = min(config.device_chunk_trajectories, request.n_traj)
            # Per-chunk device temporaries: amplitude/phase/increment buffers,
            # validity tables and the per-tau second-difference slices.
            device_bytes = (
                chunk_rows * request.saved_samples * 8 * request.real_itemsize
            )
            # Only the per-trajectory tau tables are assembled on the host.
            host_bytes = request.n_traj * 256 * 4 * 8
            return AnalyzerWorkspaceEstimate(
                device_bytes=device_bytes, host_bytes=host_bytes
            )
        return AnalyzerWorkspaceEstimate(host_bytes=5 * mode_bytes)

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(AllanVarianceConfig, self.config)
        array = getattr(data, "data", data)
        if (
            not hasattr(array, "ndim")
            or array.ndim != 3
            or not _is_complex_array(array)
        ):
            raise ValueError(
                "allan_variance expects complex shape (n_traj, n_time, n_modes)"
            )
        if array.shape[1] < 3:
            raise ValueError("allan_variance requires at least three samples")
        dt = float(getattr(data, "dt", 1.0))
        t0 = float(getattr(data, "t0", 0.0))
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        columns = resolve_mode_columns(data, config.modes)
        frequency_meta = orientation_metadata(config.orientation)
        on_device = backend.backend_name() == "cupy" and not isinstance(
            array, np.ndarray
        )
        mode_results: dict[int, dict[str, Any]] = {}
        for mode, column in zip(config.modes, columns, strict=True):
            if on_device:
                allan = calculate_allan_variance_device(
                    array[:, :, column],
                    dt,
                    taus=config.taus,
                    points=config.points,
                    min_windows=config.min_windows,
                    min_independent_windows=config.min_independent_windows,
                    amplitude_floor=config.amplitude_floor,
                    chunk_trajectories=config.device_chunk_trajectories,
                )
                phase_summary = _phase_increment_summary_device(
                    array[:, :, column],
                    dt,
                    config.amplitude_floor,
                    config.orientation,
                    config.device_chunk_trajectories,
                )
            else:
                series = _copy_mode_to_host(
                    array, column, config.transfer_chunk_samples
                )
                allan = calculate_allan_variance(
                    series,
                    dt,
                    taus=config.taus,
                    points=config.points,
                    min_windows=config.min_windows,
                    min_independent_windows=config.min_independent_windows,
                    amplitude_floor=config.amplitude_floor,
                )
                phase_summary = _phase_increment_summary(
                    series,
                    dt,
                    config.amplitude_floor,
                    config.orientation,
                )
            allan.update(frequency_meta)
            mode_results[mode] = {"allan": allan, "phase_increment": phase_summary}
        payload = {
            "modes": list(config.modes),
            "t0": t0,
            "dt": dt,
            "n_traj": int(array.shape[0]),
            "n_samples": int(array.shape[1]),
            **frequency_meta,
            "mode_results": mode_results,
        }
        return AnalysisResult(data_dict=payload, meta=dict(payload, mode_results=None))

    def create_result_accumulator(self) -> AllanResultAccumulator:
        return AllanResultAccumulator()

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int,
        label: str,
    ) -> Mapping[str, Dataset] | None:
        """Build the graph-ready typed statistics product of one payload."""
        return _build_allan_products(payload, scan_size=scan_size, label=label)


#: Trailing-axis declarations of the per-mode Allan payload leaves. The glob
#: patterns match the dotted ``mode_results.<mode>.<table>.<leaf>`` keys.
_ALLAN_DECLARED_DIMS: dict[str, tuple[str, ...]] = {
    "mode_results.*.allan.tau": ("tau",),
    "mode_results.*.allan.angular_frequency_variance": ("tau",),
    "mode_results.*.allan.angular_frequency_variance_sem": ("tau",),
    "mode_results.*.allan.trajectory_sample_count": ("tau",),
    "mode_results.*.allan.nonoverlap_angular_frequency_variance": ("tau",),
    "mode_results.*.allan.nonoverlap_angular_frequency_variance_sem": ("tau",),
    "mode_results.*.allan.nonoverlap_trajectory_sample_count": ("tau",),
    "mode_results.*.allan.nominal_independent_windows_per_trajectory": ("tau",),
    "mode_results.*.allan.total_independent_window_count": ("tau",),
    "mode_results.*.allan.per_trajectory": ("trajectory", "tau"),
    "mode_results.*.allan.valid_second_differences": ("trajectory", "tau"),
    "mode_results.*.allan.nonoverlap_per_trajectory": ("trajectory", "tau"),
    "mode_results.*.allan.nonoverlap_valid_second_differences": (
        "trajectory",
        "tau",
    ),
    "mode_results.*.phase_increment.mean_angular_frequency_per_trajectory": (
        "trajectory",
    ),
    "mode_results.*.phase_increment.max_abs_phase_step_per_trajectory": ("trajectory",),
    "mode_results.*.phase_increment.near_nyquist_fraction_per_trajectory": (
        "trajectory",
    ),
}

#: Glob patterns of the Allan variance leaves carrying the quantity and an SEM
#: uncertainty payload (``<key>_sem``). Quantities, constraints and
#: uncertainties need concrete keys, so the globs are resolved per payload.
_VARIANCE_LEAF_GLOBS = (
    "mode_results.*.allan.angular_frequency_variance",
    "mode_results.*.allan.nonoverlap_angular_frequency_variance",
)


def _variance_leaf_keys(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Enumerate the concrete Allan variance leaf keys of one stacked payload."""
    return sorted(
        key
        for key in arrays
        if any(fnmatchcase(key, pattern) for pattern in _VARIANCE_LEAF_GLOBS)
    )


def _build_allan_products(
    payload: Any,
    *,
    scan_size: int,
    label: str,
) -> dict[str, Dataset] | None:
    """Assemble the typed statistics product of one Allan analyser payload.

    Variable names keep the original dotted leaf keys (and string leaves stay
    in ``payload_meta``) so the legacy view rebuilds the exact ``analyze()``
    payload; typed ``tau``/``trajectory`` axes, the Allan quantity and the
    trajectory-sampling SEM uncertainties are layered on top.
    """
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    variance_keys = _variance_leaf_keys(leaves.arrays)
    coordinates: list[CoordinateSchema] = []
    tau_keys = sorted(
        key for key in leaves.arrays if fnmatchcase(key, "mode_results.*.allan.tau")
    )
    if tau_keys:
        leading = ("scan",) if scan_size > 1 else ()
        coordinates = [
            CoordinateSchema(
                name="tau",
                variable=tau_keys[0],
                dims=(*leading, "tau"),
                role="auxiliary" if leading else "dimension",
                units="time",
            )
        ]
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims=_ALLAN_DECLARED_DIMS,
        axis_specs={
            "tau": TypedAxisSpec("tau", AxisRole.COORDINATE, units="time"),
            "trajectory": TypedAxisSpec("trajectory", AxisRole.REALIZATION),
        },
        quantities={key: SDEQuantity.ALLAN_VARIANCE.value for key in variance_keys},
        constraints={
            key: VariableConstraints(nonnegative=True) for key in variance_keys
        },
        uncertainties=[
            UncertaintySchema(
                target=key,
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable=f"{key}_sem",
            )
            for key in variance_keys
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", source_axis="trajectory")
        ],
        coordinates=coordinates,
        attributes={"estimator": "non_overlapping_windows", "graph_ready": True},
    )
    if dataset is None:
        return None
    return {label: dataset}


def _copy_mode_to_host(array: Any, column: int, chunk_samples: int) -> np.ndarray:
    if isinstance(array, np.ndarray):
        return np.asarray(array[:, :, column])
    n_traj, n_samples = int(array.shape[0]), int(array.shape[1])
    first_stop = min(n_samples, chunk_samples)
    first = np.asarray(convert_to_numpy(array[:, :first_stop, column]))
    result = np.empty((n_traj, n_samples), dtype=first.dtype)
    result[:, :first_stop] = first
    for start in range(first_stop, n_samples, chunk_samples):
        stop = min(n_samples, start + chunk_samples)
        result[:, start:stop] = convert_to_numpy(array[:, start:stop, column])
    return result


def _is_complex_array(array: Any) -> bool:
    try:
        return bool(np.issubdtype(array.dtype, np.complexfloating))
    except TypeError:
        is_complex = getattr(array, "is_complex", None)
        return bool(is_complex()) if callable(is_complex) else False


def _masked_mean(values: np.ndarray, mask: np.ndarray, axis: int) -> np.ndarray:
    count = np.sum(mask, axis=axis)
    total = np.sum(np.where(mask, values, 0.0), axis=axis)
    result = np.full(np.shape(total), np.nan, dtype=float)
    np.divide(total, count, out=result, where=count > 0)
    return result


def _phase_increment_summary(
    series: np.ndarray,
    dt: float,
    amplitude_floor: float,
    orientation: FrequencyOrientation,
) -> dict[str, Any]:
    amplitude = np.abs(series)
    increments = np.angle(series[:, 1:] * np.conj(series[:, :-1]))
    valid = (amplitude[:, 1:] > amplitude_floor) & (amplitude[:, :-1] > amplitude_floor)
    absolute = np.abs(increments)
    return {
        "mean_angular_frequency_per_trajectory": _masked_mean(
            orientation_sign(orientation) * increments / dt, valid, axis=1
        ),
        "max_abs_phase_step_per_trajectory": np.max(
            np.where(valid, absolute, np.nan), axis=1
        ),
        "near_nyquist_fraction_per_trajectory": _masked_mean(
            (absolute >= 0.9 * np.pi).astype(float), valid, axis=1
        ),
        "near_nyquist_threshold": 0.9 * np.pi,
        **orientation_metadata(orientation),
    }


def _phase_increment_summary_device(
    series: Any,
    dt: float,
    amplitude_floor: float,
    orientation: FrequencyOrientation,
    chunk_trajectories: int,
) -> dict[str, Any]:
    """Device-resident counterpart of :func:`_phase_increment_summary`."""
    import cupy as cp

    n_traj = int(series.shape[0])
    sign = orientation_sign(orientation)
    mean_frequency = np.full(n_traj, np.nan, dtype=float)
    max_abs_step = np.full(n_traj, np.nan, dtype=float)
    near_nyquist = np.full(n_traj, np.nan, dtype=float)
    for start in range(0, n_traj, chunk_trajectories):
        stop = min(n_traj, start + chunk_trajectories)
        chunk = series[start:stop]
        amplitude = cp.abs(chunk)
        increments = cp.angle(chunk[:, 1:] * cp.conj(chunk[:, :-1]))
        valid = (amplitude[:, 1:] > amplitude_floor) & (
            amplitude[:, :-1] > amplitude_floor
        )
        absolute = cp.abs(increments)
        count = cp.sum(valid, axis=1)
        total = cp.sum(cp.where(valid, sign * increments / dt, 0.0), axis=1)
        chunk_mean = cp.where(count > 0, total / cp.maximum(count, 1), cp.nan)
        chunk_max = cp.max(cp.where(valid, absolute, cp.nan), axis=1)
        indicator = (absolute >= 0.9 * np.pi).astype(cp.float64)
        nyquist_total = cp.sum(cp.where(valid, indicator, 0.0), axis=1)
        chunk_nyquist = cp.where(
            count > 0, nyquist_total / cp.maximum(count, 1), cp.nan
        )
        mean_frequency[start:stop] = cp.asnumpy(chunk_mean)
        max_abs_step[start:stop] = cp.asnumpy(chunk_max)
        near_nyquist[start:stop] = cp.asnumpy(chunk_nyquist)
    return {
        "mean_angular_frequency_per_trajectory": mean_frequency,
        "max_abs_phase_step_per_trajectory": max_abs_step,
        "near_nyquist_fraction_per_trajectory": near_nyquist,
        "near_nyquist_threshold": 0.9 * np.pi,
        **orientation_metadata(orientation),
    }


class AllanResultAccumulator:
    """Concatenate independent trajectory-batch Allan summaries."""

    def __init__(self) -> None:
        self.payloads: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            if first["modes"] != payload["modes"]:
                raise ValueError("Allan trajectory batches use different modes")
            if first["n_samples"] != payload["n_samples"] or not math.isclose(
                first["dt"], payload["dt"]
            ):
                raise ValueError("Allan trajectory batches use different time grids")
            if first["orientation"] != payload["orientation"]:
                raise ValueError(
                    "Allan trajectory batches use different frequency orientations"
                )
            for mode in first["modes"]:
                first_tau = first["mode_results"][mode]["allan"]["tau"]
                next_tau = payload["mode_results"][mode]["allan"]["tau"]
                if not np.array_equal(first_tau, next_tau):
                    raise ValueError("Allan trajectory batches use different tau grids")
        self.payloads.append(payload)

    def finalize(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty Allan accumulator")
        first = self.payloads[0]
        mode_results: dict[int, dict[str, Any]] = {}
        for mode in first["modes"]:
            allan_payloads = [
                item["mode_results"][mode]["allan"] for item in self.payloads
            ]
            allan = dict(allan_payloads[0])
            per_trajectory = np.concatenate(
                [np.asarray(item["per_trajectory"]) for item in allan_payloads], axis=0
            )
            valid = np.concatenate(
                [
                    np.asarray(item["valid_second_differences"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            nonoverlap_per_trajectory = np.concatenate(
                [
                    np.asarray(item["nonoverlap_per_trajectory"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            nonoverlap_valid = np.concatenate(
                [
                    np.asarray(item["nonoverlap_valid_second_differences"])
                    for item in allan_payloads
                ],
                axis=0,
            )
            mean, sem, sample_count = summarize_trajectories(per_trajectory)
            nonoverlap_mean, nonoverlap_sem, nonoverlap_sample_count = (
                summarize_trajectories(nonoverlap_per_trajectory)
            )
            allan.update(
                {
                    "angular_frequency_variance": mean,
                    "angular_frequency_variance_sem": sem,
                    "per_trajectory": per_trajectory,
                    "valid_second_differences": valid,
                    "trajectory_sample_count": sample_count,
                    "nonoverlap_angular_frequency_variance": nonoverlap_mean,
                    "nonoverlap_angular_frequency_variance_sem": nonoverlap_sem,
                    "nonoverlap_per_trajectory": nonoverlap_per_trajectory,
                    "nonoverlap_valid_second_differences": nonoverlap_valid,
                    "nonoverlap_trajectory_sample_count": nonoverlap_sample_count,
                    "total_independent_window_count": np.sum(
                        nonoverlap_valid, axis=0, dtype=np.int64
                    ),
                }
            )
            phase_payloads = [
                item["mode_results"][mode]["phase_increment"] for item in self.payloads
            ]
            phase = dict(phase_payloads[0])
            for key in (
                "mean_angular_frequency_per_trajectory",
                "max_abs_phase_step_per_trajectory",
                "near_nyquist_fraction_per_trajectory",
            ):
                phase[key] = np.concatenate(
                    [np.asarray(item[key]) for item in phase_payloads], axis=0
                )
            mode_results[mode] = {"allan": allan, "phase_increment": phase}

        return {
            "modes": list(first["modes"]),
            "t0": first["t0"],
            "dt": first["dt"],
            "n_traj": sum(int(item["n_traj"]) for item in self.payloads),
            "n_samples": first["n_samples"],
            **orientation_metadata(first["orientation"]),
            "mode_results": mode_results,
        }
