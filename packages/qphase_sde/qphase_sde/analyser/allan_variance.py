"""Focused, trajectory-batchable Allan variance analyser."""

from __future__ import annotations

import math
from collections.abc import Mapping
from fnmatch import fnmatchcase
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import ConfigDict, Field, model_validator
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
    VariableConstraints,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    SDEQuantity,
    orientation_sign,
)
from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .allan_statistics import (
    calculate_allan_variance,
    calculate_allan_variance_device,
    calculate_sampled_scalar_allan,
    summarize_trajectories,
)
from .base import (
    Analyser,
    map_trajectory_product,
)
from .frequency_orientation import (
    OrientationInput,
    orientation_metadata,
    orientation_schema_extra,
    resolve_frequency_orientation,
)
from .sampled_signal import read_sampled_signal

__all__ = [
    "AllanMeasurement",
    "AllanPhaseFrequencyMeasurement",
    "AllanSampledScalarMeasurement",
    "AllanVarianceAnalyser",
    "AllanVarianceConfig",
]


class AllanPhaseFrequencyMeasurement(PluginConfigBase):
    """Phase-frequency Allan of the complex field, by physical mode index."""

    model_config = ConfigDict(extra="forbid")

    channels: list[int] = Field(
        ..., min_length=1, description="Physical mode indices for analysis"
    )
    orientation: OrientationInput = Field(
        DEFAULT_FREQUENCY_ORIENTATION,
        description=(
            "Positive-frequency phase orientation: phase_decreasing maps "
            "exp(-i*omega*t) to +omega. Input aliases: physical and fft"
        ),
        json_schema_extra=orientation_schema_extra(),
    )
    amplitude_floor: float = Field(0.0, ge=0.0)
    min_independent_windows: int = Field(
        4,
        ge=1,
        description="Minimum nominal non-overlapping windows per trajectory",
    )

    @model_validator(mode="after")
    def validate_channels(self) -> AllanPhaseFrequencyMeasurement:
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        if any(channel < 0 for channel in self.channels):
            raise ValueError("channels must be non-negative")
        return self


class AllanSampledScalarMeasurement(PluginConfigBase):
    """Sampled-scalar Allan of a real observable, by channel name."""

    model_config = ConfigDict(extra="forbid")

    channels: list[str] = Field(
        ..., min_length=1, description="Observable channel names for analysis"
    )
    min_nonoverlap_pairs: int = Field(
        4,
        ge=1,
        description=(
            "Minimum adjacent non-overlapping block-mean pairs per "
            "trajectory; adjacent pairs share one block mean, so they are "
            "nominal, not independent"
        ),
    )

    @model_validator(mode="after")
    def validate_channels(self) -> AllanSampledScalarMeasurement:
        if any(not channel.strip() for channel in self.channels):
            raise ValueError("channels must not be blank")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        return self


class AllanMeasurement(PluginConfigBase):
    """Strictly discriminated Allan measurement: exactly one variant."""

    model_config = ConfigDict(extra="forbid")

    phase_frequency: AllanPhaseFrequencyMeasurement | None = None
    sampled_scalar: AllanSampledScalarMeasurement | None = None

    @model_validator(mode="after")
    def validate_exactly_one(self) -> AllanMeasurement:
        if (self.phase_frequency is None) == (self.sampled_scalar is None):
            raise ValueError(
                "measurement requires exactly one of phase_frequency or "
                "sampled_scalar"
            )
        return self


class AllanVarianceConfig(PluginConfigBase):
    """Configuration for focused angular-frequency Allan statistics."""

    model_config = ConfigDict(extra="forbid")

    measurement: AllanMeasurement = Field(
        ..., description="Measurement model: phase_frequency or sampled_scalar"
    )
    taus: list[float] | None = Field(
        None,
        min_length=1,
        description="Requested averaging times; None selects a logarithmic grid",
    )
    points: int = Field(40, ge=2, le=256)
    min_windows: int = Field(8, ge=1)
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
        if self.taus is not None and any(value <= 0.0 for value in self.taus):
            raise ValueError("taus must contain positive values")
        return self


#: Numeric leaves of the shared per-mode Allan payload (see
#: ``allan_statistics._assemble_result``); string convention keys stay in
#: payload metadata.
_ALLAN_TABLE_LEAVES = (
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

#: Numeric leaves of the per-mode phase-increment summary.
_PHASE_INCREMENT_LEAVES = (
    "mean_angular_frequency_per_trajectory",
    "max_abs_phase_step_per_trajectory",
    "near_nyquist_fraction_per_trajectory",
    "near_nyquist_threshold",
)

#: Numeric leaves of the per-channel sampled-scalar Allan payload (see
#: ``allan_statistics._assemble_scalar_result``); string convention keys stay
#: in payload metadata. The unsuffixed leaves are the non-overlapping primary
#: estimator; the overlapping variant carries the ``overlap_`` prefix.
_SCALAR_ALLAN_TABLE_LEAVES = (
    "tau",
    "sampled_scalar_variance",
    "sampled_scalar_variance_sem",
    "per_trajectory",
    "valid_window_pairs",
    "trajectory_sample_count",
    "nominal_windows_per_trajectory",
    "total_valid_window_pairs",
    "overlap_sampled_scalar_variance",
    "overlap_sampled_scalar_variance_sem",
    "overlap_per_trajectory",
    "overlap_valid_window_pairs",
    "overlap_trajectory_sample_count",
    "overlap_total_valid_window_pairs",
    "observation_duration",
    "min_nonoverlap_pairs",
)


class AllanVarianceAnalyser(Analyser):
    """Compute compact Allan statistics without retaining complex trajectories."""

    name: ClassVar[str] = "allan_variance"
    description: ClassVar[str] = (
        "Trajectory-batchable angular-frequency Allan variance statistics"
    )
    config_schema: ClassVar[type[AllanVarianceConfig]] = AllanVarianceConfig

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(AllanVarianceConfig, self.config)
        if config.measurement.phase_frequency is not None:
            return [
                ProductRequirement(
                    name="signal",
                    kind=DataKind.TIME_SERIES,
                    quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                    value_domain="complex",
                    required_fields=["alpha", "valid_length"],
                )
            ]
        return [
            ProductRequirement(
                name="signal",
                kind=DataKind.TIME_SERIES,
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
                value_domain="real",
                required_fields=["value", "valid_length"],
            )
        ]

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
        signal = inputs.get("signal")
        if signal is None:
            raise TypeError("Allan analyser requires a 'signal' input product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            config = cast(AllanVarianceConfig, self.config)
            measurement = config.measurement
            if measurement.phase_frequency is not None:
                phase = measurement.phase_frequency
                sampled = read_sampled_signal(
                    point, variable="alpha", channels=phase.channels
                )
                payload = self._compute_phase_payload(
                    sampled.array,
                    t0=sampled.t0,
                    dt=sampled.dt,
                    modes=[int(mode) for mode in sampled.labels],
                    columns=list(sampled.columns),
                    phase=phase,
                    backend=backend,
                )
            else:
                scalar = cast(
                    AllanSampledScalarMeasurement, measurement.sampled_scalar
                )
                sampled = read_sampled_signal(
                    point, variable="value", channels=scalar.channels
                )
                payload = self._compute_scalar_payload(
                    sampled.array,
                    t0=sampled.t0,
                    dt=sampled.dt,
                    names=[str(name) for name in sampled.labels],
                    columns=list(sampled.columns),
                    scalar=scalar,
                    variable_units=sampled.variable_units,
                )
            products = _build_allan_products(payload, scan_size=None, label=label)
            if not products:
                raise ValueError("Allan analyser produced no typed product")
            return products

        return map_trajectory_product(signal, compute, context=context)

    def _compute_phase_payload(
        self,
        array: Any,
        *,
        t0: float,
        dt: float,
        modes: list[int],
        columns: list[int],
        phase: AllanPhaseFrequencyMeasurement,
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(AllanVarianceConfig, self.config)
        if array.ndim != 3 or not _is_complex_array(array):
            raise ValueError(
                "allan_variance expects complex shape (n_traj, n_time, n_modes)"
            )
        if array.shape[1] < 3:
            raise ValueError("allan_variance requires at least three samples")
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        frequency_meta = orientation_metadata(phase.orientation)
        on_device = backend.backend_name() == "cupy" and not isinstance(
            array, np.ndarray
        )
        mode_results: dict[int, dict[str, Any]] = {}
        for mode, column in zip(modes, columns, strict=True):
            if on_device:
                allan = calculate_allan_variance_device(
                    array[:, :, column],
                    dt,
                    taus=config.taus,
                    points=config.points,
                    min_windows=config.min_windows,
                    min_independent_windows=phase.min_independent_windows,
                    amplitude_floor=phase.amplitude_floor,
                    chunk_trajectories=config.device_chunk_trajectories,
                )
                phase_summary = _phase_increment_summary_device(
                    array[:, :, column],
                    dt,
                    phase.amplitude_floor,
                    phase.orientation,
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
                    min_independent_windows=phase.min_independent_windows,
                    amplitude_floor=phase.amplitude_floor,
                )
                phase_summary = _phase_increment_summary(
                    series,
                    dt,
                    phase.amplitude_floor,
                    phase.orientation,
                )
            allan.update(frequency_meta)
            mode_results[mode] = {
                "allan": allan,
                "phase_increment": phase_summary,
            }
        return {
            "modes": list(modes),
            "t0": t0,
            "dt": dt,
            "n_traj": int(array.shape[0]),
            "n_samples": int(array.shape[1]),
            "measurement_model": "phase_frequency",
            **frequency_meta,
            "mode_results": mode_results,
        }

    def _compute_scalar_payload(
        self,
        array: Any,
        *,
        t0: float,
        dt: float,
        names: list[str],
        columns: list[int],
        scalar: AllanSampledScalarMeasurement,
        variable_units: str,
    ) -> dict[str, Any]:
        config = cast(AllanVarianceConfig, self.config)
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        channel_results: dict[int, dict[str, Any]] = {}
        for index, column in enumerate(columns):
            series = _copy_mode_to_host(array, column, config.transfer_chunk_samples)
            result = calculate_sampled_scalar_allan(
                series,
                dt,
                taus=config.taus,
                points=config.points,
                min_windows=config.min_windows,
                min_nonoverlap_pairs=scalar.min_nonoverlap_pairs,
            )
            channel_results[index] = {"allan": result}
        return {
            "channels": list(range(len(names))),
            "channel_names": names,
            "t0": t0,
            "dt": dt,
            "n_traj": int(array.shape[0]),
            "n_samples": int(array.shape[1]),
            "measurement_model": "sampled_scalar",
            "value_units": variable_units,
            "channel_results": channel_results,
        }

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Declare the single graph-ready statistics product of this analyser."""
        config = cast(AllanVarianceConfig, self.config)
        phase = config.measurement.phase_frequency
        if phase is not None:
            fields = ["modes", "t0", "dt", "n_traj", "n_samples"]
            for mode in phase.channels:
                fields.extend(
                    f"mode_results.{mode}.allan.{leaf}" for leaf in _ALLAN_TABLE_LEAVES
                )
                fields.extend(
                    f"mode_results.{mode}.phase_increment.{leaf}"
                    for leaf in _PHASE_INCREMENT_LEAVES
                )
        else:
            scalar = cast(
                AllanSampledScalarMeasurement, config.measurement.sampled_scalar
            )
            fields = ["channels", "t0", "dt", "n_traj", "n_samples"]
            for index in range(len(scalar.channels)):
                fields.extend(
                    f"channel_results.{index}.allan.{leaf}"
                    for leaf in _SCALAR_ALLAN_TABLE_LEAVES
                )
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                quantity=SDEQuantity.ALLAN_VARIANCE.value,
                fields=fields,
            ),
        )

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Allan work scales with the tau grid; no total is fabricated."""
        del request
        return WorkEstimate(unit="tau", total=None)

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        config = cast(AllanVarianceConfig, self.config)
        if config.measurement.sampled_scalar is not None:
            # The scalar estimator copies each selected channel to the host
            # and runs the NumPy cumsum kernel there; a device-native variant
            # is deliberately not claimed (Phase 9).
            return AnalyserExecutionCapabilities(
                preferred_location="host",
                supports_trajectory_batching=True,
                supports_time_streaming=False,
                supports_scan_tiling=True,
            )
        return AnalyserExecutionCapabilities(
            preferred_location="backend",
            supports_trajectory_batching=True,
            supports_time_streaming=False,
            supports_scan_tiling=True,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        config = cast(AllanVarianceConfig, self.config)
        measurement = config.measurement
        n_channels = (
            len(measurement.phase_frequency.channels)
            if measurement.phase_frequency is not None
            else len(
                cast(
                    AllanSampledScalarMeasurement, measurement.sampled_scalar
                ).channels
            )
        )
        n_tau = len(config.taus) if config.taus is not None else config.points
        retained_host_bytes = (
            (request.total_n_traj or request.n_traj)
            * n_tau
            * max(n_channels, 1)
            * 8
            * 8
        )
        if measurement.sampled_scalar is not None:
            # Host estimator workspace per invocation: one full host copy of
            # the selected channel in the source dtype, the float64
            # cumulative sum, the worst-case m=1 difference temporaries
            # (non-overlap and overlap) and their boolean masks.
            host_bytes = (
                request.n_traj
                * request.saved_samples
                * (request.real_itemsize + 3 * 8 + 2)
            )
            return AnalyserWorkspaceEstimate(
                host_bytes=host_bytes,
                retained_host_bytes=retained_host_bytes,
            )
        mode_bytes = request.trajectory_bytes // max(request.n_record_modes, 1)
        if request.backend_name == "cupy":
            chunk_rows = min(config.device_chunk_trajectories, request.n_traj)
            # The phase, second differences and cumsum output are float64 even
            # for complex64 input. Include those fixed-width buffers instead
            # of scaling every temporary with the backend's real dtype.
            bytes_per_sample = 6 * request.real_itemsize + 32
            device_bytes = chunk_rows * request.saved_samples * bytes_per_sample
            # Only the per-trajectory tau tables are assembled on the host.
            host_bytes = request.n_traj * 256 * 4 * 8
            return AnalyserWorkspaceEstimate(
                device_bytes=device_bytes,
                host_bytes=host_bytes,
                retained_host_bytes=retained_host_bytes,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=5 * mode_bytes,
            retained_host_bytes=retained_host_bytes,
        )

    def create_result_accumulator(self) -> AllanResultAccumulator:
        config = cast(AllanVarianceConfig, self.config)
        measurement_model = (
            "phase_frequency"
            if config.measurement.phase_frequency is not None
            else "sampled_scalar"
        )
        return AllanResultAccumulator(measurement_model)


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
    "channel_results.*.allan.tau": ("tau",),
    "channel_results.*.allan.sampled_scalar_variance": ("tau",),
    "channel_results.*.allan.sampled_scalar_variance_sem": ("tau",),
    "channel_results.*.allan.trajectory_sample_count": ("tau",),
    "channel_results.*.allan.nominal_windows_per_trajectory": ("tau",),
    "channel_results.*.allan.total_valid_window_pairs": ("tau",),
    "channel_results.*.allan.overlap_sampled_scalar_variance": ("tau",),
    "channel_results.*.allan.overlap_sampled_scalar_variance_sem": ("tau",),
    "channel_results.*.allan.overlap_trajectory_sample_count": ("tau",),
    "channel_results.*.allan.overlap_total_valid_window_pairs": ("tau",),
    "channel_results.*.allan.per_trajectory": ("trajectory", "tau"),
    "channel_results.*.allan.valid_window_pairs": ("trajectory", "tau"),
    "channel_results.*.allan.overlap_per_trajectory": ("trajectory", "tau"),
    "channel_results.*.allan.overlap_valid_window_pairs": (
        "trajectory",
        "tau",
    ),
}

#: Glob patterns of the Allan variance leaves carrying the quantity and an SEM
#: uncertainty payload (``<key>_sem``). Quantities, constraints and
#: uncertainties need concrete keys, so the globs are resolved per payload.
_VARIANCE_LEAF_GLOBS = (
    "mode_results.*.allan.angular_frequency_variance",
    "mode_results.*.allan.nonoverlap_angular_frequency_variance",
    "channel_results.*.allan.sampled_scalar_variance",
    "channel_results.*.allan.overlap_sampled_scalar_variance",
)

#: Leaves sharing the variance units of their estimator family: the target,
#: its SEM and the per-trajectory estimates have one physical dimension.
#: Count and window leaves stay dimensionless (no units declared).
_PHASE_VARIANCE_UNIT_GLOBS = (
    "mode_results.*.allan.angular_frequency_variance",
    "mode_results.*.allan.angular_frequency_variance_sem",
    "mode_results.*.allan.nonoverlap_angular_frequency_variance",
    "mode_results.*.allan.nonoverlap_angular_frequency_variance_sem",
    "mode_results.*.allan.per_trajectory",
    "mode_results.*.allan.nonoverlap_per_trajectory",
)
_PHASE_FREQUENCY_UNIT_GLOBS = (
    "mode_results.*.phase_increment.mean_angular_frequency_per_trajectory",
)
_SCALAR_VARIANCE_UNIT_GLOBS = (
    "channel_results.*.allan.sampled_scalar_variance",
    "channel_results.*.allan.sampled_scalar_variance_sem",
    "channel_results.*.allan.overlap_sampled_scalar_variance",
    "channel_results.*.allan.overlap_sampled_scalar_variance_sem",
    "channel_results.*.allan.per_trajectory",
    "channel_results.*.allan.overlap_per_trajectory",
)

#: Units of the phase-frequency Allan variance: squared angular frequency.
_PHASE_VARIANCE_UNITS = "inverse_time^2"
#: Units of the per-trajectory mean angular frequency leaves.
_PHASE_FREQUENCY_UNITS = "inverse_time"


def _leaf_units(
    arrays: Mapping[str, np.ndarray], globs: tuple[str, ...], units: str
) -> dict[str, str]:
    """Assign one units string to every concrete leaf matching the globs."""
    return {
        key: units
        for key in arrays
        if any(fnmatchcase(key, pattern) for pattern in globs)
    }


def _variance_leaf_keys(arrays: Mapping[str, np.ndarray]) -> list[str]:
    """Enumerate the concrete Allan variance leaf keys of one stacked payload."""
    return sorted(
        key
        for key in arrays
        if any(fnmatchcase(key, pattern) for pattern in _VARIANCE_LEAF_GLOBS)
    )


def _first_point_meta(meta: Mapping[str, Any], key: str) -> Any:
    """Return one job-homogeneous metadata value from the first scan point."""
    value = meta.get(key)
    if isinstance(value, list) and value and isinstance(value[0], list | dict):
        return value[0]
    return value


def _first_point_str(meta: Mapping[str, Any], key: str) -> str | None:
    """Return one job-homogeneous scalar string from the first scan point."""
    value = meta.get(key)
    if isinstance(value, list):
        value = value[0] if value else None
    return value if isinstance(value, str) else None


def _build_allan_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset] | None:
    """Assemble the typed statistics product of one Allan analyser payload.

    Variable names keep stable dotted leaf keys and string leaves stay in
    ``payload_meta``. Typed ``tau``/``trajectory`` axes, the Allan quantity and
    trajectory-sampling SEM uncertainties are layered on top. The phase and
    sampled-scalar measurement models share the tau grid, trajectory basis and
    assembly, but record disjoint table namespaces and attributes.
    """
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    point_payloads = payload if isinstance(payload, list) else [payload]
    models = {
        point.get("measurement_model")
        for point in point_payloads
        if isinstance(point, Mapping)
    }
    if len(models) != 1:
        raise ValueError("Allan payload must declare exactly one measurement_model")
    measurement_model = models.pop()
    variance_keys = _variance_leaf_keys(leaves.arrays)
    coordinates: list[CoordinateSchema] = []
    variable_units: dict[str, str] = {}
    if measurement_model == "phase_frequency":
        if not point_payloads or not all(
            isinstance(point, Mapping) and "orientation" in point
            for point in point_payloads
        ):
            raise ValueError("Allan payload must declare its frequency orientation")
        orientations = {
            resolve_frequency_orientation({"orientation": point["orientation"]})
            for point in point_payloads
        }
        if len(orientations) != 1:
            raise ValueError("Allan scan points use different frequency orientations")
        orientation = orientations.pop()
        variable_units = {
            **_leaf_units(
                leaves.arrays, _PHASE_VARIANCE_UNIT_GLOBS, _PHASE_VARIANCE_UNITS
            ),
            **_leaf_units(
                leaves.arrays, _PHASE_FREQUENCY_UNIT_GLOBS, _PHASE_FREQUENCY_UNITS
            ),
        }
        tau_glob = "mode_results.*.allan.tau"
        attributes: dict[str, Any] = {
            "estimator": "non_overlapping_windows",
            "graph_ready": True,
            "orientation": orientation,
            "measurement_model": "phase_frequency",
        }
    elif measurement_model == "sampled_scalar":
        channel_names = _first_point_meta(leaves.payload_meta, "channel_names")
        if not isinstance(channel_names, list):
            raise ValueError(
                "sampled-scalar Allan payload must record its channel_names"
            )
        value_units = _first_point_str(leaves.payload_meta, "value_units")
        if value_units is None:
            raise ValueError(
                "sampled-scalar Allan payload must record its value_units"
            )
        variance_units = "1" if value_units in ("", "1") else f"{value_units}^2"
        variable_units = _leaf_units(
            leaves.arrays, _SCALAR_VARIANCE_UNIT_GLOBS, variance_units
        )
        tau_glob = "channel_results.*.allan.tau"
        attributes = {
            "estimator": "nonoverlapping_window_mean_difference",
            "graph_ready": True,
            "measurement_model": "sampled_scalar",
            "channel_names": [str(name) for name in channel_names],
        }
    else:
        raise ValueError(f"unsupported Allan measurement_model {measurement_model!r}")
    tau_keys = sorted(key for key in leaves.arrays if fnmatchcase(key, tau_glob))
    if tau_keys:
        leading = ("scan",) if scan_size is not None else ()
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
        variable_units=variable_units,
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
        attributes=attributes,
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
    """Concatenate independent trajectory-batch Allan summaries.

    One accumulator serves exactly one measurement model: the phase-frequency
    path merges per-mode tables keyed by physical mode, the sampled-scalar
    path merges per-channel tables keyed by channel position; their sampling
    bases and uncertainty payloads stay independent.
    """

    def __init__(self, measurement_model: str = "phase_frequency") -> None:
        if measurement_model not in {"phase_frequency", "sampled_scalar"}:
            raise ValueError(
                f"unsupported Allan measurement_model {measurement_model!r}"
            )
        self.measurement_model = measurement_model
        self.payloads: list[dict[str, Any]] = []

    def _append_payload(self, payload: dict[str, Any]) -> None:
        model = payload.get("measurement_model")
        if model != self.measurement_model:
            raise ValueError(
                f"Allan accumulator of measurement {self.measurement_model!r} "
                f"received a {model!r} payload"
            )
        payload = dict(payload)
        if self.measurement_model == "phase_frequency":
            modes = [int(mode) for mode in np.asarray(payload["modes"]).reshape(-1)]
            payload["modes"] = modes
        else:
            channels = [
                int(channel)
                for channel in np.asarray(payload["channels"]).reshape(-1)
            ]
            payload["channels"] = channels
        if self.payloads:
            first = self.payloads[0]
            if int(first["n_samples"]) != int(payload["n_samples"]) or not math.isclose(
                float(first["dt"]), float(payload["dt"])
            ):
                raise ValueError("Allan trajectory batches use different time grids")
            if self.measurement_model == "phase_frequency":
                if not np.array_equal(first["modes"], payload["modes"]):
                    raise ValueError("Allan trajectory batches use different modes")
                if first["orientation"] != payload["orientation"]:
                    raise ValueError(
                        "Allan trajectory batches use different frequency orientations"
                    )
                first_tables = first["mode_results"]
                next_tables = payload["mode_results"]
                keys = first["modes"]
            else:
                if not np.array_equal(first["channels"], payload["channels"]):
                    raise ValueError("Allan trajectory batches use different channels")
                if first["channel_names"] != payload["channel_names"]:
                    raise ValueError(
                        "Allan trajectory batches use different channel names"
                    )
                first_tables = first["channel_results"]
                next_tables = payload["channel_results"]
                keys = first["channels"]
            for key in keys:
                first_tau = first_tables[key]["allan"]["tau"]
                next_tau = next_tables[key]["allan"]["tau"]
                if not np.array_equal(first_tau, next_tau):
                    raise ValueError("Allan trajectory batches use different tau grids")
        self.payloads.append(payload)

    def consume_product(self, product: Dataset) -> None:
        """Consume one typed Allan product by its declared domain fields."""
        names = {variable.name for variable in product.variables}
        measurement_model = product.attributes.get("measurement_model")
        if not isinstance(measurement_model, str):
            raise TypeError(
                "Allan product does not declare a measurement_model attribute"
            )
        base = {
            "t0": float(product.handle("t0").materialize()),
            "dt": float(product.handle("dt").materialize()),
            "n_traj": int(product.handle("n_traj").materialize()),
            "n_samples": int(product.handle("n_samples").materialize()),
            "measurement_model": measurement_model,
        }
        if measurement_model == "phase_frequency":
            raw_modes = np.asarray(product.handle("modes").materialize()).reshape(-1)
            modes = [int(mode) for mode in raw_modes]
            mode_results: dict[int, dict[str, Any]] = {}
            for mode in modes:
                allan_prefix = f"mode_results.{mode}.allan"
                phase_prefix = f"mode_results.{mode}.phase_increment"
                allan = {
                    leaf: product.handle(f"{allan_prefix}.{leaf}").materialize()
                    for leaf in _ALLAN_TABLE_LEAVES
                    if f"{allan_prefix}.{leaf}" in names
                }
                phase = {
                    leaf: product.handle(f"{phase_prefix}.{leaf}").materialize()
                    for leaf in _PHASE_INCREMENT_LEAVES
                    if f"{phase_prefix}.{leaf}" in names
                }
                mode_results[mode] = {"allan": allan, "phase_increment": phase}
            payload_meta = product.attributes.get("payload_meta", {})
            orientation = product.attributes.get("orientation")
            if orientation is None and isinstance(payload_meta, Mapping):
                orientation = payload_meta.get("orientation")
            self._append_payload(
                {
                    **base,
                    "modes": modes,
                    "orientation": orientation,
                    "mode_results": mode_results,
                }
            )
            return
        channel_names = product.attributes.get("channel_names")
        if not isinstance(channel_names, list):
            raise TypeError(
                "sampled-scalar Allan product is missing a channel_names attribute"
            )
        payload_meta = product.attributes.get("payload_meta", {})
        value_units = (
            _first_point_str(payload_meta, "value_units")
            if isinstance(payload_meta, Mapping)
            else None
        )
        raw_channels = np.asarray(product.handle("channels").materialize()).reshape(-1)
        channels = [int(channel) for channel in raw_channels]
        channel_results: dict[int, dict[str, Any]] = {}
        for channel in channels:
            allan_prefix = f"channel_results.{channel}.allan"
            allan = {
                leaf: product.handle(f"{allan_prefix}.{leaf}").materialize()
                for leaf in _SCALAR_ALLAN_TABLE_LEAVES
                if f"{allan_prefix}.{leaf}" in names
            }
            channel_results[channel] = {"allan": allan}
        self._append_payload(
            {
                **base,
                "channels": channels,
                "channel_names": [str(name) for name in channel_names],
                "value_units": value_units,
                "channel_results": channel_results,
            }
        )

    def merge(self, other: AllanResultAccumulator) -> None:
        for payload in other.payloads:
            self._append_payload(payload)

    def finalize_product(self, label: str) -> Mapping[str, Dataset] | None:
        return _build_allan_products(
            self._finalized_payload(), scan_size=None, label=label
        )

    def checkpoint(self) -> dict[str, Any]:
        return {"payloads": list(self.payloads)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        payloads = checkpoint.get("payloads")
        if not isinstance(payloads, list):
            raise TypeError("Allan reducer checkpoint is missing payloads")
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("Allan reducer checkpoint payload must be a mapping")
            self._append_payload(payload)

    def _finalized_payload(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty Allan accumulator")
        if self.measurement_model == "sampled_scalar":
            return self._finalized_scalar_payload()
        return self._finalized_phase_payload()

    def _finalized_scalar_payload(self) -> dict[str, Any]:
        first = self.payloads[0]
        channel_results: dict[int, dict[str, Any]] = {}
        for channel in first["channels"]:
            tables = [
                item["channel_results"][channel]["allan"] for item in self.payloads
            ]
            table = dict(tables[0])
            per_trajectory = np.concatenate(
                [np.asarray(item["per_trajectory"]) for item in tables], axis=0
            )
            valid = np.concatenate(
                [np.asarray(item["valid_window_pairs"]) for item in tables], axis=0
            )
            overlap_per_trajectory = np.concatenate(
                [np.asarray(item["overlap_per_trajectory"]) for item in tables], axis=0
            )
            overlap_valid = np.concatenate(
                [np.asarray(item["overlap_valid_window_pairs"]) for item in tables],
                axis=0,
            )
            mean, sem, sample_count = summarize_trajectories(per_trajectory)
            overlap_mean, overlap_sem, overlap_sample_count = summarize_trajectories(
                overlap_per_trajectory
            )
            table.update(
                {
                    "sampled_scalar_variance": mean,
                    "sampled_scalar_variance_sem": sem,
                    "per_trajectory": per_trajectory,
                    "valid_window_pairs": valid,
                    "trajectory_sample_count": sample_count,
                    "total_valid_window_pairs": np.sum(valid, axis=0, dtype=np.int64),
                    "overlap_sampled_scalar_variance": overlap_mean,
                    "overlap_sampled_scalar_variance_sem": overlap_sem,
                    "overlap_per_trajectory": overlap_per_trajectory,
                    "overlap_valid_window_pairs": overlap_valid,
                    "overlap_trajectory_sample_count": overlap_sample_count,
                    "overlap_total_valid_window_pairs": np.sum(
                        overlap_valid, axis=0, dtype=np.int64
                    ),
                }
            )
            channel_results[channel] = {"allan": table}
        return {
            "channels": list(first["channels"]),
            "channel_names": list(first["channel_names"]),
            "t0": first["t0"],
            "dt": first["dt"],
            "n_traj": sum(int(item["n_traj"]) for item in self.payloads),
            "n_samples": first["n_samples"],
            "measurement_model": "sampled_scalar",
            "value_units": first.get("value_units"),
            "channel_results": channel_results,
        }

    def _finalized_phase_payload(self) -> dict[str, Any]:
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
            "measurement_model": "phase_frequency",
            **orientation_metadata(first["orientation"]),
            "mode_results": mode_results,
        }
