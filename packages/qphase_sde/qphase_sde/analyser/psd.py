"""qphase_sde: Power Spectral Density
---------------------------------------------------------
Compute power spectral density (PSD) from multi-trajectory time series using
FFT-based estimators.

Measurements
------------
- ``complex_field``: PSD of the complex field amplitude
  (``representation='complex'``) or of its magnitude
  (``representation='magnitude'``), selected by physical mode index. The
  phase-signed frequency orientation applies to this measurement only.
- ``sampled_scalar``: one-sided PSD of a real-valued sampled observable,
  selected by channel name; the non-negative axis is expressed by the
  ``unsigned`` orientation and ``one_sided`` sidedness attributes.

Behavior
--------
- Provide common PSD conventions: unitary/symmetric (angular frequency) and
    pragmatic (frequency f). Exact scaling, return shapes, and error semantics
    are specified by the function docstrings.

Public API
----------
``PsdAnalyzer`` : Power spectral density analyzer.
``PsdAnalyzerConfig`` : Configuration for PSD analyzer.

Notes
-----
- These utilities are backend-agnostic with NumPy implementations and are used
    by visualizer as well as analysis pipelines.

"""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as _np
from pydantic import ConfigDict, Field, model_validator
from qphase.backend.base import BackendBase
from qphase.core.protocols import (
    PluginConfigBase,
    PluginManifest,
    SubpluginSlot,
)
from qphase.data import (
    AxisRole,
    CoordinateSchema,
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SamplingBasisSchema,
    SpectralQuantity,
    UncertaintySchema,
    VariableConstraints,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..contracts.quantities import SDEQuantity
from ..math.spectral import (
    detrend_series,
    fold_one_sided_powers,
    one_sided_frequency_axis,
    sampled_scalar_psd_units,
)
from ..products import (
    TypedAxisSpec,
    assemble_typed_product,
    stack_payload_leaves,
)
from ..spectral_estimator import (
    PsdEstimate as _PsdEstimate,
)
from ..spectral_estimator import (
    SpectralEstimator,
    create_builtin_estimator,
)
from .base import (
    Analyzer,
    map_trajectory_product,
)
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    OrientationInput,
    orient_spectrum,
    orientation_metadata,
    orientation_schema_extra,
)
from .sampled_signal import read_sampled_signal

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
    "PsdComplexFieldMeasurement",
    "PsdMeasurement",
    "PsdSampledScalarMeasurement",
]


class PsdComplexFieldMeasurement(PluginConfigBase):
    """PSD of the complex field, selected by physical mode index."""

    model_config = ConfigDict(extra="forbid")

    channels: list[int] = Field(
        ..., min_length=1, description="Physical mode indices for analysis"
    )
    representation: Literal["complex", "magnitude"] = Field(
        ..., description="FFT of the complex amplitude or of |alpha|"
    )
    orientation: OrientationInput = Field(
        DEFAULT_FREQUENCY_ORIENTATION,
        description=(
            "Positive-frequency phase orientation: phase_decreasing maps "
            "exp(-i*omega*t) to +omega; phase_increasing preserves the "
            "forward-FFT convention. Input aliases: physical and fft"
        ),
        json_schema_extra=orientation_schema_extra(),
    )

    @model_validator(mode="after")
    def validate_channels(self) -> "PsdComplexFieldMeasurement":
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        if any(channel < 0 for channel in self.channels):
            raise ValueError("channels must be non-negative")
        return self


class PsdSampledScalarMeasurement(PluginConfigBase):
    """One-sided PSD of a real sampled observable, selected by channel name."""

    model_config = ConfigDict(extra="forbid")

    channels: list[str] = Field(
        ..., min_length=1, description="Observable channel names for analysis"
    )
    sidedness: Literal["one_sided"] = Field(
        "one_sided",
        description="Sampled-scalar spectra are one-sided densities",
    )
    detrend: Literal["none", "constant"] = Field(
        "none",
        description="Detrend applied before the estimator; never inferred",
    )

    @model_validator(mode="after")
    def validate_channels(self) -> "PsdSampledScalarMeasurement":
        if any(not channel.strip() for channel in self.channels):
            raise ValueError("channels must not be blank")
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels must not contain duplicates")
        return self


class PsdMeasurement(PluginConfigBase):
    """Strictly discriminated PSD measurement: exactly one variant."""

    model_config = ConfigDict(extra="forbid")

    complex_field: PsdComplexFieldMeasurement | None = None
    sampled_scalar: PsdSampledScalarMeasurement | None = None

    @model_validator(mode="after")
    def validate_exactly_one(self) -> "PsdMeasurement":
        if (self.complex_field is None) == (self.sampled_scalar is None):
            raise ValueError(
                "measurement requires exactly one of complex_field or "
                "sampled_scalar"
            )
        return self


class PsdAnalyzerConfig(PluginConfigBase):
    """Configuration for PSD Analyzer."""

    model_config = ConfigDict(extra="forbid")

    measurement: PsdMeasurement = Field(
        ..., description="Measurement model: complex_field or sampled_scalar"
    )
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="PSD convention"
    )
    dt: float | None = Field(None, description="Sampling interval (override)")
    expected_freq_max: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Optional maximum frequency magnitude expected in the output-axis "
            "units; analysis fails if it reaches the Nyquist limit"
        ),
    )


class PsdAnalyzer(Analyzer):
    """Analyzer for Power Spectral Density."""

    name: ClassVar[str] = "psd"
    description: ClassVar[str] = "Power Spectral Density analyzer"
    config_schema: ClassVar[type[PsdAnalyzerConfig]] = PsdAnalyzerConfig
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "estimator": SubpluginSlot(
                namespace="spectral_estimator",
                default="periodogram",
                protocol="qphase_sde.spectral_estimator.base:SpectralEstimator",
                description="Algorithm used to estimate the PSD",
            )
        }
    )

    def __init__(
        self,
        config: PsdAnalyzerConfig | None = None,
        *,
        subplugins: dict[str, Any] | None = None,
        **kwargs,
    ):
        super().__init__(config, **kwargs)
        if subplugins is not None:
            self.estimator: SpectralEstimator = subplugins["estimator"]
        else:
            self.estimator = create_builtin_estimator("periodogram", {})

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(PsdAnalyzerConfig, self.config)
        if config.measurement.complex_field is not None:
            return [
                ProductRequirement(
                    name="signal",
                    kind=DataKind.TIME_SERIES,
                    quantity=SDEQuantity.FIELD_AMPLITUDE.value,
                    value_domain="complex",
                    required_fields=["alpha"],
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
            raise TypeError("PSD analyser requires a 'signal' input product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            config = cast(PsdAnalyzerConfig, self.config)
            measurement = config.measurement
            if measurement.complex_field is not None:
                field = measurement.complex_field
                sampled = read_sampled_signal(
                    point, variable="alpha", channels=field.channels
                )
                dt = config.dt if config.dt is not None else sampled.dt
                payload = self._compute_field_payload(
                    sampled.array,
                    dt=float(dt),
                    modes=[int(mode) for mode in sampled.labels],
                    columns=list(sampled.columns),
                    representation=field.representation,
                    orientation=field.orientation,
                    backend=backend,
                )
            else:
                scalar = cast(
                    PsdSampledScalarMeasurement, measurement.sampled_scalar
                )
                sampled = read_sampled_signal(
                    point, variable="value", channels=scalar.channels
                )
                dt = config.dt if config.dt is not None else sampled.dt
                payload = self._compute_scalar_payload(
                    sampled.array,
                    dt=float(dt),
                    names=[str(name) for name in sampled.labels],
                    columns=list(sampled.columns),
                    detrend=scalar.detrend,
                    variable_units=sampled.variable_units,
                    backend=backend,
                )
            products = _build_psd_products(
                payload, scan_size=None, label=label, analyser=self
            )
            if not products:
                raise ValueError("PSD analyser produced no typed product")
            return products

        return map_trajectory_product(signal, compute, context=context)

    def _nyquist_guard(self, dt: float) -> float:
        config = cast(PsdAnalyzerConfig, self.config)
        convention = config.convention
        if dt <= 0.0:
            raise ValueError("PSD sampling interval must be positive")
        nyquist = _np.pi / dt if convention in ("symmetric", "unitary") else 0.5 / dt
        if config.expected_freq_max is not None and config.expected_freq_max >= nyquist:
            raise ValueError(
                f"expected_freq_max={config.expected_freq_max:.6g} reaches or "
                f"exceeds the PSD Nyquist limit {nyquist:.6g} for sample dt={dt:.6g}; "
                "reduce save_stride or the PSD dt override"
            )
        return nyquist

    def _compute_field_payload(
        self,
        data: Any,
        *,
        dt: float,
        modes: list[int],
        columns: list[int],
        representation: Literal["complex", "magnitude"],
        orientation: FrequencyOrientation,
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(PsdAnalyzerConfig, self.config)
        convention = config.convention
        nyquist = self._nyquist_guard(dt)
        estimate0 = self._estimate_single(
            data[:, :, columns[0]],
            dt,
            representation=representation,
            convention=convention,
            orientation=orientation,
            backend=backend,
        )
        axis = estimate0.axis
        output_shape = (int(axis.shape[0]), len(columns))
        mean = _np.empty(output_shape, dtype=estimate0.mean.dtype)
        std = _np.empty(output_shape, dtype=estimate0.std.dtype)
        sem = _np.empty(output_shape, dtype=estimate0.sem.dtype)
        mean[:, 0], std[:, 0], sem[:, 0] = (
            estimate0.mean,
            estimate0.std,
            estimate0.sem,
        )
        n_independent = estimate0.n_independent
        for index, column in enumerate(columns[1:], start=1):
            estimate = self._estimate_single(
                data[:, :, column],
                dt,
                representation=representation,
                convention=convention,
                orientation=orientation,
                backend=backend,
            )
            mean[:, index], std[:, index], sem[:, index] = (
                estimate.mean,
                estimate.std,
                estimate.sem,
            )
        return {
            "axis": axis,
            "psd": mean,
            "psd_std": std,
            "psd_sem": sem,
            "modes": modes,
            "measurement_model": "complex_field",
            "representation": representation,
            "convention": convention,
            **orientation_metadata(orientation),
            "estimator": self.estimator.name,
            "sample_dt": dt,
            "nyquist": nyquist,
            "uncertainty": {
                "kind": "standard_error",
                "field": "psd_sem",
                "std_field": "psd_std",
                "independent_unit": "trajectory",
                "n_independent": n_independent,
                "ddof": 1,
                "available": n_independent > 1,
            },
        }

    def _compute_scalar_payload(
        self,
        data: Any,
        *,
        dt: float,
        names: list[str],
        columns: list[int],
        detrend: Literal["none", "constant"],
        variable_units: str,
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(PsdAnalyzerConfig, self.config)
        convention = config.convention
        nyquist = self._nyquist_guard(dt)
        estimate0 = self._estimate_scalar_single(
            data[:, :, columns[0]],
            dt,
            detrend=detrend,
            convention=convention,
            backend=backend,
        )
        axis = estimate0.axis
        output_shape = (int(axis.shape[0]), len(columns))
        mean = _np.empty(output_shape, dtype=estimate0.mean.dtype)
        std = _np.empty(output_shape, dtype=estimate0.std.dtype)
        sem = _np.empty(output_shape, dtype=estimate0.sem.dtype)
        mean[:, 0], std[:, 0], sem[:, 0] = (
            estimate0.mean,
            estimate0.std,
            estimate0.sem,
        )
        n_independent = estimate0.n_independent
        for index, column in enumerate(columns[1:], start=1):
            estimate = self._estimate_scalar_single(
                data[:, :, column],
                dt,
                detrend=detrend,
                convention=convention,
                backend=backend,
            )
            mean[:, index], std[:, index], sem[:, index] = (
                estimate.mean,
                estimate.std,
                estimate.sem,
            )
        return {
            "axis": axis,
            "psd": mean,
            "psd_std": std,
            "psd_sem": sem,
            "channels": list(range(len(names))),
            "channel_names": names,
            "measurement_model": "sampled_scalar",
            "orientation": "unsigned",
            "sidedness": "one_sided",
            "detrend": detrend,
            "convention": convention,
            "psd_units": sampled_scalar_psd_units(
                variable_units, convention=convention
            ),
            "estimator": self.estimator.name,
            "sample_dt": dt,
            "nyquist": nyquist,
            "uncertainty": {
                "kind": "standard_error",
                "field": "psd_sem",
                "std_field": "psd_std",
                "independent_unit": "trajectory",
                "n_independent": n_independent,
                "ddof": 1,
                "available": n_independent > 1,
            },
        }

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Declare the graph-ready spectrum product."""
        config = cast(PsdAnalyzerConfig, self.config)
        fields = (
            list(_PSD_PRODUCT_FIELDS)
            if config.measurement.complex_field is not None
            else list(_SCALAR_PSD_PRODUCT_FIELDS)
        )
        declarations = [
            ProductDeclaration(
                name=label,
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                fields=fields,
            )
        ]
        return tuple(declarations)

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """PSD work is FFT-dominated; no total is fabricated."""
        del request
        return WorkEstimate(unit="transform", total=None)

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        """Expose the selected estimator's real execution contract."""
        capabilities = self.estimator.capabilities()
        return AnalyserExecutionCapabilities(
            preferred_location=("backend" if capabilities.backend_native else "host"),
            supports_trajectory_batching=capabilities.trajectory_batching,
            supports_time_streaming=capabilities.time_streaming,
            supports_scan_tiling=capabilities.trajectory_batching,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate FFT or host-estimator workspace for one scan point."""
        capabilities = self.estimator.capabilities()
        trajectory_bytes = request.trajectory_bytes
        retained_host_bytes = (
            request.saved_samples
            * request.n_record_modes
            * request.real_itemsize
            * 5
        )
        if not capabilities.backend_native:
            # Host estimators materialize the input and retain at least one
            # trajectory-spectrum sized temporary while the backend source is
            # still alive.
            host_bytes = 2 * trajectory_bytes
            return AnalyserWorkspaceEstimate(
                host_bytes=host_bytes,
                retained_host_bytes=retained_host_bytes,
            )

        chunk = getattr(
            getattr(self.estimator, "config", None),
            "fft_chunk_trajectories",
            None,
        )
        if chunk is not None and 0 < int(chunk) < request.n_traj:
            workspace = max(
                1,
                2 * trajectory_bytes * int(chunk) // request.n_traj,
            )
        else:
            workspace = 2 * trajectory_bytes
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(
                device_bytes=workspace,
                retained_host_bytes=retained_host_bytes,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=workspace,
            retained_host_bytes=retained_host_bytes,
        )

    def create_result_accumulator(self) -> "PsdResultAccumulator":
        """Create an exact cross-trajectory PSD statistics accumulator."""
        return PsdResultAccumulator(self)

    def _estimate_single(
        self,
        x: Any,
        dt: float,
        *,
        representation: str = "complex",
        convention: str = "symmetric",
        orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
        backend: BackendBase,
    ) -> _PsdEstimate:
        """Compute PSD and cross-trajectory uncertainty for a single mode."""
        # Convert input to backend array
        x_arr = backend.asarray(x)

        if representation == "magnitude":
            x_proc = backend.abs(x_arr)
        else:
            x_proc = x_arr

        # Ensure 2D: (n_traj, n_time)
        ndim = getattr(x_proc, "ndim", None)
        if ndim == 1:
            # Try to add dimension using slicing
            try:
                x_proc = x_proc[None, :]
            except Exception:
                # If slicing fails, we might be dealing with a backend
                # that doesn't support it
                # But standard backends (numpy, torch, cupy) do.
                pass
        elif ndim is None or ndim < 1:
            raise ValueError("[524] input `x` must be a 1-D or 2-D array")

        estimate = self.estimator.estimate(x_proc, dt, convention, backend)
        axis, mean, std, sem = orient_spectrum(
            estimate.axis,
            estimate.mean,
            estimate.std,
            estimate.sem,
            orientation=orientation,
        )
        return _PsdEstimate(
            axis=axis,
            mean=mean,
            std=std,
            sem=sem,
            n_independent=estimate.n_independent,
        )

    def _estimate_scalar_single(
        self,
        x: Any,
        dt: float,
        *,
        detrend: Literal["none", "constant"],
        convention: Literal["symmetric", "unitary", "pragmatic"],
        backend: BackendBase,
    ) -> _PsdEstimate:
        """One-sided PSD of one real sampled-observable channel.

        The estimator child produces the full two-sided spectrum in shifted
        order; folding is a per-bin reweighting that commutes with the
        cross-trajectory mean/std/SEM, so the strict one-sided reduction of
        ``math.spectral`` applies to the reduced estimate directly.
        """
        x_proc = detrend_series(backend.asarray(x), detrend)
        estimate = self.estimator.estimate(x_proc, dt, convention, backend)
        n_fft = int(_np.asarray(estimate.mean).shape[-1])
        mean = fold_one_sided_powers(_np.fft.ifftshift(_np.asarray(estimate.mean)))
        std = fold_one_sided_powers(_np.fft.ifftshift(_np.asarray(estimate.std)))
        sem = fold_one_sided_powers(_np.fft.ifftshift(_np.asarray(estimate.sem)))
        axis = one_sided_frequency_axis(n_fft, dt, convention=convention)
        return _PsdEstimate(
            axis=axis,
            mean=mean,
            std=std,
            sem=sem,
            n_independent=estimate.n_independent,
        )

#: Numeric payload leaves; string leaves stay in payload metadata.
_PSD_PRODUCT_FIELDS = (
    "axis",
    "psd",
    "modes",
    "sample_dt",
    "nyquist",
)

#: Numeric payload leaves of the sampled-scalar measurement.
_SCALAR_PSD_PRODUCT_FIELDS = (
    "axis",
    "psd",
    "channels",
    "sample_dt",
    "nyquist",
)


def _meta_scalar(meta: Mapping[str, Any], key: str, default: str) -> str:
    """Return one job-homogeneous metadata value as a scalar string.

    Scan jobs store per-point metadata lists; the attribute is homogeneous
    across the job, so the first point is authoritative.
    """
    value = meta.get(key, default)
    if isinstance(value, list):
        value = value[0] if value else default
    return default if value is None else str(value)


def _meta_list(meta: Mapping[str, Any], key: str) -> list[str]:
    """Return one job-homogeneous list metadata value from the first point."""
    value = meta.get(key)
    if isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


#: Default window identifiers of the built-in estimators, consulted only when
#: the estimator child does not expose an explicit ``window`` configuration.
_ESTIMATOR_DEFAULT_WINDOWS = {
    "periodogram": "rectangular",
    "welch": "hann",
    "multitaper": "dpss",
}


def _estimator_window(analyser: PsdAnalyzer) -> str:
    """Return the true window identifier of the analyser's estimator child."""
    estimator = analyser.estimator
    configured = getattr(getattr(estimator, "config", None), "window", None)
    if configured:
        return str(configured)
    # Fallback: report the built-in default window of the selected estimator
    # (welch -> hann, multitaper -> dpss, everything else -> rectangular).
    return _ESTIMATOR_DEFAULT_WINDOWS.get(estimator.name, "rectangular")


def _build_psd_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
    analyser: PsdAnalyzer,
) -> dict[str, Dataset] | None:
    """Assemble the typed spectral product of one PSD analyser payload.

    Variable names keep stable dotted leaf keys while string or ragged leaves
    stay in ``payload_meta``. Typed axes, the PSD quantity and the
    trajectory-sampling uncertainties are layered on top.

    ``scan_size`` is ``None`` for a job without a parameter grid, otherwise
    the exact scan size — a 1-point scan keeps a size-1 scan axis and its
    payload arrives as the bare per-point mapping.
    """
    multi_point = scan_size is not None and scan_size > 1
    points = payload if multi_point else [payload]
    cleaned_points: list[Any] = []
    for point in points:
        if isinstance(point, dict):
            clean = dict(point)
            clean.pop("peaks", None)
            cleaned_points.append(clean)
        else:
            cleaned_points.append(point)
    cleaned_payload = cleaned_points if multi_point else cleaned_points[0]
    leaves = stack_payload_leaves(label, cleaned_payload, scan_size=scan_size)
    if leaves is None:
        return None
    has_independent_count = "uncertainty.n_independent" in leaves.arrays
    has_psd_std = "psd_std" in leaves.arrays
    has_psd_sem = "psd_sem" in leaves.arrays
    sampling_bases = (
        [
            SamplingBasisSchema(
                name="trajectory", count_variable="uncertainty.n_independent"
            )
        ]
        if has_independent_count and (has_psd_std or has_psd_sem)
        else []
    )
    uncertainties = []
    if has_independent_count and has_psd_std:
        uncertainties.append(
            UncertaintySchema(
                target="psd",
                kind="sample_std",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable="psd_std",
            )
        )
    if has_independent_count and has_psd_sem:
        uncertainties.append(
            UncertaintySchema(
                target="psd",
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable="psd_sem",
            )
        )
    measurement_model = _meta_scalar(
        leaves.payload_meta, "measurement_model", "complex_field"
    )
    leading = ("scan",) if scan_size is not None else ()
    coordinates = [
        CoordinateSchema(
            name="frequency",
            variable="axis",
            dims=(*leading, "frequency"),
            role="auxiliary" if leading else "dimension",
            units="inverse_time",
        ),
    ]
    declared_dims: dict[str, tuple[str, ...]] = {
        "axis": ("frequency",),
        "psd": ("frequency", "channel"),
        "psd_std": ("frequency", "channel"),
        "psd_sem": ("frequency", "channel"),
    }
    if measurement_model == "sampled_scalar":
        declared_dims["channels"] = ("channel",)
        coordinates.append(
            CoordinateSchema(
                name="channel",
                variable="channels",
                dims=(*leading, "channel"),
                role="auxiliary" if leading else "dimension",
            )
        )
        variable_units = {
            "psd": _meta_scalar(leaves.payload_meta, "psd_units", "1/inverse_time")
        }
        attributes: dict[str, Any] = {
            "frequency_units": "inverse_time",
            "orientation": "unsigned",
            "sidedness": "one_sided",
            "normalization": "density",
            "window": _estimator_window(analyser),
            "estimator": _meta_scalar(leaves.payload_meta, "estimator", "periodogram"),
            "measurement_model": "sampled_scalar",
            "detrend": _meta_scalar(leaves.payload_meta, "detrend", "none"),
            "channel_names": _meta_list(leaves.payload_meta, "channel_names"),
            "graph_ready": True,
        }
    else:
        declared_dims["modes"] = ("channel",)
        coordinates.append(
            CoordinateSchema(
                name="mode",
                variable="modes",
                dims=(*leading, "channel"),
                role="auxiliary" if leading else "dimension",
            )
        )
        variable_units = {}
        attributes = {
            "frequency_units": "inverse_time",
            "orientation": _meta_scalar(
                leaves.payload_meta, "orientation", "phase_decreasing"
            ),
            "sidedness": "two_sided",
            "normalization": "density",
            "window": _estimator_window(analyser),
            "estimator": _meta_scalar(leaves.payload_meta, "estimator", "periodogram"),
            "measurement_model": "complex_field",
            "graph_ready": True,
        }
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.SPECTRAL,
        declared_dims=declared_dims,
        axis_specs={
            "frequency": TypedAxisSpec(
                "frequency", AxisRole.COORDINATE, units="inverse_time"
            ),
            "channel": TypedAxisSpec("channel", AxisRole.COMPONENT),
        },
        quantities={"psd": SpectralQuantity.POWER_SPECTRAL_DENSITY.value},
        variable_units=variable_units,
        constraints={"psd": VariableConstraints(nonnegative=True)},
        uncertainties=uncertainties,
        sampling_bases=sampling_bases,
        coordinates=coordinates,
        attributes=attributes,
    )
    if dataset is None:
        return None
    products: dict[str, Dataset] = {label: dataset}
    return products


class PsdResultAccumulator:
    """Merge PSD batch results without retaining trajectory spectra."""

    def __init__(self, analyzer: PsdAnalyzer) -> None:
        self.analyzer = analyzer
        config = cast(PsdAnalyzerConfig, analyzer.config)
        self.measurement_model = (
            "complex_field"
            if config.measurement.complex_field is not None
            else "sampled_scalar"
        )
        self.count = 0
        self.mean: _np.ndarray | None = None
        self.m2: _np.ndarray | None = None
        self.payload: dict[str, Any] | None = None

    def _consume_payload(self, payload: dict[str, Any]) -> None:
        measurement_model = payload.get("measurement_model")
        if measurement_model != self.measurement_model:
            raise ValueError(
                f"PSD accumulator of measurement {self.measurement_model!r} "
                f"received a {measurement_model!r} payload"
            )
        partial_count = int(payload["uncertainty"]["n_independent"])
        partial_mean = _np.asarray(payload["psd"])
        partial_std = _np.asarray(payload["psd_std"])
        if partial_count < 2:
            partial_m2 = _np.zeros_like(partial_mean)
        else:
            partial_m2 = _np.square(partial_std)
            partial_m2 *= partial_count - 1
        if self.payload is None:
            self.payload = {
                key: value
                for key, value in payload.items()
                if key not in {"psd", "psd_std", "psd_sem"}
            }
            self.mean = partial_mean.copy()
            self.m2 = partial_m2.copy()
            self.count = partial_count
            return
        if not _np.array_equal(self.payload["axis"], payload["axis"]):
            raise ValueError("PSD trajectory batches produced different axes")
        identity_key = (
            "modes" if self.measurement_model == "complex_field" else "channels"
        )
        if not _np.array_equal(
            self.payload[identity_key], payload[identity_key]
        ):
            raise ValueError("PSD trajectory batches produced different channels")
        assert self.mean is not None and self.m2 is not None
        total = self.count + partial_count
        delta = partial_mean - self.mean
        self.mean += delta * (partial_count / total)
        self.m2 += partial_m2
        _np.square(delta, out=delta)
        delta *= self.count * partial_count / total
        self.m2 += delta
        self.count = total

    def consume_product(self, product: Dataset) -> None:
        """Consume one typed PSD product without rebuilding a legacy result."""
        count = int(product.handle("uncertainty.n_independent").materialize())
        measurement_model = product.attributes.get("measurement_model")
        if not isinstance(measurement_model, str):
            raise TypeError(
                "PSD product does not declare a measurement_model attribute"
            )
        payload_meta = product.attributes.get("payload_meta", {})
        if not isinstance(payload_meta, Mapping):
            payload_meta = {}
        payload: dict[str, Any] = {
            "axis": product.handle("axis").materialize(),
            "psd": product.handle("psd").materialize(),
            "psd_std": product.handle("psd_std").materialize(),
            "psd_sem": product.handle("psd_sem").materialize(),
            "sample_dt": product.handle("sample_dt").materialize(),
            "nyquist": product.handle("nyquist").materialize(),
            "orientation": product.attributes["orientation"],
            "estimator": product.attributes["estimator"],
            "measurement_model": measurement_model,
            "uncertainty": {"n_independent": count},
        }
        if measurement_model == "sampled_scalar":
            channel_names = product.attributes.get("channel_names")
            if not isinstance(channel_names, list):
                raise TypeError(
                    "sampled-scalar PSD product is missing a channel_names "
                    "attribute"
                )
            payload["channels"] = product.handle("channels").materialize()
            payload["channel_names"] = channel_names
            payload["sidedness"] = product.attributes["sidedness"]
            payload["detrend"] = product.attributes["detrend"]
            payload["convention"] = payload_meta.get("convention")
            psd_variable = product.schema.variable("psd")
            payload["psd_units"] = psd_variable.units if psd_variable else ""
        else:
            payload["modes"] = product.handle("modes").materialize()
        self._consume_payload(payload)

    def merge(self, other: "PsdResultAccumulator") -> None:
        self._consume_payload(other._finalized_payload())

    def finalize_product(self, label: str) -> Mapping[str, Dataset] | None:
        return _build_psd_products(
            self._finalized_payload(),
            scan_size=None,
            label=label,
            analyser=self.analyzer,
        )

    def checkpoint(self) -> dict[str, Any]:
        return {"payload": self._finalized_payload()}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        payload = checkpoint.get("payload")
        if not isinstance(payload, dict):
            raise TypeError("PSD reducer checkpoint is missing its payload")
        self._consume_payload(payload)

    def _finalized_payload(self) -> dict[str, Any]:
        if self.payload is None or self.mean is None or self.m2 is None:
            raise RuntimeError("cannot finalize an empty PSD accumulator")
        payload = dict(self.payload)
        if self.count > 1:
            std = _np.sqrt(_np.maximum(self.m2 / (self.count - 1), 0.0))
            sem = std / _np.sqrt(float(self.count))
        else:
            std = _np.full(self.mean.shape, _np.nan, dtype=self.mean.dtype)
            sem = std.copy()
        payload["psd"] = self.mean
        payload["psd_std"] = std
        payload["psd_sem"] = sem
        uncertainty = dict(payload["uncertainty"])
        uncertainty["n_independent"] = self.count
        uncertainty["available"] = self.count > 1
        payload["uncertainty"] = uncertainty
        return payload
