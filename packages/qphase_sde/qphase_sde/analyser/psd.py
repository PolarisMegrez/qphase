"""qphase_sde: Power Spectral Density
---------------------------------------------------------
Compute power spectral density (PSD) from multi-trajectory time series for one
or more modes using FFT-based periodograms.

Behavior
--------
- Support two input interpretations: complex-valued directly (``kind='complex'``)
    or magnitude-based (``kind='modular'``).
- Provide common PSD conventions: unitary/symmetric (angular frequency 锠? and
    pragmatic (frequency f). Exact scaling, return shapes, and error semantics are
    specified by the function docstrings.

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
    trajectory_arrays,
    trajectory_requirement,
)
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    OrientationInput,
    orient_spectrum,
    orientation_metadata,
    orientation_schema_extra,
)

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
]


class PsdAnalyzerConfig(PluginConfigBase):
    """Configuration for PSD Analyzer."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["complex", "modular"] = Field(
        ..., description="FFT of complex signal or FFT of |signal|"
    )
    modes: list[int] = Field(..., description="Mode indices for analysis")
    convention: Literal["symmetric", "unitary", "pragmatic"] = Field(
        "symmetric", description="PSD convention"
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
    dt: float | None = Field(None, description="Sampling interval (override)")
    expected_freq_max: float | None = Field(
        None,
        gt=0.0,
        description=(
            "Optional maximum frequency magnitude expected in the output-axis "
            "units; analysis fails if it reaches the Nyquist limit"
        ),
    )
    @model_validator(mode="after")
    def validate_modes(self) -> "PsdAnalyzerConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


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
        # Both real and complex records are accepted (``kind="modular"`` uses
        # the magnitude), so no value_domain constraint is declared.
        return [trajectory_requirement(value_domain=None)]

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
            raise TypeError("PSD analyser requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            data, _, sample_dt, recorded_modes = trajectory_arrays(point)
            config = cast(PsdAnalyzerConfig, self.config)
            dt = config.dt if config.dt is not None else sample_dt
            mode_map = {mode: index for index, mode in enumerate(recorded_modes)}
            missing = [mode for mode in config.modes if mode not in mode_map]
            if missing:
                raise ValueError(f"requested PSD modes were not recorded: {missing}")
            payload = self._compute_payload(
                data,
                dt=float(dt),
                mode_columns=[mode_map[mode] for mode in config.modes],
                backend=backend,
            )
            products = _build_psd_products(
                payload, scan_size=None, label=label, analyser=self
            )
            if not products:
                raise ValueError("PSD analyser produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        data: Any,
        *,
        dt: float,
        mode_columns: list[int],
        backend: BackendBase,
    ) -> dict[str, Any]:
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
        estimate0 = self._estimate_single(
            data[:, :, mode_columns[0]],
            dt,
            kind=config.kind,
            convention=convention,
            orientation=config.orientation,
            backend=backend,
        )
        axis = estimate0.axis
        output_shape = (int(axis.shape[0]), len(mode_columns))
        mean = _np.empty(output_shape, dtype=estimate0.mean.dtype)
        std = _np.empty(output_shape, dtype=estimate0.std.dtype)
        sem = _np.empty(output_shape, dtype=estimate0.sem.dtype)
        mean[:, 0], std[:, 0], sem[:, 0] = (
            estimate0.mean,
            estimate0.std,
            estimate0.sem,
        )
        n_independent = estimate0.n_independent
        for index, column in enumerate(mode_columns[1:], start=1):
            estimate = self._estimate_single(
                data[:, :, column],
                dt,
                kind=config.kind,
                convention=convention,
                orientation=config.orientation,
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
            "modes": config.modes,
            "kind": config.kind,
            "convention": convention,
            **orientation_metadata(config.orientation),
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
        declarations = [
            ProductDeclaration(
                name=label,
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                fields=list(_PSD_PRODUCT_FIELDS),
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
        kind: str = "complex",
        convention: str = "symmetric",
        orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
        backend: BackendBase,
    ) -> _PsdEstimate:
        """Compute PSD and cross-trajectory uncertainty for a single mode."""
        # Convert input to backend array
        x_arr = backend.asarray(x)

        if kind == "modular":
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

#: Numeric payload leaves; string leaves stay in payload metadata.
_PSD_PRODUCT_FIELDS = (
    "axis",
    "psd",
    "modes",
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
    leading = ("scan",) if scan_size is not None else ()
    coordinates = [
        CoordinateSchema(
            name="frequency",
            variable="axis",
            dims=(*leading, "frequency"),
            role="auxiliary" if leading else "dimension",
            units="inverse_time",
        ),
        CoordinateSchema(
            name="mode",
            variable="modes",
            dims=(*leading, "channel"),
            role="auxiliary" if leading else "dimension",
        ),
    ]
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.SPECTRAL,
        declared_dims={
            "axis": ("frequency",),
            "psd": ("frequency", "channel"),
            "psd_std": ("frequency", "channel"),
            "psd_sem": ("frequency", "channel"),
            "modes": ("channel",),
        },
        axis_specs={
            "frequency": TypedAxisSpec(
                "frequency", AxisRole.COORDINATE, units="inverse_time"
            ),
            "channel": TypedAxisSpec("channel", AxisRole.COMPONENT),
        },
        quantities={"psd": SpectralQuantity.POWER_SPECTRAL_DENSITY.value},
        constraints={"psd": VariableConstraints(nonnegative=True)},
        uncertainties=uncertainties,
        sampling_bases=sampling_bases,
        coordinates=coordinates,
        attributes={
            "frequency_units": "inverse_time",
            "orientation": _meta_scalar(
                leaves.payload_meta, "orientation", "phase_decreasing"
            ),
            "sidedness": "two_sided",
            "normalization": "density",
            "window": _estimator_window(analyser),
            "estimator": _meta_scalar(leaves.payload_meta, "estimator", "periodogram"),
            "graph_ready": True,
        },
    )
    if dataset is None:
        return None
    products: dict[str, Dataset] = {label: dataset}
    return products


class PsdResultAccumulator:
    """Merge PSD batch results without retaining trajectory spectra."""

    def __init__(self, analyzer: PsdAnalyzer) -> None:
        self.analyzer = analyzer
        self.count = 0
        self.mean: _np.ndarray | None = None
        self.m2: _np.ndarray | None = None
        self.payload: dict[str, Any] | None = None

    def _consume_payload(self, payload: dict[str, Any]) -> None:
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
        payload = {
            "axis": product.handle("axis").materialize(),
            "psd": product.handle("psd").materialize(),
            "psd_std": product.handle("psd_std").materialize(),
            "psd_sem": product.handle("psd_sem").materialize(),
            "modes": product.handle("modes").materialize(),
            "sample_dt": product.handle("sample_dt").materialize(),
            "nyquist": product.handle("nyquist").materialize(),
            "orientation": product.attributes["orientation"],
            "estimator": product.attributes["estimator"],
            "uncertainty": {"n_independent": count},
        }
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
