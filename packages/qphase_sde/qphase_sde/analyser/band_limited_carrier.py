"""Adaptive carrier estimation from a band-limited power spectrum."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import Field, field_validator, model_validator
from qphase.backend.base import BackendBase
from qphase.core.errors import QPhaseError
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SpectralQuantity,
)

from ..contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ..contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
)
from ..numerics.band_limited import (
    BandLimitedCarrierCandidate,
    BandLimitedCarrierEstimate,
    estimate_band_limited_carrier,
    track_band_limited_carrier,
)
from ..products import assemble_row_table_product
from .base import Analyser, check_execution_cancelled, report_execution_progress
from .frequency_orientation import (
    orientation_metadata,
    resolve_frequency_orientation,
)
from .spectral_input import SpectralPoint, SpectralScanInput, spectral_scan_input
from .spectral_ridge import (
    SpectralRidgeConfig,
)

__all__ = [
    "BandLimitedCarrierAnalyser",
    "BandLimitedCarrierConfig",
    "RidgeConditionedCenterConfig",
]


def _unique_fields(*names: str) -> list[str]:
    return list(dict.fromkeys(names))


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


def _extract_readout(
    loaded: SpectralPoint, readout: int | Literal["trace"]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spectrum, _, _ = loaded.readout(readout)
    return loaded.axis, spectrum, dict(loaded.attributes)


def _ridge_candidate_rows(
    inputs: Mapping[str, Dataset], config: BandLimitedCarrierConfig
) -> list[dict[str, Any]] | None:
    """Materialize the ridge candidate rows the carrier consumes.

    Numeric columns come from the typed variables (exact floats, including
    NaN); the string columns (``measurement_name``, ``retention_tier``,
    ``tracking_path_ranks``) are not typed variables and come from the
    product's ``payload_meta`` rows instead.
    """
    ridge_label = None if config.center is None else config.center.ridge_label
    if ridge_label is None:
        return None
    product = inputs.get(f"{ridge_label}.candidates")
    if product is None:
        raise QPhaseError(
            "band_limited_carrier requires the typed spectral-ridge candidate "
            f"product {ridge_label + '.candidates'!r}; the analyser-config "
            "expander promotes center.spectral_ridge into a sibling analyser"
        )
    rows = [
        dict(row)
        for row in product.attributes.get("payload_meta", {}).get(
            "candidate_rows", []
        )
    ]
    for variable in product.variables:
        values = np.asarray(product.handle(variable.name).materialize()).reshape(-1)
        for index, value in enumerate(values):
            if index >= len(rows):
                break
            rows[index][variable.name] = (
                value.item() if hasattr(value, "item") else value
            )
    return rows


def _sort_key(value: Any) -> tuple[int, Any]:
    try:
        return 0, float(value)
    except (TypeError, ValueError):
        return 1, str(value)


def _estimate_from_config(
    axis: np.ndarray,
    spectrum: np.ndarray,
    config: BandLimitedCarrierConfig,
    *,
    fixed_center: float | None = None,
    fixed_bandwidth_max: float | None = None,
) -> tuple[BandLimitedCarrierEstimate, list[BandLimitedCarrierCandidate]]:
    return estimate_band_limited_carrier(
        axis,
        spectrum,
        freq_min=config.freq_min,
        freq_max=config.freq_max,
        bandwidth_multipliers=config.bandwidth_multipliers,
        bandwidth_min=config.bandwidth_min,
        bandwidth_max=config.bandwidth_max,
        baseline_quantile=config.baseline_quantile,
        concentration_clip_sigma=config.concentration_clip_sigma,
        concentration_iterations=config.concentration_iterations,
        peak_smoothing_bins=config.peak_smoothing_bins,
        taper_fraction=config.taper_fraction,
        maximum_lag=config.maximum_lag,
        coherence_floor=config.coherence_floor,
        minimum_lag_points=config.minimum_lag_points,
        minimum_lag_span=config.minimum_lag_span,
        lag_window_trials=config.lag_window_trials,
        max_phase_fit_rms=config.max_phase_fit_rms,
        max_frequency_drift_fraction=config.max_frequency_drift_fraction,
        max_negative_decay_fraction=config.max_negative_decay_fraction,
        consensus_count=config.consensus_count,
        minimum_log_bandwidth_span=config.minimum_log_bandwidth_span,
        stability_fraction=config.stability_fraction,
        stability_sigma=config.stability_sigma,
        platform_ambiguity_delta=config.platform_ambiguity_delta,
        fixed_center=fixed_center,
        fixed_bandwidth_max=fixed_bandwidth_max,
    )


def _build_band_limited_carrier_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
    config: BandLimitedCarrierConfig,
) -> dict[str, Dataset]:
    if not isinstance(payload, dict):
        raise TypeError("band_limited_carrier payload must be a mapping")
    rows = payload.get("carrier_rows", [])
    candidates = payload.get("candidate_rows", [])
    platforms = payload.get("platform_rows", [])
    if not all(isinstance(value, list) for value in (rows, candidates, platforms)):
        raise TypeError("band_limited_carrier row tables must be lists")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"carrier_rows", "candidate_rows", "platform_rows"}
    }
    result = assemble_row_table_product(
        label,
        rows,
        scan_size=scan_size,
        fields=_carrier_result_fields(config.scan_param),
        integer_fields=(
            "consensus_count",
            "lag_points",
            "touched_maximum_lag",
            "platform_count",
            "tracked_platform_index",
        ),
        scan_value_field=config.scan_param,
        quantities={"frequency": "coherence_frequency"},
        payload_key="carrier_rows",
        payload_meta=metadata,
        attributes={"table": "band_limited_carrier"},
    )
    candidate = assemble_row_table_product(
        f"{label}.candidates",
        candidates,
        scan_size=scan_size,
        fields=_carrier_candidate_fields(config.scan_param),
        integer_fields=("lag_points", "touched_maximum_lag"),
        scan_value_field=config.scan_param,
        quantities={"frequency": "coherence_frequency"},
        payload_key="candidate_rows",
        payload_meta=metadata,
        attributes={"table": "band_limited_carrier_candidates"},
    )
    platform = assemble_row_table_product(
        f"{label}.platforms",
        platforms,
        scan_size=scan_size,
        fields=_carrier_platform_fields(config.scan_param),
        integer_fields=(
            "platform_index",
            "first_candidate",
            "last_candidate",
            "candidate_count",
            "touched_maximum_lag",
        ),
        scan_value_field=config.scan_param,
        quantities={"frequency": "coherence_frequency"},
        payload_key="platform_rows",
        payload_meta=metadata,
        attributes={"table": "band_limited_carrier_platforms"},
    )
    return {
        label: result,
        f"{label}.candidates": candidate,
        f"{label}.platforms": platform,
    }


class BandLimitedCarrierAnalyser(Analyser):
    """Cross-result adaptive carrier estimator for saved PSD datasets."""

    name: ClassVar[str] = "band_limited_carrier"
    description: ClassVar[str] = (
        "Adaptive band-limited long-time first-order-coherence carrier"
    )
    config_schema: ClassVar[type[BandLimitedCarrierConfig]] = BandLimitedCarrierConfig

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        metadata = metadata or {}
        config = cast(BandLimitedCarrierConfig, self.config)
        typed = spectral_scan_input(
            inputs,
            psd_key=config.psd_key,
            scan_param=config.scan_param,
            metadata=metadata,
        )
        ridge_rows = _ridge_candidate_rows(inputs, config)
        payload, _ = self._compute_payload(
            typed,
            backend,
            ridge_rows=ridge_rows,
            context=context,
        )
        products = _build_band_limited_carrier_products(
            payload,
            scan_size=None,
            label=label,
            config=config,
        )
        if not products:
            raise ValueError(f"analyser {label!r} produced no typed products")
        return products

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Work scales with the number of loaded PSD spectra; no total is fabricated."""
        del request
        return WorkEstimate(unit="spectrum", total=None)

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(BandLimitedCarrierConfig, self.config)
        required = ["axis", "psd"]
        if isinstance(config.readout, int):
            required.append("modes")
        requirements = [
            ProductRequirement(
                name=config.psd_key,
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                required_fields=required,
            )
        ]
        ridge_label = None if config.center is None else config.center.ridge_label
        if ridge_label is not None:
            requirements.append(
                ProductRequirement(
                    name=f"{ridge_label}.candidates",
                    kind=DataKind.STATISTICS,
                    required_fields=[
                        "candidate_index",
                        "retained_for_association",
                        "frequency_std",
                        "confidence_lower",
                        "confidence_upper",
                    ],
                )
            )
            # The ridge node's selected-ridge product is consumed too: every
            # output of a graph node must have a consumer, otherwise the
            # executor publishes it as a terminal product. Its data is unused.
            requirements.append(
                ProductRequirement(name=ridge_label, kind=DataKind.STATISTICS)
            )
        return requirements

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        config = cast(BandLimitedCarrierConfig, self.config)
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=_carrier_result_fields(config.scan_param),
            ),
            ProductDeclaration(
                name=f"{label}.candidates",
                kind=DataKind.STATISTICS,
                fields=_carrier_candidate_fields(config.scan_param),
            ),
            ProductDeclaration(
                name=f"{label}.platforms",
                kind=DataKind.STATISTICS,
                fields=_carrier_platform_fields(config.scan_param),
            ),
        )

    def _compute_payload(
        self,
        data: SpectralScanInput,
        backend: BackendBase,
        *,
        ridge_rows: list[dict[str, Any]] | None = None,
        context: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        config = cast(BandLimitedCarrierConfig, self.config)
        center_config = config.center
        loaded_results = data.points
        if not loaded_results:
            raise QPhaseError("band_limited_carrier received no input results")

        ridge_by_scan: dict[float, list[dict[str, Any]]] = {}
        all_ridges_by_scan: dict[float, list[dict[str, Any]]] = {}
        if center_config is not None:
            if ridge_rows is None:
                raise QPhaseError(
                    "ridge-conditioned carrier center requires the typed "
                    "spectral-ridge candidate product"
                )
            measurement_name = (
                "trace" if config.readout == "trace" else f"mode_{config.readout}"
            )
            for ridge_row in ridge_rows:
                if ridge_row["measurement_name"] != measurement_name:
                    continue
                scan_key = float(ridge_row[config.scan_param])
                all_ridges_by_scan.setdefault(scan_key, []).append(ridge_row)
                if ridge_row["retained_for_association"]:
                    ridge_by_scan.setdefault(scan_key, []).append(ridge_row)

        rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        platform_rows: list[dict[str, Any]] = []
        estimates: list[BandLimitedCarrierEstimate] = []
        reference_orientation: FrequencyOrientation | None = None
        for point_index, loaded in enumerate(loaded_results, start=1):
            check_execution_cancelled(context)
            scan_value = loaded.scan_value
            job_name = loaded.job_name
            axis, spectrum, payload = _extract_readout(loaded, config.readout)
            orientation = resolve_frequency_orientation(payload)
            if reference_orientation is None:
                reference_orientation = orientation
            elif reference_orientation != orientation:
                raise QPhaseError("PSD inputs use different frequency orientations")
            if config.center is None:
                center_rows: list[dict[str, Any] | None] = [None]
            else:
                center_rows = []
                center_rows.extend(ridge_by_scan.get(float(scan_value), []))
                if not center_rows:
                    raise QPhaseError(
                        f"spectral ridge retained no center for {config.scan_param}="
                        f"{scan_value!r}"
                    )
            neighbor_frequencies = np.asarray(
                [
                    float(item["frequency"])
                    for item in all_ridges_by_scan.get(float(scan_value), [])
                ]
            )
            for ridge_order, center_row in enumerate(center_rows):
                fixed_center = (
                    None if center_row is None else float(center_row["frequency"])
                )
                neighbor_distance = math.nan
                fixed_bandwidth_max = None
                if fixed_center is not None and neighbor_frequencies.size > 1:
                    distances = np.abs(neighbor_frequencies - fixed_center)
                    distances = distances[distances > np.finfo(float).eps]
                    if distances.size:
                        neighbor_distance = float(np.min(distances))
                        assert center_config is not None
                        fixed_bandwidth_max = (
                            center_config.maximum_neighbor_fraction * neighbor_distance
                        )
                estimate, candidates = _estimate_from_config(
                    axis,
                    spectrum,
                    config,
                    fixed_center=fixed_center,
                    fixed_bandwidth_max=fixed_bandwidth_max,
                )
                estimates.append(estimate)
                ridge_fields: dict[str, Any] = {}
                if center_row is not None:
                    assert fixed_center is not None
                    ridge_fields = {
                        "ridge_candidate_index": int(center_row["candidate_index"]),
                        "ridge_order": ridge_order,
                        "ridge_center": fixed_center,
                        "ridge_center_std": float(center_row["frequency_std"]),
                        "ridge_confidence_lower": float(center_row["confidence_lower"]),
                        "ridge_confidence_upper": float(center_row["confidence_upper"]),
                        "ridge_retention_tier": center_row["retention_tier"],
                        "ridge_tracking_path_ranks": center_row["tracking_path_ranks"],
                        "ridge_neighbor_distance": neighbor_distance,
                        "ridge_bandwidth_max": (
                            fixed_bandwidth_max
                            if fixed_bandwidth_max is not None
                            else math.nan
                        ),
                        "ridge_carrier_correction": (
                            estimate.frequency - fixed_center
                            if np.isfinite(estimate.frequency)
                            else math.nan
                        ),
                        "ridge_conditioned_uncertainty_upper": (
                            estimate.diagnostic_uncertainty
                            + float(center_row["frequency_std"])
                            if np.isfinite(estimate.diagnostic_uncertainty)
                            else math.nan
                        ),
                        "ridge_platform_selection": "nearest_ridge_center",
                    }
                rows.append(
                    {
                        "job_name": job_name,
                        config.scan_param: scan_value,
                        "readout": config.readout,
                        "orientation": orientation,
                        **ridge_fields,
                        **estimate.as_dict(),
                    }
                )
                for candidate in candidates:
                    candidate_rows.append(
                        {
                            "job_name": job_name,
                            config.scan_param: scan_value,
                            "readout": config.readout,
                            **ridge_fields,
                            **asdict(candidate),
                        }
                    )
                for platform_index, platform in enumerate(estimate.platforms):
                    platform_rows.append(
                        {
                            "job_name": job_name,
                            config.scan_param: scan_value,
                            "readout": config.readout,
                            **ridge_fields,
                            "platform_index": platform_index,
                            **asdict(platform),
                        }
                    )
            report_execution_progress(
                context,
                completed=point_index,
                unit="spectrum",
                message=f"band_limited_carrier: estimated {point_index} spectra",
            )

        records = sorted(
            zip(rows, estimates, strict=True),
            key=lambda item: _sort_key(item[0][config.scan_param]),
        )
        rows = [item[0] for item in records]
        estimates = [item[1] for item in records]
        if (
            config.center is None
            and config.tracking_enabled
            and all(
                isinstance(row[config.scan_param], (int, float, np.number))
                for row in rows
            )
        ):
            tracked = track_band_limited_carrier(
                np.asarray([row[config.scan_param] for row in rows], dtype=float),
                estimates,
                frequency_scale=config.tracking_frequency_scale,
                curvature_weight=config.tracking_curvature_weight,
                ambiguity_delta=config.tracking_ambiguity_delta,
                max_normalized_curvature=config.tracking_max_normalized_curvature,
                cancel_check=lambda: check_execution_cancelled(context),
            )
            for row, tracking in zip(rows, tracked, strict=True):
                row.update(tracking)
        else:
            for row in rows:
                row.update(
                    {
                        "tracked_frequency": math.nan,
                        "tracked_diagnostic_uncertainty": math.nan,
                        "tracked_platform_index": -1,
                        "tracked_status": (
                            "ridge_conditioned"
                            if config.center is not None
                            else "disabled"
                        ),
                        "tracked_path_cost_gap": math.nan,
                    }
                )
        candidate_rows.sort(
            key=lambda row: (
                _sort_key(row[config.scan_param]),
                int(row.get("ridge_order", -1)),
                float(row["half_bandwidth"]),
            )
        )
        platform_rows.sort(
            key=lambda row: (
                _sort_key(row[config.scan_param]),
                int(row.get("ridge_order", -1)),
                int(row["platform_index"]),
            )
        )
        # Empty scans use this analyser's declared frequency-axis convention.
        metadata = orientation_metadata(
            reference_orientation or DEFAULT_FREQUENCY_ORIENTATION
        )
        return (
            {
                "carrier_rows": rows,
                "candidate_rows": candidate_rows,
                "platform_rows": platform_rows,
                **metadata,
            },
            {
                "scan_param": config.scan_param,
                "readout": config.readout,
                "count": len(rows),
                "center_method": (
                    "spectral_ridge" if config.center is not None else "adaptive_peak"
                ),
                "uncertainty_note": (
                    "regression_std is HAC phase-regression uncertainty; "
                    "bandwidth_std is within-platform estimator sensitivity; "
                    "neither quantity is trajectory SEM"
                ),
                **metadata,
            },
        )
