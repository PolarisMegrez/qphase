"""The registered spectral-ridge analyser plugin."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict
from typing import Any, ClassVar, cast

import numpy as np
from qphase.backend.base import BackendBase
from qphase.core.errors import QPhaseError
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    SpectralQuantity,
)

from ...contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceRequest,
    WorkEstimate,
)
from ...contracts.quantities import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
)
from ...products import assemble_row_table_product
from ..base import Analyser, check_execution_cancelled, report_execution_progress
from ..frequency_orientation import (
    orientation_metadata,
    resolve_frequency_orientation,
)
from ..spectral_input import SpectralPoint, SpectralScanInput, spectral_scan_input
from .config import (
    Readout,
    SpectralRidgeConfig,
    _ridge_candidate_fields,
    _ridge_selected_fields,
)
from .tracking import (
    SpectralRidgeEstimate,
    _candidate_retention_tiers,
    _track_candidate_paths,
    _tracking_segments,
    estimate_spectral_ridges,
)


def _extract_readout(
    source_data: SpectralPoint,
    readout: Readout,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any], str]:
    spectrum, sem, source = source_data.readout(readout)
    return (
        source_data.axis,
        spectrum,
        sem,
        dict(source_data.attributes),
        source,
    )

def _build_spectral_ridge_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
    config: SpectralRidgeConfig,
) -> dict[str, Dataset]:
    if not isinstance(payload, dict):
        raise TypeError("spectral_ridge payload must be a mapping")
    rows = payload.get("ridge_rows", [])
    candidates = payload.get("candidate_rows", [])
    if not isinstance(rows, list) or not isinstance(candidates, list):
        raise TypeError("spectral_ridge row tables must be lists")
    metadata = {
        key: value
        for key, value in payload.items()
        if key not in {"ridge_rows", "candidate_rows"}
    }
    selected = assemble_row_table_product(
        label,
        rows,
        scan_size=scan_size,
        fields=_ridge_selected_fields(config.scan_param),
        integer_fields=("point_index",),
        scan_index_field="point_index",
        scan_value_field=config.scan_param,
        quantities={"frequency": "peak_location"},
        payload_key="ridge_rows",
        payload_meta=metadata,
        attributes={"table": "spectral_ridge"},
    )
    candidate = assemble_row_table_product(
        f"{label}.candidates",
        candidates,
        scan_size=scan_size,
        fields=_ridge_candidate_fields(config.scan_param),
        integer_fields=(
            "point_index",
            "candidate_index",
            "selected",
            "retained_for_association",
        ),
        scan_index_field="point_index",
        scan_value_field=config.scan_param,
        quantities={"frequency": "peak_location"},
        payload_key="candidate_rows",
        payload_meta=metadata,
        attributes={"table": "spectral_ridge_candidates"},
    )
    return {label: selected, f"{label}.candidates": candidate}

class SpectralRidgeAnalyser(Analyser):
    """Extract model-independent spectral ridges from logical PSD scans."""

    name: ClassVar[str] = "spectral_ridge"
    description: ClassVar[str] = (
        "Scale-space spectral ridges with curvature diagnostics"
    )
    config_schema: ClassVar[type[SpectralRidgeConfig]] = SpectralRidgeConfig

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
        typed = spectral_scan_input(
            inputs,
            psd_key=cast(SpectralRidgeConfig, self.config).psd_key,
            scan_param=cast(SpectralRidgeConfig, self.config).scan_param,
            metadata=metadata,
        )
        payload, _ = self._compute_payload(typed, backend, context=context)
        products = _build_spectral_ridge_products(
            payload,
            scan_size=None,
            label=label,
            config=cast(SpectralRidgeConfig, self.config),
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
        config = cast(SpectralRidgeConfig, self.config)
        required = ["axis", "psd"]
        if any(isinstance(readout, int) for readout in config.readouts):
            required.append("modes")
        return [
            ProductRequirement(
                name=config.psd_key,
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                required_fields=required,
            )
        ]

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        config = cast(SpectralRidgeConfig, self.config)
        selected_fields = _ridge_selected_fields(config.scan_param)
        candidate_fields = _ridge_candidate_fields(config.scan_param)
        return (
            ProductDeclaration(
                name=label, kind=DataKind.STATISTICS, fields=selected_fields
            ),
            ProductDeclaration(
                name=f"{label}.candidates",
                kind=DataKind.STATISTICS,
                fields=candidate_fields,
            ),
        )

    def _compute_payload(
        self,
        data: SpectralScanInput,
        backend: BackendBase,
        *,
        context: Any | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del backend
        config = cast(SpectralRidgeConfig, self.config)
        loaded = data.points
        if not loaded:
            raise QPhaseError("spectral_ridge received no input results")
        point_data: dict[str, list[tuple[float, Any, list[SpectralRidgeEstimate]]]] = {
            str(readout): [] for readout in config.readouts
        }
        reference_orientation: FrequencyOrientation | None = None
        uncertainty_sources: set[str] = set()
        spectra_done = 0
        for item in loaded:
            check_execution_cancelled(context)
            scan_value = item.scan_value
            job_name = item.job_name
            for readout in config.readouts:
                axis, spectrum, sem, payload, source = _extract_readout(
                    item, readout
                )
                orientation = resolve_frequency_orientation(payload)
                if reference_orientation is None:
                    reference_orientation = orientation
                elif reference_orientation != orientation:
                    raise QPhaseError("PSD inputs use different frequency orientations")
                mask = np.ones(axis.shape, dtype=bool)
                if config.freq_min is not None:
                    mask &= axis >= config.freq_min
                if config.freq_max is not None:
                    mask &= axis <= config.freq_max
                if int(np.count_nonzero(mask)) < 16:
                    raise QPhaseError(
                        "spectral ridge search window has fewer than 16 bins"
                    )
                estimates = estimate_spectral_ridges(
                    axis[mask],
                    spectrum[mask],
                    sem=None if sem is None else sem[mask],
                    maximum_profile_bins=config.maximum_profile_bins,
                    smoothing_scale_bins=config.smoothing_scale_bins,
                    local_window_scale=config.local_window_scale,
                    maximum_candidates=config.maximum_candidates,
                    minimum_scale_support=config.minimum_scale_support,
                    minimum_prominence_fraction=config.minimum_prominence_fraction,
                    cluster_scale_factor=config.cluster_scale_factor,
                    confidence_sigma=config.confidence_sigma,
                    plateau_fraction=config.plateau_fraction,
                    frequency_bin_covariance=config.frequency_bin_covariance,
                )
                point_data[str(readout)].append((scan_value, job_name, estimates))
                uncertainty_sources.add(
                    f"{source}+frequency_{config.frequency_bin_covariance}"
                    if sem is not None
                    else "fit_residual"
                )
            spectra_done += len(config.readouts)
            report_execution_progress(
                context,
                completed=spectra_done,
                unit="spectrum",
                message=f"spectral_ridge: extracted {spectra_done} spectra",
            )

        rows: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        for readout in config.readouts:
            check_execution_cancelled(context)
            readout_key = str(readout)
            points = sorted(point_data[readout_key], key=lambda item: item[0])
            candidate_sets = [item[2] for item in points]
            accepted_path_indices: list[set[int]] = [set() for _ in points]
            path_ranks: list[dict[int, list[int]]] = [dict() for _ in points]
            path_cost_deltas: list[dict[int, float]] = [dict() for _ in points]
            if config.tracking_enabled:
                frequency_scale = config.tracking_frequency_scale
                if frequency_scale is None:
                    widths = [
                        estimate.plateau_upper - estimate.plateau_lower
                        for estimates in candidate_sets
                        for estimate in estimates
                        if np.isfinite(estimate.plateau_upper)
                        and np.isfinite(estimate.plateau_lower)
                        and estimate.plateau_upper > estimate.plateau_lower
                    ]
                    # Plateau width is a full width; use its half-width as the
                    # natural transition scale for one ridge.
                    frequency_scale = 0.5 * float(np.median(widths)) if widths else 1.0
                selected: list[int] = []
                scan_values = np.asarray([item[0] for item in points], dtype=float)
                for segment in _tracking_segments(
                    scan_values, config.tracking_gap_factor
                ):
                    check_execution_cancelled(context)
                    paths = _track_candidate_paths(
                        candidate_sets[segment],
                        frequency_scale=max(frequency_scale, np.finfo(float).eps),
                        tracking_weight=config.tracking_weight,
                        huber_delta=config.tracking_huber_delta,
                        path_count=config.tracking_path_count,
                    )
                    best_cost = paths[0].cost
                    accepted_paths = [
                        path
                        for path in paths
                        if path.cost <= best_cost + config.tracking_max_cost_delta
                    ]
                    selected.extend(paths[0].indices)
                    segment_start = int(segment.start or 0)
                    for rank, path in enumerate(accepted_paths):
                        cost_delta = path.cost - best_cost
                        for local_index, candidate_index in enumerate(path.indices):
                            global_index = segment_start + local_index
                            accepted_path_indices[global_index].add(candidate_index)
                            path_ranks[global_index].setdefault(
                                candidate_index, []
                            ).append(rank)
                            current_delta = path_cost_deltas[global_index].get(
                                candidate_index, math.inf
                            )
                            path_cost_deltas[global_index][candidate_index] = min(
                                current_delta, cost_delta
                            )
            else:
                selected = [0 for _ in points]
                for point_index in range(len(points)):
                    accepted_path_indices[point_index].add(0)
                    path_ranks[point_index][0] = [0]
                    path_cost_deltas[point_index][0] = 0.0
            for point_index, ((scan_value, job_name, estimates), chosen) in enumerate(
                zip(points, selected, strict=True)
            ):
                check_execution_cancelled(context)
                chosen = min(chosen, len(estimates) - 1)
                estimate = estimates[chosen]
                retention_tiers = _candidate_retention_tiers(
                    estimates,
                    accepted_path_indices[point_index],
                    relative_height_fraction=(
                        config.retention_relative_height_fraction
                    ),
                    minimum_scale_support=config.retention_min_scale_support,
                    minimum_curvature_significance=(
                        config.retention_min_curvature_significance
                    ),
                    continuity_rescue=config.continuity_rescue,
                )
                retained_indices = [
                    index
                    for index, tier in enumerate(retention_tiers)
                    if tier != "excluded"
                ]
                competitive = [
                    candidate
                    for index, candidate in enumerate(estimates)
                    if index in retained_indices
                ]
                ambiguity_lower = min(
                    candidate.confidence_lower
                    if np.isfinite(candidate.confidence_lower)
                    else candidate.frequency - candidate.frequency_std
                    for candidate in competitive
                )
                ambiguity_upper = max(
                    candidate.confidence_upper
                    if np.isfinite(candidate.confidence_upper)
                    else candidate.frequency + candidate.frequency_std
                    for candidate in competitive
                )
                measurement_name = "trace" if readout == "trace" else f"mode_{readout}"
                row = {
                    "point_index": point_index,
                    "job_name": job_name,
                    config.scan_param: scan_value,
                    "readout": readout,
                    "measurement_name": measurement_name,
                    "measurement_kind": (
                        "incoherent_trace" if readout == "trace" else "bare_mode"
                    ),
                    "frequency": estimate.frequency,
                    "frequency_std": estimate.frequency_std,
                    "local_frequency_std": estimate.local_frequency_std,
                    "scale_frequency_std": estimate.scale_frequency_std,
                    "peak_value": estimate.peak_value,
                    "relative_height": estimate.relative_height,
                    "curvature": estimate.curvature,
                    "curvature_std": estimate.curvature_std,
                    "curvature_significance": estimate.curvature_significance,
                    "scale_support": estimate.scale_support,
                    "confidence_lower": estimate.confidence_lower,
                    "confidence_upper": estimate.confidence_upper,
                    "confidence_width": (
                        estimate.confidence_upper - estimate.confidence_lower
                    ),
                    "ambiguity_lower": ambiguity_lower,
                    "ambiguity_upper": ambiguity_upper,
                    "ambiguity_width": ambiguity_upper - ambiguity_lower,
                    "ambiguity_candidate_count": len(competitive),
                    "ambiguity_peak_fraction": (
                        config.retention_relative_height_fraction
                    ),
                    "plateau_lower": estimate.plateau_lower,
                    "plateau_upper": estimate.plateau_upper,
                    "plateau_width": estimate.plateau_upper - estimate.plateau_lower,
                    "selected_candidate": chosen,
                    "candidate_count": len(estimates),
                    "retained_candidate_count": len(retained_indices),
                    "accepted_tracking_path_count": len(
                        {
                            rank
                            for ranks in path_ranks[point_index].values()
                            for rank in ranks
                        }
                    ),
                    "status": estimate.status,
                    "error": estimate.error,
                }
                rows.append(row)
                for candidate_index, candidate in enumerate(estimates):
                    candidate_rows.append(
                        {
                            "point_index": point_index,
                            config.scan_param: scan_value,
                            "measurement_name": measurement_name,
                            "candidate_index": candidate_index,
                            "selected": candidate_index == chosen,
                            "retained_for_association": (
                                retention_tiers[candidate_index] != "excluded"
                            ),
                            "retention_tier": retention_tiers[candidate_index],
                            "tracking_path_ranks": "|".join(
                                str(rank)
                                for rank in path_ranks[point_index].get(
                                    candidate_index, []
                                )
                            ),
                            "tracking_best_cost_delta": path_cost_deltas[
                                point_index
                            ].get(candidate_index, math.nan),
                            **asdict(candidate),
                        }
                    )

        # Empty scans use this analyser's declared frequency-axis convention.
        metadata = orientation_metadata(
            reference_orientation or DEFAULT_FREQUENCY_ORIENTATION
        )
        return (
            {
                "ridge_rows": rows,
                "candidate_rows": candidate_rows,
                **metadata,
            },
            {
                "scan_param": config.scan_param,
                "readouts": config.readouts,
                "method": "Gaussian scale-space plus local quadratic ridge",
                "selection": (
                    "peak evidence plus optional scan continuity; no model target"
                ),
                "tracking_path_count": config.tracking_path_count,
                "tracking_max_cost_delta": config.tracking_max_cost_delta,
                "tracking_transition": "Huber loss",
                "retention": {
                    "relative_height_fraction": (
                        config.retention_relative_height_fraction
                    ),
                    "minimum_scale_support": config.retention_min_scale_support,
                    "minimum_curvature_significance": (
                        config.retention_min_curvature_significance
                    ),
                    "continuity_rescue": config.continuity_rescue,
                    "fallback_guarantees_one_candidate": True,
                },
                "uncertainty_sources": sorted(uncertainty_sources),
                "frequency_bin_covariance": config.frequency_bin_covariance,
                **metadata,
            },
        )
