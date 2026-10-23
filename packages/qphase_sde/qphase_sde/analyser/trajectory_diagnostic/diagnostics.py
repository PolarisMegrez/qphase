"""The registered diagnostics analyser and its internal children.

``TrajectoryDiagnostics`` owns the plugin surface (requirements, workspace,
output declaration) and dispatches to one child per diagnostic block; the
numeric work itself lives in :mod:`..estimators`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.data import (
    AxisRole,
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
)

from ...contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
)
from ...numerics.coordinates import canonical_r_coordinates
from ...products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from ..allan_statistics import calculate_allan_variance
from ..base import (
    Analyser,
    map_trajectory_product,
    select_mode_columns,
    trajectory_arrays,
    trajectory_requirement,
)
from ..frequency_orientation import orientation_metadata
from .base import TrajectoryDiagnosticChild, TrajectoryDiagnosticContext
from .config import (
    _ALLAN_TABLE_FIELDS,
    _G1_MODEL_FIT_FIELDS,
    _PHASE_INCREMENT_FIELDS,
    TrajectoryDiagnosticsConfig,
)
from .estimators import (
    _G1_MODEL_FUNCTIONS,
    _block_spectrum,
    _block_statistics,
    _coherence,
    _matrix_projection,
    _phase_increment_summary,
    _stationarity_details,
)


def _build_trajectory_diagnostics_products(
    payload: Any, *, scan_size: int | None, label: str
) -> dict[str, Dataset] | None:
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    declared_dims: dict[str, tuple[str, ...]] = {}
    for key, array in leaves.arrays.items():
        if key == "modes":
            declared_dims[key] = ("mode",)
        elif ".allan." in key and array.ndim == 2:
            declared_dims[key] = ("trajectory", "tau")
        elif ".allan." in key and array.ndim == 1:
            declared_dims[key] = ("tau",)
        elif ".coherence." in key and array.ndim == 2:
            declared_dims[key] = ("trajectory", "lag")
        elif key.endswith(
            ("mean_amplitude_per_trajectory", "mean_power_per_trajectory")
        ):
            declared_dims[key] = ("trajectory",)
        elif "per_trajectory" in key and ".phase_increment." in key:
            declared_dims[key] = ("trajectory",)
        elif "per_trajectory" in key and ".coherence." in key:
            declared_dims[key] = (
                ("trajectory", "lag") if array.ndim == 2 else ("trajectory",)
            )
        elif key.endswith("lag_window_seconds") and array.ndim == 1:
            declared_dims[key] = ("fit_parameter",)
        elif key.endswith("block_statistics") and array.ndim == 1:
            declared_dims[key] = ("block",)
        elif ".coherence." in key and array.ndim == 1:
            declared_dims[key] = ("lag",)
    axis_specs = {
        "mode": TypedAxisSpec("mode", AxisRole.COMPONENT),
        "trajectory": TypedAxisSpec("trajectory", AxisRole.REALIZATION),
        "tau": TypedAxisSpec("tau", AxisRole.COORDINATE, units="time"),
        "lag": TypedAxisSpec("lag", AxisRole.COORDINATE, units="time"),
        "fit_parameter": TypedAxisSpec("fit_parameter", AxisRole.INDEX),
        "block": TypedAxisSpec("block", AxisRole.INDEX),
    }
    product = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims=declared_dims,
        axis_specs=axis_specs,
        attributes={
            "graph_ready": True,
            "typed_builder": "trajectory_diagnostics/1",
        },
    )
    return None if product is None else {label: product}


class TrajectoryDiagnostics(Analyser):
    """Analyze saved complex trajectories without assuming a spectral line shape."""

    name: ClassVar[str] = "trajectory_diagnostics"
    description: ClassVar[str] = "Stationarity, coherence, and Allan diagnostics"
    config_schema: ClassVar[type[TrajectoryDiagnosticsConfig]] = (
        TrajectoryDiagnosticsConfig
    )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        """Declare the current full-record host implementation."""
        return AnalyserExecutionCapabilities(
            preferred_location="host",
            supports_trajectory_batching=False,
            supports_time_streaming=False,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate D2H, canonical-coordinate, and diagnostic temporaries."""
        materialization = (
            request.trajectory_bytes if request.backend_name == "cupy" else 0
        )
        coordinate_bytes = request.trajectory_bytes * request.n_record_modes // 2
        scratch = request.trajectory_bytes
        return AnalyserWorkspaceEstimate(
            host_bytes=materialization + coordinate_bytes + scratch
        )

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
        del backend
        trajectory = inputs.get("trajectories")
        if trajectory is None:
            raise TypeError("trajectory_diagnostics requires trajectories")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            array, t0, dt, recorded = trajectory_arrays(point)
            modes, columns = select_mode_columns(
                recorded, cast(TrajectoryDiagnosticsConfig, self.config).modes
            )
            payload = self._compute_payload(array, modes, columns, t0=t0, dt=dt)
            products = _build_trajectory_diagnostics_products(
                payload, scan_size=None, label=label
            )
            if not products:
                raise ValueError("trajectory_diagnostics produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        array: Any,
        modes: list[int],
        columns: list[int],
        *,
        t0: float,
        dt: float,
    ) -> dict[str, Any]:
        config = cast(TrajectoryDiagnosticsConfig, self.config)
        values = np.asarray(convert_to_numpy(array))
        if values.ndim != 3:
            raise ValueError(
                "trajectory diagnostics expects shape (n_traj, n_time, n_modes)"
            )
        if values.shape[1] < 3:
            raise ValueError("trajectory diagnostics requires at least three samples")
        if not np.iscomplexobj(values):
            raise ValueError("trajectory diagnostics requires complex mode amplitudes")
        if dt <= 0.0:
            raise ValueError("trajectory sample spacing must be positive")
        diagnostic_context = TrajectoryDiagnosticContext(
            values=values,
            dt=dt,
            t0=t0,
            modes=tuple(modes),
            mode_columns=tuple(columns),
            coordinate_builder=canonical_r_coordinates,
        )
        result: dict[str, Any] = {
            "modes": modes,
            "t0": t0,
            "dt": dt,
            "n_traj": int(values.shape[0]),
            "n_samples": int(values.shape[1]),
            **orientation_metadata(config.orientation),
            "mode_results": {},
        }
        children: list[TrajectoryDiagnosticChild] = [_ModeSummaryChild()]
        if config.coherence:
            children.append(_CoherenceChild())
        if config.allan:
            children.append(_AllanChild())
        if config.block_spectrum:
            children.append(_BlockSpectrumChild())
        if config.matrix_projection:
            children.append(_MatrixProjectionChild())
        if config.stationarity_details:
            children.append(_StationarityChild())
        for child in children:
            child.apply(diagnostic_context, result, config)
        return result

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        config = cast(TrajectoryDiagnosticsConfig, self.config)
        fields = ["modes", "t0", "dt", "n_traj", "n_samples"]
        for mode in config.modes:
            prefix = f"mode_results.{mode}"
            fields += [
                f"{prefix}.mean_amplitude_per_trajectory",
                f"{prefix}.mean_power_per_trajectory",
            ]
            fields += [
                f"{prefix}.phase_increment.{leaf}" for leaf in _PHASE_INCREMENT_FIELDS
            ]
            if not config.block_durations:
                # An empty block-statistics list stacks as an empty float array.
                fields.append(f"{prefix}.block_statistics")
            if config.coherence:
                coherence = f"{prefix}.coherence"
                fields += [
                    f"{coherence}.{leaf}"
                    for leaf in (
                        "lag",
                        "g1",
                        "g1_normalized",
                        "g1_sem_magnitude",
                        "centered",
                    )
                ]
                fits = f"{coherence}.model_fits"
                fields += [f"{fits}.{leaf}" for leaf in _G1_MODEL_FIT_FIELDS]
                for model in _G1_MODEL_FUNCTIONS:
                    fields += [
                        f"{fits}.models.{model}.n_train",
                        f"{fits}.models.{model}.n_holdout",
                    ]
                if config.keep_coherence_per_trajectory:
                    fields.append(f"{coherence}.g1_per_trajectory")
            if config.allan:
                fields += [f"{prefix}.allan.{leaf}" for leaf in _ALLAN_TABLE_FIELDS]
        if config.matrix_projection:
            fields += [
                "matrix_projection.n_modes",
                "matrix_projection.n_coordinates",
                "matrix_projection.reference_vector",
                "matrix_projection.reference_norm",
                "matrix_projection.dt",
                "matrix_projection.mean_coordinates_per_trajectory",
            ]
            if config.matrix_projection_left_vector is not None:
                fields += [
                    "matrix_projection.left_vector",
                    "matrix_projection.left_vector_norm",
                    "matrix_projection.projection_per_trajectory",
                    "matrix_projection.projection_mean_per_trajectory",
                    "matrix_projection.projection_std_per_trajectory",
                    "matrix_projection.projection_min_per_trajectory",
                    "matrix_projection.projection_max_per_trajectory",
                ]
            if config.matrix_projection_keep_coordinates:
                fields.append("matrix_projection.coordinates_per_trajectory")
        if config.stationarity_details and config.block_durations:
            fields.append("stationarity.n_features")
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=fields,
            ),
        )


class _ModeSummaryChild:
    name = "mode_summary"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            series = context.series(mode)
            result["mode_results"][mode] = {
                "mean_amplitude_per_trajectory": np.mean(np.abs(series), axis=1),
                "mean_power_per_trajectory": np.mean(np.abs(series) ** 2, axis=1),
                "phase_increment": _phase_increment_summary(
                    series,
                    context.dt,
                    config.amplitude_floor,
                    config.orientation,
                ),
                "block_statistics": _block_statistics(
                    series,
                    context.dt,
                    config.block_durations,
                    config.amplitude_floor,
                    config.orientation,
                ),
            }


class _CoherenceChild:
    name = "coherence"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            result["mode_results"][mode]["coherence"] = _coherence(
                context.series(mode),
                context.dt,
                max_lag=config.coherence_max_lag,
                center=config.center_coherence,
                keep_per_trajectory=config.keep_coherence_per_trajectory,
            )


class _AllanChild:
    name = "allan"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            allan = calculate_allan_variance(
                context.series(mode),
                context.dt,
                taus=config.allan_taus,
                points=config.allan_points,
                min_windows=config.allan_min_windows,
                min_independent_windows=config.allan_min_independent_windows,
                amplitude_floor=config.amplitude_floor,
            )
            allan.update(orientation_metadata(config.orientation))
            result["mode_results"][mode]["allan"] = allan


class _BlockSpectrumChild:
    name = "block_spectrum"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        for mode in context.modes:
            result["mode_results"][mode]["block_spectrum"] = _block_spectrum(
                context.series(mode),
                context.dt,
                config.block_durations,
                wing_factor=config.block_spectrum_wing_factor,
                orientation=config.orientation,
            )


class _MatrixProjectionChild:
    name = "matrix_projection"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        result[self.name] = _matrix_projection(
            context.values,
            context.canonical_coordinates(),
            context.dt,
            config,
        )


class _StationarityChild:
    name = "stationarity"

    def apply(
        self,
        context: TrajectoryDiagnosticContext,
        result: dict[str, Any],
        config: TrajectoryDiagnosticsConfig,
    ) -> None:
        result[self.name] = _stationarity_details(
            context.canonical_coordinates(),
            context.dt,
            config.block_durations,
            n_modes=context.values.shape[-1],
        )
