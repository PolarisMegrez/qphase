"""Moments of user-defined Hermitian quadratic observables."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_serializer, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    AxisRole,
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
from ..contracts.quantities import SDEMomentFamilySchema, SDEQuantity
from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .base import (
    Analyzer,
    map_trajectory_product,
    select_mode_columns,
    trajectory_arrays,
    trajectory_requirement,
)

__all__ = [
    "QuadraticMomentAnalyzer",
    "QuadraticMomentConfig",
    "QuadraticObservableSpec",
]

_QUADRATIC_MOMENT_FIELDS: tuple[str, ...] = (
    "modes",
    "matrices",
    "centers",
    "reference_matrices",
    "max_order",
    "n_traj",
    "n_samples",
    "t0",
    "dt",
    "observation_duration",
    "raw_moments",
    "raw_moment_sem",
    "central_moments",
    "cumulants",
    "cumulant_sem",
    "per_trajectory_raw_moments",
    "time_blocks.count",
    "time_blocks.start_index",
    "time_blocks.stop_index",
    "time_blocks.start_time",
    "time_blocks.stop_time",
    "time_blocks.raw_moments",
    "time_blocks.central_moments",
    "time_blocks.cumulants",
    "time_blocks.relative_mean_distance",
    "time_blocks.first_last_mean_distance",
    "uncertainty.available",
    "uncertainty.n_independent",
    "uncertainty.time_blocks_are_independent",
)


class QuadraticObservableSpec(BaseModel):
    """One real observable ``alpha^dagger Q alpha - center``."""

    model_config = ConfigDict(extra="forbid")

    matrix: list[list[complex]] = Field(
        description="Hermitian matrix Q in the configured mode order",
    )
    center: float | None = Field(
        None,
        description="Optional scalar center subtracted from alpha^dagger Q alpha",
    )
    reference_matrix: list[list[complex]] | None = Field(
        None,
        description="Optional Hermitian R_ref defining center=Tr(Q R_ref)",
    )

    @model_validator(mode="after")
    def validate_center(self) -> QuadraticObservableSpec:
        if self.center is not None and self.reference_matrix is not None:
            raise ValueError("center and reference_matrix are mutually exclusive")
        return self

    @field_serializer("matrix", "reference_matrix")
    def serialize_complex_matrix(
        self, value: list[list[complex]] | None
    ) -> list[list[str]] | None:
        if value is None:
            return None
        return [[str(entry) for entry in row] for row in value]


class QuadraticMomentConfig(PluginConfigBase):
    """Configuration for Hermitian quadratic-observable moments."""

    observables: dict[str, QuadraticObservableSpec] = Field(
        min_length=1,
        description="Named Hermitian quadratic observables",
    )
    modes: list[int] | None = Field(
        None,
        min_length=1,
        description="Physical mode indices; None uses every recorded mode",
    )
    max_order: int = Field(
        4,
        ge=1,
        le=4,
        description="Highest raw moment and cumulant order to retain",
    )
    time_blocks: int = Field(
        8,
        ge=1,
        le=256,
        description="Contiguous blocks retained for stationarity diagnostics",
    )
    min_block_samples: int = Field(
        32,
        ge=2,
        description="Minimum saved samples in each requested time block",
    )
    time_chunk_samples: int = Field(
        8192,
        ge=1,
        description="Maximum saved samples reduced in one backend operation",
    )
    hermitian_tolerance: float = Field(
        1e-10,
        ge=0.0,
        description="Relative tolerance for Hermitian matrix validation",
    )

    @model_validator(mode="after")
    def validate_configuration(self) -> QuadraticMomentConfig:
        if self.modes is not None:
            if len(set(self.modes)) != len(self.modes):
                raise ValueError("modes must not contain duplicates")
            if any(mode < 0 for mode in self.modes):
                raise ValueError("modes must be non-negative")
        if any(not name.strip() for name in self.observables):
            raise ValueError("observable names must not be blank")
        return self


class QuadraticMomentAnalyzer(Analyzer):
    """Estimate moments of ``alpha^dagger Q alpha`` without retaining trajectories."""

    name: ClassVar[str] = "quadratic_moments"
    description: ClassVar[str] = (
        "Raw moments and cumulants of named Hermitian quadratic observables"
    )
    config_schema: ClassVar[type[QuadraticMomentConfig]] = QuadraticMomentConfig

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(
            preferred_location="backend",
            supports_trajectory_batching=True,
            supports_time_streaming=False,
            supports_scan_tiling=True,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        config = cast(QuadraticMomentConfig, self.config)
        n_observables = len(config.observables)
        n_modes = (
            len(config.modes) if config.modes is not None else request.n_record_modes
        )
        chunk = min(config.time_chunk_samples, request.saved_samples)
        chunk_values = (
            request.n_traj * chunk * n_observables * request.real_itemsize * 3
        )
        matrices = n_observables * n_modes * n_modes * request.real_itemsize * 2
        centers = n_observables * request.real_itemsize
        summaries = (
            request.n_traj
            * n_observables
            * config.max_order
            * request.real_itemsize
            * 2
        )
        block_payload = (
            config.time_blocks
            * n_observables
            * config.max_order
            * request.real_itemsize
        )
        total_n_traj = request.total_n_traj or request.n_traj
        batch_count = math.ceil(total_n_traj / request.n_traj)
        trajectory_payload = (
            total_n_traj
            * n_observables
            * config.max_order
            * request.real_itemsize
        )
        fixed_payload = (
            2 * matrices
            + centers
            + block_payload
            + 2 * config.time_blocks * 8
            + n_modes * 8
            + n_observables * 64
        )
        # Finalize holds payloads plus concatenate, then creates leave_raw,
        # cumulant values, and full-shape subtraction/square temporaries. The
        # extra construction-sized term covers the leave_raw expression peak.
        payload_and_concatenate = 2 * trajectory_payload
        leave_raw_and_construction = 2 * trajectory_payload
        cumulant_values = trajectory_payload
        variance_temporaries = 2 * trajectory_payload
        retained_host_bytes = (
            payload_and_concatenate
            + leave_raw_and_construction
            + cumulant_values
            + variance_temporaries
            + batch_count * fixed_payload
        )
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(
                device_bytes=chunk_values + summaries + matrices + centers,
                retained_host_bytes=retained_host_bytes,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=chunk_values + summaries,
            retained_host_bytes=retained_host_bytes,
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
        trajectory = inputs.get("trajectories")
        if trajectory is None:
            raise TypeError("quadratic_moments requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            values, t0, dt, recorded = trajectory_arrays(point)
            modes, columns = select_mode_columns(
                recorded, cast(QuadraticMomentConfig, self.config).modes
            )
            payload = self._compute_payload(
                values,
                modes=modes,
                columns=columns,
                t0=t0,
                dt=dt,
                backend=backend,
            )
            products = _build_quadratic_moments_products(
                payload, scan_size=None, label=label
            )
            if not products:
                raise ValueError("quadratic_moments produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        values: Any,
        *,
        modes: list[int],
        columns: list[int],
        t0: float,
        dt: float,
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(QuadraticMomentConfig, self.config)
        if values.ndim != 3 or not np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError(
                "quadratic_moments expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, _ = map(int, values.shape)
        if n_traj < 1 or n_samples < 1:
            raise ValueError("quadratic_moments requires non-empty trajectories")
        names, matrices, centers, references = _compile_observables(
            config.observables, len(modes), config.hermitian_tolerance
        )
        backend_matrices = backend.asarray(matrices, dtype=values.dtype)
        backend_centers = backend.asarray(centers, dtype=values.real.dtype)
        trajectory_sums = backend.zeros(
            (n_traj, len(names), config.max_order), dtype=values.real.dtype
        )
        boundaries = _block_boundaries(
            n_samples, config.time_blocks, config.min_block_samples
        )
        block_raw_moments: list[np.ndarray] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            block_sums = backend.zeros(
                (n_traj, len(names), config.max_order), dtype=values.real.dtype
            )
            for chunk_start in range(start, stop, config.time_chunk_samples):
                chunk_stop = min(stop, chunk_start + config.time_chunk_samples)
                selected = values[:, chunk_start:chunk_stop, columns]
                observable = backend.real(
                    backend.einsum(
                        "rti,oij,rtj->rto",
                        selected.conj(),
                        backend_matrices,
                        selected,
                    )
                )
                observable -= backend_centers[None, None, :]
                power = observable
                for order in range(config.max_order):
                    block_sums[:, :, order] += backend.einsum("rto->ro", power)
                    if order + 1 < config.max_order:
                        power = power * observable
            trajectory_sums += block_sums
            block_raw_moments.append(
                np.asarray(convert_to_numpy(backend.mean(block_sums, axis=0)))
                / float(stop - start)
            )
        return self._summarize(
            per_trajectory_raw_moments=np.asarray(convert_to_numpy(trajectory_sums))
            / float(n_samples),
            block_raw_moments=np.asarray(block_raw_moments),
            names=names,
            matrices=matrices,
            centers=centers,
            reference_matrices=references,
            modes=modes,
            n_samples=n_samples,
            t0=t0,
            dt=dt,
            boundaries=boundaries,
        )

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                quantity=SDEQuantity.MOMENTS.value,
                fields=list(_QUADRATIC_MOMENT_FIELDS),
            ),
        )

    def create_result_accumulator(self) -> QuadraticMomentResultAccumulator:
        return QuadraticMomentResultAccumulator(self)

    def _summarize(
        self,
        *,
        per_trajectory_raw_moments: np.ndarray,
        block_raw_moments: np.ndarray,
        names: list[str],
        matrices: np.ndarray,
        centers: np.ndarray,
        reference_matrices: np.ndarray,
        modes: list[int],
        n_samples: int,
        t0: float,
        dt: float,
        boundaries: np.ndarray,
    ) -> dict[str, Any]:
        n_traj = int(per_trajectory_raw_moments.shape[0])
        raw_moments = np.mean(per_trajectory_raw_moments, axis=0)
        cumulants = _raw_to_cumulants(raw_moments)
        central_moments = _raw_to_central_moments(raw_moments)
        if n_traj > 1:
            raw_sem = np.std(per_trajectory_raw_moments, axis=0, ddof=1) / math.sqrt(
                n_traj
            )
            cumulant_sem = _cumulant_jackknife_sem(
                per_trajectory_raw_moments, raw_moments
            )
        else:
            raw_sem = np.full(raw_moments.shape, np.nan)
            cumulant_sem = np.full(cumulants.shape, np.nan)

        block_cumulants = _raw_to_cumulants(block_raw_moments)
        block_central = _raw_to_central_moments(block_raw_moments)
        mean_scale = max(float(np.linalg.norm(raw_moments[:, 0])), 1e-14)
        block_distance = (
            np.linalg.norm(block_raw_moments[:, :, 0] - raw_moments[None, :, 0], axis=1)
            / mean_scale
        )
        starts = boundaries[:-1]
        stops = boundaries[1:]
        return {
            "quantity": "hermitian_quadratic_observable_moments",
            "definition": "x_o = alpha^dagger Q_o alpha - center_o",
            "ordering_correction": "none",
            "observable_names": list(names),
            "modes": list(modes),
            "matrices": matrices,
            "centers": centers,
            "reference_matrices": reference_matrices,
            "max_order": int(raw_moments.shape[-1]),
            "n_traj": n_traj,
            "n_samples": int(n_samples),
            "t0": float(t0),
            "dt": float(dt),
            "observation_duration": float(max(0, n_samples - 1) * dt),
            "raw_moments": raw_moments,
            "raw_moment_sem": raw_sem,
            "central_moments": central_moments,
            "cumulants": cumulants,
            "cumulant_sem": cumulant_sem,
            "per_trajectory_raw_moments": per_trajectory_raw_moments,
            "time_blocks": {
                "count": int(block_raw_moments.shape[0]),
                "start_index": starts,
                "stop_index": stops,
                "start_time": t0 + starts * dt,
                "stop_time": t0 + np.maximum(stops - 1, starts) * dt,
                "raw_moments": block_raw_moments,
                "central_moments": block_central,
                "cumulants": block_cumulants,
                "relative_mean_distance": block_distance,
                "first_last_mean_distance": float(
                    np.linalg.norm(
                        block_raw_moments[-1, :, 0] - block_raw_moments[0, :, 0]
                    )
                    / mean_scale
                ),
            },
            "uncertainty": {
                "available": n_traj > 1,
                "independent_unit": "trajectory",
                "n_independent": n_traj,
                "raw_moment_method": "sample_sem_across_trajectory_time_means",
                "cumulant_method": "leave_one_trajectory_out_jackknife",
                "time_blocks_are_independent": False,
            },
        }


def _moment_orders(leaves_arrays: dict[str, np.ndarray]) -> list[int]:
    """Resolve the explicit moment-order coordinates of one stacked payload."""
    raw = leaves_arrays.get("raw_moments")
    if raw is not None and raw.ndim >= 1:
        return list(range(1, int(raw.shape[-1]) + 1))
    scalar = int(np.ravel(np.asarray(leaves_arrays["max_order"]))[0])
    return list(range(1, scalar + 1))


def _build_quadratic_moments_products(
    payload: Any,
    *,
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset] | None:
    """Assemble the typed quadratic-moments product of one analyser payload."""
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    moments = SDEQuantity.MOMENTS.value
    orders = _moment_orders(leaves.arrays)
    kinds: tuple[tuple[str, Literal["raw", "central", "cumulant"]], ...] = (
        ("raw_moments", "raw"),
        ("central_moments", "central"),
        ("cumulants", "cumulant"),
    )
    moment_families = {
        name: SDEMomentFamilySchema(
            family_id="sde-quadratic-moments",
            moment_kind=kind,
            ordering="c_number",
            orders=orders,
        ).model_dump(mode="json")
        for name, kind in kinds
    }
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims={
            "raw_moments": ("channel", "order"),
            "raw_moment_sem": ("channel", "order"),
            "central_moments": ("channel", "order"),
            "cumulants": ("channel", "order"),
            "cumulant_sem": ("channel", "order"),
            "per_trajectory_raw_moments": ("trajectory", "channel", "order"),
        },
        axis_specs={
            "channel": TypedAxisSpec("channel", AxisRole.COMPONENT),
            "order": TypedAxisSpec(
                "order",
                AxisRole.INDEX,
                coordinate="regular",
                start=1.0,
                step=1.0,
            ),
            "trajectory": TypedAxisSpec("trajectory", AxisRole.REALIZATION),
        },
        quantities={
            "raw_moments": moments,
            "central_moments": moments,
            "cumulants": moments,
        },
        uncertainties=[
            UncertaintySchema(
                target="raw_moments",
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable="raw_moment_sem",
            ),
            UncertaintySchema(
                target="cumulants",
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable="cumulant_sem",
            ),
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", source_axis="trajectory")
        ],
        attributes={"graph_ready": True, "moment_families": moment_families},
    )
    if dataset is None:
        return None
    return {label: dataset}


class QuadraticMomentResultAccumulator:
    """Merge trajectory batches and recompute nonlinear cumulants."""

    def __init__(self, analyzer: QuadraticMomentAnalyzer) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def _consume_payload(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            for key in ("observable_names", "modes", "n_samples", "max_order"):
                if first[key] != payload[key]:
                    raise ValueError(f"quadratic-moment batches used different {key}")
            for key in ("matrices", "centers", "reference_matrices"):
                if not np.allclose(first[key], payload[key], rtol=0.0, atol=0.0):
                    raise ValueError(f"quadratic-moment batches used different {key}")
            for key in ("start_index", "stop_index"):
                if not np.array_equal(
                    first["time_blocks"][key], payload["time_blocks"][key]
                ):
                    raise ValueError(
                        "quadratic-moment batches used different time blocks"
                    )
        self.payloads.append(payload)

    def consume_product(self, product: Dataset) -> None:
        def value(name: str) -> Any:
            return product.handle(name).materialize()

        self._consume_payload(
            {
                "observable_names": product.attributes["payload_meta"][
                    "observable_names"
                ],
                "modes": [int(mode) for mode in np.asarray(value("modes"))],
                "matrices": value("matrices"),
                "centers": value("centers"),
                "reference_matrices": value("reference_matrices"),
                "max_order": int(value("max_order")),
                "n_traj": int(value("n_traj")),
                "n_samples": int(value("n_samples")),
                "t0": float(value("t0")),
                "dt": float(value("dt")),
                "per_trajectory_raw_moments": value("per_trajectory_raw_moments"),
                "time_blocks": {
                    "start_index": value("time_blocks.start_index"),
                    "stop_index": value("time_blocks.stop_index"),
                    "raw_moments": value("time_blocks.raw_moments"),
                },
            }
        )

    def merge(self, other: QuadraticMomentResultAccumulator) -> None:
        for payload in other.payloads:
            self._consume_payload(payload)

    def finalize_product(self, label: str) -> Mapping[str, Dataset] | None:
        return _build_quadratic_moments_products(
            self._finalize_payload(), scan_size=None, label=label
        )

    def checkpoint(self) -> dict[str, Any]:
        return {"payloads": list(self.payloads)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        payloads = checkpoint.get("payloads")
        if not isinstance(payloads, list):
            raise TypeError("quadratic moments checkpoint is missing payloads")
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("quadratic moments checkpoint payload is invalid")
            self._consume_payload(payload)

    def _finalize_payload(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty quadratic-moment accumulator")
        first = self.payloads[0]
        counts = np.asarray([int(item["n_traj"]) for item in self.payloads])
        total = int(np.sum(counts))
        block_raw = sum(
            np.asarray(item["time_blocks"]["raw_moments"]) * count
            for item, count in zip(self.payloads, counts, strict=True)
        ) / float(total)
        return self.analyzer._summarize(
            per_trajectory_raw_moments=np.concatenate(
                [
                    np.asarray(item["per_trajectory_raw_moments"])
                    for item in self.payloads
                ],
                axis=0,
            ),
            block_raw_moments=block_raw,
            names=list(first["observable_names"]),
            matrices=np.asarray(first["matrices"]),
            centers=np.asarray(first["centers"]),
            reference_matrices=np.asarray(first["reference_matrices"]),
            modes=list(first["modes"]),
            n_samples=int(first["n_samples"]),
            t0=float(first["t0"]),
            dt=float(first["dt"]),
            boundaries=np.concatenate(
                (
                    np.asarray(first["time_blocks"]["start_index"]),
                    np.asarray(first["time_blocks"]["stop_index"])[-1:],
                )
            ),
        )


def _compile_observables(
    specs: dict[str, QuadraticObservableSpec],
    n_modes: int,
    tolerance: float,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names: list[str] = []
    matrices: list[np.ndarray] = []
    centers: list[float] = []
    references: list[np.ndarray] = []
    for name, spec in specs.items():
        matrix = _validate_hermitian(
            spec.matrix, n_modes, tolerance, f"observable {name!r} matrix"
        )
        if spec.reference_matrix is None:
            reference = np.zeros_like(matrix)
            center = 0.0 if spec.center is None else float(spec.center)
        else:
            reference = _validate_hermitian(
                spec.reference_matrix,
                n_modes,
                tolerance,
                f"observable {name!r} reference_matrix",
            )
            traced = np.trace(matrix @ reference)
            if abs(float(np.imag(traced))) > tolerance * max(
                1.0, abs(float(np.real(traced)))
            ):
                raise ValueError(f"observable {name!r} reference center is not real")
            center = float(np.real(traced))
        names.append(name)
        matrices.append(matrix)
        centers.append(center)
        references.append(reference)
    return (
        names,
        np.asarray(matrices),
        np.asarray(centers),
        np.asarray(references),
    )


def _validate_hermitian(
    value: Any, n_modes: int, tolerance: float, label: str
) -> np.ndarray:
    matrix = np.asarray(value, dtype=complex)
    if matrix.shape != (n_modes, n_modes):
        raise ValueError(f"{label} must have shape ({n_modes}, {n_modes})")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain finite values")
    scale = max(float(np.linalg.norm(matrix)), 1.0)
    residual = float(np.linalg.norm(matrix - matrix.conj().T)) / scale
    if residual > tolerance:
        raise ValueError(f"{label} is not Hermitian: relative residual {residual:.3e}")
    return (matrix + matrix.conj().T) / 2.0


def _raw_to_cumulants(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    result = np.empty_like(raw)
    mean = raw[..., 0]
    result[..., 0] = mean
    if raw.shape[-1] >= 2:
        result[..., 1] = raw[..., 1] - mean**2
    if raw.shape[-1] >= 3:
        result[..., 2] = raw[..., 2] - 3.0 * raw[..., 1] * mean + 2.0 * mean**3
    if raw.shape[-1] >= 4:
        result[..., 3] = (
            raw[..., 3]
            - 4.0 * raw[..., 2] * mean
            - 3.0 * raw[..., 1] ** 2
            + 12.0 * raw[..., 1] * mean**2
            - 6.0 * mean**4
        )
    return result


def _raw_to_central_moments(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    result = np.empty_like(raw)
    mean = raw[..., 0]
    result[..., 0] = mean
    if raw.shape[-1] >= 2:
        result[..., 1] = raw[..., 1] - mean**2
    if raw.shape[-1] >= 3:
        result[..., 2] = raw[..., 2] - 3.0 * raw[..., 1] * mean + 2.0 * mean**3
    if raw.shape[-1] >= 4:
        result[..., 3] = (
            raw[..., 3]
            - 4.0 * raw[..., 2] * mean
            + 6.0 * raw[..., 1] * mean**2
            - 3.0 * mean**4
        )
    return result


def _cumulant_jackknife_sem(
    per_trajectory: np.ndarray, raw_moments: np.ndarray
) -> np.ndarray:
    count = int(per_trajectory.shape[0])
    leave_raw = (count * raw_moments[None, :, :] - per_trajectory) / float(count - 1)
    values = _raw_to_cumulants(leave_raw)
    mean = np.mean(values, axis=0)
    return np.sqrt(
        (count - 1) / count * np.sum((values - mean[None, :, :]) ** 2, axis=0)
    )


def _block_boundaries(n_samples: int, requested: int, minimum: int) -> np.ndarray:
    maximum = max(1, n_samples // minimum)
    count = min(requested, maximum)
    return np.linspace(0, n_samples, count + 1, dtype=int)
