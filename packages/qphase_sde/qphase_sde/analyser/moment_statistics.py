"""Time-averaged c-number occupation moments and intensity correlations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import Field, model_validator
from qphase.backend.base import BackendBase
from qphase.backend.xputil import convert_to_numpy
from qphase.core.protocols import PluginConfigBase
from qphase.data import (
    AxisRole,
    DataKind,
    Dataset,
    SamplingBasisSchema,
    UncertaintySchema,
)

from ..contracts.quantities import SDEMomentFamilySchema, SDEQuantity
from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
    resolve_mode_columns,
)
from .result import AnalysisResult

__all__ = ["MomentStatisticsAnalyzer", "MomentStatisticsConfig"]


class MomentStatisticsConfig(PluginConfigBase):
    """Configuration for c-number occupation and fourth-order moments."""

    modes: list[int] | None = Field(
        None,
        min_length=1,
        description="Physical mode indices; None analyzes every recorded mode",
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
    denominator_tolerance: float = Field(
        1e-14,
        ge=0.0,
        description="Minimum occupation product used to normalize g2",
    )

    @model_validator(mode="after")
    def validate_modes(self) -> MomentStatisticsConfig:
        if self.modes is not None and len(set(self.modes)) != len(self.modes):
            raise ValueError("modes must not contain duplicates")
        return self


class MomentStatisticsAnalyzer(Analyzer):
    """Estimate occupation products without materializing full intensity data."""

    name: ClassVar[str] = "moment_statistics"
    description: ClassVar[str] = (
        "C-number occupation, fourth-order moments, covariance, and g2"
    )
    config_schema: ClassVar[type[MomentStatisticsConfig]] = MomentStatisticsConfig

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
        config = cast(MomentStatisticsConfig, self.config)
        n_modes = (
            len(config.modes)
            if config.modes is not None
            else request.n_record_modes
        )
        chunk = min(config.time_chunk_samples, request.saved_samples)
        # Selected complex values, real intensities, and conservative contraction
        # workspace. These arrays are bounded by time_chunk_samples.
        backend_workspace = (
            request.n_traj
            * chunk
            * n_modes
            * request.real_itemsize
            * 4
        )
        summaries = (
            request.n_traj
            * (n_modes + n_modes * n_modes)
            * request.real_itemsize
            * 2
        )
        blocks = (
            config.time_blocks
            * (n_modes + n_modes * n_modes)
            * request.real_itemsize
        )
        host_bytes = summaries + blocks
        if request.backend_name == "cupy":
            return AnalyzerWorkspaceEstimate(
                device_bytes=backend_workspace + summaries,
                host_bytes=host_bytes,
            )
        return AnalyzerWorkspaceEstimate(
            host_bytes=backend_workspace + 2 * summaries + blocks
        )

    def analyze(self, data: Any, backend: BackendBase) -> AnalysisResult:
        config = cast(MomentStatisticsConfig, self.config)
        values = getattr(data, "data", data)
        if (
            not hasattr(values, "ndim")
            or values.ndim != 3
            or not np.issubdtype(values.dtype, np.complexfloating)
        ):
            raise ValueError(
                "moment_statistics expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, _ = map(int, values.shape)
        if n_traj < 1 or n_samples < 1:
            raise ValueError("moment_statistics requires non-empty trajectories")

        modes = _resolve_modes(data, config.modes, int(values.shape[2]))
        columns = resolve_mode_columns(data, modes)
        n_modes = len(modes)
        real_dtype = values.real.dtype
        trajectory_sum = backend.zeros((n_traj, n_modes), dtype=real_dtype)
        trajectory_product_sum = backend.zeros(
            (n_traj, n_modes, n_modes), dtype=real_dtype
        )

        boundaries = _block_boundaries(
            n_samples, config.time_blocks, config.min_block_samples
        )
        block_occupation: list[np.ndarray] = []
        block_product: list[np.ndarray] = []
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            block_sum = backend.zeros((n_traj, n_modes), dtype=real_dtype)
            block_product_sum = backend.zeros(
                (n_traj, n_modes, n_modes), dtype=real_dtype
            )
            for chunk_start in range(start, stop, config.time_chunk_samples):
                chunk_stop = min(stop, chunk_start + config.time_chunk_samples)
                selected = values[:, chunk_start:chunk_stop, columns]
                intensity = backend.abs(selected)
                intensity *= intensity
                block_sum += backend.einsum("rti->ri", intensity)
                block_product_sum += backend.einsum(
                    "rti,rtj->rij", intensity, intensity
                )

            count = float(stop - start)
            trajectory_sum += block_sum
            trajectory_product_sum += block_product_sum
            block_occupation.append(
                np.asarray(convert_to_numpy(backend.mean(block_sum, axis=0))) / count
            )
            block_product.append(
                np.asarray(
                    convert_to_numpy(backend.mean(block_product_sum, axis=0))
                )
                / count
            )

        per_trajectory = np.asarray(convert_to_numpy(trajectory_sum)) / float(
            n_samples
        )
        per_trajectory_product = np.asarray(
            convert_to_numpy(trajectory_product_sum)
        ) / float(n_samples)
        payload = self._summarize(
            per_trajectory=per_trajectory,
            per_trajectory_product=per_trajectory_product,
            block_occupation=np.asarray(block_occupation),
            block_product=np.asarray(block_product),
            modes=modes,
            n_samples=n_samples,
            t0=float(getattr(data, "t0", 0.0)),
            dt=float(getattr(data, "dt", 1.0)),
            boundaries=boundaries,
        )
        meta = {
            "quantity": payload["quantity"],
            "modes": payload["modes"],
            "n_traj": payload["n_traj"],
            "n_samples": payload["n_samples"],
            "uncertainty_unit": "trajectory",
        }
        return AnalysisResult(data_dict=payload, meta=meta)

    def create_result_accumulator(self) -> MomentStatisticsResultAccumulator:
        return MomentStatisticsResultAccumulator(self)

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int,
        label: str,
    ) -> Mapping[str, Dataset] | None:
        """Build the graph-ready typed product of one ``analyze()`` payload."""
        return _build_moment_statistics_products(
            payload, scan_size=scan_size, label=label
        )

    def _summarize(
        self,
        *,
        per_trajectory: np.ndarray,
        per_trajectory_product: np.ndarray,
        block_occupation: np.ndarray,
        block_product: np.ndarray,
        modes: list[int],
        n_samples: int,
        t0: float,
        dt: float,
        boundaries: np.ndarray,
    ) -> dict[str, Any]:
        config = cast(MomentStatisticsConfig, self.config)
        n_traj = int(per_trajectory.shape[0])
        occupation = np.mean(per_trajectory, axis=0)
        product = _symmetric(np.mean(per_trajectory_product, axis=0))
        covariance = _symmetric(product - np.outer(occupation, occupation))
        g2 = _normalized_product(
            product, occupation, config.denominator_tolerance
        )

        if n_traj > 1:
            occupation_sem = np.std(per_trajectory, axis=0, ddof=1) / math.sqrt(
                n_traj
            )
            product_sem = np.std(
                per_trajectory_product, axis=0, ddof=1
            ) / math.sqrt(n_traj)
            g2_sem = _g2_jackknife_sem(
                per_trajectory,
                per_trajectory_product,
                occupation,
                product,
                config.denominator_tolerance,
            )
        else:
            occupation_sem = np.full(occupation.shape, np.nan)
            product_sem = np.full(product.shape, np.nan)
            g2_sem = np.full(product.shape, np.nan)

        block_product = np.asarray([_symmetric(item) for item in block_product])
        block_g2 = np.asarray(
            [
                _normalized_product(
                    item,
                    block_occupation[index],
                    config.denominator_tolerance,
                )
                for index, item in enumerate(block_product)
            ]
        )
        occupation_scale = max(
            float(np.linalg.norm(occupation)), config.denominator_tolerance
        )
        block_distance = np.asarray(
            [
                float(np.linalg.norm(item - occupation) / occupation_scale)
                for item in block_occupation
            ]
        )
        starts = boundaries[:-1]
        stops = boundaries[1:]
        diagonal = np.arange(len(modes))
        return {
            "quantity": "c_number_occupation_moments",
            "definition": "n_i = |alpha_i|^2; G2_ij = mean(n_i * n_j)",
            "ordering_correction": "none",
            "modes": list(modes),
            "n_traj": n_traj,
            "n_samples": int(n_samples),
            "t0": float(t0),
            "dt": float(dt),
            "observation_duration": float(max(0, n_samples - 1) * dt),
            "occupation": occupation,
            "occupation_sem": occupation_sem,
            "fourth_moment": product[diagonal, diagonal],
            "fourth_moment_sem": product_sem[diagonal, diagonal],
            "occupation_product": product,
            "occupation_product_sem": product_sem,
            "occupation_covariance": covariance,
            "g2": g2,
            "g2_sem": g2_sem,
            "per_trajectory_occupation": per_trajectory,
            "per_trajectory_occupation_product": per_trajectory_product,
            "time_blocks": {
                "count": int(block_occupation.shape[0]),
                "start_index": starts,
                "stop_index": stops,
                "start_time": t0 + starts * dt,
                "stop_time": t0 + np.maximum(stops - 1, starts) * dt,
                "occupation": block_occupation,
                "fourth_moment": block_product[:, diagonal, diagonal],
                "occupation_product": block_product,
                "g2": block_g2,
                "relative_occupation_distance": block_distance,
                "first_last_occupation_distance": float(
                    np.linalg.norm(block_occupation[-1] - block_occupation[0])
                    / occupation_scale
                ),
            },
            "uncertainty": {
                "available": n_traj > 1,
                "independent_unit": "trajectory",
                "n_independent": n_traj,
                "moment_method": "sample_sem_across_trajectory_time_means",
                "g2_method": "leave_one_trajectory_out_jackknife",
                "time_blocks_are_independent": False,
            },
        }


def _build_moment_statistics_products(
    payload: Any,
    *,
    scan_size: int,
    label: str,
) -> dict[str, Dataset] | None:
    """Assemble the typed moment-statistics product of one analyser payload."""
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    moments = SDEQuantity.MOMENTS.value
    dataset = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims={
            "occupation": ("channel",),
            "occupation_sem": ("channel",),
            "fourth_moment": ("channel",),
            "fourth_moment_sem": ("channel",),
            "occupation_product": ("channel", "channel_2"),
            "occupation_product_sem": ("channel", "channel_2"),
            "occupation_covariance": ("channel", "channel_2"),
            "g2": ("channel", "channel_2"),
            "g2_sem": ("channel", "channel_2"),
            "per_trajectory_occupation": ("trajectory", "channel"),
            "per_trajectory_occupation_product": (
                "trajectory",
                "channel",
                "channel_2",
            ),
        },
        axis_specs={
            "channel": TypedAxisSpec("channel", AxisRole.COMPONENT),
            "channel_2": TypedAxisSpec("channel_2", AxisRole.COMPONENT),
            "trajectory": TypedAxisSpec("trajectory", AxisRole.REALIZATION),
        },
        quantities={
            "occupation": moments,
            "fourth_moment": moments,
            "occupation_product": moments,
            "g2": moments,
        },
        uncertainties=[
            UncertaintySchema(
                target=target,
                kind="sem",
                sampling_basis="trajectory",
                covariance="real",
                scope="sampling",
                data_variable=data_variable,
            )
            for target, data_variable in (
                ("occupation", "occupation_sem"),
                ("fourth_moment", "fourth_moment_sem"),
                ("occupation_product", "occupation_product_sem"),
                ("g2", "g2_sem"),
            )
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", source_axis="trajectory")
        ],
        attributes={
            "graph_ready": True,
            "moment_family": SDEMomentFamilySchema(
                family_id="sde-occupation-moments",
                moment_kind="raw",
                ordering="normal",
                orders=[1, 2, 4],
            ).model_dump(mode="json"),
            # g2 is a normalized ratio of the order-2 family, not a raw moment.
            "normalized_variables": ["g2"],
        },
    )
    if dataset is None:
        return None
    return {label: dataset}


class MomentStatisticsResultAccumulator:
    """Merge independent trajectory batches and recompute nonlinear ratios."""

    def __init__(self, analyzer: MomentStatisticsAnalyzer) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def update(self, payload: dict[str, Any]) -> None:
        if self.payloads:
            first = self.payloads[0]
            if first["modes"] != payload["modes"]:
                raise ValueError("moment-statistics batches used different modes")
            if first["n_samples"] != payload["n_samples"]:
                raise ValueError("moment-statistics batches used different time grids")
            for key in ("start_index", "stop_index"):
                if not np.array_equal(
                    first["time_blocks"][key], payload["time_blocks"][key]
                ):
                    raise ValueError(
                        "moment-statistics batches used different time blocks"
                    )
        self.payloads.append(payload)

    def finalize(self) -> dict[str, Any]:
        if not self.payloads:
            raise RuntimeError("cannot finalize an empty moment-statistics accumulator")
        first = self.payloads[0]
        counts = np.asarray([int(item["n_traj"]) for item in self.payloads])
        total = int(np.sum(counts))
        block_occupation = sum(
            np.asarray(item["time_blocks"]["occupation"]) * count
            for item, count in zip(self.payloads, counts, strict=True)
        ) / float(total)
        block_product = sum(
            np.asarray(item["time_blocks"]["occupation_product"]) * count
            for item, count in zip(self.payloads, counts, strict=True)
        ) / float(total)
        return self.analyzer._summarize(
            per_trajectory=np.concatenate(
                [
                    np.asarray(item["per_trajectory_occupation"])
                    for item in self.payloads
                ],
                axis=0,
            ),
            per_trajectory_product=np.concatenate(
                [
                    np.asarray(item["per_trajectory_occupation_product"])
                    for item in self.payloads
                ],
                axis=0,
            ),
            block_occupation=block_occupation,
            block_product=block_product,
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


def _resolve_modes(data: Any, configured: list[int] | None, stored: int) -> list[int]:
    if configured is not None:
        return list(configured)
    meta = getattr(data, "meta", None)
    mode_indices = meta.get("mode_indices") if isinstance(meta, dict) else None
    if mode_indices is None:
        return list(range(stored))
    if len(mode_indices) != stored:
        raise ValueError("trajectory mode_indices do not match the stored mode count")
    return [int(mode) for mode in mode_indices]


def _block_boundaries(n_samples: int, requested: int, minimum: int) -> np.ndarray:
    maximum = max(1, n_samples // minimum)
    count = min(requested, maximum)
    return np.linspace(0, n_samples, count + 1, dtype=int)


def _symmetric(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    return (matrix + matrix.T) / 2.0


def _normalized_product(
    product: np.ndarray, occupation: np.ndarray, tolerance: float
) -> np.ndarray:
    denominator = np.outer(occupation, occupation)
    result = np.full(product.shape, np.nan, dtype=float)
    np.divide(product, denominator, out=result, where=denominator > tolerance)
    return _symmetric(result)


def _g2_jackknife_sem(
    per_trajectory: np.ndarray,
    per_trajectory_product: np.ndarray,
    occupation: np.ndarray,
    product: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    count = int(per_trajectory.shape[0])
    leave_occupation = (
        count * occupation[None, :] - per_trajectory
    ) / (count - 1)
    leave_product = (
        count * product[None, :, :] - per_trajectory_product
    ) / (count - 1)
    values = np.asarray(
        [
            _normalized_product(item, leave_occupation[index], tolerance)
            for index, item in enumerate(leave_product)
        ]
    )
    finite = np.isfinite(values)
    finite_count = np.sum(finite, axis=0)
    mean = np.full(values.shape[1:], np.nan, dtype=float)
    np.divide(
        np.sum(np.where(finite, values, 0.0), axis=0),
        finite_count,
        out=mean,
        where=finite_count > 0,
    )
    sem = np.sqrt(
        (count - 1) / count * np.nansum((values - mean[None, :, :]) ** 2, axis=0)
    )
    sem[finite_count == 0] = np.nan
    return sem
