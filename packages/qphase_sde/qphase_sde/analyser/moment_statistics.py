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
    Analyser,
    map_trajectory_product,
    select_mode_columns,
    trajectory_arrays,
    trajectory_requirement,
)

__all__ = ["MomentStatisticsAnalyser", "MomentStatisticsConfig"]

_MOMENT_STATISTICS_FIELDS: tuple[str, ...] = (
    "modes",
    "n_traj",
    "n_samples",
    "t0",
    "dt",
    "observation_duration",
    "occupation",
    "occupation_sem",
    "fourth_moment",
    "fourth_moment_sem",
    "occupation_product",
    "occupation_product_sem",
    "occupation_covariance",
    "g2",
    "g2_sem",
    "per_trajectory_occupation",
    "per_trajectory_occupation_product",
    "time_blocks.count",
    "time_blocks.start_index",
    "time_blocks.stop_index",
    "time_blocks.start_time",
    "time_blocks.stop_time",
    "time_blocks.occupation",
    "time_blocks.fourth_moment",
    "time_blocks.occupation_product",
    "time_blocks.g2",
    "time_blocks.relative_occupation_distance",
    "time_blocks.first_last_occupation_distance",
    "uncertainty.available",
    "uncertainty.n_independent",
    "uncertainty.time_blocks_are_independent",
)


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


class MomentStatisticsAnalyser(Analyser):
    """Estimate occupation products without materializing full intensity data."""

    name: ClassVar[str] = "moment_statistics"
    description: ClassVar[str] = (
        "C-number occupation, fourth-order moments, covariance, and g2"
    )
    config_schema: ClassVar[type[MomentStatisticsConfig]] = MomentStatisticsConfig

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(
            preferred_location="backend",
            supports_trajectory_batching=True,
            supports_time_streaming=False,
            supports_scan_tiling=True,
        )

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        config = cast(MomentStatisticsConfig, self.config)
        n_modes = (
            len(config.modes) if config.modes is not None else request.n_record_modes
        )
        chunk = min(config.time_chunk_samples, request.saved_samples)
        # Selected complex values, real intensities, and conservative contraction
        # workspace. These arrays are bounded by time_chunk_samples.
        backend_workspace = request.n_traj * chunk * n_modes * request.real_itemsize * 4
        summaries = (
            request.n_traj * (n_modes + n_modes * n_modes) * request.real_itemsize * 2
        )
        block_payload = (
            config.time_blocks * (n_modes + n_modes * n_modes) * request.real_itemsize
        )
        total_n_traj = request.total_n_traj or request.n_traj
        batch_count = math.ceil(total_n_traj / request.n_traj)
        occupation_payload = total_n_traj * n_modes * request.real_itemsize
        product_payload = (
            total_n_traj * n_modes * n_modes * request.real_itemsize
        )
        trajectory_payload = occupation_payload + product_payload
        fixed_payload = (
            block_payload
            + 2 * config.time_blocks * 8
            + n_modes * 8
        )
        # Finalize holds the batch payloads and concatenated arrays together.
        # The g2 jackknife then adds leave-one-out occupation/product arrays,
        # product-shaped values, and conservative finite/arithmetic temporaries.
        payload_and_concatenate = 2 * trajectory_payload
        jackknife_workspace = occupation_payload + 5 * product_payload
        retained_host_bytes = (
            payload_and_concatenate
            + jackknife_workspace
            + batch_count * fixed_payload
        )
        if request.backend_name == "cupy":
            return AnalyserWorkspaceEstimate(
                device_bytes=backend_workspace + summaries,
                retained_host_bytes=retained_host_bytes,
            )
        return AnalyserWorkspaceEstimate(
            host_bytes=backend_workspace + summaries,
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
            raise TypeError("moment_statistics requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            values, t0, dt, recorded = trajectory_arrays(point)
            modes, columns = select_mode_columns(
                recorded, cast(MomentStatisticsConfig, self.config).modes
            )
            payload = self._compute_payload(
                values,
                modes=modes,
                columns=columns,
                t0=t0,
                dt=dt,
                backend=backend,
            )
            products = _build_moment_statistics_products(
                payload, scan_size=None, label=label
            )
            if not products:
                raise ValueError("moment_statistics produced no typed product")
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
        config = cast(MomentStatisticsConfig, self.config)
        if values.ndim != 3 or not np.issubdtype(values.dtype, np.complexfloating):
            raise ValueError(
                "moment_statistics expects complex shape (n_traj, n_time, n_modes)"
            )
        n_traj, n_samples, _ = map(int, values.shape)
        if n_traj < 1 or n_samples < 1:
            raise ValueError("moment_statistics requires non-empty trajectories")
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
                np.asarray(convert_to_numpy(backend.mean(block_product_sum, axis=0)))
                / count
            )
        return self._summarize(
            per_trajectory=np.asarray(convert_to_numpy(trajectory_sum))
            / float(n_samples),
            per_trajectory_product=np.asarray(convert_to_numpy(trajectory_product_sum))
            / float(n_samples),
            block_occupation=np.asarray(block_occupation),
            block_product=np.asarray(block_product),
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
                fields=list(_MOMENT_STATISTICS_FIELDS),
            ),
        )

    def create_result_accumulator(self) -> MomentStatisticsResultAccumulator:
        return MomentStatisticsResultAccumulator(self)

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
        g2 = _normalized_product(product, occupation, config.denominator_tolerance)

        if n_traj > 1:
            occupation_sem = np.std(per_trajectory, axis=0, ddof=1) / math.sqrt(n_traj)
            product_sem = np.std(per_trajectory_product, axis=0, ddof=1) / math.sqrt(
                n_traj
            )
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
    scan_size: int | None,
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

    def __init__(self, analyzer: MomentStatisticsAnalyser) -> None:
        self.analyzer = analyzer
        self.payloads: list[dict[str, Any]] = []

    def _consume_payload(self, payload: dict[str, Any]) -> None:
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

    def consume_product(self, product: Dataset) -> None:
        def value(name: str) -> Any:
            return product.handle(name).materialize()

        self._consume_payload(
            {
                "modes": [int(mode) for mode in np.asarray(value("modes"))],
                "n_traj": int(value("n_traj")),
                "n_samples": int(value("n_samples")),
                "t0": float(value("t0")),
                "dt": float(value("dt")),
                "per_trajectory_occupation": value("per_trajectory_occupation"),
                "per_trajectory_occupation_product": value(
                    "per_trajectory_occupation_product"
                ),
                "time_blocks": {
                    "start_index": value("time_blocks.start_index"),
                    "stop_index": value("time_blocks.stop_index"),
                    "occupation": value("time_blocks.occupation"),
                    "occupation_product": value("time_blocks.occupation_product"),
                },
            }
        )

    def merge(self, other: MomentStatisticsResultAccumulator) -> None:
        for payload in other.payloads:
            self._consume_payload(payload)

    def finalize_product(self, label: str) -> Mapping[str, Dataset] | None:
        return _build_moment_statistics_products(
            self._finalize_payload(), scan_size=None, label=label
        )

    def checkpoint(self) -> dict[str, Any]:
        return {"payloads": list(self.payloads)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        payloads = checkpoint.get("payloads")
        if not isinstance(payloads, list):
            raise TypeError("moment statistics checkpoint is missing payloads")
        for payload in payloads:
            if not isinstance(payload, dict):
                raise TypeError("moment statistics checkpoint payload is invalid")
            self._consume_payload(payload)

    def _finalize_payload(self) -> dict[str, Any]:
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
    leave_occupation = (count * occupation[None, :] - per_trajectory) / (count - 1)
    leave_product = (count * product[None, :, :] - per_trajectory_product) / (count - 1)
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
