"""qphase_sde: Polar Distribution Analyser
---------------------------------------------------------
Compute longitudinal (magnitude) distribution from multi-trajectory time series.
Includes peak finding and width estimation for the distribution.

Public API
----------
``PolarDistAnalyser`` : Polar Distribution analyser.
``PolarDistAnalyserConfig`` : Configuration for Polar Dist analyser.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal, cast

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
)

from ..products import TypedAxisSpec, assemble_typed_product, stack_payload_leaves
from .base import (
    Analyser,
    map_trajectory_product,
    select_mode_columns,
    trajectory_arrays,
    trajectory_requirement,
)

__all__ = [
    "PolarDistAnalyser",
    "PolarDistAnalyserConfig",
]


class PolarDistAnalyserConfig(PluginConfigBase):
    """Configuration for Polar Distribution Analyser.

    Analyzes the distribution of the magnitude (r = |z|) of the modes.
    """

    modes: list[int] = Field(..., description="Mode indices for analysis")
    bins: (
        int
        | Literal["auto", "fd", "doane", "scott", "stone", "rice", "sturges", "sqrt"]
    ) = Field("auto", description="Number of bins or method string for histogram")
    range: list[tuple[float, float]] | None = Field(
        None,
        description="Range for each mode. If None, auto-calculated from data.",
    )
    density: bool = Field(True, description="Normalize histogram to form a PDF")

    @model_validator(mode="after")
    def validate_modes(self) -> "PolarDistAnalyserConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


def _build_polar_dist_products(
    payload: Any,
    *,
    modes: list[int],
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset] | None:
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    declared_dims: dict[str, tuple[str, ...]] = {"modes": ("mode",)}
    axis_specs: dict[str, TypedAxisSpec] = {
        "mode": TypedAxisSpec("mode", AxisRole.COMPONENT)
    }
    for mode in modes:
        prefix = f"distributions.{mode}"
        declared_dims[f"{prefix}.hist"] = ("bin",)
        declared_dims[f"{prefix}.edges"] = ("edge",)
        declared_dims[f"{prefix}.bin_centers"] = ("bin",)
        declared_dims[f"{prefix}.mean"] = ()
        declared_dims[f"{prefix}.std"] = ()
    axis_specs["bin"] = TypedAxisSpec("bin", AxisRole.COORDINATE)
    axis_specs["edge"] = TypedAxisSpec("edge", AxisRole.COORDINATE)
    product = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims=declared_dims,
        axis_specs=axis_specs,
        attributes={"graph_ready": True, "typed_builder": "pdist/1"},
    )
    return None if product is None else {label: product}


class PolarDistAnalyser(Analyser):
    """Analyser for Polar (Longitudinal/Magnitude) Distribution."""

    name: ClassVar[str] = "polar_dist"
    description: ClassVar[str] = "Magnitude Distribution Analyser with Peak Finding"
    config_schema: ClassVar[type[PolarDistAnalyserConfig]] = PolarDistAnalyserConfig

    def __init__(self, config: PolarDistAnalyserConfig | None = None, **kwargs):
        super().__init__(config, **kwargs)

    def input_requirements(self) -> list[ProductRequirement]:
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
            raise TypeError("polar_dist requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            data, _, _, recorded = trajectory_arrays(point)
            modes, columns = select_mode_columns(
                recorded, cast(PolarDistAnalyserConfig, self.config).modes
            )
            payload = self._compute_payload(data, modes, columns, backend)
            products = _build_polar_dist_products(
                payload, modes=modes, scan_size=None, label=label
            )
            if not products:
                raise ValueError("polar_dist produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        data: Any,
        modes: list[int],
        mode_columns: list[int],
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(PolarDistAnalyserConfig, self.config)
        bins = config.bins
        density = config.density
        range_list = config.range
        use_backend = hasattr(backend, "histogram") and hasattr(backend, "std")
        if not use_backend:
            data_np = convert_to_numpy(data)
        results = {}
        resolved_bins = bins
        for index, (mode, column) in enumerate(zip(modes, mode_columns, strict=True)):
            if use_backend:
                samples = backend.abs(data[:, :, column].reshape(-1))
            else:
                samples = np.abs(data_np[:, :, column].flatten())
            selected_range = (
                range_list[index] if range_list and index < len(range_list) else None
            )
            if use_backend:
                if isinstance(resolved_bins, str):
                    edges = np.histogram_bin_edges(
                        convert_to_numpy(samples),
                        bins=resolved_bins,
                        range=selected_range,
                    )
                    resolved_bins = len(edges) - 1
                hist, edges = backend.histogram(
                    samples,
                    bins=resolved_bins,
                    range=selected_range,
                    density=density,
                )
                hist, edges = convert_to_numpy(hist), convert_to_numpy(edges)
                mean = float(convert_to_numpy(backend.mean(samples)))
                std = float(convert_to_numpy(backend.std(samples)))
            else:
                hist, edges = np.histogram(
                    samples,
                    bins=resolved_bins,
                    range=selected_range,
                    density=density,
                )
                mean, std = float(np.mean(samples)), float(np.std(samples))
            results[mode] = {
                "hist": hist,
                "edges": edges,
                "bin_centers": (edges[:-1] + edges[1:]) / 2,
                "mean": mean,
                "std": std,
            }
        return {
            "distributions": results,
            "modes": modes,
            "bins_config": resolved_bins,
            "density": density,
        }

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        config = cast(PolarDistAnalyserConfig, self.config)
        fields: list[str] = []
        if isinstance(config.bins, int):
            fields.append("bins_config")
        for mode in config.modes:
            fields += [
                f"distributions.{mode}.hist",
                f"distributions.{mode}.edges",
                f"distributions.{mode}.bin_centers",
                f"distributions.{mode}.mean",
                f"distributions.{mode}.std",
            ]
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=fields,
            ),
        )
