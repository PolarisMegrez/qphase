"""qphase_sde: Distribution Analyser
---------------------------------------------------------
Compute phase space distribution (histogram) from multi-trajectory time series.

Public API
----------
``DistAnalyser`` : Distribution analyzer.
``DistAnalyserConfig`` : Configuration for Distribution analyzer.
"""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as _np
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

from ..products import (
    TypedAxisSpec,
    assemble_typed_product,
    stack_payload_leaves,
)
from .base import (
    Analyser,
    map_trajectory_product,
    select_mode_columns,
    trajectory_arrays,
    trajectory_requirement,
)

__all__ = [
    "DistAnalyser",
    "DistAnalyserConfig",
]


def _is_complex(arr: Any) -> bool:
    """Return True if ``arr`` has a complex dtype, backend-agnostic."""
    dtype = getattr(arr, "dtype", None)
    if dtype is None:
        return False
    return str(getattr(dtype, "kind", dtype)) in {"c", "complex"}


class DistAnalyserConfig(PluginConfigBase):
    """Configuration for Distribution Analyser."""

    modes: list[int] = Field(..., description="Mode indices for analysis")
    bins: int = Field(50, description="Number of bins for histogram")
    range: list[list[float]] | None = Field(
        None, description="Range for each mode [[min, max], ...]"
    )
    density: bool = Field(True, description="Normalize histogram to form a PDF")

    @model_validator(mode="after")
    def validate_modes(self) -> "DistAnalyserConfig":
        if not self.modes:
            raise ValueError("modes must be non-empty")
        return self


def _build_dist_products(
    payload: Any,
    *,
    modes: list[int],
    scan_size: int | None,
    label: str,
) -> dict[str, Dataset] | None:
    leaves = stack_payload_leaves(label, payload, scan_size=scan_size)
    if leaves is None:
        return None
    sample_payload = payload[0] if isinstance(payload, list) else payload
    distributions = (
        sample_payload.get("distributions", {})
        if isinstance(sample_payload, dict)
        else {}
    )
    declared_dims: dict[str, tuple[str, ...]] = {"modes": ("mode",)}
    axis_specs: dict[str, TypedAxisSpec] = {
        "mode": TypedAxisSpec("mode", AxisRole.COMPONENT)
    }
    for mode in modes:
        prefix = f"distributions.{mode}"
        kind = distributions.get(mode, {}).get("type")
        if kind == "2d_complex":
            declared_dims[f"{prefix}.hist"] = ("bin_x", "bin_y")
            axis_specs["bin_x"] = TypedAxisSpec("bin_x", AxisRole.COORDINATE)
            axis_specs["bin_y"] = TypedAxisSpec("bin_y", AxisRole.COORDINATE)
            declared_dims[f"{prefix}.xedges"] = ("edge_x",)
            declared_dims[f"{prefix}.yedges"] = ("edge_y",)
            axis_specs["edge_x"] = TypedAxisSpec("edge_x", AxisRole.COORDINATE)
            axis_specs["edge_y"] = TypedAxisSpec("edge_y", AxisRole.COORDINATE)
        else:
            declared_dims[f"{prefix}.hist"] = ("bin",)
            declared_dims[f"{prefix}.edges"] = ("edge",)
            axis_specs["bin"] = TypedAxisSpec("bin", AxisRole.COORDINATE)
            axis_specs["edge"] = TypedAxisSpec("edge", AxisRole.COORDINATE)
    product = assemble_typed_product(
        label,
        leaves,
        scan_size=scan_size,
        kind=DataKind.STATISTICS,
        declared_dims=declared_dims,
        axis_specs=axis_specs,
        attributes={"graph_ready": True, "typed_builder": "dist/1"},
    )
    return None if product is None else {label: product}


class DistAnalyser(Analyser):
    """Analyser for Phase Space Distribution."""

    name: ClassVar[str] = "dist"
    description: ClassVar[str] = "Phase Space Distribution analyzer"
    config_schema: ClassVar[type[DistAnalyserConfig]] = DistAnalyserConfig

    def __init__(self, config: DistAnalyserConfig | None = None, **kwargs):
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
            raise TypeError("dist requires a trajectories product")

        def compute(point: Dataset) -> Mapping[str, Dataset]:
            data, _, _, recorded = trajectory_arrays(point)
            modes, columns = select_mode_columns(
                recorded, cast(DistAnalyserConfig, self.config).modes
            )
            payload = self._compute_payload(data, modes, columns, backend)
            products = _build_dist_products(
                payload, modes=modes, scan_size=None, label=label
            )
            if not products:
                raise ValueError("dist produced no typed product")
            return products

        return map_trajectory_product(trajectory, compute, context=context)

    def _compute_payload(
        self,
        data: Any,
        modes: list[int],
        mode_columns: list[int],
        backend: BackendBase,
    ) -> dict[str, Any]:
        config = cast(DistAnalyserConfig, self.config)
        bins = config.bins
        density = config.density
        range_list = config.range
        use_backend_hist = hasattr(backend, "histogram") and hasattr(
            backend, "histogram2d"
        )
        if not use_backend_hist:
            data_np = convert_to_numpy(data)
        results = {}
        for index, (mode, column) in enumerate(zip(modes, mode_columns, strict=True)):
            samples = (
                data[:, :, column].reshape(-1)
                if use_backend_hist
                else data_np[:, :, column].flatten()
            )
            is_complex = (
                _is_complex(samples) if use_backend_hist else _np.iscomplexobj(samples)
            )
            if is_complex:
                x = backend.real(samples) if use_backend_hist else samples.real
                y = backend.imag(samples) if use_backend_hist else samples.imag
                selected_range = None
                if range_list and index < len(range_list):
                    item = cast(tuple[float, float], tuple(range_list[index]))
                    selected_range = [item, item]
                if use_backend_hist:
                    hist, xedges, yedges = backend.histogram2d(
                        x, y, bins=bins, range=selected_range, density=density
                    )
                    hist = convert_to_numpy(hist)
                    xedges = convert_to_numpy(xedges)
                    yedges = convert_to_numpy(yedges)
                else:
                    hist, xedges, yedges = _np.histogram2d(
                        x, y, bins=bins, range=selected_range, density=density
                    )
                results[mode] = {
                    "hist": hist,
                    "xedges": xedges,
                    "yedges": yedges,
                    "type": "2d_complex",
                }
            else:
                selected_range_1d: tuple[float, float] | None = None
                if range_list and index < len(range_list):
                    selected_range_1d = cast(
                        tuple[float, float], tuple(range_list[index])
                    )
                if use_backend_hist:
                    hist, edges = backend.histogram(
                        samples,
                        bins=bins,
                        range=selected_range_1d,
                        density=density,
                    )
                    hist, edges = convert_to_numpy(hist), convert_to_numpy(edges)
                else:
                    hist, edges = _np.histogram(
                        samples,
                        bins=bins,
                        range=selected_range_1d,
                        density=density,
                    )
                results[mode] = {
                    "hist": hist,
                    "edges": edges,
                    "type": "1d_real",
                }
        return {
            "distributions": results,
            "modes": modes,
            "bins": bins,
            "density": density,
        }

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        config = cast(DistAnalyserConfig, self.config)
        fields: list[str] = []
        for mode in config.modes:
            fields += [
                f"distributions.{mode}.hist",
                f"distributions.{mode}.xedges",
                f"distributions.{mode}.yedges",
            ]
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=fields,
            ),
        )
