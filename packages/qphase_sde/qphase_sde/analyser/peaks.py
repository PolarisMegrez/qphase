"""Standalone grouped peak analysis from a typed spectral product."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.core.protocols import (
    PluginConfigBase,
    PluginManifest,
    ResultProtocol,
    SubpluginSlot,
)
from qphase.data import DataKind, Dataset, ProductDeclaration, ProductRequirement

from ..contracts.analyser import AnalyserExecutionCapabilities, WorkEstimate
from ..peak_finder import materialize_candidate_product
from .base import Analyzer, check_execution_cancelled, report_execution_progress
from .result import AnalysisResult
from .result_input import load_sde_results

__all__ = ["PeaksAnalyzer", "PeaksAnalyzerConfig"]


class PeaksAnalyzerConfig(PluginConfigBase):
    """Configuration for the standalone PSD peak analyser."""

    model_config = ConfigDict(extra="forbid")

    scan_param: str = Field(..., description="Parameter used as the scan axis")
    psd_key: str = Field("psd", description="Spectral input product key")
    modes: list[int] | None = Field(
        None, description="Physical modes to analyse; otherwise use PSD metadata"
    )
    finder: dict[str, Any] = Field(
        default_factory=lambda: {"scipy": {}},
        description="One selected typed peak-finder child configuration",
    )


class PeaksAnalyzer(Analyzer):
    """Run a typed peak finder independently of PSD estimation."""

    name: ClassVar[str] = "peaks"
    description: ClassVar[str] = "Grouped peak candidates from a spectral product"
    config_schema: ClassVar[type[PeaksAnalyzerConfig]] = PeaksAnalyzerConfig
    preserves_input_scan: ClassVar[bool] = True
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "finder": SubpluginSlot(
                namespace="peak_finder",
                default="scipy",
                protocol="qphase_sde.peak_finder.base:TypedPeakFinder",
                description="Peak candidate finder child",
            )
        }
    )

    def __init__(
        self,
        config: PeaksAnalyzerConfig | None = None,
        *,
        subplugins: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._finder = None if subplugins is None else subplugins.get("finder")

    def _resolve_finder(self) -> Any:
        if self._finder is None:
            raise RuntimeError(
                "peaks requires a resolved peak_finder child; construct it "
                "through the plugin graph or pass subplugins"
            )
        return self._finder

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(PeaksAnalyzerConfig, self.config)
        return [
            ProductRequirement(
                name=config.psd_key,
                kind=DataKind.SPECTRAL,
                required_fields=["axis", "psd"],
            )
        ]

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.STATISTICS,
                fields=[
                    "group_scan_index",
                    "group_channel_index",
                    "candidate_offsets",
                    "location",
                    "intensity",
                    "sample_index",
                    "status_code",
                ],
            ),
        )

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        """Find candidates directly from the typed spectral Dataset."""
        del backend
        config = cast(PeaksAnalyzerConfig, self.config)
        finder = self._resolve_finder()
        product = inputs.get(config.psd_key)
        if product is None:
            raise ValueError(f"peaks input has no product {config.psd_key!r}")
        has_scan = "scan" in {axis.name for axis in product.axes}
        raw_scan_size = product.axis("scan").size if has_scan else 1
        if raw_scan_size is None:
            raise ValueError("peaks input has an open scan axis")
        scan_size = int(raw_scan_size)
        tables: list[Mapping[str, Any]] = []
        selected_modes: list[int] | None = None
        for scan_index in range(scan_size):
            check_execution_cancelled(context)
            point = product.point_view(scan=scan_index) if has_scan else product
            axis = np.asarray(point.handle("axis").materialize(), dtype=float)
            spectrum = np.asarray(point.handle("psd").materialize(), dtype=float)
            if spectrum.ndim == 1:
                spectrum = spectrum[:, None]
            if "modes" in {variable.name for variable in point.variables}:
                available = [
                    int(value)
                    for value in np.asarray(point.handle("modes").materialize())
                ]
            else:
                available = list(range(spectrum.shape[1]))
            modes = list(config.modes) if config.modes is not None else available
            if selected_modes is None:
                selected_modes = modes
            elif selected_modes != modes:
                raise ValueError("peak inputs use different mode selections")
            tables.append(
                finder.find_candidate_table(
                    axis,
                    spectrum[:, [available.index(mode) for mode in modes]].T,
                    scan_index=scan_index,
                    scan_count=scan_size,
                    channel_names=[f"mode_{mode}" for mode in modes],
                )
            )
            report_execution_progress(
                context,
                completed=scan_index + 1,
                unit="spectrum",
                message=f"peaks: searched {scan_index + 1} spectra",
            )
        return {
            label: materialize_candidate_product(
                candidate_tables=tables,
                scan_size=scan_size if has_scan else None,
                label=label,
                modes=selected_modes or [],
                source_product=(
                    str(metadata["source_product"])
                    if metadata is not None and "source_product" in metadata
                    else config.psd_key
                ),
            )
        }

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: Any) -> WorkEstimate:
        del request
        return WorkEstimate(unit="spectrum", total=None)

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(PeaksAnalyzerConfig, self.config)
        finder = self._resolve_finder()
        if finder is None:
            raise ValueError("peaks analyser requires a peak-finder child")
        loaded = load_sde_results(data)
        if not loaded:
            raise ValueError("peaks analyser received no spectral input")
        candidate_tables: list[Mapping[str, Any]] = []
        modes: list[int] | None = None
        for item_index, item in enumerate(loaded):
            payload = item.analysis[config.psd_key]
            axis = np.asarray(payload["axis"], dtype=float)
            spectrum = np.asarray(payload["psd"], dtype=float)
            if spectrum.ndim == 1:
                spectrum = spectrum[:, None]
            available = [
                int(mode) for mode in payload.get("modes", range(spectrum.shape[1]))
            ]
            selected = list(config.modes) if config.modes is not None else available
            if modes is None:
                modes = selected
            elif modes != selected:
                raise ValueError("peak inputs use different mode selections")
            table = finder.find_candidate_table(
                axis,
                spectrum[:, [available.index(mode) for mode in selected]].T,
                scan_index=item_index,
                scan_count=len(loaded),
                channel_names=[f"mode_{mode}" for mode in selected],
            )
            candidate_tables.append(table)
        return AnalysisResult(
            data_dict={"candidate_tables": candidate_tables, "modes": modes or []},
            meta={"scan_param": config.scan_param, "count": len(candidate_tables)},
        )

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int | None,
        label: str,
    ) -> dict[str, Dataset] | None:
        if isinstance(payload, list):
            candidate_tables: list[Mapping[str, Any]] = []
            modes: list[int] | None = None
            for scan_index, item in enumerate(payload):
                if not isinstance(item, Mapping):
                    raise TypeError("peaks scan payload entries must be mappings")
                tables = item.get("candidate_tables")
                if not isinstance(tables, list):
                    raise TypeError("peaks scan payload must contain candidate_tables")
                for table in tables:
                    if not isinstance(table, Mapping):
                        raise TypeError("peaks candidate table must be a mapping")
                    rebased = dict(table)
                    rebased["group_scan_index"] = np.full_like(
                        np.asarray(table["group_scan_index"], dtype=np.int64),
                        scan_index,
                    )
                    rebased["scan_count"] = len(payload)
                    candidate_tables.append(rebased)
                item_modes = [int(mode) for mode in item.get("modes", [])]
                if modes is None:
                    modes = item_modes
                elif modes != item_modes:
                    raise ValueError("peak scan points use different mode selections")
            payload = {"candidate_tables": candidate_tables, "modes": modes or []}
        if not isinstance(payload, dict):
            raise TypeError("peaks payload must be a mapping")
        tables = payload.get("candidate_tables")
        if not isinstance(tables, list):
            raise TypeError("peaks payload must contain candidate_tables")
        return {
            label: materialize_candidate_product(
                candidate_tables=tables,
                scan_size=scan_size,
                label=label,
                modes=[int(mode) for mode in payload["modes"]],
            )
        }
