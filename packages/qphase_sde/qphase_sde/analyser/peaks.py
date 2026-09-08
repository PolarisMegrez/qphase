"""Standalone grouped peak analysis from a typed spectral product."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
from pydantic import ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginConfigBase, ResultProtocol
from qphase.data import DataKind, Dataset, ProductDeclaration, ProductRequirement

from ..contracts.analyser import AnalyserExecutionCapabilities, WorkEstimate
from ..peak_finder import (
    RationalPeakFinderConfig,
    ScipyPeakFinderConfig,
    create_peak_finder,
    materialize_candidate_product,
)
from .base import Analyzer
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
    finder: bool | str | ScipyPeakFinderConfig | RationalPeakFinderConfig = Field(
        "scipy", description="Typed peak-finder child configuration"
    )


class PeaksAnalyzer(Analyzer):
    """Run a typed peak finder independently of PSD estimation."""

    name: ClassVar[str] = "peaks"
    description: ClassVar[str] = "Grouped peak candidates from a spectral product"
    config_schema: ClassVar[type[PeaksAnalyzerConfig]] = PeaksAnalyzerConfig
    preserves_input_scan: ClassVar[bool] = True

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
                    "status_code",
                ],
            ),
        )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: Any) -> WorkEstimate:
        del request
        return WorkEstimate(unit="spectrum", total=None)

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(PeaksAnalyzerConfig, self.config)
        finder = create_peak_finder(config.finder)
        if finder is None:
            raise ValueError("peaks analyser requires a peak-finder child")
        loaded = load_sde_results(data)
        if not loaded:
            raise ValueError("peaks analyser received no spectral input")
        peak_points: list[dict[int, dict[str, list[float]]]] = []
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
            offsets = table["candidate_offsets"]
            locations = table["columns"]["location"]
            intensities = table["columns"]["intensity"]
            point: dict[int, dict[str, list[float]]] = {}
            for channel, mode in enumerate(selected):
                start, stop = int(offsets[channel]), int(offsets[channel + 1])
                point[mode] = {
                    "frequencies": locations[start:stop].tolist(),
                    "values": intensities[start:stop].tolist(),
                }
            peak_points.append(point)
        return AnalysisResult(
            data_dict={"peak_points": peak_points, "modes": modes or []},
            meta={"scan_param": config.scan_param, "count": len(peak_points)},
        )

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int | None,
        label: str,
    ) -> dict[str, Dataset] | None:
        if not isinstance(payload, dict):
            raise TypeError("peaks payload must be a mapping")
        points = payload["peak_points"]
        return {
            label: materialize_candidate_product(
                points,
                scan_size=scan_size,
                label=label,
                modes=[int(mode) for mode in payload["modes"]],
            )
        }
