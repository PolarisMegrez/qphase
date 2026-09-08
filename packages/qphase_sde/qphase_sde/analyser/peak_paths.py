"""Typed peak-path parent analyser."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.core.aggregation import DirectoryInputResult
from qphase.core.protocols import (
    PluginConfigBase,
    PluginManifest,
    ResultProtocol,
    SubpluginSlot,
)
from qphase.core.registry import discovery, registry
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    load_bundle,
)

from ..contracts.analyser import AnalyserExecutionCapabilities, WorkEstimate
from ..contracts.peaks import PEAK_PATH_PRODUCT, PeakPathResult
from ..peak_tracker import materialize_path_product
from .base import Analyzer
from .result import AnalysisResult

__all__ = ["PeakPathsAnalyzer", "PeakPathsAnalyzerConfig"]


class PeakPathsAnalyzerConfig(PluginConfigBase):
    """Configuration for the typed peak-path parent analyser."""

    model_config = ConfigDict(extra="forbid")

    scan_param: str = Field(..., description="Name of the input scan axis")
    candidates_key: str = Field("peaks", description="Input peak product label")
    tracker: dict[str, Any] = Field(
        default_factory=lambda: {"ridge": {}},
        description="Configuration of the peak-tracker child plugin",
    )


def _products_from_input(data: Any) -> Mapping[str, Dataset]:
    if isinstance(data, DirectoryInputResult):
        restored = load_bundle(Path(data.path))
        products = getattr(restored, "products", None)
    else:
        products = getattr(data, "products", None)
        if products is None and isinstance(data, Mapping):
            products = data
    if not isinstance(products, Mapping) or not all(
        isinstance(value, Dataset) for value in products.values()
    ):
        raise ValueError("peak_paths requires a typed SDE data bundle input")
    return products


def _make_tracker(config: PeakPathsAnalyzerConfig, injected: Any | None) -> Any:
    if injected is not None:
        return injected
    if not registry.list("peak_tracker"):
        discovery.discover_plugins()
    names = set(registry.list("peak_tracker"))
    selected = set(config.tracker) & names
    if selected:
        if len(config.tracker) != 1:
            raise ValueError("peak_paths tracker must select exactly one child")
        name, values = next(iter(config.tracker.items()))
        if not isinstance(values, Mapping):
            raise ValueError("peak_paths tracker child configuration must be a mapping")
    else:
        name, values = "ridge", config.tracker
    return registry.create(f"peak_tracker:{name}", **values)


class PeakPathsAnalyzer(Analyzer):
    """Link typed peak candidates into scan-axis paths."""

    name: ClassVar[str] = "peak_paths"
    description: ClassVar[str] = "Typed scan-axis peak-path analyser"
    config_schema: ClassVar[type[PeakPathsAnalyzerConfig]] = PeakPathsAnalyzerConfig
    preserves_input_scan: ClassVar[bool] = False
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "tracker": SubpluginSlot(
                namespace="peak_tracker",
                default="ridge",
                protocol="qphase_sde.peak_tracker.base:PeakTrackerProtocol",
                description="Peak path tracker child",
            )
        }
    )

    def __init__(
        self,
        config: PeakPathsAnalyzerConfig | None = None,
        *,
        subplugins: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._tracker = None if subplugins is None else subplugins.get("tracker")

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(PeakPathsAnalyzerConfig, self.config)
        return [
            ProductRequirement(
                name=config.candidates_key,
                kind=DataKind.STATISTICS,
                required_fields=[
                    "candidate_offsets",
                    "group_scan_index",
                    "group_channel_index",
                    "location",
                    "intensity",
                    "status_code",
                ],
            )
        ]

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: BackendBase,
        label: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Dataset]:
        del metadata
        """Execute directly on a typed candidate product mapping."""
        result = cast(AnalysisResult, self.analyze(inputs, backend))
        products = self.build_products(result.data_dict, scan_size=None, label=label)
        if products is None:
            raise ValueError("peak_paths produced no typed path product")
        return products

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        return (
            ProductDeclaration(
                name=label,
                kind=PEAK_PATH_PRODUCT.kind,
                fields=[
                    "path_offsets",
                    "candidate_row",
                    "scan_position",
                    "path_status_code",
                    "path_quality",
                ],
            ),
        )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: Any) -> WorkEstimate:
        del request
        return WorkEstimate(unit="candidate", total=None)

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(PeakPathsAnalyzerConfig, self.config)
        products = _products_from_input(data)
        candidates = products.get(config.candidates_key)
        if candidates is None:
            raise ValueError(
                f"peak_paths input has no product {config.candidates_key!r}"
            )
        required = (
            "candidate_offsets",
            "group_scan_index",
            "group_channel_index",
            "location",
            "intensity",
            "status_code",
        )
        columns = {
            name: np.asarray(candidates.handle(name).materialize())
            for name in required
            if name
            not in {"candidate_offsets", "group_scan_index", "group_channel_index"}
        }
        offsets = np.asarray(candidates.handle("candidate_offsets").materialize())
        group_scan = np.asarray(candidates.handle("group_scan_index").materialize())
        group_channel = np.asarray(
            candidates.handle("group_channel_index").materialize()
        )
        scan_axis = (
            candidates.axis("scan")
            if "scan" in {axis.name for axis in candidates.axes}
            else None
        )
        scan_count = 1
        if scan_axis is not None:
            size = scan_axis.size
            if size is None:
                raise ValueError("peak-path input scan axis has no size")
            scan_count = int(size)
        channel_names = candidates.attributes.get("channel_names")
        if not isinstance(channel_names, list) or not channel_names:
            channel_count = int(group_channel.max()) + 1 if group_channel.size else 1
            channel_names = [f"channel_{index}" for index in range(channel_count)]
        tracker = _make_tracker(config, self._tracker)
        paths = tracker.track(
            offsets,
            group_scan,
            group_channel,
            columns,
            scan_count=scan_count,
            channel_names=[str(name) for name in channel_names],
        )
        return AnalysisResult(
            data_dict={
                "paths": paths,
                "candidate_count": int(columns["location"].size),
            },
            meta={"scan_param": config.scan_param, "scan_count": scan_count},
        )

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int | None,
        label: str,
    ) -> dict[str, Dataset] | None:
        del scan_size
        if not isinstance(payload, Mapping):
            raise TypeError("peak_paths payload must be a mapping")
        paths = payload.get("paths")
        if not isinstance(paths, list) or not all(
            isinstance(path, PeakPathResult) for path in paths
        ):
            raise TypeError("peak_paths payload must contain PeakPathResult rows")
        return {
            label: materialize_path_product(
                paths,
                candidate_count=int(payload["candidate_count"]),
                label=label,
                source_product=cast(
                    PeakPathsAnalyzerConfig, self.config
                ).candidates_key,
            )
        }
