"""Typed coherence-frequency parent analyser."""

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
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    StatisticsDataset,
    load_bundle,
)

from ..coherence_frequency.base import CoherenceFrequencyProtocol
from ..contracts.analyser import AnalyserExecutionCapabilities, WorkEstimate
from ..contracts.coherence import (
    COHERENCE_FREQUENCY_PRODUCT,
    CoherenceFrequencyEstimate,
)
from .base import Analyzer
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)
from .result import AnalysisResult

__all__ = ["CoherenceFrequencyAnalyzer", "CoherenceFrequencyAnalyzerConfig"]


class CoherenceFrequencyAnalyzerConfig(PluginConfigBase):
    """Configuration for the typed coherence-frequency parent."""

    model_config = ConfigDict(extra="forbid")

    scan_param: str = Field(..., description="Name of the input scan axis")
    input_key: str = Field("spectrum", description="Typed input product label")
    estimator: dict[str, Any] = Field(
        default_factory=lambda: {"band_limited": {}},
        description="One selected coherence-frequency child configuration",
    )
    channel: int = Field(0, ge=0)

    @property
    def estimator_name(self) -> str:
        """Return the selected estimator child name."""
        if len(self.estimator) != 1:
            raise ValueError("coherence_frequency.estimator must select one child")
        return str(next(iter(self.estimator)))


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
        raise ValueError("coherence_frequency requires a typed SDE data bundle input")
    return products


def _make_child(
    config: CoherenceFrequencyAnalyzerConfig,
    injected: CoherenceFrequencyProtocol | None,
) -> CoherenceFrequencyProtocol:
    del config
    if injected is None:
        raise RuntimeError(
            "coherence_frequency requires a resolved estimator child; "
            "construct it through the plugin graph or pass subplugins"
        )
    return injected


def _point_product(product: Dataset, index: int) -> Dataset:
    if "scan" in {axis.name for axis in product.axes}:
        return product.point_view(scan=index)
    return product


def _spectral_channel(array: np.ndarray, channel: int) -> np.ndarray:
    values = np.asarray(array)
    if values.ndim == 1:
        return values
    if values.ndim == 2:
        if not 0 <= channel < values.shape[1]:
            raise ValueError(f"channel {channel} is not present in the input product")
        return values[:, channel]
    raise ValueError("coherence-frequency input must be one- or two-dimensional")


def _estimate_input(
    product: Dataset,
    config: CoherenceFrequencyAnalyzerConfig,
    estimator_name: str | None = None,
) -> Mapping[str, Any]:
    child = estimator_name or config.estimator_name
    if child == "short_delay":
        correlations = np.asarray(
            product.handle("per_trajectory_correlation").materialize()
        )
        if correlations.ndim == 2:
            selected = correlations
        elif correlations.ndim == 3:
            if not 0 <= config.channel < correlations.shape[1]:
                raise ValueError(
                    f"channel {config.channel} is not present in the input product"
                )
            selected = correlations[:, config.channel, :]
        else:
            raise ValueError(
                "per_trajectory_correlation must have shape "
                "(trajectory, measurement, lag)"
            )
        return {
            "per_trajectory_correlation": selected,
            "dt": float(np.asarray(product.handle("dt").materialize()).reshape(-1)[0]),
        }
    spectrum = _spectral_channel(product.handle("psd").materialize(), config.channel)
    return {
        "axis": np.asarray(product.handle("axis").materialize(), dtype=float),
        "spectrum": np.asarray(spectrum, dtype=float),
    }


def _materialize_estimates(
    estimates: list[CoherenceFrequencyEstimate],
    *,
    scan_size: int,
    label: str,
    orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
    channel_name: str = "channel_0",
    channel_index: int = 0,
) -> Dataset:
    schema_data = COHERENCE_FREQUENCY_PRODUCT.model_dump(mode="python")
    schema_data["axes"] = [
        {**axis, "size": scan_size if axis["name"] == "scan" else 1}
        for axis in schema_data["axes"]
    ]
    schema_data["attributes"] = {
        **schema_data["attributes"],
        "graph_ready": True,
        "estimator": estimates[0].estimator if estimates else "",
        **orientation_metadata(orientation),
        "channel_names": [channel_name],
        "channel_indices": [int(channel_index)],
    }
    has_sampling = all(
        item.independent_count is not None and item.frequency_sem is not None
        for item in estimates
    )
    if not has_sampling:
        schema_data["sampling_bases"] = []
        schema_data["uncertainties"] = [
            item for item in schema_data["uncertainties"] if item["scope"] != "sampling"
        ]
        schema_data["variables"] = [
            item
            for item in schema_data["variables"]
            if item["name"] not in {"frequency_sem", "independent_count"}
        ]
    else:
        schema_data["sampling_bases"] = [
            {"name": "trajectory", "count_variable": "independent_count"}
        ]
    from qphase.data import ProductSchema

    schema = ProductSchema.model_validate(schema_data)

    def values(name: str, default: float = np.nan) -> np.ndarray:
        return np.asarray(
            [[getattr(item, name, default)] for item in estimates], dtype=float
        )

    arrays = {
        "frequency": values("frequency"),
        "conditional_std": values("conditional_std"),
        "confidence_lower": values("confidence_lower"),
        "confidence_upper": values("confidence_upper"),
        "lag": values("lag"),
        "bandwidth": values("bandwidth"),
        "status_code": np.asarray(
            [[int(item.status)] for item in estimates], dtype=np.int64
        ),
    }
    if has_sampling:
        arrays["frequency_sem"] = values("frequency_sem")
        arrays["independent_count"] = np.asarray(
            [[item.independent_count] for item in estimates], dtype=np.int64
        )
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.sde",
        provenance={"analyser": "coherence_frequency", "product": label},
    )


class CoherenceFrequencyAnalyzer(Analyzer):
    """Run one coherence-frequency child against a typed input product."""

    name: ClassVar[str] = "coherence_frequency"
    description: ClassVar[str] = "Typed coherence-frequency parent analyser"
    config_schema: ClassVar[type[CoherenceFrequencyAnalyzerConfig]] = (
        CoherenceFrequencyAnalyzerConfig
    )
    preserves_input_scan: ClassVar[bool] = True
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "estimator": SubpluginSlot(
                namespace="coherence_frequency",
                default="band_limited",
                protocol="qphase_sde.coherence_frequency.base:CoherenceFrequencyProtocol",
                description="Coherence-frequency estimator child",
            )
        }
    )

    def __init__(
        self,
        config: CoherenceFrequencyAnalyzerConfig | None = None,
        *,
        subplugins: Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        self._estimator = None if subplugins is None else subplugins.get("estimator")

    def _selected_estimator_name(self) -> str:
        """Return the resolved child name, including graph-injected children."""
        if self._estimator is not None:
            return str(self._estimator.name)
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        return config.estimator_name

    def input_requirements(self) -> list[ProductRequirement]:
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        estimator_name = self._selected_estimator_name()
        fields = (
            ["per_trajectory_correlation", "dt"]
            if estimator_name == "short_delay"
            else ["axis", "psd"]
        )
        return [
            ProductRequirement(
                name=config.input_key,
                kind=DataKind.STATISTICS
                if estimator_name == "short_delay"
                else DataKind.SPECTRAL,
                required_fields=fields,
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
        """Execute directly on typed input products."""
        result = cast(AnalysisResult, self.analyze(inputs, backend))
        scan_size = result.data_dict["estimates"].__len__()
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        source = inputs[config.input_key]
        orientation = resolve_frequency_orientation(source.attributes)
        products = self.build_products(
            result.data_dict,
            scan_size=scan_size,
            label=label,
            orientation=orientation,
            channel_name=f"channel_{config.channel}",
            channel_index=config.channel,
        )
        if products is None:
            raise ValueError("coherence_frequency produced no typed product")
        return products

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        estimator_name = self._selected_estimator_name()
        fields = [
            variable.name
            for variable in COHERENCE_FREQUENCY_PRODUCT.variables
            if estimator_name == "short_delay"
            or variable.name not in {"frequency_sem", "independent_count"}
        ]
        return (
            ProductDeclaration(
                name=label,
                kind=COHERENCE_FREQUENCY_PRODUCT.kind,
                fields=fields,
            ),
        )

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        return AnalyserExecutionCapabilities(preferred_location="host")

    def work_estimate(self, request: Any) -> WorkEstimate:
        del request
        return WorkEstimate(unit="spectrum", total=None)

    def analyze(self, data: Any, backend: BackendBase) -> ResultProtocol:
        del backend
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        products = _products_from_input(data)
        product = products.get(config.input_key)
        if product is None:
            raise ValueError(
                f"coherence_frequency input has no product {config.input_key!r}"
            )
        scan_axis = (
            product.axis("scan")
            if "scan" in {axis.name for axis in product.axes}
            else None
        )
        scan_count = 1
        if scan_axis is not None:
            size = scan_axis.size
            if size is None:
                raise ValueError("coherence-frequency input scan axis has no size")
            scan_count = int(size)
        child = _make_child(config, self._estimator)
        estimator_name = self._selected_estimator_name()
        estimates = [
            child.estimate(
                _estimate_input(_point_product(product, index), config, estimator_name)
            )
            for index in range(scan_count)
        ]
        return AnalysisResult(
            data_dict={"estimates": estimates},
            meta={"scan_param": config.scan_param, "scan_count": scan_count},
        )

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int | None,
        label: str,
        orientation: FrequencyOrientation = DEFAULT_FREQUENCY_ORIENTATION,
        channel_name: str = "channel_0",
        channel_index: int = 0,
    ) -> dict[str, Dataset] | None:
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("estimates"), list
        ):
            raise TypeError("coherence_frequency payload must contain estimates")
        estimates = payload["estimates"]
        if not all(isinstance(item, CoherenceFrequencyEstimate) for item in estimates):
            raise TypeError("coherence_frequency estimates have an invalid type")
        size = len(estimates) if scan_size is None else scan_size
        if size != len(estimates):
            raise ValueError("coherence_frequency estimates do not match scan size")
        return {
            label: _materialize_estimates(
                estimates,
                scan_size=size,
                label=label,
                orientation=orientation,
                channel_name=channel_name,
                channel_index=channel_index,
            )
        }
