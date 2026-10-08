"""Typed coherence-frequency parent analyser."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np
from pydantic import ConfigDict, Field
from qphase.backend.base import BackendBase
from qphase.core.protocols import (
    PluginConfigBase,
    PluginManifest,
    SubpluginSlot,
)
from qphase.data import (
    DataKind,
    Dataset,
    ProductDeclaration,
    ProductRequirement,
    StatisticsDataset,
)

from ..coherence_frequency.base import CoherenceFrequencyProtocol
from ..contracts.analyser import AnalyserExecutionCapabilities, WorkEstimate
from ..contracts.coherence import (
    COHERENCE_FREQUENCY_PRODUCT,
    CoherenceFrequencyEstimate,
)
from .base import Analyzer, check_execution_cancelled, report_execution_progress
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    FrequencyOrientation,
    orientation_metadata,
    resolve_frequency_orientation,
)

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
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        """Execute directly on typed input products."""
        del backend, metadata
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        source = inputs[config.input_key]
        estimates = self._compute_estimates(source, context=context)
        orientation = resolve_frequency_orientation(source.attributes)
        channel_name = f"channel_{config.channel}"
        configured_names = source.attributes.get("channel_names")
        if isinstance(configured_names, list) and config.channel < len(
            configured_names
        ):
            channel_name = str(configured_names[config.channel])
        elif "modes" in {variable.name for variable in source.variables}:
            modes = np.asarray(source.handle("modes").materialize()).reshape(-1)
            if config.channel < modes.size:
                channel_name = f"mode_{int(modes[config.channel])}"
        return {
            label: _materialize_estimates(
                estimates,
                scan_size=len(estimates),
                label=label,
                orientation=orientation,
                channel_name=channel_name,
                channel_index=config.channel,
            )
        }

    def _compute_estimates(
        self,
        product: Dataset,
        *,
        context: Any | None = None,
    ) -> list[CoherenceFrequencyEstimate]:
        config = cast(CoherenceFrequencyAnalyzerConfig, self.config)
        scan_axis = (
            product.axis("scan")
            if "scan" in {axis.name for axis in product.axes}
            else None
        )
        scan_count = 1
        if scan_axis is not None:
            if scan_axis.size is None:
                raise ValueError("coherence-frequency input scan axis has no size")
            scan_count = int(scan_axis.size)
        child = _make_child(config, self._estimator)
        estimator_name = self._selected_estimator_name()
        estimates: list[CoherenceFrequencyEstimate] = []
        for index in range(scan_count):
            check_execution_cancelled(context)
            estimates.append(
                child.estimate(
                    _estimate_input(
                        _point_product(product, index), config, estimator_name
                    )
                )
            )
            report_execution_progress(
                context,
                completed=index + 1,
                unit="spectrum",
                message=f"coherence_frequency: estimated {index + 1} spectra",
            )
        return estimates

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
