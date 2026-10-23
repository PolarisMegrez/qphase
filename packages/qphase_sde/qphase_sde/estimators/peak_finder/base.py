"""qphase_sde: Peak Finder Plugin Classes
---------------------------------------------------------
:class:`TypedPeakFinder` is the only public finder contract: a runtime
structural protocol whose ``find_candidate_table`` returns the grouped
candidate-table payload of ``qphase_sde.contracts.peaks`` — exactly the
mapping :func:`~qphase_sde.contracts.peaks.validate_candidate_table` accepts.

Everything else in this module is implementation detail of the built-in
finders: the numerical ``_find_peaks`` step and its ``_PeakInfo`` container
are private, and the concrete finders reach the typed contract through the
shared ``_NumericalPeakFinder`` base.

Public API
----------
``BasePeakFinderConfig`` : Base configuration for peak finders.
``TypedPeakFinder`` : The candidate-table plugin contract.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.core.protocols import PluginConfigBase
from qphase.data import Dataset, ProductSchema, StatisticsDataset

from ...contracts.peaks import (
    PEAK_PRODUCT,
    CandidateTablePayload,
    validate_candidate_table,
)


class _PeakInfo(BaseModel):
    """Private peak container returned by ``_find_peaks``."""

    indices: list[int]
    frequencies: list[float]
    values: list[float]
    properties: dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, **kwargs) -> dict[str, Any]:
        """Serialize peak info, converting numpy/complex values to plain Python."""
        data = super().model_dump(**kwargs)
        data["properties"] = _serialize_peak_value(self.properties)
        return data


def _serialize_peak_value(value: Any) -> Any:
    """Recursively convert numpy arrays and complex numbers to JSON-safe types."""
    if isinstance(value, np.ndarray):
        return _serialize_peak_value(value.tolist())
    if isinstance(value, np.complexfloating | complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    if isinstance(value, dict):
        return {k: _serialize_peak_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_serialize_peak_value(v) for v in value]
    if isinstance(value, np.floating | np.integer):
        return value.item()
    return value


class BasePeakFinderConfig(PluginConfigBase):
    """Base configuration for peak finders.

    Strict like the other plugin config bases in this repo (e.g. the CAM
    solver/postprocessor bases): unknown fields are rejected instead of
    being silently absorbed by the permissive ``PluginConfigBase`` default.
    """

    model_config = ConfigDict(extra="forbid")

    method: str


def materialize_candidate_product(
    peak_points: Sequence[Mapping[Any, Any]] | None = None,
    *,
    scan_size: int | None,
    label: str,
    modes: Sequence[int],
    source_product: str | None = None,
    candidate_tables: Sequence[Mapping[str, Any]] | None = None,
) -> Dataset:
    """Materialize per-scan finder results as a grouped peak product."""
    if candidate_tables is None and peak_points is None:
        raise ValueError("candidate product needs peak points or candidate tables")
    scan_count = 1 if scan_size is None else scan_size
    group_scan_index: list[int] = []
    group_channel_index: list[int] = []
    candidate_offsets = [0]
    locations: list[float] = []
    intensities: list[float] = []
    candidate_columns: dict[str, list[Any]] = {}
    if candidate_tables is not None:
        if len(candidate_tables) != scan_count:
            raise ValueError("candidate tables must match the scan cardinality")
        for table in candidate_tables:
            columns = table["columns"]
            offsets = np.asarray(table["candidate_offsets"], dtype=np.int64)
            group_scan = np.asarray(table["group_scan_index"], dtype=np.int64)
            group_channel = np.asarray(table["group_channel_index"], dtype=np.int64)
            for group, (scan, channel) in enumerate(
                zip(group_scan, group_channel, strict=True)
            ):
                start, stop = int(offsets[group]), int(offsets[group + 1])
                group_scan_index.append(int(scan))
                group_channel_index.append(int(channel))
                candidate_offsets.append(candidate_offsets[-1] + stop - start)
                for name, values in columns.items():
                    candidate_columns.setdefault(name, []).extend(
                        np.asarray(values)[start:stop].tolist()
                    )
        locations = [float(value) for value in candidate_columns.pop("location", [])]
        intensities = [float(value) for value in candidate_columns.pop("intensity", [])]
    else:
        points = list(peak_points or [])
        if len(points) != scan_count:
            raise ValueError("peak_points must match the scan cardinality")
        for scan_index, point in enumerate(points):
            for channel_index, mode in enumerate(modes):
                info = point.get(mode, point.get(str(mode), {}))
                frequencies = info.get("frequencies", [])
                values = info.get("values", [])
                if len(frequencies) != len(values):
                    raise ValueError(
                        "peak frequencies and values must have equal lengths"
                    )
                group_scan_index.append(scan_index)
                group_channel_index.append(channel_index)
                locations.extend(float(value) for value in frequencies)
                intensities.extend(float(value) for value in values)
                candidate_offsets.append(len(locations))

    candidate_count = len(locations)
    columns = {
        "location": np.asarray(locations, dtype=float),
        "intensity": np.asarray(intensities, dtype=float),
        "status_code": np.asarray(
            candidate_columns.pop(
                "status_code", np.full(candidate_count, 3, dtype=np.int64)
            ),
            dtype=np.int64,
        ),
        "sample_index": np.asarray(
            candidate_columns.pop(
                "sample_index", np.full(candidate_count, -1, dtype=np.int64)
            ),
            dtype=np.int64,
        ),
    }
    for name, values in candidate_columns.items():
        columns[name] = np.asarray(values)
    channel_names = [f"mode_{mode}" for mode in modes]
    validate_candidate_table(
        np.asarray(candidate_offsets, dtype=np.int64),
        np.asarray(group_scan_index, dtype=np.int64),
        np.asarray(group_channel_index, dtype=np.int64),
        columns,
        scan_count=scan_count,
        channel_names=channel_names,
    )
    schema_data = PEAK_PRODUCT.model_dump(mode="python")
    sizes = {
        "scan": scan_count,
        "group": len(group_scan_index),
        "group_offset": len(candidate_offsets),
        "candidate": candidate_count,
    }
    schema_data["axes"] = [
        {**axis, "size": sizes[axis["name"]]}
        for axis in schema_data["axes"]
        if axis["name"] != "scan" or scan_size is not None
    ]
    schema_data["sampling_bases"] = []
    schema_data["uncertainties"] = [
        item for item in schema_data["uncertainties"] if item["scope"] != "sampling"
    ]
    schema_data["variables"] = [
        variable
        for variable in schema_data["variables"]
        if variable["name"]
        not in {"sampling_location_std", "sampling_location_std_valid"}
    ]
    schema_data["attributes"] = {
        **schema_data["attributes"],
        "channel_names": channel_names,
        "mode_labels": channel_names,
        "graph_ready": True,
        "source_finder": "typed",
        "source_product": source_product or label,
    }
    schema = ProductSchema.model_validate(schema_data)
    arrays: dict[str, Any] = {**columns}
    arrays["group_scan_index"] = np.asarray(group_scan_index, dtype=np.int64)
    arrays["group_channel_index"] = np.asarray(group_channel_index, dtype=np.int64)
    arrays["candidate_offsets"] = np.asarray(candidate_offsets, dtype=np.int64)
    for variable in schema.variables:
        if variable.name in arrays:
            continue
        if variable.name.endswith("_valid"):
            arrays[variable.name] = np.zeros(candidate_count, dtype=np.int8)
        else:
            arrays[variable.name] = np.full(candidate_count, np.nan, dtype=float)
    return StatisticsDataset.from_arrays(
        schema,
        arrays,
        owner="engine.sde",
        provenance={
            "source_finder": "typed",
            "product": label,
            "source_product": source_product or label,
        },
    )


class _NumericalPeakFinder(ABC):
    """Private base: numerical peak finding plus the typed-table adapter."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BasePeakFinderConfig]] = BasePeakFinderConfig

    def __init__(self, config: BasePeakFinderConfig | None = None, **kwargs: Any):
        if config is None:
            config = self.config_schema(**kwargs)
        self.config = config

    @abstractmethod
    def _find_peaks(self, freqs: np.ndarray, psd: np.ndarray) -> _PeakInfo:
        """Find peaks in power spectral density data.

        Parameters
        ----------
        freqs : np.ndarray
            Frequency axis.
        psd : np.ndarray
            Power spectral density values.

        Returns
        -------
        _PeakInfo
            Detected peaks and their properties.

        """
        pass

    def find_candidate_table(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        *,
        scan_index: int,
        scan_count: int,
        channel_names: Sequence[str],
        mode_labels: Sequence[str] | None = None,
        channel_measurement_kinds: Sequence[str] | None = None,
        channel_mode_weights: Sequence[Sequence[float]] | None = None,
    ) -> CandidateTablePayload:
        """Return the legacy finder's detections as a grouped typed table."""
        axis = np.asarray(frequencies, dtype=float).reshape(-1)
        values = np.asarray(power, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if values.ndim != 2 or values.shape[1] != axis.size:
            raise ValueError("power must have shape (channel, frequency)")
        if values.shape[0] != len(channel_names):
            raise ValueError("channel_names must match the power channel count")

        offsets = [0]
        group_scan_index: list[int] = []
        group_channel_index: list[int] = []
        locations: list[float] = []
        intensities: list[float] = []
        sample_indices: list[int] = []
        statuses: list[int] = []
        for channel, spectrum in enumerate(values):
            result = self._find_peaks(axis, spectrum)
            sample_indices.extend(int(value) for value in result.indices)
            locations.extend(float(value) for value in result.frequencies)
            intensities.extend(float(value) for value in result.values)
            statuses.extend([3] * len(result.frequencies))
            offsets.append(len(locations))
            group_scan_index.append(scan_index)
            group_channel_index.append(channel)
        payload: CandidateTablePayload = {
            "candidate_offsets": np.asarray(offsets, dtype=np.int64),
            "group_scan_index": np.asarray(group_scan_index, dtype=np.int64),
            "group_channel_index": np.asarray(group_channel_index, dtype=np.int64),
            "columns": {
                "location": np.asarray(locations, dtype=float),
                "intensity": np.asarray(intensities, dtype=float),
                "sample_index": np.asarray(sample_indices, dtype=np.int64),
                "status_code": np.asarray(statuses, dtype=np.int64),
            },
            "scan_count": int(scan_count),
            "channel_names": list(channel_names),
        }
        if mode_labels is not None:
            payload["mode_labels"] = list(mode_labels)
        if channel_measurement_kinds is not None:
            payload["channel_measurement_kinds"] = list(channel_measurement_kinds)
        if channel_mode_weights is not None:
            payload["channel_mode_weights"] = [
                list(row) for row in channel_mode_weights
            ]
        validate_candidate_table(**payload)
        return payload


@runtime_checkable
class TypedPeakFinder(Protocol):
    """Typed contract for unified peak-candidate finders.

    Implementations return the grouped candidate-table payload of
    ``qphase_sde.contracts.peaks`` — a :class:`CandidateTablePayload` mapping
    that must unpack into ``validate_candidate_table`` without raising.
    Built-in finders satisfy this protocol through the shared
    ``_NumericalPeakFinder`` base, which owns the single implementation.
    """

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BasePeakFinderConfig]]

    def find_candidate_table(
        self,
        frequencies: np.ndarray,
        power: np.ndarray,
        *,
        scan_index: int,
        scan_count: int,
        channel_names: Sequence[str],
        mode_labels: Sequence[str] | None = None,
        channel_measurement_kinds: Sequence[str] | None = None,
        channel_mode_weights: Sequence[Sequence[float]] | None = None,
    ) -> CandidateTablePayload:
        """Find peaks in one scan point's spectra and return the payload.

        Parameters
        ----------
        frequencies : np.ndarray
            Shared frequency axis.
        power : np.ndarray
            Power spectral density values, one row per channel.
        scan_index : int
            Scan point these spectra belong to (``group_scan_index``).
        scan_count : int
            Total number of scan points of the run.
        channel_names : Sequence[str]
            Stable channel labels of ``power``, echoed into the payload.
        mode_labels : Sequence[str] | None
            Modes the readout channels act on, when a measurement basis is
            declared.
        channel_mode_weights : Sequence[Sequence[float]] | None
            ``[re, im]`` readout weights flattened row-major over
            (channel x mode), echoed into the payload.
        channel_measurement_kinds : Sequence[str] | None
            Measurement representation kind for each channel when a basis is
            supplied.

        Returns
        -------
        CandidateTablePayload
            The grouped candidate-table payload;
            ``qphase_sde.contracts.peaks.validate_candidate_table(**payload)``
            must not raise.

        """
        ...
