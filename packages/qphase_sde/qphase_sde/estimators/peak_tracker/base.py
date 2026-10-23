"""qphase_sde: Peak Tracker Plugin Class (2.0)
---------------------------------------------------------
Plugin-class protocol for scan-axis peak-path trackers. A peak tracker
consumes one validated grouped candidate table — the materialized payload of
``PEAK_PRODUCT`` from ``qphase_sde.contracts.peaks``, checked with
``validate_candidate_table`` — and links its candidate rows into scan-axis
paths, the rows of ``PEAK_PATH_PRODUCT``. Paths reference candidate rows
only; a path is never a global physical cross-scan branch identity.

The first concrete implementation is ``peak_tracker.ridge``; this module
declares the plugin-class surface and keeps concrete tracking logic in its
own child-plugin module.

Public API
----------
``BasePeakTrackerConfig`` : Base configuration for peak trackers.
``PeakTrackerProtocol`` : Abstract peak-tracker plugin class.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase
from qphase.data import Dataset, ProductSchema, StatisticsDataset

from ...contracts.peaks import PEAK_PATH_PRODUCT, PeakPathResult, validate_path_table

__all__ = [
    "BasePeakTrackerConfig",
    "PeakTrackerProtocol",
    "materialize_path_product",
]


class BasePeakTrackerConfig(PluginConfigBase):
    """Base configuration for peak trackers.

    Strict like the other plugin config bases in this repo (e.g. the CAM
    solver/postprocessor bases): unknown fields are rejected instead of
    being silently absorbed by the permissive ``PluginConfigBase`` default.
    """

    model_config = ConfigDict(extra="forbid")


def materialize_path_product(
    paths: Sequence[PeakPathResult],
    *,
    candidate_count: int,
    label: str = "peak_paths",
    source_product: str = "",
) -> Dataset:
    """Materialize tracker paths using the formal ``PEAK_PATH_PRODUCT`` schema."""
    path_offsets = [0]
    candidate_rows: list[int] = []
    scan_positions: list[int] = []
    status_codes: list[int] = []
    qualities: list[float] = []
    for path in paths:
        candidate_rows.extend(path.candidate_rows)
        scan_positions.extend(path.scan_positions)
        path_offsets.append(len(candidate_rows))
        status_codes.append(int(path.status))
        qualities.append(np.nan if path.quality is None else float(path.quality))
    validate_path_table(
        np.asarray(path_offsets, dtype=np.int64),
        np.asarray(candidate_rows, dtype=np.int64),
        np.asarray(scan_positions, dtype=np.int64),
        candidate_count,
        path_status_code=np.asarray(status_codes, dtype=np.int64),
    )
    schema_data = PEAK_PATH_PRODUCT.model_dump(mode="python")
    sizes = {
        "path": len(paths),
        "path_offset": len(path_offsets),
        "path_member": len(candidate_rows),
    }
    schema_data["axes"] = [
        {**axis, "size": sizes[axis["name"]]} for axis in schema_data["axes"]
    ]
    schema_data["attributes"] = {
        **schema_data["attributes"],
        "graph_ready": True,
        "source_product": source_product,
    }
    schema = ProductSchema.model_validate(schema_data)
    return StatisticsDataset.from_arrays(
        schema,
        {
            "path_offsets": np.asarray(path_offsets, dtype=np.int64),
            "candidate_row": np.asarray(candidate_rows, dtype=np.int64),
            "scan_position": np.asarray(scan_positions, dtype=np.int64),
            "path_status_code": np.asarray(status_codes, dtype=np.int64),
            "path_quality": np.asarray(qualities, dtype=float),
        },
        owner="engine.sde",
        provenance={"product": label},
    )


class PeakTrackerProtocol(ABC):
    """Abstract base class for scan-axis peak-path trackers."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[BasePeakTrackerConfig]] = BasePeakTrackerConfig

    def __init__(self, config: BasePeakTrackerConfig | None = None, **kwargs: Any):
        if config is None:
            config = self.config_schema(**kwargs)
        self.config = config

    @abstractmethod
    def track(
        self,
        candidate_offsets: np.ndarray,
        group_scan_index: np.ndarray,
        group_channel_index: np.ndarray,
        columns: Mapping[str, np.ndarray],
        *,
        scan_count: int,
        channel_names: Sequence[str],
    ) -> list[PeakPathResult]:
        """Link the rows of a validated candidate table into scan-axis paths.

        Parameters
        ----------
        candidate_offsets : np.ndarray
            Prefix sums over groups: group ``g`` owns candidate rows
            ``[offsets[g], offsets[g + 1])``.
        group_scan_index : np.ndarray
            Scan point of each group, indexing into the scan axis.
        group_channel_index : np.ndarray
            Channel of each group, indexing into ``channel_names``.
        columns : Mapping[str, np.ndarray]
            Candidate columns (``location``, ``intensity``, ``status_code``
            plus optional capability columns with their ``<name>_valid``
            masks), one entry per candidate row.
        scan_count : int
            Number of scan points the group indices reference.
        channel_names : Sequence[str]
            Stable index-to-label mapping of the candidate channels.

        Returns
        -------
        list[PeakPathResult]
            Scan-axis paths over candidate rows. The caller materializes the
            path table and validates it with ``validate_path_table``.

        """
        pass
