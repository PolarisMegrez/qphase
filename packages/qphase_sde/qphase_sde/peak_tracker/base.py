"""qphase_sde: Peak Tracker Plugin Class (2.0)
---------------------------------------------------------
Plugin-class protocol for scan-axis peak-path trackers. A peak tracker
consumes one validated grouped candidate table — the materialized payload of
``PEAK_PRODUCT`` from ``qphase_sde.contracts.peaks``, checked with
``validate_candidate_table`` — and links its candidate rows into scan-axis
paths, the rows of ``PEAK_PATH_PRODUCT``. Paths reference candidate rows
only; a path is never a global physical cross-scan branch identity.

Concrete implementations arrive in Wave 5B: ``peak_tracker.ridge`` will host
the path-tracking logic of the current ``analyser.spectral_ridge`` analyser.
This module declares the plugin-class surface only and ships no concrete
tracker.

Public API
----------
``BasePeakTrackerConfig`` : Base configuration for peak trackers.
``PeakTrackerProtocol`` : Abstract peak-tracker plugin class.
"""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar

import numpy as np
from qphase.core.protocols import PluginConfigBase

from ..contracts.peaks import PeakPathResult

__all__ = [
    "BasePeakTrackerConfig",
    "PeakTrackerProtocol",
]


class BasePeakTrackerConfig(PluginConfigBase):
    """Base configuration for peak trackers."""


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
