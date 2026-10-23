"""qphase_sde: Peak Tracker Subpackage
---------------------------------------------------------
Scan-axis peak-path trackers: child plugins that link the candidate rows of a
validated grouped candidate table (``PEAK_PRODUCT``) into scan-axis paths
(``PEAK_PATH_PRODUCT``). The initial ``peak_tracker.ridge`` implementation
uses data-only continuity and leaves physical branch interpretation to later
postprocessing.

Public API
----------
``BasePeakTrackerConfig`` : Base configuration for peak trackers.
``PeakTrackerProtocol`` : Abstract peak-tracker plugin class.
"""

from .base import (
    BasePeakTrackerConfig,
    PeakTrackerProtocol,
    materialize_path_product,
)
from .ridge import RidgePeakTracker, RidgePeakTrackerConfig

__all__ = [
    "BasePeakTrackerConfig",
    "PeakTrackerProtocol",
    "RidgePeakTracker",
    "RidgePeakTrackerConfig",
    "materialize_path_product",
]
