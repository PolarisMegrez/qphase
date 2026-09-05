"""qphase_sde: Peak Tracker Subpackage
---------------------------------------------------------
Scan-axis peak-path trackers: child plugins that link the candidate rows of a
validated grouped candidate table (``PEAK_PRODUCT``) into scan-axis paths
(``PEAK_PATH_PRODUCT``). Concrete implementations arrive in Wave 5B
(``peak_tracker.ridge``); the subpackage currently declares the plugin-class
protocol only.

Public API
----------
``BasePeakTrackerConfig`` : Base configuration for peak trackers.
``PeakTrackerProtocol`` : Abstract peak-tracker plugin class.
"""

from .base import BasePeakTrackerConfig, PeakTrackerProtocol

__all__ = [
    "BasePeakTrackerConfig",
    "PeakTrackerProtocol",
]
