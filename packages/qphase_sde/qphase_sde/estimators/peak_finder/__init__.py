"""qphase_sde: Peak Finder Plugins
---------------------------------------------------------
``TypedPeakFinder`` is the only public finder contract; the built-in finders
below satisfy it. Their numerical ``find_peaks`` step and its ``PeakInfo``
container are private implementation details of ``base`` and are not
re-exported here.

Public API
----------
BasePeakFinderConfig
    Base configuration for peak finders.
TypedPeakFinder
    Candidate-table plugin contract.
ScipyPeakFinder / ScipyPeakFinderConfig
    ``scipy.signal.find_peaks`` based finder.
RationalPeakFinder / RationalPeakFinderConfig
    Rational-function (Padé) based finder.
materialize_candidate_product
    Materialize per-scan finder results as a grouped peak product.
create_peak_finder
    Build a finder from a plugin configuration value.
"""

from typing import Any

from .base import (
    BasePeakFinderConfig,
    TypedPeakFinder,
    materialize_candidate_product,
)
from .rational import RationalPeakFinder, RationalPeakFinderConfig
from .scipy import ScipyPeakFinder, ScipyPeakFinderConfig

__all__ = [
    "BasePeakFinderConfig",
    "RationalPeakFinder",
    "RationalPeakFinderConfig",
    "ScipyPeakFinder",
    "ScipyPeakFinderConfig",
    "TypedPeakFinder",
    "create_peak_finder",
    "materialize_candidate_product",
]


def create_peak_finder(config: Any) -> TypedPeakFinder | None:
    """Create a peak finder from configuration."""
    if isinstance(config, str | type(None)):
        if config is None or (isinstance(config, str) and config.lower() == "none"):
            return None
        if isinstance(config, str) and config.lower() in ["scipy", "standard"]:
            return ScipyPeakFinder(ScipyPeakFinderConfig())
        if isinstance(config, str) and config.lower() == "rational":
            return RationalPeakFinder(RationalPeakFinderConfig())

    if isinstance(config, dict):
        method = config.get("method", "scipy")
        if method == "scipy":
            return ScipyPeakFinder(ScipyPeakFinderConfig(**config))
        elif method == "rational":
            return RationalPeakFinder(RationalPeakFinderConfig(**config))

    # If passed a Config object directly
    if isinstance(config, ScipyPeakFinderConfig):
        return ScipyPeakFinder(config)
    if isinstance(config, RationalPeakFinderConfig):
        return RationalPeakFinder(config)

    return None
