
from typing import Any

from .base import BasePeakFinderConfig, PeakFinder, PeakInfo
from .scipy import ScipyPeakFinder, ScipyPeakFinderConfig
from .rational import RationalPeakFinder, RationalPeakFinderConfig

__all__ = [
    "PeakFinder",
    "PeakInfo",
    "BasePeakFinderConfig",
    "ScipyPeakFinder",
    "ScipyPeakFinderConfig",
    "RationalPeakFinder",
    "RationalPeakFinderConfig",
    "create_peak_finder",
]

def create_peak_finder(config: Any) -> PeakFinder | None:
    """Factory to create a peak finder from configuration."""
    if isinstance(config, (str, type(None))):
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
