"""Adaptive carrier estimation from a band-limited power spectrum.

The analyser plugin is registered from this package; its configuration models
live in :mod:`config`.
"""

from .analyser import BandLimitedCarrierAnalyser
from .config import BandLimitedCarrierConfig, RidgeConditionedCenterConfig

__all__ = [
    "BandLimitedCarrierAnalyser",
    "BandLimitedCarrierConfig",
    "RidgeConditionedCenterConfig",
]
