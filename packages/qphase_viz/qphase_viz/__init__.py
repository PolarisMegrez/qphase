"""Visualization Package
=====================

Visualization engine and plotters for qphase, providing time-series, phase-plane,
and power spectrum analysis visualization capabilities.

Public API
----------
VizEngine
    Main visualization engine.
VizEngineConfig
    Configuration for the visualization engine.
"""

from .config import (
    PhasePlaneConfig,
    PowerSpectrumConfig,
    TimeSeriesConfig,
    VizEngineConfig,
)
from .engine import VizEngine

__all__ = [
    "VizEngine",
    "VizEngineConfig",
    "TimeSeriesConfig",
    "PhasePlaneConfig",
    "PowerSpectrumConfig",
]
