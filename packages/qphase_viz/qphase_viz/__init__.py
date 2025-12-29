"""qphase_viz: Visualization Package
--------------------------------

Visualization engine and plotters for qphase.

Usage
-----
>>> from qphase_viz import VizEngine, VizEngineConfig
>>> config = VizEngineConfig(output_dir="plots", specs=[...])
>>> engine = VizEngine(config)
>>> engine.run(data)
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
