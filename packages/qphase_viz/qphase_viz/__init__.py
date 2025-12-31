"""qphase-viz - Visualization Package
=================================
Visualization engine and plotters for qphase, providing time-series, phase-plane,
and power spectrum analysis visualization capabilities.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 1.0.0 (Jan 2026)
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
