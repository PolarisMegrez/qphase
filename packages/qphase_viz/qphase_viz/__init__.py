"""qphase-viz - Visualization Package
=================================
Visualization engine and plotters for qphase, providing time-series, phase-plane,
and power spectrum analysis visualization capabilities.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 1.0.1 (Jan 2026)
"""

from qphase_sde.analyser import PsdAnalyzer, PsdAnalyzerConfig

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
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
]
