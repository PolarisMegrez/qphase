"""qphase-viz - Visualization Package
=================================
Visualization engine and plotters for qphase, providing time-series, phase-plane,
and power spectrum analysis visualization capabilities.

Author : Yu Xue-hao (GitHub: @PolarisMegrez)
Affiliation : School of Physical Sciences, UCAS
Contact : yuxuehao23@mails.ucas.ac.cn
License : MIT
Version : 2.0.0 (Oct 2026)
"""

from .config import (
    AnnotationSpec,
    PhasePlaneConfig,
    PowerSpectrumConfig,
    SpectralScanConfig,
    SpectralScanSpec,
    TimeSeriesConfig,
    VizEngineConfig,
)
from .engine import VizEngine
from .styles import (
    DEFAULT_STYLE,
    PALETTE,
    STYLE_NAMES,
    RenderOptions,
    StyleName,
    allan_scan_colormap,
    cam_flow_colormap,
    rc_params,
)

__all__ = [
    "DEFAULT_STYLE",
    "PALETTE",
    "STYLE_NAMES",
    "AnnotationSpec",
    "PhasePlaneConfig",
    "PowerSpectrumConfig",
    "RenderOptions",
    "SpectralScanConfig",
    "SpectralScanSpec",
    "StyleName",
    "TimeSeriesConfig",
    "VizEngine",
    "VizEngineConfig",
    "allan_scan_colormap",
    "cam_flow_colormap",
    "rc_params",
]
