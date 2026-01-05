"""qphase_sde: Analysis Subpackage
---------------------------------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and phase space distribution.

Registry integration
--------------------
On import, register available analysis routines into the central registry
under the ``analysis`` namespace for scheduler-driven discovery and dispatch.

Public API
----------
``PsdAnalyzer`` : Power spectral density analyzer.
``PsdAnalyzerConfig`` : Configuration for PSD analyzer.
``DistAnalyzer`` : Distribution analyzer.
``DistAnalyzerConfig`` : Configuration for Distribution analyzer.
"""

from .dist import DistAnalyzer, DistAnalyzerConfig
from .psd import PsdAnalyzer, PsdAnalyzerConfig

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
    "DistAnalyzer",
    "DistAnalyzerConfig",
]
