"""qphase_sde: Analysis Subpackage
------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and future time/frequency-domain diagnostics.

Registry integration
--------------------
On import, register available analysis routines into the central registry
under the ``analysis`` namespace for scheduler-driven discovery and dispatch.
"""

from .psd import PsdAnalyzer, PsdAnalyzerConfig  # noqa: F401

__all__ = [
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
]
