"""qphase_sde: Analysis Subpackage
------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and future time/frequency-domain diagnostics.

Registry integration
--------------------
On import, register available analysis routines into the central registry
under the ``analysis`` namespace for scheduler-driven discovery and dispatch.
"""

from ..core.registry import namespaced
from .psd import PsdAnalyzer, PsdAnalyzerConfig  # noqa: F401

register, register_lazy = namespaced("analysis")

# Register built-in analysis entries using the helper to align with other packages.
try:
    register(
        "psd",
        PsdAnalyzer,
        tags=["spectral"],
    )
except Exception:
    # Avoid hard failures at import time in unusual environments
    pass
