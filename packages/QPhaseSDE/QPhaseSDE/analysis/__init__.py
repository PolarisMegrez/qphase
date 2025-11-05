"""
QPhaseSDE: Analysis Subpackage
------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and future time/frequency-domain diagnostics.

Registry integration
--------------------
On import, register available analysis routines into the central registry
under the ``analysis`` namespace for scheduler-driven discovery and dispatch.
"""

from .psd import compute_psd_for_modes, compute_psd_single  # noqa: F401
from ..core.registry import namespaced
register, register_lazy = namespaced("analysis")

# Register built-in analysis entries using the helper to align with other packages.
try:
	register("psd", lambda: compute_psd_for_modes, return_callable=True, tags=["spectral"])  # type: ignore[arg-type]
	register("psd_single", lambda: compute_psd_single, return_callable=True, tags=["spectral"])  # type: ignore[arg-type]
except Exception:
	# Avoid hard failures at import time in unusual environments
	pass
