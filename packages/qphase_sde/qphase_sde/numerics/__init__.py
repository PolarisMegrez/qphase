"""qphase_sde: Backend-Neutral Numerical Cores
---------------------------------------------------------
Backend-agnostic numerical definitions shared across the SDE resource
package: canonical phase-space coordinate conventions, the backend-aware
primitives used by all integrators, and the array-in/estimate-out cores that
both the analyser and the estimator layers call.

Nothing here imports a plugin class: the modules take arrays and return typed
estimates or arrays. Execution helpers live in ``qphase_sde.runtime``.

Public API
----------
coordinates
    Canonical phase-space real-coordinate conventions.
observable
    Hermitian quadratic observable spec, validation and backend evaluation.
ops
    Backend-agnostic SDE operations (noise expansion and contraction).
spectral
    Frozen sampled-scalar spectral math (detrend, one-sided folding, axes
    and canonical units).
band_limited
    Band-limited carrier estimate.
finite_delay
    Finite-delay carrier estimate from a spectrum.
short_delay
    Short-delay coherence-carrier estimate.
"""

from .band_limited import estimate_band_limited_carrier
from .finite_delay import finite_delay_carrier_from_spectrum
from .short_delay import CoherenceCarrierEstimate, estimate_coherence_carrier

__all__ = [
    "CoherenceCarrierEstimate",
    "band_limited",
    "coordinates",
    "estimate_band_limited_carrier",
    "estimate_coherence_carrier",
    "finite_delay",
    "finite_delay_carrier_from_spectrum",
    "observable",
    "ops",
    "short_delay",
    "spectral",
]
