"""Backend-neutral numerical cores shared by coherence-frequency plugins and
analysers.

Nothing here imports a plugin class: the modules take arrays and return typed
estimates, so the analyser and estimator layers can both call them.
"""

from .band_limited import estimate_band_limited_carrier
from .finite_delay import finite_delay_carrier_from_spectrum
from .short_delay import CoherenceCarrierEstimate, estimate_coherence_carrier

__all__ = [
    "CoherenceCarrierEstimate",
    "estimate_band_limited_carrier",
    "estimate_coherence_carrier",
    "finite_delay_carrier_from_spectrum",
]
