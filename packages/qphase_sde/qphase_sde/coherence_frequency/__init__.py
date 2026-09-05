"""qphase_sde: Coherence-Frequency Subpackage
---------------------------------------------------------
Coherence-frequency estimators: child plugins that consume the typed inputs
they declare (time series, spectrum, peak candidates and/or peak paths) and
produce coherence-frequency estimate rows of ``COHERENCE_FREQUENCY_PRODUCT``.
Concrete implementations arrive in Wave 5B (``short_delay``/``band_limited``/
``finite_delay``); the subpackage currently declares the plugin-class
protocol only.

Public API
----------
``BaseCoherenceFrequencyConfig`` : Base configuration for coherence-frequency
    estimators.
``CoherenceFrequencyProtocol`` : Abstract coherence-frequency plugin class.
"""

from .base import BaseCoherenceFrequencyConfig, CoherenceFrequencyProtocol

__all__ = [
    "BaseCoherenceFrequencyConfig",
    "CoherenceFrequencyProtocol",
]
