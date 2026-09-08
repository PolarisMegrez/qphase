"""qphase_sde: Coherence-Frequency Subpackage
---------------------------------------------------------
Coherence-frequency estimators: child plugins that consume the typed inputs
they declare (time series, spectrum, peak candidates and/or peak paths) and
produce coherence-frequency estimate rows of ``COHERENCE_FREQUENCY_PRODUCT``.
Built-in ``short_delay``, ``band_limited`` and ``finite_delay`` child plugins
implement the shared protocol and reuse the package's established numerical
estimators.

Public API
----------
``BaseCoherenceFrequencyConfig`` : Base configuration for coherence-frequency
    estimators.
``CoherenceFrequencyProtocol`` : Abstract coherence-frequency plugin class.
"""

from .band_limited import (
    BandLimitedCoherenceFrequency,
    BandLimitedCoherenceFrequencyConfig,
)
from .base import BaseCoherenceFrequencyConfig, CoherenceFrequencyProtocol
from .finite_delay import (
    FiniteDelayCoherenceFrequency,
    FiniteDelayCoherenceFrequencyConfig,
)
from .short_delay import (
    ShortDelayCoherenceFrequency,
    ShortDelayCoherenceFrequencyConfig,
)

__all__ = [
    "BaseCoherenceFrequencyConfig",
    "CoherenceFrequencyProtocol",
    "ShortDelayCoherenceFrequency",
    "ShortDelayCoherenceFrequencyConfig",
    "BandLimitedCoherenceFrequency",
    "BandLimitedCoherenceFrequencyConfig",
    "FiniteDelayCoherenceFrequency",
    "FiniteDelayCoherenceFrequencyConfig",
]
