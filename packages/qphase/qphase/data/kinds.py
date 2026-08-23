"""qphase: Data Kind Definitions
---------------------------------------------------------
Defines the three public data kinds (``time_series``, ``spectral``,
``statistics``) shared by the scheduler, engines, artifact store, session cache
and cross-package data selection, plus the frozen minimal set of spectral
quantities. Data kinds belong to the core; physical quantities, provenance and
reducers belong to resource packages.

Public API
----------
DataKind
    The three public data kinds.
SpectralQuantity
    Frozen minimal spectral quantity enumeration.
"""

from enum import Enum

__all__ = [
    "DataKind",
    "SpectralQuantity",
]


class DataKind(str, Enum):
    """The three public data kinds of typed data products."""

    TIME_SERIES = "time_series"
    SPECTRAL = "spectral"
    STATISTICS = "statistics"


class SpectralQuantity(str, Enum):
    """Frozen minimal set of spectral quantities.

    PSD variables declare real/nonnegative; cross-spectrum variables declare
    complex with a Hermitian layout. The distinction between quantities is
    carried by the variable schema, never by incompatible result classes.
    """

    FOURIER_AMPLITUDE = "fourier_amplitude"
    POWER_SPECTRAL_DENSITY = "power_spectral_density"
    CROSS_SPECTRAL_DENSITY = "cross_spectral_density"
    COHERENCE = "coherence"
