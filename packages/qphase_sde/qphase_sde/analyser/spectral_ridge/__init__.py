"""Scale-space spectral-ridge estimation from saved power spectra.

The analyser plugin is registered from this package; the estimation core and
its path tracker live in :mod:`tracking`, the configuration model in
:mod:`config`.
"""

from .analyser import SpectralRidgeAnalyser
from .config import SpectralRidgeConfig
from .tracking import SpectralRidgeEstimate, estimate_spectral_ridges

__all__ = [
    "SpectralRidgeAnalyser",
    "SpectralRidgeConfig",
    "SpectralRidgeEstimate",
    "estimate_spectral_ridges",
]
