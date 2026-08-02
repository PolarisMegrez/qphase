"""Built-in spectral-estimator subplugins for the PSD analyser."""

from .base import PsdEstimate, SpectralEstimator, SpectralEstimatorCapabilities
from .builtin import (
    MultitaperEstimator,
    MultitaperEstimatorConfig,
    PeriodogramEstimator,
    PeriodogramEstimatorConfig,
    WelchEstimator,
    WelchEstimatorConfig,
    create_builtin_estimator,
)

__all__ = [
    "MultitaperEstimator",
    "MultitaperEstimatorConfig",
    "PeriodogramEstimator",
    "PeriodogramEstimatorConfig",
    "PsdEstimate",
    "SpectralEstimator",
    "SpectralEstimatorCapabilities",
    "WelchEstimator",
    "WelchEstimatorConfig",
    "create_builtin_estimator",
]
