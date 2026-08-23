"""qphase_sde: Analysis Subpackage
---------------------------------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and phase space distribution.

Registry integration
--------------------
On import, register available analysis routines into the central registry
under the ``analysis`` namespace for scheduler-driven discovery and dispatch.

Public API
----------
``PsdAnalyzer`` : Power spectral density analyzer.
``PsdAnalyzerConfig`` : Configuration for PSD analyzer.
``DistAnalyzer`` : Distribution analyzer.
``DistAnalyzerConfig`` : Configuration for Distribution analyzer.
"""

from .allan_scaling import (
    AllanScalingAnalyzer,
    AllanScalingConfig,
    NormalFormExpectation,
)
from .allan_variance import AllanVarianceAnalyzer, AllanVarianceConfig
from .band_limited_carrier import (
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierConfig,
    RidgeConditionedCenterConfig,
)
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerProtocol,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
)
from .coherence_carrier import CoherenceCarrierAnalyzer, CoherenceCarrierConfig
from .coherence_matrix import CoherenceMatrixAnalyzer, CoherenceMatrixConfig
from .dist import DistAnalyzer, DistAnalyzerConfig
from .finite_delay_carrier import (
    FiniteDelayCarrierAnalyzer,
    FiniteDelayCarrierConfig,
)
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    LEGACY_FREQUENCY_ORIENTATION,
    ORIENTATION_ALIASES,
    FrequencyOrientation,
    OrientationInput,
    resolve_frequency_orientation,
)
from .moment_statistics import MomentStatisticsAnalyzer, MomentStatisticsConfig
from .polar_dist import PolarDistAnalyzer, PolarDistAnalyzerConfig
from .psd import PsdAnalyzer, PsdAnalyzerConfig
from .quadratic_moments import (
    QuadraticMomentAnalyzer,
    QuadraticMomentConfig,
    QuadraticObservableSpec,
)
from .spectral_ridge import SpectralRidgeAnalyzer, SpectralRidgeConfig
from .trajectory_diagnostics import (
    TrajectoryDiagnostics,
    TrajectoryDiagnosticsConfig,
)

__all__ = [
    "Analyzer",
    "AnalyzerExecutionCapabilities",
    "AnalyzerProtocol",
    "AnalyzerWorkspaceEstimate",
    "AnalyzerWorkspaceRequest",
    "AllanVarianceAnalyzer",
    "AllanVarianceConfig",
    "AllanScalingAnalyzer",
    "AllanScalingConfig",
    "BandLimitedCarrierAnalyzer",
    "BandLimitedCarrierConfig",
    "RidgeConditionedCenterConfig",
    "CoherenceMatrixAnalyzer",
    "CoherenceMatrixConfig",
    "CoherenceCarrierAnalyzer",
    "CoherenceCarrierConfig",
    "MomentStatisticsAnalyzer",
    "MomentStatisticsConfig",
    "NormalFormExpectation",
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
    "QuadraticMomentAnalyzer",
    "QuadraticMomentConfig",
    "QuadraticObservableSpec",
    "SpectralRidgeAnalyzer",
    "SpectralRidgeConfig",
    "DistAnalyzer",
    "DistAnalyzerConfig",
    "DEFAULT_FREQUENCY_ORIENTATION",
    "LEGACY_FREQUENCY_ORIENTATION",
    "ORIENTATION_ALIASES",
    "OrientationInput",
    "FrequencyOrientation",
    "FiniteDelayCarrierAnalyzer",
    "FiniteDelayCarrierConfig",
    "PolarDistAnalyzer",
    "PolarDistAnalyzerConfig",
    "TrajectoryDiagnostics",
    "TrajectoryDiagnosticsConfig",
    "resolve_frequency_orientation",
]
