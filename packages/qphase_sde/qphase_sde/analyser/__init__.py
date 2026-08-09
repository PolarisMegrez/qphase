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
from .base import (
    Analyzer,
    AnalyzerExecutionCapabilities,
    AnalyzerProtocol,
    AnalyzerWorkspaceEstimate,
    AnalyzerWorkspaceRequest,
)
from .coherence_matrix import CoherenceMatrixAnalyzer, CoherenceMatrixConfig
from .dist import DistAnalyzer, DistAnalyzerConfig
from .polar_dist import PolarDistAnalyzer, PolarDistAnalyzerConfig
from .psd import PsdAnalyzer, PsdAnalyzerConfig
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
    "CoherenceMatrixAnalyzer",
    "CoherenceMatrixConfig",
    "NormalFormExpectation",
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
    "DistAnalyzer",
    "DistAnalyzerConfig",
    "PolarDistAnalyzer",
    "PolarDistAnalyzerConfig",
    "TrajectoryDiagnostics",
    "TrajectoryDiagnosticsConfig",
]
