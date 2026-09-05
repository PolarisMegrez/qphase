"""qphase_sde: Analysis Subpackage
---------------------------------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and phase space distribution.

The engine-facing contract types (``AnalyserExecutionCapabilities``,
``AnalyserWorkspaceRequest``, ``AnalyserWorkspaceEstimate`` and
``AnalyserContract``) are re-exported lazily from
``qphase_sde.contracts.analyser`` — the single source of truth — so that
importing this package never pulls the contracts package before the core
schema packages are fully initialized.

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

from typing import TYPE_CHECKING, Any

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
from .base import Analyzer
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

if TYPE_CHECKING:
    from ..contracts.analyser import (
        AnalyserContract,
        AnalyserExecutionCapabilities,
        AnalyserWorkspaceEstimate,
        AnalyserWorkspaceRequest,
    )

__all__ = [
    "Analyzer",
    "AnalyserContract",
    "AnalyserExecutionCapabilities",
    "AnalyserWorkspaceEstimate",
    "AnalyserWorkspaceRequest",
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

#: Contract re-exports resolved lazily (PEP 562) from the contracts package.
_LAZY_CONTRACT_EXPORTS = {
    "AnalyserContract": "..contracts.analyser",
    "AnalyserExecutionCapabilities": "..contracts.analyser",
    "AnalyserWorkspaceEstimate": "..contracts.analyser",
    "AnalyserWorkspaceRequest": "..contracts.analyser",
}


def __getattr__(name: str) -> Any:
    """Resolve contract re-exports on first access."""
    module_suffix = _LAZY_CONTRACT_EXPORTS.get(name)
    if module_suffix is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    module = importlib.import_module(module_suffix, __name__)
    return getattr(module, name)
