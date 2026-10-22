"""qphase_sde: Analysis Subpackage
---------------------------------------------------------
Numerical analysis utilities for simulated data, including power spectral
density (PSD) computation and phase space distribution.

The analyser declaration types (``AnalyserExecutionCapabilities``,
``AnalyserWorkspaceRequest``, ``AnalyserWorkspaceEstimate`` and
``AnalyserDeclarationContract``) are re-exported lazily from
``qphase_sde.contracts.analyser`` — the single source of truth — so that
importing this package never pulls the contracts package before the core
schema packages are fully initialized.

Registry integration
--------------------
Analyser plugins are discovered by the core registry through the
``analyser.<name>`` entry points declared in the ``qphase`` entry-point group
of this package's ``pyproject.toml``. Importing this package performs no
registration itself.

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
from .allan_variance import (
    AllanMeasurement,
    AllanPhaseFrequencyMeasurement,
    AllanSampledScalarMeasurement,
    AllanVarianceAnalyzer,
    AllanVarianceConfig,
)
from .band_limited_carrier import (
    BandLimitedCarrierAnalyzer,
    BandLimitedCarrierConfig,
    RidgeConditionedCenterConfig,
)
from .base import Analyzer
from .coherence_carrier import CoherenceCarrierAnalyzer, CoherenceCarrierConfig
from .coherence_frequency import (
    CoherenceFrequencyAnalyzer,
    CoherenceFrequencyAnalyzerConfig,
)
from .coherence_matrix import CoherenceMatrixAnalyzer, CoherenceMatrixConfig
from .dist import DistAnalyzer, DistAnalyzerConfig
from .finite_delay_carrier import (
    FiniteDelayCarrierAnalyzer,
    FiniteDelayCarrierConfig,
)
from .frequency_orientation import (
    DEFAULT_FREQUENCY_ORIENTATION,
    ORIENTATION_ALIASES,
    FrequencyOrientation,
    OrientationInput,
    resolve_frequency_orientation,
)
from .moment_statistics import MomentStatisticsAnalyzer, MomentStatisticsConfig
from .observable_transform import (
    ObservableTransformAnalyzer,
    ObservableTransformConfig,
)
from .peak_paths import PeakPathsAnalyzer, PeakPathsAnalyzerConfig
from .peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from .polar_dist import PolarDistAnalyzer, PolarDistAnalyzerConfig
from .psd import (
    PsdAnalyzer,
    PsdAnalyzerConfig,
    PsdComplexFieldMeasurement,
    PsdMeasurement,
    PsdSampledScalarMeasurement,
)
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
        AnalyserDeclarationContract,
        AnalyserExecutionCapabilities,
        AnalyserWorkspaceEstimate,
        AnalyserWorkspaceRequest,
    )

__all__ = [
    "Analyzer",
    "AnalyserDeclarationContract",
    "AnalyserExecutionContract",
    "AnalyserExecutionCapabilities",
    "AnalyserWorkspaceEstimate",
    "AnalyserWorkspaceRequest",
    "AllanMeasurement",
    "AllanPhaseFrequencyMeasurement",
    "AllanSampledScalarMeasurement",
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
    "CoherenceFrequencyAnalyzer",
    "CoherenceFrequencyAnalyzerConfig",
    "MomentStatisticsAnalyzer",
    "MomentStatisticsConfig",
    "NormalFormExpectation",
    "ObservableTransformAnalyzer",
    "ObservableTransformConfig",
    "PsdAnalyzer",
    "PsdAnalyzerConfig",
    "PsdComplexFieldMeasurement",
    "PsdMeasurement",
    "PsdSampledScalarMeasurement",
    "QuadraticMomentAnalyzer",
    "QuadraticMomentConfig",
    "QuadraticObservableSpec",
    "SpectralRidgeAnalyzer",
    "SpectralRidgeConfig",
    "DistAnalyzer",
    "DistAnalyzerConfig",
    "DEFAULT_FREQUENCY_ORIENTATION",
    "ORIENTATION_ALIASES",
    "OrientationInput",
    "FrequencyOrientation",
    "FiniteDelayCarrierAnalyzer",
    "FiniteDelayCarrierConfig",
    "PolarDistAnalyzer",
    "PolarDistAnalyzerConfig",
    "PeaksAnalyzer",
    "PeaksAnalyzerConfig",
    "PeakPathsAnalyzer",
    "PeakPathsAnalyzerConfig",
    "TrajectoryDiagnostics",
    "TrajectoryDiagnosticsConfig",
    "resolve_frequency_orientation",
]

#: Contract re-exports resolved lazily (PEP 562) from the contracts package.
_LAZY_CONTRACT_EXPORTS = {
    "AnalyserDeclarationContract": "..contracts.analyser",
    "AnalyserExecutionContract": "..contracts.analyser",
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
