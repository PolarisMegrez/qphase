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
``PsdAnalyser`` : Power spectral density analyzer.
``PsdAnalyserConfig`` : Configuration for PSD analyzer.
``DistAnalyser`` : Distribution analyzer.
``DistAnalyserConfig`` : Configuration for Distribution analyzer.
"""

from typing import TYPE_CHECKING, Any

from .allan_scaling import (
    AllanScalingAnalyser,
    AllanScalingConfig,
    NormalFormExpectation,
)
from .allan_variance import (
    AllanMeasurement,
    AllanPhaseFrequencyMeasurement,
    AllanSampledScalarMeasurement,
    AllanVarianceAnalyser,
    AllanVarianceConfig,
)
from .band_limited_carrier import (
    BandLimitedCarrierAnalyser,
    BandLimitedCarrierConfig,
    RidgeConditionedCenterConfig,
)
from .base import Analyser
from .coherence_carrier import CoherenceCarrierAnalyser, CoherenceCarrierConfig
from .coherence_frequency import (
    CoherenceFrequencyAnalyser,
    CoherenceFrequencyAnalyserConfig,
)
from .coherence_matrix import CoherenceMatrixAnalyser, CoherenceMatrixConfig
from .dist import DistAnalyser, DistAnalyserConfig
from .finite_delay_carrier import (
    FiniteDelayCarrierAnalyser,
    FiniteDelayCarrierConfig,
)
from .frequency_orientation import (
    ORIENTATION_ALIASES,
    OrientationInput,
    resolve_frequency_orientation,
)
from .moment_statistics import MomentStatisticsAnalyser, MomentStatisticsConfig
from .observable_transform import (
    ObservableTransformAnalyser,
    ObservableTransformConfig,
)
from .peak_paths import PeakPathsAnalyser, PeakPathsAnalyserConfig
from .peaks import PeaksAnalyser, PeaksAnalyserConfig
from .polar_dist import PolarDistAnalyser, PolarDistAnalyserConfig
from .psd import (
    PsdAnalyser,
    PsdAnalyserConfig,
    PsdComplexFieldMeasurement,
    PsdMeasurement,
    PsdSampledScalarMeasurement,
)
from .quadratic_moments import (
    QuadraticMomentAnalyser,
    QuadraticMomentConfig,
    QuadraticObservableSpec,
)
from .spectral_ridge import SpectralRidgeAnalyser, SpectralRidgeConfig
from .trajectory_diagnostic import (
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
    "Analyser",
    "AnalyserDeclarationContract",
    "AnalyserExecutionContract",
    "AnalyserExecutionCapabilities",
    "AnalyserWorkspaceEstimate",
    "AnalyserWorkspaceRequest",
    "AllanMeasurement",
    "AllanPhaseFrequencyMeasurement",
    "AllanSampledScalarMeasurement",
    "AllanVarianceAnalyser",
    "AllanVarianceConfig",
    "AllanScalingAnalyser",
    "AllanScalingConfig",
    "BandLimitedCarrierAnalyser",
    "BandLimitedCarrierConfig",
    "RidgeConditionedCenterConfig",
    "CoherenceMatrixAnalyser",
    "CoherenceMatrixConfig",
    "CoherenceCarrierAnalyser",
    "CoherenceCarrierConfig",
    "CoherenceFrequencyAnalyser",
    "CoherenceFrequencyAnalyserConfig",
    "MomentStatisticsAnalyser",
    "MomentStatisticsConfig",
    "NormalFormExpectation",
    "ObservableTransformAnalyser",
    "ObservableTransformConfig",
    "PsdAnalyser",
    "PsdAnalyserConfig",
    "PsdComplexFieldMeasurement",
    "PsdMeasurement",
    "PsdSampledScalarMeasurement",
    "QuadraticMomentAnalyser",
    "QuadraticMomentConfig",
    "QuadraticObservableSpec",
    "SpectralRidgeAnalyser",
    "SpectralRidgeConfig",
    "DistAnalyser",
    "DistAnalyserConfig",
    "ORIENTATION_ALIASES",
    "OrientationInput",
    "FiniteDelayCarrierAnalyser",
    "FiniteDelayCarrierConfig",
    "PolarDistAnalyser",
    "PolarDistAnalyserConfig",
    "PeaksAnalyser",
    "PeaksAnalyserConfig",
    "PeakPathsAnalyser",
    "PeakPathsAnalyserConfig",
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
