"""Coherent-amplitude matrix analysis for the QPhase workspace."""

from .bifurcation_result import CAMBifurcationResult, CAMBifurcationScanResult
from .errors import (
    BifurcationCapabilityError,
    JacobianUnavailableError,
    QPSGenCompatibilityError,
    SolutionCapacityError,
)
from .model import (
    CAMBifurcationModel,
    CAMBogoliubovModel,
    CAMModel,
    CAMSymbolicSpec,
    CAMVectorModel,
)
from .result import CAMResult

__all__ = [
    "BifurcationCapabilityError",
    "CAMBifurcationModel",
    "CAMBifurcationResult",
    "CAMBifurcationScanResult",
    "CAMBogoliubovModel",
    "CAMModel",
    "CAMResult",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "QPSGenCompatibilityError",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "0.1.0"
