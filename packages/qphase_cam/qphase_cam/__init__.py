"""Coherent-amplitude matrix analysis for the QPhase workspace."""

from .bifurcation_result import CAMBifurcationResult, CAMBifurcationScanResult
from .errors import (
    BifurcationCapabilityError,
    FPGenCompatibilityError,
    JacobianUnavailableError,
    SolutionCapacityError,
)
from .model import CAMBifurcationModel, CAMModel, CAMSymbolicSpec, CAMVectorModel
from .result import CAMResult

__all__ = [
    "BifurcationCapabilityError",
    "CAMBifurcationModel",
    "CAMBifurcationResult",
    "CAMBifurcationScanResult",
    "CAMModel",
    "CAMResult",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "FPGenCompatibilityError",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "0.1.0"
