"""Coherent-amplitude matrix analysis for the QPhase workspace."""

from .bifurcation_result import CAMBifurcationResult
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
    "CAMModel",
    "CAMResult",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "FPGenCompatibilityError",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "0.1.0"
