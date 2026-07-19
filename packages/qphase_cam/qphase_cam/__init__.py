"""Coherent-amplitude matrix analysis for the QPhase workspace."""

from .errors import JacobianUnavailableError, SolutionCapacityError
from .model import CAMModel, CAMSymbolicSpec, CAMVectorModel
from .result import CAMResult

__all__ = [
    "CAMModel",
    "CAMResult",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "0.1.0"
