"""Coherent-amplitude matrix analysis for the QPhase workspace."""

from .errors import JacobianUnavailableError, SolutionCapacityError
from .model import CAMModel, CAMSymbolicSpec
from .result import CAMResult

__all__ = [
    "CAMModel",
    "CAMResult",
    "CAMSymbolicSpec",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "0.1.0"
