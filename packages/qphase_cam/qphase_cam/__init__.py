"""Coherent-amplitude matrix analysis for the QPhase workspace."""

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

__all__ = [
    "BifurcationCapabilityError",
    "CAMBifurcationModel",
    "CAMBogoliubovModel",
    "CAMModel",
    "CAMSymbolicSpec",
    "CAMVectorModel",
    "QPSGenCompatibilityError",
    "JacobianUnavailableError",
    "SolutionCapacityError",
]

__version__ = "2.0.0"
