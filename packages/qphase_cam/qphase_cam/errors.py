"""Resource-specific CAM errors."""

from qphase.core.errors import QPhaseRuntimeError


class CAMError(QPhaseRuntimeError):
    """Base error for coherent-amplitude matrix operations."""


class SolutionCapacityError(CAMError):
    """Raised when a model produces more states than its declared capacity."""


class JacobianUnavailableError(CAMError):
    """Raised when an operation requires an unavailable Jacobian."""


class BifurcationCapabilityError(CAMError):
    """Raised when a model cannot provide exact bifurcation dynamics."""


class QPSGenCompatibilityError(BifurcationCapabilityError):
    """Raised when qpsgen exposes an incompatible numerical contract."""
