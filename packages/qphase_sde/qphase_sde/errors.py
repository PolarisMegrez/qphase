"""qphase_sde: SDE Package Errors (2.0)
---------------------------------------------------------
Error hierarchy of the SDE resource package. All package errors derive from
``SDEError`` — itself a core ``QPhaseError`` — so callers can catch SDE
failures specifically while existing core error handling keeps working.

Public API
----------
SDEError
    Base exception for all qphase_sde errors.
SDEConfigError
    SDE configuration and validation errors.
SDEExecutionError
    SDE engine execution errors.
"""

from qphase.core.errors import QPhaseError

__all__ = [
    "SDEConfigError",
    "SDEError",
    "SDEExecutionError",
]


class SDEError(QPhaseError):
    """Base exception for all qphase_sde resource package errors."""


class SDEConfigError(SDEError):
    """Configuration and validation errors of the SDE resource package."""


class SDEExecutionError(SDEError):
    """SDE engine execution errors (integration, analysis, scan assembly)."""
