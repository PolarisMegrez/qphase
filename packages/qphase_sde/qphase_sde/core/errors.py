"""qphase_sde: Error Taxonomy and Logging
-------------------------------------

Independent error system for qphase_sde package.
This module defines a complete, standalone error hierarchy without any
dependencies on qphase or other external packages.

Architecture
------------
The qphase_sde error system is completely independent and self-contained.
Each package (ctl, sde, viz) defines its own error hierarchy:

- qphase_sde: Uses QPSError as base (inherits from Exception)
- qphase: Uses QPSError as base (inherits from Exception)
- qphase_viz: Uses QPSError as base (inherits from Exception)

This design ensures:
1. No circular dependencies between packages
2. Each package can operate independently
3. Clear error boundaries and responsibilities

Error Hierarchy
---------------
- QPSError: Base exception for all qphase_sde errors
- QPSBackendError: Backend-related errors (200-299)
- QPSIntegratorError: Integrator-related errors (300-399)
- QPSModelError: Model-related errors (600-699)
- QPSStateError: State-related errors (700-799)
- QPSVisualizerError: Visualization-related errors (800-899)
- QPSRequirementError: Requirements and dependency errors (000)

Warning Hierarchy
-----------------
- QPSWarning: Base warning for all qphase_sde warnings

Logging
-------
The shared logger is named "qphase_sde" and can be configured for
console and file output with optional JSON formatting.
Python warnings are captured into logging with adjustable levels.
"""

import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = [
    "QPSError",
    "QPSRequirementError",
    "QPSBackendError",
    "QPSIntegratorError",
    "QPSConfigError",
    "QPSModelError",
    "QPSStateError",
    "QPSWarning",
    "get_logger",
    "configure_logging",
    "deprecated",
]


# =============================================================================
# Exception Hierarchy (Independent - No External Dependencies)
# =============================================================================


class QPSError(Exception):
    """Base exception for all qphase_sde errors.

    This is the root of the qphase_sde error hierarchy. All errors raised
    by the qphase_sde package inherit from this class.

    The error system is:
    - Independent: No dependencies on qphase or other packages
    - Focused: SDE-specific error categorization
    - Backend-agnostic: Works with any backend implementation
    - Actionable: Clear error messages for debugging

    Examples
    --------
    >>> try:
    ...     # some SDE operation
    ...     pass
    ... except QPSError as e:
    ...     print(f"SDE error occurred: {e}")

    """

    pass


class QPSRequirementError(QPSError):
    """Requirements and dependency errors (Code 000).

    Raised when required dependencies are missing or incompatible.
    Examples: missing backend library, insufficient Python version, etc.
    """

    pass


class QPSBackendError(QPSError):
    """Backend-related errors (Code 200-299).

    Raised when backend creation, initialization, or operation fails.
    Examples: GPU not available, incompatible dtype, initialization failure.
    """

    pass


class QPSIntegratorError(QPSError):
    """Integrator-related errors (Code 300-399).

    Raised when integrator configuration or execution fails.
    Examples: invalid time step, integration divergence, unsupported method.
    """

    pass


class QPSConfigError(QPSError):
    """Configuration-related errors (Code 500-599).

    Raised when configuration validation or resolution fails.
    Examples: missing fields, invalid values, incompatible options.
    """

    pass


class QPSModelError(QPSError):
    """Model-related errors (Code 600-699).

    Raised when model definition, validation, or simulation fails.
    Examples: invalid parameters, simulation instability, dimension mismatch.
    """

    pass


class QPSStateError(QPSError):
    """State-related errors (Code 700-799).

    Raised when state operations, transformations, or validation fails.
    Examples: incompatible shapes, backend mismatch, serialization errors.
    """

    pass


# =============================================================================
# Warning Hierarchy
# =============================================================================


class QPSWarning(Warning):
    """Base warning for all qphase_sde warnings.

    All warnings issued by the qphase_sde package inherit from this class.
    """

    pass


# =============================================================================
# Logger Configuration
# =============================================================================

_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the shared qphase_sde logger instance.

    Returns
    -------
    logging.Logger
        The singleton logger named "qphase_sde" configured at INFO level by
        default with a console handler. Handlers are created lazily on first use.

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.name
    'qphase_sde'

    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("qphase_sde")
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            h.setFormatter(fmt)
            _logger.addHandler(h)
    return _logger


def configure_logging(
    verbose: bool = False,
    log_file: str | None = None,
    as_json: bool = False,
    suppress_warnings: bool = False,
) -> None:
    """Configure the shared logger outputs and warning capture.

    Parameters
    ----------
    verbose : bool, default False
        When True, set logger level to DEBUG; otherwise INFO.
    log_file : str or None, default None
        Optional file path to append logs. Invalid paths are ignored silently.
    as_json : bool, default False
        Emit logs in a compact JSON line format when True; otherwise plain text.
    suppress_warnings : bool, default False
        Route Python warnings into logging and raise their level to ERROR when
        True; otherwise capture warnings at WARNING level.

    Examples
    --------
    >>> configure_logging(verbose=True, as_json=False)  # doctest: +SKIP
    >>> logger = get_logger()
    >>> logger.level in (logging.INFO, logging.DEBUG)
    True

    """
    logger = get_logger()
    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # Console handler
    ch = logging.StreamHandler()
    if as_json:
        fmt = logging.Formatter(
            '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
        )
    else:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler
    if log_file:
        try:
            path = os.fspath(log_file)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception:
            # Ignore invalid file handler targets
            pass

    if suppress_warnings:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
    else:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.WARNING)


# =============================================================================
# Deprecation Utilities
# =============================================================================

T = TypeVar("T")


def deprecated(reason: str) -> Callable[[T], T]:
    """Mark a function or class as deprecated.

    On first call/instantiation, emits a ``QPSWarning`` (code [990]) via
    Python's warnings subsystem and logs the same message through the shared
    logger. Subsequent calls will not repeat the warning.

    Parameters
    ----------
    reason : str
        Human-readable explanation of the deprecation and suggested alternative.

    Returns
    -------
    Callable[[T], T]
        A decorator that wraps a function/class to emit the deprecation warning
        once, then delegates to the original object.

    Examples
    --------
    >>> @deprecated("Use new_api() instead")
    ... def old_api():
    ...     return 42
    >>> isinstance(old_api(), int)
    True

    """

    def _decorator(obj: T) -> T:
        logger = get_logger()
        warned_attr = "__qps_deprecated_warned__"

        if callable(obj):

            def _wrapped(*args, **kwargs):
                if not getattr(_wrapped, warned_attr, False):
                    name = getattr(obj, "__name__", str(obj))
                    msg = f"[990] DEPRECATED: {name}: " + str(reason)
                    warnings.warn(msg, QPSWarning, stacklevel=2)
                    logger.warning(msg)
                    setattr(_wrapped, warned_attr, True)
                return cast(Callable[..., Any], obj)(*args, **kwargs)

            try:
                _wrapped.__name__ = getattr(obj, "__name__", _wrapped.__name__)
            except Exception:
                pass
            return cast(T, _wrapped)
        return obj

    return _decorator
