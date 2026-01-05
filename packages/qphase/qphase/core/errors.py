"""qphase: Exceptions and Logging
---------------------------------------------------------
Establishes the unified exception hierarchy and logging infrastructure for the
control layer. It defines categorized error types (e.g., ``QPhaseConfigError``,
``QPhasePluginError``) to facilitate precise error handling and provides a
centralized logging configuration utility that supports multiple output formats
(console, file, JSON).

Public API
----------
QPhaseError
    Base exception class for all framework errors.
QPhaseConfigError, QPhaseIOError, QPhasePluginError
    Specific error types.
QPhaseSchedulerError, QPhaseRuntimeError, QPhaseCLIError
    Execution errors.
QPhaseWarning
    Base warning class for all framework warnings.
get_logger, configure_logging
    Logging configuration utilities.
deprecated
    Decorator for marking deprecated functions.
"""

import logging
import os
import warnings
from collections.abc import Callable
from typing import Any, TypeVar, cast

__all__ = [
    "QPhaseError",
    "QPhaseWarning",
    "QPhaseIOError",
    "QPhaseConfigError",
    "QPhasePluginError",
    "QPhaseSchedulerError",
    "QPhaseRuntimeError",
    "QPhaseCLIError",
    "get_logger",
    "configure_logging",
    "deprecated",
]


# Base exception hierarchy
class QPhaseError(Exception):
    """Base exception for all qphase framework errors.

    This is the root exception class for all framework-specific errors.
    All other framework exceptions should inherit from this class.

    Examples
    --------
    >>> try:
    ...     raise QPhaseError("Something went wrong")
    ... except QPhaseError as e:
    ...     print(e)
    Something went wrong

    """

    pass


class QPhaseWarning(Warning):
    """Base warning for all qphase framework warnings.

    This is the root warning class for all framework-specific warnings.

    Examples
    --------
    >>> warnings.warn("This is a warning", QPhaseWarning)

    """

    pass


# Control layer specific errors
class QPhaseIOError(QPhaseError):
    """Input/output related errors.

    Raised when file operations, network requests, or other I/O operations fail.

    Examples
    --------
    >>> raise QPhaseIOError("Failed to read file")
    Traceback (most recent call last):
    ...
    QPhaseIOError: Failed to read file

    """

    pass


class QPhaseConfigError(QPhaseError):
    """Configuration and validation errors.

    Raised when configuration files are invalid, missing required fields,
    or contain incompatible settings.

    Examples
    --------
    >>> raise QPhaseConfigError("Invalid configuration value")
    Traceback (most recent call last):
    ...
    QPhaseConfigError: Invalid configuration value

    """

    pass


class QPhasePluginError(QPhaseError):
    """Plugin-related errors.

    Raised when plugin operations fail, including:
    - Plugin not found during lookup
    - Plugin instantiation failures
    - Plugin execution errors

    Note: This wraps plugin-specific errors from the registry layer.

    Examples
    --------
    >>> raise QPhasePluginError("Failed to instantiate plugin")
    Traceback (most recent call last):
    ...
    QPhasePluginError: Failed to instantiate plugin

    """

    pass


class QPhaseSchedulerError(QPhaseError):
    """Scheduler and job orchestration errors.

    Raised when job scheduling, execution, or orchestration fails,
    including job dependency resolution and resource allocation issues.

    Examples
    --------
    >>> raise QPhaseSchedulerError("Job execution failed")
    Traceback (most recent call last):
    ...
    QPhaseSchedulerError: Job execution failed

    """

    pass


class QPhaseRuntimeError(QPhaseError):
    """Resource package execution errors (wrapped).

    Raised when a resource package (e.g., SDE models) fails during execution.
    This error wraps exceptions from external resource packages to provide
    context while maintaining error boundaries.

    Examples
    --------
    >>> raise QPhaseRuntimeError("Model execution failed")
    Traceback (most recent call last):
    ...
    QPhaseRuntimeError: Model execution failed

    """

    pass


class QPhaseCLIError(QPhaseError):
    """Command-line interface errors.

    Raised when CLI commands fail, arguments are invalid, or
    command execution encounters errors.

    Examples
    --------
    >>> raise QPhaseCLIError("Invalid command arguments")
    Traceback (most recent call last):
    ...
    QPhaseCLIError: Invalid command arguments

    """

    pass


# Logger
_logger: logging.Logger | None = None


def get_logger() -> logging.Logger:
    """Get the shared qphase logger instance.

    Returns
    -------
    logging.Logger
        The singleton logger named ``"qphase"`` configured at INFO level by
        default with a console handler. Handlers are created lazily on first use.

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.name
    'qphase'

    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("qphase")
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
        warned_attr = "__qphase_deprecated_warned__"

        if callable(obj):

            def _wrapped(*args, **kwargs):
                if not getattr(_wrapped, warned_attr, False):
                    name = getattr(obj, "__name__", str(obj))
                    msg = f"[990] DEPRECATED: {name}: " + str(reason)
                    warnings.warn(msg, QPhaseWarning, stacklevel=2)
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
