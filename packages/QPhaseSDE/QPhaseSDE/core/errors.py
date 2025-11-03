"""
QPhaseSDE: Error Taxonomy and Logging
-------------------------------------
Unified exception types, warnings, and a shared logger for the framework. This
module centralizes error categorization and provides configuration helpers and
a deprecation decorator for standardized messaging.

Behavior
--------
- The shared logger is named "QPhaseSDE" and can be configured to console and
  file with optional JSON formatting.
- Python warnings are captured into logging with adjustable levels.
"""

import logging
import os
from typing import Optional, Callable, TypeVar, cast, Any
import warnings

__all__ = [
    "QPSError",
    "QPSIOError",
    "QPSBackendError",
    "QPSIntegratorError",
    "QPSModelError",
    "QPSConfigError",
    "QPSRegistryError",
    "QPSStateError",
    "QPSVisualizerError",
    "VisualizerConfigError",
    "QPSWarning",
    "get_logger",
    "configure_logging",
    "deprecated",
]

# Exception hierarchy
class QPSError(Exception):
    pass

class QPSRequirementError(QPSError): # Code Numbering: 000
    pass

class QPSIOError(QPSError): # Code Numbering: 100-103
    pass

class QPSBackendError(QPSError): # Code Numbering: 2xx
    pass

class QPSIntegratorError(QPSError):
    pass

class QPSRegistryError(QPSError):
    pass

class QPSConfigError(QPSError): # Code Numbering: 5xx
    pass

class QPSModelError(QPSError): # Code Numbering: 6xx, empty for now
    pass

class QPSStateError(QPSError): # Code Numbering: 7xx
    pass

class QPSVisualizerError(QPSError): # Code Numbering: 8xx
    pass

class VisualizerConfigError(QPSVisualizerError): # Code Numbering: 5xx
    pass

# Warning hierarchy
class QPSWarning(Warning): # Code Numbering: 9xx
    pass

# Logger
_logger: Optional[logging.Logger] = None

def get_logger() -> logging.Logger:
    """Get the shared QPhaseSDE logger instance.

    Returns
    -------
    logging.Logger
        The singleton logger named ``"QPhaseSDE"`` configured at INFO level by
        default with a console handler. Handlers are created lazily on first use.

    Examples
    --------
    >>> logger = get_logger()
    >>> logger.name
    'QPhaseSDE'
    """
    global _logger
    if _logger is None:
        _logger = logging.getLogger("QPhaseSDE")
        _logger.setLevel(logging.INFO)
        if not _logger.handlers:
            h = logging.StreamHandler()
            fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
            h.setFormatter(fmt)
            _logger.addHandler(h)
    return _logger

def configure_logging(verbose: bool = False,
                      log_file: Optional[str] = None,
                      as_json: bool = False,
                      suppress_warnings: bool = False) -> None:
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
        fmt = logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}')
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
    """Decorator to mark functions/classes as deprecated.

    On first call/instantiation, emits a ``QPhaseSDEWarning`` (code [990]) via
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
            def _wrapped(*args, **kwargs):  # type: ignore[misc]
                if not getattr(_wrapped, warned_attr, False):  # type: ignore[attr-defined]
                    msg = f"[990] DEPRECATED: {getattr(obj, '__name__', str(obj))}: {reason}"
                    warnings.warn(msg, QPSWarning, stacklevel=2)
                    logger.warning(msg)
                    setattr(_wrapped, warned_attr, True)  # type: ignore[attr-defined]
                return cast(Callable[..., Any], obj)(*args, **kwargs)  # type: ignore[call-arg]
            try:
                _wrapped.__name__ = getattr(obj, "__name__", _wrapped.__name__)  # type: ignore[attr-defined]
            except Exception:
                pass
            return cast(T, _wrapped)
        return obj
    return _decorator