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
from pathlib import Path
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
    "ErrorCode",
    "get_logger",
    "configure_logging",
    "attach_session_log",
    "deprecated",
]


class ErrorCode:
    """Stable machine-readable error codes, partitioned by failure boundary."""

    CONFIG = "config"
    PLUGIN_DISCOVERY = "plugin_discovery"
    PLUGIN_CREATION = "plugin_creation"
    INPUT = "input"
    ENGINE_RUNTIME = "engine_runtime"
    ARTIFACT_IO = "artifact_io"
    CHECKPOINT = "checkpoint"
    CANCELLATION = "cancellation"
    UNKNOWN = "unknown"


# Base exception hierarchy
class QPhaseError(Exception):
    """Base exception for all qphase framework errors.

    This is the root exception class for all framework-specific errors.
    All other framework exceptions should inherit from this class.

    Parameters
    ----------
    message : str
        User-readable, single-line error summary.
    code : str | None
        Stable machine-readable error code (see ``ErrorCode``). When omitted,
        the reporting layer derives one from the exception type.
    hint : str | None
        Optional actionable suggestion shown in CLI error briefs.
    context : dict | None
        Small structured context (engine, stage, plugin, scan point...) merged
        into the final error report. Never store large arrays here.

    Examples
    --------
    >>> try:
    ...     raise QPhaseError("Something went wrong")
    ... except QPhaseError as e:
    ...     print(e)
    Something went wrong

    """

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        hint: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.hint = hint
        self.context: dict[str, Any] = dict(context) if context else {}


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


class _BriefFormatter(logging.Formatter):
    """Console formatter that never emits tracebacks.

    Full exception details belong to the log file and the structured error
    report; the console only shows the single-line message.
    """

    def format(self, record: logging.LogRecord) -> str:
        exc_info, exc_text = record.exc_info, record.exc_text
        record.exc_info, record.exc_text = None, None
        try:
            return super().format(record)
        finally:
            record.exc_info, record.exc_text = exc_info, exc_text


class _JsonLogFormatter(logging.Formatter):
    """Compact JSON-lines formatter for file logs."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "session": getattr(record, "session_id", None),
            "job": getattr(record, "job", None),
            "engine": getattr(record, "engine", None),
            "stage": getattr(record, "stage", None),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_FILE_TEXT_FORMAT = (
    "%(asctime)s %(levelname)s %(name)s "
    "[session=%(session_id)s job=%(job)s engine=%(engine)s stage=%(stage)s] "
    "%(message)s"
)
_CONSOLE_TEXT_FORMAT = "[%(levelname)s] %(message)s"


def _file_formatter(as_json: bool) -> logging.Formatter:
    if as_json:
        return _JsonLogFormatter()
    return logging.Formatter(_FILE_TEXT_FORMAT)


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
    *,
    console_level: int | str = logging.WARNING,
    file_level: int | str = logging.DEBUG,
    console_handler: logging.Handler | None = None,
) -> None:
    """Configure the shared logger outputs and warning capture.

    The logger itself runs at DEBUG so attached file handlers capture
    everything. The console handler only carries warnings and errors as
    brief one-line messages; normal lifecycle output belongs to the CLI
    renderer, and full DEBUG content belongs to the session log file.

    Parameters
    ----------
    verbose : bool, default False
        Kept for CLI compatibility. When True the console handler level is
        lowered to INFO.
    log_file : str or None, default None
        Optional explicit file path for an additional file log. Failures to
        create it produce one explicit warning instead of being ignored.
    as_json : bool, default False
        Emit file logs in a compact JSON line format. Only affects file
        handlers; the console stays human-readable.
    suppress_warnings : bool, default False
        Route Python warnings into logging and raise their level to ERROR when
        True; otherwise capture warnings at WARNING level.
    console_level : int or str, default logging.WARNING
        Level for the console handler.
    file_level : int or str, default logging.DEBUG
        Level for file handlers.
    console_handler : logging.Handler or None, default None
        Optional pre-built handler (e.g. one routing through the CLI progress
        renderer) replacing the default stderr console handler.

    Examples
    --------
    >>> configure_logging(verbose=True, as_json=False)  # doctest: +SKIP
    >>> logger = get_logger()
    >>> logger.level == logging.DEBUG
    True

    """
    from .logging_context import LogContextFilter

    logger = get_logger()
    # Clear existing handlers
    for h in list(logger.handlers):
        logger.removeHandler(h)
    logger.setLevel(logging.DEBUG)

    # Console handler: brief, warnings and errors only, never a traceback.
    if verbose and console_level == logging.WARNING:
        console_level = logging.INFO
    ch = console_handler if console_handler is not None else logging.StreamHandler()
    ch.setLevel(console_level)
    if console_handler is None:
        ch.setFormatter(_BriefFormatter(_CONSOLE_TEXT_FORMAT))
        ch.addFilter(LogContextFilter())
    logger.addHandler(ch)

    # Explicit file handler (in addition to the per-session log file).
    if log_file:
        try:
            path = os.fspath(log_file)
            fh = logging.FileHandler(path, encoding="utf-8")
            fh.setLevel(file_level)
            fh.setFormatter(_file_formatter(as_json))
            fh.addFilter(LogContextFilter())
            logger.addHandler(fh)
        except Exception as exc:
            logger.warning(
                f"Could not create log file '{log_file}': {exc}. "
                "File logging for this path is disabled."
            )

    if suppress_warnings:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.ERROR)
    else:
        logging.captureWarnings(True)
        logging.getLogger("py.warnings").setLevel(logging.WARNING)


def attach_session_log(
    session_dir: str | Path,
    *,
    filename: str = "qphase.log",
    level: int | str = logging.DEBUG,
    as_json: bool = False,
) -> tuple[Path | None, logging.Handler | None]:
    """Attach the per-session DEBUG log file under ``session_dir``.

    Returns
    -------
    tuple[Path | None, logging.Handler | None]
        The log path and its handler (so callers can detach it later). Both
        are None when the file cannot be created; in that case one explicit
        warning is emitted instead of failing silently.

    """
    from .logging_context import LogContextFilter

    logger = get_logger()
    path = Path(session_dir) / filename
    try:
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(_file_formatter(as_json))
        handler.addFilter(LogContextFilter())
        logger.addHandler(handler)
        return path, handler
    except Exception as exc:
        logger.warning(
            f"Could not create session log file '{path}': {exc}. "
            "Session file logging is disabled."
        )
        return None, None


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
