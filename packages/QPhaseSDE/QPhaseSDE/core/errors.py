from __future__ import annotations

"""Centralized errors, warnings, and logging for QPhaseSDE.

All modules must raise subclasses of QPhaseSDEError and emit warnings via
QPhaseSDEWarning (or the logger at WARNING level). A single logger is exposed
and configurable.
"""

import logging
import os
from typing import Optional

__all__ = [
    "QPhaseSDEError",
    "EngineError",
    "BackendError",
    "BackendCapabilityError",
    "IntegratorError",
    "NoiseModelError",
    "StateError",
    "RegistryError",
    "ConfigError",
    "ConflictError",
    "ImportBackendError",
    "ImportVisualizerError",
    "SDEIOError",
    "QPhaseSDEWarning",
    "get_logger",
    "configure_logging",
]


# Exception hierarchy
class QPhaseSDEError(Exception):
    pass


class EngineError(QPhaseSDEError):
    pass


class BackendError(QPhaseSDEError):
    pass


class BackendCapabilityError(BackendError):
    pass


class IntegratorError(QPhaseSDEError):
    pass


class NoiseModelError(QPhaseSDEError):
    pass


class StateError(QPhaseSDEError):
    pass


class RegistryError(QPhaseSDEError):
    pass


class ConfigError(QPhaseSDEError):
    pass


class ConflictError(ConfigError):
    pass


class ImportBackendError(BackendError):
    pass


class ImportVisualizerError(QPhaseSDEError):
    pass


class SDEIOError(QPhaseSDEError):
    pass


# Warning hierarchy
class QPhaseSDEWarning(Warning):
    pass


# Logger
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
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
