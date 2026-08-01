"""qphase: Logging Context Propagation
---------------------------------------------------------
Carries session/job/engine/stage context through ``contextvars`` so that log
records can be enriched centrally instead of every call site stitching context
into its messages. The scheduler binds context around job execution; the
``LogContextFilter`` copies the current values onto each log record for the
file formatters.

Public API
----------
``bind_log_context`` : Context manager that sets context fields for a scope.
``set_log_context`` : Update individual fields in the current scope.
``LogContextFilter`` : Logging filter injecting context fields into records.
"""

from __future__ import annotations

import contextvars
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

__all__ = ["LogContextFilter", "bind_log_context", "set_log_context"]

_FIELDS = ("session_id", "job", "engine", "stage")

_log_context: contextvars.ContextVar[dict[str, Any] | None] = contextvars.ContextVar(
    "qphase_log_context", default=None
)


@contextmanager
def bind_log_context(**fields: Any) -> Iterator[None]:
    """Bind context fields (session_id, job, engine, stage) for a scope."""
    current = _log_context.get() or {}
    merged = {**current, **{k: v for k, v in fields.items() if v is not None}}
    token = _log_context.set(merged)
    try:
        yield
    finally:
        _log_context.reset(token)


def set_log_context(**fields: Any) -> None:
    """Update individual context fields within the current scope."""
    current = _log_context.get() or {}
    _log_context.set({**current, **{k: v for k, v in fields.items() if v is not None}})


class LogContextFilter(logging.Filter):
    """Ensure every record carries the context attributes used by formatters."""

    def filter(self, record: logging.LogRecord) -> bool:
        context = _log_context.get() or {}
        for name in _FIELDS:
            value = context.get(name)
            setattr(record, name, "-" if value in (None, "") else str(value))
        return True
