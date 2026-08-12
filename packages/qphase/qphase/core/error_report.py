"""qphase: Structured Error Reports
---------------------------------------------------------
Defines the error reporting protocol for failed jobs. A single ``ErrorReport``
is constructed at the job boundary (where full context is available), stored
as ``error_report.json`` inside the failed job directory, and referenced by
summary from the session manifest, ``JobResult``, and the CLI brief. The CLI
never prints raw tracebacks; the full cause chain and traceback always live in
the report file and the session log.

Public API
----------
``ErrorReport`` : Full structured failure record with cause chain/traceback.
``ErrorSummary`` : Small reference DTO shared with CLI/manifest/snapshots.
``build_error_report`` : Construct a report from an exception plus context.
``save_error_report`` : Persist a report as ``error_report.json``.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import traceback as traceback_module
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .errors import (
    ErrorCode,
    QPhaseConfigError,
    QPhaseError,
    QPhaseIOError,
    QPhasePluginError,
    QPhaseRuntimeError,
    get_logger,
)

__all__ = [
    "ErrorReport",
    "ErrorSummary",
    "build_error_report",
    "classify_exception",
    "save_error_report",
]

log = get_logger()

ERROR_REPORT_FILENAME = "error_report.json"


@dataclass(frozen=True)
class ErrorSummary:
    """Small error reference embedded in snapshots, manifests, and results."""

    error_id: str
    code: str
    summary: str
    hint: str | None = None
    report_path: str | None = None
    log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ErrorReport:
    """Full structured record of one job failure."""

    error_id: str
    code: str
    category: str
    timestamp: str
    summary: str
    exception_type: str
    hint: str | None = None
    session_id: str | None = None
    job_name: str | None = None
    engine: str | None = None
    stage: str | None = None
    plugin: str | None = None
    job_dir: str | None = None
    scan_context: dict[str, Any] | None = None
    context: dict[str, Any] = field(default_factory=dict)
    cause_chain: list[dict[str, str]] = field(default_factory=list)
    traceback: str = ""
    python_version: str = ""
    qphase_version: str = ""
    log_file: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_dto(
        self,
        *,
        report_path: str | None = None,
        log_file: str | None = None,
    ) -> ErrorSummary:
        """Build the small reference DTO for manifests and CLI briefs."""
        return ErrorSummary(
            error_id=self.error_id,
            code=self.code,
            summary=self.summary,
            hint=self.hint,
            report_path=report_path,
            log_file=log_file if log_file is not None else self.log_file,
        )


def classify_exception(exc: BaseException) -> str:
    """Map an exception to a stable error code by failure boundary."""
    if isinstance(exc, QPhaseError) and exc.code:
        return exc.code
    if isinstance(exc, QPhaseConfigError):
        return ErrorCode.CONFIG
    if isinstance(exc, QPhasePluginError):
        return ErrorCode.PLUGIN_CREATION
    if isinstance(exc, QPhaseIOError):
        return ErrorCode.ARTIFACT_IO
    if isinstance(exc, QPhaseRuntimeError):
        return ErrorCode.ENGINE_RUNTIME
    if isinstance(exc, KeyboardInterrupt):
        return ErrorCode.CANCELLATION
    try:
        from pydantic import ValidationError

        if isinstance(exc, ValidationError):
            return ErrorCode.CONFIG
    except ImportError:  # pragma: no cover - pydantic is a hard dependency
        pass
    return ErrorCode.UNKNOWN


def _cause_chain(exc: BaseException) -> list[dict[str, str]]:
    """Flatten the ``raise ... from ...`` chain into type/message records."""
    chain: list[dict[str, str]] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append({"type": type(current).__name__, "message": str(current)})
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return chain


def _merged_context(exc: BaseException) -> dict[str, Any]:
    """Merge ``QPhaseError.context`` dicts along the chain, outermost wins."""
    merged: dict[str, Any] = {}
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        context = getattr(current, "context", None)
        if isinstance(current, QPhaseError) and context:
            for key, value in context.items():
                merged.setdefault(key, value)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return merged


def _find_hint(exc: BaseException) -> str | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        hint = getattr(current, "hint", None)
        if hint:
            return str(hint)
        if current.__cause__ is not None:
            current = current.__cause__
        elif not current.__suppress_context__:
            current = current.__context__
        else:
            current = None
    return None


def _qphase_version() -> str:
    for distribution in ("qphase", "qphase-workspace"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "unknown"


def build_error_report(
    exc: BaseException,
    *,
    session_id: str | None = None,
    job_name: str | None = None,
    engine: str | None = None,
    stage: str | None = None,
    plugin: str | None = None,
    job_dir: str | Path | None = None,
    scan_context: dict[str, Any] | None = None,
    log_file: str | Path | None = None,
) -> ErrorReport:
    """Construct an ``ErrorReport`` from an exception and job context.

    The summary is the first line of the outermost exception message; the full
    cause chain and traceback are always captured for the file report.
    """
    code = classify_exception(exc)
    raw_summary = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    if not raw_summary:
        raw_summary = type(exc).__name__
    context = _merged_context(exc)
    return ErrorReport(
        error_id=uuid.uuid4().hex[:12],
        code=code,
        category=code.split("_")[0],
        timestamp=datetime.now(UTC).isoformat(),
        summary=raw_summary,
        hint=_find_hint(exc),
        exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
        session_id=session_id,
        job_name=job_name,
        engine=engine or _string_or_none(context.get("engine")),
        stage=stage or _string_or_none(context.get("stage")),
        plugin=plugin or _string_or_none(context.get("plugin")),
        job_dir=str(job_dir) if job_dir is not None else None,
        scan_context=scan_context,
        context=context,
        cause_chain=_cause_chain(exc),
        traceback="".join(
            traceback_module.format_exception(type(exc), exc, exc.__traceback__)
        ),
        python_version=platform.python_version(),
        qphase_version=_qphase_version(),
        log_file=str(log_file) if log_file is not None else None,
    )


def save_error_report(report: ErrorReport, directory: str | Path) -> Path | None:
    """Persist the report as ``error_report.json`` inside ``directory``.

    Returns the written path, or None when writing fails (a warning is logged;
    reporting a failure must never mask the original error).
    """
    try:
        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / ERROR_REPORT_FILENAME
        path.write_text(
            json.dumps(report.to_dict(), indent=2, default=str),
            encoding="utf-8",
        )
        return path
    except Exception as exc:  # pragma: no cover - defensive
        log.warning(f"Could not write error report for job failure: {exc}")
        return None


def format_validation_issues(exc: BaseException, *, max_issues: int = 8) -> str | None:
    """Compact field-path list for Pydantic configuration errors."""
    try:
        from pydantic import ValidationError
    except ImportError:  # pragma: no cover
        return None
    if not isinstance(exc, ValidationError):
        return None
    lines = []
    for error in exc.errors()[:max_issues]:
        location = ".".join(str(part) for part in error.get("loc", ()))
        lines.append(f"  - {location or '<root>'}: {error.get('msg', '')}")
    remaining = len(exc.errors()) - max_issues
    if remaining > 0:
        lines.append(f"  ... and {remaining} more")
    return "\n".join(lines)


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)
