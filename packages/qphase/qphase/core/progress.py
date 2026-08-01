"""qphase: Structured Progress Model
---------------------------------------------------------
Defines the structured progress pipeline shared by resource engines, the core
scheduler, and output renderers. Engines emit immutable ``ProgressEvent`` work
records (counts, units, stages) instead of interpreting time semantics; the core
``ProgressTracker`` aggregates events per ``(stage, unit)`` scope and produces
low-cost rate/ETA estimates; consumers receive a ``ProgressSnapshot`` DTO that
carries exactly the information needed to render a brief CLI view or a service
response.

Public API
----------
``ProgressEvent`` : Immutable work-progress record emitted by engines.
``ProgressReporter`` : Engine-facing API for emitting progress events.
``ProgressTracker`` : Core aggregator with EMA rate and remaining-time estimates.
``ProgressSnapshot`` : Outward-facing DTO consumed by CLI, service, and GUI.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from .errors import QPhaseWarning, get_logger

__all__ = [
    "ProgressEvent",
    "ProgressEventKind",
    "ProgressReporter",
    "ProgressSnapshot",
    "ProgressTracker",
    "SnapshotKind",
]

log = get_logger()

ProgressEventKind = Literal["stage_started", "progress", "status", "stage_completed"]
SnapshotKind = Literal[
    "job_started",
    "job_progress",
    "job_status",
    "job_completed",
    "job_failed",
    "job_skipped",
]

#: Stage ID reserved for job-spanning, monotonically increasing work. Only
#: events in this scope license a whole-job ETA.
JOB_STAGE = "job"


@dataclass(frozen=True)
class ProgressEvent:
    """One immutable work-progress record emitted by a resource engine.

    Attributes
    ----------
    kind : ProgressEventKind
        ``stage_started`` opens a new estimation scope, ``progress`` carries
        work counts, ``status`` carries a message-only update, and
        ``stage_completed`` closes a scope.
    stage : str | None
        Stable, machine-readable stage ID (e.g. ``"solve"``, ``"integrate"``).
    completed, total : float | None
        Work counts in ``unit``. When ``total`` is known the invariant
        ``0 <= completed <= total`` holds.
    unit : str | None
        Natural work unit (``"tile"``, ``"step"``, ``"chunk"``, ``"plot"``...).
    message : str
        Human-readable supplement; never parsed for counts.
    metadata : Mapping[str, Any]
        Small structured extras (worker count, scan shape). Large arrays are
        forbidden.
    monotonic_time : float
        Filled by core when the event is received; ``0.0`` means unset.

    """

    kind: ProgressEventKind = "progress"
    stage: str | None = None
    completed: float | None = None
    total: float | None = None
    unit: str | None = None
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    monotonic_time: float = 0.0

    def __post_init__(self) -> None:
        """Validate work-count invariants."""
        if self.completed is not None and self.completed < 0:
            raise ValueError("completed must be >= 0")
        if self.total is not None:
            if self.total < 0:
                raise ValueError("total must be >= 0")
            if self.completed is not None and self.completed > self.total:
                raise ValueError("completed must not exceed total")


@dataclass(frozen=True)
class ProgressSnapshot:
    """Outward DTO describing the progress state of one logical job.

    ``remaining`` estimates the current stage only; it equals a whole-job ETA
    exclusively when ``stage == "job"``. ``rate``/``remaining`` are ``None``
    until the tracker has warmed up, and consumers must not fabricate values.
    """

    kind: SnapshotKind
    job_name: str
    job_index: int
    total_jobs: int
    engine: str | None = None
    stage: str | None = None
    completed: float | None = None
    total: float | None = None
    unit: str | None = None
    fraction: float | None = None
    elapsed: float = 0.0
    rate: float | None = None
    remaining: float | None = None
    message: str = ""
    scan_summary: dict[str, Any] | None = None
    duration: float | None = None
    run_dir: str | None = None
    error: Any | None = None
    monotonic_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        error = self.error
        if error is not None and hasattr(error, "to_dict"):
            error = error.to_dict()
        return {
            "kind": self.kind,
            "job_name": self.job_name,
            "job_index": self.job_index,
            "total_jobs": self.total_jobs,
            "engine": self.engine,
            "stage": self.stage,
            "completed": self.completed,
            "total": self.total,
            "unit": self.unit,
            "fraction": self.fraction,
            "elapsed": self.elapsed,
            "rate": self.rate,
            "remaining": self.remaining,
            "message": self.message,
            "scan_summary": self.scan_summary,
            "duration": self.duration,
            "run_dir": self.run_dir,
            "error": error,
        }


class ProgressReporter:
    """Stable progress API handed to resource engines via ``ExecutionContext``.

    Engines report work with :meth:`update`/:meth:`advance` and group it with
    the :meth:`stage` context manager. The legacy ``report(percent,
    total_duration_estimate, ...)`` call keeps working for one deprecation
    cycle and is converted to a fractional ``completed/total`` event; the
    engine-provided duration estimate is ignored.
    """

    def __init__(
        self,
        emit: Callable[[ProgressEvent], None] | None = None,
    ) -> None:
        self._emit = emit
        self._stage: str | None = None
        self._stage_total: float | None = None
        self._stage_unit: str | None = None
        self._completed: float = 0.0
        self._legacy_warned = False

    # ------------------------------------------------------------------
    # New work-based API
    # ------------------------------------------------------------------
    def update(
        self,
        completed: float | None = None,
        total: float | None = None,
        *,
        unit: str | None = None,
        stage: str | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Report absolute work counts for the current or given stage."""
        effective_stage = stage if stage is not None else self._stage
        if effective_stage == self._stage:
            if total is None:
                total = self._stage_total
            if unit is None:
                unit = self._stage_unit
            if completed is not None:
                self._completed = float(completed)
        self._send(
            ProgressEvent(
                kind="progress",
                stage=effective_stage,
                completed=completed,
                total=total,
                unit=unit,
                message=message,
                metadata=dict(metadata or {}),
            )
        )

    def advance(
        self,
        amount: float = 1.0,
        *,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Increment the work counter of the current stage and report it."""
        self._completed += amount
        self.update(completed=self._completed, message=message, metadata=metadata)

    def status(
        self,
        message: str,
        *,
        stage: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit a message-only update that does not affect work estimates."""
        self._send(
            ProgressEvent(
                kind="status",
                stage=stage if stage is not None else self._stage,
                message=message,
                metadata=dict(metadata or {}),
            )
        )

    @contextmanager
    def stage(
        self,
        stage_id: str,
        *,
        total: float | None = None,
        unit: str | None = None,
        message: str = "",
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[ProgressReporter]:
        """Open a progress stage; resets the :meth:`advance` counter."""
        previous = (self._stage, self._stage_total, self._stage_unit, self._completed)
        self._stage, self._stage_total, self._stage_unit = stage_id, total, unit
        self._completed = 0.0
        self._send(
            ProgressEvent(
                kind="stage_started",
                stage=stage_id,
                total=total,
                unit=unit,
                message=message,
                metadata=dict(metadata or {}),
            )
        )
        try:
            yield self
        finally:
            self._send(
                ProgressEvent(
                    kind="stage_completed",
                    stage=stage_id,
                    completed=self._completed,
                    total=total,
                    unit=unit,
                )
            )
            (
                self._stage,
                self._stage_total,
                self._stage_unit,
                self._completed,
            ) = previous

    # ------------------------------------------------------------------
    # Legacy percent API (one deprecation cycle)
    # ------------------------------------------------------------------
    def report(
        self,
        percent: float | None,
        *,
        total_duration_estimate: float | None = None,
        message: str = "",
        stage: str | None = None,
    ) -> None:
        """Adapt a deprecated percent-based report to a fractional event."""
        del total_duration_estimate  # engine duration estimates are ignored
        self._warn_legacy_once()
        if percent is None:
            self._send(ProgressEvent(kind="status", stage=stage, message=message))
            return
        clamped = min(max(float(percent), 0.0), 1.0)
        self._send(
            ProgressEvent(
                kind="progress",
                stage=stage,
                completed=clamped,
                total=1.0,
                message=message,
            )
        )

    def __call__(
        self,
        percent: float | None,
        total_duration_estimate: float | None = None,
        message: str = "",
        stage: str | None = None,
    ) -> None:
        """Support the legacy ``(percent, total_est, message, stage)`` call."""
        self.report(
            percent,
            total_duration_estimate=total_duration_estimate,
            message=message,
            stage=stage,
        )

    def legacy_callback(
        self,
    ) -> Callable[[float | None, float | None, str, str | None], None]:
        """Return an old-signature callback that feeds this reporter.

        Used by the scheduler to support engines that still accept
        ``progress_cb`` instead of consuming ``context.progress``.
        """

        def _callback(
            percent: float | None,
            total_duration_estimate: float | None,
            message: str,
            stage: str | None,
        ) -> None:
            self.report(
                percent,
                total_duration_estimate=total_duration_estimate,
                message=message,
                stage=stage,
            )

        return _callback

    @classmethod
    def wrap_legacy(
        cls,
        callback: Callable[[float | None, float | None, str, str | None], None] | None,
    ) -> ProgressReporter:
        """Adapt an old percent callback so new-API callers can drive it."""
        reporter = cls()
        if callback is not None:

            def _emit(event: ProgressEvent) -> None:
                fraction = None
                if (
                    event.completed is not None
                    and event.total is not None
                    and event.total > 0
                ):
                    fraction = min(max(event.completed / event.total, 0.0), 1.0)
                callback(fraction, None, event.message, event.stage)

            reporter._emit = _emit
        return reporter

    # ------------------------------------------------------------------
    def _send(self, event: ProgressEvent) -> None:
        if self._emit is not None:
            self._emit(event)

    def _warn_legacy_once(self) -> None:
        if self._legacy_warned:
            return
        self._legacy_warned = True
        message = (
            "[991] DEPRECATED: ProgressReporter.report(percent, "
            "total_duration_estimate=...) is deprecated; use "
            "update(completed, total, unit=..., stage=...) instead."
        )
        warnings.warn(message, QPhaseWarning, stacklevel=3)
        log.warning(message)


@dataclass
class _StageStats:
    """Estimator state for one ``(stage, unit)`` scope."""

    started_at: float
    total: float | None = None
    last_completed: float | None = None
    last_time: float | None = None
    rate_ema: float | None = None
    samples: int = 0


class ProgressTracker:
    """Aggregate progress events for one job and estimate stage rate/ETA.

    Statistics are scoped per ``(stage, unit)`` and reset when a stage is
    re-opened, the total changes, or the completed count regresses, so
    heterogeneous algorithm phases never mix into one estimate. Estimates use
    a light EMA and stay hidden until both the warm-up period has passed and
    enough valid samples exist. Unknown totals yield no fabricated fraction or
    ETA.
    """

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        eta_warmup_seconds: float = 2.0,
        eta_min_samples: int = 3,
        eta_smoothing: float = 0.25,
    ) -> None:
        self._clock = clock
        self._warmup = float(eta_warmup_seconds)
        self._min_samples = int(eta_min_samples)
        self._smoothing = float(eta_smoothing)
        self._started_at: float | None = None
        self._stages: dict[tuple[str | None, str | None], _StageStats] = {}
        self._last_event: ProgressEvent | None = None
        self._current_stage: str | None = None

    @property
    def current_stage(self) -> str | None:
        """Most recent non-empty stage ID observed, for error context."""
        return self._current_stage

    @property
    def started_at(self) -> float | None:
        return self._started_at

    def observe(self, event: ProgressEvent) -> ProgressEvent:
        """Ingest one event, filling ``monotonic_time`` when unset."""
        if event.monotonic_time <= 0.0:
            event = replace(event, monotonic_time=self._clock())
        now = event.monotonic_time
        if self._started_at is None:
            self._started_at = now
        if event.stage:
            self._current_stage = event.stage

        if event.kind == "status":
            self._last_event = event
            return event

        key = (event.stage, event.unit)
        stats = self._stages.get(key)
        reset = stats is None or event.kind == "stage_started"
        if not reset and stats is not None:
            if (
                event.total is not None
                and stats.total is not None
                and event.total != stats.total
            ):
                reset = True
            elif (
                event.completed is not None
                and stats.last_completed is not None
                and event.completed < stats.last_completed
            ):
                reset = True

        if reset or stats is None:
            stats = _StageStats(started_at=now, total=event.total)
            self._stages[key] = stats
            stats.last_completed = event.completed
            stats.last_time = now
        else:
            if (
                event.kind == "progress"
                and event.completed is not None
                and stats.last_completed is not None
                and stats.last_time is not None
            ):
                delta_completed = event.completed - stats.last_completed
                delta_time = now - stats.last_time
                if delta_completed > 0 and delta_time > 0:
                    instant_rate = delta_completed / delta_time
                    if stats.rate_ema is None:
                        stats.rate_ema = instant_rate
                    else:
                        stats.rate_ema = (
                            self._smoothing * instant_rate
                            + (1.0 - self._smoothing) * stats.rate_ema
                        )
                    stats.samples += 1
            if event.completed is not None:
                stats.last_completed = event.completed
            if event.total is not None:
                stats.total = event.total
            stats.last_time = now

        self._last_event = event
        return event

    def elapsed(self, event: ProgressEvent) -> float:
        """Seconds between job start and this event."""
        if self._started_at is None:
            return 0.0
        return max(event.monotonic_time - self._started_at, 0.0)

    def estimates(
        self, event: ProgressEvent
    ) -> tuple[float | None, float | None, float | None]:
        """Return ``(fraction, rate, remaining)`` for the event's stage scope.

        All three are ``None`` when they cannot be derived honestly: unknown
        or zero total (fraction/remaining), warm-up not finished, or too few
        samples (rate/remaining).
        """
        fraction = None
        if (
            event.completed is not None
            and event.total is not None
            and event.total > 0
        ):
            fraction = min(max(event.completed / event.total, 0.0), 1.0)

        stats = self._stages.get((event.stage, event.unit))
        if (
            stats is None
            or stats.rate_ema is None
            or stats.samples < self._min_samples
            or event.monotonic_time - stats.started_at < self._warmup
        ):
            return fraction, None, None

        rate = stats.rate_ema
        remaining = None
        if (
            event.total is not None
            and event.completed is not None
            and rate > 0
        ):
            remaining = max(event.total - event.completed, 0.0) / rate
        return fraction, rate, remaining
