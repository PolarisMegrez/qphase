"""Terminal rendering for structured scheduler progress snapshots."""

from __future__ import annotations

import logging
import sys
import time
from typing import TextIO

from qphase.core.progress import ProgressSnapshot


class CliProgressRenderer:
    """Render concise progress without turning the terminal into a log stream."""

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        is_tty: bool | None = None,
        verbose: bool = False,
        refresh_interval: float = 0.5,
        milestone_percent: float = 10.0,
        clock=time.monotonic,
    ) -> None:
        self.stream = stream or sys.stdout
        self.is_tty = self.stream.isatty() if is_tty is None else is_tty
        self.verbose = verbose
        self.refresh_interval = refresh_interval
        self.milestone_percent = milestone_percent
        self.clock = clock
        self._active = False
        self._last_refresh = 0.0
        self._last_stage: tuple[str, str | None] | None = None
        self._last_milestone: dict[tuple[str, str | None], int] = {}

    def handle(self, snapshot: ProgressSnapshot) -> None:
        """Render one scheduler snapshot."""
        if snapshot.kind == "job_started":
            self._write_line(self._format_start(snapshot))
            return
        if snapshot.kind in {"job_completed", "job_failed", "job_skipped"}:
            self._clear_active()
            self._write_line(self._format_terminal(snapshot))
            return
        if snapshot.kind == "job_status" and not self.verbose:
            return
        if self.is_tty:
            self._render_tty(snapshot)
        else:
            self._render_non_tty(snapshot)

    def write_log(self, level: str, message: str) -> None:
        """Print one warning/error without corrupting the active progress line."""
        self._clear_active()
        self._write_line(f"{level}: {message}")

    def _render_tty(self, snapshot: ProgressSnapshot) -> None:
        now = self.clock()
        stage_changed = self._stage_key(snapshot) != self._last_stage
        if (
            not stage_changed
            and now - self._last_refresh < self.refresh_interval
            and snapshot.fraction not in {1.0}
        ):
            return
        self._last_stage = self._stage_key(snapshot)
        self._last_refresh = now
        text = self._format_progress(snapshot)
        self.stream.write(f"\r{text:<100}")
        self.stream.flush()
        self._active = True

    def _render_non_tty(self, snapshot: ProgressSnapshot) -> None:
        key = self._stage_key(snapshot)
        stage_changed = key != self._last_stage
        self._last_stage = key
        milestone = None
        if snapshot.fraction is not None:
            milestone = int(
                snapshot.fraction * 100.0 // self.milestone_percent
            )
        if not stage_changed and milestone is not None:
            if milestone <= self._last_milestone.get(key, -1):
                return
        elif not stage_changed and snapshot.kind != "job_status":
            return
        if milestone is not None:
            self._last_milestone[key] = milestone
        self._write_line(self._format_progress(snapshot))

    def _format_start(self, snapshot: ProgressSnapshot) -> str:
        prefix = self._job_prefix(snapshot)
        parts = [prefix, snapshot.engine or "unknown engine"]
        summary = snapshot.scan_summary
        if summary:
            shape = "x".join(str(value) for value in summary.get("shape", ()))
            parts.append(f"scan {shape} ({summary.get('size', '?')} points)")
        return " | ".join(parts)

    def _format_progress(self, snapshot: ProgressSnapshot) -> str:
        parts = [self._job_prefix(snapshot)]
        if snapshot.stage:
            parts.append(snapshot.stage)
        if snapshot.completed is not None:
            completed = _format_count(snapshot.completed)
            if snapshot.total is not None:
                completed += f"/{_format_count(snapshot.total)}"
            if snapshot.unit:
                completed += f" {snapshot.unit}"
            parts.append(completed)
        if snapshot.fraction is not None:
            parts.append(f"{snapshot.fraction * 100.0:.1f}%")
        if snapshot.rate is not None and snapshot.unit:
            parts.append(f"{snapshot.rate:.2g} {snapshot.unit}/s")
        if snapshot.remaining is not None:
            parts.append(f"remaining ~{_format_duration(snapshot.remaining)}")
        else:
            parts.append(f"elapsed {_format_duration(snapshot.elapsed)}")
        if snapshot.message and (self.verbose or snapshot.completed is None):
            parts.append(snapshot.message)
        return " | ".join(parts)

    def _format_terminal(self, snapshot: ProgressSnapshot) -> str:
        prefix = self._job_prefix(snapshot)
        if snapshot.kind == "job_completed":
            text = f"{prefix} completed in {_format_duration(snapshot.duration or 0)}"
            return f"{text} | {snapshot.run_dir}" if snapshot.run_dir else text
        if snapshot.kind == "job_skipped":
            return f"{prefix} skipped | {snapshot.message}"
        error = snapshot.error
        code = getattr(error, "code", None)
        summary = getattr(error, "summary", None) or snapshot.message
        error_id = getattr(error, "error_id", None)
        report_path = getattr(error, "report_path", None)
        label = " ".join(value for value in (code, error_id) if value)
        text = f"{prefix} failed{f' [{label}]' if label else ''} | {summary}"
        return f"{text} | details: {report_path}" if report_path else text

    @staticmethod
    def _stage_key(snapshot: ProgressSnapshot) -> tuple[str, str | None]:
        return snapshot.job_name, snapshot.stage

    @staticmethod
    def _job_prefix(snapshot: ProgressSnapshot) -> str:
        return (
            f"[{snapshot.job_index + 1}/{snapshot.total_jobs}] "
            f"{snapshot.job_name}"
        )

    def _clear_active(self) -> None:
        if not self._active:
            return
        self.stream.write("\r" + " " * 100 + "\r")
        self.stream.flush()
        self._active = False

    def _write_line(self, text: str) -> None:
        self.stream.write(text + "\n")
        self.stream.flush()


class ProgressLogHandler(logging.Handler):
    """Route console warnings/errors through the progress renderer."""

    def __init__(self, renderer: CliProgressRenderer) -> None:
        super().__init__()
        self.renderer = renderer

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.renderer.write_log(record.levelname, record.getMessage())
        except Exception:
            self.handleError(record)


def _format_count(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )
