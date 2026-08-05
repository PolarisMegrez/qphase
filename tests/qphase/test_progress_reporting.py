from __future__ import annotations

import io

import pytest
from qphase.commands.progress import CliProgressRenderer
from qphase.core.progress import ProgressEvent, ProgressSnapshot, ProgressTracker

pytestmark = pytest.mark.unit


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_tracker_estimates_only_the_current_stage_after_warmup() -> None:
    clock = FakeClock()
    tracker = ProgressTracker(
        clock=clock,
        eta_warmup_seconds=2.0,
        eta_min_samples=3,
        eta_smoothing=1.0,
    )

    event = tracker.observe(
        ProgressEvent(stage="solve", completed=0, total=10, unit="tile")
    )
    assert tracker.estimates(event) == (0.0, None, None)

    for completed in (1, 2, 3):
        clock.advance(1.0)
        event = tracker.observe(
            ProgressEvent(
                stage="solve", completed=completed, total=10, unit="tile"
            )
        )

    fraction, rate, remaining = tracker.estimates(event)
    assert fraction == pytest.approx(0.3)
    assert rate == pytest.approx(1.0)
    assert remaining == pytest.approx(7.0)

    clock.advance(1.0)
    next_stage = tracker.observe(
        ProgressEvent(stage="refine", completed=0, total=4, unit="point")
    )
    assert tracker.estimates(next_stage) == (0.0, None, None)


def test_tracker_does_not_invent_progress_for_unknown_total() -> None:
    tracker = ProgressTracker(eta_warmup_seconds=0, eta_min_samples=1)
    event = tracker.observe(
        ProgressEvent(stage="discover", completed=3, unit="candidate")
    )
    assert tracker.estimates(event) == (None, None, None)


def test_tracker_excludes_other_stage_time_when_sampling_resumes() -> None:
    clock = FakeClock()
    tracker = ProgressTracker(
        clock=clock,
        eta_warmup_seconds=0.0,
        eta_min_samples=1,
        eta_smoothing=1.0,
    )
    tracker.observe(ProgressEvent(stage="sampling", completed=0, total=30, unit="step"))
    clock.advance(1.0)
    first = tracker.observe(
        ProgressEvent(stage="sampling", completed=10, total=30, unit="step")
    )
    assert tracker.estimates(first)[1] == pytest.approx(10.0)

    clock.advance(20.0)
    tracker.observe(ProgressEvent(kind="status", stage="analysis"))
    clock.advance(1.0)
    resumed = tracker.observe(
        ProgressEvent(stage="sampling", completed=20, total=30, unit="step")
    )

    assert tracker.estimates(resumed)[1] == pytest.approx(10.0)


def test_non_tty_renderer_prints_milestones_without_carriage_returns() -> None:
    stream = io.StringIO()
    renderer = CliProgressRenderer(
        stream=stream,
        is_tty=False,
        milestone_percent=10.0,
    )
    renderer.handle(
        ProgressSnapshot(
            kind="job_started",
            job_name="scan",
            job_index=0,
            total_jobs=1,
            engine="cam",
            scan_summary={"shape": [101, 101], "size": 10201},
        )
    )
    for fraction in (0.01, 0.05, 0.11, 0.15, 0.21):
        renderer.handle(
            ProgressSnapshot(
                kind="job_progress",
                job_name="scan",
                job_index=0,
                total_jobs=1,
                stage="solve_tiles",
                completed=fraction * 100,
                total=100,
                unit="tile",
                fraction=fraction,
                elapsed=1.0,
            )
        )

    output = stream.getvalue()
    assert "scan 101x101 (10201 points)" in output
    assert output.count("solve_tiles") == 3
    assert "\r" not in output
    assert "global" not in output.lower()


def test_non_tty_renderer_shows_only_normal_status_in_brief_mode() -> None:
    stream = io.StringIO()
    renderer = CliProgressRenderer(stream=stream, is_tty=False, verbose=False)
    common = {
        "kind": "job_status",
        "job_name": "simulation",
        "job_index": 0,
        "total_jobs": 1,
        "engine": "sde",
        "stage": "planning",
    }

    renderer.handle(
        ProgressSnapshot(**common, message="hidden detail", importance="detail")
    )
    renderer.handle(
        ProgressSnapshot(**common, message="visible plan", importance="normal")
    )

    assert "hidden detail" not in stream.getvalue()
    assert "visible plan" in stream.getvalue()
