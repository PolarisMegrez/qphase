from __future__ import annotations

import io
from types import SimpleNamespace

import pytest
from qphase.commands.progress import CliProgressRenderer
from qphase.core.progress import (
    ProgressEvent,
    ProgressReporter,
    ProgressSnapshot,
    ProgressTracker,
)

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
            ProgressEvent(stage="solve", completed=completed, total=10, unit="tile")
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


def test_progress_snapshot_preserves_plugin_activity_metadata() -> None:
    snapshot = ProgressSnapshot(
        kind="job_status",
        job_name="simulation",
        job_index=0,
        total_jobs=1,
        metadata={"active_plugins": ["analyser.psd", "analyser.allan_variance"]},
    )

    assert snapshot.to_dict()["metadata"]["active_plugins"] == [
        "analyser.psd",
        "analyser.allan_variance",
    ]


# --- persistence stage of the ResultRouter save boundary -------------------


class _SavingResult:
    """Result whose only job is to show whether a save reached the boundary."""

    data = None
    metadata: dict = {}
    label = None

    def __init__(self) -> None:
        self.saved: list = []

    def save(self, path) -> None:
        self.saved.append(path)


def _router_context(events: list, saves: list):
    """Return an ExecutionContext-shaped object recording events and saves."""

    class _Artifacts:
        def save_result(self, result, name) -> None:
            saves.append((name, len(events)))

    return SimpleNamespace(
        progress=ProgressReporter(events.append), artifacts=_Artifacts()
    )


def _save_job(**kwargs):
    from qphase.core.config import JobConfig

    return JobConfig(name="save_job", engine={"dummy": {}}, **kwargs)


def test_router_reports_the_artifact_save_as_a_persistence_stage(tmp_path) -> None:
    from qphase.core.result_router import ResultRouter
    from qphase.core.system_config import SystemConfig

    events: list = []
    saves: list = []
    result = _SavingResult()
    ResultRouter(SystemConfig()).route_output(
        _save_job(save=True),
        result,
        {},
        tmp_path,
        context=_router_context(events, saves),
    )

    # The save happens inside the stage, after its start and before the advance.
    assert saves == [("save_job", 1)]
    assert [event.kind for event in events] == [
        "stage_started",
        "progress",
        "stage_completed",
    ]
    started, progress, completed = events
    assert (started.stage, started.total, started.unit) == (
        "persistence",
        1,
        "artifact",
    )
    assert "save_job" in started.message
    assert (progress.stage, progress.completed, progress.total, progress.unit) == (
        "persistence",
        1.0,
        1.0,
        "artifact",
    )
    assert (completed.stage, completed.completed, completed.total) == (
        "persistence",
        1.0,
        1.0,
    )
    assert result.saved == []


def test_router_reports_no_persistence_stage_when_it_does_not_save(tmp_path) -> None:
    from qphase.core.result_router import ResultRouter
    from qphase.core.system_config import SystemConfig

    events: list = []
    saves: list = []
    result = _SavingResult()
    ResultRouter(SystemConfig()).route_output(
        _save_job(save=False),
        result,
        {},
        tmp_path,
        context=_router_context(events, saves),
    )

    assert events == []
    assert saves == []
    assert result.saved == []


def test_router_keeps_the_legacy_save_path_free_of_progress(tmp_path) -> None:
    from qphase.core.result_router import ResultRouter
    from qphase.core.system_config import SystemConfig

    result = _SavingResult()
    ResultRouter(SystemConfig()).route_output(
        _save_job(save=True), result, {}, tmp_path
    )

    assert result.saved == [tmp_path / "save_job"]


def test_router_does_not_count_a_failed_save(tmp_path) -> None:
    from qphase.core.errors import QPhaseIOError
    from qphase.core.result_router import ResultRouter
    from qphase.core.system_config import SystemConfig

    events: list = []

    class _FailingArtifacts:
        def save_result(self, result, name) -> None:
            raise OSError("disk full")

    context = SimpleNamespace(
        progress=ProgressReporter(events.append), artifacts=_FailingArtifacts()
    )
    with pytest.raises(QPhaseIOError):
        ResultRouter(SystemConfig()).route_output(
            _save_job(save=True), _SavingResult(), {}, tmp_path, context=context
        )

    # The stage still closes, but it never claims the artifact was written.
    assert [event.kind for event in events] == ["stage_started", "stage_completed"]
    assert events[-1].completed == 0.0
