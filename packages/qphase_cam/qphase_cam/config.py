"""qphase_cam: Task Selection Planning Contract
---------------------------------------------------------
The CAM v2 task vocabulary and the selector that maps a job to one profile id.

Six tasks replace the current implicit behaviour: ``solve`` (one steady state),
``scan`` (multistability across a scan axis), ``search`` (bifurcation search),
``continue`` (continuation branches), ``analyze`` (postprocess existing typed
products) and ``summarize_search`` (summarize candidates/evidence).

This module is **planning only**. The task is selected from the explicit
``engine.cam.task`` key: there is *no default*, because the v2 contract
requires every CAM job to state what it does. The current runtime does not read
this key yet — Phase 10E activates it together with the typed runtime — so a
job may still carry ``engine.cam`` settings the running engine ignores, and the
selector must never be used to validate a live job before that handover.

Public API
----------
CAM_TASKS
    The six task ids, in declaration order.
select_cam_task_profile
    Map a job-explicit config view to one task id.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["CAM_TASKS", "select_cam_task_profile"]

#: The CAM task ids, frozen for Phase 10.
CAM_TASKS: tuple[str, ...] = (
    "solve",
    "scan",
    "search",
    "continue",
    "analyze",
    "summarize_search",
)


def select_cam_task_profile(job_explicit_config: dict[str, Any]) -> str:
    """Return the CAM task id declared by one job-explicit config view.

    ``job_explicit_config`` is the compiler's job-explicit view: the engine
    section lives under ``engine.cam``. The task must be stated explicitly; a
    missing, non-mapping or unknown value is an error rather than a silent
    fallback, because each task consumes different plugins and produces
    different products.

    Raises
    ------
    ValueError
        If ``engine.cam.task`` is missing, not a string, or not one of
        :data:`CAM_TASKS`.

    """
    engines = job_explicit_config.get("engine")
    engine_config = engines.get("cam") if isinstance(engines, Mapping) else None
    if not isinstance(engine_config, Mapping):
        raise ValueError(
            "CAM jobs must declare an explicit task: expected an "
            "'engine.cam' mapping with a 'task' key"
        )
    task = engine_config.get("task")
    if task is None:
        raise ValueError(
            "'engine.cam.task' is required; CAM jobs declare one of "
            f"{list(CAM_TASKS)}"
        )
    if not isinstance(task, str):
        raise ValueError(
            f"'engine.cam.task' must be a string, got {type(task).__name__}"
        )
    if task not in CAM_TASKS:
        raise ValueError(
            f"unknown CAM task {task!r}; expected one of {list(CAM_TASKS)}"
        )
    return task
