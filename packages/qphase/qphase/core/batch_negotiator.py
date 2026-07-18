"""qphase: Job batching negotiator for the scheduler.

The negotiator groups expanded jobs into single jobs or batches by consulting
the resource-pack-provided :class:`~qphase.core.batching.BatchPlanner` for each
engine. It preserves dependency ordering so that downstream jobs never run
before their upstream inputs are available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qphase.core.batching import BatchPlan
from qphase.core.errors import QPhasePluginError
from qphase.core.registry import RegistryCenter

from .config import JobConfig


@dataclass
class SingleJob:
    """Wrapper for a job that must be executed individually."""

    job: JobConfig


@dataclass
class BatchJob:
    """Wrapper for a group of jobs that will be executed as one batch."""

    plan: BatchPlan
    original_jobs: list[JobConfig]


JobGroup = SingleJob | BatchJob


class BatchNegotiator:
    """Groups expanded jobs into single or batched execution units.

    The negotiator is conservative: it only batches contiguous blocks of jobs
    that share the same engine and for which that engine's ``BatchPlanner``
    reports ``can_batch(...) == True``. A block is also interrupted by any job
    that is referenced as ``input`` by a later job that has not yet been grouped,
    ensuring downstream aggregation jobs see all upstream results.

    Parameters
    ----------
    registry : RegistryCenter
        Registry used to discover ``BatchPlanner`` instances.

    """

    def __init__(self, registry: RegistryCenter) -> None:
        self._registry = registry
        # Cache planners by engine name to avoid repeated registry lookups.
        self._planner_cache: dict[str, Any] = {}

    def _engine_name(self, job: JobConfig) -> str:
        """Return the engine name declared by a job config."""
        engine_cfg = job.engine
        if isinstance(engine_cfg, dict):
            # Engine config is a single-key dict like {"sde": {...}}
            names = [k for k in engine_cfg.keys() if k not in ("name", "params")]
            if names:
                return names[0].lower()
        # Fallback: try attribute access
        if hasattr(engine_cfg, "name"):
            return str(engine_cfg.name).lower()
        return ""

    def _get_planner(self, engine_name: str) -> Any | None:
        """Return the BatchPlanner class/instance for an engine, or None."""
        if engine_name in self._planner_cache:
            return self._planner_cache[engine_name]

        try:
            planner = self._registry.get_batch_planner(engine_name)
        except QPhasePluginError:
            planner = None

        self._planner_cache[engine_name] = planner
        return planner

    def _downstream_names(self, jobs: list[JobConfig]) -> set[str]:
        """Collect all job names referenced as input by later jobs."""
        referenced: set[str] = set()
        for job in jobs:
            if job.input:
                referenced.add(job.input)
        return referenced

    def group_jobs(self, expanded_jobs: list[JobConfig]) -> list[JobGroup]:
        """Partition *expanded_jobs* into single/batched execution groups.

        The algorithm walks the job list left-to-right and greedily forms the
        largest possible batch for each engine, stopping before any job that is
        referenced by a downstream ``input`` field.

        Returns
        -------
        list[JobGroup]
            Ordered list of ``SingleJob`` or ``BatchJob`` groups.

        """
        if not expanded_jobs:
            return []

        downstream_inputs = self._downstream_names(expanded_jobs)
        groups: list[JobGroup] = []
        i = 0
        n = len(expanded_jobs)

        while i < n:
            job = expanded_jobs[i]

            # A job referenced downstream must remain individually addressable.
            if job.name in downstream_inputs:
                groups.append(SingleJob(job))
                i += 1
                continue

            engine_name = self._engine_name(job)
            planner = self._get_planner(engine_name) if engine_name else None

            if planner is None:
                groups.append(SingleJob(job))
                i += 1
                continue

            # Try to extend the batch with subsequent compatible jobs.
            candidate_jobs = [job]
            j = i + 1
            while j < n:
                next_job = expanded_jobs[j]
                if next_job.name in downstream_inputs:
                    break
                if self._engine_name(next_job) != engine_name:
                    break
                candidate_jobs.append(next_job)
                j += 1

            # Trim the candidate list to the largest prefix the planner accepts.
            batch_jobs = self._trim_batch(planner, candidate_jobs)

            if len(batch_jobs) > 1:
                plan = planner.plan_batch(batch_jobs)
                groups.append(BatchJob(plan=plan, original_jobs=batch_jobs))
                i += len(batch_jobs)
            else:
                groups.append(SingleJob(job))
                i += 1

        return groups

    def _trim_batch(
        self, planner: Any, candidate_jobs: list[JobConfig]
    ) -> list[JobConfig]:
        """Return the largest prefix of *candidate_jobs* accepted by *planner*.

        The planner is asked repeatedly with shrinking prefixes until it accepts
        one. If even a single job is not accepted, that job is returned alone
        (the caller will wrap it as ``SingleJob``).
        """
        for size in range(len(candidate_jobs), 0, -1):
            prefix = candidate_jobs[:size]
            try:
                if planner.can_batch(prefix):
                    return prefix
            except Exception:
                # Treat planner exceptions as "cannot batch".
                pass
        return [candidate_jobs[0]]
