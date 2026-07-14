"""qphase: Batching protocol for resource-pack-driven parallel execution.

This module defines the contracts through which a resource pack (e.g.
``qphase_sde``) can tell the generic scheduler that a group of expanded jobs can
be executed as a single batch, and how the combined result should be split back
into per-job results.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from qphase.core.protocols import ResultProtocol

from .config import JobConfig


class BatchPlanner(Protocol):
    """Implemented by engines that can execute several expanded jobs together.

    A ``BatchPlanner`` is discovered by the scheduler through the engine plugin's
    metadata. It inspects a list of expanded job configurations and, if they are
    compatible, returns a ``BatchPlan`` describing how to merge them.
    """

    @classmethod
    def can_batch(cls, jobs: list[JobConfig]) -> bool:
        """Return ``True`` if this planner can handle *jobs* as one batch."""
        ...

    def plan_batch(self, jobs: list[JobConfig]) -> BatchPlan:
        """Return a concrete plan for *jobs*.

        Must only be called when :meth:`can_batch` returned ``True``.
        """
        ...

    def run_batch(
        self,
        batch_job: JobConfig,
        original_jobs: list[JobConfig],
        input_result: ResultProtocol | None,
        **run_kwargs: Any,
    ) -> ResultProtocol:
        """Execute the merged ``batch_job`` and return one combined result.

        Parameters
        ----------
        batch_job : JobConfig
            The merged job configuration produced by :meth:`plan_batch`.
        original_jobs : list[JobConfig]
            The original expanded jobs, in the order they were batched.
        input_result : ResultProtocol | None
            Resolved input from an upstream job, if any.
        **run_kwargs
            Extra arguments the scheduler may pass (e.g. ``progress_cb``).
        """
        ...


@dataclass
class BatchPlan:
    """Description of a batch produced by a :class:`BatchPlanner`."""

    # The merged job config that the engine will actually run.
    batch_job: JobConfig

    # Names of the original expanded jobs, in the order they are batched.
    original_names: list[str]

    # Name of the registered ResultSplitter that can split the merged result.
    result_splitter: str


class ResultSplitter(Protocol):
    """Splits a batched result back into per-job results."""

    def split(
        self,
        batched_result: ResultProtocol,
        original_jobs: list[JobConfig],
    ) -> dict[str, ResultProtocol]:
        """Return a mapping ``original_job_name -> result``.

        The returned results must be valid inputs for downstream jobs and must
        carry the metadata of the corresponding original job.
        """
        ...
