"""qphase_sde: Analyser Declaration Contract (2.0)
---------------------------------------------------------
The planner/bundle-facing analyser declaration contract. An analyser is an
adapter between the engine and replaceable child plugins: it declares typed
input requirements and (multi-)output specs, execution capabilities,
workspace and work estimates, and optionally a reducer for batched/streaming
execution. Planners and the bundle adapter consume only these declarations —
they never branch on analyser names.

This protocol freezes the declaration surface only — what an analyser
advertises, not how it runs. ``AnalyserExecutionContract.execute`` is the
runtime boundary used by ``ProductGraphExecutor``; numerical implementation
details remain inside the resource package.

This module is the sole source of truth for those types: the legacy 1.x
duplicates in ``qphase_sde.analyser.base`` have been removed. The plugin base class
(:class:`qphase_sde.analyser.base.Analyzer`) consumes these contract types
directly. Built-in numerical kernels remain private implementation details
behind the typed execution boundary.

Public API
----------
AnalyserExecutionCapabilities
    Planner-visible execution capabilities.
AnalyserWorkspaceRequest
    Shape/dtype facts for workspace estimation.
AnalyserWorkspaceEstimate
    Estimated peak workspace of one invocation.
WorkEstimate
    Natural work-unit estimate for structured progress.
AnalyserDeclarationContract
    The 2.0 analyser declaration protocol.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qphase.data import Dataset, ProductDeclaration, ProductRequirement

    from .reducer import ReducerProtocol

__all__ = [
    "AnalyserDeclarationContract",
    "AnalyserExecutionContract",
    "AnalyserExecutionCapabilities",
    "AnalyserWorkspaceEstimate",
    "AnalyserWorkspaceRequest",
    "WorkEstimate",
]

#: Natural work units understood by the progress model. Stages never mix
#: heterogeneous units into one ETA.
WorkUnit = Literal[
    "trajectory-step",
    "sample",
    "segment",
    "transform",
    "byte",
    "trajectory-tau",
    "tau",
    "candidate",
    "spectrum",
    "chunk",
]


@dataclass(frozen=True)
class AnalyserExecutionCapabilities:
    """Planner-visible execution capabilities of an analyser."""

    preferred_location: Literal["host", "backend", "either"] = "either"
    requires_materialized_time_axis: bool = False
    supports_trajectory_batching: bool = False
    supports_time_streaming: bool = False
    supports_scan_tiling: bool = False
    supports_masked_or_ragged: bool = False
    deterministic_merge: bool = True


@dataclass(frozen=True)
class AnalyserWorkspaceRequest:
    """Shape and dtype facts supplied to analyser workspace estimators."""

    trajectory_bytes: int
    n_traj: int
    saved_samples: int
    n_record_modes: int
    real_itemsize: int
    backend_name: str
    total_n_traj: int | None = None


@dataclass(frozen=True)
class AnalyserWorkspaceEstimate:
    """Per-point invocation/output and cross-batch retained memory."""

    device_bytes: int = 0
    host_bytes: int = 0
    retained_device_bytes: int = 0
    retained_host_bytes: int = 0


@dataclass(frozen=True)
class WorkEstimate:
    """Natural work-unit estimate for one analyser invocation.

    ``total=None`` means the stage reports heartbeats and completed chunks
    only — it must not fabricate an ETA.
    """

    unit: WorkUnit
    total: int | None = None
    stage_prefix: str = "analyze"
    sub_stages: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class AnalyserDeclarationContract(Protocol):
    """The 2.0 analyser declaration protocol.

    Planner- and bundle-facing only: implementations declare typed
    inputs/outputs, execution capabilities, workspace and work estimates,
    and an optional reducer. Runtime execution is specified by the separate
    ``AnalyserExecutionContract``.
    """

    def input_requirements(self) -> list[ProductRequirement]:
        """Typed input products this analyser consumes."""
        ...

    def output_specs(self, *, label: str) -> tuple[ProductDeclaration, ...]:
        """Typed products emitted by this analyser."""
        ...

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        """Planner-visible execution capabilities."""
        ...

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate temporary workspace for one invocation."""
        ...

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Estimate natural work units for structured progress."""
        ...

    def reducer(self) -> ReducerProtocol | None:
        """Return the optional batch/stream reducer."""
        ...


@runtime_checkable
class AnalyserExecutionContract(AnalyserDeclarationContract, Protocol):
    """Typed execution surface used by the core ProductGraph executor.

    ``context`` is the scheduler-supplied execution context (``None`` for
    direct calls). Long-running ``execute()`` implementations MUST poll
    ``context.cancellation.raise_if_cancelled()`` inside their internal loops
    — guarded as ``getattr(context, "cancellation", None)`` so direct calls
    and attribute-poor contexts keep working — at their natural work-unit
    cadence (per scan point, per fit, per lag), and SHOULD report progress
    through ``context.progress`` at the same cadence using the unit and stage
    prefix declared by :meth:`AnalyserDeclarationContract.work_estimate`.
    When the work total is unknown the stage reports heartbeats and completed
    counts only; implementations must not fabricate a total. Behaviour must be
    identical when ``context`` is ``None``.
    """

    def execute(
        self,
        inputs: Mapping[str, Dataset],
        *,
        backend: Any,
        label: str,
        metadata: Mapping[str, Any] | None = None,
        context: Any | None = None,
    ) -> Mapping[str, Dataset]:
        """Consume named typed products and return declared typed products."""
        ...
