"""qphase_sde: Analyser Contract (2.0)
---------------------------------------------------------
Freezes the engine-facing analyser contract. An analyser is an adapter between
the engine and replaceable child plugins: it declares typed input requirements
and output specs, execution capabilities, workspace and work estimates, and
optionally a reducer for batched/streaming execution. The engine consumes only
these declarations — it never branches on analyser names.

The legacy ``analyze(data, backend)`` entry point remains as a convenience
adapter but is no longer the only execution interface the engine knows.

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
AnalyserContract
    The 2.0 analyser protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from qphase.data import ProductDeclaration, ProductRequirement

    from .reducer import ReducerProtocol

__all__ = [
    "AnalyserContract",
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


@dataclass(frozen=True)
class AnalyserWorkspaceEstimate:
    """Peak temporary memory attributed to one analyser invocation."""

    device_bytes: int = 0
    host_bytes: int = 0


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
class AnalyserContract(Protocol):
    """The 2.0 analyser protocol consumed by the SDE engine."""

    def input_requirements(self) -> list[ProductRequirement]:
        """Typed input products this analyser consumes."""
        ...

    def output_spec(self) -> ProductDeclaration:
        """Return the declared output product of this analyser."""
        ...

    def execution_capabilities(self) -> AnalyserExecutionCapabilities:
        """Planner-visible execution capabilities."""
        ...

    def workspace(self, request: AnalyserWorkspaceRequest) -> AnalyserWorkspaceEstimate:
        """Estimate peak workspace for one invocation."""
        ...

    def work_estimate(self, request: AnalyserWorkspaceRequest) -> WorkEstimate:
        """Estimate natural work units for structured progress."""
        ...

    def reducer(self) -> ReducerProtocol | None:
        """Return the reducer for batched/streaming execution, if supported."""
        ...
