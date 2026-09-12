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
duplicates in ``qphase_sde.analyser.base`` (``AnalyzerExecutionCapabilities``,
``AnalyzerWorkspaceRequest``, ``AnalyzerWorkspaceEstimate`` and
``AnalyzerProtocol``) have been removed. The plugin base class
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
AnalyserProductBuilderProtocol
    Internal product assembly hook used by the built-in analyser base.
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
    "AnalyserProductBuilderProtocol",
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
    """Typed execution surface used by the core ProductGraph executor."""

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


@runtime_checkable
class AnalyserProductBuilderProtocol(Protocol):
    """Temporary hook converting direct-API numerical output into products.

    The production ProductGraph entry is ``AnalyserExecutionContract.execute``
    and does not route through this hook. Built-in analysers retain it only for
    the direct ``analyze()`` compatibility API until the Wave 5D deletion.

    This protocol is not part of the registered analyser execution contract,
    and new third-party plugins must not depend on it.
    """

    def build_products(
        self,
        payload: Any,
        *,
        scan_size: int | None,
        label: str,
    ) -> Mapping[str, Dataset] | None:
        """Build typed products from one ``analyze()`` payload.

        ``scan_size`` is ``None`` when the job had no parameter grid (the
        payload is the bare per-point mapping and products carry no scan
        axis) and the exact scan size of a genuine scan otherwise — a
        1-point scan keeps a scan axis of size 1. Scan payloads are the bare
        per-point mapping for ``scan_size == 1`` and the list of
        ``scan_size`` per-point payloads for larger scans; ``label`` is the
        job-local analyser label used as the default product name.
        Returning ``None`` or an empty mapping is a producer failure: the
        bundle adapter raises and the job ends failed. Invalid payloads
        raise, never silently degrade.
        """
        ...
