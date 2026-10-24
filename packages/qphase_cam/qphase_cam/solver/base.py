"""Public base class for CAM solver plugins.

Phase 10D2 introduces one solver contract: a solver declares what it can do and
answers a :class:`CAMSolveRequest` with a :class:`CAMProductFragment` of typed
products. Nothing is dispatched by ``solver.name`` or ``output_kind`` any more;
the engine reads the capability declarations below.

The declarations are static: they are the vocabulary the engine and the task
profiles consume, and the tests freeze them against the actual behaviour of each
solver so a solver cannot claim a capability it does not implement.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar, Generic, TypeVar

from pydantic import ConfigDict
from qphase.core.protocols import PluginConfigBase
from qphase.core.scan import ParameterGrid
from qphase.data import ProductDeclaration, ProductRequirement

from qphase_cam.data import (
    CAM_CANDIDATE_DECLARATION,
    CAM_CONTINUATION_DECLARATION,
    CAM_STATE_DECLARATION,
    CAM_STATE_SCAN_DECLARATION,
)

__all__ = [
    "CAM_SOLVER_TASKS",
    "CAMProductFragment",
    "CAMSolver",
    "CAMSolverConfig",
    "CAMSolveRequest",
    "SOLVER_OUTPUT_DECLARATIONS",
]


class CAMSolverConfig(PluginConfigBase):
    """Strict base schema for CAM solver configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


SolverConfigT = TypeVar("SolverConfigT", bound=CAMSolverConfig)

#: Tasks a CAM solver may declare. The engine picks one from the capability list.
CAM_SOLVER_TASKS: tuple[str, ...] = ("solve", "scan", "search", "continue")

#: Product declarations a solver may name in ``output_products``.
SOLVER_OUTPUT_DECLARATIONS: dict[str, ProductDeclaration] = {
    "state": CAM_STATE_DECLARATION,
    "state_scan": CAM_STATE_SCAN_DECLARATION,
    "candidates": CAM_CANDIDATE_DECLARATION,
    "continuation": CAM_CONTINUATION_DECLARATION,
}


@dataclass(frozen=True)
class CAMSolveRequest:
    """Everything one solver invocation needs.

    ``parameter_grid`` is the compiled ``ScanSpec`` grid (``None`` for a single
    point). Solvers that cannot consume a grid declare ``supports_parameter_grid
    = False`` and the engine rejects the combination instead of silently
    dropping the scan.
    """

    task: str
    model: Any
    backend: Any
    parameter_grid: ParameterGrid | None = None
    data: Any | None = None
    context: Any | None = None
    case_failure_policy: str = "abort"

    @property
    def progress(self) -> Any | None:
        """Return the progress reporter of the execution context, if any."""
        return getattr(self.context, "progress", None)


@dataclass
class CAMProductFragment:
    """Typed products produced by one solver call.

    ``products`` is keyed by product role (``state``/``state_scan``/``candidates``
    /``continuation``). ``audits`` carries one :class:`SearchAuditFragment` per
    search case; the evidence product that consumes them belongs to Phase 10E, so
    the engine only carries them forward. ``metadata`` is execution metadata, not
    a payload: nothing scientific rides there.
    """

    products: dict[str, Any] = field(default_factory=dict)
    audits: tuple[Any, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


class CAMSolver(ABC, Generic[SolverConfigT]):
    """Base class for solver plugins selected by the CAM engine."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[SolverConfigT]]

    #: Tasks this solver implements, taken from :data:`CAM_SOLVER_TASKS`.
    supported_tasks: ClassVar[tuple[str, ...]] = ("solve",)
    #: Typed products this solver consumes (empty for a from-scratch solve).
    input_product_requirements: ClassVar[tuple[ProductRequirement, ...]] = ()
    #: Output declarations, one per product this solver can emit.
    output_products: ClassVar[tuple[ProductDeclaration, ...]] = (
        SOLVER_OUTPUT_DECLARATIONS["state"],
    )
    #: Whether the solver consumes a compiled ``ScanSpec`` grid itself.
    supports_parameter_grid: ClassVar[bool] = False
    #: Whether the solver batches several parameter points in one call.
    supports_batch: ClassVar[bool] = False
    #: Whether the solver needs a model Jacobian.
    requires_jacobian: ClassVar[bool] = False
    #: Whether the solver needs symbolically derived dynamics (qpsgen).
    requires_symbolic_dynamics: ClassVar[bool] = False
    #: Whether the solver emits a :class:`SearchAuditFragment` per case.
    search_audit_capability: ClassVar[bool] = False
    #: Progress unit reported while the solver runs.
    progress_unit: ClassVar[str] = "solve"

    config: SolverConfigT

    def __init__(self, config: SolverConfigT | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword options, not both")
        source = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)

    @abstractmethod
    def solve(self, request: CAMSolveRequest) -> CAMProductFragment:
        """Solve one request and return its typed products.

        This is the only solver entry point: no ``solve(model, backend)`` overload
        and no result-object output survive Phase 10D2.
        """
