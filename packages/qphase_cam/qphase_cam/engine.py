"""QPhase engine for coherent-amplitude matrix analysis.

The engine dispatches by capability, not by name: it asks the solver which tasks
it supports, hands it a :class:`~qphase_cam.solver.base.CAMSolveRequest` and
assembles the returned typed products into one
:class:`~qphase.data.GenericDataBundle`. It never builds a result object, never
writes a postprocess dictionary and never inspects a solver's numeric output.

Postprocessors are annotation producers: each one receives the product it declares
it annotates and returns an annotation product, which joins the same bundle.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.data import Dataset

from qphase_cam.data import cam_bundle
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.solver.base import (
    CAM_SOLVER_TASKS,
    CAMProductFragment,
    CAMSolver,
    CAMSolveRequest,
)


class EngineConfig(BaseModel):
    """CAM engine configuration; task details belong to solver plugins."""

    model_config = ConfigDict(extra="allow")
    case_failure_policy: Literal["abort", "record"] = "abort"


def select_solver_task(solver: CAMSolver, parameter_grid: Any | None) -> str:
    """Return the task one solver invocation performs.

    ``search`` and ``continue`` win over a grid, because a search or a
    continuation consumes the grid itself; otherwise a grid means ``scan`` and no
    grid means ``solve``. A combination the solver cannot perform raises instead of
    silently dropping the grid.
    """
    supported = tuple(solver.supported_tasks)
    unknown = sorted(set(supported) - set(CAM_SOLVER_TASKS))
    if unknown or not supported:
        raise RuntimeError(
            f"CAM solver {solver.name!r} declares unsupported tasks "
            f"{unknown or '(none)'}"
        )
    if parameter_grid is not None and not solver.supports_parameter_grid:
        # Every grid path checks the capability: a solver that cannot consume a
        # ScanSpec must be rejected here, not half-run with the grid dropped.
        raise ValueError(
            f"CAM solver {solver.name!r} does not support a parameter grid "
            f"(supports_parameter_grid=False); declared tasks: {supported}"
        )
    if "search" in supported:
        return "search"
    if "continue" in supported:
        if parameter_grid is not None:
            raise ValueError(
                "continuation cannot be combined with an external ScanSpec"
            )
        return "continue"
    if parameter_grid is not None:
        return "scan"
    if "solve" in supported:
        return "solve"
    if "scan" in supported:
        return "scan"
    raise ValueError(
        f"CAM solver {solver.name!r} supports neither 'solve' nor 'scan' without a "
        f"grid; declared tasks: {supported}"
    )


class Engine(EngineBase):
    name: ClassVar[str] = "CAM"
    description: ClassVar[str] = "Coherent-amplitude matrix analysis engine"
    config_schema: ClassVar[type[EngineConfig]] = EngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"backend", "model", "cam_solver"},
        optional_plugins={"cam_postprocessor"},
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        self.config = config or EngineConfig()
        self.plugins = plugins or {}

    def run(
        self,
        data: Any | None = None,
        context: Any | None = None,
    ) -> ResultProtocol:
        model = self._required("model")
        backend = self._required("backend")
        solver = self._required("cam_solver")
        if not isinstance(solver, CAMSolver):
            raise TypeError(
                "the CAM engine requires a CAMSolver plugin, got "
                f"{type(solver).__name__}"
            )
        grid = context.parameter_grid if context is not None else None
        task = select_solver_task(solver, grid)
        reporter = getattr(context, "progress", None) if context is not None else None
        if reporter is not None:
            reporter.status(f"Solving CAM {task}", stage="solve")
        fragment = solver.solve(
            CAMSolveRequest(
                task=task,
                model=model,
                backend=backend,
                parameter_grid=grid,
                data=data,
                context=context,
                case_failure_policy=self.config.case_failure_policy,
            )
        )
        if not isinstance(fragment, CAMProductFragment):
            raise TypeError(
                f"CAM solver {solver.name!r} returned "
                f"{type(fragment).__name__}, expected a CAMProductFragment"
            )
        products: dict[str, Dataset] = dict(fragment.products)
        if not products:
            raise RuntimeError(f"CAM solver {solver.name!r} returned no products")
        metadata = {
            "engine": "cam",
            "model": getattr(model, "name", type(model).__name__),
            "solver": solver.name,
            "task": task,
            **dict(fragment.metadata),
        }
        annotations = self._annotate(products, metadata, model, backend, reporter)
        products.update(annotations)
        bundle = cam_bundle(
            products,
            provenance={
                "engine": "cam",
                "task": task,
                "model": metadata["model"],
                "solver": solver.name,
                "search_audits": len(fragment.audits),
                "audits": [audit.model_dump(mode="json") for audit in fragment.audits],
            },
            metadata=metadata,
        )
        if reporter is not None:
            reporter.status("CAM analysis complete", stage="complete")
        return bundle

    def _annotate(
        self,
        products: dict[str, Dataset],
        metadata: dict[str, Any],
        model: Any,
        backend: Any,
        reporter: Any,
    ) -> dict[str, Dataset]:
        """Run every configured postprocessor and collect its annotation product."""
        configured = self.plugins.get("cam_postprocessor")
        if not configured:
            return {}
        processors = (
            configured
            if isinstance(configured, dict)
            else {configured.name: configured}
        )
        annotations: dict[str, Dataset] = {}
        for name, processor in processors.items():
            role = self._annotation_source(processor, products)
            context = CAMAnnotationContext(
                model=model,
                backend=backend,
                metadata=metadata,
                products=products,
                source_role=role,
            )
            if reporter is not None:
                reporter.status(f"Annotating {role} with {name}", stage="annotate")
            product = processor.process(products[role], context)
            if not isinstance(product, Dataset):
                raise TypeError(
                    f"postprocessor {name!r} returned {type(product).__name__}, "
                    "expected an annotation Dataset"
                )
            kind = str(processor.annotation_kind or name)
            if kind in annotations or kind in products:
                raise ValueError(
                    f"postprocessor {name!r} produced duplicate product {kind!r}"
                )
            annotations[kind] = product
            metadata.setdefault("annotations", []).append(kind)
            if processor.result_metadata:
                metadata.setdefault("annotation_metadata", {})[kind] = dict(
                    processor.result_metadata
                )
        return annotations

    @staticmethod
    def _annotation_source(processor: Any, products: dict[str, Dataset]) -> str:
        """Return the product role one postprocessor annotates."""
        accepted = tuple(processor.accepted_product_roles)
        for role in accepted:
            if role in products:
                return role
        raise ValueError(
            f"postprocessor {processor.name!r} accepts {accepted} but the bundle "
            f"has {sorted(products)}"
        )

    def _required(self, name: str) -> Any:
        plugin = self.plugins.get(name)
        if plugin is None or isinstance(plugin, dict):
            raise RuntimeError(f"CAM engine requires exactly one {name!r} plugin")
        return plugin


__all__ = ["Engine", "EngineConfig", "select_solver_task"]
