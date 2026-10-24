"""QPhase engine for coherent-amplitude matrix analysis.

The engine executes exactly the task the job declares: ``EngineConfig.task`` has
no default, so no job relies on an implicit "a grid means scan" rule any more.
Two tasks run a solver (``solve``/``scan``/``search``/``continue``) and two do
not (``analyze`` postprocesses an existing bundle, ``summarize_search``
assembles evidence from upstream artifacts).

The engine dispatches by capability, not by name: it asks the solver which tasks
it supports, hands it a :class:`~qphase_cam.solver.base.CAMSolveRequest` and
assembles the returned typed products into one
:class:`~qphase.data.GenericDataBundle`. It never builds a result object, never
writes a postprocess dictionary and never inspects a solver's numeric output.

Postprocessors are annotation producers: each one receives the product it declares
it annotates and returns an annotation product, which joins the same bundle.

A ``search`` always publishes evidence: the assembler turns the solver's
:class:`~qphase_cam.evidence.SearchAuditFragment` records into the evidence
product of the same bundle, and the fragment payload is not repeated in the
manifest provenance. Every bundle the engine builds carries the ``artifact_id``
it will be published under, so the evidence can cite its own artifact and the
artifact store persists exactly that id.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.protocols import EngineBase, EngineManifest, ResultProtocol
from qphase.data import Dataset

from qphase_cam.data import cam_bundle
from qphase_cam.evidence_assembler import (
    CANDIDATE_PRODUCT,
    EVIDENCE_PRODUCT,
    EvidenceAssembler,
    artifact_reference,
)
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.solver.base import (
    CAM_SOLVER_TASKS,
    CAMProductFragment,
    CAMSolver,
    CAMSolveRequest,
)

__all__ = ["Engine", "EngineConfig", "SOLVER_TASKS", "validate_solver_task"]

#: Tasks that run a solver; the rest are the engine's own product assembly.
SOLVER_TASKS: tuple[str, ...] = tuple(CAM_SOLVER_TASKS)

#: The six CAM tasks, as a literal for the config schema.
TaskName = Literal["solve", "scan", "search", "continue", "analyze", "summarize_search"]


class EngineConfig(BaseModel):
    """CAM engine configuration.

    ``task`` is required and has no default: each task consumes different
    plugins and produces different products, so "what does this job do" is a
    configuration decision, never an inference from the input shape.
    """

    model_config = ConfigDict(extra="allow")
    task: TaskName
    case_failure_policy: Literal["abort", "record"] = "abort"


def validate_solver_task(
    solver: CAMSolver,
    task: str,
    parameter_grid: Any | None,
    *,
    model: Any | None = None,
) -> None:
    """Check that one solver can run the declared task with the given input.

    The task is explicit, so this only validates the combination: the solver
    must declare the task, the parameter grid must match it, and a single-point
    task must not carry an array-valued model parameter. A grid with ``solve``
    is rejected instead of being read as an implicit scan, ``scan`` requires a
    grid, ``continue`` forbids one, and ``search`` accepts either a single point
    or a grid.

    ``model`` is optional so the grid checks stay usable without one; with a
    ``solve`` task it is what makes "an array parameter is an implicit scan"
    detectable. ``model.params`` holding a non-scalar value would silently solve
    a whole scan from a task that claims one point.
    """
    supported = tuple(solver.supported_tasks)
    unknown = sorted(set(supported) - set(CAM_SOLVER_TASKS))
    if unknown or not supported:
        raise RuntimeError(
            f"CAM solver {solver.name!r} declares unsupported tasks "
            f"{unknown or '(none)'}"
        )
    if task not in supported:
        raise ValueError(
            f"CAM solver {solver.name!r} does not implement the {task!r} task; "
            f"declared tasks: {supported}"
        )
    if parameter_grid is not None and not solver.supports_parameter_grid:
        raise ValueError(
            f"CAM solver {solver.name!r} does not support a parameter grid "
            f"(supports_parameter_grid=False); declared tasks: {supported}"
        )
    if task == "solve" and parameter_grid is not None:
        raise ValueError(
            "the 'solve' task solves one parameter point and does not accept a "
            "grid; declare task: scan instead of relying on an implicit scan"
        )
    if task == "solve" and model is not None:
        array_parameters = _array_valued_parameters(model)
        if array_parameters:
            raise ValueError(
                "the 'solve' task solves one parameter point, but the model "
                f"declares array-valued parameters {array_parameters}; declare "
                "task: scan with a parameter grid instead of relying on an "
                "implicit scan"
            )
    if task == "scan" and parameter_grid is None:
        raise ValueError(
            "the 'scan' task needs a parameter grid; a single point is the 'solve' task"
        )
    if task == "continue" and parameter_grid is not None:
        raise ValueError(
            "continuation cannot be combined with an external parameter grid"
        )


def _array_valued_parameters(model: Any) -> list[str]:
    """Return the model parameters whose value is not a single number."""
    params = getattr(model, "params", None)
    if not isinstance(params, Mapping):
        return []
    return sorted(
        str(name) for name, value in params.items() if np.asarray(value).ndim != 0
    )


class Engine(EngineBase):
    name: ClassVar[str] = "CAM"
    description: ClassVar[str] = "Coherent-amplitude matrix analysis engine"
    config_schema: ClassVar[type[EngineConfig]] = EngineConfig
    manifest: ClassVar[EngineManifest] = EngineManifest(
        required_plugins={"backend", "model", "cam_solver"},
        optional_plugins={"cam_postprocessor"},
        # The six task profiles are the engine's only requirement declaration;
        # the compiler selects one from the explicit ``engine.cam.task`` key.
        task_profiles="qphase_cam.manifest:CAM_TASK_PROFILES",
        profile_selector="qphase_cam.config:select_cam_task_profile",
    )

    def __init__(
        self,
        config: EngineConfig | None = None,
        plugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        del kwargs
        if config is None:
            raise TypeError(
                "the CAM engine requires an explicit task: construct "
                "EngineConfig(task=...) instead of relying on an inferred one"
            )
        self.config = config
        self.plugins = plugins or {}

    def run(
        self,
        data: Any | None = None,
        context: Any | None = None,
    ) -> ResultProtocol:
        task = str(self.config.task)
        reporter = getattr(context, "progress", None) if context is not None else None
        # The artifact id is decided before the products exist, because the
        # evidence product cites the artifact that will carry it.
        artifact_id = uuid.uuid4().hex
        if task in SOLVER_TASKS:
            return self._run_solver(task, data, context, artifact_id, reporter)
        if task == "analyze":
            return self._run_analyze(data, artifact_id, reporter)
        if task == "summarize_search":
            return self._run_summarize(data, artifact_id, reporter)
        raise ValueError(
            f"CAM engine cannot run task {task!r}; the accepted tasks are "
            f"{list(CAM_SOLVER_TASKS) + ['analyze', 'summarize_search']}"
        )

    # -- solver tasks -----------------------------------------------------

    def _run_solver(
        self,
        task: str,
        data: Any | None,
        context: Any | None,
        artifact_id: str,
        reporter: Any,
    ) -> ResultProtocol:
        """Run one solver task and assemble its products."""
        model = self._required("model")
        backend = self._required("backend")
        solver = self._required("cam_solver")
        if not isinstance(solver, CAMSolver):
            raise TypeError(
                "the CAM engine requires a CAMSolver plugin, got "
                f"{type(solver).__name__}"
            )
        grid = context.parameter_grid if context is not None else None
        validate_solver_task(solver, task, grid, model=model)
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
            "task": task,
            "model": getattr(model, "name", type(model).__name__),
            "solver": solver.name,
            "artifact_id": artifact_id,
            **dict(fragment.metadata),
        }
        if task == "search":
            products[EVIDENCE_PRODUCT] = self._search_evidence(
                solver, fragment, products, artifact_id
            )
        annotations = self._annotate(products, metadata, model, backend, reporter)
        products.update(annotations)
        bundle = cam_bundle(
            products,
            provenance={
                "engine": "cam",
                "task": task,
                "artifact_id": artifact_id,
                "model": metadata["model"],
                "solver": solver.name,
                # The audit payload lives in the evidence product; the manifest
                # only records how many case audits the run produced.
                "search_audit_count": len(fragment.audits),
            },
            metadata=metadata,
        )
        if reporter is not None:
            reporter.status("CAM analysis complete", stage="complete")
        return bundle

    def _search_evidence(
        self,
        solver: CAMSolver,
        fragment: CAMProductFragment,
        products: Mapping[str, Dataset],
        artifact_id: str,
    ) -> Dataset:
        """Assemble the mandatory evidence product of one search."""
        if not fragment.audits:
            raise RuntimeError(
                f"CAM solver {solver.name!r} declared the 'search' task but "
                "produced no search audit fragment; a search publishes evidence"
            )
        candidate = products.get(CANDIDATE_PRODUCT)
        return EvidenceAssembler().from_fragments(
            fragment.audits,
            artifact_id=artifact_id,
            candidate_product=candidate if isinstance(candidate, Dataset) else None,
            provenance={"engine": "cam", "task": "search", "solver": solver.name},
        )

    # -- analyze ----------------------------------------------------------

    def _run_analyze(
        self,
        data: Any | None,
        artifact_id: str,
        reporter: Any,
    ) -> ResultProtocol:
        """Postprocess an existing typed bundle without a solver."""
        products, input_artifact = self._input_products(data)
        processors = self.plugins.get("cam_postprocessor")
        if not processors:
            raise RuntimeError(
                "the 'analyze' task needs at least one cam_postprocessor plugin"
            )
        model = self.plugins.get("model")
        backend = self.plugins.get("backend")
        self._check_annotation_capabilities(processors, model, backend)
        metadata: dict[str, Any] = {
            "engine": "cam",
            "task": "analyze",
            "artifact_id": artifact_id,
            "input_products": sorted(products),
            "input_artifact_id": input_artifact,
        }
        annotations = self._annotate(products, metadata, model, backend, reporter)
        if not annotations:
            raise RuntimeError(
                "the 'analyze' task produced no annotation; configure at least "
                "one cam_postprocessor that accepts "
                f"{sorted(products)}"
            )
        provenance = {
            "engine": "cam",
            "task": "analyze",
            "artifact_id": artifact_id,
            "input_artifact_id": input_artifact,
            "annotations": sorted(annotations),
        }
        if reporter is not None:
            reporter.status("CAM analysis complete", stage="complete")
        # Only the annotations are produced: the upstream products stay in the
        # artifact they came from and are never copied into a second artifact.
        return cam_bundle(annotations, provenance=provenance, metadata=metadata)

    # -- summarize_search -------------------------------------------------

    def _run_summarize(
        self,
        data: Any | None,
        artifact_id: str,
        reporter: Any,
    ) -> ResultProtocol:
        """Assemble one evidence product from upstream search artifacts.

        The evidence product is mandatory; ``cam_postprocessor`` plugins stay
        optional annotation providers. Each upstream candidate product is
        annotated on its own — two searches are never merged into one product —
        and each annotation product name carries its source artifact id once more
        than one upstream is summarized.
        """
        inputs = self._input_results(data)
        assembler = EvidenceAssembler(allow_multiple_searches=True)
        fragments: list[Any] = []
        references = []
        sources: list[tuple[str, dict[str, Dataset]]] = []
        for result in inputs:
            reference = artifact_reference(result)
            references.append(reference)
            products = self._products_of(result)
            sources.append((reference.artifact_id, products))
            fragments.extend(assembler.fragments_from_products(products))
        if reporter is not None:
            reporter.status("Assembling search evidence", stage="summarize")
        evidence = assembler.from_fragments(
            fragments,
            artifact_id=artifact_id,
            source_artifacts=references,
            provenance={
                "engine": "cam",
                "task": "summarize_search",
                "summarized_artifact_ids": sorted(
                    reference.artifact_id for reference in references
                ),
            },
        )
        metadata = {
            "engine": "cam",
            "task": "summarize_search",
            "artifact_id": artifact_id,
            "input_artifact_ids": sorted(
                reference.artifact_id for reference in references
            ),
            "case_count": int(evidence.attributes["case_count"]),
        }
        annotations = self._annotate_sources(sources, metadata, reporter)
        if reporter is not None:
            reporter.status("CAM analysis complete", stage="complete")
        return cam_bundle(
            {EVIDENCE_PRODUCT: evidence, **annotations},
            provenance={
                "engine": "cam",
                "task": "summarize_search",
                "artifact_id": artifact_id,
                "summarized_artifact_ids": sorted(
                    reference.artifact_id for reference in references
                ),
                "annotations": sorted(annotations),
                **dict(evidence.provenance),
            },
            metadata=metadata,
        )

    # -- inputs -----------------------------------------------------------

    def _input_products(self, data: Any | None) -> tuple[dict[str, Dataset], str]:
        """Return the products and artifact id of one upstream bundle."""
        inputs = self._input_results(data)
        if len(inputs) != 1:
            raise ValueError(
                "the 'analyze' task annotates one upstream bundle; pass one "
                f"artifact, not {len(inputs)}"
            )
        products = self._products_of(inputs[0])
        reference = artifact_reference(inputs[0])
        return products, reference.artifact_id

    def _input_results(self, data: Any | None) -> list[Any]:
        """Return the upstream results of one job input.

        A typed bundle is one input. A mapped result (the core's multi-view
        input form) is a mapping of label to result, and each of its views is an
        input in its own right.
        """
        if data is None:
            raise ValueError(
                f"the {str(self.config.task)!r} task needs an upstream artifact; "
                "it runs no solver"
            )
        results = getattr(data, "results", None)
        if isinstance(results, Mapping):
            if not results:
                raise ValueError("the mapped input carries no results")
            return list(results.values())
        if isinstance(getattr(data, "products", None), Mapping):
            return [data]
        raise TypeError(
            "the CAM engine takes a typed product bundle or a mapped result; got "
            f"{type(data).__name__}"
        )

    @staticmethod
    def _products_of(result: Any) -> dict[str, Dataset]:
        """Return the typed products of one upstream result."""
        products = getattr(result, "products", None)
        if not isinstance(products, Mapping) or not products:
            raise TypeError(
                "an upstream result must carry a non-empty product mapping; got "
                f"{type(result).__name__}"
            )
        return dict(products)

    # -- annotations ------------------------------------------------------

    def _check_annotation_capabilities(
        self, configured: Any, model: Any, backend: Any
    ) -> None:
        """Reject a postprocessor whose declared needs are not available."""
        processors = (
            configured
            if isinstance(configured, Mapping)
            else {configured.name: configured}
        )
        for name, processor in processors.items():
            if processor.requires_model and model is None:
                raise ValueError(
                    f"postprocessor {name!r} requires a model, but the task did "
                    "not load one"
                )
            if processor.requires_backend and backend is None:
                raise ValueError(
                    f"postprocessor {name!r} requires a backend, but the task did "
                    "not load one"
                )

    def _annotate(
        self,
        products: Mapping[str, Dataset],
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
    def _annotation_source(processor: Any, products: Mapping[str, Dataset]) -> str:
        """Return the product role one postprocessor annotates."""
        role = Engine._matching_role(processor, products)
        if role is None:
            raise ValueError(
                f"postprocessor {processor.name!r} accepts "
                f"{tuple(processor.accepted_product_roles)} but the bundle has "
                f"{sorted(products)}"
            )
        return role

    @staticmethod
    def _matching_role(processor: Any, products: Mapping[str, Dataset]) -> Any | None:
        """Return the first accepted role present in the products, if any."""
        for role in tuple(processor.accepted_product_roles):
            if role in products:
                return role
        return None

    def _annotate_sources(
        self,
        sources: Sequence[tuple[str, Mapping[str, Dataset]]],
        metadata: dict[str, Any],
        reporter: Any,
    ) -> dict[str, Dataset]:
        """Annotate every upstream candidate product of a summary.

        A summary has no model and no backend (its task profile forbids both), so
        a provider that declares it needs one is rejected instead of being run
        against ``None``. Only the products a provider accepts are annotated, and
        a configured provider that matches no upstream is an error: silently
        ignoring a configured provider is exactly what a provider must not do.
        """
        configured = self.plugins.get("cam_postprocessor")
        if not configured:
            return {}
        processors = (
            configured
            if isinstance(configured, Mapping)
            else {configured.name: configured}
        )
        self._check_annotation_capabilities(processors, None, None)
        annotations: dict[str, Dataset] = {}
        matched: dict[str, list[str]] = {str(name): [] for name in processors}
        for source_id, products in sources:
            for name, processor in processors.items():
                role = self._matching_role(processor, products)
                if role is None:
                    continue
                context = CAMAnnotationContext(
                    model=None,
                    backend=None,
                    metadata=metadata,
                    products=products,
                    source_role=role,
                )
                if reporter is not None:
                    reporter.status(
                        f"Annotating {role} of {source_id} with {name}",
                        stage="annotate",
                    )
                product = processor.process(products[role], context)
                if not isinstance(product, Dataset):
                    raise TypeError(
                        f"postprocessor {name!r} returned "
                        f"{type(product).__name__}, expected an annotation Dataset"
                    )
                kind = str(processor.annotation_kind or name)
                base = kind if len(sources) == 1 else f"{kind}__{source_id}"
                key = self._annotation_product_name(base, str(name), annotations)
                annotations[key] = product
                matched[str(name)].append(key)
                if processor.result_metadata:
                    metadata.setdefault("annotation_metadata", {})[key] = dict(
                        processor.result_metadata
                    )
        unmatched = sorted(name for name, keys in matched.items() if not keys)
        if unmatched:
            available = sorted({role for _, products in sources for role in products})
            raise ValueError(
                f"postprocessor(s) {unmatched} matched no upstream product; the "
                f"summarized artifacts carry {available}"
            )
        metadata["annotations"] = sorted(annotations)
        return annotations

    @staticmethod
    def _annotation_product_name(
        base: str, provider: str, annotations: Mapping[str, Dataset]
    ) -> str:
        """Return a stable, collision-free product name for one annotation.

        The base is ``kind`` for a single upstream and ``kind__<artifact id>`` for
        several; two providers that emit the same kind (or two views of the same
        artifact) are separated by the provider name, never by evaluation order
        alone.
        """
        taken = set(annotations) | {EVIDENCE_PRODUCT}
        if base not in taken:
            return base
        scoped = f"{base}__{provider}"
        if scoped not in taken:
            return scoped
        index = 2
        while f"{scoped}__{index}" in taken:
            index += 1
        return f"{scoped}__{index}"

    def _required(self, name: str) -> Any:
        plugin = self.plugins.get(name)
        if plugin is None or isinstance(plugin, dict):
            raise RuntimeError(f"CAM engine requires exactly one {name!r} plugin")
        return plugin
