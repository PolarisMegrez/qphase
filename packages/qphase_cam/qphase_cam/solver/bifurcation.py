"""High-order CAM equilibrium bifurcation solver plugin."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Sequence
from itertools import product
from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.core.protocols import PluginManifest, SubpluginSlot
from scipy.optimize import least_squares, root

from qphase_cam.core.bordered import BorderedMultiplicitySystem
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.core.reduction import (
    CondensedScalarReduction,
    FractionFreeScalarReduction,
    SeedGenerationStats,
    finite_vector,
    reduction_path,
    scaled_distance,
)
from qphase_cam.errors import BifurcationCapabilityError
from qphase_cam.state import (
    CAMBifurcationCandidate,
    CAMBifurcationOutput,
)

from ..evidence import fragment_from_bifurcation_metadata
from .base import (
    SOLVER_OUTPUT_DECLARATIONS,
    CAMProductFragment,
    CAMSolver,
    CAMSolverConfig,
    CAMSolveRequest,
)
from .bifurcation_audit import (
    AUDIT_CONSERVATION,
    AUDIT_FIELD_UNITS,
    AUDIT_SCHEMA,
    EMPTY_RESULT_NOTE,
    REJECTION_COUNTER_SEMANTICS,
    AuditConfig,
    NearMissStore,
    near_miss_json,
    near_miss_selection_metadata,
)
from .bifurcation_classifier import BifurcationClassifier
from .bifurcation_discovery import BifurcationDiscovery
from .bifurcation_strategy import BifurcationStrategy
from .bifurcation_target import BifurcationTarget
from .execution import (
    build_candidate_from_cases,
    candidate_case_arrays,
    grid_model_parameters,
    model_parameters,
    run_cases,
)
from .identity import (
    candidate_content as derive_candidate_content,
)
from .identity import (
    candidate_identity as derive_candidate_identity,
)
from .identity import (
    case_identity as derive_case_identity,
)
from .identity import (
    search_identity as derive_search_identity,
)
from .identity import (
    search_scope as derive_search_scope,
)
from .identity import (
    start_identity as derive_start_identity,
)

#: Audit count fields sharing the ``candidate start`` unit.  These are the
#: only count fields aggregated into ``audit["totals"]``; generation-trial
#: and workload counts keep per-path units and are merged per key only (see
#: ``AUDIT_CONSERVATION``).
_CANDIDATE_COUNT_FIELDS = (
    "generated_candidate_count",
    "prefilter_pass_count",
    "prefilter_rejected_count",
    "refinement_start_count",
    "refinement_duplicate_count",
    "accepted_count",
    "rejected_count",
)


def _workload_fields(stats: SeedGenerationStats) -> dict[str, int]:
    """Audit workload subfields for one seed pass; one unit per key."""
    return {
        f"{unit}_count": int(count) for unit, count in sorted(stats.workload.items())
    }


def _candidate_chart_id(candidate: Any) -> str:
    """Return the qpsgen chart id of one linear reduction candidate.

    The chart id names the equation/state partition (``ret:<ids>|eq:<equations>``)
    the reduction was built from, which is what makes two reductions of the same
    retained state two different generation paths. It is carried explicitly into the
    reduction, so no later step has to reach back into the qpsgen candidate and no
    slot index is ever used as a path.
    """
    chart = str(getattr(candidate, "chart_id", "") or "")
    if not chart:
        raise BifurcationCapabilityError(
            "a linear reduction candidate carries no chart id, so its generation "
            "path cannot be named"
        )
    return chart


class ControlRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float
    max: float
    scale: float | None = Field(None, gt=0.0)
    domain: Literal["auto", "real", "nonnegative"] = "auto"
    sampling: Literal["linear", "log", "log_abs"] = "linear"

    @model_validator(mode="after")
    def validate_bounds(self) -> ControlRange:
        if self.max <= self.min:
            raise ValueError("control max must be greater than min")
        if self.sampling == "log" and self.min <= 0.0:
            raise ValueError("log-sampled control min must be positive")
        if self.sampling == "log_abs" and (
            self.min == 0.0
            or self.max == 0.0
            or np.signbit(self.min) != np.signbit(self.max)
        ):
            raise ValueError(
                "log_abs-sampled control bounds must be nonzero and have the same sign"
            )
        return self

    def sample_values(self, count: int) -> np.ndarray:
        if self.sampling == "log":
            return np.geomspace(self.min, self.max, count)
        if self.sampling == "log_abs":
            sign = -1.0 if self.max < 0.0 else 1.0
            magnitudes = np.geomspace(abs(self.min), abs(self.max), count)
            return sign * magnitudes
        return np.linspace(self.min, self.max, count)


class RefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tolerance: float = Field(1e-11, gt=0.0)
    max_iterations: int = Field(100, ge=1)


class VerificationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    initial_digits: int = Field(50, ge=20)
    max_digits: int = Field(200, ge=20)

    @model_validator(mode="after")
    def validate_digits(self) -> VerificationConfig:
        if self.max_digits < self.initial_digits:
            raise ValueError("max_digits must be at least initial_digits")
        return self


class StateDomainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["auto", "hermitian_psd"] = "auto"
    enforcement: Literal["filter"] = "filter"
    psd_tolerance: float = Field(1e-10, ge=0.0)


class PerturbationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    parameter: str
    scale: float = Field(1.0, gt=0.0)
    side: Literal["negative", "positive", "both"] = "both"


class BifurcationSolverConfig(CAMSolverConfig):
    controls: dict[str, ControlRange]
    perturbation: PerturbationConfig
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    state_domain: StateDomainConfig = Field(default_factory=StateDomainConfig)
    audit: AuditConfig = Field(default_factory=AuditConfig)


class BifurcationSolver(CAMSolver[BifurcationSolverConfig]):
    """Search model parameters for high-order CAM equilibrium roots."""

    name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "High-order CAM equilibrium bifurcation search"
    config_schema: ClassVar[type[BifurcationSolverConfig]] = BifurcationSolverConfig
    supported_tasks: ClassVar[tuple[str, ...]] = ("search",)
    output_products: ClassVar[tuple[Any, ...]] = (
        SOLVER_OUTPUT_DECLARATIONS["candidates"],
    )
    requires_jacobian: ClassVar[bool] = True
    requires_symbolic_dynamics: ClassVar[bool] = True
    search_audit_capability: ClassVar[bool] = True
    supports_parameter_grid: ClassVar[bool] = True
    progress_unit: ClassVar[str] = "candidate"
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "target": SubpluginSlot(
                namespace="bifurcation_target",
                protocol=("qphase_cam.solver.bifurcation_target:BifurcationTarget"),
                allowed=frozenset({"equilibrium_multiplicity"}),
            ),
            "strategy": SubpluginSlot(
                namespace="bifurcation_strategy",
                default="auto",
                protocol=("qphase_cam.solver.bifurcation_strategy:BifurcationStrategy"),
                allowed=frozenset({"auto", "reduced", "full"}),
            ),
            "discovery": SubpluginSlot(
                namespace="bifurcation_discovery",
                default="seeds",
                protocol=(
                    "qphase_cam.solver.bifurcation_discovery:BifurcationDiscovery"
                ),
                allowed=frozenset({"seeds"}),
            ),
            "classifier": SubpluginSlot(
                namespace="bifurcation_classifier",
                default="scaling_signature",
                protocol=(
                    "qphase_cam.solver.bifurcation_classifier:BifurcationClassifier"
                ),
                allowed=frozenset({"scaling_signature"}),
            ),
        }
    )

    def __init__(
        self,
        config: BifurcationSolverConfig | None = None,
        *,
        subplugins: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, **kwargs)
        if subplugins is None:
            raise TypeError("bifurcation solver requires resolved subplugins")
        self.target: BifurcationTarget = subplugins["target"]
        self.strategy: BifurcationStrategy = subplugins["strategy"]
        self.discovery: BifurcationDiscovery = subplugins["discovery"]
        self.classifier: BifurcationClassifier = subplugins["classifier"]
        self._reduction_cache: dict[
            tuple[Any, ...],
            tuple[
                tuple[FractionFreeScalarReduction | CondensedScalarReduction, ...],
                dict[str, Any],
            ],
        ] = {}
        if len(self.config.controls) != self.target.order - 1:
            raise ValueError(
                "equilibrium multiplicity order m requires exactly m-1 controls"
            )

    def solve(self, request: CAMSolveRequest) -> CAMProductFragment:
        """Search one case or a grid of cases; return typed candidates + audits."""
        model = request.model
        backend = request.backend
        if request.parameter_grid is not None:
            return self._grid_fragment(request)
        output = self.search(model, backend, data=request.data, context=request.context)
        return self._fragment(
            model,
            backend,
            [self._case_payload(model, output, flat_index=0, index=(), values={})],
            case_axes={},
            case_params={},
            case_shape=(),
            scan_metadata={},
        )

    def _grid_fragment(self, request: CAMSolveRequest) -> CAMProductFragment:
        """Run one search per grid case and merge the typed products."""
        model = request.model
        backend = request.backend
        grid = request.parameter_grid
        assert grid is not None
        base_params = dict(model.params)
        # The scan axes are the case axis of this job: their current value per case
        # belongs to the case identity, never to the search design.
        swept = tuple(
            grid_model_parameters(model, grid, forbidden=tuple(self.config.controls))
        )
        context = request.context

        def run_case(current: Any, point: Any) -> dict[str, Any]:
            # Pack while the case parameters are the model's: the candidate state
            # matrix is reconstructed from them.
            try:
                output = self.search(
                    current,
                    backend,
                    data=request.data,
                    context=context,
                    swept_parameters=swept,
                )
            except MemoryError:
                raise
            except Exception as exc:  # noqa: BLE001 - policy decides
                if request.case_failure_policy == "abort":
                    raise
                return self._case_payload(
                    current,
                    None,
                    flat_index=point.flat_index,
                    index=point.index,
                    values=point.values,
                    error=exc,
                )
            return self._case_payload(
                current,
                output,
                flat_index=point.flat_index,
                index=point.index,
                values=point.values,
            )

        cases = run_cases(
            grid,
            model=model,
            base_params=base_params,
            call=run_case,
            context=context,
            policy=request.case_failure_policy,
        )
        model_parameters(model, {**base_params, **grid_model_parameters(model, grid)})
        case_params: dict[str, Any] = {}
        for name, value in base_params.items():
            if np.asarray(value).ndim == 0:
                case_params[name] = np.full(grid.size, value)
        parameter_arrays = grid.parameter_arrays(flatten=True)
        for axis_name, target in grid.targets.items():
            case_params[target.rsplit(".", 1)[-1]] = np.asarray(
                parameter_arrays[axis_name]
            )
        return self._fragment(
            model,
            backend,
            [outcome.result for outcome in cases],
            case_axes={name: np.asarray(values) for name, values in grid.axes.items()},
            case_params=case_params,
            case_shape=tuple(grid.shape),
            scan_metadata={
                "scan_shape": tuple(grid.shape),
                "scan_combine": grid.combine,
                "scan_targets": dict(grid.targets),
            },
        )

    def _case_payload(
        self,
        model: Any,
        output: CAMBifurcationOutput | None,
        *,
        flat_index: int,
        index: tuple[int, ...],
        values: dict[str, Any],
        error: Exception | None = None,
    ) -> dict[str, Any]:
        """Return the per-case payload used to build the merged product."""
        metadata: dict[str, Any] = {
            "case_flat_index": int(flat_index),
            "case_index": [int(value) for value in index],
            "case_parameters": dict(values),
        }
        if output is None:
            metadata.update(
                {
                    "case_status": "error",
                    "case_error_type": type(error).__name__,
                    "case_error_message": str(error),
                }
            )
            return {
                "case_identity": "",
                "search_identity": "",
                "arrays": None,
                "metadata": metadata,
                "output": None,
            }
        search = str(output.metadata.get("search_identity", ""))
        case = str(output.metadata.get("case_identity", ""))
        metadata.update(
            {
                "case_identity": case,
                "search_identity": search,
                "case_status": "complete",
                "strategy": output.metadata.get("strategy"),
                "coverage": output.metadata.get("coverage"),
            }
        )
        return {
            "case_identity": case,
            "search_identity": search,
            "arrays": candidate_case_arrays(model, output),
            "metadata": metadata,
            "output": output,
        }

    def _fragment(
        self,
        model: Any,
        backend: Any,
        cases: list[dict[str, Any]],
        *,
        case_axes: dict[str, Any],
        case_params: dict[str, Any],
        case_shape: tuple[int, ...],
        scan_metadata: dict[str, Any],
    ) -> CAMProductFragment:
        """Merge per-case payloads into one candidate product and its audits."""
        arrays = [
            payload["arrays"]
            if payload["arrays"] is not None
            else candidate_case_arrays(model, _empty_candidate_output(self, model))
            for payload in cases
        ]
        dtype = str(np.asarray(arrays[0]["states"]).dtype) if arrays else "complex128"
        fragments: dict[int, Any] = {}
        for index, payload in enumerate(cases):
            if payload["output"] is None:
                continue
            fragments[index] = fragment_from_bifurcation_metadata(
                payload["output"].metadata,
                case_identity=payload["case_identity"],
                search_identity=payload["search_identity"],
                accepted_candidate_identities=payload["output"].metadata.get(
                    "accepted_candidate_identities", ()
                ),
                accepted_start_identities=payload["output"].metadata.get(
                    "accepted_start_identities", ()
                ),
                backend=str(backend.backend_name()),
                dtype=dtype,
                plugins={
                    "discovery": type(self.discovery).__name__,
                    "classifier": type(self.classifier).__name__,
                    "target": type(self.target).__name__,
                    "strategy": type(self.strategy).__name__,
                },
            )
        # The fragment is the case's own evidence contract: storing it in the
        # case metadata is what lets a summary read a candidate-only artifact
        # back without re-deriving an event. A failed case has no fragment, and
        # therefore no evidence to guess from.
        product = build_candidate_from_cases(
            case_arrays=arrays,
            case_meta=[
                (
                    {
                        **payload["metadata"],
                        "search_fragment": fragments[index].model_dump(mode="json"),
                    }
                    if index in fragments
                    else payload["metadata"]
                )
                for index, payload in enumerate(cases)
            ],
            case_axes=case_axes,
            case_params=case_params,
            case_shape=case_shape,
        )
        audits = list(fragments.values())
        first: dict[str, Any] = {}
        for payload in cases:
            if payload["output"] is not None:
                first = dict(payload["output"].metadata)
                break
        metadata = {
            **scan_metadata,
            "control_names": tuple(first.get("control_names", ())),
            "perturbation_parameter": first.get("perturbation_parameter", ""),
            "perturbation_scale": first.get("perturbation_scale", 1.0),
            "perturbation_side": first.get("perturbation_side", "both"),
            "closure": first.get("closure", {}),
            "fixed_params": first.get("fixed_params", {}),
            "state_ids": list(first.get("state_ids", ())),
            "target": first.get("target", self.target.name),
            "order": int(first.get("order", self.target.order)),
            "case_count": len(cases),
            "candidate_count": int(
                sum(len(item["candidate_identity"]) for item in arrays)
            ),
            "search_identities": sorted(
                {
                    payload["search_identity"]
                    for payload in cases
                    if payload["search_identity"]
                }
            ),
            "case_identities": [
                payload["case_identity"]
                for payload in cases
                if payload["case_identity"]
            ],
            "audit_count": len(audits),
        }
        return CAMProductFragment(
            products={"candidates": product}, audits=tuple(audits), metadata=metadata
        )

    def search(
        self,
        model: Any,
        backend: Any,
        *,
        data: Any | None = None,
        context: Any | None = None,
        swept_parameters: Sequence[str] = (),
    ) -> CAMBifurcationOutput:
        """Run the numeric search of one case (the core ``solve`` builds on).

        This is the documented numeric entry point: it returns the solver's own
        candidate output for one model state. The typed product and its audit are
        produced by :meth:`solve`, which is what the engine calls.

        ``swept_parameters`` names the model scalar parameters whose current values
        are one point of an external scan (the job's scan axes). They are fixed input
        of this call rather than search design: they leave the search identity and
        enter the case identity. A single, unscanned call passes none.
        """
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("bifurcation solver currently requires numpy")
        adapter = QPSGenDynamicsAdapter.from_model(model)
        self._validate_controls(adapter)
        reporter = context.progress if context is not None else None
        swept = tuple(str(name) for name in swept_parameters)
        scope_search, scope_case = self._scope_identities(model, swept_parameters=swept)
        if self.strategy.mode == "full":
            full_candidates, metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            full_candidates, metadata = self._finalize_identities(
                full_candidates,
                metadata,
                model=model,
                swept_parameters=swept,
            )
            return CAMBifurcationOutput(
                candidates=full_candidates,
                target=self.target.name,
                order=self.target.order,
                metadata=metadata,
            )
        if reporter is not None:
            reporter.status("Preparing scalar reductions", stage="reduce")
        try:
            reductions, reduction_search = self._select_reductions(adapter)
        except BifurcationCapabilityError:
            if self.strategy.mode != "auto":
                raise
            full_candidates, metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            metadata["fallback_reason"] = "no_regular_scalar_reduction"
            full_candidates, metadata = self._finalize_identities(
                full_candidates,
                metadata,
                model=model,
                swept_parameters=swept,
            )
            return CAMBifurcationOutput(
                candidates=full_candidates,
                target=self.target.name,
                order=self.target.order,
                metadata=metadata,
            )
        if not reductions:
            if self.strategy.mode != "auto":
                detail = "; ".join(reduction_search.get("consumer_errors", ()))
                message = "no regular scalar fraction-free reduction is available"
                if detail:
                    message = f"{message}: {detail}"
                raise BifurcationCapabilityError(message)
            full_candidates, metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            metadata["fallback_reason"] = "no_regular_scalar_reduction"
            metadata["reduction_search"] = reduction_search
            full_candidates, metadata = self._finalize_identities(
                full_candidates,
                metadata,
                model=model,
                swept_parameters=swept,
            )
            return CAMBifurcationOutput(
                candidates=full_candidates,
                target=self.target.name,
                order=self.target.order,
                metadata=metadata,
            )
        paths = [reduction_path(reduction) for reduction in reductions]
        duplicates = sorted({path for path in paths if paths.count(path) > 1})
        if duplicates:
            # One chart is one generation path: two audit entries with the same path
            # would pool two attempts under one identity, so this is an error.
            raise BifurcationCapabilityError(
                "the reduction search returned duplicated generation charts: "
                f"{duplicates}"
            )
        candidates: list[CAMBifurcationCandidate] = []
        reduction_runs: list[dict[str, Any]] = []
        rejected_count = 0
        rejected_by_reason: Counter[str] = Counter()
        near_misses = NearMissStore(
            per_reason=self.config.audit.near_miss_per_reason,
            total=self.config.audit.near_miss_total,
        )
        reduction_audits: list[dict[str, Any]] = []
        prefilter_rejected_total: Counter[str] = Counter()
        seed_skips_total: Counter[str] = Counter()
        workload_total: Counter[str] = Counter()
        for index, reduction in enumerate(reductions):
            path = paths[index]
            seed_stats = SeedGenerationStats(source=reduction.seed_source)
            starts = reduction.initial_starts(
                self._control_bounds(),
                samples_per_control=self.discovery.config.samples_per_control,
                max_starts=self.discovery.config.max_starts,
                order_parameter_bounds=self.strategy.config.order_parameter_bounds,
                order_parameter_samples=(self.discovery.config.order_parameter_samples),
                stats=seed_stats,
                control_axes=self._control_seed_axes(),
            )
            generated_count = len(starts)
            prefilter_rejected: Counter[str] = Counter()
            physical_starts: list[np.ndarray] = []
            for start in starts:
                reason = self._reduced_start_prefilter_reason(reduction, adapter, start)
                if reason is None:
                    physical_starts.append(start)
                else:
                    prefilter_rejected[reason] += 1
            starts = physical_starts
            if reporter is not None:
                with reporter.stage(
                    f"refine_{index + 1}",
                    total=len(starts),
                    unit="candidate",
                    message=(
                        f"Refining {len(starts)} candidates from "
                        f"{reduction.retained_id}"
                    ),
                ):
                    found, rejected, run_rejected_count, dedup_skipped = self._refine(
                        reduction,
                        adapter,
                        starts,
                        reporter=reporter,
                        near_misses=near_misses,
                        seed_source=seed_stats.source,
                        path=path,
                    )
            else:
                found, rejected, run_rejected_count, dedup_skipped = self._refine(
                    reduction,
                    adapter,
                    starts,
                    reporter=None,
                    near_misses=near_misses,
                    seed_source=seed_stats.source,
                    path=path,
                )
            candidates.extend(found)
            rejected_count += run_rejected_count
            rejected_by_reason.update(rejected)
            prefilter_rejected_total.update(prefilter_rejected)
            seed_skips_total.update(seed_stats.skipped)
            workload_total.update(_workload_fields(seed_stats))
            reduction_runs.append(
                {
                    "path": path,
                    "state_id": reduction.retained_id,
                    "method": reduction.method,
                    "degree": reduction.degree,
                    "refinement_start_count": len(starts),
                    "prefilter_rejected_count": generated_count - len(starts),
                    "accepted_count": len(found),
                    "rejected_count": run_rejected_count,
                    "rejected_by_reason": dict(rejected),
                }
            )
            reduction_audits.append(
                {
                    "path": path,
                    "state_id": reduction.retained_id,
                    "method": reduction.method,
                    "seed_source": seed_stats.source,
                    "physical_status": "checked",
                    "physical_check_stage": "prefilter",
                    "generation_trial_count": (
                        generated_count + seed_stats.skipped_total
                    ),
                    "generated_candidate_count": generated_count,
                    "prefilter_pass_count": len(starts),
                    "prefilter_rejected_count": generated_count - len(starts),
                    "prefilter_rejected": dict(prefilter_rejected),
                    "refinement_start_count": len(starts),
                    "refinement_duplicate_count": dedup_skipped,
                    "accepted_count": len(found),
                    "rejected_count": run_rejected_count,
                    "rejected_by_reason": dict(rejected),
                    "seed_skips": dict(seed_stats.skipped),
                    "seed_truncated": seed_stats.truncated,
                    "workload": _workload_fields(seed_stats),
                }
            )
        # Start identities exist before deduplication: a start that merges into
        # another candidate keeps its own identity on the survivor.
        self._assign_start_identities(candidates, case=scope_case, model=model)
        candidates = self._deduplicate_candidates(candidates)
        metadata = {
            "control_names": tuple(self.config.controls),
            "search_identity": scope_search,
            "case_identity": scope_case,
            "strategy": self.strategy.mode,
            "reduction_runs": reduction_runs,
            "reduction_state_id": (
                reductions[0].retained_id if len(reductions) == 1 else None
            ),
            "reduced_degree": (reductions[0].degree if len(reductions) == 1 else None),
            "refinement_start_count": sum(
                item["refinement_start_count"] for item in reduction_runs
            ),
            "rejected_count": rejected_count,
            "rejected_by_reason": dict(rejected_by_reason),
            "coverage": self._reduction_coverage(reduction_search),
            "structural_coverage": self._reduction_coverage(reduction_search),
            "numerical_coverage": "finite_control_grid_local_refinement",
            "domain_coverage": {
                "control_bounds": self._control_bounds(),
                "control_sampling": {
                    name: control.sampling
                    for name, control in self.config.controls.items()
                },
                "physical_status": "checked",
                "physical_check_stage": "prefilter",
                "generated_candidate_count": sum(
                    item["generated_candidate_count"] for item in reduction_audits
                ),
                "refinement_start_count": sum(
                    item["refinement_start_count"] for item in reduction_audits
                ),
            },
            "singular_coverage": "not_covered_by_regular_reduction",
            "reduction_search": reduction_search,
            "compile_cache_hit": bool(reduction_search.get("compile_cache_hit", False)),
            "qpsgen": adapter.provenance(),
            "closure": adapter.closure_provenance(),
            "state_scale_source": self._model_state_scales(adapter)[1],
            "perturbation_parameter": self.config.perturbation.parameter,
            "perturbation_scale": self.config.perturbation.scale,
            "perturbation_side": self.config.perturbation.side,
        }
        if self.strategy.mode == "auto":
            full_candidates, full_metadata = self._solve_full(
                adapter, data=data, reporter=reporter, near_misses=near_misses
            )
            self._assign_start_identities(full_candidates, case=scope_case, model=model)
            candidates = self._deduplicate_candidates([*candidates, *full_candidates])
            metadata["full_fallback"] = full_metadata
            metadata["coverage"] = f"{metadata['coverage']}_and_bordered_local_search"
        full_audit = (
            metadata.get("full_fallback", {}).get("audit")
            if self.strategy.mode == "auto"
            else None
        )
        near_miss_records = near_misses.finalize()
        totals: dict[str, Any] = {
            field: sum(item[field] for item in reduction_audits)
            for field in _CANDIDATE_COUNT_FIELDS
        }
        rejected_total_by_reason: Counter[str] = Counter(rejected_by_reason)
        seed_truncated = any(item["seed_truncated"] for item in reduction_audits)
        if full_audit is not None:
            for field in _CANDIDATE_COUNT_FIELDS:
                totals[field] += full_audit[field]
            prefilter_rejected_total.update(full_audit["prefilter_rejected"])
            rejected_total_by_reason.update(full_audit["rejected_by_reason"])
            seed_skips_total.update(full_audit["seed_skips"])
            workload_total.update(full_audit["workload"])
            seed_truncated = seed_truncated or full_audit["seed_truncated"]
        totals.update(
            {
                "prefilter_rejected": dict(prefilter_rejected_total),
                "rejected_by_reason": dict(rejected_total_by_reason),
                "seed_skips": dict(seed_skips_total),
                "seed_truncated": seed_truncated,
                "workload": dict(workload_total),
                "near_miss_saved": len(near_miss_records),
                "near_miss_dropped": near_misses.dropped,
                "unique_candidate_count": len(candidates),
            }
        )
        metadata["audit"] = {
            "schema": AUDIT_SCHEMA,
            "path": self.strategy.mode,
            "physical_status": "checked",
            "field_units": AUDIT_FIELD_UNITS,
            "conservation": AUDIT_CONSERVATION,
            "reductions": reduction_audits,
            "totals": totals,
            "rejection_counter_semantics": REJECTION_COUNTER_SEMANTICS,
            "near_miss_selection": near_miss_selection_metadata(self.config.audit),
            "near_misses": near_miss_records,
            "result_note": (
                EMPTY_RESULT_NOTE if not candidates else "candidates_found"
            ),
        }
        metadata["near_misses_json"] = near_miss_json(near_miss_records)
        candidates, metadata = self._finalize_identities(
            candidates, metadata, model=model, swept_parameters=swept
        )
        return CAMBifurcationOutput(
            candidates=candidates,
            target=self.target.name,
            order=self.target.order,
            metadata=metadata,
        )

    @staticmethod
    def _reduction_coverage(search: dict[str, Any]) -> str:
        if search["complete"]:
            return "regular_reduction_exhaustive"
        if search["truncation_reasons"] == ["return_limit"]:
            return "regular_reduction_ranked_prefix"
        return "regular_reduction_budget_limited"

    def _solve_full(
        self,
        adapter: QPSGenDynamicsAdapter,
        *,
        data: Any | None,
        reporter: Any | None,
        near_misses: NearMissStore | None = None,
    ) -> tuple[list[CAMBifurcationCandidate], dict[str, Any]]:
        system = BorderedMultiplicitySystem(
            adapter,
            n_state=adapter.state_size,
            order=self.target.order,
            control_names=tuple(self.config.controls),
            base_params=adapter.model.params,
        )
        owns_near_misses = near_misses is None
        if near_misses is None:
            near_misses = NearMissStore(
                per_reason=self.config.audit.near_miss_per_reason,
                total=self.config.audit.near_miss_total,
            )
        seed_stats = SeedGenerationStats(source="upstream")
        seeds = self._upstream_seeds(system, adapter, data, stats=seed_stats)
        seed_source = "upstream"
        if not seeds:
            seed_stats = SeedGenerationStats(source="domain_sampling")
            seeds = self._discover_full_seeds(system, adapter, stats=seed_stats)
            seed_source = "domain_sampling"
        candidates: list[CAMBifurcationCandidate] = []
        solved: list[np.ndarray] = []
        rejected_count = 0
        dedup_skipped = 0
        rejected_by_reason: Counter[str] = Counter()
        if reporter is not None:
            stage = reporter.stage(
                "bordered_refine",
                total=len(seeds),
                unit="candidate",
                message=f"Refining {len(seeds)} bordered candidates",
            )
        else:
            from contextlib import nullcontext

            stage = nullcontext()
        lower, upper = self._full_bounds(system, adapter)
        with stage:
            for seed in seeds:
                initial = np.minimum(np.maximum(seed, lower), upper)
                result = least_squares(
                    system.residual,
                    initial,
                    bounds=(lower, upper),
                    x_scale="jac",
                    jac="3-point",
                    ftol=self.config.refinement.tolerance,
                    xtol=self.config.refinement.tolerance,
                    gtol=self.config.refinement.tolerance,
                    max_nfev=self.config.refinement.max_iterations,
                )
                if reporter is not None:
                    reporter.advance()
                value = np.asarray(result.x)
                if any(
                    scaled_distance(value, previous, self._full_scale(system, adapter))
                    <= 1e-6
                    for previous in solved
                ):
                    dedup_skipped += 1
                    continue
                solved.append(value)
                candidate = self._full_candidate(
                    system, adapter, value, result.cost, result.success
                )
                candidate.metadata["generation_path"] = "full"
                if candidate.success:
                    candidates.append(candidate)
                else:
                    rejected_count += 1
                    reasons = tuple(
                        candidate.metadata.get("rejection_reasons", ("unknown",))
                    )
                    rejected_by_reason.update(reasons)
                    near_misses.add(
                        path="full",
                        reduction=None,
                        seed_source=seed_source,
                        controls=candidate.controls,
                        rejection_reasons=reasons,
                        search_residual=candidate.search_residual_norm,
                        full_residual=candidate.full_residual_norm,
                        min_state_eigenvalue=candidate.metadata.get(
                            "minimum_physical_eigenvalue", np.nan
                        ),
                    )
        audit: dict[str, Any] = {
            "schema": AUDIT_SCHEMA,
            "path": "full",
            "seed_source": seed_source,
            "physical_status": "checked",
            "physical_check_stage": "generation",
            "generation_trial_count": len(seeds) + seed_stats.skipped_total,
            "generated_candidate_count": len(seeds),
            "prefilter_pass_count": len(seeds),
            "prefilter_rejected_count": 0,
            "prefilter_rejected": {},
            "refinement_start_count": len(seeds),
            "refinement_duplicate_count": dedup_skipped,
            "accepted_count": len(candidates),
            "rejected_count": rejected_count,
            "rejected_by_reason": dict(rejected_by_reason),
            "seed_skips": dict(seed_stats.skipped),
            "seed_truncated": seed_stats.truncated,
            "workload": _workload_fields(seed_stats),
            "rejection_counter_semantics": REJECTION_COUNTER_SEMANTICS,
            "near_miss_selection": near_miss_selection_metadata(self.config.audit),
        }
        metadata: dict[str, Any] = {
            "control_names": tuple(self.config.controls),
            "strategy": "full",
            "refinement_start_count": len(seeds),
            "rejected_count": rejected_count,
            "rejected_by_reason": dict(rejected_by_reason),
            "coverage": f"{seed_source}_bordered_local_search",
            "structural_coverage": "full_bordered_local_system",
            "numerical_coverage": f"{seed_source}_bordered_local_search",
            "domain_coverage": {
                "control_bounds": self._control_bounds(),
                "control_sampling": {
                    name: control.sampling
                    for name, control in self.config.controls.items()
                },
                "physical_status": "checked",
                "physical_check_stage": "generation",
                "seed_source": seed_source,
                "generated_candidate_count": len(seeds),
                "refinement_start_count": len(seeds),
            },
            "singular_coverage": "local_bordered_search",
            "qpsgen": adapter.provenance(),
            "closure": adapter.closure_provenance(),
            "state_scale_source": self._model_state_scales(adapter)[1],
            "perturbation_parameter": self.config.perturbation.parameter,
            "perturbation_scale": self.config.perturbation.scale,
            "perturbation_side": self.config.perturbation.side,
            "audit": audit,
        }
        if owns_near_misses:
            records = near_misses.finalize()
            audit["near_misses"] = records
            audit["result_note"] = (
                EMPTY_RESULT_NOTE if not candidates else "candidates_found"
            )
            audit["field_units"] = AUDIT_FIELD_UNITS
            audit["conservation"] = AUDIT_CONSERVATION
            audit["totals"] = {
                **{field: audit[field] for field in _CANDIDATE_COUNT_FIELDS},
                "prefilter_rejected": {},
                "rejected_by_reason": dict(rejected_by_reason),
                "seed_skips": dict(seed_stats.skipped),
                "seed_truncated": seed_stats.truncated,
                "workload": _workload_fields(seed_stats),
                "near_miss_saved": len(records),
                "near_miss_dropped": near_misses.dropped,
                "unique_candidate_count": len(candidates),
            }
            metadata["near_misses_json"] = near_miss_json(records)
        return candidates, metadata

    def _upstream_seeds(
        self,
        system: BorderedMultiplicitySystem,
        adapter: QPSGenDynamicsAdapter,
        data: Any | None,
        stats: SeedGenerationStats | None = None,
    ) -> list[np.ndarray]:
        """Build bordered-system seeds from upstream steady states.

        Every upstream state is individually checked against the physical
        domain (``adapter.state_matrix`` PSD via ``_state_is_physical``)
        before it becomes a seed; states that fail are counted under the
        ``non_physical_state`` skip reason, never as physical starts.  The
        ``upstream_seed`` workload unit counts evaluated upstream states.
        """
        if data is None or not hasattr(data, "states"):
            return []
        seeds = []
        grid_shape = tuple(getattr(data, "grid_shape", ()))
        indices = np.ndindex(grid_shape) if grid_shape else [()]
        for grid_index in indices:
            point = data.point_view(grid_index) if grid_shape else data
            states = np.asarray(point.states)
            valid = np.asarray(
                getattr(point, "valid_mask", np.ones(len(states), dtype=bool))
            )
            params = (
                data.params_at(grid_index)
                if hasattr(data, "params_at")
                else dict(getattr(data, "params", {}))
            )
            if not all(name in params for name in self.config.controls):
                if stats is not None:
                    stats.skip("missing_controls")
                continue
            controls = {name: float(params[name]) for name in self.config.controls}
            if not self._controls_in_bounds(controls):
                if stats is not None:
                    stats.skip("controls_out_of_bounds")
                continue
            check_params = {**adapter.model.params, **params}
            for state, is_valid in zip(states, valid, strict=True):
                if stats is not None:
                    stats.work("upstream_seed")
                if not is_valid:
                    if stats is not None:
                        stats.skip("invalid_state")
                    continue
                vector = matrix_to_vector(state)
                if not self._state_is_physical(adapter, vector, check_params):
                    if stats is not None:
                        stats.skip("non_physical_state")
                    continue
                try:
                    seeds.append(system.seed(vector, controls))
                except np.linalg.LinAlgError:
                    if stats is not None:
                        stats.skip("seed_failed")
                    continue
        limit = self.discovery.config.max_starts
        if stats is not None and len(seeds) > limit:
            stats.truncated = True
        return seeds[:limit]

    def _discover_full_seeds(
        self,
        system: BorderedMultiplicitySystem,
        adapter: QPSGenDynamicsAdapter,
        stats: SeedGenerationStats | None = None,
    ) -> list[np.ndarray]:
        axes = [
            control.sample_values(self.discovery.config.samples_per_control)
            for control in self.config.controls.values()
        ]
        starts = []
        for control_values in product(*axes):
            if stats is not None:
                stats.work("control_point")
            known_states: list[np.ndarray] = []
            controls = dict(zip(self.config.controls, control_values, strict=True))
            params = {**adapter.model.params, **controls}
            for guess in self._state_guesses(adapter):
                if stats is not None:
                    stats.work("fixed_point_guess")
                solution = root(
                    lambda value, params=params: adapter.rhs(value, params),
                    guess,
                    jac=lambda value, params=params: adapter.jacobian(value, params),
                    options={"maxfev": self.config.refinement.max_iterations},
                )
                state = np.asarray(solution.x)
                if (
                    not solution.success
                    or np.linalg.norm(adapter.rhs(state, params)) > 1e-7
                ):
                    if stats is not None:
                        stats.skip("fixed_point_not_converged")
                    continue
                if not self._state_is_physical(adapter, state, params):
                    if stats is not None:
                        stats.skip("non_physical_state")
                    continue
                if any(np.linalg.norm(state - item) < 1e-7 for item in known_states):
                    if stats is not None:
                        stats.skip("duplicate_state")
                    continue
                known_states.append(state)
                try:
                    starts.append(system.seed(state, controls))
                except np.linalg.LinAlgError:
                    if stats is not None:
                        stats.skip("seed_failed")
                    continue
                if len(starts) >= self.discovery.config.max_starts:
                    if stats is not None:
                        stats.truncated = True
                    return starts
        return starts

    def _full_candidate(
        self,
        system: BorderedMultiplicitySystem,
        adapter: QPSGenDynamicsAdapter,
        value: np.ndarray,
        cost: float,
        optimizer_success: bool,
        verification: Any | None = None,
    ) -> CAMBifurcationCandidate:
        if verification is not None:
            value = verification.value
        state, control_values, right, left = system.unpack(value)
        controls = dict(zip(self.config.controls, control_values, strict=True))
        params = {**adapter.model.params, **controls}
        diagnostics = system.diagnostics(value)
        residual = system.residual(value)
        full_residual = adapter.rhs(state, params)
        physical_eigenvalues = adapter.physical_eigenvalues(state, params)
        jacobian_eigenvalues = np.linalg.eigvals(adapter.jacobian(state, params))
        threshold = max(np.sqrt(self.config.refinement.tolerance), 1e-9)
        singular_tolerance = self.strategy.config.singular_tolerance
        simple_zero = bool(
            diagnostics.singular_values[-1] <= threshold
            and (
                len(diagnostics.singular_values) == 1
                or diagnostics.singular_values[-2] > singular_tolerance
            )
        )
        is_physical = bool(
            np.min(physical_eigenvalues) >= -self.config.state_domain.psd_tolerance
        )
        rejection_reasons = []
        if not optimizer_success:
            rejection_reasons.append("optimizer_failed")
        if verification is not None and not verification.success:
            rejection_reasons.append("verification_failed")
        if np.linalg.norm(residual) > threshold:
            rejection_reasons.append("search_residual")
        if np.linalg.norm(full_residual) > threshold:
            rejection_reasons.append("full_residual")
        if abs(diagnostics.coefficients[self.target.order]) <= singular_tolerance:
            rejection_reasons.append("higher_multiplicity")
        if not simple_zero:
            rejection_reasons.append("non_simple_kernel")
        if not is_physical:
            rejection_reasons.append("non_physical")
        success = not rejection_reasons
        if success and verification is None:
            verified = system.verify(
                value,
                initial_digits=self.config.verification.initial_digits,
                max_digits=self.config.verification.max_digits,
            )
            return self._full_candidate(
                system,
                adapter,
                verified.value,
                cost,
                optimizer_success,
                verification=verified,
            )
        return CAMBifurcationCandidate(
            state_vector=state,
            controls={name: float(item) for name, item in controls.items()},
            full_residual_norm=float(np.linalg.norm(full_residual)),
            search_residual_norm=float(np.linalg.norm(residual)),
            success=success,
            status="verified" if success else "rejected",
            method="bordered_full",
            metadata={
                "optimizer_cost": float(cost),
                "verification_digits": (
                    verification.digits if verification is not None else 0
                ),
                "verification_working_digits": (
                    verification.digits if verification is not None else 0
                ),
                "verified_unknown_decimal_values": (
                    verification.unknown_decimal_values
                    if verification is not None
                    else ()
                ),
                "verified_full_state_decimal_values": (
                    verification.full_state_decimal_values
                    if verification is not None
                    else ()
                ),
                "multiplicity_residual_norm": (
                    verification.multiplicity_residual_norm
                    if verification is not None
                    else np.nan
                ),
                "verified_full_residual_norm": (
                    verification.full_residual_norm
                    if verification is not None
                    else np.nan
                ),
                "verification_status": (
                    "verified"
                    if verification is not None and verification.success
                    else "failed"
                ),
                "reduced_coefficients": diagnostics.coefficients,
                "jacobian_singular_values": diagnostics.singular_values,
                "right_null_vector": right,
                "left_null_vector": left,
                "is_physical": is_physical,
                "is_stable": bool(np.max(np.real(jacobian_eigenvalues)) <= threshold),
                "maximum_jacobian_real_part": float(
                    np.max(np.real(jacobian_eigenvalues))
                ),
                "minimum_physical_eigenvalue": float(np.min(physical_eigenvalues)),
                "rejection_reasons": tuple(rejection_reasons),
                "classification_status": (
                    "full_strategy_unavailable" if success else "not_run"
                ),
                "scaling_signatures": (),
                "classification_accepted": False,
            },
        )

    def _full_bounds(
        self,
        system: BorderedMultiplicitySystem,
        adapter: QPSGenDynamicsAdapter,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = system.n_state
        lower_state = np.full(n, -np.inf)
        lower_state[list(adapter.diagonal_state_indices)] = 0.0
        lower = np.concatenate(
            (
                lower_state,
                np.asarray([item.min for item in self.config.controls.values()]),
                np.full(2 * n, -np.inf),
            )
        )
        upper = np.concatenate(
            (
                np.full(n, np.inf),
                np.asarray([item.max for item in self.config.controls.values()]),
                np.full(2 * n, np.inf),
            )
        )
        return lower, upper

    def _full_scale(
        self, system: BorderedMultiplicitySystem, adapter: QPSGenDynamicsAdapter
    ) -> np.ndarray:
        n = system.n_state
        state_scales, _ = self._model_state_scales(adapter)
        return np.asarray(
            [
                *state_scales,
                *(
                    self._control_scale(control)
                    for control in self.config.controls.values()
                ),
                *([1.0] * (2 * n)),
            ]
        )

    def _state_guesses(self, adapter: QPSGenDynamicsAdapter) -> list[np.ndarray]:
        n_state = adapter.state_size
        state_scales, _ = self._model_state_scales(adapter)
        scale = float(np.max(state_scales))
        guesses = [np.zeros(n_state)]
        for value in (0.5, 1.0, scale):
            guess = np.zeros(n_state)
            guess[list(adapter.diagonal_state_indices)] = value
            guesses.append(guess)
        return guesses

    def _controls_in_bounds(self, controls: dict[str, float]) -> bool:
        return all(
            self.config.controls[name].min <= value <= self.config.controls[name].max
            for name, value in controls.items()
        )

    def _validate_controls(self, adapter: QPSGenDynamicsAdapter) -> None:
        unknown = set(self.config.controls) - set(adapter.parameter_names)
        if unknown:
            raise ValueError(
                f"unknown bifurcation controls for {adapter.model.name}: "
                f"{sorted(unknown)}"
            )
        if self.config.perturbation.parameter not in adapter.parameter_names:
            raise ValueError(
                "unknown perturbation parameter "
                f"{self.config.perturbation.parameter!r} for {adapter.model.name}"
            )
        nonscalar = [
            name
            for name, value in adapter.model.params.items()
            if np.asarray(value).ndim != 0
        ]
        if nonscalar:
            raise ValueError(
                "bifurcation model parameters must be scalar; "
                f"found arrays for {nonscalar}"
            )
        domains = adapter.parameter_domains
        for name, control in self.config.controls.items():
            model_domain = domains[name]
            nonnegative = (
                model_domain in {"positive", "nonnegative"}
                or control.domain == "nonnegative"
            )
            if control.domain == "real" and model_domain in {
                "positive",
                "nonnegative",
            }:
                raise ValueError(
                    f"control {name!r} cannot relax model domain "
                    f"{model_domain!r} to real"
                )
            if nonnegative and control.min < 0.0:
                raise ValueError(
                    f"control {name!r} is nonnegative but has min={control.min}"
                )
        for name, domain in domains.items():
            if name in self.config.controls or domain not in {
                "positive",
                "nonnegative",
            }:
                continue
            value = float(adapter.model.params[name])
            if value < 0.0:
                raise ValueError(
                    f"model parameter {name!r} is {domain} but has value {value}"
                )

    def _select_reductions(
        self, adapter: QPSGenDynamicsAdapter
    ) -> tuple[
        tuple[FractionFreeScalarReduction | CondensedScalarReduction, ...],
        dict[str, Any],
    ]:
        strategy_config = self.strategy.config
        provenance = adapter.provenance()
        cache_key = (
            provenance.get("fingerprint"),
            self.target.order,
            tuple(self.config.controls),
            repr(strategy_config.model_dump()),
        )
        cached = self._reduction_cache.get(cache_key)
        if cached is not None:
            cached_reductions, cached_manifest = cached
            manifest = dict(cached_manifest)
            manifest["compile_cache_hit"] = True
            return self._bind_reductions(cached_reductions, adapter), manifest
        search = adapter.search_linear_reductions(
            retained_dimension=1,
            retained_ids=(
                None
                if strategy_config.order_parameter is None
                else (strategy_config.order_parameter,)
            ),
            return_limit=strategy_config.max_candidates,
            partition_limit=strategy_config.partition_limit,
            materialization_limit=strategy_config.materialization_limit,
            equation_partitions="all",
        )
        candidates = search.candidates
        errors = []
        reductions: list[FractionFreeScalarReduction | CondensedScalarReduction] = []
        for candidate in candidates:
            # A candidate without a chart id cannot name its generation path, and
            # that is a contract violation rather than a per-candidate materialization
            # failure: fail here instead of dropping the chart silently.
            chart_id = _candidate_chart_id(candidate)
            if len(candidate.eliminated_indices) > 3:
                try:
                    reductions.append(
                        CondensedScalarReduction(
                            adapter.linear_reduction(candidate=candidate),
                            order=self.target.order,
                            control_names=tuple(self.config.controls),
                            base_params=adapter.model.params,
                            chart_id=chart_id,
                        )
                    )
                except Exception as exc:
                    errors.append(f"{candidate.retained_ids}: {exc}")
                continue
            if (
                candidate.reduced_degree is not None
                and candidate.reduced_degree < self.target.order
            ):
                continue
            try:
                materialized = adapter.materialized_linear_reduction(
                    candidate=candidate
                )
                reduction = FractionFreeScalarReduction(
                    materialized,
                    order=self.target.order,
                    control_names=tuple(self.config.controls),
                    base_params=adapter.model.params,
                    chart_id=chart_id,
                )
            except Exception as exc:
                errors.append(f"{candidate.retained_ids}: {exc}")
                continue
            if reduction.degree >= self.target.order:
                reductions.append(reduction)
        manifest = search.manifest()
        manifest["consumer_error_count"] = len(errors)
        manifest["consumer_errors"] = tuple(errors[:3])
        # qpsgen reduction contract: the manifest carries
        # ``materialization_failure_count`` and the search result carries
        # ``materialization_failures`` (entries with ``chart_id`` and
        # ``error``); ``rejected_reason_counts`` never contains
        # ``materialization_failed``.  Keep a bounded serializable summary
        # here; the count always reflects the full number of failures.
        materialization_failures = tuple(
            getattr(search, "materialization_failures", ())
        )
        manifest["materialization_failure_count"] = int(
            manifest.get("materialization_failure_count", len(materialization_failures))
        )
        manifest["materialization_failures"] = tuple(
            {
                "chart_id": str(getattr(failure, "chart_id", "unknown")),
                "error": str(getattr(failure, "error", failure)),
            }
            for failure in materialization_failures[:3]
        )
        manifest["compile_cache_hit"] = False
        output = tuple(reductions)
        if len(self._reduction_cache) >= 8:
            self._reduction_cache.pop(next(iter(self._reduction_cache)))
        self._reduction_cache[cache_key] = (output, dict(manifest))
        return self._bind_reductions(output, adapter), manifest

    def _bind_reductions(
        self,
        reductions: tuple[FractionFreeScalarReduction | CondensedScalarReduction, ...],
        adapter: QPSGenDynamicsAdapter,
    ) -> tuple[FractionFreeScalarReduction | CondensedScalarReduction, ...]:
        """Bind runtime parameters without mutating cached structure templates."""
        state_scales, _ = self._model_state_scales(adapter)
        scale_by_id = dict(zip(adapter.state_ids, state_scales, strict=True))
        bound = []
        for reduction in reductions:
            runtime_reduction = copy.copy(reduction)
            runtime_reduction.base_params = dict(adapter.model.params)
            runtime_reduction.retained_scale = float(
                scale_by_id.get(reduction.retained_id, np.max(state_scales))
            )
            bound.append(runtime_reduction)
        return tuple(bound)

    @staticmethod
    def _model_state_scales(
        adapter: QPSGenDynamicsAdapter,
    ) -> tuple[np.ndarray, str]:
        provider = getattr(adapter.model, "cam_bifurcation_scales", None)
        if not callable(provider):
            return np.ones(adapter.state_size), "heuristic:unit"
        payload = provider(dict(adapter.model.params))
        source = "model"
        if isinstance(payload, dict):
            source = str(payload.get("source", source))
            payload = payload.get("state", 1.0)
        values = np.asarray(payload, dtype=float)
        if values.ndim == 0:
            values = np.full(adapter.state_size, float(values))
        values = np.broadcast_to(values, (adapter.state_size,)).copy()
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("CAM bifurcation state scales must be finite and positive")
        return values, source

    def _refine(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: QPSGenDynamicsAdapter,
        starts: list[np.ndarray],
        *,
        reporter: Any | None,
        near_misses: NearMissStore | None = None,
        seed_source: str = "unknown",
        path: str,
    ) -> tuple[list[CAMBifurcationCandidate], Counter[str], int, int]:
        if not starts:
            return [], Counter(), 0, 0
        tolerance = self.config.refinement.tolerance
        variable_scale = np.asarray(
            [
                max(1.0, float(np.nanmedian(np.abs([item[0] for item in starts])))),
                *(
                    self._control_scale(control)
                    for control in self.config.controls.values()
                ),
            ]
        )
        jacobian_samples = np.asarray(
            [np.abs(reduction.jacobian(start)) for start in starts]
        )
        equation_scale = np.maximum(
            np.nanmedian(
                np.linalg.norm(
                    jacobian_samples * variable_scale[None, None, :],
                    axis=-1,
                ),
                axis=0,
            ),
            np.sqrt(tolerance),
        )
        lower = np.asarray(
            [
                0.0 if reduction.retained_id.startswith("r_diag_") else -np.inf,
                *(control.min for control in self.config.controls.values()),
            ]
        )
        upper = np.asarray(
            [
                np.inf,
                *(control.max for control in self.config.controls.values()),
            ]
        )
        accepted: list[CAMBifurcationCandidate] = []
        solved: list[np.ndarray] = []
        rejected: Counter[str] = Counter()
        rejected_count = 0
        dedup_skipped = 0
        for start in starts:
            initial = np.minimum(np.maximum(start, lower), upper)
            result = least_squares(
                lambda value: reduction.equations(value) / equation_scale,
                initial,
                jac=lambda value: reduction.jacobian(value) / equation_scale[:, None],
                bounds=(lower, upper),
                x_scale=variable_scale,
                ftol=tolerance,
                xtol=tolerance,
                gtol=tolerance,
                max_nfev=self.config.refinement.max_iterations,
            )
            value = np.asarray(result.x)
            if reporter is not None:
                reporter.advance()
            if any(
                scaled_distance(value, previous, variable_scale) <= 1e-6
                for previous in solved
            ):
                dedup_skipped += 1
                continue
            solved.append(value)
            candidate = self._candidate(
                reduction, adapter, value, result.cost, result.success
            )
            # The start's identity needs the chart it came from, not just the
            # method name: two charts of one retained state are two starts.
            candidate.metadata["generation_path"] = path
            if candidate.success:
                accepted.append(candidate)
            else:
                rejected_count += 1
                reasons = tuple(
                    candidate.metadata.get("rejection_reasons", ("unknown",))
                )
                rejected.update(reasons)
                if near_misses is not None:
                    near_misses.add(
                        path=path,
                        reduction=reduction.retained_id,
                        seed_source=seed_source,
                        controls=candidate.controls,
                        rejection_reasons=reasons,
                        search_residual=candidate.search_residual_norm,
                        full_residual=candidate.full_residual_norm,
                        min_state_eigenvalue=candidate.metadata.get(
                            "minimum_physical_eigenvalue", np.nan
                        ),
                    )
        return accepted, rejected, rejected_count, dedup_skipped

    def _candidate(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: QPSGenDynamicsAdapter,
        value: np.ndarray,
        cost: float,
        optimizer_success: bool,
        verification: Any | None = None,
    ) -> CAMBifurcationCandidate:
        if verification is not None:
            value = verification.value
        controls = dict(zip(self.config.controls, value[1:], strict=True))
        params = {**adapter.model.params, **controls}
        vector = reduction.reconstruct(value)
        equations = reduction.equations(value)
        diagnostics = reduction.diagnostics(value)
        full_residual = adapter.rhs(vector, params)
        jacobian = adapter.jacobian(vector, params)
        physical_eigenvalues = adapter.physical_eigenvalues(vector, params)
        jacobian_eigenvalues = np.linalg.eigvals(jacobian)
        search_norm = float(np.linalg.norm(equations))
        full_norm = float(np.linalg.norm(full_residual))
        threshold = max(np.sqrt(self.config.refinement.tolerance), 1e-9)
        singular_tolerance = self.strategy.config.singular_tolerance
        next_coefficient = abs(
            float(diagnostics.reduced_coefficients[self.target.order])
        )
        is_physical = bool(
            np.min(physical_eigenvalues) >= -self.config.state_domain.psd_tolerance
        )
        regular = bool(
            abs(diagnostics.regularity_determinant) > singular_tolerance
            and diagnostics.denominator_margin > singular_tolerance
            and diagnostics.condition_number <= self.strategy.config.condition_limit
        )
        rejection_reasons = []
        if not optimizer_success:
            rejection_reasons.append("optimizer_failed")
        if verification is not None and not verification.success:
            rejection_reasons.append("verification_failed")
        if not finite_vector(vector):
            rejection_reasons.append("non_finite_state")
        if search_norm > threshold:
            rejection_reasons.append("search_residual")
        if full_norm > threshold:
            rejection_reasons.append("full_residual")
        if next_coefficient <= singular_tolerance:
            rejection_reasons.append("higher_multiplicity")
        if not regular:
            rejection_reasons.append("singular_reduction")
        if not is_physical:
            rejection_reasons.append("non_physical")
        success = not rejection_reasons
        if success and verification is None:
            verified = reduction.verify(
                value,
                initial_digits=self.config.verification.initial_digits,
                max_digits=self.config.verification.max_digits,
            )
            return self._candidate(
                reduction,
                adapter,
                verified.value,
                cost,
                optimizer_success,
                verification=verified,
            )
        metadata = {
            "optimizer_cost": float(cost),
            "verification_digits": (
                verification.digits if verification is not None else 0
            ),
            "verification_working_digits": (
                verification.digits if verification is not None else 0
            ),
            "verified_unknown_decimal_values": (
                verification.unknown_decimal_values if verification is not None else ()
            ),
            "verified_full_state_decimal_values": (
                verification.full_state_decimal_values
                if verification is not None
                else ()
            ),
            "multiplicity_residual_norm": (
                verification.multiplicity_residual_norm
                if verification is not None
                else np.nan
            ),
            "verified_full_residual_norm": (
                verification.full_residual_norm if verification is not None else np.nan
            ),
            "verification_status": (
                "verified"
                if verification is not None and verification.success
                else "failed"
            ),
            "regularity_determinant": diagnostics.regularity_determinant,
            "denominator_margin": diagnostics.denominator_margin,
            "reduction_condition_number": diagnostics.condition_number,
            "reduced_coefficients": diagnostics.reduced_coefficients,
            "jacobian_singular_values": np.linalg.svd(jacobian, compute_uv=False),
            "is_physical": is_physical,
            "is_stable": bool(np.max(np.real(jacobian_eigenvalues)) <= threshold),
            "maximum_jacobian_real_part": float(np.max(np.real(jacobian_eigenvalues))),
            "minimum_physical_eigenvalue": float(np.min(physical_eigenvalues)),
            "rejection_reasons": tuple(rejection_reasons),
        }
        if success:
            metadata.update(
                self.classifier.classify(
                    reduction,
                    value,
                    vector,
                    params,
                    adapter,
                    perturbation=self.config.perturbation.parameter,
                    scale=self.config.perturbation.scale,
                    side=self.config.perturbation.side,
                    verification_digits=(
                        verification.digits if verification is not None else 30
                    ),
                )
            )
        else:
            metadata.update(
                {
                    "classification_status": "not_run",
                    "scaling_signatures": (),
                    "classification_accepted": False,
                }
            )
        return CAMBifurcationCandidate(
            state_vector=vector,
            controls={name: float(item) for name, item in controls.items()},
            full_residual_norm=full_norm,
            search_residual_norm=search_norm,
            success=success,
            status="verified" if success else "rejected",
            method=reduction.method,
            metadata=metadata,
        )

    # -- solver-owned identities -------------------------------------------

    def _scope_identities(
        self, model: Any, *, swept_parameters: Sequence[str] = ()
    ) -> tuple[str, str]:
        """Return the ``(search, case)`` identities of this solve call.

        The case values are the controls plus the swept model parameters of the
        job's scan: two scan points that differ in one of them are two cases of one
        search. A swept parameter without a finite scalar value is an error, never a
        silently ignored case axis.
        """
        swept = tuple(str(name) for name in swept_parameters)
        scope = derive_search_scope(
            model,
            target=self.target.name,
            order=int(self.target.order),
            controls=self.config.controls,
            perturbation={
                "parameter": self.config.perturbation.parameter,
                "scale": float(self.config.perturbation.scale),
                "side": str(self.config.perturbation.side),
            },
            strategy=str(self.strategy.mode),
            swept_parameters=swept,
        )
        search = derive_search_identity(scope)
        params = getattr(model, "params", {}) or {}
        case_params = {
            name: float(np.asarray(params[name]).item())
            for name in self.config.controls
            if name in params and np.asarray(params[name]).ndim == 0
        }
        for name in swept:
            if name not in params:
                raise ValueError(
                    f"swept parameter {name!r} is missing from the model parameters"
                )
            value = np.asarray(params[name])
            if value.ndim != 0:
                raise ValueError(
                    f"swept case parameter {name!r} must be a scalar; got shape "
                    f"{value.shape}"
                )
            number = float(value.item())
            if not np.isfinite(number):
                raise ValueError(
                    f"swept case parameter {name!r} must be finite; got {number!r}"
                )
            case_params[name] = number
        return search, derive_case_identity(search, case_params)

    def _identity_content(
        self, model: Any, candidate: CAMBifurcationCandidate
    ) -> tuple[dict[str, Any], str]:
        """Return the content payload and basis of one candidate."""
        return derive_candidate_content(
            model_name=str(getattr(model, "name", type(model).__name__)),
            target=self.target.name,
            order=int(self.target.order),
            state_vector=candidate.state_vector,
            controls=candidate.controls,
            verified_full_state_decimal_values=candidate.metadata.get(
                "verified_full_state_decimal_values", ()
            ),
        )

    def _generation_path(self, candidate: CAMBifurcationCandidate) -> str:
        """Return the generation path a candidate's start came from.

        The path is written by the refinement pass that produced the candidate
        (``<method>:<chart id>`` for a reduction, ``full`` for the bordered search).
        A candidate without one is an internal contract violation: falling back to
        the method name would merge two charts of one retained state into one start
        identity.
        """
        path = candidate.metadata.get("generation_path")
        if not path:
            raise RuntimeError(
                "candidate "
                f"{candidate.metadata.get('candidate_identity') or '<unidentified>'!r} "
                "carries no generation path; an accepted start cannot be named"
            )
        return str(path)

    def _assign_start_identities(
        self,
        candidates: list[CAMBifurcationCandidate],
        *,
        case: str,
        model: Any,
    ) -> None:
        """Give every accepted start its own identity (idempotent).

        The identity's path component is the candidate's real generation path, so two
        charts that converge to the same candidate keep two start identities.
        """
        for candidate in candidates:
            if candidate.metadata.get("accepted_start_identities"):
                continue
            content, _ = self._identity_content(model, candidate)
            candidate.metadata["accepted_start_identities"] = (
                derive_start_identity(
                    case=case,
                    path=self._generation_path(candidate),
                    content=content,
                ),
            )

    def _assign_candidate_identities(
        self, candidates: list[CAMBifurcationCandidate], *, model: Any
    ) -> None:
        """Give every surviving candidate its content identity and basis."""
        for candidate in candidates:
            content, basis = self._identity_content(model, candidate)
            candidate.metadata["candidate_identity"] = derive_candidate_identity(
                content
            )
            candidate.metadata["identity_basis"] = basis

    def _finalize_identities(
        self,
        candidates: list[CAMBifurcationCandidate],
        metadata: dict[str, Any],
        *,
        model: Any,
        swept_parameters: Sequence[str] = (),
    ) -> tuple[list[CAMBifurcationCandidate], dict[str, Any]]:
        """Attach both identity layers and expose them in the output metadata."""
        search, case = self._scope_identities(model, swept_parameters=swept_parameters)
        metadata["search_identity"] = search
        metadata["case_identity"] = case
        self._assign_start_identities(candidates, case=case, model=model)
        self._assign_candidate_identities(candidates, model=model)
        metadata["accepted_start_identities"] = tuple(
            identity
            for candidate in candidates
            for identity in candidate.metadata.get("accepted_start_identities", ())
        )
        metadata["accepted_candidate_identities"] = tuple(
            str(candidate.metadata["candidate_identity"]) for candidate in candidates
        )
        return candidates, metadata

    def _deduplicate_candidates(
        self, candidates: list[CAMBifurcationCandidate]
    ) -> list[CAMBifurcationCandidate]:
        unique: list[CAMBifurcationCandidate] = []
        for candidate in sorted(candidates, key=lambda item: item.full_residual_norm):
            vector = np.concatenate(
                (
                    np.asarray(candidate.state_vector, dtype=float).reshape(-1),
                    np.asarray(
                        [candidate.controls[name] for name in self.config.controls],
                        dtype=float,
                    ),
                )
            )
            duplicate = None
            for previous in unique:
                previous_vector = np.concatenate(
                    (
                        np.asarray(previous.state_vector, dtype=float).reshape(-1),
                        np.asarray(
                            [previous.controls[name] for name in self.config.controls],
                            dtype=float,
                        ),
                    )
                )
                if np.allclose(vector, previous_vector, rtol=1e-6, atol=1e-8):
                    duplicate = previous
                    break
            if duplicate is None:
                candidate.metadata["discovery_methods"] = (candidate.method,)
                candidate.metadata["discovery_paths"] = (
                    self._generation_path(candidate),
                )
                unique.append(candidate)
                continue
            methods = tuple(
                dict.fromkeys(
                    (
                        *duplicate.metadata.get(
                            "discovery_methods", (duplicate.method,)
                        ),
                        candidate.method,
                    )
                )
            )
            duplicate.metadata["discovery_methods"] = methods
            # Every chart that reached the survivor stays visible, so a merged
            # candidate never loses the generation paths behind it.
            paths = tuple(
                dict.fromkeys(
                    (
                        *duplicate.metadata.get(
                            "discovery_paths",
                            (self._generation_path(duplicate),),
                        ),
                        self._generation_path(candidate),
                    )
                )
            )
            duplicate.metadata["discovery_paths"] = paths
            # The merged start keeps its identity on the survivor: a duplicate
            # must never erase an accepted start from the evidence.
            merged = list(duplicate.metadata.get("accepted_start_identities", ()))
            for identity in candidate.metadata.get("accepted_start_identities", ()):
                if identity in merged:
                    raise RuntimeError(
                        "duplicate accepted-start identity "
                        f"{identity!r}: one start was counted twice"
                    )
                merged.append(identity)
            duplicate.metadata["accepted_start_identities"] = tuple(merged)
            if (
                duplicate.metadata.get("classification_status") != "classified"
                and candidate.metadata.get("classification_status") == "classified"
            ):
                for name in (
                    "classification_status",
                    "classification_accepted",
                    "coefficient_threshold",
                    "state_tangent_vector",
                    "state_tangent_matrix",
                    "scaling_signatures",
                ):
                    duplicate.metadata[name] = candidate.metadata.get(name)
        return unique

    def _control_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (control.min, control.max) for control in self.config.controls.values()
        )

    def _control_seed_axes(self) -> tuple[np.ndarray, ...]:
        count = self.discovery.config.samples_per_control
        return tuple(
            item.sample_values(count) for item in self.config.controls.values()
        )

    def _state_is_physical(
        self,
        adapter: QPSGenDynamicsAdapter,
        vector: np.ndarray,
        params: dict[str, Any],
    ) -> bool:
        try:
            eigenvalues = adapter.physical_eigenvalues(vector, params)
        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
            ZeroDivisionError,
        ):
            return False
        return bool(
            np.all(np.isfinite(eigenvalues))
            and np.min(eigenvalues) >= -self.config.state_domain.psd_tolerance
        )

    def _reduced_start_prefilter_reason(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: QPSGenDynamicsAdapter,
        start: np.ndarray,
    ) -> str | None:
        """Return the physical-prefilter rejection reason, or None if physical."""
        try:
            vector = reduction.reconstruct(start)
        except (
            np.linalg.LinAlgError,
            ValueError,
            FloatingPointError,
            ZeroDivisionError,
        ):
            return "reconstruction_error"
        controls = dict(zip(self.config.controls, start[1:], strict=True))
        if self._state_is_physical(
            adapter, vector, {**adapter.model.params, **controls}
        ):
            return None
        return "psd_violation"

    def _reduced_start_is_physical(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: QPSGenDynamicsAdapter,
        start: np.ndarray,
    ) -> bool:
        return self._reduced_start_prefilter_reason(reduction, adapter, start) is None

    @staticmethod
    def _control_scale(control: ControlRange) -> float:
        return float(control.scale or (control.max - control.min))


def _empty_candidate_output(solver: Any, model: Any) -> CAMBifurcationOutput:
    """Return an empty candidate output for a case the policy recorded as failed."""
    return CAMBifurcationOutput(
        candidates=[],
        target=solver.target.name,
        order=int(solver.target.order),
        metadata={"control_names": tuple(solver.config.controls)},
    )
