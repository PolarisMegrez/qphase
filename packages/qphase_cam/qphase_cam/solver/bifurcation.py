"""High-order CAM equilibrium bifurcation solver plugin."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.core.protocols import PluginManifest, SubpluginSlot
from scipy.optimize import least_squares

from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.core.reduction import (
    CondensedScalarReduction,
    FractionFreeScalarReduction,
    finite_vector,
    scaled_distance,
)
from qphase_cam.errors import BifurcationCapabilityError
from qphase_cam.state import (
    CAMBifurcationCandidate,
    CAMBifurcationOutput,
)

from .base import CAMSolver, CAMSolverConfig
from .bifurcation_discovery import BifurcationDiscovery
from .bifurcation_strategy import BifurcationStrategy
from .bifurcation_target import BifurcationTarget


class ControlRange(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: float
    max: float
    scale: float | None = Field(None, gt=0.0)

    @model_validator(mode="after")
    def validate_bounds(self) -> ControlRange:
        if self.max <= self.min:
            raise ValueError("control max must be greater than min")
        return self


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


class BifurcationSolverConfig(CAMSolverConfig):
    controls: dict[str, ControlRange]
    refinement: RefinementConfig = Field(default_factory=RefinementConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)


class BifurcationSolver(CAMSolver):
    """Search model parameters for high-order CAM equilibrium roots."""

    name: ClassVar[str] = "bifurcation"
    description: ClassVar[str] = "High-order CAM equilibrium bifurcation search"
    config_schema: ClassVar[type[BifurcationSolverConfig]] = BifurcationSolverConfig
    output_kind: ClassVar[str] = "bifurcation_candidates"
    manifest: ClassVar[PluginManifest] = PluginManifest(
        subplugins={
            "target": SubpluginSlot(
                namespace="bifurcation_target",
                protocol=(
                    "qphase_cam.solver.bifurcation_target:BifurcationTarget"
                ),
                allowed=frozenset({"equilibrium_multiplicity"}),
            ),
            "strategy": SubpluginSlot(
                namespace="bifurcation_strategy",
                default="auto",
                protocol=(
                    "qphase_cam.solver.bifurcation_strategy:BifurcationStrategy"
                ),
                allowed=frozenset({"auto", "reduced", "full"}),
            ),
            "discovery": SubpluginSlot(
                namespace="bifurcation_discovery",
                default="seeds",
                protocol=(
                    "qphase_cam.solver.bifurcation_discovery:BifurcationDiscovery"
                ),
                allowed=frozenset({"seeds", "continuation"}),
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
        if len(self.config.controls) != self.target.order - 1:
            raise ValueError(
                "equilibrium multiplicity order m requires exactly m-1 controls"
            )

    def solve(
        self,
        model: Any,
        backend: Any,
        *,
        data: Any | None = None,
        context: Any | None = None,
    ) -> CAMBifurcationOutput:
        del data
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("bifurcation solver currently requires numpy")
        if self.strategy.mode == "full":
            raise BifurcationCapabilityError(
                "full bordered-system strategy is not available yet"
            )
        adapter = FPGenDynamicsAdapter.from_model(model)
        self._validate_controls(adapter)
        reporter = context.progress if context is not None else None
        if reporter is not None:
            reporter.status("Preparing scalar reduction", stage="reduce")
        reduction = self._select_reduction(adapter)
        starts = reduction.initial_starts(
            self._control_bounds(),
            samples_per_control=self.discovery.config.samples_per_control,
            max_starts=self.discovery.config.max_starts,
            order_parameter_bounds=self.strategy.config.order_parameter_bounds,
            order_parameter_samples=(
                self.discovery.config.order_parameter_samples
            ),
        )
        if reporter is not None:
            with reporter.stage(
                "refine",
                total=len(starts),
                unit="candidate",
                message=f"Refining {len(starts)} reduced candidates",
            ):
                candidates, rejected = self._refine(
                    reduction, adapter, starts, reporter=reporter
                )
        else:
            candidates, rejected = self._refine(
                reduction, adapter, starts, reporter=None
            )
        metadata = {
            "control_names": tuple(self.config.controls),
            "strategy": self.strategy.mode,
            "reduction_state_id": reduction.retained_id,
            "reduced_degree": reduction.degree,
            "start_count": len(starts),
            "rejected_count": rejected,
            "coverage": "sampled_reduced_regular_branch",
            "fpgen": adapter.provenance(),
        }
        return CAMBifurcationOutput(
            candidates=candidates,
            target=self.target.name,
            order=self.target.order,
            metadata=metadata,
        )

    def _validate_controls(self, adapter: FPGenDynamicsAdapter) -> None:
        unknown = set(self.config.controls) - set(adapter.parameter_names)
        if unknown:
            raise ValueError(
                f"unknown bifurcation controls for {adapter.model.name}: "
                f"{sorted(unknown)}"
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

    def _select_reduction(
        self, adapter: FPGenDynamicsAdapter
    ) -> FractionFreeScalarReduction | CondensedScalarReduction:
        strategy_config = self.strategy.config
        candidates = adapter.dynamics.find_linear_reductions(
            retained_dimension=1,
            max_candidates=strategy_config.max_candidates,
            equation_partitions="all",
        )
        if strategy_config.order_parameter is not None:
            candidates = tuple(
                candidate
                for candidate in candidates
                if candidate.retained_ids == (strategy_config.order_parameter,)
            )
        errors = []
        for candidate in candidates:
            if len(candidate.eliminated_indices) > 3:
                return CondensedScalarReduction(
                    adapter.dynamics.linear_reduce(candidate=candidate),
                    order=self.target.order,
                    control_names=tuple(self.config.controls),
                    base_params=adapter.model.params,
                )
            if (
                candidate.reduced_degree is not None
                and candidate.reduced_degree < self.target.order
            ):
                continue
            try:
                materialized = adapter.dynamics.linear_reduce(
                    candidate=candidate
                ).materialize()
                reduction = FractionFreeScalarReduction(
                    materialized,
                    order=self.target.order,
                    control_names=tuple(self.config.controls),
                    base_params=adapter.model.params,
                )
            except Exception as exc:
                errors.append(f"{candidate.retained_ids}: {exc}")
                continue
            if reduction.degree >= self.target.order:
                return reduction
        detail = "; ".join(errors[:3])
        message = "no regular scalar fraction-free reduction is available"
        if detail:
            message = f"{message}: {detail}"
        if self.strategy.mode == "auto":
            message += "; full-system fallback is required"
        raise BifurcationCapabilityError(message)

    def _refine(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: FPGenDynamicsAdapter,
        starts: list[np.ndarray],
        *,
        reporter: Any | None,
    ) -> tuple[list[CAMBifurcationCandidate], int]:
        if not starts:
            return [], 0
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
        rejected = 0
        for start in starts:
            initial = np.minimum(np.maximum(start, lower), upper)
            result = least_squares(
                lambda value: reduction.equations(value) / equation_scale,
                initial,
                jac=lambda value: reduction.jacobian(value)
                / equation_scale[:, None],
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
                continue
            solved.append(value)
            candidate = self._candidate(
                reduction, adapter, value, result.cost, result.success
            )
            if candidate.success:
                accepted.append(candidate)
            else:
                rejected += 1
        return accepted, rejected

    def _candidate(
        self,
        reduction: FractionFreeScalarReduction | CondensedScalarReduction,
        adapter: FPGenDynamicsAdapter,
        value: np.ndarray,
        cost: float,
        optimizer_success: bool,
    ) -> CAMBifurcationCandidate:
        controls = dict(
            zip(self.config.controls, value[1:], strict=True)
        )
        params = {**adapter.model.params, **controls}
        vector = reduction.reconstruct(value)
        equations = reduction.equations(value)
        diagnostics = reduction.diagnostics(value)
        full_residual = adapter.rhs(vector, params)
        jacobian = adapter.jacobian(vector, params)
        matrix = self._vector_to_matrix(vector, int(adapter.model.n_modes))
        physical_eigenvalues = np.linalg.eigvalsh(matrix)
        jacobian_eigenvalues = np.linalg.eigvals(jacobian)
        search_norm = float(np.linalg.norm(equations))
        full_norm = float(np.linalg.norm(full_residual))
        threshold = max(np.sqrt(self.config.refinement.tolerance), 1e-9)
        singular_tolerance = self.strategy.config.singular_tolerance
        next_coefficient = abs(
            float(diagnostics.reduced_coefficients[self.target.order])
        )
        is_physical = bool(np.min(physical_eigenvalues) >= -threshold)
        regular = bool(
            abs(diagnostics.regularity_determinant) > singular_tolerance
            and diagnostics.denominator_margin > singular_tolerance
            and diagnostics.condition_number <= self.strategy.config.condition_limit
        )
        success = bool(
            optimizer_success
            and finite_vector(vector)
            and search_norm <= threshold
            and full_norm <= threshold
            and next_coefficient > singular_tolerance
            and regular
            and is_physical
        )
        return CAMBifurcationCandidate(
            state_vector=vector,
            controls={name: float(item) for name, item in controls.items()},
            full_residual_norm=full_norm,
            search_residual_norm=search_norm,
            success=success,
            status="candidate" if success else "rejected",
            method=reduction.method,
            metadata={
                "optimizer_cost": float(cost),
                "regularity_determinant": diagnostics.regularity_determinant,
                "denominator_margin": diagnostics.denominator_margin,
                "reduction_condition_number": diagnostics.condition_number,
                "reduced_coefficients": diagnostics.reduced_coefficients,
                "jacobian_singular_values": np.linalg.svd(
                    jacobian, compute_uv=False
                ),
                "is_physical": is_physical,
                "is_stable": bool(
                    np.max(np.real(jacobian_eigenvalues)) <= threshold
                ),
                "minimum_physical_eigenvalue": float(
                    np.min(physical_eigenvalues)
                ),
            },
        )

    def _control_bounds(self) -> tuple[tuple[float, float], ...]:
        return tuple(
            (control.min, control.max)
            for control in self.config.controls.values()
        )

    @staticmethod
    def _control_scale(control: ControlRange) -> float:
        return float(control.scale or (control.max - control.min))

    @staticmethod
    def _vector_to_matrix(vector: np.ndarray, n_modes: int) -> np.ndarray:
        from qphase_cam.core.coordinates import vector_to_matrix

        return np.asarray(vector_to_matrix(vector, n_modes))
