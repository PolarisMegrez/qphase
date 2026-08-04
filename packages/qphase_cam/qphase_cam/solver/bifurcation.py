"""High-order CAM equilibrium bifurcation solver plugin."""

from __future__ import annotations

from itertools import product
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qphase.core.protocols import PluginManifest, SubpluginSlot
from scipy.optimize import least_squares, root

from qphase_cam.core.bordered import BorderedMultiplicitySystem
from qphase_cam.core.coordinates import matrix_to_vector
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
        if str(backend.backend_name()).lower() != "numpy":
            raise ValueError("bifurcation solver currently requires numpy")
        adapter = FPGenDynamicsAdapter.from_model(model)
        self._validate_controls(adapter)
        reporter = context.progress if context is not None else None
        if self.strategy.mode == "full":
            candidates, metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            return CAMBifurcationOutput(
                candidates=candidates,
                target=self.target.name,
                order=self.target.order,
                metadata=metadata,
            )
        if reporter is not None:
            reporter.status("Preparing scalar reduction", stage="reduce")
        try:
            reduction = self._select_reduction(adapter)
        except BifurcationCapabilityError:
            if self.strategy.mode != "auto":
                raise
            candidates, metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            metadata["fallback_reason"] = "no_regular_scalar_reduction"
            return CAMBifurcationOutput(
                candidates=candidates,
                target=self.target.name,
                order=self.target.order,
                metadata=metadata,
            )
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
        if self.strategy.mode == "auto" and not candidates:
            full_candidates, full_metadata = self._solve_full(
                adapter, data=data, reporter=reporter
            )
            if full_candidates:
                candidates = full_candidates
                metadata = full_metadata
                metadata["fallback_reason"] = "reduced_search_found_no_candidates"
            else:
                metadata["fallback_reason"] = (
                    "reduced_search_found_no_candidates"
                )
                metadata["full_fallback"] = full_metadata
                metadata["coverage"] = (
                    "sampled_reduced_and_bordered_local_search"
                )
        return CAMBifurcationOutput(
            candidates=candidates,
            target=self.target.name,
            order=self.target.order,
            metadata=metadata,
        )

    def _solve_full(
        self,
        adapter: FPGenDynamicsAdapter,
        *,
        data: Any | None,
        reporter: Any | None,
    ) -> tuple[list[CAMBifurcationCandidate], dict[str, Any]]:
        system = BorderedMultiplicitySystem(
            adapter,
            n_state=int(adapter.model.n_modes) ** 2,
            order=self.target.order,
            control_names=tuple(self.config.controls),
            base_params=adapter.model.params,
        )
        seeds = self._upstream_seeds(system, data)
        seed_source = "upstream"
        if not seeds:
            seeds = self._discover_full_seeds(system, adapter)
            seed_source = "domain_sampling"
        candidates: list[CAMBifurcationCandidate] = []
        solved: list[np.ndarray] = []
        rejected = 0
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
        lower, upper = self._full_bounds(system)
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
                    scaled_distance(value, previous, self._full_scale(system))
                    <= 1e-6
                    for previous in solved
                ):
                    continue
                solved.append(value)
                candidate = self._full_candidate(
                    system, adapter, value, result.cost, result.success
                )
                if candidate.success:
                    candidates.append(candidate)
                else:
                    rejected += 1
        return candidates, {
            "control_names": tuple(self.config.controls),
            "strategy": "full",
            "start_count": len(seeds),
            "rejected_count": rejected,
            "coverage": f"{seed_source}_bordered_local_search",
            "fpgen": adapter.provenance(),
        }

    def _upstream_seeds(
        self, system: BorderedMultiplicitySystem, data: Any | None
    ) -> list[np.ndarray]:
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
                continue
            controls = {
                name: float(params[name]) for name in self.config.controls
            }
            if not self._controls_in_bounds(controls):
                continue
            for state, is_valid in zip(states, valid, strict=True):
                if not is_valid:
                    continue
                try:
                    seeds.append(
                        system.seed(matrix_to_vector(state), controls)
                    )
                except np.linalg.LinAlgError:
                    continue
        return seeds[: self.discovery.config.max_starts]

    def _discover_full_seeds(
        self,
        system: BorderedMultiplicitySystem,
        adapter: FPGenDynamicsAdapter,
    ) -> list[np.ndarray]:
        axes = [
            np.linspace(
                control.min,
                control.max,
                self.discovery.config.samples_per_control,
            )
            for control in self.config.controls.values()
        ]
        starts = []
        known_states: list[np.ndarray] = []
        for control_values in product(*axes):
            controls = dict(
                zip(self.config.controls, control_values, strict=True)
            )
            params = {**adapter.model.params, **controls}
            for guess in self._state_guesses(adapter.model):
                solution = root(
                    lambda value, params=params: adapter.rhs(value, params),
                    guess,
                    jac=lambda value, params=params: adapter.jacobian(
                        value, params
                    ),
                    options={"maxfev": self.config.refinement.max_iterations},
                )
                state = np.asarray(solution.x)
                if not solution.success or np.linalg.norm(
                    adapter.rhs(state, params)
                ) > 1e-7:
                    continue
                if any(np.linalg.norm(state - item) < 1e-7 for item in known_states):
                    continue
                known_states.append(state)
                try:
                    starts.append(system.seed(state, controls))
                except np.linalg.LinAlgError:
                    continue
                if len(starts) >= self.discovery.config.max_starts:
                    return starts
        return starts

    def _full_candidate(
        self,
        system: BorderedMultiplicitySystem,
        adapter: FPGenDynamicsAdapter,
        value: np.ndarray,
        cost: float,
        optimizer_success: bool,
        verification: Any | None = None,
    ) -> CAMBifurcationCandidate:
        if verification is not None:
            value = verification.value
        state, control_values, right, left = system.unpack(value)
        controls = dict(
            zip(self.config.controls, control_values, strict=True)
        )
        params = {**adapter.model.params, **controls}
        diagnostics = system.diagnostics(value)
        residual = system.residual(value)
        full_residual = adapter.rhs(state, params)
        matrix = self._vector_to_matrix(state, int(adapter.model.n_modes))
        physical_eigenvalues = np.linalg.eigvalsh(matrix)
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
        is_physical = bool(np.min(physical_eigenvalues) >= -threshold)
        success = bool(
            optimizer_success
            and (verification is None or verification.success)
            and np.linalg.norm(residual) <= threshold
            and np.linalg.norm(full_residual) <= threshold
            and abs(diagnostics.coefficients[self.target.order])
            > singular_tolerance
            and simple_zero
            and is_physical
        )
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
                "verification_residual_norm": (
                    verification.residual_norm
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
                "is_stable": bool(
                    np.max(np.real(jacobian_eigenvalues)) <= threshold
                ),
                "minimum_physical_eigenvalue": float(
                    np.min(physical_eigenvalues)
                ),
            },
        )

    def _full_bounds(
        self, system: BorderedMultiplicitySystem
    ) -> tuple[np.ndarray, np.ndarray]:
        n = system.n_state
        lower_state = np.full(n, -np.inf)
        lower_state[: int(np.sqrt(n))] = 0.0
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

    def _full_scale(self, system: BorderedMultiplicitySystem) -> np.ndarray:
        n = system.n_state
        nonlinear = max(
            (
                abs(float(system.base_params[name]))
                for name in ("chi", "Gamma")
                if name in system.base_params
            ),
            default=1.0,
        )
        state_scale = max(1.0, 1.0 / max(nonlinear, 1e-8))
        return np.asarray(
            [
                *([state_scale] * n),
                *(
                    self._control_scale(control)
                    for control in self.config.controls.values()
                ),
                *([1.0] * (2 * n)),
            ]
        )

    def _state_guesses(self, model: Any) -> list[np.ndarray]:
        n_modes = int(model.n_modes)
        n_state = n_modes**2
        nonlinear = max(
            (
                abs(float(model.params[name]))
                for name in ("chi", "Gamma")
                if name in model.params
            ),
            default=1.0,
        )
        scale = max(1.0, 1.0 / max(nonlinear, 1e-8))
        guesses = [np.zeros(n_state)]
        for value in (0.5, 1.0, scale):
            guess = np.zeros(n_state)
            guess[:n_modes] = value
            guesses.append(guess)
        return guesses

    def _controls_in_bounds(self, controls: dict[str, float]) -> bool:
        return all(
            self.config.controls[name].min
            <= value
            <= self.config.controls[name].max
            for name, value in controls.items()
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
        verification: Any | None = None,
    ) -> CAMBifurcationCandidate:
        if verification is not None:
            value = verification.value
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
            and (verification is None or verification.success)
            and finite_vector(vector)
            and search_norm <= threshold
            and full_norm <= threshold
            and next_coefficient > singular_tolerance
            and regular
            and is_physical
        )
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
        return CAMBifurcationCandidate(
            state_vector=vector,
            controls={name: float(item) for name, item in controls.items()},
            full_residual_norm=full_norm,
            search_residual_norm=search_norm,
            success=success,
            status="verified" if success else "rejected",
            method=reduction.method,
            metadata={
                "optimizer_cost": float(cost),
                "verification_digits": (
                    verification.digits if verification is not None else 0
                ),
                "verification_residual_norm": (
                    verification.residual_norm
                    if verification is not None
                    else np.nan
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
