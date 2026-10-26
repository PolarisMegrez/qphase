"""High-precision local response validation for CAM bifurcation branches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from qphase_cam.branch_diagnostics import (
    BranchDiagnostics,
    BranchDiagnosticsConfig,
    branch_diagnostics,
)
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
from qphase_cam.outcomes import CAMOutcomeStatus

from ..solver.high_precision import (
    HighPrecisionNewtonConfig,
    solve_high_precision,
)
from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .candidates import (
    CandidateState,
    CandidateView,
    candidate_parameters,
    candidate_state,
    candidate_view,
    pivot_branch_rows,
)
from .petermann import eigensystem_petermann


class LocalResponseValidationConfig(CAMPostprocessorConfig):
    epsilon_min: float = Field(default=1e-12, gt=0.0)
    epsilon_max: float = Field(default=1e-4, gt=0.0)
    epsilon_points: int = Field(default=9, ge=3, le=100)
    precision_digits: int = Field(default=80, ge=30, le=500)
    max_iterations: int = Field(default=40, ge=1, le=500)
    residual_tolerance: float = Field(default=1e-30, gt=0.0)
    fit_points: int = Field(default=4, ge=3)
    psd_tolerance: float = Field(default=1e-9, ge=0.0)
    stability_tolerance: float = Field(default=1e-9, ge=0.0)
    continuity_ratio_limit: float = Field(default=10.0, gt=1.0)
    diagnostic_observables: tuple[Any, ...] = Field(
        default=(),
        exclude=True,
        description=(
            "Optional CAM observables to diagnose on every solved branch point. Each "
            "one is evaluated on the state the branch solve already produced, so the "
            "default (empty) keeps the product, its cost and its numbers unchanged, "
            "and a non-empty tuple adds one readout axis of diagnostic fields."
        ),
    )
    diagnostic_config: BranchDiagnosticsConfig = Field(
        default_factory=BranchDiagnosticsConfig,
        description="Steps and floors of the per-observable branch diagnostics",
    )

    @model_validator(mode="after")
    def validate_interval(self) -> LocalResponseValidationConfig:
        if self.epsilon_max <= self.epsilon_min:
            raise ValueError("epsilon_max must exceed epsilon_min")
        if self.fit_points > self.epsilon_points:
            raise ValueError("fit_points cannot exceed epsilon_points")
        names = []
        for item in self.diagnostic_observables:
            identity = getattr(item, "identity", None)
            names.append(str(getattr(identity, "name", "") or ""))
        if any(not name for name in names):
            raise ValueError("every diagnostic observable needs a named identity")
        if len(set(names)) != len(names):
            raise ValueError(
                f"diagnostic observable names must be unique; got {names}"
            )
        return self


#: Branch-diagnostic fields the annotation carries, in a stable order. Every one is a
#: scalar per (candidate, branch, epsilon_sample, readout); ``None`` (an undefined
#: quantity) is padded as NaN by the pivot, never as a number.
DIAGNOSTIC_FIELDS = (
    "diagnostic_status",
    "diagnostic_reason_code",
    "diagnostic_reason",
    "soft_eigenvalue_real",
    "soft_eigenvalue_imag",
    "soft_rate",
    "noncritical_spectral_gap",
    "critical_mode_condition_number",
    "eigenvector_condition_number",
    "projected_noise_intensity",
    "local_ou_sigma_x",
    "soft_coordinate",
    "rho_x",
    "eta2",
    "minimum_physical_eigenvalue",
    "frequency",
    "frequency_soft_projection",
    "frequency_curvature",
    "eta_readout",
    "soft_ou_intensity",
    "full_ou_intensity",
    "soft_response",
    "explicit_response",
    "full_response",
    "regular_response",
    "ou_snr_coefficient",
)


def _diagnostic_fields(diagnostic: BranchDiagnostics) -> dict[str, Any]:
    """Return one diagnostic as annotation fields, ``None`` kept as "undefined"."""
    fields: dict[str, Any] = {
        "diagnostic_status": diagnostic.status.value,
        "diagnostic_reason_code": diagnostic.reason_code or "",
        "diagnostic_reason": diagnostic.reason or "",
    }
    for name in DIAGNOSTIC_FIELDS[3:]:
        fields[name] = getattr(diagnostic, name)
    return fields


def _undefined_diagnostic_fields() -> dict[str, Any]:
    """Return an empty diagnostic slot: no status, every field undefined.

    A branch point that did not converge is not diagnosed, but its slot in the
    readout axis is kept so the sample axis stays aligned with the base fields.
    """
    fields: dict[str, Any] = {
        "diagnostic_status": "",
        "diagnostic_reason_code": "",
        "diagnostic_reason": "",
    }
    for name in DIAGNOSTIC_FIELDS[3:]:
        fields[name] = None
    return fields


class LocalResponseValidation(CAMPostprocessor[LocalResponseValidationConfig]):
    """Validate classified local branches against the complete CAM residual."""

    name: ClassVar[str] = "local_response_validation"
    description: ClassVar[str] = (
        "Solve complete CAM branches and measure state/Rayleigh response exponents"
    )
    config_schema: ClassVar[type[LocalResponseValidationConfig]] = (
        LocalResponseValidationConfig
    )
    annotation_kind: ClassVar[str] = "local_response_validation"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("candidates",)
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        model = context.model
        adapter = QPSGenDynamicsAdapter.from_model(model)
        view = candidate_view(product, metadata=dict(context.metadata))
        candidates = view
        branches = candidates.branches
        if branches.size == 0:
            self.result_metadata = {"status": "no_real_branches", "sample_count": 0}
            return self._annotation(product, view, [], status=self._status(view, []))

        magnitudes = np.geomspace(
            self.config.epsilon_min,
            self.config.epsilon_max,
            self.config.epsilon_points,
        )
        observables = tuple(self.config.diagnostic_observables)
        observable_names = tuple(
            str(item.identity.name) for item in observables
        )
        rows: list[dict[str, Any]] = []
        diagnostic_rows: list[dict[str, Any]] = []
        for branch_index in range(branches.size):
            if not bool(branches.real_branch[branch_index]):
                continue
            candidate_index = int(branches.candidate_index[branch_index])
            params = self._critical_params(view, candidate_index, model)
            design = candidates.design_at(candidate_index)
            if not design.is_resolved:
                raise ValueError(
                    "the case of candidate "
                    f"{candidates.rows[candidate_index].identity!r} records no "
                    "perturbation parameter; the local-response branch cannot be "
                    "probed along an axis it does not name"
                )
            perturbation = design.parameter
            scale = design.scale
            critical_candidate = candidate_state(candidates, candidate_index)
            critical_decimals = critical_candidate.decimals
            critical = critical_candidate.vector
            exponent = float(
                branches.exponent_numerator[branch_index]
                / branches.exponent_denominator[branch_index]
            )
            tangent = np.asarray(
                candidates.diagnostics["state_tangent_vector"][candidate_index],
                dtype=float,
            )
            coefficient = float(branches.amplitude[branch_index]) * tangent
            side = int(branches.epsilon_side[branch_index])
            critical_rayleigh = self._rayleigh(adapter, model, critical, params)
            critical_petermann = self._hamiltonian_petermann_max(
                adapter, model, critical, params
            )
            visibility = self._rayleigh_visibility(
                adapter, model, critical, params, tangent
            )
            branch_rows = []
            for magnitude in magnitudes:
                epsilon = side * float(magnitude)
                initial = critical + coefficient * magnitude**exponent
                local_params = dict(params)
                local_params[perturbation] = params[perturbation] + scale * epsilon
                state, residual, converged = self._newton(
                    adapter,
                    initial,
                    local_params,
                    critical_decimals=critical_decimals,
                    leading_coefficient=coefficient,
                    epsilon_magnitude=magnitude,
                    exponent=exponent,
                )
                expected_norm = max(
                    np.linalg.norm(coefficient) * magnitude**exponent, 1e-300
                )
                delta_norm = float(np.linalg.norm(state - critical))
                ratio = delta_norm / expected_norm
                continuous = bool(
                    converged
                    and 1.0 / self.config.continuity_ratio_limit
                    <= ratio
                    <= self.config.continuity_ratio_limit
                )
                physical = adapter.physical_eigenvalues(state, local_params)
                jacobian = np.linalg.eigvals(adapter.jacobian(state, local_params))
                rayleigh = self._rayleigh(adapter, model, state, local_params)
                petermann = self._hamiltonian_petermann_max(
                    adapter, model, state, local_params
                )
                if observables:
                    # One diagnostic per observable on the *same* solved state: the
                    # branch is continued once and every readout reuses it. A branch
                    # point that did not converge is still placed on its sample slot —
                    # with no status and no fields — so the readout axis stays aligned
                    # with the base sample axis instead of shifting later samples.
                    for observable in observables:
                        if converged:
                            fields = _diagnostic_fields(
                                branch_diagnostics(
                                    adapter,
                                    model,
                                    observable,
                                    state=state,
                                    params=local_params,
                                    parameter=perturbation,
                                    scale=scale,
                                    critical_state=critical,
                                    config=self.config.diagnostic_config,
                                )
                            )
                        else:
                            fields = _undefined_diagnostic_fields()
                        diagnostic_rows.append(
                            {
                                "candidate_index": candidate_index,
                                "branch_index": branch_index,
                                "readout": str(observable.identity.name),
                                **fields,
                            }
                        )
                branch_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "branch_index": branch_index,
                        "epsilon_side": side,
                        "epsilon": epsilon,
                        "abs_epsilon": magnitude,
                        "asymptotic_exponent": exponent,
                        "converged": converged,
                        "continuous": continuous,
                        "full_residual_norm": residual,
                        "is_physical": bool(
                            np.min(physical) >= -self.config.psd_tolerance
                        ),
                        "is_stable": bool(
                            np.max(np.real(jacobian)) <= self.config.stability_tolerance
                        ),
                        "minimum_physical_eigenvalue": float(np.min(physical)),
                        "maximum_jacobian_real_part": float(np.max(np.real(jacobian))),
                        "delta_state_norm": delta_norm,
                        "state_effective_exponent": np.nan,
                        "rayleigh_frequency": rayleigh,
                        "delta_rayleigh_frequency": rayleigh - critical_rayleigh,
                        "rayleigh_effective_exponent": np.nan,
                        "state_fit_exponent": np.nan,
                        "rayleigh_fit_exponent": np.nan,
                        "rayleigh_visibility": visibility,
                        "rayleigh_projection_status": (
                            "weak_projection" if visibility < 1e-3 else "resolved"
                        ),
                        "critical_hamiltonian_petermann_max": critical_petermann,
                        "hamiltonian_petermann_max": petermann,
                        "hamiltonian_petermann_effective_exponent": np.nan,
                        "hamiltonian_petermann_fit_exponent": np.nan,
                    }
                )
            self._add_exponents(branch_rows)
            self._add_petermann_exponents(branch_rows)
            rows.extend(branch_rows)

        output = self._columns(rows)
        designs = {
            (design.parameter, design.scale, design.side): design
            for design in (
                candidates.design_at(index)
                for index in range(len(candidates.rows))
            )
            if design.is_resolved
        }
        self.result_metadata = {
            "status": "complete" if rows else "no_real_branches",
            "sample_count": len(rows),
            "precision_digits": self.config.precision_digits,
            "perturbation_designs": [
                {"parameter": item.parameter, "scale": item.scale, "side": item.side}
                for _, item in sorted(designs.items())
            ],
        }
        del output
        return self._annotation(
            product,
            view,
            rows,
            diagnostic_rows=diagnostic_rows,
            observable_names=observable_names,
            status=self._status(view, rows),
        )

    def _status(self, view: CandidateView, rows: list[dict[str, Any]]) -> list[str]:
        """Return one outcome status per candidate."""
        statuses: list[str] = []
        for row in view.rows:
            sample = [
                item for item in rows if int(item["candidate_index"]) == row.index
            ]
            if not view.branches_of(row.index):
                statuses.append(CAMOutcomeStatus.NOT_FOUND.value)
            elif any(bool(item["converged"]) for item in sample):
                statuses.append(CAMOutcomeStatus.VALID.value)
            else:
                statuses.append(CAMOutcomeStatus.NOT_CONVERGED.value)
        return statuses

    def _annotation(
        self,
        product: Any,
        view: CandidateView,
        rows: list[dict[str, Any]],
        *,
        diagnostic_rows: list[dict[str, Any]] | None = None,
        observable_names: tuple[str, ...] = (),
        status: list[str],
    ) -> Any:
        """Return the typed local-response annotation over the candidate rows.

        The base fields are the per-(candidate, branch, epsilon_sample) trace
        quantities. When diagnostic observables were configured, their fields are
        added on a ``readout`` axis — the same branch solve, one entry per readout —
        with the observable identities recorded in the method metadata so a stored
        annotation states which readout each coordinate is.
        """
        del product
        fields, field_axes, axes = pivot_branch_rows(
            view, rows, sample_axis="epsilon_sample"
        )
        resolved = list(diagnostic_rows or [])
        if resolved:
            diagnostic_fields, diagnostic_axes, diagnostic_sizes = pivot_branch_rows(
                view,
                resolved,
                sample_axis="epsilon_sample",
                axis_values={"readout": observable_names},
            )
            # The base fields already carry the row anchors and the physical
            # eigenvalue per (candidate, branch, sample); the diagnostic rows repeat
            # them for every readout, so those duplicates are dropped instead of
            # shadowing the base column.
            for name in set(fields) & set(diagnostic_fields):
                diagnostic_fields.pop(name)
                diagnostic_axes.pop(name, None)
            fields = {**fields, **diagnostic_fields}
            field_axes = {**field_axes, **diagnostic_axes}
            axes = {**axes, **diagnostic_sizes}
        identities = [
            {
                "name": str(item.identity.name),
                "kind": item.identity.kind.value,
                "sector": item.identity.sector.value,
                "measurement_mapping": item.identity.measurement_mapping,
            }
            for item in self.config.diagnostic_observables
        ]
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind="candidate",
            target_identity=[row.identity for row in view.rows],
            case_identity=[row.case_identity for row in view.rows],
            status=status,
            fields=fields,
            field_axes=field_axes,
            axes=axes,
            method_metadata=[
                {
                    **self.result_metadata,
                    "perturbation_parameter": view.design_at(row.index).parameter,
                    "perturbation_scale": view.design_at(row.index).scale,
                    "perturbation_side": view.design_at(row.index).side,
                    "diagnostic_observables": identities,
                    "diagnostic_config": self.config.diagnostic_config.model_dump(
                        mode="json"
                    ),
                }
                for row in view.rows
            ],
        )

    def _newton(
        self,
        adapter: QPSGenDynamicsAdapter,
        initial: np.ndarray,
        params: dict[str, Any],
        *,
        critical_decimals: tuple[str, ...] | None = None,
        leading_coefficient: np.ndarray | None = None,
        epsilon_magnitude: float | None = None,
        exponent: float | None = None,
    ) -> tuple[np.ndarray, float, bool]:
        """Solve one branch point through the public high-precision service.

        Kept as the postprocessor's internal entry point; the implementation and
        its numerics live in :mod:`qphase_cam.solver.high_precision`.
        """
        solution = solve_high_precision(
            adapter,
            initial,
            params,
            config=self._newton_config(),
            critical_decimals=critical_decimals,
            leading_coefficient=leading_coefficient,
            epsilon_magnitude=epsilon_magnitude,
            exponent=exponent,
        )
        return solution.state, solution.residual, solution.converged

    def _newton_config(self) -> HighPrecisionNewtonConfig:
        """Return the solver configuration this plugin's config describes."""
        return HighPrecisionNewtonConfig(
            precision_digits=self.config.precision_digits,
            residual_tolerance=self.config.residual_tolerance,
            max_iterations=self.config.max_iterations,
        )

    def _critical_params(
        self, view: CandidateView, candidate_index: int, model: Any
    ) -> dict[str, Any]:
        """Return the model parameters of one candidate of the typed candidate view.

        The view resolves the candidate's own case, so a candidate index is never
        read as if it were a case index.
        """
        return candidate_parameters(view, candidate_index, model)

    @staticmethod
    def _critical_state_decimals(candidates: Any, index: int) -> tuple[str, ...] | None:
        """Return the candidate's verified decimal expansion, if it has one."""
        return candidate_state(candidates, index).decimals

    @staticmethod
    def _critical_state(
        candidates: Any,
        index: int,
        decimals: tuple[str, ...] | None,
    ) -> np.ndarray:
        """Return the state one branch starts from.

        ``decimals`` is honoured when given; otherwise the candidate's own state is
        read through the shared candidate-state service.
        """
        state: CandidateState = candidate_state(candidates, index)
        if decimals is not None:
            return np.asarray([float(item) for item in decimals], dtype=float)
        return state.chart_vector

    @staticmethod
    def _rayleigh(
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        state: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        matrix = adapter.state_matrix(state, params)
        denominator = np.trace(matrix)
        if abs(denominator) < 1e-300:
            return np.nan
        hamiltonian = np.asarray(model.cam_hamiltonian(matrix, params))
        return float(np.real(np.trace(hamiltonian @ matrix) / denominator))

    def _rayleigh_visibility(
        self,
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        state: np.ndarray,
        params: dict[str, Any],
        tangent: np.ndarray,
    ) -> float:
        gradient = np.empty_like(state)
        for index in range(len(state)):
            step = 1e-5 * max(1.0, abs(state[index]))
            offset = np.zeros_like(state)
            offset[index] = step
            gradient[index] = (
                self._rayleigh(adapter, model, state + offset, params)
                - self._rayleigh(adapter, model, state - offset, params)
            ) / (2.0 * step)
        denominator = np.linalg.norm(gradient) * np.linalg.norm(tangent)
        if denominator == 0.0:
            return 0.0
        return float(abs(np.dot(gradient, tangent)) / denominator)

    @staticmethod
    def _hamiltonian_petermann_max(
        adapter: QPSGenDynamicsAdapter,
        model: Any,
        state: np.ndarray,
        params: dict[str, Any],
    ) -> float:
        matrix = adapter.state_matrix(state, params)
        hamiltonian = np.asarray(model.cam_hamiltonian(matrix, params))
        _, factors = eigensystem_petermann(hamiltonian)
        return float(np.max(factors))

    def _add_exponents(self, rows: list[dict[str, Any]]) -> None:
        for name, output_name in (
            ("delta_state_norm", "state_effective_exponent"),
            ("delta_rayleigh_frequency", "rayleigh_effective_exponent"),
        ):
            values = np.abs([float(row[name]) for row in rows])
            epsilon = np.asarray([row["abs_epsilon"] for row in rows])
            validated = np.asarray(
                [row["converged"] and row["continuous"] for row in rows]
            )
            valid = np.isfinite(values) & (values > 0.0) & validated
            indices = np.flatnonzero(valid)
            for left, right in zip(indices[:-1], indices[1:], strict=True):
                rows[right][output_name] = float(
                    np.log(values[right] / values[left])
                    / np.log(epsilon[right] / epsilon[left])
                )
            fit = indices[: self.config.fit_points]
            exponent = (
                float(np.polyfit(np.log(epsilon[fit]), np.log(values[fit]), 1)[0])
                if len(fit) >= 3
                else np.nan
            )
            fit_name = (
                "state_fit_exponent"
                if name == "delta_state_norm"
                else "rayleigh_fit_exponent"
            )
            for row in rows:
                row[fit_name] = exponent

    def _add_petermann_exponents(self, rows: list[dict[str, Any]]) -> None:
        values = np.asarray([float(row["hamiltonian_petermann_max"]) for row in rows])
        epsilon = np.asarray([float(row["abs_epsilon"]) for row in rows])
        valid = (
            np.isfinite(values)
            & (values > 0.0)
            & np.asarray([row["converged"] and row["continuous"] for row in rows])
        )
        indices = np.flatnonzero(valid)
        for left, right in zip(indices[:-1], indices[1:], strict=True):
            rows[right]["hamiltonian_petermann_effective_exponent"] = float(
                -np.log(values[right] / values[left])
                / np.log(epsilon[right] / epsilon[left])
            )
        fit = indices[: self.config.fit_points]
        exponent = (
            float(-np.polyfit(np.log(epsilon[fit]), np.log(values[fit]), 1)[0])
            if len(fit) >= 3
            else np.nan
        )
        for row in rows:
            row["hamiltonian_petermann_fit_exponent"] = exponent

    @staticmethod
    def _columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not rows:
            return LocalResponseValidation._empty()
        names = tuple(rows[0])
        return {name: np.asarray([row[name] for row in rows]) for name in names}

    @staticmethod
    def _summary(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not rows:
            return LocalResponseValidation._empty_summary()
        output = []
        for candidate_index in sorted({int(row["candidate_index"]) for row in rows}):
            selected = [
                row for row in rows if int(row["candidate_index"]) == candidate_index
            ]
            sample_count = len(selected)
            converged = sum(bool(row["converged"]) for row in selected)
            continuous = sum(bool(row["continuous"]) for row in selected)
            physical = sum(bool(row["is_physical"]) for row in selected)
            stable = sum(bool(row["is_stable"]) for row in selected)
            state_fits = np.asarray(
                [float(row["state_fit_exponent"]) for row in selected], dtype=float
            )
            rayleigh_fits = np.asarray(
                [float(row["rayleigh_fit_exponent"]) for row in selected],
                dtype=float,
            )
            state_fit_min, state_fit_max = LocalResponseValidation._finite_range(
                state_fits
            )
            rayleigh_fit_min, rayleigh_fit_max = LocalResponseValidation._finite_range(
                rayleigh_fits
            )
            petermann_fits = np.asarray(
                [float(row["hamiltonian_petermann_fit_exponent"]) for row in selected],
                dtype=float,
            )
            petermann_fit_min, petermann_fit_max = (
                LocalResponseValidation._finite_range(petermann_fits)
            )
            output.append(
                {
                    "candidate_index": candidate_index,
                    "branch_count": len({int(row["branch_index"]) for row in selected}),
                    "sample_count": sample_count,
                    "converged_count": converged,
                    "continuous_count": continuous,
                    "physical_count": physical,
                    "stable_count": stable,
                    "converged_fraction": converged / sample_count,
                    "continuous_fraction": continuous / sample_count,
                    "physical_fraction": physical / sample_count,
                    "stable_fraction": stable / sample_count,
                    "all_converged": converged == sample_count,
                    "all_continuous": continuous == sample_count,
                    "all_physical": physical == sample_count,
                    "all_stable": stable == sample_count,
                    "minimum_physical_eigenvalue": min(
                        float(row["minimum_physical_eigenvalue"]) for row in selected
                    ),
                    "maximum_jacobian_real_part": max(
                        float(row["maximum_jacobian_real_part"]) for row in selected
                    ),
                    "minimum_state_fit_exponent": state_fit_min,
                    "maximum_state_fit_exponent": state_fit_max,
                    "minimum_rayleigh_fit_exponent": rayleigh_fit_min,
                    "maximum_rayleigh_fit_exponent": rayleigh_fit_max,
                    "minimum_rayleigh_visibility": min(
                        float(row["rayleigh_visibility"]) for row in selected
                    ),
                    "minimum_hamiltonian_petermann_max": min(
                        float(row["hamiltonian_petermann_max"]) for row in selected
                    ),
                    "maximum_hamiltonian_petermann_max": max(
                        float(row["hamiltonian_petermann_max"]) for row in selected
                    ),
                    "minimum_hamiltonian_petermann_fit_exponent": petermann_fit_min,
                    "maximum_hamiltonian_petermann_fit_exponent": petermann_fit_max,
                }
            )
        names = tuple(output[0])
        return {name: np.asarray([row[name] for row in output]) for name in names}

    @staticmethod
    def _finite_range(values: np.ndarray) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        if not finite.size:
            return np.nan, np.nan
        return float(np.min(finite)), float(np.max(finite))

    @staticmethod
    def _empty() -> dict[str, np.ndarray]:
        fields: dict[str, Any] = defaultdict(lambda: float)
        fields.update(
            {
                "candidate_index": int,
                "branch_index": int,
                "epsilon_side": int,
                "converged": bool,
                "continuous": bool,
                "is_physical": bool,
                "is_stable": bool,
                "rayleigh_projection_status": str,
            }
        )
        names = (
            "candidate_index",
            "branch_index",
            "epsilon_side",
            "epsilon",
            "abs_epsilon",
            "asymptotic_exponent",
            "converged",
            "continuous",
            "full_residual_norm",
            "is_physical",
            "is_stable",
            "minimum_physical_eigenvalue",
            "maximum_jacobian_real_part",
            "delta_state_norm",
            "state_effective_exponent",
            "rayleigh_frequency",
            "delta_rayleigh_frequency",
            "rayleigh_effective_exponent",
            "state_fit_exponent",
            "rayleigh_fit_exponent",
            "rayleigh_visibility",
            "rayleigh_projection_status",
            "critical_hamiltonian_petermann_max",
            "hamiltonian_petermann_max",
            "hamiltonian_petermann_effective_exponent",
            "hamiltonian_petermann_fit_exponent",
        )
        return {name: np.asarray([], dtype=fields[name]) for name in names}

    @staticmethod
    def _empty_summary() -> dict[str, np.ndarray]:
        integer = {
            "candidate_index",
            "branch_count",
            "sample_count",
            "converged_count",
            "continuous_count",
            "physical_count",
            "stable_count",
        }
        boolean = {
            "all_converged",
            "all_continuous",
            "all_physical",
            "all_stable",
        }
        names = (
            "candidate_index",
            "branch_count",
            "sample_count",
            "converged_count",
            "continuous_count",
            "physical_count",
            "stable_count",
            "converged_fraction",
            "continuous_fraction",
            "physical_fraction",
            "stable_fraction",
            "all_converged",
            "all_continuous",
            "all_physical",
            "all_stable",
            "minimum_physical_eigenvalue",
            "maximum_jacobian_real_part",
            "minimum_state_fit_exponent",
            "maximum_state_fit_exponent",
            "minimum_rayleigh_fit_exponent",
            "maximum_rayleigh_fit_exponent",
            "minimum_rayleigh_visibility",
            "minimum_hamiltonian_petermann_max",
            "maximum_hamiltonian_petermann_max",
            "minimum_hamiltonian_petermann_fit_exponent",
            "maximum_hamiltonian_petermann_fit_exponent",
        )
        return {
            name: np.asarray(
                [],
                dtype=int if name in integer else bool if name in boolean else float,
            )
            for name in names
        }
