"""High-precision local response validation for CAM bifurcation branches."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
from pydantic import Field, model_validator

from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from .base import CAMPostprocessor, CAMPostprocessorConfig


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

    @model_validator(mode="after")
    def validate_interval(self) -> LocalResponseValidationConfig:
        if self.epsilon_max <= self.epsilon_min:
            raise ValueError("epsilon_max must exceed epsilon_min")
        if self.fit_points > self.epsilon_points:
            raise ValueError("fit_points cannot exceed epsilon_points")
        return self


class LocalResponseValidation(CAMPostprocessor):
    """Validate classified local branches against the complete CAM residual."""

    name: ClassVar[str] = "local_response_validation"
    description: ClassVar[str] = (
        "Solve complete CAM branches and measure state/Rayleigh response exponents"
    )
    config_schema: ClassVar[type[LocalResponseValidationConfig]] = (
        LocalResponseValidationConfig
    )
    accepted_result_kinds: ClassVar[frozenset[str]] = frozenset(
        {"bifurcation_candidates", "bifurcation_scan"}
    )

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        adapter = FPGenDynamicsAdapter.from_model(model)
        candidates = getattr(result, "candidates", result)
        branches = candidates.branches
        if branches is None or branches.size == 0:
            self.result_metadata = {"status": "no_real_branches", "sample_count": 0}
            return {self.name: self._empty()}

        magnitudes = np.geomspace(
            self.config.epsilon_min,
            self.config.epsilon_max,
            self.config.epsilon_points,
        )
        rows: list[dict[str, Any]] = []
        for branch_index in range(branches.size):
            if not bool(branches.real_branch[branch_index]):
                continue
            candidate_index = int(branches.candidate_index[branch_index])
            params = self._critical_params(result, candidate_index, model)
            perturbation = str(candidates.meta.get("perturbation_parameter", ""))
            if not perturbation:
                raise ValueError("bifurcation result has no perturbation_parameter")
            scale = float(candidates.meta.get("perturbation_scale", 1.0))
            critical_decimals = self._critical_state_decimals(
                candidates, candidate_index
            )
            critical = self._critical_state(
                candidates, candidate_index, critical_decimals
            )
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
                            np.max(np.real(jacobian))
                            <= self.config.stability_tolerance
                        ),
                        "minimum_physical_eigenvalue": float(np.min(physical)),
                        "maximum_jacobian_real_part": float(
                            np.max(np.real(jacobian))
                        ),
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
                    }
                )
            self._add_exponents(branch_rows)
            rows.extend(branch_rows)

        output = self._columns(rows)
        self.result_metadata = {
            "status": "complete" if rows else "no_real_branches",
            "sample_count": len(rows),
            "precision_digits": self.config.precision_digits,
            "perturbation_parameter": candidates.meta.get(
                "perturbation_parameter"
            ),
        }
        return {self.name: output}

    def _newton(
        self,
        adapter: FPGenDynamicsAdapter,
        initial: np.ndarray,
        params: dict[str, Any],
        *,
        critical_decimals: tuple[str, ...] | None,
        leading_coefficient: np.ndarray,
        epsilon_magnitude: float,
        exponent: float,
    ) -> tuple[np.ndarray, float, bool]:
        import mpmath as mp

        previous_digits = mp.mp.dps
        mp.mp.dps = self.config.precision_digits
        if critical_decimals is None:
            state = mp.matrix([mp.mpf(str(item)) for item in initial])
        else:
            magnitude = mp.mpf(str(epsilon_magnitude)) ** mp.mpf(str(exponent))
            state = mp.matrix(
                [
                    mp.mpf(value) + mp.mpf(str(coefficient)) * magnitude
                    for value, coefficient in zip(
                        critical_decimals, leading_coefficient, strict=True
                    )
                ]
            )
        mp_params = {
            name: mp.mpf(str(value)) if np.asarray(value).ndim == 0 else value
            for name, value in params.items()
        }
        tolerance = mp.mpf(str(self.config.residual_tolerance))
        try:
            residual = mp.inf
            for _ in range(self.config.max_iterations):
                values = adapter.mpmath_rhs(state, mp_params)
                residual = mp.sqrt(sum(item * item for item in values))
                if residual <= tolerance:
                    break
                jacobian = adapter.mpmath_jacobian(state, mp_params)
                try:
                    step = mp.lu_solve(jacobian, -values)
                except ZeroDivisionError:
                    break
                accepted = False
                damping = mp.mpf("1")
                for _ in range(16):
                    trial = state + damping * step
                    trial_values = adapter.mpmath_rhs(trial, mp_params)
                    trial_residual = mp.sqrt(
                        sum(item * item for item in trial_values)
                    )
                    if trial_residual < residual:
                        state = trial
                        residual = trial_residual
                        accepted = True
                        break
                    damping /= 2
                if not accepted:
                    break
            output = np.asarray([float(item) for item in state], dtype=float)
            return output, float(residual), bool(residual <= tolerance)
        finally:
            mp.mp.dps = previous_digits

    def _critical_params(
        self, result: Any, candidate_index: int, model: Any
    ) -> dict[str, Any]:
        candidates = getattr(result, "candidates", result)
        params = dict(model.params)
        params.update(candidates.meta.get("fixed_params", {}))
        if hasattr(result, "candidate_offsets"):
            case = int(
                np.searchsorted(
                    result.candidate_offsets, candidate_index, side="right"
                )
                - 1
            )
            params.update(
                {
                    name: np.asarray(values).reshape(-1)[case]
                    for name, values in result.case_params.items()
                }
            )
        params.update(
            zip(
                candidates.control_names,
                candidates.control_values[candidate_index],
                strict=True,
            )
        )
        return params

    @staticmethod
    def _critical_state_decimals(
        candidates: Any, index: int
    ) -> tuple[str, ...] | None:
        decimals = candidates.diagnostics.get("verified_full_state_decimal_values")
        if decimals is not None:
            values = np.asarray(decimals, dtype=object)[index]
            if isinstance(values, np.ndarray):
                values = values.tolist()
            if values and not isinstance(values, float):
                return tuple(str(item) for item in values)
        return None

    @staticmethod
    def _critical_state(
        candidates: Any,
        index: int,
        decimals: tuple[str, ...] | None,
    ) -> np.ndarray:
        if decimals is not None:
            return np.asarray([float(item) for item in decimals], dtype=float)
        return np.asarray(candidates.state_vectors[index], dtype=float)

    @staticmethod
    def _rayleigh(
        adapter: FPGenDynamicsAdapter,
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
        adapter: FPGenDynamicsAdapter,
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

    @staticmethod
    def _columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not rows:
            return LocalResponseValidation._empty()
        names = tuple(rows[0])
        return {name: np.asarray([row[name] for row in rows]) for name in names}

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
            "candidate_index", "branch_index", "epsilon_side", "epsilon",
            "abs_epsilon", "asymptotic_exponent", "converged", "continuous",
            "full_residual_norm", "is_physical", "is_stable",
            "minimum_physical_eigenvalue", "maximum_jacobian_real_part",
            "delta_state_norm", "state_effective_exponent", "rayleigh_frequency",
            "delta_rayleigh_frequency", "rayleigh_effective_exponent",
            "state_fit_exponent", "rayleigh_fit_exponent", "rayleigh_visibility",
            "rayleigh_projection_status",
        )
        return {name: np.asarray([], dtype=fields[name]) for name in names}
