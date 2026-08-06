"""Finite-noise diagnostics for CAM equilibrium bifurcation candidates."""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from .base import CAMPostprocessor, CAMPostprocessorConfig


class StochasticValidityConfig(CAMPostprocessorConfig):
    """Numerical tolerances and an optional experimental perturbation scale."""

    probe_epsilon: float | None = Field(
        default=None,
        gt=0.0,
        description="Optional |epsilon| used to label the predicted noise regime",
    )
    critical_eigenvalue_tolerance: float = Field(
        default=1e-6,
        gt=0.0,
        description="Maximum absolute critical eigenvalue accepted as zero",
    )
    covariance_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        description="Tolerance for covariance PSD and projected-noise checks",
    )
    overlap_tolerance: float = Field(
        default=1e-10,
        gt=0.0,
        description="Minimum left-right critical eigenvector overlap",
    )


class StochasticValidity(CAMPostprocessor):
    """Estimate when projected sample-matrix noise masks local CAM response."""

    name: ClassVar[str] = "stochastic_validity"
    description: ClassVar[str] = (
        "Project approximate sample-matrix noise onto CAM critical modes"
    )
    config_schema: ClassVar[type[StochasticValidityConfig]] = StochasticValidityConfig
    accepted_result_kinds: ClassVar[frozenset[str]] = frozenset(
        {"bifurcation_candidates", "bifurcation_scan"}
    )

    def process(self, result: Any, model: Any, backend: Any) -> dict[str, Any]:
        del backend
        adapter = FPGenDynamicsAdapter.from_model(model)
        candidates = getattr(result, "candidates", result)
        closure = dict(candidates.meta.get("closure", adapter.closure_provenance()))
        rows: list[dict[str, Any]] = []
        for candidate_index in range(len(candidates.state_vectors)):
            branch_indices = self._branch_indices(candidates, candidate_index)
            if not branch_indices:
                rows.append(
                    self._base_row(
                        candidate_index,
                        branch_index=-1,
                        epsilon_side=0,
                        closure=closure,
                        status="no_real_branch",
                    )
                )
                continue
            for branch_index in branch_indices:
                rows.append(
                    self._evaluate_branch(
                        result,
                        model,
                        adapter,
                        candidate_index,
                        branch_index,
                        closure,
                    )
                )
        output = self._columns(rows)
        complete = int(np.count_nonzero(output["status"] == "complete"))
        self.result_metadata = {
            "status": "complete" if complete else "no_supported_branches",
            "row_count": len(rows),
            "complete_count": complete,
            "noise_semantics": "factorized_sample_matrix",
            "closure": closure,
        }
        return {self.name: output}

    def _evaluate_branch(
        self,
        result: Any,
        model: Any,
        adapter: FPGenDynamicsAdapter,
        candidate_index: int,
        branch_index: int,
        closure: dict[str, Any],
    ) -> dict[str, Any]:
        candidates = getattr(result, "candidates", result)
        branches = candidates.branches
        assert branches is not None
        side = int(branches.epsilon_side[branch_index])
        row = self._base_row(
            candidate_index,
            branch_index=branch_index,
            epsilon_side=side,
            closure=closure,
            status="unsupported",
        )
        if adapter.moment_layout != "normal":
            row["status"] = "unsupported_moment_layout"
            return row
        diffusion_provider = getattr(model, "cam_diffusion", None)
        if not callable(diffusion_provider):
            row["status"] = "diffusion_unavailable"
            return row
        if (
            int(branches.perturbation_order[branch_index]) != 1
            or int(branches.coupling_state_order[branch_index]) != 0
            or int(branches.state_order[branch_index]) % 2 != 1
        ):
            row["status"] = "unsupported_signature"
            return row

        params = self._critical_params(result, candidate_index, model)
        state = np.asarray(candidates.state_vectors[candidate_index], dtype=float)
        matrix = adapter.state_matrix(state, params)
        jacobian = np.asarray(adapter.jacobian(state, params), dtype=float)
        mode = self._critical_mode(jacobian)
        if mode is None:
            row["status"] = "critical_mode_unresolved"
            return row
        critical_value, right, left, condition, spectral_gap, matrix_condition = mode
        row.update(
            {
                "critical_eigenvalue_real": float(np.real(critical_value)),
                "critical_eigenvalue_imag": float(np.imag(critical_value)),
                "noncritical_spectral_gap": spectral_gap,
                "critical_mode_condition_number": condition,
                "eigenvector_condition_number": matrix_condition,
            }
        )
        if abs(critical_value) > self.config.critical_eigenvalue_tolerance:
            row["status"] = "critical_eigenvalue_not_zero"
            return row

        diffusion = np.asarray(diffusion_provider(matrix, params), dtype=complex)
        anomalous_provider = getattr(model, "cam_anomalous_diffusion", None)
        if callable(anomalous_provider):
            anomalous = np.asarray(anomalous_provider(matrix, params), dtype=complex)
            if np.linalg.norm(anomalous) > self.config.covariance_tolerance:
                row["status"] = "unsupported_anomalous_diffusion"
                return row
        covariance = canonical_sample_matrix_covariance(matrix, diffusion)
        covariance_eigenvalues = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
        minimum_covariance = float(np.min(covariance_eigenvalues))
        row["noise_covariance_minimum_eigenvalue"] = minimum_covariance
        if minimum_covariance < -self.config.covariance_tolerance:
            row["status"] = "noise_covariance_not_psd"
            return row
        projected_noise = float(np.real(left @ covariance @ left))
        row["projected_noise_intensity"] = projected_noise
        if projected_noise <= self.config.covariance_tolerance:
            row["status"] = "critical_noise_unresolved"
            return row

        perturbation = str(candidates.meta.get("perturbation_parameter", ""))
        if not perturbation:
            row["status"] = "perturbation_parameter_missing"
            return row
        scale = float(candidates.meta.get("perturbation_scale", 1.0))
        parameter_direction = adapter.parameter_direction(perturbation, scale)
        forcing = float(
            np.real(
                left @ adapter.parameter_jacobian(state, params) @ parameter_direction
            )
        )
        coefficient_matrix = np.asarray(
            branches.leading_state_coefficient[branch_index]
        )
        coefficient = np.asarray(matrix_to_vector(coefficient_matrix), dtype=float)
        center_coefficient = float(left @ coefficient)
        row["parameter_forcing"] = forcing
        row["branch_center_coefficient"] = center_coefficient
        if (
            abs(forcing) <= np.finfo(float).eps
            or abs(center_coefficient) <= np.finfo(float).eps
        ):
            row["status"] = "normal_form_projection_unresolved"
            return row

        order = int(branches.state_order[branch_index])
        normal_coefficient = float(-forcing * side / center_coefficient**order)
        confining = bool(normal_coefficient < 0.0)
        fluctuation_scale = float(
            ((order + 1) * projected_noise / (2.0 * abs(normal_coefficient)))
            ** (1.0 / (order + 1))
        )
        crossover = float((fluctuation_scale / abs(center_coefficient)) ** order)
        probe = self.config.probe_epsilon
        regime = (
            "unassessed"
            if probe is None
            else "noise_dominated"
            if probe <= crossover
            else "response_dominated"
        )
        row.update(
            {
                "normal_form_state_coefficient": normal_coefficient,
                "normal_form_confining": confining,
                "critical_fluctuation_scale": fluctuation_scale,
                "epsilon_crossover": crossover,
                "probe_epsilon": np.nan if probe is None else probe,
                "regime": regime,
                "status": "complete" if confining else "nonconfining_normal_form",
            }
        )
        return row

    def _critical_mode(
        self, jacobian: np.ndarray
    ) -> tuple[complex, np.ndarray, np.ndarray, float, float, float] | None:
        values, right_vectors = np.linalg.eig(jacobian)
        index = int(np.argmin(np.abs(values)))
        critical = values[index]
        right = np.real_if_close(right_vectors[:, index])
        left_values, left_vectors = np.linalg.eig(jacobian.T)
        left_index = int(np.argmin(np.abs(left_values - critical)))
        left = np.real_if_close(left_vectors[:, left_index])
        if np.iscomplexobj(right) or np.iscomplexobj(left):
            return None
        right = np.asarray(right, dtype=float)
        left = np.asarray(left, dtype=float)
        right /= np.linalg.norm(right)
        overlap = float(left @ right)
        if abs(overlap) <= self.config.overlap_tolerance:
            return None
        left /= overlap
        remaining = np.delete(values, index)
        spectral_gap = float(-np.max(np.real(remaining))) if remaining.size else np.nan
        return (
            critical,
            right,
            left,
            float(np.linalg.norm(left)),
            spectral_gap,
            float(np.linalg.cond(right_vectors)),
        )

    @staticmethod
    def _branch_indices(candidates: Any, candidate_index: int) -> list[int]:
        branches = candidates.branches
        if branches is None:
            return []
        return [
            index
            for index in range(branches.size)
            if int(branches.candidate_index[index]) == candidate_index
            and bool(branches.real_branch[index])
        ]

    @staticmethod
    def _critical_params(
        result: Any, candidate_index: int, model: Any
    ) -> dict[str, Any]:
        candidates = getattr(result, "candidates", result)
        params = dict(model.params)
        params.update(candidates.meta.get("fixed_params", {}))
        if hasattr(result, "candidate_offsets"):
            case = int(
                np.searchsorted(result.candidate_offsets, candidate_index, side="right")
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
    def _base_row(
        candidate_index: int,
        *,
        branch_index: int,
        epsilon_side: int,
        closure: dict[str, Any],
        status: str,
    ) -> dict[str, Any]:
        return {
            "candidate_index": candidate_index,
            "branch_index": branch_index,
            "epsilon_side": epsilon_side,
            "status": status,
            "critical_eigenvalue_real": np.nan,
            "critical_eigenvalue_imag": np.nan,
            "noncritical_spectral_gap": np.nan,
            "critical_mode_condition_number": np.nan,
            "eigenvector_condition_number": np.nan,
            "noise_covariance_minimum_eigenvalue": np.nan,
            "projected_noise_intensity": np.nan,
            "parameter_forcing": np.nan,
            "branch_center_coefficient": np.nan,
            "normal_form_state_coefficient": np.nan,
            "normal_form_confining": False,
            "critical_fluctuation_scale": np.nan,
            "epsilon_crossover": np.nan,
            "probe_epsilon": np.nan,
            "regime": "unassessed",
            "noise_semantics": "factorized_sample_matrix",
            "representation": str(closure.get("representation", "unknown")),
            "fpe_is_exact": bool(closure.get("fpe_is_exact", False)),
            "moment_closure": str(closure.get("moment_closure", "unknown")),
            "moment_closure_is_exact": bool(
                closure.get("moment_closure_is_exact", False)
            ),
            "deterministic_cam_is_exact": bool(
                closure.get("deterministic_cam_is_exact", False)
            ),
        }

    @staticmethod
    def _columns(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        if not rows:
            return {"candidate_index": np.asarray([], dtype=int)}
        names = tuple(rows[0])
        return {name: np.asarray([row[name] for row in rows]) for name in names}


def canonical_sample_matrix_covariance(
    state: np.ndarray, diffusion: np.ndarray
) -> np.ndarray:
    """Return the M=0 sample-matrix noise covariance in canonical coordinates."""
    state = np.asarray(state, dtype=complex)
    diffusion = np.asarray(diffusion, dtype=complex)
    if state.ndim != 2 or state.shape[0] != state.shape[1]:
        raise ValueError("CAM state must be a square matrix")
    if diffusion.shape != state.shape:
        raise ValueError("CAM diffusion must have the same shape as the state")
    modes = state.shape[0]
    complex_covariance = np.empty((modes * modes, modes * modes), dtype=complex)
    for i in range(modes):
        for j in range(modes):
            for k in range(modes):
                for right in range(modes):
                    complex_covariance[i * modes + j, k * modes + right] = (
                        diffusion[i, right] * state[k, j]
                        + diffusion[k, j] * state[i, right]
                    )
    transform = _canonical_hermitian_transform(modes)
    covariance = np.real_if_close(transform @ complex_covariance @ transform.T)
    if np.iscomplexobj(covariance):
        raise ValueError("canonical CAM noise covariance is not real")
    return np.asarray(covariance, dtype=float)


def _canonical_hermitian_transform(modes: int) -> np.ndarray:
    transform = np.zeros((modes * modes, modes * modes), dtype=complex)
    row = 0
    for mode in range(modes):
        transform[row, mode * modes + mode] = 1.0
        row += 1
    for left in range(modes):
        for right in range(left + 1, modes):
            transform[row, left * modes + right] = 0.5
            transform[row, right * modes + left] = 0.5
            row += 1
    for left in range(modes):
        for right in range(left + 1, modes):
            transform[row, left * modes + right] = -0.5j
            transform[row, right * modes + left] = 0.5j
            row += 1
    return transform
