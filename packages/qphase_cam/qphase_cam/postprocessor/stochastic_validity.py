"""Finite-noise diagnostics for CAM equilibrium bifurcation candidates."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, ClassVar

import numpy as np
from pydantic import Field

from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.noise import canonical_sample_matrix_covariance
from qphase_cam.core.qpsgen import QPSGenDynamicsAdapter
from qphase_cam.data import build_annotation_product
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.validity import critical_mode, minimum_eigenvalue, projected_intensity

from .base import CAMAnnotationContext, CAMPostprocessor, CAMPostprocessorConfig
from .candidates import (
    CandidateView,
    candidate_parameters,
    candidate_view,
    pivot_branch_rows,
)


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


class StochasticValidity(CAMPostprocessor[StochasticValidityConfig]):
    """Estimate when projected sample-matrix noise masks local CAM response."""

    name: ClassVar[str] = "stochastic_validity"
    description: ClassVar[str] = (
        "Project approximate sample-matrix noise onto CAM critical modes"
    )
    config_schema: ClassVar[type[StochasticValidityConfig]] = StochasticValidityConfig
    annotation_kind: ClassVar[str] = "stochastic_validity"
    accepted_product_roles: ClassVar[tuple[str, ...]] = ("candidates",)
    requires_model: ClassVar[bool] = True
    requires_backend: ClassVar[bool] = False

    def process(self, product: Any, context: CAMAnnotationContext) -> Any:
        model = context.model
        adapter = QPSGenDynamicsAdapter.from_model(model)
        view = candidate_view(product, metadata=dict(context.metadata))
        result = view
        candidates = view
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
        status = output.get("status")
        complete = (
            int(np.count_nonzero(status == "complete")) if status is not None else 0
        )
        self.result_metadata = {
            "status": "complete" if complete else "no_supported_branches",
            "row_count": len(rows),
            "complete_count": complete,
            "noise_semantics": "factorized_sample_matrix",
            "closure": closure,
        }
        del output
        return self._annotation(view, rows)

    def _annotation(self, view: CandidateView, rows: list[dict[str, Any]]) -> Any:
        """Return the typed stochastic-validity annotation over the candidates."""
        status: list[str] = []
        for row in view.rows:
            sample = [
                item for item in rows if int(item["candidate_index"]) == row.index
            ]
            if not sample:
                status.append(CAMOutcomeStatus.NOT_FOUND.value)
            elif any(str(item["status"]) == "complete" for item in sample):
                status.append(CAMOutcomeStatus.VALID.value)
            else:
                status.append(CAMOutcomeStatus.UNCHECKED.value)
        fields, field_axes, axes = pivot_branch_rows(
            view, rows, rename={"status": "branch_status"}
        )
        if not fields:
            # An empty candidate set still declares the full field schema, with a
            # zero-length branch axis, so a consumer sees the same columns.
            template = self._base_row(
                0,
                branch_index=-1,
                epsilon_side=0,
                closure={},
                status="no_real_branch",
            )
            for name, value in template.items():
                resolved = "branch_status" if name == "status" else name
                if isinstance(value, bool):
                    dtype: Any = bool
                elif isinstance(value, int):
                    dtype = np.int64
                elif isinstance(value, str):
                    dtype = f"U{max(len(value), 1)}"
                else:
                    dtype = float
                fields[resolved] = np.zeros((len(view.rows), 0), dtype=dtype)
                field_axes[resolved] = ("branch",)
            axes["branch"] = 0
        return build_annotation_product(
            annotation_kind=self.annotation_kind,
            target_kind="candidate",
            target_identity=[row.identity for row in view.rows],
            case_identity=[row.case_identity for row in view.rows],
            status=status,
            fields=fields,
            field_axes=field_axes,
            axes=axes,
            method_metadata=[dict(self.result_metadata) for _ in view.rows],
        )

    def _evaluate_branch(
        self,
        result: Any,
        model: Any,
        adapter: QPSGenDynamicsAdapter,
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
        # The shared critical-mode solver owns the eigenproblem; the plugin keeps
        # its historical "nearest zero" selection rule so its published numbers do
        # not move, and the service's default "slowest" rule stays separate.
        mode = critical_mode(
            jacobian,
            overlap_tolerance=self.config.overlap_tolerance,
            rule="nearest_zero",
        )
        if mode is None:
            row["status"] = "critical_mode_unresolved"
            return row
        left = mode.left
        row.update(
            {
                "critical_eigenvalue_real": float(np.real(mode.value)),
                "critical_eigenvalue_imag": float(np.imag(mode.value)),
                "noncritical_spectral_gap": mode.spectral_gap,
                "critical_mode_condition_number": mode.condition_number,
                "eigenvector_condition_number": mode.eigenvector_condition_number,
            }
        )
        if abs(mode.value) > self.config.critical_eigenvalue_tolerance:
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
        minimum_covariance, _ = minimum_eigenvalue(covariance)
        row["noise_covariance_minimum_eigenvalue"] = minimum_covariance
        if minimum_covariance < -self.config.covariance_tolerance:
            row["status"] = "noise_covariance_not_psd"
            return row
        projected_noise = projected_intensity(covariance, left, left)
        row["projected_noise_intensity"] = projected_noise
        if projected_noise <= self.config.covariance_tolerance:
            row["status"] = "critical_noise_unresolved"
            return row

        design = candidates.design_at(candidate_index)
        if not design.is_resolved:
            row["status"] = "perturbation_parameter_missing"
            return row
        perturbation = design.parameter
        scale = design.scale
        parameter_direction = adapter.parameter_direction(perturbation, scale)
        forcing = float(
            np.real(
                left @ adapter.parameter_jacobian(state, params) @ parameter_direction
            )
        )
        # The product stores each coefficient matrix as a per-row array; a single
        # row arrives as an object array of Python complex numbers, so normalise
        # the dtype before the canonical real chart is read from it.
        coefficient_matrix = np.asarray(
            branches.leading_state_coefficient[branch_index], dtype=complex
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
        """Return the model parameters of one candidate.

        The typed path resolves the candidate through the candidate view; a
        legacy scan result maps the candidate index to its case grid index first,
        so a candidate is never read as if it were a case.
        """
        return candidate_parameters(result, candidate_index, model)

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
            return StochasticValidity._empty()
        names = tuple(rows[0])
        return {name: np.asarray([row[name] for row in rows]) for name in names}

    @staticmethod
    def _empty() -> dict[str, np.ndarray]:
        """Return the stable empty-table schema.

        Every ``_base_row`` field is present with the same dtype kind the
        non-empty table would produce.
        """
        fields: dict[str, Any] = defaultdict(lambda: float)
        fields.update(
            {
                "candidate_index": int,
                "branch_index": int,
                "epsilon_side": int,
                "status": str,
                "normal_form_confining": bool,
                "regime": str,
                "noise_semantics": str,
                "representation": str,
                "fpe_is_exact": bool,
                "moment_closure": str,
                "moment_closure_is_exact": bool,
                "deterministic_cam_is_exact": bool,
            }
        )
        names = (
            "candidate_index",
            "branch_index",
            "epsilon_side",
            "status",
            "critical_eigenvalue_real",
            "critical_eigenvalue_imag",
            "noncritical_spectral_gap",
            "critical_mode_condition_number",
            "eigenvector_condition_number",
            "noise_covariance_minimum_eigenvalue",
            "projected_noise_intensity",
            "parameter_forcing",
            "branch_center_coefficient",
            "normal_form_state_coefficient",
            "normal_form_confining",
            "critical_fluctuation_scale",
            "epsilon_crossover",
            "probe_epsilon",
            "regime",
            "noise_semantics",
            "representation",
            "fpe_is_exact",
            "moment_closure",
            "moment_closure_is_exact",
            "deterministic_cam_is_exact",
        )
        return {name: np.asarray([], dtype=fields[name]) for name in names}
