"""qphase_cam: path-information bounds of the full CAM covariance
--------------------------------------------------------------------
A scalar reconstruction reduces the CAM noise to one projected intensity ``q(x)`` and
reports the path-information rate ``∫ F(x)² π(x) / q(x) dx`` of that reduction. The
reduction throws away the noise in the other canonical directions, so the number is a
*bound* only for the one-dimensional problem it states. This module reports the same
quantity for the **full canonical covariance**, together with the reduced value it
contains, so a caller can say how much information the reduction discards.

Everything is evaluated on the caller's reconstruction and on nothing else:

* the manifold gives the states ``x`` (``manifold_embedding``),
* the reconstruction's ``generator_table`` gives the grid, the drift and the readout,
  and the shared :func:`~qphase_cam.nonlinear.evaluate_generator` gives the density and
  the reduced path information — there is no second grid, no second density and no
  second integration rule;
* the noise is the public batch transform
  :func:`~qphase_cam.core.noise.canonical_sample_matrix_covariances`, fed by
  :func:`~qphase_cam.nonlinear.noise_stack`, so the "batch when the model supports it,
  row by row otherwise" policy is the reconstruction's own;
* the parameter forcing is :func:`~qphase_cam.nonlinear.parameter_forcing`, the same
  batched ``parameter_jacobian``/``einsum`` order the reconstruction uses.

A source reconstruction that is not ``valid`` never yields a ``valid`` bound: when its
table is usable the numbers are still computed (that is the diagnostic a caller asking
for them wants) and the result repeats the source's verdict. A singular full covariance
is reported as singular — it is never regularized, clamped or solved in a least-squares
sense, because a full bound that needs a repaired covariance is not the bound of this
noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core.noise import (
    canonical_sample_matrix_covariance,
    canonical_sample_matrix_covariances,
)
from .nonlinear import (
    ScalarGeneratorEvaluation,
    ScalarReconstruction,
    ScalarReconstructionConfig,
    evaluate_generator,
    manifold_embedding,
    noise_stack,
    parameter_forcing,
)
from .outcomes import CAMOutcomeStatus

__all__ = [
    "PathInformationBounds",
    "RELATIVE_PSD_TOLERANCE",
    "path_information_bounds",
]

#: Largest negative eigenvalue, relative to the largest positive one, that still counts
#: as a positive-semidefinite covariance. The historical figure script rejected a
#: minimum relative eigenvalue below ``-1e-9``; the same explicit tolerance is used
#: here, and a covariance outside it is reported instead of repaired.
RELATIVE_PSD_TOLERANCE = 1.0e-9


class PathInformationBounds(BaseModel):
    """The reduced and full path-information bounds of one reconstruction.

    ``reduced`` is the reconstruction's own one-dimensional rate (``leftᵀ Σ left``),
    ``full`` the same integral with the full canonical covariance, and
    ``full_at_center`` the unweighted ``F₀ᵀ Σ(0)⁻¹ F₀`` at the reference state.
    ``minimum_relative_eigenvalue`` is the smallest ``λ_min/λ_max`` of the full
    covariance over the grid, i.e. the conditioning the linear solves had to face.

    ``source_status`` is the reconstruction's own status. A source that was not
    ``valid`` can still carry diagnostic numbers, but then ``status`` repeats the
    source's verdict rather than upgrading it. ``reduced_reconstruction`` is the value
    the source reported for the same integral; the two differences are ``None`` when the
    source did not provide it, never ``0``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None
    source_status: CAMOutcomeStatus

    reduced: float | None = None
    full: float | None = None
    full_at_center: float | None = None
    minimum_relative_eigenvalue: float | None = None

    reduced_reconstruction: float | None = None
    reduced_absolute_difference: float | None = None
    reduced_relative_difference: float | None = None

    evaluation: ScalarGeneratorEvaluation | None = None
    method: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def path_information_bounds(
    adapter: Any,
    model: Any,
    *,
    state: Any,
    params: Mapping[str, Any],
    reconstruction: ScalarReconstruction,
    parameter_direction: Any,
    config: ScalarReconstructionConfig | None = None,
) -> PathInformationBounds:
    """Return the reduced and full path-information bounds of one reconstruction.

    The reconstruction supplies the manifold, the grid (`generator_table`), the drift
    and the readout, so this function never resolves its own domain: a reconstruction
    that stopped before its grid cannot be bounded at all and says so.

    ``parameter_direction`` is the parameter-space direction of the driven parameter,
    e.g. ``adapter.parameter_direction(name, scale)`` — one entry per model parameter,
    which the parameter Jacobian maps onto the canonical coordinates. The reduced
    forcing is ``left · parameter_jacobian(x) · direction`` and the full one is the same
    vector before the projection.

    The full bound is ``∫ Fᵀ Σ_full(x)⁻¹ F π(x) dx`` with the canonical
    ``Σ_full(x)`` and the density and reduced rate of the shared
    :func:`~qphase_cam.nonlinear.evaluate_generator`. A singular ``Σ_full`` — the
    physical boundary is exactly where the critical direction has no noise — is
    reported as singular instead of being regularized, and a covariance outside
    :data:`RELATIVE_PSD_TOLERANCE` is reported as not positive semidefinite.
    """
    settings = config or ScalarReconstructionConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    direction = np.asarray(parameter_direction, dtype=float).reshape(-1)
    manifold = reconstruction.manifold
    table = reconstruction.generator_table
    source_valid = reconstruction.status is CAMOutcomeStatus.VALID
    method: dict[str, Any] = {
        "service": "qphase_cam.path_information.path_information_bounds",
        "source": "the caller's scalar_reconstruction (manifold and generator table)",
        "grid": (
            "the reconstruction's generator table"
            if table is None
            else f"{len(table.x)} points of the reconstruction's generator table"
        ),
        "state_stack": "qphase_cam.nonlinear.noise_stack (batch when declared)",
        "noise": "qphase_cam.core.noise.canonical_sample_matrix_covariances",
        "noise_at_center": "qphase_cam.core.noise.canonical_sample_matrix_covariance",
        "density_and_reduced": (
            "evaluate_generator(table, forcing=reduced forcing, config=config); the "
            "reduced bound is its path_information_rate"
        ),
        "forcing": (
            "parameter_forcing(adapter, vectors, params, left, direction): "
            "left . parameter_jacobian(x) . direction, the reconstruction's order"
        ),
        "full": "int F^T Sigma_full(x)^-1 F density(x) dx",
        "full_at_center": "F0^T Sigma_full(state)^-1 F0",
        "minimum_relative_eigenvalue": "min(eig_min/eig_max) of Sigma_full(x)",
        "integration_split": settings.integration_split,
        "config": settings.model_dump(mode="json"),
        "relative_psd_tolerance": float(RELATIVE_PSD_TOLERANCE),
        "parameter_direction": [float(value) for value in direction],
        "parameter_direction_norm": float(np.linalg.norm(direction)),
        "batch_noise": {
            "adapter_supports_batched_states": bool(
                getattr(adapter, "supports_batched_states", False)
            ),
            "model_supports_batched_matrices": bool(
                getattr(model, "supports_batched_matrices", False)
            ),
        },
        "source_status": reconstruction.status.value,
        "source_valid": source_valid,
        "source_reason_code": reconstruction.reason_code,
        "source_validity_domain": reconstruction.validity_domain,
    }
    diagnostics: dict[str, Any] = {}

    def failed(
        status: CAMOutcomeStatus,
        code: str,
        reason: str,
        *,
        evaluation: ScalarGeneratorEvaluation | None = None,
    ) -> PathInformationBounds:
        return PathInformationBounds(
            status=status,
            reason=reason,
            reason_code=code,
            source_status=reconstruction.status,
            evaluation=evaluation,
            method=dict(method),
            diagnostics=dict(diagnostics),
        )

    if not np.all(np.isfinite(point)):
        raise ValueError("the state must be finite")
    if not np.all(np.isfinite(direction)):
        raise ValueError("the parameter direction must be finite")
    left = np.asarray(manifold.left, dtype=float)
    if left.size != point.size:
        raise ValueError(
            f"the reconstruction's left vector has {left.size} entries for a "
            f"{point.size}-coordinate state"
        )
    if table is None:
        detail = (
            "the source reconstruction carries no generator table, so there is no "
            "grid, density or forcing to bound"
        )
        if not source_valid:
            diagnostics["missing_source_fields"] = ["generator_table"]
            return failed(
                reconstruction.status,
                "no_generator_table",
                f"{reconstruction.reason or 'the reconstruction did not resolve'}; "
                f"{detail}",
            )
        raise ValueError(
            f"a valid reconstruction must carry its generator table; {detail}"
        )

    x = np.asarray(table.x, dtype=float)
    if x.size < 3 or not np.all(np.isfinite(x)) or np.any(np.diff(x) <= 0.0):
        raise ValueError("the generator table's grid is not finite and increasing")
    vectors = manifold_embedding(manifold, point, x)
    matrices, diffusions, physical_minimum = noise_stack(
        adapter, model, vectors, parameter_map
    )
    covariance = canonical_sample_matrix_covariances(matrices, diffusions)
    if covariance.shape[0] != x.size:
        raise ValueError(
            f"the noise covariance has {covariance.shape[0]} rows for {x.size} states"
        )
    if np.any(physical_minimum < -settings.physical_tolerance):
        diagnostics["minimum_physical_eigenvalue"] = float(np.min(physical_minimum))
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "state_not_physical",
            "the state leaves the physical region inside the reconstruction's grid "
            f"(minimum eigenvalue {float(np.min(physical_minimum))!r} < "
            f"{-settings.physical_tolerance!r})",
        )

    forcing, forcing_vectors = parameter_forcing(
        adapter, vectors, parameter_map, left, direction
    )
    evaluation = evaluate_generator(table, forcing=forcing, config=settings)
    method["grid"] = f"{x.size} points of the reconstruction's generator table"
    method["core_point_count"] = evaluation.core_point_count
    method["edge_point_count"] = evaluation.edge_point_count
    method["density_normalization_raw"] = evaluation.density_normalization
    if not evaluation.normalizable or evaluation.scalar_intensity is None:
        diagnostics["density_normalization_raw"] = evaluation.density_normalization
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "density_not_normalizable",
            "the reconstruction's stationary density does not normalize on its grid",
            evaluation=evaluation,
        )
    density = np.asarray(evaluation.density, dtype=float)
    if not np.all(np.isfinite(density)) or np.any(density < 0.0):
        diagnostics["minimum_density"] = float(np.min(density))
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "density_not_finite",
            "the stationary density is not finite and non-negative",
            evaluation=evaluation,
        )
    reduced = evaluation.path_information_rate
    if reduced is None or not bool(np.isfinite(reduced)):
        diagnostics["reduced"] = reduced
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "reduced_path_information_not_finite",
            f"the reduced path-information rate is {reduced!r}",
            evaluation=evaluation,
        )
    reduced_value = float(reduced)

    eigenvalues = np.linalg.eigvalsh(covariance)
    if not np.all(np.isfinite(eigenvalues)):
        diagnostics["eigenvalue_range"] = [
            float(np.nanmin(eigenvalues)),
            float(np.nanmax(eigenvalues)),
        ]
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "covariance_spectrum_not_finite",
            "the full canonical covariance has a non-finite eigenvalue",
            evaluation=evaluation,
        )
    largest = eigenvalues[:, -1]
    if np.any(largest <= 0.0):
        diagnostics["largest_eigenvalue_minimum"] = float(np.min(largest))
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "covariance_not_positive_semidefinite",
            "the full canonical covariance has no positive eigenvalue at "
            f"{int(np.count_nonzero(largest <= 0.0))} grid points",
            evaluation=evaluation,
        )
    relative_eigenvalue = eigenvalues[:, 0] / largest
    minimum_relative_eigenvalue = float(np.min(relative_eigenvalue))
    diagnostics["minimum_relative_eigenvalue"] = minimum_relative_eigenvalue
    if minimum_relative_eigenvalue < -RELATIVE_PSD_TOLERANCE:
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "covariance_not_positive_semidefinite",
            "the full canonical covariance is not positive semidefinite: minimum "
            f"relative eigenvalue {minimum_relative_eigenvalue!r} < "
            f"{-RELATIVE_PSD_TOLERANCE!r}",
            evaluation=evaluation,
        )

    try:
        solved = np.linalg.solve(covariance, forcing_vectors[:, :, None])[:, :, 0]
    except np.linalg.LinAlgError as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "covariance_singular",
            "the full canonical covariance is singular on the reconstruction's grid, "
            f"so F^T Sigma^-1 F has no value ({exc})",
            evaluation=evaluation,
        )
    if not np.all(np.isfinite(solved)):
        diagnostics["solved_maximum"] = float(np.max(np.abs(solved)))
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "covariance_solve_not_finite",
            "solving the full canonical covariance did not give a finite vector",
            evaluation=evaluation,
        )
    integrand = np.einsum("ni,ni->n", forcing_vectors, solved, optimize=True)
    if not np.all(np.isfinite(integrand)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "full_integrand_not_finite",
            "the full path-information integrand is not finite",
            evaluation=evaluation,
        )
    if np.any(integrand < 0.0):
        diagnostics["minimum_full_integrand"] = float(np.min(integrand))
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "full_integrand_not_positive",
            "the full path-information integrand is negative, which an inverse "
            "covariance cannot produce: the linear solve is not the bound's",
            evaluation=evaluation,
        )
    full = float(np.trapezoid(integrand * density, x))
    if not np.isfinite(full):
        diagnostics["full"] = full
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "full_bound_not_finite",
            f"the full path-information bound is {full!r}",
            evaluation=evaluation,
        )

    center_matrix = np.asarray(
        adapter.state_matrix(point, parameter_map), dtype=complex
    )
    center_diffusion = np.asarray(
        model.cam_diffusion(center_matrix, parameter_map), dtype=complex
    )
    center_covariance = canonical_sample_matrix_covariance(
        center_matrix, center_diffusion
    )
    center_forcing = np.einsum(
        "ij,j->i",
        np.asarray(adapter.parameter_jacobian(point, parameter_map), dtype=float),
        direction,
        optimize=True,
    )
    try:
        center_solved = np.linalg.solve(center_covariance, center_forcing)
    except np.linalg.LinAlgError as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "center_covariance_singular",
            "the canonical covariance at the reference state is singular, so the "
            f"centre bound has no value ({exc})",
            evaluation=evaluation,
        )
    if not np.all(np.isfinite(center_solved)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "center_covariance_solve_not_finite",
            "solving the canonical covariance at the reference state did not give a "
            "finite vector",
            evaluation=evaluation,
        )
    full_at_center = float(center_forcing @ center_solved)
    if not np.isfinite(full_at_center):
        diagnostics["full_at_center"] = full_at_center
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "center_bound_not_finite",
            f"the centre path-information bound is {full_at_center!r}",
            evaluation=evaluation,
        )

    source_reduced = reconstruction.path_information_rate
    reduced_reconstruction: float | None = None
    absolute_difference: float | None = None
    relative_difference: float | None = None
    if source_reduced is not None and bool(np.isfinite(source_reduced)):
        reduced_reconstruction = float(source_reduced)
        absolute_difference = float(reduced_value - reduced_reconstruction)
        if reduced_reconstruction != 0.0:
            relative_difference = float(absolute_difference / reduced_reconstruction)
    method["reduced_reconstruction"] = (
        "reconstruction.path_information_rate"
        if reduced_reconstruction is not None
        else "the source did not provide a path-information rate"
    )
    method["full_integrand_range"] = [
        float(np.min(integrand)),
        float(np.max(integrand)),
    ]
    method["eigenvalue_range"] = [
        float(np.min(eigenvalues[:, 0])),
        float(np.max(largest)),
    ]

    status = CAMOutcomeStatus.VALID if source_valid else reconstruction.status
    reason: str | None = None
    reason_code: str | None = None
    if not source_valid:
        reason = (
            "diagnostic bounds of a source that is not valid "
            f"({reconstruction.status.value}): "
            f"{reconstruction.reason or 'no reason given'}"
        )
        reason_code = reconstruction.reason_code or "source_reconstruction_not_valid"
    return PathInformationBounds(
        status=status,
        reason=reason,
        reason_code=reason_code,
        source_status=reconstruction.status,
        reduced=reduced_value,
        full=full,
        full_at_center=full_at_center,
        minimum_relative_eigenvalue=minimum_relative_eigenvalue,
        reduced_reconstruction=reduced_reconstruction,
        reduced_absolute_difference=absolute_difference,
        reduced_relative_difference=relative_difference,
        evaluation=evaluation,
        method=dict(method),
        diagnostics=dict(diagnostics),
    )
