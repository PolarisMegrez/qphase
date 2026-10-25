"""qphase_cam: the unified stochastic-validity service
---------------------------------------------------------
One linear-noise assessment answers "can this state carry the observed readout at
all, and with which noise scale?". It is a **root service**, not a plugin: the
``stochastic_validity`` postprocessor and any script or test call the same
functions, so there is exactly one PSD test, one critical-mode solver and one
noise-timescale definition in the package.

Conventions (stated once, used everywhere):

* ``J`` is the dynamics Jacobian at the state and ``D`` the noise covariance in
  the *same* canonical coordinates, with the Itô convention
  ``J Σ + Σ Jᵀ + D = 0`` for the stationary covariance ``Σ``. A stable ``J`` is
  solved directly by LAPACK's Bartels-Stewart Sylvester solver, which is the solver
  the historical centre-manifold probe uses; a singular spectrum falls back to the
  Kronecker least-squares form and is reported through its residual.
* The zero-frequency response of an observable with gradient ``g`` is the static
  susceptibility ``gᵀ(-J)⁻¹``; its norm is reported.
* The integrated noise intensity is the zero-frequency spectrum
  ``S(0) = 2∫₀^∞ C(τ)dτ = 2 gᵀ(-J)⁻¹Σg``.
* The correlation time is ``τ_c = ∫₀^∞ C(τ)dτ / C(0) = gᵀ(-J)⁻¹Σg / (gᵀΣg)``.

For the scalar Ornstein-Uhlenbeck case ``J = -1``, ``D = 1``, unit gradient this
gives ``Σ = 0.5``, response ``1``, ``S(0) = 1`` and ``τ_c = 1`` — the benchmark the
tests freeze.

Three rules shape the report:

* **Nothing is clamped.** A negative diffusion eigenvalue, a non-positive
  projected variance, a non-finite Lyapunov solution or a zero response is
  reported as it is; the status says the assessment failed instead of the number
  being made plausible.
* **A failure is partial, not empty.** Everything computed before the failing
  check stays in the assessment, with the ``reason`` naming the check.
* **The status vocabulary is the shared one.** ``valid``, ``stochastic_invalid``,
  ``unidentifiable``, ``uncontrolled_approximation`` and ``not_converged`` all
  mean what :class:`~qphase_cam.outcomes.CAMOutcomeStatus` says they mean.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.data import Dataset
from scipy.linalg import solve_continuous_lyapunov

from .data import build_annotation_product
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus

__all__ = [
    "CriticalMode",
    "StochasticAssessment",
    "StochasticAssessmentConfig",
    "assess_stochastic_state",
    "build_stochastic_assessment_annotation",
    "critical_mode",
    "lyapunov_covariance",
    "minimum_eigenvalue",
    "noise_timescales",
    "numeric_column",
    "projected_intensity",
    "tristate_column",
    "zero_frequency_gain",
]

#: Moment layouts whose canonical coordinates carry a physical diffusion.
_SUPPORTED_MOMENT_LAYOUTS = frozenset({"normal", "augmented"})


class StochasticAssessmentConfig(BaseModel):
    """Tolerances and step control of one assessment.

    The defaults are the ones the existing ``stochastic_validity`` plugin uses, so
    the plugin and the service cannot drift apart by accident.
    """

    model_config = ConfigDict(extra="forbid")

    critical_eigenvalue_tolerance: float = Field(
        default=1e-6,
        ge=0.0,
        description="Largest real part of the slowest mode still treated as marginal",
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
    critical_mode_rule: Literal["slowest", "nearest_zero"] = Field(
        default="slowest",
        description=(
            "Which eigenvalue is the critical one. 'slowest' (the default) is the "
            "linear-noise rule; 'nearest_zero' is the historical bifurcation-probe "
            "rule, which a caller reproducing that study selects explicitly"
        ),
    )
    response_tolerance: float = Field(
        default=1e-12,
        ge=0.0,
        description="Smallest zero-frequency response norm treated as resolved",
    )
    lyapunov_relative_residual_tolerance: float = Field(
        default=1e-8,
        ge=0.0,
        description="Largest relative residual of the stationary covariance solution",
    )
    relative_step: float = Field(
        default=1e-5,
        gt=0.0,
        description="Relative finite-difference step for the observable gradient",
    )


class StochasticAssessment(BaseModel):
    """The full report of one linear-noise assessment.

    Every measured quantity is optional: a check that never ran because an
    earlier one failed stays ``None`` instead of being filled with a default. The
    ``status`` and ``reason`` say what happened; ``diagnostics`` keeps the raw
    values a caller may want (eigenvalues, residuals) without pretending they are
    a verdict.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None

    # -- linearization ----------------------------------------------------
    critical_mode_resolved: bool
    critical_eigenvalue_real: float | None = None
    critical_eigenvalue_imag: float | None = None
    noncritical_spectral_gap: float | None = None
    critical_mode_condition_number: float | None = None
    eigenvector_condition_number: float | None = None

    # -- diffusion --------------------------------------------------------
    diffusion_is_psd: bool | None = None
    diffusion_minimum_eigenvalue: float | None = None

    # -- stationary covariance -------------------------------------------
    lyapunov_solution_finite: bool | None = None
    lyapunov_relative_residual: float | None = None
    lyapunov_covariance_trace: float | None = None
    lyapunov_covariance_minimum_eigenvalue: float | None = None
    lyapunov_is_psd: bool | None = None

    # -- observable -------------------------------------------------------
    observable_value: float | None = None
    observable_gradient: tuple[float, ...] | None = None
    projected_variance: float | None = None
    projected_variance_is_positive: bool | None = None

    # -- response and noise ----------------------------------------------
    zero_frequency_response_norm: float | None = None
    zero_frequency_response_finite: bool | None = None
    integrated_noise_intensity: float | None = None
    correlation_time: float | None = None
    integrated_noise_intensity_finite: bool | None = None
    correlation_time_positive: bool | None = None

    # -- representation ---------------------------------------------------
    representation_supported: bool
    anomalous_diffusion_norm: float | None = None

    # -- method -----------------------------------------------------------
    derivative: DerivativeRecord | None = None
    method: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True)
class CriticalMode:
    """The resolved slowest mode of one linearization."""

    value: complex
    right: np.ndarray
    left: np.ndarray
    condition_number: float
    spectral_gap: float
    eigenvector_condition_number: float


def numeric_column(values: Sequence[Any]) -> np.ndarray:
    """Return one float column in which ``None`` becomes ``NaN``, never zero."""
    return np.asarray(
        [np.nan if value is None else float(value) for value in values], dtype=float
    )


def tristate_column(values: Sequence[Any]) -> np.ndarray:
    """Return one tri-state ``int8`` column: -1 unknown, 0 false, 1 true."""
    return np.asarray(
        [-1 if value is None else int(bool(value)) for value in values], dtype=np.int8
    )


def minimum_eigenvalue(covariance: Any) -> tuple[float, np.ndarray]:
    """Return the smallest eigenvalue and the spectrum of a symmetric matrix.

    The matrix is symmetrized first (the Lyapunov solution and the canonical
    sample-matrix covariance are symmetric by construction; a small asymmetry is
    rounding, not physics).
    """
    matrix = np.asarray(covariance, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"a covariance must be square; got shape {matrix.shape}")
    symmetric = (matrix + matrix.T) / 2.0
    spectrum = np.linalg.eigvalsh(symmetric)
    return float(np.min(spectrum)), spectrum


def projected_intensity(covariance: Any, left: Any, right: Any) -> float:
    """Return one projected noise intensity ``leftᵀ Σ right`` as a real number.

    No imaginary part is tolerated: a complex projection means the caller mixed
    coordinates that do not belong together, which is an error rather than a
    number to round away.
    """
    value = (
        np.asarray(left, dtype=float)
        @ np.asarray(covariance, dtype=float)
        @ np.asarray(right, dtype=float)
    )
    return float(np.real(value))


def critical_mode(
    jacobian: Any,
    *,
    overlap_tolerance: float = 1e-10,
    rule: str = "slowest",
) -> CriticalMode | None:
    """Return the critical mode of one Jacobian, or ``None`` if unresolved.

    ``rule="slowest"`` selects the eigenvalue with the largest real part, which is
    what a linear-noise assessment needs. ``rule="nearest_zero"`` keeps the
    historical selection of the ``stochastic_validity`` plugin (smallest ``|λ|``,
    i.e. the marginal mode at a bifurcation point). The two agree whenever a zero
    mode exists; the option exists so the plugin's published numbers keep their
    original definition.

    A mode whose left/right eigenvectors are not real, or whose left-right overlap
    is at rounding scale, is reported as unresolved instead of being normalised
    into something that looks usable.
    """
    matrix = np.asarray(jacobian, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"a Jacobian must be square; got shape {matrix.shape}")
    values, right_vectors = np.linalg.eig(matrix)
    if rule == "slowest":
        index = int(np.argmax(np.real(values)))
    elif rule == "nearest_zero":
        index = int(np.argmin(np.abs(values)))
    else:
        raise ValueError(f"unknown critical-mode rule {rule!r}")
    critical = values[index]
    right = np.real_if_close(right_vectors[:, index])
    left_values, left_vectors = np.linalg.eig(matrix.T)
    left_index = int(np.argmin(np.abs(left_values - critical)))
    left = np.real_if_close(left_vectors[:, left_index])
    if np.iscomplexobj(right) or np.iscomplexobj(left):
        return None
    right = np.asarray(right, dtype=float)
    left = np.asarray(left, dtype=float)
    right_norm = float(np.linalg.norm(right))
    if right_norm == 0.0 or not np.isfinite(right_norm):
        return None
    right = right / right_norm
    overlap = float(left @ right)
    if abs(overlap) <= overlap_tolerance:
        return None
    left = left / overlap
    remaining = np.delete(values, index)
    spectral_gap = float(-np.max(np.real(remaining))) if remaining.size else np.nan
    return CriticalMode(
        value=complex(critical),
        right=right,
        left=left,
        condition_number=float(np.linalg.norm(left)),
        spectral_gap=spectral_gap,
        eigenvector_condition_number=float(np.linalg.cond(right_vectors)),
    )


def _lyapunov_covariance(
    jacobian: Any, diffusion: Any
) -> tuple[np.ndarray, float, str]:
    """Return ``(Σ, relative residual, solver name)`` of ``J Σ + Σ Jᵀ + D = 0``.

    A stable Jacobian is solved by LAPACK's Bartels-Stewart Sylvester solver
    (:func:`scipy.linalg.solve_continuous_lyapunov`). That is the solver the
    historical centre-manifold probe uses, so the covariance of a stable system is
    the probe's covariance rather than a second solution of the same equation.

    The Sylvester solver *perturbs the coefficients* when a pair of eigenvalues sums
    to zero — a singular Jacobian with no stationary covariance — and reports that as
    a warning; it can also overflow. Neither is a solve, so those cases fall back to
    the Kronecker least-squares solve, which always answers and whose returned
    residual then says the equation was not satisfied. The solver name travels with
    the result so a report can say which one produced it.
    """
    matrix = np.asarray(jacobian, dtype=float)
    noise = np.asarray(diffusion, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"a Jacobian must be square; got shape {matrix.shape}")
    if noise.shape != matrix.shape:
        raise ValueError(
            f"the diffusion shape {noise.shape} does not match the Jacobian "
            f"shape {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(noise)):
        raise ValueError("the Jacobian and the diffusion must be finite")
    size = matrix.shape[0]
    solution: np.ndarray | None = None
    solver = "continuous lyapunov (LAPACK trsyl)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            try:
                solution = np.asarray(
                    solve_continuous_lyapunov(matrix, -noise), dtype=float
                )
            except (ValueError, np.linalg.LinAlgError):
                solution = None
    if solution is not None and caught:
        # The one warning this solver raises is that it perturbed the coefficients;
        # its answer is then a solve of a nearby problem, not of this one.
        solution = None
    if solution is None or not np.all(np.isfinite(solution)):
        operator = np.kron(np.eye(size), matrix) + np.kron(matrix, np.eye(size))
        solution, *_ = np.linalg.lstsq(operator, -noise.reshape(-1), rcond=None)
        solution = solution.reshape(size, size)
        solver = "kronecker least squares (singular spectrum fallback)"
    covariance = (solution + solution.T) / 2.0
    # The residual is the equation itself, evaluated on the closed-form matrix: no
    # Kronecker operator is involved, so it measures the answer rather than the
    # solver's internal scaling.
    equation = matrix @ covariance + covariance @ matrix.T + noise
    scale = float(np.linalg.norm(noise))
    relative = float(np.linalg.norm(equation) / scale) if scale > 0.0 else np.inf
    return covariance, relative, solver


def lyapunov_covariance(jacobian: Any, diffusion: Any) -> tuple[np.ndarray, float]:
    """Solve ``J Σ + Σ Jᵀ + D = 0`` and return ``(Σ, relative residual)``.

    A stable Jacobian is solved directly (:func:`_lyapunov_covariance`), so the
    covariance is the one the historical probe reports. A singular Jacobian has no
    stationary covariance; the solve then falls back to the least-squares Kronecker
    form, which yields a residual instead of an exception or a silently regularised
    answer. The residual is relative to ``‖D‖``: it is the honest measure of "this
    equation was not satisfied" and is never clamped.

    Raises
    ------
    ValueError
        If the Jacobian is not square, the diffusion has a different shape, or either
        is not finite.

    """
    covariance, relative, _ = _lyapunov_covariance(jacobian, diffusion)
    return covariance, relative


def zero_frequency_gain(jacobian: Any, gradient: Any) -> tuple[np.ndarray, float]:
    """Return the static susceptibility row ``gᵀ(-J)⁻¹`` and its norm.

    A singular ``J`` is solved in the least-squares sense and reported as
    non-finite: a zero-frequency response that does not exist must not be
    presented as a large one.
    """
    matrix = np.asarray(jacobian, dtype=float)
    vector = np.asarray(gradient, dtype=float).reshape(-1)
    if matrix.shape[0] != vector.size:
        raise ValueError(
            f"the gradient has {vector.size} entries for a {matrix.shape[0]}x"
            f"{matrix.shape[1]} Jacobian"
        )
    try:
        response = np.linalg.solve(-matrix, np.eye(matrix.shape[0])).T @ vector
        finite = bool(np.all(np.isfinite(response)))
    except np.linalg.LinAlgError:
        response = np.full(vector.size, np.nan)
        finite = False
    norm = float(np.linalg.norm(response)) if finite else float("inf")
    return response, norm


def noise_timescales(
    jacobian: Any, covariance: Any, gradient: Any
) -> tuple[float, float]:
    """Return ``(integrated noise intensity, correlation time)`` of the readout.

    ``S(0) = 2 gᵀ(-J)⁻¹Σg`` and ``τ_c = gᵀ(-J)⁻¹Σg / (gᵀΣg)``. A singular ``J`` or
    a vanishing variance yields ``inf``/``nan`` rather than a substituted small
    denominator.
    """
    matrix = np.asarray(jacobian, dtype=float)
    vector = np.asarray(gradient, dtype=float).reshape(-1)
    variance = projected_intensity(covariance, vector, vector)
    try:
        # gᵀ(-J)⁻¹ = ((-J)⁻ᵀ g)ᵀ, so one transposed solve gives the whole integral.
        stationary_gain = np.linalg.solve(-matrix.T, vector)
        integral = float(stationary_gain @ np.asarray(covariance, dtype=float) @ vector)
        integrated = 2.0 * integral
        correlation_time = (
            float(integral / variance) if variance != 0.0 else float("nan")
        )
    except np.linalg.LinAlgError:
        integrated = float("inf")
        correlation_time = float("nan")
    return integrated, correlation_time


def assess_stochastic_state(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    model: Any | None = None,
    stationary_covariance: Any | None = None,
    mode: CriticalMode | None = None,
    config: StochasticAssessmentConfig | None = None,
) -> StochasticAssessment:
    """Assess one state's linear-noise validity for one observable.

    ``diffusion`` is the noise covariance in the coordinates the adapter's
    Jacobian acts on (the canonical Hermitian basis for a CAM moment model). When
    it is omitted, ``model.cam_diffusion`` is transformed through
    :func:`~qphase_cam.core.noise.canonical_sample_matrix_covariance`, which is
    the same covariance the ``stochastic_validity`` postprocessor uses.

    ``stationary_covariance`` and ``mode`` are optional *precomputed* inputs for a
    caller that assesses several readouts at one state and does not want to repeat the
    Lyapunov solve or the eigenproblem per readout. They are verified here, not
    trusted: the covariance must match the Jacobian's shape, be finite, be positive
    semidefinite and satisfy the Lyapunov equation to the configured residual, and the
    mode must match the coordinate count with finite eigenvectors. The provenance of
    both is recorded in ``method``.

    Raises
    ------
    ValueError
        If neither a diffusion nor a model providing ``cam_diffusion`` is given,
        or the inputs are not the shapes the adapter describes.

    """
    settings = config or StochasticAssessmentConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.validity.assess_stochastic_state",
        "quadrature": "lyapunov",
        "noise_semantics": "linear_diffusion_covariance",
        "lyapunov_convention": "J Sigma + Sigma J^T + D = 0",
        "observable": observable.identity.model_dump(mode="json"),
        "critical_mode_rule": settings.critical_mode_rule,
    }
    diagnostics: dict[str, Any] = {}

    def report(
        status: CAMOutcomeStatus,
        reason: str | None,
        **updates: Any,
    ) -> StochasticAssessment:
        payload: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "critical_mode_resolved": diagnostics.get("critical_mode_resolved", False),
            "representation_supported": diagnostics.get(
                "representation_supported", False
            ),
            "method": dict(method),
            # A copy: a later diagnostic must never rewrite a returned report.
            "diagnostics": dict(diagnostics),
        }
        payload.update(updates)
        return StochasticAssessment(**payload)

    if point.size == 0 or not np.all(np.isfinite(point)):
        return report(CAMOutcomeStatus.NOT_CONVERGED, "the state is not finite")
    layout = getattr(adapter, "moment_layout", None)
    representation_supported = layout in _SUPPORTED_MOMENT_LAYOUTS
    diagnostics["moment_layout"] = layout
    diagnostics["representation_supported"] = representation_supported
    if not representation_supported:
        return report(
            CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
            f"moment layout {layout!r} has no physical diffusion to assess",
        )
    anomalous_provider = getattr(model, "cam_anomalous_diffusion", None)
    anomalous_norm: float | None = None
    if callable(anomalous_provider):
        anomalous_norm = float(
            np.linalg.norm(
                np.asarray(
                    anomalous_provider(
                        np.asarray(adapter.state_matrix(point, parameter_map)),
                        parameter_map,
                    )
                )
            )
        )
        diagnostics["anomalous_diffusion_norm"] = anomalous_norm
        if anomalous_norm > settings.covariance_tolerance:
            return report(
                CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
                "the model declares anomalous diffusion; a linear noise assessment "
                "does not apply",
                anomalous_diffusion_norm=anomalous_norm,
            )

    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        return report(
            CAMOutcomeStatus.NOT_CONVERGED, "the dynamics Jacobian is not finite"
        )
    resolved_mode: CriticalMode | None
    if mode is not None:
        left_vector = np.asarray(mode.left, dtype=float).reshape(-1)
        right_vector = np.asarray(mode.right, dtype=float).reshape(-1)
        if left_vector.size != point.size or right_vector.size != point.size:
            raise ValueError(
                "the precomputed critical mode has "
                f"{left_vector.size}/{right_vector.size} coordinates for a "
                f"{point.size}-coordinate state"
            )
        if not np.all(np.isfinite(left_vector)) or not np.all(
            np.isfinite(right_vector)
        ):
            raise ValueError("the precomputed critical mode is not finite")
        method["critical_mode_source"] = "precomputed"
        resolved_mode = mode
    else:
        method["critical_mode_source"] = "resolved"
        resolved_mode = critical_mode(
            jacobian,
            overlap_tolerance=settings.overlap_tolerance,
            rule=settings.critical_mode_rule,
        )
    mode = resolved_mode
    if mode is None:
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the critical mode could not be resolved (complex or overlapping "
            "left/right eigenvectors)",
        )
    diagnostics["critical_mode_resolved"] = True
    diagnostics["jacobian"] = jacobian.tolist()
    diagnostics["jacobian_eigenvalues"] = [
        [float(np.real(value)), float(np.imag(value))]
        for value in np.linalg.eigvals(jacobian)
    ]
    diagnostics["critical_mode_left"] = [float(value) for value in mode.left]
    diagnostics["critical_mode_right"] = [float(value) for value in mode.right]
    mode_fields: dict[str, Any] = {
        "critical_eigenvalue_real": float(np.real(mode.value)),
        "critical_eigenvalue_imag": float(np.imag(mode.value)),
        "noncritical_spectral_gap": mode.spectral_gap,
        "critical_mode_condition_number": mode.condition_number,
        "eigenvector_condition_number": mode.eigenvector_condition_number,
    }
    if float(np.real(mode.value)) > settings.critical_eigenvalue_tolerance:
        return report(
            CAMOutcomeStatus.UNSTABLE,
            "the slowest mode grows, so the state has no stationary noise",
            **mode_fields,
        )

    noise = diffusion
    if noise is None:
        if model is None or not callable(getattr(model, "cam_diffusion", None)):
            raise ValueError(
                "an assessment needs either the diffusion covariance or a model "
                "that implements cam_diffusion(state, params)"
            )
        from .core.noise import canonical_sample_matrix_covariance

        state_matrix = np.asarray(adapter.state_matrix(point, parameter_map))
        noise = canonical_sample_matrix_covariance(
            state_matrix,
            model.cam_diffusion(state_matrix, parameter_map),
        )
    covariance_of_noise = np.asarray(noise, dtype=float)
    if covariance_of_noise.shape != jacobian.shape:
        raise ValueError(
            f"the diffusion shape {covariance_of_noise.shape} does not match the "
            f"Jacobian shape {jacobian.shape}"
        )
    if not np.all(np.isfinite(covariance_of_noise)):
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the diffusion covariance is not finite",
            anomalous_diffusion_norm=anomalous_norm,
            **mode_fields,
        )
    minimum_noise, noise_spectrum = minimum_eigenvalue(covariance_of_noise)
    diagnostics["diffusion_covariance"] = covariance_of_noise.tolist()
    diffusion_is_psd = minimum_noise >= -settings.covariance_tolerance
    diagnostics["diffusion_eigenvalues"] = [float(value) for value in noise_spectrum]
    noise_fields: dict[str, Any] = {
        "diffusion_is_psd": diffusion_is_psd,
        "diffusion_minimum_eigenvalue": minimum_noise,
        "anomalous_diffusion_norm": anomalous_norm,
    }
    if not diffusion_is_psd:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"the diffusion covariance is not positive semidefinite "
            f"(minimum eigenvalue {minimum_noise:.3e})",
            **mode_fields,
            **noise_fields,
        )

    if stationary_covariance is None:
        covariance, residual, solver = _lyapunov_covariance(
            jacobian, covariance_of_noise
        )
        method["lyapunov_source"] = "solved"
    else:
        covariance = np.asarray(stationary_covariance, dtype=float)
        if covariance.shape != jacobian.shape:
            raise ValueError(
                f"the precomputed stationary covariance has shape {covariance.shape} "
                f"for a {jacobian.shape[0]}-coordinate Jacobian"
            )
        if not np.all(np.isfinite(covariance)):
            return report(
                CAMOutcomeStatus.STOCHASTIC_INVALID,
                "the precomputed stationary covariance is not finite",
                **mode_fields,
                **noise_fields,
            )
        equation = jacobian @ covariance + covariance @ jacobian.T + covariance_of_noise
        scale = float(np.linalg.norm(covariance_of_noise))
        residual = (
            float(np.linalg.norm(equation) / scale) if scale > 0.0 else np.inf
        )
        solver = "precomputed"
        method["lyapunov_source"] = "precomputed"
        if residual > settings.lyapunov_relative_residual_tolerance:
            return report(
                CAMOutcomeStatus.STOCHASTIC_INVALID,
                "the precomputed stationary covariance does not solve the Lyapunov "
                f"equation (relative residual {residual:.3e})",
                **mode_fields,
                **noise_fields,
                lyapunov_solution_finite=True,
                lyapunov_relative_residual=residual,
            )
    solution_finite = bool(np.all(np.isfinite(covariance)))
    method["lyapunov_solver"] = solver
    diagnostics["lyapunov_covariance"] = covariance.tolist()
    diagnostics["lyapunov_relative_residual"] = residual
    diagnostics["lyapunov_solver"] = solver
    lyapunov_fields: dict[str, Any] = {
        "lyapunov_solution_finite": solution_finite,
        "lyapunov_relative_residual": residual,
    }
    if not solution_finite:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the stationary covariance solution is not finite",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
        )
    minimum_covariance, covariance_spectrum = minimum_eigenvalue(covariance)
    lyapunov_is_psd = minimum_covariance >= -settings.covariance_tolerance
    diagnostics["lyapunov_eigenvalues"] = [
        float(value) for value in covariance_spectrum
    ]
    lyapunov_fields.update(
        {
            "lyapunov_covariance_trace": float(np.trace(covariance)),
            "lyapunov_covariance_minimum_eigenvalue": minimum_covariance,
            "lyapunov_is_psd": lyapunov_is_psd,
        }
    )
    if residual > settings.lyapunov_relative_residual_tolerance:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"no stationary covariance solves the Lyapunov equation (relative "
            f"residual {residual:.3e})",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
        )
    if not lyapunov_is_psd:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            f"the stationary covariance is not positive semidefinite (minimum "
            f"eigenvalue {minimum_covariance:.3e})",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
        )

    try:
        observable_value = float(observable.value(adapter, point, parameter_map))
        record = observable.gradient(adapter, point, parameter_map)
    except (ValueError, TypeError, FloatingPointError) as exc:
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            f"the observable cannot be evaluated at this state: {exc}",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
        )
    gradient = np.asarray(record.gradient, dtype=float)
    observable_fields: dict[str, Any] = {
        "observable_value": observable_value,
        "observable_gradient": tuple(float(value) for value in gradient),
        "derivative": record,
    }
    diagnostics["derivative"] = record.model_dump(mode="json")
    if not np.isfinite(observable_value):
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the observable value is not finite",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )
    if gradient.size != jacobian.shape[0]:
        raise ValueError(
            f"the observable gradient has {gradient.size} entries for a "
            f"{jacobian.shape[0]}-coordinate state"
        )
    gradient_norm = float(np.linalg.norm(gradient))
    if not np.all(np.isfinite(gradient)) or gradient_norm == 0.0:
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the observable gradient is not resolved, so the projection would be "
            "an artefact of the readout",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )
    if record.repeat_relative_error is not None and not np.isfinite(
        record.repeat_relative_error
    ):
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the observable gradient did not converge under step refinement",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )
    variance = projected_intensity(covariance, gradient, gradient)
    positive_variance = variance > settings.covariance_tolerance
    diagnostics["projected_variance"] = variance
    observable_fields.update(
        {
            "projected_variance": variance,
            "projected_variance_is_positive": positive_variance,
        }
    )
    if not positive_variance:
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            f"the projected variance is not positive ({variance:.3e}); the readout "
            "carries no stationary noise",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )

    response, response_norm = zero_frequency_gain(jacobian, gradient)
    response_finite = bool(np.all(np.isfinite(response)))
    diagnostics["zero_frequency_response"] = [float(value) for value in response]
    noise_fields.update(
        {
            "zero_frequency_response_norm": response_norm,
            "zero_frequency_response_finite": response_finite,
        }
    )
    if not response_finite or not response_norm > settings.response_tolerance:
        return report(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the zero-frequency response does not exist or is below the "
            "resolution tolerance",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )

    integrated, correlation_time = noise_timescales(jacobian, covariance, gradient)
    integrated_finite = bool(np.isfinite(integrated))
    timescale_positive = bool(np.isfinite(correlation_time) and correlation_time > 0.0)
    diagnostics["integrated_noise_intensity"] = integrated
    diagnostics["correlation_time"] = correlation_time
    noise_fields.update(
        {
            "integrated_noise_intensity": integrated,
            "correlation_time": correlation_time,
            "integrated_noise_intensity_finite": integrated_finite,
            "correlation_time_positive": timescale_positive,
        }
    )
    if not integrated_finite or integrated <= 0.0:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the integrated noise intensity is not finite and positive",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )
    if not timescale_positive:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the correlation time is not finite and positive",
            **mode_fields,
            **noise_fields,
            **lyapunov_fields,
            **observable_fields,
        )
    return report(
        CAMOutcomeStatus.VALID,
        None,
        **mode_fields,
        **noise_fields,
        **lyapunov_fields,
        **observable_fields,
    )


#: What a row status means in an assessment annotation.
_STATUS_SEMANTICS = (
    "one CAMOutcomeStatus per assessment; a failed assessment keeps its reason and "
    "every diagnostic computed before the failing check"
)

#: Tri-state encoding of a boolean an assessment could not decide.
_TRISTATE_SEMANTICS = (
    "-1 means the check never ran, 0 means it ran and failed, 1 means it passed; "
    "a boolean is never collapsed to False when the answer is unknown"
)

#: Numeric fields the annotation carries, in column order.
_NUMERIC_FIELDS: tuple[str, ...] = (
    "critical_eigenvalue_real",
    "critical_eigenvalue_imag",
    "noncritical_spectral_gap",
    "critical_mode_condition_number",
    "eigenvector_condition_number",
    "diffusion_minimum_eigenvalue",
    "lyapunov_relative_residual",
    "lyapunov_covariance_trace",
    "lyapunov_covariance_minimum_eigenvalue",
    "observable_value",
    "projected_variance",
    "zero_frequency_response_norm",
    "integrated_noise_intensity",
    "correlation_time",
    "anomalous_diffusion_norm",
)

#: Boolean fields the annotation carries as tri-state integers.
_TRISATE_FIELDS: tuple[str, ...] = (
    "critical_mode_resolved",
    "representation_supported",
    "diffusion_is_psd",
    "lyapunov_solution_finite",
    "lyapunov_is_psd",
    "projected_variance_is_positive",
    "zero_frequency_response_finite",
    "integrated_noise_intensity_finite",
    "correlation_time_positive",
)


def build_stochastic_assessment_annotation(
    assessments: Sequence[StochasticAssessment],
    *,
    target_identity: Sequence[str],
    case_identity: Sequence[str] | None = None,
    annotation_kind: str = "stochastic_validity",
    target_kind: str = "state",
    method_metadata: Sequence[Mapping[str, Any]] | None = None,
) -> Dataset:
    """Turn assessments into a typed annotation product.

    One row per assessment, in order. The status column is the shared
    ``CAMOutcomeStatus`` vocabularly, so a failed assessment is a status with a
    ``reason``, never a silently missing row. Unavailable numbers are ``NaN`` and
    undecided booleans are ``-1``; nothing is clamped or defaulted.

    Raises
    ------
    ValueError
        If the row counts disagree or the assessment list is empty.

    """
    cases = list(assessments)
    if not cases:
        raise ValueError("an assessment annotation needs at least one assessment")
    identities = [str(value) for value in target_identity]
    if len(identities) != len(cases):
        raise ValueError(
            f"{len(identities)} target identities for {len(cases)} assessments"
        )
    fields: dict[str, Any] = {
        "reason": np.asarray([item.reason or "" for item in cases], dtype=str),
        "derivative_method": np.asarray(
            [
                "" if item.derivative is None else item.derivative.method
                for item in cases
            ],
            dtype=str,
        ),
    }
    for name in _NUMERIC_FIELDS:
        fields[name] = numeric_column([getattr(item, name) for item in cases])
    for name in _TRISATE_FIELDS:
        fields[name] = tristate_column([getattr(item, name) for item in cases])
    fields["derivative_condition"] = np.asarray(
        [
            np.nan
            if item.derivative is None or item.derivative.condition is None
            else float(item.derivative.condition)
            for item in cases
        ],
        dtype=float,
    )
    fields["derivative_repeat_relative_error"] = np.asarray(
        [
            np.nan
            if item.derivative is None or item.derivative.repeat_relative_error is None
            else float(item.derivative.repeat_relative_error)
            for item in cases
        ],
        dtype=float,
    )
    gradients = [
        None
        if item.observable_gradient is None
        else np.asarray(item.observable_gradient, dtype=float)
        for item in cases
    ]
    if all(gradient is not None for gradient in gradients):
        width = int(
            max(len(gradient) for gradient in gradients if gradient is not None)
        )
        fields["observable_gradient"] = np.asarray(
            [
                np.pad(
                    np.asarray(gradient, dtype=float),
                    (0, width - len(gradient)),
                    constant_values=np.nan,
                )
                for gradient in gradients
                if gradient is not None
            ],
            dtype=float,
        )
    metadata = (
        [
            {
                **dict(item.method),
                "tristate_semantics": _TRISTATE_SEMANTICS,
                "status_semantics": _STATUS_SEMANTICS,
            }
            for item in cases
        ]
        if method_metadata is None
        else [dict(item) for item in method_metadata]
    )
    if len(metadata) != len(cases):
        raise ValueError(
            f"{len(metadata)} method metadata entries for {len(cases)} assessments"
        )
    return build_annotation_product(
        annotation_kind=annotation_kind,
        target_kind=target_kind,
        target_identity=identities,
        case_identity=None if case_identity is None else list(case_identity),
        status=[item.status.value for item in cases],
        fields=fields,
        method_metadata=metadata,
    )
