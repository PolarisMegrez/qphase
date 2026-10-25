"""qphase_cam: the closure-risk service
---------------------------------------------------------
Closure risk asks "how much does the deterministic CAM equation misplace the
observable once the noise of the full system is included?". It is the second-order
correction of a linear-noise assessment: the Itô drift the deterministic equation
drops (the *closure contraction*), the state displacement that drift implies, and
the curvature of the readout (its Jensen bias). Together they are the closure bias
of one observable at one state.

The numbers are the ones
``scripts/tools/screen_kerr_full_3mode_closure_risk.py`` computes, generalised
away from that script's model and readout:

* ``Σ`` and the zero-frequency response come from
  :func:`~qphase_cam.validity.assess_stochastic_state` — this service never
  re-solves the Lyapunov equation or re-tests PSD.
* the closure contraction is ``b = ½ Σ_k (∂J/∂x_k) Σ[:, k]``, evaluated with a
  central difference of the adapter Jacobian;
* the displacement is ``δx = (-J)⁻¹ b``;
* the drift bias is ``g·δx``, the Jensen bias is ``½ Σ_k λ_k ∂²f/∂u_k²`` over the
  positive eigen-directions of ``Σ``, and the closure bias is their sum;
* the shape fraction is how much of ``δR`` is orthogonal to the radial direction
  of the state matrix (0 = pure amplitude, 1 = pure shape);
* the soft-mode quantities use the critical left/right eigenvectors, so
  ``σ_x = sqrt(leftᵀ D left / (2γ))`` and the Kubo numbers say whether the linear
  response dominates the noise.

Nothing is a verdict about acceptance: the Kubo numbers label the regime with a
configured threshold and the status only says whether the numbers could be
computed. Screening a campaign is the caller's job, with its own thresholds.

``ClosureAssessment`` is a frozen report; a failure keeps every quantity computed
before the failing check, with ``reason`` naming it. The status vocabulary is the
shared :class:`~qphase_cam.outcomes.CAMOutcomeStatus`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qphase.data import Dataset

from .core.noise import canonical_sample_matrix_covariance
from .data import build_annotation_product
from .observable import CAMObservable, DerivativeRecord, coordinate_scale
from .outcomes import CAMOutcomeStatus
from .validity import (
    StochasticAssessment,
    StochasticAssessmentConfig,
    _lyapunov_covariance,
    assess_stochastic_state,
    critical_mode,
    minimum_eigenvalue,
    numeric_column,
    projected_intensity,
    tristate_column,
)

__all__ = [
    "ClosureAssessment",
    "ClosureAssessmentConfig",
    "ClosureState",
    "assess_closure",
    "closure_state",
    "build_closure_assessment_annotation",
    "closure_contraction",
    "closure_shape_fraction",
    "observable_jensen_bias",
]


class ClosureAssessmentConfig(BaseModel):
    """Numeric controls of one closure assessment.

    The stochastic tolerances are nested because this service reuses the
    linear-noise assessment unchanged; the closure fields are the finite-difference
    steps and the regime label. ``kubo_threshold`` only names the regime — it never
    rejects or ranks a state.
    """

    model_config = ConfigDict(extra="forbid")

    stochastic: StochasticAssessmentConfig = Field(
        default_factory=StochasticAssessmentConfig
    )
    closure_relative_step: float = Field(
        default=1e-5,
        gt=0.0,
        description="Relative step for the Jacobian derivative (the script's 2e-5 "
        "absolute step scaled by max(|x_i|, 1))",
    )
    jensen_relative_step: float = Field(
        default=1e-4,
        gt=0.0,
        description="Relative step of the readout curvature, scaled by max(1, |x|)",
    )
    jensen_eigenvalue_tolerance: float = Field(
        default=1e-12,
        ge=0.0,
        description="Covariance eigenvalues below this are not expanded",
    )
    kubo_threshold: float = Field(
        default=1.0,
        gt=0.0,
        description="Label threshold for the validity domain (never a filter)",
    )
    physicality_tolerance: float = Field(
        default=1e-9,
        ge=0.0,
        description="Tolerance for the physical (Hermitian PSD) spectrum",
    )
    evaluate_jensen: bool = Field(
        default=True, description="Whether the readout curvature is evaluated"
    )
    evaluate_shape: bool = Field(
        default=True, description="Whether the closure shape fraction is evaluated"
    )


class ClosureAssessment(BaseModel):
    """The full report of one closure assessment.

    ``stochastic`` is the linear-noise assessment this report builds on, so the
    covariance, response, projection and derivative record travel with the closure
    numbers instead of being recomputed or dropped.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    stochastic: StochasticAssessment

    # -- closure drift ----------------------------------------------------
    closure_contraction_norm: float | None = None
    closure_displacement: tuple[float, ...] | None = None
    closure_displacement_norm: float | None = None
    drift_bias: float | None = None
    jensen_bias: float | None = None
    closure_bias_total: float | None = None
    closure_shape_fraction: float | None = None

    # -- soft-mode noise and regime --------------------------------------
    soft_mode_rate: float | None = None
    soft_mode_sigma: float | None = None
    soft_sigma: float | None = None
    soft_kubo: float | None = None
    full_kubo: float | None = None
    validity_domain: str | None = None

    # -- conditioning and method -----------------------------------------
    jacobian_condition_number: float | None = None
    covariance_condition_number: float | None = None
    approximation_order: int = 2
    closure_derivative: DerivativeRecord | None = None
    jensen_step: float | None = None
    jensen_repeat_relative_error: float | None = None
    jensen_eigenvalue_count: int | None = None

    # -- physicality ------------------------------------------------------
    physical_minimum_eigenvalue: float | None = None
    physical_is_psd: bool | None = None

    method: dict[str, Any] = Field(default_factory=dict)
    diagnostics: dict[str, Any] = Field(default_factory=dict)


def closure_contraction(
    jacobian_at: Any,
    state: Any,
    covariance: Any,
    *,
    relative_step: float = 1e-5,
) -> tuple[np.ndarray, DerivativeRecord]:
    """Return the Itô closure drift ``½ Σ_k (∂J/∂x_k) Σ[:, k]`` and its record.

    ``jacobian_at`` is a callable ``x -> J(x)``; each coordinate is differentiated
    by a central difference whose step follows that coordinate's scale, and the
    contraction is recomputed at half the step so the record says whether the
    derivative is resolved. An unresolved derivative is reported through
    ``repeat_relative_error``, never smoothed away.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    covariance_matrix = np.asarray(covariance, dtype=float)
    if covariance_matrix.shape != (point.size, point.size):
        raise ValueError(
            f"the covariance shape {covariance_matrix.shape} does not match the "
            f"{point.size}-coordinate state"
        )
    scale = coordinate_scale(point)
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("the coordinate scale must be finite and positive")

    def contraction_at(step: np.ndarray) -> np.ndarray:
        total = np.zeros(point.size)
        for index in range(point.size):
            offset = np.zeros(point.size)
            offset[index] = step[index]
            plus = np.asarray(jacobian_at(point + offset), dtype=float)
            minus = np.asarray(jacobian_at(point - offset), dtype=float)
            if plus.shape != covariance_matrix.shape or minus.shape != (
                covariance_matrix.shape
            ):
                raise ValueError(
                    "the Jacobian shape changed under a finite-difference step: "
                    f"{plus.shape} and {minus.shape} for {covariance_matrix.shape}"
                )
            derivative = (plus - minus) / (2.0 * step[index])
            total += 0.5 * derivative @ covariance_matrix[:, index]
        return total

    step = relative_step * scale
    contraction = contraction_at(step)
    refined = contraction_at(step / 2.0)
    norm = float(np.linalg.norm(contraction))
    repeat_error = (
        float(np.linalg.norm(contraction - refined) / norm)
        if norm > 0.0 and np.all(np.isfinite(contraction))
        else None
    )
    magnitude = np.abs(contraction)
    nonzero = magnitude[magnitude > 0.0]
    condition = (
        float(np.max(nonzero) / np.min(nonzero))
        if nonzero.size == magnitude.size and nonzero.size
        else None
    )
    record = DerivativeRecord(
        method="central_difference",
        order=2,
        gradient=tuple(float(value) for value in contraction),
        step=tuple(float(value) for value in step),
        scale=tuple(float(value) for value in scale),
        condition=condition,
        repeat_relative_error=repeat_error,
        metadata={
            "quantity": "closure_drift_contraction",
            "identity": "b = 1/2 sum_k (dJ/dx_k) Sigma[:, k]",
            "relative_step": float(relative_step),
            "repeat_check": "recomputed at half the step",
        },
    )
    return contraction, record


def observable_jensen_bias(
    value_at: Any,
    state: Any,
    covariance: Any,
    *,
    relative_step: float = 1e-4,
    eigenvalue_tolerance: float = 1e-12,
) -> tuple[float, float, float | None, int]:
    """Return ``(Jensen bias, step, repeat relative error, expanded count)``.

    ``value_at`` is a callable ``x -> f(x)``. The bias is
    ``½ Σ_k λ_k ∂²f/∂u_k²`` over the *positive* eigen-directions of the
    covariance: a zero or negative eigenvalue contributes no noise direction, so
    expanding it would invent curvature the noise never samples. The curvature is a
    central second difference at ``relative_step * max(1, |x|)`` and is recomputed
    at half the step.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    covariance_matrix = np.asarray(covariance, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    scale = max(1.0, float(np.linalg.norm(point)))
    step = relative_step * scale
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError(f"the Jensen step must be finite and positive; got {step}")
    tolerance = eigenvalue_tolerance * max(
        float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0, 1.0
    )

    def bias_at(width: float) -> float:
        center = float(value_at(point))
        total = 0.0
        for eigenvalue, direction in zip(eigenvalues, eigenvectors.T, strict=True):
            if eigenvalue <= tolerance:
                continue
            plus = float(value_at(point + width * direction))
            minus = float(value_at(point - width * direction))
            curvature = (plus - 2.0 * center + minus) / width**2
            total += 0.5 * float(eigenvalue) * curvature
        return total

    bias = bias_at(step)
    refined = bias_at(step / 2.0)
    denominator = abs(bias)
    repeat_error = (
        float(abs(bias - refined) / denominator)
        if denominator > 0.0 and np.isfinite(bias)
        else None
    )
    expanded = int(np.count_nonzero(eigenvalues > tolerance))
    return bias, step, repeat_error, expanded


def closure_shape_fraction(matrix: Any, displacement_matrix: Any) -> float | None:
    """Return how much of ``δR`` is orthogonal to the radial direction.

    ``0`` is a purely amplitude (radial) change, ``1`` a purely shape-changing one.
    ``None`` when either matrix has a vanishing norm or the state matrix is not
    positive definite in the ⟨R, R⟩ sense: the fraction is then undefined rather
    than zero.
    """
    state_matrix = np.asarray(matrix, dtype=complex)
    delta = np.asarray(displacement_matrix, dtype=complex)
    if state_matrix.shape != delta.shape:
        raise ValueError(
            f"state matrix shape {state_matrix.shape} does not match the "
            f"displacement shape {delta.shape}"
        )
    denominator = float(np.real(np.vdot(state_matrix, state_matrix)))
    delta_norm = float(np.linalg.norm(delta))
    if denominator <= 0.0 or delta_norm <= 0.0:
        return None
    radial = float(np.real(np.vdot(state_matrix, delta))) / denominator * state_matrix
    return float(np.linalg.norm(delta - radial) / delta_norm)


@dataclass(frozen=True)
class ClosureState:
    """The state-level closure quantities of one state, computed once.

    This is a *precomputed value*, not a runtime or a product: it carries the Jacobian,
    the canonical noise, the critical mode, the stationary covariance, the closure
    contraction/displacement/shape and the state's physicality, so several readouts at
    the same state share one Lyapunov solve, one eigenproblem and one contraction. The
    readout-specific work — its gradient, its projection, its drift and Jensen bias and
    its Kubo numbers — is *not* here and is evaluated per observable.

    ``status`` reports a state-level failure (an unresolved mode, an unusable noise
    covariance, a failed Lyapunov solve, a non-finite contraction or a singular
    displacement). Physicality is recorded but not turned into a verdict here, because
    :func:`assess_closure` decides it at the point of its own flow.
    """

    point: np.ndarray
    params: dict[str, Any]
    config: ClosureAssessmentConfig
    jacobian: np.ndarray | None
    noise_covariance: np.ndarray | None
    stationary_covariance: np.ndarray | None
    mode: Any
    covariance_minimum_eigenvalue: float | None
    covariance_spectrum: tuple[float, ...]
    covariance_condition_number: float | None
    jacobian_condition_number: float | None
    closure_contraction: np.ndarray | None
    closure_derivative: DerivativeRecord | None
    closure_displacement: np.ndarray | None
    closure_shape_fraction: float | None
    physical_minimum_eigenvalue: float | None
    physical_is_psd: bool | None
    soft_mode_rate: float | None
    projected_noise: float | None
    soft_mode_sigma: float | None
    status: CAMOutcomeStatus
    reason: str | None
    diagnostics: dict[str, Any]
    method: dict[str, Any]


def closure_state(
    adapter: Any,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    model: Any | None = None,
    config: ClosureAssessmentConfig | None = None,
) -> ClosureState:
    """Return the state-level closure quantities of one state.

    The checks are the ones :func:`~qphase_cam.validity.assess_stochastic_state` makes
    for the same state — a finite Jacobian, a resolved critical mode, a finite and
    positive-semidefinite noise covariance, and a stationary covariance that is finite,
    positive semidefinite and solves the Lyapunov equation to the configured residual —
    plus the closure contraction and displacement. A failure is reported through
    ``status``/``reason`` with the fields computed before it, and the readouts that
    share this state all see the same verdict.
    """
    settings = config or ClosureAssessmentConfig()
    model_copy = settings.model_copy(deep=True)
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    diagnostics: dict[str, Any] = {}
    method: dict[str, Any] = {
        "service": "qphase_cam.closure.closure_state",
        "critical_mode_rule": settings.stochastic.critical_mode_rule,
        "noise_covariance": "canonical diffusion covariance",
        "lyapunov_convention": "J Sigma + Sigma J^T + D = 0",
        "closure_definition": "b = 1/2 sum_k (dJ/dx_k) Sigma[:, k]; dx = (-J)^-1 b",
        "shared": "one Jacobian/noise/mode/Lyapunov/contraction per state",
    }

    def failed(
        status: CAMOutcomeStatus, reason: str, **updates: Any
    ) -> ClosureState:
        fields: dict[str, Any] = {
            "point": point,
            "params": parameter_map,
            "config": model_copy,
            "jacobian": None,
            "noise_covariance": None,
            "stationary_covariance": None,
            "mode": None,
            "covariance_minimum_eigenvalue": None,
            "covariance_spectrum": (),
            "covariance_condition_number": None,
            "jacobian_condition_number": None,
            "closure_contraction": None,
            "closure_derivative": None,
            "closure_displacement": None,
            "closure_shape_fraction": None,
            "physical_minimum_eigenvalue": None,
            "physical_is_psd": None,
            "soft_mode_rate": None,
            "projected_noise": None,
            "soft_mode_sigma": None,
            "status": status,
            "reason": reason,
            "diagnostics": dict(diagnostics),
            "method": dict(method),
        }
        fields.update(updates)
        return ClosureState(**fields)

    if point.size == 0 or not np.all(np.isfinite(point)):
        return failed(CAMOutcomeStatus.NOT_CONVERGED, "the state is not finite")
    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        return failed(
            CAMOutcomeStatus.NOT_CONVERGED, "the dynamics Jacobian is not finite"
        )
    diagnostics["jacobian"] = jacobian.tolist()
    if diffusion is None:
        if model is None or not callable(getattr(model, "cam_diffusion", None)):
            raise ValueError(
                "a closure state needs either the noise covariance or a model"
            )
        matrix = np.asarray(adapter.state_matrix(point, parameter_map), dtype=complex)
        noise = canonical_sample_matrix_covariance(
            matrix, model.cam_diffusion(matrix, parameter_map)
        )
    else:
        noise = np.asarray(diffusion, dtype=float)
    if noise.shape != jacobian.shape:
        raise ValueError(
            f"the diffusion shape {noise.shape} does not match the Jacobian shape "
            f"{jacobian.shape}"
        )
    if not np.all(np.isfinite(noise)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the diffusion covariance is not finite",
            jacobian=jacobian,
            noise_covariance=noise,
        )
    minimum_noise, noise_spectrum = minimum_eigenvalue(noise)
    diagnostics["diffusion_eigenvalues"] = [float(value) for value in noise_spectrum]
    if minimum_noise < -settings.stochastic.covariance_tolerance:
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the diffusion covariance is not positive semidefinite "
            f"(minimum eigenvalue {minimum_noise:.3e})",
            jacobian=jacobian,
            noise_covariance=noise,
        )
    mode = critical_mode(
        jacobian,
        overlap_tolerance=settings.stochastic.overlap_tolerance,
        rule=settings.stochastic.critical_mode_rule,
    )
    if mode is None:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            "the critical mode could not be resolved (complex or overlapping "
            "left/right eigenvectors)",
            jacobian=jacobian,
            noise_covariance=noise,
        )
    diagnostics["critical_mode_left"] = [float(value) for value in mode.left]
    diagnostics["critical_mode_right"] = [float(value) for value in mode.right]
    covariance, residual, solver = _lyapunov_covariance(jacobian, noise)
    diagnostics["lyapunov_relative_residual"] = residual
    diagnostics["lyapunov_solver"] = solver
    method["lyapunov_solver"] = solver
    if not np.all(np.isfinite(covariance)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the stationary covariance solution is not finite",
            mode=mode,
            jacobian=jacobian,
            noise_covariance=noise,
        )
    covariance_minimum, covariance_spectrum = minimum_eigenvalue(covariance)
    diagnostics["covariance_eigenvalues"] = [
        float(value) for value in covariance_spectrum
    ]
    if covariance_minimum < -settings.stochastic.covariance_tolerance:
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the stationary covariance is not positive semidefinite (minimum "
            f"eigenvalue {covariance_minimum:.3e})",
            mode=mode,
            stationary_covariance=covariance,
            covariance_minimum_eigenvalue=covariance_minimum,
            jacobian=jacobian,
            noise_covariance=noise,
        )
    if residual > settings.stochastic.lyapunov_relative_residual_tolerance:
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "no stationary covariance solves the Lyapunov equation (relative "
            f"residual {residual:.3e})",
            mode=mode,
            jacobian=jacobian,
            noise_covariance=noise,
        )
    left = np.asarray(mode.left, dtype=float)
    rate = -float(np.real(mode.value))
    projected_noise = projected_intensity(noise, left, left)
    sigma = (
        float(np.sqrt(projected_noise / (2.0 * rate)))
        if rate > 0.0 and projected_noise > 0.0
        else None
    )
    contraction, record = closure_contraction(
        lambda vector: adapter.jacobian(vector, parameter_map),
        point,
        covariance,
        relative_step=settings.closure_relative_step,
    )
    diagnostics["closure_contraction"] = [float(value) for value in contraction]
    if not np.all(np.isfinite(contraction)):
        return failed(
            CAMOutcomeStatus.NOT_CONVERGED,
            "the closure contraction is not finite",
            mode=mode,
            stationary_covariance=covariance,
            covariance_minimum_eigenvalue=covariance_minimum,
            covariance_spectrum=tuple(float(value) for value in covariance_spectrum),
            closure_derivative=record,
        )
    try:
        displacement = np.linalg.solve(-jacobian, contraction)
    except np.linalg.LinAlgError:
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the closure displacement is undefined: -J is singular",
            mode=mode,
            stationary_covariance=covariance,
            covariance_minimum_eigenvalue=covariance_minimum,
            covariance_spectrum=tuple(float(value) for value in covariance_spectrum),
            closure_contraction=contraction,
            closure_derivative=record,
        )
    if not np.all(np.isfinite(displacement)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the closure displacement is not finite",
            mode=mode,
            stationary_covariance=covariance,
            covariance_minimum_eigenvalue=covariance_minimum,
            covariance_spectrum=tuple(float(value) for value in covariance_spectrum),
            closure_contraction=contraction,
            closure_derivative=record,
        )
    diagnostics["closure_displacement"] = [float(value) for value in displacement]
    shape_fraction: float | None = None
    if settings.evaluate_shape:
        state_matrix = np.asarray(adapter.state_matrix(point, parameter_map))
        displacement_matrix = np.asarray(
            adapter.state_matrix(displacement, parameter_map)
        )
        shape_fraction = closure_shape_fraction(state_matrix, displacement_matrix)
        if shape_fraction is None:
            diagnostics["closure_shape_fraction_status"] = (
                "undefined: the displacement or the state matrix has a vanishing norm"
            )
    physical_minimum: float | None = None
    physical_is_psd: bool | None = None
    physical_provider = getattr(adapter, "physical_eigenvalues", None)
    if callable(physical_provider):
        try:
            physical = np.asarray(physical_provider(point, parameter_map), dtype=float)
        except (ValueError, RuntimeError) as exc:
            diagnostics["physicality_status"] = f"unavailable: {exc}"
        else:
            physical_minimum = float(np.min(physical))
            physical_is_psd = physical_minimum >= -settings.physicality_tolerance
    else:
        diagnostics["physicality_status"] = "unavailable: adapter has no spectrum"
    covariance_condition = (
        float(np.linalg.cond(covariance))
        if covariance_minimum > 0.0
        else float("inf")
    )
    return ClosureState(
        point=point,
        params=parameter_map,
        config=model_copy,
        jacobian=jacobian,
        noise_covariance=noise,
        stationary_covariance=covariance,
        mode=mode,
        covariance_minimum_eigenvalue=covariance_minimum,
        covariance_spectrum=tuple(float(value) for value in covariance_spectrum),
        covariance_condition_number=covariance_condition,
        jacobian_condition_number=float(np.linalg.cond(jacobian)),
        closure_contraction=contraction,
        closure_derivative=record,
        closure_displacement=displacement,
        closure_shape_fraction=shape_fraction,
        physical_minimum_eigenvalue=physical_minimum,
        physical_is_psd=physical_is_psd,
        soft_mode_rate=rate,
        projected_noise=projected_noise,
        soft_mode_sigma=sigma,
        status=CAMOutcomeStatus.VALID,
        reason=None,
        diagnostics=dict(diagnostics),
        method=dict(method),
    )


def _check_context(
    context: ClosureState,
    *,
    state: Any,
    params: Mapping[str, Any],
    config: ClosureAssessmentConfig | None,
) -> None:
    """Refuse a shared state that was computed for something else.

    A context is only reusable for the exact state, parameters and configuration it
    was built from: a different state of the same shape, a different parameter value or
    a different step would silently mix two assessments. Values are compared with
    ``np.array_equal``, so an array-like parameter equal to a float is accepted.
    """
    point = np.asarray(state, dtype=float).reshape(-1)
    if point.shape != context.point.shape or not np.array_equal(
        point, context.point
    ):
        raise ValueError(
            "the shared closure state was computed at a different state; build a new "
            "context for this one"
        )
    requested = dict(params or {})
    if set(requested) != set(context.params):
        raise ValueError(
            "the shared closure state was computed with different parameter names: "
            f"{sorted(set(requested) ^ set(context.params))}"
        )
    for name, value in requested.items():
        if not np.array_equal(np.asarray(value), np.asarray(context.params[name])):
            raise ValueError(
                f"the shared closure state was computed with a different value for "
                f"{name!r}"
            )
    if config is not None and config != context.config:
        raise ValueError(
            "an explicit config disagrees with the config the shared closure state was "
            "computed with"
        )


def assess_closure(
    adapter: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    diffusion: Any | None = None,
    model: Any | None = None,
    state_context: ClosureState | None = None,
    config: ClosureAssessmentConfig | None = None,
) -> ClosureAssessment:
    """Assess the closure bias of one observable at one state.

    The linear-noise part is delegated to
    :func:`~qphase_cam.validity.assess_stochastic_state`, which is handed the noise
    covariance, the stationary covariance and the critical mode of the shared
    :class:`ClosureState` — and verifies all three before using them. A state that
    fails that assessment returns the same status and every diagnostic it managed to
    compute, with no closure numbers invented on top.

    ``state_context`` reuses the state-level closure quantities of a previous call at
    the same state, so assessing three readouts costs one Jacobian, one noise
    covariance, one eigenproblem, one Lyapunov solve and one closure contraction
    instead of three. The shared path and the fresh path are the same code: this
    function always goes through :func:`closure_state`, and passing the context only
    skips recomputing it.
    """
    if state_context is not None:
        _check_context(state_context, state=state, params=params, config=config)
        settings = (
            state_context.config if config is None else config
        )
    else:
        settings = config or ClosureAssessmentConfig()
    context = state_context or closure_state(
        adapter,
        state=state,
        params=params,
        diffusion=diffusion,
        model=model,
        config=settings,
    )
    if context.status is CAMOutcomeStatus.VALID:
        stochastic = assess_stochastic_state(
            adapter,
            observable,
            state=state,
            params=params,
            diffusion=context.noise_covariance,
            stationary_covariance=context.stationary_covariance,
            mode=context.mode,
            config=settings.stochastic,
        )
    else:
        # The state pass already failed. The readout assessment is still run, on the
        # caller's own inputs, so the report carries the assessment's own reason
        # instead of a second-hand one.
        stochastic = assess_stochastic_state(
            adapter,
            observable,
            state=state,
            params=params,
            diffusion=diffusion,
            model=model,
            config=settings.stochastic,
        )
    method: dict[str, Any] = {
        "service": "qphase_cam.closure.assess_closure",
        "approximation": "ito_taylor_second_order",
        "closure_definition": "b = 1/2 sum_k (dJ/dx_k) Sigma[:, k]; dx = (-J)^-1 b",
        "jensen_definition": "1/2 sum_k lambda_k d2f/du_k^2 over positive lambda",
        "shape_definition": "||dR - radial(dR)|| / ||dR||",
        "kubo_definition": "kubo_full = sigma_full * tau_full; kubo_soft = "
        "|g.right| sigma_x / gamma",
        "observable": observable.identity.model_dump(mode="json"),
        "regime_threshold": float(settings.kubo_threshold),
    }
    diagnostics: dict[str, Any] = {"stochastic_status": stochastic.status.value}

    def report(
        status: CAMOutcomeStatus, reason: str | None, **updates: Any
    ) -> ClosureAssessment:
        payload: dict[str, Any] = {
            "status": status,
            "reason": reason,
            "stochastic": stochastic,
            "method": dict(method),
            "diagnostics": dict(diagnostics),
        }
        payload.update(updates)
        return ClosureAssessment(**payload)

    if stochastic.status is not CAMOutcomeStatus.VALID:
        # The shared state's own failure is the authoritative reason when it failed;
        # the readout assessment may fail for a second-hand reason of its own.
        if context.status is not CAMOutcomeStatus.VALID:
            return report(
                context.status,
                context.reason or "the closure state is not valid",
            )
        return report(
            stochastic.status,
            stochastic.reason or "the linear-noise assessment is not valid",
        )
    point = context.point
    jacobian = context.jacobian
    covariance = context.stationary_covariance
    noise = context.noise_covariance
    right = (
        np.asarray(context.mode.right, dtype=float)
        if context.mode is not None
        else np.zeros(0)
    )
    if jacobian is None or noise is None or covariance is None:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the closure state does not carry a Jacobian, a noise covariance and a "
            "stationary covariance",
        )
    if jacobian.shape != (point.size, point.size):
        raise ValueError(
            "the closure state does not record a Jacobian matching the "
            f"{point.size}-coordinate state"
        )
    gradient = np.asarray(stochastic.observable_gradient, dtype=float)
    diagnostics.update(
        {
            key: value
            for key, value in context.diagnostics.items()
            if key
            in {
                "jacobian",
                "diffusion_eigenvalues",
                "critical_mode_left",
                "critical_mode_right",
                "lyapunov_relative_residual",
                "lyapunov_solver",
                "covariance_eigenvalues",
                "closure_contraction",
                "closure_displacement",
                "closure_shape_fraction_status",
                "physicality_status",
            }
        }
    )

    contraction = context.closure_contraction
    record = context.closure_derivative
    displacement = context.closure_displacement
    if context.status is not CAMOutcomeStatus.VALID:
        # A state-level failure is the same verdict for every readout that shares it,
        # and it keeps whatever the shared pass computed.
        partial: dict[str, Any] = {}
        if contraction is not None:
            partial["closure_contraction_norm"] = float(np.linalg.norm(contraction))
        if record is not None:
            partial["closure_derivative"] = record
        return report(
            context.status, context.reason or "the closure state failed", **partial
        )
    if contraction is None or displacement is None or record is None:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the closure state does not carry a contraction and displacement",
        )
    drift_bias = float(gradient @ displacement)
    closure_fields: dict[str, Any] = {
        "closure_contraction_norm": float(np.linalg.norm(contraction)),
        "closure_displacement": tuple(float(value) for value in displacement),
        "closure_displacement_norm": float(np.linalg.norm(displacement)),
        "drift_bias": drift_bias,
        "closure_derivative": record,
    }
    if settings.evaluate_shape:
        closure_fields["closure_shape_fraction"] = context.closure_shape_fraction
    if settings.evaluate_jensen:
        jensen, jensen_step, jensen_repeat, expanded = observable_jensen_bias(
            lambda vector: observable.value(adapter, vector, dict(params)),
            point,
            covariance,
            relative_step=settings.jensen_relative_step,
            eigenvalue_tolerance=settings.jensen_eigenvalue_tolerance,
        )
        closure_fields.update(
            {
                "jensen_bias": float(jensen),
                "jensen_step": float(jensen_step),
                "jensen_repeat_relative_error": jensen_repeat,
                "jensen_eigenvalue_count": expanded,
            }
        )
        diagnostics["jensen_eigenvalue_count"] = expanded
        diagnostics["jensen_repeat_relative_error"] = jensen_repeat
    jensen_bias = closure_fields.get("jensen_bias")
    # The total needs both terms: reporting the drift alone as "the closure bias"
    # would overstate the assessment when the curvature was not evaluated.
    closure_fields["closure_bias_total"] = (
        None if jensen_bias is None else float(drift_bias + jensen_bias)
    )
    if jensen_bias is None:
        diagnostics["closure_bias_total_status"] = (
            "unavailable: the readout curvature (Jensen term) was not evaluated"
        )

    rate = float(context.soft_mode_rate or 0.0)
    if rate <= 0.0:
        return report(
            CAMOutcomeStatus.UNSTABLE,
            f"the soft-mode decay rate is not positive (gamma={rate:.3e})",
            **closure_fields,
        )
    projected_noise = float(context.projected_noise or 0.0)
    if projected_noise < -settings.stochastic.covariance_tolerance:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the projected noise along the critical mode is negative",
            **closure_fields,
        )
    if context.soft_mode_sigma is None:
        return report(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            "the soft-mode noise scale is not positive",
            **closure_fields,
        )
    sigma_x = float(context.soft_mode_sigma)
    projection = float(gradient @ right)
    soft_sigma = abs(projection) * sigma_x
    soft_kubo = soft_sigma / rate
    full_sigma = float(np.sqrt(stochastic.projected_variance or 0.0))
    full_tau = stochastic.correlation_time
    full_kubo = float(full_sigma * full_tau) if full_tau is not None else None
    closure_fields.update(
        {
            "soft_mode_rate": rate,
            "soft_mode_sigma": sigma_x,
            "soft_sigma": soft_sigma,
            "soft_kubo": soft_kubo,
            "full_kubo": full_kubo,
        }
    )
    kubo_values = [value for value in (soft_kubo, full_kubo) if value is not None]
    if not all(np.isfinite(value) for value in kubo_values):
        closure_fields["validity_domain"] = "unresolved"
    elif all(value < settings.kubo_threshold for value in kubo_values):
        closure_fields["validity_domain"] = "weak_noise"
    else:
        closure_fields["validity_domain"] = "noise_dominated"

    closure_fields.update(
        {
            "jacobian_condition_number": context.jacobian_condition_number,
            "covariance_condition_number": context.covariance_condition_number,
        }
    )
    physical_minimum = context.physical_minimum_eigenvalue
    physical_is_psd = context.physical_is_psd
    closure_fields.update(
        {
            "physical_minimum_eigenvalue": physical_minimum,
            "physical_is_psd": physical_is_psd,
        }
    )
    if physical_is_psd is False:
        return report(
            CAMOutcomeStatus.NONPHYSICAL,
            f"the physical spectrum is not positive semidefinite (minimum "
            f"eigenvalue {physical_minimum:.3e})",
            **closure_fields,
        )
    return report(CAMOutcomeStatus.VALID, None, **closure_fields)


#: Boolean fields of a closure annotation, encoded as -1/0/1.
_CLOSURE_TRISTATE_FIELDS: tuple[str, ...] = ("physical_is_psd",)

#: Numeric fields of a closure annotation.
_CLOSURE_NUMERIC_FIELDS: tuple[str, ...] = (
    "closure_contraction_norm",
    "closure_displacement_norm",
    "drift_bias",
    "jensen_bias",
    "closure_bias_total",
    "closure_shape_fraction",
    "soft_mode_rate",
    "soft_mode_sigma",
    "soft_sigma",
    "soft_kubo",
    "full_kubo",
    "jacobian_condition_number",
    "covariance_condition_number",
    "jensen_step",
    "jensen_repeat_relative_error",
    "physical_minimum_eigenvalue",
)

#: Integer fields of a closure annotation.
_CLOSURE_INTEGER_FIELDS: tuple[str, ...] = (
    "approximation_order",
    "jensen_eigenvalue_count",
)


def build_closure_assessment_annotation(
    assessments: Sequence[ClosureAssessment],
    *,
    target_identity: Sequence[str],
    case_identity: Sequence[str] | None = None,
    annotation_kind: str = "closure_risk",
    target_kind: str = "state",
) -> Dataset:
    """Turn closure assessments into a typed annotation product.

    One row per assessment in order. The status column is the shared
    ``CAMOutcomeStatus`` vocabulary, unavailable numbers are ``NaN``, undecided
    booleans are ``-1`` and every row carries the method metadata of its
    assessment. Nothing here ranks or filters: the annotation is a diagnostic.
    """
    cases = list(assessments)
    identities = [str(value) for value in target_identity]
    if not cases:
        # A product with no filled target still declares the same field set, so a
        # consumer sees a stable schema instead of a missing column.
        empty = {name: np.asarray([], dtype=float) for name in _CLOSURE_NUMERIC_FIELDS}
        empty.update(
            {name: np.asarray([], dtype=np.int8) for name in _CLOSURE_TRISTATE_FIELDS}
        )
        empty.update(
            {name: np.asarray([], dtype=np.int64) for name in _CLOSURE_INTEGER_FIELDS}
        )
        for name in (
            "reason",
            "validity_domain",
            "stochastic_status",
            "closure_derivative_method",
        ):
            empty[name] = np.asarray([], dtype="U1")
        return build_annotation_product(
            annotation_kind=annotation_kind,
            target_kind=target_kind,
            target_identity=[],
            case_identity=None if case_identity is None else [],
            status=[],
            fields=empty,
            method_metadata=[],
        )
    if len(identities) != len(cases):
        raise ValueError(
            f"{len(identities)} target identities for {len(cases)} assessments"
        )
    if case_identity is not None and len(list(case_identity)) != len(cases):
        raise ValueError(
            f"{len(list(case_identity))} case identities for {len(cases)} assessments"
        )
    fields: dict[str, Any] = {
        "reason": np.asarray([item.reason or "" for item in cases], dtype=str),
        "validity_domain": np.asarray(
            [item.validity_domain or "" for item in cases], dtype=str
        ),
        "stochastic_status": np.asarray(
            [item.stochastic.status.value for item in cases], dtype=str
        ),
        "closure_derivative_method": np.asarray(
            [
                ""
                if item.closure_derivative is None
                else item.closure_derivative.method
                for item in cases
            ],
            dtype=str,
        ),
    }
    for name in _CLOSURE_NUMERIC_FIELDS:
        fields[name] = numeric_column([getattr(item, name) for item in cases])
    for name in _CLOSURE_TRISTATE_FIELDS:
        fields[name] = tristate_column([getattr(item, name) for item in cases])
    for name in _CLOSURE_INTEGER_FIELDS:
        fields[name] = np.asarray(
            [
                -1 if getattr(item, name) is None else int(getattr(item, name))
                for item in cases
            ],
            dtype=np.int64,
        )
    displacements = [
        None
        if item.closure_displacement is None
        else np.asarray(item.closure_displacement, dtype=float)
        for item in cases
    ]
    if all(value is not None for value in displacements):
        width = int(max(len(value) for value in displacements if value is not None))
        fields["closure_displacement"] = np.asarray(
            [
                np.pad(value, (0, width - len(value)), constant_values=np.nan)
                for value in displacements
                if value is not None
            ],
            dtype=float,
        )
    return build_annotation_product(
        annotation_kind=annotation_kind,
        target_kind=target_kind,
        target_identity=identities,
        case_identity=None if case_identity is None else list(case_identity),
        status=[item.status.value for item in cases],
        fields=fields,
        method_metadata=[dict(item.method) for item in cases],
    )
