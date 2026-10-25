"""qphase_cam: branch-point diagnostics of one readout
---------------------------------------------------------
A branch point is a solved state on a Puiseux branch of a CAM model. Assessing what a
readout sees there needs the same handful of quantities every time: the critical mode
and its conditioning, the physicality of the state, the canonical noise projected on
that mode, the Gaussian soft/full intensities, the explicit and static parameter
response, and the two second-order coefficients — the branch curvature ``eta2`` along
the critical direction and the readout curvature ``eta_readout``.

The Gaussian intensities and the parameter response are **not** re-derived here. They
come from :func:`qphase_cam.nonlinear.ou_intensity_algebra` and
:func:`qphase_cam.nonlinear.parameter_response`, which are the same implementations
:func:`~qphase_cam.nonlinear.ou_intensities` and
:func:`~qphase_cam.nonlinear.response_prediction` use; those two raise when the soft
mode does not decay, so this service calls the cores directly and *reports* the
condition instead of pretending the mode is stable. The mode comes from
:func:`~qphase_cam.validity.critical_mode` with the historical ``nearest_zero`` rule,
and the noise from the shared canonical sample-matrix covariance.

What is genuinely new here is only the second-order pair: both curvatures use a central
difference along the critical right vector with
``direction_step_relative * max(|x|, 1)``
— *not* the unit-offset slow-manifold bilinear form.

Failures are stated, never smoothed: a non-finite state or critical state, a non-finite
Jacobian, an unresolved mode, a state below ``physical_tolerance``, a non-positive
projected noise and a vanishing readout projection each set ``status`` / ``reason_code``
and leave every field they make undefined as ``None``. Nothing is clipped to zero and
nothing is ordered by a NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core.noise import canonical_sample_matrix_covariance
from .nonlinear import ou_intensity_algebra, parameter_response
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus
from .validity import critical_mode, minimum_eigenvalue

__all__ = [
    "BranchDiagnostics",
    "BranchDiagnosticsConfig",
    "branch_diagnostics",
]

FAILURE_STATE = "state_not_finite"
FAILURE_CRITICAL_STATE = "critical_state_not_finite"
FAILURE_JACOBIAN = "jacobian_not_finite"
FAILURE_MODE = "critical_mode_unresolved"
FAILURE_PHYSICAL_SPECTRUM = "physical_spectrum_unreadable"
FAILURE_PHYSICAL_EIGENVALUE = "physical_eigenvalue_not_finite"
FAILURE_NONPHYSICAL = "nonphysical_state"
FAILURE_RATE = "soft_mode_not_decaying"
FAILURE_NOISE = "projected_noise_not_positive"
FAILURE_NOISE_NOT_FINITE = "noise_covariance_not_finite"
FAILURE_NOISE_NOT_PSD = "noise_covariance_not_psd"
FAILURE_FULL_NOISE = "full_noise_intensity_not_finite"
FAILURE_RESPONSE = "zero_frequency_response_unresolved"
FAILURE_READOUT = "readout_not_evaluable"


class BranchDiagnosticsConfig(BaseModel):
    """Steps and floors of one branch-point diagnostic.

    ``relative_step`` is the step of the **explicit parameter response** only:
    ``relative_step * max(1, |parameter value|)``, the same convention
    :func:`~qphase_cam.nonlinear.response_prediction` uses. The readout *gradient* is
    the observable's own record, with whatever step and repeat check that observable
    carries — this config does not impose one. ``direction_step_relative`` is the step
    of both second derivatives along the critical right vector, relative to
    ``max(‖state‖, 1)``. ``projection_floor`` is the readout projection below which
    ``eta_readout`` has no denominator. ``physical_tolerance`` is how far below zero
    the state's Hermitian spectrum may go and still count as physical.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_step: float = Field(
        default=2.0e-6,
        gt=0.0,
        description="Central-difference step of the explicit response",
    )
    direction_step_relative: float = Field(default=1.0e-5, gt=0.0)
    projection_floor: float = Field(default=1.0e-14, gt=0.0)
    overlap_tolerance: float = Field(default=1.0e-10, gt=0.0)
    physical_tolerance: float = Field(default=1.0e-9, ge=0.0)
    covariance_tolerance: float = Field(
        default=1.0e-9,
        ge=0.0,
        description=(
            "How far below zero the canonical noise covariance may go and "
            "still count as positive semidefinite"
        ),
    )


class BranchDiagnostics(BaseModel):
    """What one readout sees at one solved branch point.

    ``status`` is ``valid`` only when the state is physical, the mode decays, the
    projected noise is positive and the readout resolved. A failure keeps the fields
    computed before it and states the check that stopped the rest, so a partial
    diagnostic is never mistaken for a complete one. Every ``float | None`` field is
    ``None`` when it is not defined rather than a NaN a caller could sort.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CAMOutcomeStatus
    reason: str | None = None
    reason_code: str | None = None

    # -- critical mode -----------------------------------------------------
    soft_eigenvalue_real: float | None = None
    soft_eigenvalue_imag: float | None = None
    soft_rate: float | None = None
    noncritical_spectral_gap: float | None = None
    critical_mode_condition_number: float | None = None
    eigenvector_condition_number: float | None = None

    # -- state physicality and mode-projected noise -----------------------
    minimum_physical_eigenvalue: float | None = None
    noise_covariance_minimum_eigenvalue: float | None = None
    projected_noise_intensity: float | None = None
    local_ou_sigma_x: float | None = None
    soft_coordinate: float | None = None
    rho_x: float | None = None
    eta2: float | None = None

    # -- readout -----------------------------------------------------------
    frequency: float | None = None
    frequency_soft_projection: float | None = None
    frequency_curvature: float | None = None
    eta_readout: float | None = None
    soft_ou_intensity: float | None = None
    full_ou_intensity: float | None = None
    soft_response: float | None = None
    explicit_response: float | None = None
    full_response: float | None = None
    regular_response: float | None = None
    ou_snr_coefficient: float | None = None
    gradient: DerivativeRecord | None = None
    derivative: DerivativeRecord | None = None
    method: dict[str, Any] = Field(default_factory=dict)


def _finite(value: Any) -> float | None:
    """Return ``value`` as a float, or ``None`` when it is not finite."""
    resolved = float(value)
    return resolved if np.isfinite(resolved) else None


def branch_diagnostics(
    adapter: Any,
    model: Any,
    observable: CAMObservable,
    *,
    state: Any,
    params: Mapping[str, Any],
    parameter: str,
    scale: float = 1.0,
    critical_state: Any | None = None,
    config: BranchDiagnosticsConfig | None = None,
) -> BranchDiagnostics:
    """Return the branch-point diagnostics of one readout at one state.

    ``state`` is the solved branch point and ``critical_state`` the branch's critical
    state, used only for the soft coordinate ``l·(state − critical)``; without it that
    coordinate (and ``rho_x``) stays ``None``, and a non-finite critical state is
    reported instead of silently poisoning the coordinate. ``parameter`` names the scan
    parameter the branch was continued along and ``scale`` multiplies an epsilon into
    its offset, exactly as the candidate case records it.
    """
    settings = config or BranchDiagnosticsConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.branch_diagnostics.branch_diagnostics",
        "critical_mode_rule": "nearest_zero",
        "noise_covariance": "canonical diffusion covariance",
        "gaussian_intensities": "qphase_cam.nonlinear.ou_intensity_algebra",
        "parameter_response": "qphase_cam.nonlinear.parameter_response",
        "readout": observable.identity.model_dump(mode="json"),
        "parameter": parameter,
        "scale": float(scale),
        "explicit_response_step": "relative_step * max(1, |parameter value|)",
        "relative_step": float(settings.relative_step),
        "direction_step_relative": float(settings.direction_step_relative),
        "projection_floor": float(settings.projection_floor),
        "physical_tolerance": float(settings.physical_tolerance),
        "covariance_tolerance": float(settings.covariance_tolerance),
        "gradient": "the observable's own derivative record",
        "eta2": "0.5 l·[(J(x+d r) − J(x−d r)) r] / (2d) with d = step·max(‖x‖,1)",
        "readout_curvature": "central second difference along the right vector",
    }

    def failed(
        status: CAMOutcomeStatus, code: str, reason: str, **fields: Any
    ) -> BranchDiagnostics:
        return BranchDiagnostics(
            status=status,
            reason=reason,
            reason_code=code,
            method=dict(method),
            **fields,
        )

    if point.size == 0 or not np.all(np.isfinite(point)):
        return failed(
            CAMOutcomeStatus.NOT_CONVERGED, FAILURE_STATE, "the state is not finite"
        )
    if parameter not in parameter_map:
        raise ValueError(f"the parameter {parameter!r} has no value in params")
    critical: np.ndarray | None = None
    if critical_state is not None:
        critical = np.asarray(critical_state, dtype=float).reshape(-1)
        if critical.size != point.size:
            raise ValueError(
                f"the critical state has {critical.size} coordinates for a "
                f"{point.size}-coordinate state"
            )
    jacobian = np.asarray(adapter.jacobian(point, parameter_map), dtype=float)
    if not np.all(np.isfinite(jacobian)):
        return failed(
            CAMOutcomeStatus.NOT_CONVERGED,
            FAILURE_JACOBIAN,
            "the dynamics Jacobian is not finite",
        )
    mode = critical_mode(
        jacobian, overlap_tolerance=settings.overlap_tolerance, rule="nearest_zero"
    )
    if mode is None:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_MODE,
            "the soft mode is unresolved (complex or overlapping left/right "
            "eigenvectors)",
        )
    eigenvalue = complex(mode.value)
    rate = -float(np.real(eigenvalue))
    right = np.asarray(mode.right, dtype=float)
    left = np.asarray(mode.left, dtype=float)
    mode_fields: dict[str, Any] = {
        "soft_eigenvalue_real": float(np.real(eigenvalue)),
        "soft_eigenvalue_imag": float(np.imag(eigenvalue)),
        "soft_rate": rate,
        "noncritical_spectral_gap": (
            _finite(mode.spectral_gap) if np.isfinite(mode.spectral_gap) else None
        ),
        "critical_mode_condition_number": _finite(mode.condition_number),
        "eigenvector_condition_number": _finite(mode.eigenvector_condition_number),
    }
    try:
        spectrum = np.asarray(adapter.physical_eigenvalues(point, parameter_map))
    except (AttributeError, TypeError, ValueError, np.linalg.LinAlgError) as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_PHYSICAL_SPECTRUM,
            f"the physical spectrum cannot be read at this state: {exc}",
            **mode_fields,
        )
    if spectrum.size == 0 or not np.all(np.isfinite(spectrum)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_PHYSICAL_EIGENVALUE,
            "the physical spectrum is not finite",
            **mode_fields,
        )
    minimum_physical = float(np.min(spectrum))
    mode_fields["minimum_physical_eigenvalue"] = minimum_physical
    # A non-physical state still has an algebra: the verdict is recorded and the
    # readout quantities are computed, because they are what a caller inspects when
    # deciding whether a slightly negative eigenvalue matters.
    physical_verdict: tuple[CAMOutcomeStatus, str, str] | None = None
    if minimum_physical < -settings.physical_tolerance:
        physical_verdict = (
            CAMOutcomeStatus.NONPHYSICAL,
            FAILURE_NONPHYSICAL,
            "the state is not physical (minimum eigenvalue "
            f"{minimum_physical:.3e} < -{settings.physical_tolerance:.1e})",
        )
    if critical is not None and not np.all(np.isfinite(critical)):
        return failed(
            CAMOutcomeStatus.NOT_CONVERGED,
            FAILURE_CRITICAL_STATE,
            "the critical state is not finite",
            **mode_fields,
        )
    matrix = np.asarray(adapter.state_matrix(point, parameter_map), dtype=complex)
    diffusion = np.asarray(model.cam_diffusion(matrix, parameter_map), dtype=complex)
    covariance = canonical_sample_matrix_covariance(matrix, diffusion)
    if not np.all(np.isfinite(covariance)):
        return failed(
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_NOISE_NOT_FINITE,
            "the canonical noise covariance is not finite",
            **mode_fields,
        )
    minimum_covariance, _ = minimum_eigenvalue(covariance)
    mode_fields["noise_covariance_minimum_eigenvalue"] = float(minimum_covariance)
    covariance_verdict: tuple[CAMOutcomeStatus, str, str] | None = None
    if minimum_covariance < -settings.covariance_tolerance:
        covariance_verdict = (
            CAMOutcomeStatus.STOCHASTIC_INVALID,
            FAILURE_NOISE_NOT_PSD,
            "the canonical noise covariance is not positive semidefinite "
            f"(minimum eigenvalue {minimum_covariance:.3e})",
        )

    try:
        algebra = ou_intensity_algebra(
            adapter,
            observable,
            state=point,
            params=parameter_map,
            diffusion=covariance,
            mode=mode,
            jacobian=jacobian,
            overlap_tolerance=settings.overlap_tolerance,
        )
    except ValueError as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_RESPONSE,
            str(exc),
            **mode_fields,
        )
    projected_noise = algebra.projected_noise
    noise_is_usable = bool(np.isfinite(projected_noise) and projected_noise > 0.0)
    # A non-positive projected noise does not erase the readout: the frequency and the
    # projection are defined, and only the noise scale and what divides by it are not.
    sigma = algebra.sigma if noise_is_usable else None
    coordinate: float | None = None
    if critical is not None:
        coordinate = float(left @ (point - critical))
    noise_fields: dict[str, Any] = {
        "projected_noise_intensity": projected_noise,
        "local_ou_sigma_x": sigma,
        "soft_coordinate": coordinate,
        "rho_x": (
            abs(coordinate) / sigma if (coordinate is not None and sigma) else None
        ),
    }
    if physical_verdict is not None:
        status, reason_code, reason = physical_verdict
    elif covariance_verdict is not None:
        status, reason_code, reason = covariance_verdict
    elif not np.isfinite(algebra.full_intensity) or algebra.full_intensity < 0.0:
        status = CAMOutcomeStatus.STOCHASTIC_INVALID
        reason_code = FAILURE_FULL_NOISE
        reason = (
            "the full-mode noise intensity is not finite and non-negative "
            f"({algebra.full_intensity!r})"
        )
    elif rate <= 0.0:
        status = CAMOutcomeStatus.UNSTABLE
        reason_code = FAILURE_RATE
        reason = f"the soft mode does not decay (rate {rate:.3e})"
    elif not noise_is_usable:
        status = CAMOutcomeStatus.STOCHASTIC_INVALID
        reason_code = FAILURE_NOISE
        reason = (
            "the projected noise along the critical mode is not finite and positive "
            f"({projected_noise!r})"
        )
    else:
        status = CAMOutcomeStatus.VALID
        reason_code = None
        reason = None

    # Second derivative of the drift along the critical direction: a central
    # difference of the Jacobian, *not* the unit-offset slow-manifold bilinear form.
    direction_step = settings.direction_step_relative * max(
        1.0, float(np.linalg.norm(point))
    )
    plus_jacobian = np.asarray(
        adapter.jacobian(point + direction_step * right, parameter_map), dtype=float
    )
    minus_jacobian = np.asarray(
        adapter.jacobian(point - direction_step * right, parameter_map), dtype=float
    )
    along_second = (plus_jacobian - minus_jacobian) @ right / (2.0 * direction_step)
    quadratic_drift = 0.5 * float(left @ along_second)
    noise_fields["eta2"] = (
        abs(quadratic_drift) * sigma / rate if sigma is not None else None
    )

    try:
        response = parameter_response(
            adapter,
            observable,
            state=point,
            params=parameter_map,
            parameter=parameter,
            scale=scale,
            relative_step=settings.relative_step,
            jacobian=jacobian,
            gradient=algebra.gradient,
        )
    except ValueError as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_RESPONSE,
            str(exc),
            **mode_fields,
            **noise_fields,
        )
    projected_drive = float(left @ np.asarray(response.drive, dtype=float))
    projection = algebra.projection
    soft_response = (
        projection * projected_drive / rate if rate > 0.0 else None
    )
    frequency = float(observable.value(adapter, point, parameter_map))
    plus_frequency = float(
        observable.value(adapter, point + direction_step * right, parameter_map)
    )
    minus_frequency = float(
        observable.value(adapter, point - direction_step * right, parameter_map)
    )
    if not all(
        np.isfinite(value) for value in (frequency, plus_frequency, minus_frequency)
    ):
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_READOUT,
            "the readout is not finite on the state or on its right-vector offsets",
            **mode_fields,
            **noise_fields,
            gradient=algebra.gradient,
        )
    if not np.all(np.isfinite(algebra.gradient.gradient)):
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_READOUT,
            "the readout gradient is not finite",
            **mode_fields,
            **noise_fields,
            gradient=algebra.gradient,
        )
    curvature = (plus_frequency - 2.0 * frequency + minus_frequency) / (
        direction_step**2
    )
    if not np.isfinite(curvature):
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_READOUT,
            f"the readout curvature is not finite ({curvature!r})",
            **mode_fields,
            **noise_fields,
            gradient=algebra.gradient,
        )
    eta_readout: float | None = None
    if (
        sigma is not None
        and np.isfinite(curvature)
        and abs(projection) > settings.projection_floor
    ):
        eta_readout = 0.5 * abs(curvature) * sigma / abs(projection)
    full_response = response.full_response
    readout_fields: dict[str, Any] = {
        "frequency": frequency,
        "frequency_soft_projection": projection,
        "frequency_curvature": curvature,
        "eta_readout": eta_readout,
        "soft_ou_intensity": (
            algebra.soft_intensity if noise_is_usable else None
        ),
        "full_ou_intensity": algebra.full_intensity,
        "soft_response": soft_response,
        "explicit_response": response.explicit_response,
        "full_response": full_response,
        "regular_response": (
            full_response - soft_response
            if (full_response is not None and soft_response is not None)
            else None
        ),
        "ou_snr_coefficient": (
            full_response**2 / algebra.full_intensity
            if (
                full_response is not None
                and np.isfinite(algebra.full_intensity)
                and algebra.full_intensity > 0.0
            )
            else None
        ),
        "gradient": algebra.gradient,
        "derivative": response.derivative,
    }
    return BranchDiagnostics(
        status=status,
        reason=reason,
        reason_code=reason_code,
        method=dict(method),
        **mode_fields,
        **noise_fields,
        **readout_fields,
    )
