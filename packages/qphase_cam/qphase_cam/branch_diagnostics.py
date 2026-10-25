"""qphase_cam: branch-point diagnostics of one readout
---------------------------------------------------------
A branch point is a solved state on a Puiseux branch of a CAM model. Assessing what a
readout sees there needs the same handful of quantities every time: the critical mode
and its conditioning, the canonical noise projected on that mode, the Gaussian
soft/full intensities, the explicit and static parameter response, and the two
second-order coefficients (the branch curvature ``eta2`` along the critical
direction and the readout curvature ``eta_readout``).

This module is the one service that computes them, so a script or a postprocessor
cannot drift from "the" definition:

* the critical mode is :func:`~qphase_cam.validity.critical_mode` with its historical
  ``nearest_zero`` rule — no mode index, no trace shortcut, no model name;
* the noise is the shared canonical sample-matrix covariance
  (:mod:`qphase_cam.core.noise`);
* the Gaussian terms are :func:`~qphase_cam.nonlinear.ou_intensities`' definitions,
  re-derived on the state's own Jacobian and covariance rather than re-implemented
  (the intensities are per observable, so they are evaluated from the same
  ``projection``/``response`` the service already has);
* the readout gradient is the observable's own record (its step and repeat check
  travel with it), and the parameter response is a central difference in the named
  parameter with ``relative_step`` times its magnitude;
* both curvatures use a central difference along the critical right vector with
  ``direction_step_relative * max(‖state‖, 1)`` — *not* the slow-manifold bilinear
  form, which uses unit offsets.

Unavailability is stated, not faked: a non-positive soft rate, a non-positive
projected noise, a vanishing readout projection (``|projection| <= projection_floor``)
or a singular response solve leave the dependent fields ``None`` and put the reason in
``status``/``reason``. Nothing is clipped to zero and nothing is sorted by a NaN.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from .core.noise import canonical_sample_matrix_covariance
from .observable import CAMObservable, DerivativeRecord
from .outcomes import CAMOutcomeStatus
from .validity import critical_mode

__all__ = [
    "BranchDiagnostics",
    "BranchDiagnosticsConfig",
    "branch_diagnostics",
]

FAILURE_STATE = "state_not_finite"
FAILURE_JACOBIAN = "jacobian_not_finite"
FAILURE_MODE = "critical_mode_unresolved"
FAILURE_RATE = "soft_mode_not_decaying"
FAILURE_NOISE = "projected_noise_not_positive"
FAILURE_RESPONSE = "parameter_response_unresolved"
FAILURE_READOUT = "readout_not_evaluable"


class BranchDiagnosticsConfig(BaseModel):
    """Steps and floors of one branch-point diagnostic.

    ``relative_step`` is the central-difference step of the observable gradient and of
    the explicit parameter response, relative to the parameter's own magnitude (the
    paper chain's ``2e-6``). ``direction_step_relative`` is the step of both second
    derivatives along the critical right vector, relative to the state norm.
    ``projection_floor`` is the readout projection below which ``eta_readout`` has no
    denominator.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_step: float = Field(default=2.0e-6, gt=0.0)
    direction_step_relative: float = Field(default=1.0e-5, gt=0.0)
    projection_floor: float = Field(default=1.0e-14, gt=0.0)
    overlap_tolerance: float = Field(default=1.0e-10, gt=0.0)


class BranchDiagnostics(BaseModel):
    """What one readout sees at one solved branch point.

    ``status`` is ``valid`` when the mode, the noise and the readout all resolved. A
    failure keeps the fields computed before it and states the check that stopped the
    rest, so a partial diagnostic is never mistaken for a complete one. Every
    ``float | None`` field is ``None`` when it is not defined rather than a NaN that a
    caller could sort or compare.
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

    # -- mode-projected noise ---------------------------------------------
    projected_noise_intensity: float | None = None
    local_ou_sigma_x: float | None = None
    soft_coordinate: float | None = None
    rho_x: float | None = None
    eta2: float | None = None
    minimum_physical_eigenvalue: float | None = None

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
    coordinate (and ``rho_x``) stays ``None``. ``parameter`` names the scan parameter
    the branch was continued along and ``scale`` multiplies an epsilon into its offset,
    exactly as the candidate case records it.
    """
    settings = config or BranchDiagnosticsConfig()
    point = np.asarray(state, dtype=float).reshape(-1)
    parameter_map = dict(params or {})
    method: dict[str, Any] = {
        "service": "qphase_cam.branch_diagnostics.branch_diagnostics",
        "critical_mode_rule": "nearest_zero",
        "noise_covariance": "canonical diffusion covariance",
        "readout": observable.identity.model_dump(mode="json"),
        "parameter": parameter,
        "scale": float(scale),
        "relative_step": float(settings.relative_step),
        "direction_step_relative": float(settings.direction_step_relative),
        "projection_floor": float(settings.projection_floor),
        "eta2": "0.5 l·[(J(x+d r) − J(x−d r)) r] / (2d) with d = step·max(‖x‖,1)",
        "readout_curvature": "central second difference along the right vector",
        "soft_ou": "projection^2 * projected_noise / rate^2",
        "full_ou": "v^T Sigma v with v = g^T (-J)^{-1}",
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
    matrix = np.asarray(adapter.state_matrix(point, parameter_map), dtype=complex)
    diffusion = np.asarray(model.cam_diffusion(matrix, parameter_map), dtype=complex)
    covariance = canonical_sample_matrix_covariance(matrix, diffusion)
    projected_noise = float(np.real(left @ covariance @ left))
    physical = _finite(np.min(adapter.physical_eigenvalues(point, parameter_map)))
    mode_fields: dict[str, Any] = {
        "soft_eigenvalue_real": float(np.real(eigenvalue)),
        "soft_eigenvalue_imag": float(np.imag(eigenvalue)),
        "soft_rate": rate,
        "noncritical_spectral_gap": (
            _finite(mode.spectral_gap)
            if np.isfinite(mode.spectral_gap)
            else None
        ),
        "critical_mode_condition_number": _finite(mode.condition_number),
        "eigenvector_condition_number": _finite(mode.eigenvector_condition_number),
        "minimum_physical_eigenvalue": physical,
    }
    # A non-positive rate or a non-positive projected noise makes the *noise scale*
    # undefined, not the readout: the frequency, its projection and the second
    # derivatives are still computable and are kept. Only the quantities that divide
    # by the rate or by sigma stay None.
    sigma: float | None = None
    if rate > 0.0 and np.isfinite(projected_noise) and projected_noise > 0.0:
        sigma = float(np.sqrt(projected_noise / (2.0 * rate)))
    coordinate: float | None = None
    if critical_state is not None:
        critical = np.asarray(critical_state, dtype=float).reshape(-1)
        coordinate = float(left @ (point - critical))
    noise_fields: dict[str, Any] = {
        "projected_noise_intensity": _finite(projected_noise),
        "local_ou_sigma_x": sigma,
        "soft_coordinate": coordinate,
        "rho_x": (
            abs(coordinate) / sigma
            if (coordinate is not None and sigma)
            else None
        ),
    }

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

    if rate <= 0.0:
        status = CAMOutcomeStatus.UNSTABLE
        reason_code = FAILURE_RATE
        reason = f"the soft mode does not decay (rate {rate:.3e})"
    elif noise_fields["projected_noise_intensity"] is None:
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

    try:
        direction = np.asarray(
            adapter.parameter_direction(parameter, float(scale)), dtype=float
        )
        drive = (
            np.asarray(adapter.parameter_jacobian(point, parameter_map), dtype=float)
            @ direction
        )
        state_response = np.linalg.solve(-jacobian, drive)
    except (np.linalg.LinAlgError, ValueError) as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_RESPONSE,
            f"the parameter response does not exist: {exc}",
            **mode_fields,
            **noise_fields,
        )
    if not np.all(np.isfinite(state_response)):
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_RESPONSE,
            "the parameter response is not finite",
            **mode_fields,
            **noise_fields,
        )

    try:
        frequency = float(observable.value(adapter, point, parameter_map))
        record = observable.gradient(adapter, point, parameter_map)
    except (ValueError, TypeError, FloatingPointError) as exc:
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_READOUT,
            f"the readout cannot be evaluated at this state: {exc}",
            **mode_fields,
            **noise_fields,
        )
    gradient = np.asarray(record.gradient, dtype=float)
    if not np.all(np.isfinite(gradient)) or not np.isfinite(frequency):
        return failed(
            CAMOutcomeStatus.UNIDENTIFIABLE,
            FAILURE_READOUT,
            "the readout or its gradient is not finite",
            **mode_fields,
            **noise_fields,
            derivative=record,
        )
    projection = float(gradient @ right)
    # The Gaussian intensities divide by the rate; without a positive rate they are
    # undefined while the projection and the raw response stay measurable.
    soft_response = (
        projection * float(left @ drive) / rate if rate > 0.0 else None
    )
    soft_ou = (
        projection**2 * projected_noise / rate**2 if rate > 0.0 else None
    )
    try:
        zero_frequency = np.linalg.solve(-jacobian.T, gradient)
        full_ou = float(zero_frequency @ covariance @ zero_frequency)
    except np.linalg.LinAlgError:
        full_ou = None
    parameter_value = float(parameter_map[parameter])
    parameter_step = settings.relative_step * max(1.0, abs(parameter_value))
    plus_params = dict(parameter_map)
    minus_params = dict(parameter_map)
    plus_params[parameter] = parameter_value + parameter_step
    minus_params[parameter] = parameter_value - parameter_step
    explicit = (
        float(observable.value(adapter, point, plus_params))
        - float(observable.value(adapter, point, minus_params))
    ) / (2.0 * parameter_step)
    full_response = explicit + float(gradient @ state_response)
    plus_frequency = float(
        observable.value(adapter, point + direction_step * right, parameter_map)
    )
    minus_frequency = float(
        observable.value(adapter, point - direction_step * right, parameter_map)
    )
    curvature = (plus_frequency - 2.0 * frequency + minus_frequency) / (
        direction_step**2
    )
    eta_readout: float | None = None
    if (
        sigma is not None
        and np.isfinite(curvature)
        and abs(projection) > settings.projection_floor
    ):
        eta_readout = 0.5 * abs(curvature) * sigma / abs(projection)
    readout_fields: dict[str, Any] = {
        "frequency": frequency,
        "frequency_soft_projection": projection,
        "frequency_curvature": curvature,
        "eta_readout": eta_readout,
        "soft_ou_intensity": soft_ou,
        "full_ou_intensity": full_ou,
        "soft_response": soft_response,
        "explicit_response": explicit,
        "full_response": full_response,
        "regular_response": (
            full_response - soft_response if soft_response is not None else None
        ),
        "ou_snr_coefficient": (
            full_response**2 / full_ou
            if full_ou is not None and full_ou > 0.0
            else None
        ),
        "derivative": record,
    }
    method["eta_readout"] = (
        "0.5 |d^2 omega| sigma / |projection| when |projection| > projection_floor"
    )
    return BranchDiagnostics(
        status=status,
        reason=reason,
        reason_code=reason_code,
        method=dict(method),
        **mode_fields,
        **noise_fields,
        **readout_fields,
    )
