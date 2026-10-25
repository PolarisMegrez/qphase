"""High-precision CAM Newton solve as an ordinary numerical service.

The local-response validation, the long31 and Q1 paper chains and the branch-IC
generators all need the same thing: continue one CAM equilibrium with mpmath at a
requested working precision, from either a plain state vector or the branch's
verified decimal expansion. That implementation used to live only inside
``LocalResponseValidation``, so a retained script had to reach into a
postprocessor's private method to solve a branch.

This module is that one implementation, exposed as plain functions: no entry point,
no engine task, no scheduler, no result protocol and no persistence. Nothing about
the numerics changed — the working precision is raised and restored, the equation is
the model's residual, the step is a damped Newton step with bisection backtracking,
and a solve reports ``converged`` only when the residual reached the tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "HighPrecisionNewtonConfig",
    "HighPrecisionSolution",
    "solve_high_precision",
]


class HighPrecisionNewtonConfig(BaseModel):
    """Working precision, residual target and step budget of one solve.

    The defaults are the local-response validation's, so a caller migrating from it
    keeps the numbers it had.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    precision_digits: int = Field(
        default=80, ge=30, le=500, description="mpmath working decimal digits"
    )
    residual_tolerance: float = Field(
        default=1e-30, gt=0.0, description="Residual norm a solve must reach"
    )
    max_iterations: int = Field(
        default=40, ge=1, le=500, description="Newton step budget"
    )


@dataclass(frozen=True)
class HighPrecisionSolution:
    """One high-precision continuation result.

    ``residual`` is the last residual the iteration evaluated, in the same units as
    the model's residual vector, and ``converged`` says whether it reached the
    configured tolerance. ``iterations`` counts the steps actually taken; a solve
    that stopped on a rejected damping step reports the steps it completed rather
    than a claim that it converged.
    """

    state: np.ndarray
    residual: float
    converged: bool
    iterations: int


def _predicted_initial(
    critical_decimals: tuple[str, ...],
    leading_coefficient: Any,
    epsilon_magnitude: float | None,
    exponent: float | None,
) -> Any:
    """Return the mpmath initial state of the documented prediction."""
    import mpmath as mp

    if epsilon_magnitude is None or exponent is None:
        raise ValueError(
            "the predicted initial state needs epsilon_magnitude and exponent "
            "together with critical_decimals"
        )
    coefficient = np.asarray(leading_coefficient, dtype=float).reshape(-1)
    if coefficient.size != len(critical_decimals):
        raise ValueError(
            f"leading_coefficient has {coefficient.size} entries for "
            f"{len(critical_decimals)} critical decimals"
        )
    magnitude = mp.mpf(str(epsilon_magnitude)) ** mp.mpf(str(exponent))
    return mp.matrix(
        [
            mp.mpf(value) + mp.mpf(str(item)) * magnitude
            for value, item in zip(critical_decimals, coefficient, strict=True)
        ]
    )


def solve_high_precision(
    adapter: Any,
    initial: Any,
    params: dict[str, Any],
    *,
    config: HighPrecisionNewtonConfig | None = None,
    critical_decimals: tuple[str, ...] | None = None,
    leading_coefficient: Any | None = None,
    epsilon_magnitude: float | None = None,
    exponent: float | None = None,
) -> HighPrecisionSolution:
    """Continue one CAM equilibrium with mpmath Newton steps.

    ``critical_decimals`` selects the branch's verified decimal expansion as the
    starting point: the initial state is ``critical_decimal + coefficient *
    epsilon**exponent`` per coordinate. Without it the starting point is ``initial``.
    The three prediction arguments are optional but must appear together — a caller
    that passes one of them and expects the others to be guessed gets an error.
    """
    import mpmath as mp

    resolved = config or HighPrecisionNewtonConfig()
    provided = (
        critical_decimals is not None,
        leading_coefficient is not None,
        epsilon_magnitude is not None,
        exponent is not None,
    )
    if any(provided) and not all(provided):
        raise ValueError(
            "the predicted initial state needs critical_decimals, "
            "leading_coefficient, epsilon_magnitude and exponent together; got "
            f"critical_decimals={critical_decimals is not None}, "
            f"leading_coefficient={leading_coefficient is not None}, "
            f"epsilon_magnitude={epsilon_magnitude is not None}, "
            f"exponent={exponent is not None}"
        )

    previous_digits = mp.mp.dps
    mp.mp.dps = resolved.precision_digits
    try:
        if critical_decimals is None:
            vector = np.asarray(initial, dtype=float).reshape(-1)
            state = mp.matrix([mp.mpf(str(item)) for item in vector])
        else:
            state = _predicted_initial(
                critical_decimals,
                leading_coefficient,
                epsilon_magnitude,
                exponent,
            )
        mp_params = {
            name: mp.mpf(str(value)) if np.asarray(value).ndim == 0 else value
            for name, value in params.items()
        }
        tolerance = mp.mpf(str(resolved.residual_tolerance))
        residual = mp.inf
        iterations = 0
        for _ in range(resolved.max_iterations):
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
                trial_residual = mp.sqrt(sum(item * item for item in trial_values))
                if trial_residual < residual:
                    state = trial
                    residual = trial_residual
                    accepted = True
                    break
                damping /= 2
            if not accepted:
                break
            iterations += 1
        output = np.asarray([float(item) for item in state], dtype=float)
        return HighPrecisionSolution(
            state=output,
            residual=float(residual),
            converged=bool(residual <= tolerance),
            iterations=iterations,
        )
    finally:
        mp.mp.dps = previous_digits
