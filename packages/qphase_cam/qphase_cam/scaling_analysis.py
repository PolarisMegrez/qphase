"""qphase_cam: readout-resolved scaling analysis
---------------------------------------------------------
The numerical layer behind a readout-scaling annotation. It answers "how does one
observable scale with one control coordinate, and over which window does a simple
power law describe it?" — as ordinary functions, not as a plugin.

The reference implementation is
``scripts/tools/analyze_kerr_full_3mode_cam_r_scaling.py``: a log-space power fit
(its ``_free_power_fit``), a tolerance-bounded prefix (``_prefix_upper``) and a
fixed-exponent single-amplitude window (``_fixed_exponent_fit_upper``). This module
reproduces those three definitions with the caller's exponent and tolerance as
arguments instead of the script's hard-coded one third, and adds the two pieces a
generic analysis needs:

* :func:`local_log_exponents` — the per-point effective exponent
  ``d log y / d log x`` with second-order edge stencils;
* :func:`window_log_scaling_fits` — the log-space fit of every requested window;
* :func:`predict_puiseux` and :func:`puiseux_outer_extrapolation` — the generic
  extension of the 10D3 Puiseux layer: fit inside a window and predict outside it,
  so "does the inner window describe the outer points?" is answered for any readout
  (the Allan-specific window diagnostic is not reused for a general law).

Every function validates its input strictly: one-dimensional, equal-length,
finite, strictly positive and strictly increasing ``x``; strictly positive ``y``
(a log fit of a non-positive response is undefined, and returning a number for it
would be a fabrication). A window that holds fewer points than a fit needs raises
instead of returning a zero exponent.

This module is *analysis*: it fits the series it is given. The nonlinear Puiseux
fit itself belongs to :mod:`qphase_cam.solver.scaling` and is reused here, never
re-implemented.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.core.utils import canonical_json
from scipy.optimize import minimize_scalar

from .solver.scaling import ScalingFit, free_exponent_fit

__all__ = [
    "ALLAN_PUISEUX_FAMILY_NOTE",
    "LogLinearFit",
    "OuterExtrapolation",
    "allan_free_exponent_puiseux_fit",
    "evaluate_allan_free_exponent_puiseux",
    "fixed_exponent_tolerance_upper",
    "local_log_exponents",
    "local_slope",
    "log_linear_scaling_fit",
    "predict_puiseux",
    "puiseux_outer_extrapolation",
    "series_identity",
    "tolerance_prefix_upper",
    "window_log_scaling_fits",
]


class LogLinearFit(BaseModel):
    """One log-space power fit: ``log y = intercept + exponent * log x``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    exponent: float
    log_rmse: float
    intercept: float
    point_count: int
    upper: float | None = None


class OuterExtrapolation(BaseModel):
    """How well a Puiseux fit of an inner window predicts the outer points."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    inner_max: float
    inner_point_count: int
    outer_point_count: int
    exponent: float
    bounds: tuple[float, float]
    offset: float
    coefficients: tuple[float, ...]
    relative_rmse: float
    max_abs_relative_error: float
    relative_residuals: tuple[float, ...]
    at_exponent_bound: bool


def _as_series(x: Any, y: Any, *, minimum_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Return validated ``(x, y)`` arrays; raise on anything a fit cannot use."""
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    if xs.size != ys.size:
        raise ValueError(f"x has {xs.size} points for {ys.size} responses")
    if xs.size < minimum_points:
        raise ValueError(
            f"a scaling series needs at least {minimum_points} points; got {xs.size}"
        )
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("x and y must be finite everywhere")
    if np.any(xs <= 0.0):
        raise ValueError("x must be strictly positive")
    if np.any(ys <= 0.0):
        raise ValueError(
            "y must be strictly positive: a log-space scaling fit of a "
            "non-positive response is undefined"
        )
    if np.any(np.diff(xs) <= 0.0):
        raise ValueError(
            "x must be strictly increasing; sort the series instead of letting a "
            "window mean two different things"
        )
    return xs, ys


def _validated_tolerance(tolerance: Any, name: str = "tolerance") -> float:
    value = float(tolerance)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative; got {tolerance}")
    return value


def log_linear_scaling_fit(
    x: Any, y: Any, *, upper: float | None = None
) -> LogLinearFit:
    """Fit ``log y`` on ``log x`` over ``x <= upper`` and return the exponent.

    This is the script's ``_free_power_fit`` without its hard-wired window: the
    result is the log-space slope and the RMS of the log residuals. It is *not* a
    Puiseux fit — no offset, no higher orders; that is
    :func:`qphase_cam.solver.scaling.free_exponent_fit`.
    """
    xs, ys = _as_series(x, y, minimum_points=2)
    resolved_upper = None if upper is None else float(upper)
    if resolved_upper is not None:
        if not np.isfinite(resolved_upper) or resolved_upper <= 0.0:
            raise ValueError(f"upper must be finite and positive; got {resolved_upper}")
        selected = xs <= resolved_upper
    else:
        selected = np.ones(xs.size, dtype=bool)
    if int(np.count_nonzero(selected)) < 2:
        raise ValueError(
            f"the window upper={resolved_upper} keeps "
            f"{int(np.count_nonzero(selected))} points; a log fit needs two"
        )
    log_x = np.log(xs[selected])
    log_y = np.log(ys[selected])
    design = np.column_stack((np.ones(log_x.size), log_x))
    coefficients, *_ = np.linalg.lstsq(design, log_y, rcond=None)
    residual = design @ coefficients - log_y
    return LogLinearFit(
        exponent=float(coefficients[1]),
        log_rmse=float(np.sqrt(np.mean(residual**2))),
        intercept=float(coefficients[0]),
        point_count=int(np.count_nonzero(selected)),
        upper=resolved_upper,
    )


def local_log_exponents(x: Any, y: Any) -> np.ndarray:
    """Return the per-point effective exponent ``d log y / d log x``.

    The stencil is second order at the edges (``edge_order=2``), which needs at
    least three points; the result keeps every point instead of collapsing the
    series into one number, so a crossover is visible rather than averaged away.
    """
    xs, ys = _as_series(x, y, minimum_points=3)
    return np.gradient(np.log(ys), np.log(xs), edge_order=2)


def tolerance_prefix_upper(x: Any, accepted: Any) -> float | None:
    """Return the last ``x`` before the first rejected point.

    ``None`` when the first point is already rejected: nothing is accepted, so
    there is no window (an empty window is not the same as a window at the origin).
    """
    xs, _ = _as_series(x, x, minimum_points=1)
    flags = np.asarray(accepted, dtype=bool).reshape(-1)
    if flags.size != xs.size:
        raise ValueError(f"accepted has {flags.size} entries for {xs.size} coordinates")
    rejected = np.flatnonzero(~flags)
    stop = int(rejected[0]) if rejected.size else xs.size
    return float(xs[stop - 1]) if stop > 0 else None


def fixed_exponent_tolerance_upper(
    x: Any,
    y: Any,
    exponent: float,
    tolerance: float,
    *,
    minimum_points: int = 5,
) -> float | None:
    """Return the furthest prefix a fixed-exponent single amplitude describes.

    For every prefix of at least ``minimum_points`` points the amplitude is fitted
    by least squares against ``x**exponent`` and the relative error
    ``max|prediction / y - 1|`` is compared with ``tolerance``; the last prefix that
    passes wins. The exponent is the caller's: no default target exponent is baked
    in.
    """
    xs, ys = _as_series(x, y, minimum_points=2)
    resolved_exponent = float(exponent)
    if not np.isfinite(resolved_exponent) or resolved_exponent <= 0.0:
        raise ValueError(f"exponent must be finite and positive; got {exponent}")
    resolved_tolerance = _validated_tolerance(tolerance)
    if minimum_points < 2:
        raise ValueError(f"minimum_points must be at least 2; got {minimum_points}")
    if xs.size < minimum_points:
        raise ValueError(
            f"the series has {xs.size} points but minimum_points={minimum_points}"
        )
    basis = xs**resolved_exponent
    upper: float | None = None
    for stop in range(minimum_points, xs.size + 1):
        local_basis = basis[:stop]
        local_values = ys[:stop]
        amplitude = float(local_basis @ local_values / (local_basis @ local_basis))
        relative = np.abs(amplitude * local_basis / local_values - 1.0)
        if float(np.max(relative)) <= resolved_tolerance:
            upper = float(xs[stop - 1])
    return upper


def window_log_scaling_fits(
    x: Any, y: Any, uppers: Sequence[float]
) -> list[LogLinearFit]:
    """Return one log-space fit per requested window."""
    xs, ys = _as_series(x, y, minimum_points=2)
    windows = [float(value) for value in uppers]
    for value in windows:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"a window upper must be finite and positive; got {value}")
    return [log_linear_scaling_fit(xs, ys, upper=value) for value in windows]


def predict_puiseux(fit: ScalingFit, x: Any) -> np.ndarray:
    """Evaluate one Puiseux fit at arbitrary coordinates.

    ``fit.coefficients`` live in the fit's own normalized coordinate
    (``x / fit.epsilon_scale``), so the same scale is applied here — a prediction
    that quietly used a different normalization would be a different law.
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    if xs.size and (not np.all(np.isfinite(xs)) or np.any(xs <= 0.0)):
        raise ValueError("x must be finite and strictly positive")
    coordinate = xs / float(fit.epsilon_scale)
    columns = [
        coordinate ** (power * fit.exponent) for power in range(1, fit.order + 1)
    ]
    design = np.column_stack(columns) if columns else np.empty((xs.size, 0))
    return float(fit.offset) + design @ np.asarray(fit.coefficients, dtype=float)


def puiseux_outer_extrapolation(
    x: Any,
    y: Any,
    *,
    inner_max: float,
    order: int,
    bounds: tuple[float, float],
    weighted: bool = False,
    profile_points: int = 401,
) -> OuterExtrapolation:
    """Fit the inner window and score its prediction of the outer points.

    This is the generic (non-Allan) version of the window extrapolation check: the
    law fitted inside ``x <= inner_max`` is evaluated outside it, and the relative
    residuals are reported. A window that leaves no inner or no outer points
    raises instead of scoring an empty set.
    """
    xs, ys = _as_series(x, y, minimum_points=2)
    resolved_inner = float(inner_max)
    if not np.isfinite(resolved_inner) or resolved_inner <= 0.0:
        raise ValueError(f"inner_max must be finite and positive; got {inner_max}")
    inner = xs <= resolved_inner
    outer = ~inner
    if int(np.count_nonzero(outer)) < 1:
        raise ValueError(
            f"inner_max={resolved_inner} leaves no outer point to extrapolate to"
        )
    fit = free_exponent_fit(
        xs[inner],
        ys[inner],
        order=order,
        bounds=bounds,
        weighted=weighted,
        profile_points=profile_points,
        calculate_loo=False,
    )
    prediction = predict_puiseux(fit, xs[outer])
    relative = (ys[outer] - prediction) / ys[outer]
    return OuterExtrapolation(
        inner_max=resolved_inner,
        inner_point_count=int(np.count_nonzero(inner)),
        outer_point_count=int(np.count_nonzero(outer)),
        exponent=float(fit.exponent),
        bounds=tuple(bounds),
        offset=float(fit.offset),
        coefficients=tuple(float(value) for value in fit.coefficients),
        relative_rmse=float(np.sqrt(np.mean(relative**2))),
        max_abs_relative_error=float(np.max(np.abs(relative))),
        relative_residuals=tuple(float(value) for value in relative),
        at_exponent_bound=bool(fit.at_exponent_bound),
    )


def series_identity(
    observable: Any,
    *,
    parameter: str,
    x: Any,
    y: Any,
) -> str:
    """Return the content identity of one scaling series.

    The identity covers the observable identity, the control name and every
    ``(x, y)`` pair — never a path, a job name or a slot position, so the same
    series recomputed elsewhere carries the same identity and a different series
    cannot inherit it.
    """
    xs = np.asarray(x, dtype=float).reshape(-1)
    ys = np.asarray(y, dtype=float).reshape(-1)
    if xs.size != ys.size:
        raise ValueError(f"x has {xs.size} points for {ys.size} responses")
    # A point the fit could not use is not part of the series content, and NaN is
    # not JSON: the identity covers the usable pairs only.
    usable = np.isfinite(xs) & np.isfinite(ys)
    digest_payload = {
        "observable": observable.identity.model_dump(mode="json"),
        "parameter": str(parameter),
        "point_count": int(np.count_nonzero(usable)),
        "points": [
            [float(a), float(b)]
            for a, b in zip(xs[usable].tolist(), ys[usable].tolist(), strict=True)
        ],
    }
    text = canonical_json(digest_payload)
    return "cam:scaling:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


#: The Allan Puiseux exponent here is a *different fit family* from the
#: ``-2(1-p) + k p`` basis of :mod:`qphase_cam.solver.scaling`: this one uses
#: ``(k - 4) p`` and therefore absorbs the transfer-function prefactor into the
#: exponent instead of carrying it as a separate leading power. The two bases are
#: not interchangeable and must never be substituted for one another; the paper
#: summary's Allan fields are defined by the ``(k - 4) p`` family below.
ALLAN_PUISEUX_FAMILY_NOTE = (
    "Allan free-exponent Puiseux with the (k - 4) p basis; a different fit family "
    "from the -2(1-p) + k p basis of solver.scaling, not interchangeable"
)


def allan_free_exponent_puiseux_fit(
    epsilon: Any,
    values: Any,
    *,
    order: int,
    exponent_bounds: tuple[float, float],
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Fit the ``(k - 4) p`` Allan Puiseux family on relative residuals.

    The design column ``k`` is ``epsilon ** ((k - 4) * p)`` for ``k = 0..order``,
    which is deliberately **not** the ``-2(1-p) + k p`` basis of the solver-layer
    Allan fit (see :data:`ALLAN_PUISEUX_FAMILY_NOTE`). The exponent minimises the
    relative residual, and its uncertainty comes from the profile curvature at
    step ``1e-4`` with ``n - (order + 2)`` degrees of freedom.
    """
    x, y = _as_series(epsilon, values, minimum_points=max(3, order + 3))
    resolved_order = int(order)
    if resolved_order < 0:
        raise ValueError(f"order must be non-negative; got {order}")
    low, high = (float(exponent_bounds[0]), float(exponent_bounds[1]))
    if not (np.isfinite(low) and np.isfinite(high)) or low <= 0.0 or low >= high:
        raise ValueError(
            f"exponent_bounds must be finite, positive and increasing; got "
            f"{exponent_bounds}"
        )

    def projected_fit(exponent: float) -> tuple[float, np.ndarray, np.ndarray]:
        design = np.column_stack(
            [x ** ((power - 4) * exponent) for power in range(resolved_order + 1)]
        )
        coefficients, *_ = np.linalg.lstsq(
            design / y[:, None], np.ones(y.size), rcond=None
        )
        prediction = design @ coefficients
        objective = float(np.sum(((y - prediction) / y) ** 2))
        return objective, coefficients, prediction

    optimum = minimize_scalar(
        lambda exponent: projected_fit(float(exponent))[0],
        bounds=(low, high),
        method="bounded",
        options={"xatol": 1e-13},
    )
    exponent = float(optimum.x)
    objective, coefficients, prediction = projected_fit(exponent)
    step = 1e-4
    curvature = (
        projected_fit(exponent + step)[0]
        - 2.0 * objective
        + projected_fit(exponent - step)[0]
    ) / step**2
    if not np.isfinite(curvature) or curvature <= 0.0:
        raise ValueError("the Allan Puiseux profile curvature is not positive")
    degrees_of_freedom = y.size - (resolved_order + 2)
    if degrees_of_freedom < 1:
        raise ValueError(
            f"order {resolved_order} leaves no residual degrees of freedom"
        )
    residual_scale = float(np.sqrt(objective / degrees_of_freedom))
    uncertainty = float(np.sqrt(2.0 / curvature) * residual_scale)
    return exponent, uncertainty, coefficients, prediction


def evaluate_allan_free_exponent_puiseux(
    epsilon: Any, coefficients: Any, exponent: float
) -> np.ndarray:
    """Evaluate the ``(k - 4) p`` Allan Puiseux sum, matching its fit."""
    x = np.asarray(epsilon, dtype=float).reshape(-1)
    return sum(
        coefficient * x ** ((power - 4) * float(exponent))
        for power, coefficient in enumerate(np.asarray(coefficients, dtype=float))
    )


def local_slope(
    epsilon: Any,
    values: Any,
    sem: Any,
    accepted: Any,
    half_window: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted local-quadratic slope and its uncertainty.

    For every accepted point the slope comes from a second-order Taylor fit over
    the accepted neighbours inside ``half_window`` (clamped at the sequence ends),
    weighted by ``1/sem``; a window with fewer than three points stays ``NaN``
    instead of being fitted with a lower order.
    """
    x = np.asarray(epsilon, dtype=float).reshape(-1)
    y = np.asarray(values, dtype=float).reshape(-1)
    sigma = np.asarray(sem, dtype=float).reshape(-1)
    mask = np.asarray(accepted, dtype=bool).reshape(-1)
    if not (x.size == y.size == sigma.size == mask.size):
        raise ValueError("epsilon, values, sem and accepted must have equal length")
    if np.any(sigma <= 0.0) or not np.all(np.isfinite(sigma)):
        raise ValueError("sem must be finite and strictly positive")
    if int(half_window) < 1:
        raise ValueError(f"half_window must be at least 1; got {half_window}")
    index = np.flatnonzero(mask)
    slope = np.full(x.size, np.nan)
    slope_sem = np.full(x.size, np.nan)
    for position, center in enumerate(index):
        width = min(2 * int(half_window) + 1, index.size)
        offset = min(max(0, position - int(half_window)), index.size - width)
        window = index[offset : offset + width]
        if window.size < 3:
            continue
        local_x = x[window] - x[center]
        local_y = y[window]
        weight = 1.0 / sigma[window]
        coefficients, covariance = np.polyfit(
            local_x, local_y, min(2, window.size - 1), w=weight, cov="unscaled"
        )
        slope[center] = coefficients[-2]
        slope_sem[center] = float(np.sqrt(covariance[-2, -2]))
    return slope, slope_sem
