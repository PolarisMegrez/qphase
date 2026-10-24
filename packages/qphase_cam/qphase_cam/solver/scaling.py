"""qphase_cam: general scaling (Puiseux) fitting
---------------------------------------------------------
The numeric layer the scaling classifier exposes to callers. It generalises the
five one-off evaluation scripts (``scripts/tools/evaluate_*puiseux*`` and
``evaluate_*two_stage*``) into one strict, model-agnostic API:

* :func:`fixed_exponent_fit` — ``x**(k*p)`` for ``k=1..order`` with an optional
  offset and optional weights. A fixed-``p`` G1 fit *is* this fit, so there is no
  model-specific variant.
* :func:`free_exponent_fit` — the same Puiseux basis with the base exponent
  fitted by a profile search plus bounded refinement, either anchored on a known
  critical value (offset held) or with a free offset.
* :func:`allan_relative_fit` — the Allan-variance variant
  ``x**[-2(1-p) + k*p]`` for ``k=0..order`` scored on relative residuals, with a
  window diagnostic that reports how the exponent moves with the fit window.
* :func:`two_stage_frequency_fit` — a fixed one-third offset stage followed by a
  free exponent and amplitude on the full range.

Nothing here knows about a model, a target exponent or a default window: bounds
and windows are the caller's, and every input is validated instead of being
coerced. A failed fit raises; a degenerate one is reported through ``at_bound``,
the condition number and the leave-one-out figures rather than hidden.

Results are frozen data objects, not products: turning a fit into an annotation
belongs to the postprocessor layer.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from scipy.optimize import minimize_scalar

__all__ = [
    "PRIMARY_EXPONENT_BOUNDS",
    "WIDE_EXPONENT_BOUNDS",
    "ScalingFit",
    "ScalingWindowDiagnostics",
    "TwoStageFrequencyFit",
    "allan_relative_fit",
    "allan_window_diagnostics",
    "fixed_exponent_fit",
    "free_exponent_fit",
    "two_stage_frequency_fit",
]

#: Conventional narrow exponent window used by the historical CAM/G1 scripts.
PRIMARY_EXPONENT_BOUNDS: tuple[float, float] = (0.25, 0.45)

#: Conventional wide window used to check for an exponent alias.
WIDE_EXPONENT_BOUNDS: tuple[float, float] = (0.10, 0.70)

#: Fit families this module produces.
FitModel = Literal["fixed_exponent", "free_exponent", "allan_relative"]

#: Relative tolerance for "the optimum sits on the bound".
BOUND_TOLERANCE = 1.0e-5

#: Profile resolution of the leave-one-out refits. Slightly coarser than the main
#: Allan profile (801 points) on purpose: nine extra global searches at the main
#: resolution buy no extra information about a single omitted point.
LOO_PROFILE_POINTS = 401


@dataclass(frozen=True)
class ScalingFit:
    """One Puiseux fit, with its diagnostics and its method metadata."""

    model: FitModel
    order: int
    point_count: int
    parameter_count: int
    epsilon_scale: float
    exponent: float
    exponent_sem: float | None
    exponent_bounds: tuple[float, float] | None
    at_exponent_bound: bool
    offset: float
    coefficients: tuple[float, ...]
    prediction: np.ndarray
    residual: np.ndarray
    rmse: float
    max_abs_residual: float
    loo_rmse: float | None
    relative_rmse: float | None
    loo_relative_rmse: float | None
    aicc: float
    bic: float | None
    weighted_reduced_chi2: float | None
    condition: float
    objective: float
    weighted: bool
    anchor: float | None = None
    profile_points: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def leading_coefficient(self) -> float:
        """Coefficient of the leading Puiseux term."""
        return float(self.coefficients[0])

    @property
    def degrees_of_freedom(self) -> int:
        """Residual degrees of freedom."""
        return self.point_count - self.parameter_count


@dataclass(frozen=True)
class ScalingWindowDiagnostics:
    """How the fitted exponent moves with the fitting window."""

    window_maxima: tuple[float, ...]
    window_exponents: tuple[float, ...]
    span: float
    inner_max: float
    outer_extrapolation_relative_rmse: float
    outer_extrapolation_max_abs_relative_error: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TwoStageFrequencyFit:
    """A fixed-offset stage followed by a free exponent on the full range."""

    lower: float
    upper: float
    stage1_exponent: float
    stage1_point_count: int
    stage1_offset: float
    stage1_amplitude: float
    stage1_rmse: float
    stage2_exponent: float
    stage2_amplitude: float
    stage2_full_range_rmse: float
    stage2_max_abs_residual: float
    stage2_prediction: np.ndarray
    stage2_residual: np.ndarray
    exponent_bounds: tuple[float, float]
    epsilon_scale: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


# --- validation ------------------------------------------------------------


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional; got shape {array.shape}")
    if array.size < 2:
        raise ValueError(f"{name} needs at least two points; got {array.size}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite everywhere")
    return array


def _validated_coordinate(epsilon: Any, *, minimum_points: int) -> np.ndarray:
    x = _as_vector(epsilon, "x")
    if np.any(x <= 0.0):
        raise ValueError("x must be strictly positive")
    if x.size < minimum_points:
        raise ValueError(
            f"a fit of this order needs at least {minimum_points} points; got {x.size}"
        )
    return x


def _validated_response(
    value: Any, name: str, *, count: int, positive: bool = False
) -> np.ndarray:
    y = _as_vector(value, name)
    if y.size != count:
        raise ValueError(f"{name} has {y.size} points for {count} coordinates")
    if positive and np.any(y <= 0.0):
        raise ValueError(f"{name} must be strictly positive")
    return y


def _validated_uncertainty(value: Any, *, count: int) -> np.ndarray:
    if value is None:
        return np.ones(count)
    sigma = _as_vector(value, "uncertainty")
    if sigma.size != count:
        raise ValueError(f"uncertainty has {sigma.size} points for {count} coordinates")
    if np.any(sigma <= 0.0):
        raise ValueError("uncertainty must be strictly positive")
    return sigma


def _validated_order(order: Any, *, point_count: int, parameters: int) -> int:
    if isinstance(order, bool) or not isinstance(order, (int, np.integer)):
        raise ValueError(f"order must be an integer; got {order!r}")
    resolved = int(order)
    if resolved < 1:
        raise ValueError(f"order must be at least 1; got {resolved}")
    parameter_count = resolved + parameters
    if point_count - parameter_count < 1:
        raise ValueError(
            f"order {resolved} needs more than {parameter_count} points for a "
            f"non-degenerate fit; got {point_count}"
        )
    return resolved


def _validated_bounds(bounds: Any) -> tuple[float, float]:
    try:
        values = tuple(float(item) for item in bounds)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bounds must be a (low, high) pair; got {bounds!r}") from exc
    if len(values) != 2:
        raise ValueError(f"bounds must be a (low, high) pair; got {bounds!r}")
    low, high = values
    if not (np.isfinite(low) and np.isfinite(high)):
        raise ValueError(f"bounds must be finite; got {values}")
    if low <= 0.0:
        raise ValueError(f"bounds must be strictly positive; got {values}")
    if low >= high:
        raise ValueError(f"bounds must be increasing; got {values}")
    return (low, high)


def _validated_scale(scale: float | None, x: np.ndarray) -> float:
    if scale is None:
        return float(np.max(x))
    resolved = float(scale)
    if not np.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"scale must be finite and positive; got {resolved}")
    return resolved


# --- designs and profile search --------------------------------------------


def _puiseux_design(
    coordinate: np.ndarray, order: int, exponent: float, *, offset_column: bool
) -> np.ndarray:
    columns = [coordinate ** (power * exponent) for power in range(1, order + 1)]
    if offset_column:
        return np.column_stack((np.ones(coordinate.size), *columns))
    return np.column_stack(columns)


def _allan_design(coordinate: np.ndarray, order: int, exponent: float) -> np.ndarray:
    powers = -2.0 * (1.0 - exponent) + exponent * np.arange(order + 1)
    return coordinate[:, None] ** powers[None, :]


def _allan_projection(
    coordinate: np.ndarray, values: np.ndarray, order: int, exponent: float
) -> tuple[float, np.ndarray]:
    """Relative-residual objective and coefficients for one Allan exponent."""
    design = _allan_design(coordinate, order, exponent)
    coefficients, *_ = np.linalg.lstsq(
        design / values[:, None], np.ones(values.size), rcond=None
    )
    relative = (values - design @ coefficients) / values
    return float(relative @ relative), coefficients


def _allan_exponent(
    coordinate: np.ndarray,
    values: np.ndarray,
    *,
    order: int,
    bounds: tuple[float, float],
    profile_points: int,
) -> float:
    """Best relative-residual exponent on the given points."""

    def objective(exponent: float) -> float:
        return _allan_projection(coordinate, values, order, exponent)[0]

    return float(
        min(_profile_minima(objective, bounds, profile_points=profile_points))[1]
    )


def _allan_loo_residual(
    coordinate: np.ndarray,
    values: np.ndarray,
    *,
    order: int,
    bounds: tuple[float, float],
    profile_points: int,
    omitted: int,
) -> float:
    """Relative prediction error at one omitted point, refitting the rest."""
    retained = np.arange(coordinate.size) != omitted
    local_coordinate = coordinate[retained]
    local_values = values[retained]
    exponent = _allan_exponent(
        local_coordinate,
        local_values,
        order=order,
        bounds=bounds,
        profile_points=profile_points,
    )
    _, coefficients = _allan_projection(local_coordinate, local_values, order, exponent)
    prediction = float(
        (_allan_design(coordinate[[omitted]], order, exponent) @ coefficients)[0]
    )
    return float((values[omitted] - prediction) / values[omitted])


def _profile_minima(
    objective: Any, bounds: tuple[float, float], *, profile_points: int
) -> list[tuple[float, float]]:
    """Return every local minimum of a bounded scalar objective.

    A coarse profile first, then a bounded refinement of every local minimum (and
    of both bounds). Refining only the global profile minimum would let a
    high-order relabelling of the exponent win by accident, so the caller gets
    all of them and picks deterministically.
    """
    if profile_points < 3:
        raise ValueError(f"profile_points must be at least 3; got {profile_points}")
    grid = np.linspace(bounds[0], bounds[1], profile_points)
    values = np.asarray([objective(float(exponent)) for exponent in grid])
    candidates = {0, grid.size - 1, int(np.argmin(values))}
    candidates.update(
        int(index)
        for index in np.flatnonzero(
            (values[1:-1] <= values[:-2]) & (values[1:-1] <= values[2:])
        )
        + 1
    )
    minima: list[tuple[float, float]] = []
    for index in sorted(candidates):
        if index in (0, grid.size - 1):
            minima.append((float(values[index]), float(grid[index])))
            continue
        optimum = minimize_scalar(
            objective,
            bounds=(float(grid[index - 1]), float(grid[index + 1])),
            method="bounded",
            options={"xatol": 1.0e-13},
        )
        minima.append((float(optimum.fun), float(optimum.x)))
    return minima


def _aicc(point_count: int, parameter_count: int, criterion: float) -> float:
    aic = (
        point_count * np.log(max(criterion / point_count, np.finfo(float).tiny))
        + 2.0 * parameter_count
    )
    return float(
        aic
        + 2.0
        * parameter_count
        * (parameter_count + 1)
        / (point_count - parameter_count - 1)
    )


def _bic(point_count: int, parameter_count: int, criterion: float) -> float:
    return float(
        point_count * np.log(max(criterion / point_count, np.finfo(float).tiny))
        + parameter_count * np.log(point_count)
    )


def _at_bound(exponent: float, bounds: tuple[float, float]) -> bool:
    return bool(min(exponent - bounds[0], bounds[1] - exponent) < BOUND_TOLERANCE)


# --- absolute-residual Puiseux --------------------------------------------


def _puiseux_fit_impl(
    *,
    model: FitModel,
    x: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    order: int,
    exponent: float | None,
    bounds: tuple[float, float] | None,
    profile_points: int,
    weighted: bool,
    anchor: float | None,
    calculate_loo: bool,
) -> ScalingFit:
    """Fit the Puiseux basis at a given exponent or with the base exponent free."""
    scale = float(np.max(x))
    coordinate = x / scale
    offset_column = anchor is None
    resolved_anchor = 0.0 if anchor is None else float(anchor)
    parameter_count = order + (2 if offset_column else 1)
    weights = 1.0 / sigma if weighted else np.ones(coordinate.size)

    def projected(
        value: float,
    ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        design = _puiseux_design(coordinate, order, value, offset_column=offset_column)
        # An anchored fit is solved in the shifted space (response minus anchor):
        # re-adding the anchor before taking the residual would lose the small
        # difference to cancellation.
        target = y if offset_column else y - resolved_anchor
        coefficients, *_ = np.linalg.lstsq(
            design * weights[:, None], target * weights, rcond=None
        )
        shifted = design @ coefficients
        objective = float(np.sum(((target - shifted) * weights) ** 2))
        return objective, coefficients, shifted, design

    if exponent is not None:
        fixed = float(exponent)
        if not np.isfinite(fixed) or fixed <= 0.0:
            raise ValueError(f"exponent must be finite and positive; got {exponent}")
        if bounds is not None and not (bounds[0] <= fixed <= bounds[1]):
            raise ValueError(f"fixed exponent {fixed} lies outside the bounds {bounds}")
        resolved = fixed
        resolved_bounds = bounds
        objective, coefficients, shifted, design = projected(resolved)
    else:
        if bounds is None:
            raise ValueError(
                "a free-exponent fit requires finite positive exponent bounds"
            )
        minima = _profile_minima(
            lambda value: projected(float(value))[0],
            bounds,
            profile_points=profile_points,
        )
        objective, resolved = min(minima)
        objective, coefficients, shifted, design = projected(resolved)
        resolved_bounds = bounds
    prediction = shifted if offset_column else shifted + resolved_anchor
    residual = y - prediction

    if not np.isfinite(objective):
        raise ValueError("the fit produced a non-finite objective")
    degrees_of_freedom = coordinate.size - parameter_count
    if degrees_of_freedom < 1:
        raise ValueError(
            f"the fit leaves no residual degrees of freedom ({degrees_of_freedom})"
        )
    puiseux = coefficients if not offset_column else coefficients[1:]
    offset = float(coefficients[0]) if offset_column else resolved_anchor

    exponent_derivative = np.sum(
        np.asarray(
            [
                puiseux[power - 1]
                * power
                * np.log(coordinate)
                * coordinate ** (power * resolved)
                for power in range(1, order + 1)
            ]
        ),
        axis=0,
    )
    jacobian = np.column_stack((design, exponent_derivative))
    weighted_jacobian = jacobian * weights[:, None]
    covariance = np.linalg.pinv(weighted_jacobian.T @ weighted_jacobian)
    covariance *= objective / degrees_of_freedom
    exponent_sem = (
        float(np.sqrt(max(float(covariance[-1, -1]), 0.0)))
        if resolved_bounds is not None
        else None
    )

    loo_residual = np.full(coordinate.size, np.nan)
    if calculate_loo:
        for omitted in range(coordinate.size):
            retained = np.arange(coordinate.size) != omitted
            local = _subset_exponent(
                coordinate[retained],
                y[retained],
                sigma[retained],
                order=order,
                exponent=exponent,
                bounds=bounds,
                profile_points=profile_points,
                weighted=weighted,
                anchor=anchor,
            )
            prediction_omitted = _puiseux_prediction(
                coordinate[[omitted]],
                order,
                local["exponent"],
                local["coefficients"],
                offset_column=offset_column,
                anchor=resolved_anchor,
            )
            loo_residual[omitted] = float(y[omitted] - prediction_omitted[0])

    rmse = float(np.sqrt(np.mean(residual**2)))
    loo_rmse = float(np.sqrt(np.mean(loo_residual**2))) if calculate_loo else None
    metadata = {
        "basis": "puiseux",
        "design_powers": list(range(1, order + 1)),
        "offset_column": offset_column,
        "residual_kind": "absolute",
        "sem_method": (
            "linearised_gauss_newton" if resolved_bounds is not None else "fixed"
        ),
        "loo_method": "refit" if calculate_loo else "disabled",
    }
    return ScalingFit(
        model=model,
        order=order,
        point_count=int(coordinate.size),
        parameter_count=parameter_count,
        epsilon_scale=scale,
        exponent=resolved,
        exponent_sem=exponent_sem,
        exponent_bounds=resolved_bounds,
        at_exponent_bound=(
            _at_bound(resolved, resolved_bounds) if resolved_bounds else False
        ),
        offset=offset,
        coefficients=tuple(float(value) for value in puiseux),
        prediction=np.asarray(prediction, dtype=float),
        residual=np.asarray(residual, dtype=float),
        rmse=rmse,
        max_abs_residual=float(np.max(np.abs(residual))),
        loo_rmse=loo_rmse,
        relative_rmse=None,
        loo_relative_rmse=None,
        aicc=_aicc(coordinate.size, parameter_count, objective),
        bic=None,
        weighted_reduced_chi2=float(
            np.sum((residual / sigma) ** 2) / degrees_of_freedom
        ),
        condition=float(np.linalg.cond(weighted_jacobian)),
        objective=float(objective),
        weighted=weighted,
        anchor=None if offset_column else resolved_anchor,
        profile_points=None if exponent is not None else profile_points,
        metadata=metadata,
    )


def _subset_exponent(
    coordinate: np.ndarray,
    y: np.ndarray,
    sigma: np.ndarray,
    *,
    order: int,
    exponent: float | None,
    bounds: tuple[float, float] | None,
    profile_points: int,
    weighted: bool,
    anchor: float | None,
) -> dict[str, Any]:
    """Refit one leave-one-out subset; return its exponent and coefficients."""
    offset_column = anchor is None
    resolved_anchor = 0.0 if anchor is None else float(anchor)
    weights = 1.0 / sigma if weighted else np.ones(coordinate.size)

    def projected(value: float) -> tuple[float, np.ndarray]:
        design = _puiseux_design(coordinate, order, value, offset_column=offset_column)
        target = y if offset_column else y - resolved_anchor
        coefficients, *_ = np.linalg.lstsq(
            design * weights[:, None], target * weights, rcond=None
        )
        shifted = design @ coefficients
        return float(np.sum(((target - shifted) * weights) ** 2)), coefficients

    if exponent is not None:
        resolved = float(exponent)
    else:
        assert bounds is not None
        _, resolved = min(
            _profile_minima(
                lambda value: projected(float(value))[0],
                bounds,
                profile_points=profile_points,
            )
        )
    _, coefficients = projected(resolved)
    return {"exponent": resolved, "coefficients": coefficients}


def _puiseux_prediction(
    coordinate: np.ndarray,
    order: int,
    exponent: float,
    coefficients: Any,
    *,
    offset_column: bool,
    anchor: float,
) -> np.ndarray:
    """Predict from a fitted coefficient vector of the fitted parametrisation."""
    design = _puiseux_design(coordinate, order, exponent, offset_column=offset_column)
    shifted = design @ np.asarray(coefficients, dtype=float)
    return shifted if offset_column else shifted + anchor


def fixed_exponent_fit(
    epsilon: Any,
    response: Any,
    *,
    exponent: float,
    order: int,
    uncertainty: Any | None = None,
    weighted: bool = False,
    offset: bool = True,
    anchor: float | None = None,
    calculate_loo: bool = True,
    scale: float | None = None,
) -> ScalingFit:
    """Fit ``response = offset + sum_k c_k x**(k*exponent)`` at a fixed exponent.

    This is the fixed-``p`` Puiseux fit (a G1 one-third fit is the case
    ``exponent=1/3``): ``offset=True`` fits the constant term, ``offset=False``
    requires ``anchor`` and holds it. Leave-one-out uses the analytic leverage
    formula of the linear design. A fixed exponent is not a fitted parameter, so a
    free-offset fit spends ``order + 1`` parameters and an anchored one ``order``.
    """
    if offset and anchor is not None:
        raise ValueError("pass either offset=True or anchor=..., not both")
    if not offset and anchor is None:
        raise ValueError("offset=False requires the anchor value to hold")
    x = _validated_coordinate(epsilon, minimum_points=2)
    y = _validated_response(response, "response", count=x.size)
    sigma = _validated_uncertainty(uncertainty, count=x.size)
    variable_count = 1 if offset else 0
    resolved_order = _validated_order(
        order, point_count=x.size, parameters=variable_count
    )
    fixed = float(exponent)
    if not np.isfinite(fixed) or fixed <= 0.0:
        raise ValueError(f"exponent must be finite and positive; got {exponent}")
    resolved_scale = _validated_scale(scale, x)
    coordinate = x / resolved_scale
    weights = 1.0 / sigma if weighted else np.ones(x.size)
    design = _puiseux_design(coordinate, resolved_order, fixed, offset_column=offset)
    weighted_design = design * weights[:, None]
    resolved_anchor = 0.0 if anchor is None else float(anchor)
    target = y if offset else y - resolved_anchor
    coefficients, *_ = np.linalg.lstsq(weighted_design, target * weights, rcond=None)
    prediction = design @ coefficients
    if not offset:
        prediction = prediction + resolved_anchor
    residual = y - prediction
    parameter_count = resolved_order + variable_count
    degrees_of_freedom = x.size - parameter_count
    if degrees_of_freedom < 1:
        raise ValueError("the fit leaves no residual degrees of freedom")
    gram_inverse = np.linalg.pinv(weighted_design.T @ weighted_design)
    leverage = np.einsum("ij,jk,ik->i", weighted_design, gram_inverse, weighted_design)
    # A zero leverage would make the analytic LOO residual meaningless (it would
    # mean the point carries no information at all), so the guard is explicit.
    loo_residual = residual / np.maximum(1.0 - leverage, 1.0e-12)
    rss = float(residual @ residual)
    weighted_rss = float(np.sum((residual / sigma) ** 2))
    criterion = weighted_rss if weighted else rss
    puiseux = coefficients if not offset else coefficients[1:]
    offset_value = float(coefficients[0]) if offset else resolved_anchor
    return ScalingFit(
        model="fixed_exponent",
        order=resolved_order,
        point_count=int(x.size),
        parameter_count=parameter_count,
        epsilon_scale=resolved_scale,
        exponent=fixed,
        exponent_sem=None,
        exponent_bounds=None,
        at_exponent_bound=False,
        offset=offset_value,
        coefficients=tuple(float(value) for value in puiseux),
        prediction=np.asarray(prediction, dtype=float),
        residual=np.asarray(residual, dtype=float),
        rmse=float(np.sqrt(np.mean(residual**2))),
        max_abs_residual=float(np.max(np.abs(residual))),
        loo_rmse=(float(np.sqrt(np.mean(loo_residual**2))) if calculate_loo else None),
        relative_rmse=None,
        loo_relative_rmse=None,
        aicc=_aicc(x.size, parameter_count, criterion),
        bic=None,
        weighted_reduced_chi2=weighted_rss / degrees_of_freedom,
        condition=float(np.linalg.cond(weighted_design)),
        objective=float(criterion),
        weighted=weighted,
        anchor=None if offset else resolved_anchor,
        profile_points=None,
        metadata={
            "basis": "puiseux",
            "design_powers": list(range(1, resolved_order + 1)),
            "offset_column": offset,
            "residual_kind": "absolute",
            "loo_method": "leverage" if calculate_loo else "disabled",
        },
    )


def free_exponent_fit(
    epsilon: Any,
    response: Any,
    *,
    order: int,
    bounds: tuple[float, float],
    uncertainty: Any | None = None,
    weighted: bool = False,
    anchor: float | None = None,
    profile_points: int = 401,
    calculate_loo: bool = True,
) -> ScalingFit:
    """Fit ``response = offset + sum_k c_k x**(k*p)`` with ``p`` free.

    With ``anchor`` the offset is held at that value (a known critical
    frequency); without it the constant term is fitted. Every local minimum of the
    profiled objective is refined and the best one wins, so a high-order relabelling
    of the exponent cannot be picked by accident. The coordinate scale is always the
    data maximum, so the fitted exponent is scale-invariant.
    """
    x = _validated_coordinate(epsilon, minimum_points=2)
    y = _validated_response(response, "response", count=x.size)
    sigma = _validated_uncertainty(uncertainty, count=x.size)
    resolved_bounds = _validated_bounds(bounds)
    resolved_order = _validated_order(
        order, point_count=x.size, parameters=2 if anchor is None else 1
    )
    return _puiseux_fit_impl(
        model="free_exponent",
        x=x,
        y=y,
        sigma=sigma,
        order=resolved_order,
        exponent=None,
        bounds=resolved_bounds,
        profile_points=profile_points,
        weighted=weighted,
        anchor=anchor,
        calculate_loo=calculate_loo,
    )


def allan_relative_fit(
    epsilon: Any,
    intensity: Any,
    *,
    order: int,
    bounds: tuple[float, float],
    profile_points: int = 801,
    calculate_loo: bool = True,
) -> ScalingFit:
    """Fit the Allan intensity ``x**[-2(1-p) + k*p]``, ``k=0..order``.

    The objective is the relative residual (the intensity spans decades), the
    exponent uncertainty comes from the curvature of the profiled objective, and
    the reported condition is scaled by the column norms. No target exponent and
    no default window are assumed.
    """
    x = _validated_coordinate(epsilon, minimum_points=2)
    values = _validated_response(intensity, "intensity", count=x.size, positive=True)
    resolved_bounds = _validated_bounds(bounds)
    resolved_order = _validated_order(order, point_count=x.size, parameters=2)
    scale = float(np.max(x))
    coordinate = x / scale
    parameter_count = resolved_order + 2
    degrees_of_freedom = x.size - parameter_count
    if degrees_of_freedom < 1:
        raise ValueError("the fit leaves no residual degrees of freedom")
    loo_profile_points = min(profile_points, LOO_PROFILE_POINTS)
    if calculate_loo and x.size < parameter_count + 2:
        raise ValueError(
            "a leave-one-out check needs at least one more point than the fit"
        )

    minima = _profile_minima(
        lambda value: _allan_projection(
            coordinate, values, resolved_order, float(value)
        )[0],
        resolved_bounds,
        profile_points=profile_points,
    )
    objective, exponent = min(minima)
    objective, coefficients = _allan_projection(
        coordinate, values, resolved_order, exponent
    )
    if not np.isfinite(objective):
        raise ValueError("the fit produced a non-finite objective")
    prediction = _allan_design(coordinate, resolved_order, exponent) @ coefficients
    relative_residual = (values - prediction) / values

    step = 1.0e-4
    lower = _allan_projection(coordinate, values, resolved_order, exponent - step)[0]
    upper = _allan_projection(coordinate, values, resolved_order, exponent + step)[0]
    curvature = (upper - 2.0 * objective + lower) / step**2
    exponent_sem = (
        float(np.sqrt(2.0 * (objective / degrees_of_freedom) / curvature))
        if np.isfinite(curvature) and curvature > 0.0
        else None
    )

    basis = _allan_design(coordinate, resolved_order, exponent)
    powers = 2.0 + np.arange(resolved_order + 1)
    exponent_derivative = np.sum(
        coefficients[None, :] * powers[None, :] * np.log(coordinate)[:, None] * basis,
        axis=1,
    )
    relative_jacobian = np.column_stack(
        (basis / values[:, None], exponent_derivative / values)
    )
    norms = np.linalg.norm(relative_jacobian, axis=0)
    condition = (
        float(np.linalg.cond(relative_jacobian / norms[None, :]))
        if not np.any(norms == 0.0)
        else float("inf")
    )

    loo_relative = np.full(x.size, np.nan)
    if calculate_loo:
        for omitted in range(x.size):
            loo_relative[omitted] = _allan_loo_residual(
                coordinate,
                values,
                order=resolved_order,
                bounds=resolved_bounds,
                profile_points=loo_profile_points,
                omitted=omitted,
            )

    metadata = {
        "basis": "allan_puiseux",
        "leading_power": float(-2.0 * (1.0 - exponent)),
        "leading_power_sem": (
            float(2.0 * exponent_sem) if exponent_sem is not None else None
        ),
        "design_powers": [
            float(-2.0 * (1.0 - exponent) + exponent * index)
            for index in range(resolved_order + 1)
        ],
        "residual_kind": "relative",
        "objective": "relative_residual_sum_of_squares",
        "sem_method": "profile_curvature",
        "loo_method": "refit" if calculate_loo else "disabled",
        "loo_profile_points": loo_profile_points if calculate_loo else None,
    }
    return ScalingFit(
        model="allan_relative",
        order=resolved_order,
        point_count=int(x.size),
        parameter_count=parameter_count,
        epsilon_scale=scale,
        exponent=float(exponent),
        exponent_sem=exponent_sem,
        exponent_bounds=resolved_bounds,
        at_exponent_bound=_at_bound(float(exponent), resolved_bounds),
        offset=0.0,
        coefficients=tuple(float(value) for value in coefficients),
        prediction=np.asarray(prediction, dtype=float),
        residual=np.asarray(values - prediction, dtype=float),
        rmse=float(np.sqrt(np.mean((values - prediction) ** 2))),
        max_abs_residual=float(np.max(np.abs(values - prediction))),
        loo_rmse=None,
        relative_rmse=float(np.sqrt(np.mean(relative_residual**2))),
        loo_relative_rmse=(
            float(np.sqrt(np.mean(loo_relative**2))) if calculate_loo else None
        ),
        aicc=_aicc(x.size, parameter_count, objective),
        bic=_bic(x.size, parameter_count, objective),
        weighted_reduced_chi2=None,
        condition=condition,
        objective=float(objective),
        weighted=False,
        anchor=None,
        profile_points=profile_points,
        metadata=metadata,
    )


def allan_window_diagnostics(
    epsilon: Any,
    intensity: Any,
    *,
    order: int,
    bounds: tuple[float, float],
    window_maxima: Sequence[float],
    inner_max: float,
    profile_points: int = 801,
) -> ScalingWindowDiagnostics:
    """Report how the Allan exponent moves with the fitting window.

    The windows are the caller's: nothing here assumes a target exponent or a
    sample window. ``inner_max`` splits the data for the extrapolation check.
    """
    x = _validated_coordinate(epsilon, minimum_points=2)
    values = _validated_response(intensity, "intensity", count=x.size, positive=True)
    resolved_bounds = _validated_bounds(bounds)
    maxima = tuple(float(item) for item in window_maxima)
    if not maxima:
        raise ValueError("window_maxima must not be empty")
    for item in maxima:
        if not np.isfinite(item) or item <= 0.0:
            raise ValueError(f"window maxima must be finite and positive; got {item}")
    resolved_inner = float(inner_max)
    if not np.isfinite(resolved_inner) or resolved_inner <= 0.0:
        raise ValueError(f"inner_max must be finite and positive; got {inner_max}")
    scale = float(np.max(x))
    coordinate = x / scale
    exponents: list[float] = []
    for value in maxima:
        selected = x <= value * (1.0 + 1.0e-12)
        count = int(np.count_nonzero(selected))
        if count < order + 3:
            raise ValueError(
                f"window {value} keeps {count} points, too few for order {order}"
            )
        exponents.append(
            _allan_exponent(
                coordinate[selected],
                values[selected],
                order=order,
                bounds=resolved_bounds,
                profile_points=profile_points,
            )
        )
    inner = x <= resolved_inner * (1.0 + 1.0e-12)
    inner_count = int(np.count_nonzero(inner))
    if inner_count < order + 3 or int(np.count_nonzero(~inner)) < 1:
        raise ValueError(
            f"inner_max {resolved_inner} does not split the data usefully "
            f"({inner_count} inner, {int(np.count_nonzero(~inner))} outer points)"
        )
    inner_exponent = _allan_exponent(
        coordinate[inner],
        values[inner],
        order=order,
        bounds=resolved_bounds,
        profile_points=profile_points,
    )
    _, coefficients = _allan_projection(
        coordinate[inner], values[inner], order, inner_exponent
    )
    outer_prediction = (
        _allan_design(coordinate[~inner], order, inner_exponent) @ coefficients
    )
    outer_relative = (values[~inner] - outer_prediction) / values[~inner]
    return ScalingWindowDiagnostics(
        window_maxima=maxima,
        window_exponents=tuple(exponents),
        span=float(max(exponents) - min(exponents)),
        inner_max=resolved_inner,
        outer_extrapolation_relative_rmse=float(np.sqrt(np.mean(outer_relative**2))),
        outer_extrapolation_max_abs_relative_error=float(
            np.max(np.abs(outer_relative))
        ),
        metadata={
            "basis": "allan_puiseux",
            "residual_kind": "relative",
            "window_rule": "x <= maximum * (1 + 1e-12)",
            "inner_rule": "x <= inner_max * (1 + 1e-12)",
        },
    )


def two_stage_frequency_fit(
    epsilon: Any,
    frequency: Any,
    *,
    lower: float,
    upper: float,
    exponent_bounds: tuple[float, float],
    stage1_exponent: float = 1.0 / 3.0,
    profile_points: int = 401,
) -> TwoStageFrequencyFit:
    """Fit a fixed one-third offset first, then a free exponent on all points.

    Stage one fits ``offset + a*x**(1/3)`` inside ``[lower, upper]``; stage two
    holds that offset and fits ``a*x**p`` over the **full** data range. The
    coordinate scale is the full range's maximum, as the historical script used.
    """
    x = _validated_coordinate(epsilon, minimum_points=3)
    y = _validated_response(frequency, "frequency", count=x.size)
    lo = float(lower)
    hi = float(upper)
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo >= hi:
        raise ValueError(
            f"the stage-one window must be finite and increasing: {lower}, {upper}"
        )
    resolved_bounds = _validated_bounds(exponent_bounds)
    stage1_power = float(stage1_exponent)
    if not np.isfinite(stage1_power) or stage1_power <= 0.0:
        raise ValueError(
            f"stage1_exponent must be finite and positive; got {stage1_exponent}"
        )
    scale = float(np.max(x))
    coordinate = x / scale
    selected = (x >= lo * (1.0 - 1.0e-12)) & (x <= hi * (1.0 + 1.0e-12))
    count = int(np.count_nonzero(selected))
    if count < 3:
        raise ValueError(
            f"the stage-one window keeps {count} points, too few for an offset fit"
        )
    stage1_design = np.column_stack(
        (np.ones(count), coordinate[selected] ** stage1_power)
    )
    stage1_coefficients, *_ = np.linalg.lstsq(stage1_design, y[selected], rcond=None)
    stage1_residual = stage1_design @ stage1_coefficients - y[selected]
    offset = float(stage1_coefficients[0])

    def profile(exponent: float) -> tuple[float, float, np.ndarray]:
        basis = coordinate**exponent
        amplitude = float(basis @ (y - offset) / (basis @ basis))
        residual = offset + amplitude * basis - y
        return float(residual @ residual), amplitude, residual

    minima = _profile_minima(
        lambda value: profile(float(value))[0],
        resolved_bounds,
        profile_points=profile_points,
    )
    _, exponent = min(minima)
    _, amplitude, residual = profile(exponent)
    prediction = offset + amplitude * coordinate**exponent
    return TwoStageFrequencyFit(
        lower=lo,
        upper=hi,
        stage1_exponent=stage1_power,
        stage1_point_count=count,
        stage1_offset=offset,
        stage1_amplitude=float(stage1_coefficients[1]),
        stage1_rmse=float(np.sqrt(np.mean(stage1_residual**2))),
        stage2_exponent=float(exponent),
        stage2_amplitude=amplitude,
        stage2_full_range_rmse=float(np.sqrt(np.mean(residual**2))),
        stage2_max_abs_residual=float(np.max(np.abs(residual))),
        stage2_prediction=np.asarray(prediction, dtype=float),
        stage2_residual=np.asarray(residual, dtype=float),
        exponent_bounds=resolved_bounds,
        epsilon_scale=scale,
        metadata={
            "stage1_model": "offset + amplitude * x**stage1_exponent (windowed)",
            "stage2_model": "offset(fixed) + amplitude * x**exponent (full range)",
            "stage2_range": [float(np.min(x)), float(np.max(x))],
            "residual_kind": "absolute",
            "point_count": int(x.size),
        },
    )
