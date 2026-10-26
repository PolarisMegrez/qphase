"""Scaling-signature classifier and scaling-fit numeric tests.

The numeric half is split by what an assertion can honestly claim:

* on exactly representable families (below) the fit must *recover* the generating
  exponent, coefficients and offset, and its residual is cancellation noise, so
  that residual is bounded relative to the data scale rather than frozen;
* on a family with a planted residual the statistics that need a known answer
  (rmse, leave-one-out, AICc/BIC, the exponent SEM) are checked against closed
  forms or against a reference computed independently in this file.

The frozen values are the recovery facts the historical scripts established; they
are properties of the design and the generating family, not of one BLAS build.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.reduction import LocalScalingSeries
from qphase_cam.solver.bifurcation_classifier import (
    ScalingSignatureClassifier,
    ScalingSignatureConfig,
)
from qphase_cam.solver.scaling import (
    allan_relative_fit,
    allan_window_diagnostics,
    fixed_exponent_fit,
    free_exponent_fit,
    two_stage_frequency_fit,
)
from scipy.optimize import minimize_scalar


class PolynomialReduction:
    def __init__(self, coefficients):
        self.coefficients = coefficients

    def local_scaling_series(self, value, **kwargs):
        del value, kwargs
        return LocalScalingSeries(
            coefficients=dict(self.coefficients),
            coefficient_decimals={
                key: str(value) for key, value in self.coefficients.items()
            },
            state_tangent=np.asarray([1.0]),
        )


class ScalarStateAdapter:
    @staticmethod
    def state_matrix(vector, params):
        del params
        return np.asarray([[np.asarray(vector)[0]]])


@pytest.mark.parametrize(
    ("coefficients", "expected", "exponent"),
    [
        ({(2, 0): 1.0, (0, 1): -1.0}, (2, 1, 0), 0.5),
        ({(2, 0): 1.0, (1, 1): -1.0}, (2, 1, 1), 1.0),
        ({(3, 0): 1.0, (1, 1): -1.0}, (3, 1, 1), 0.5),
        ({(3, 0): 1.0, (0, 1): -1.0}, (3, 1, 0), 1.0 / 3.0),
    ],
)
def test_scaling_signatures_match_canonical_normal_forms(
    coefficients, expected, exponent
):
    classifier = ScalingSignatureClassifier(ScalingSignatureConfig())
    result = classifier.classify(
        PolynomialReduction(coefficients),
        np.asarray([0.0]),
        np.asarray([0.0]),
        {},
        ScalarStateAdapter(),
        perturbation="mu",
        scale=1.0,
        side="both",
        verification_digits=50,
    )
    signature = result["scaling_signatures"][0]
    assert (
        signature["state_order"],
        signature["perturbation_order"],
        signature["coupling_state_order"],
    ) == expected
    assert signature["exponent"] == pytest.approx(exponent)
    assert signature["sublinear"] is (exponent < 1.0)


def test_transcritical_is_rejected_by_sublinear_acceptance():
    classifier = ScalingSignatureClassifier(ScalingSignatureConfig(max_exponent=0.999))
    result = classifier.classify(
        PolynomialReduction({(2, 0): 1.0, (1, 1): -1.0}),
        np.asarray([0.0]),
        np.asarray([0.0]),
        {},
        ScalarStateAdapter(),
        perturbation="mu",
        scale=1.0,
        side="both",
        verification_digits=50,
    )
    assert not result["classification_accepted"]


# --- frozen synthetic references for the numeric scaling layer -------------

X = np.geomspace(1.0e-4, 1.0e-1, 9)
CRITICAL = 1.234
UNCERTAINTY = 0.01 * np.ones_like(X)
CAM_FREQUENCY = CRITICAL + 2.5 * X ** (1.0 / 3.0) - 0.8 * X ** (2.0 / 3.0)
G1_FREQUENCY = 0.75 + 2.5 * X ** (1.0 / 3.0) - 0.8 * X ** (2.0 / 3.0)
ALLAN_INTENSITY = X ** (-4.0 / 3.0) + 0.5 * X ** (-1.0) + 0.25 * X ** (-2.0 / 3.0)
THIRD = 1.0 / 3.0

#: Recovery facts — the exponent, the coefficients, the offset, the condition
#: number and the counts are properties of the design and the generating family.
EXACT = {"rel": 1.0e-9, "abs": 0.0}
#: Profile-search resolution on the two-stage objective (401-point grid).
PROFILE = {"rel": 1.0e-8, "abs": 0.0}
#: A residual that is cancellation noise, stated as a fraction of the response
#: scale. These data are exact polynomials, so a faithful fit reproduces them at
#: this level; the last digits of that residual belong to the platform, not to the
#: result, which is why it is bounded here and given a known answer below.
ROUNDING = 1.0e-12
#: A deterministic non-zero residual (the anchored and two-stage references
#: below) reproduces at a relative tolerance far above the platform's last digits.
STABLE = {"rel": 1.0e-6, "abs": 0.0}


def _aicc_of(sse: float, count: int, parameters: int) -> float:
    """AICc from a residual sum of squares, as the module documents it."""
    aic = count * np.log(max(sse / count, np.finfo(float).tiny)) + 2.0 * parameters
    return float(aic + 2.0 * parameters * (parameters + 1) / (count - parameters - 1))


def _bic_of(sse: float, count: int, parameters: int) -> float:
    """BIC from a residual sum of squares, as the module documents it."""
    return float(
        count * np.log(max(sse / count, np.finfo(float).tiny))
        + parameters * np.log(count)
    )


def test_free_exponent_fit_with_anchor_recovers_the_generating_family():
    """The anchored fit reproduces CAM's exact Puiseux family.

    The exponent lands at 1/3 and the coefficients and offset are the generating
    ones; the residual of such a fit is cancellation noise, so its size is bounded
    relative to the response scale and the criterion is checked as a definition.
    """
    fit = free_exponent_fit(
        X,
        CAM_FREQUENCY,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=UNCERTAINTY,
        anchor=CRITICAL,
    )
    assert fit.model == "free_exponent"
    assert fit.parameter_count == 3
    assert fit.degrees_of_freedom == 6
    assert fit.epsilon_scale == pytest.approx(0.1, **EXACT)
    assert fit.exponent == pytest.approx(0.3333333333236878, **EXACT)
    assert fit.exponent_bounds == (0.25, 0.45)
    assert fit.at_exponent_bound is False
    assert fit.offset == pytest.approx(CRITICAL, **EXACT)
    assert fit.anchor == CRITICAL
    assert fit.coefficients == pytest.approx(
        (1.1603972083531455, -0.17235477515035624), **EXACT
    )
    assert fit.leading_coefficient == pytest.approx(1.1603972083531455, **EXACT)
    assert fit.condition == pytest.approx(33.20959999861707, **EXACT)
    assert fit.profile_points == 401
    # The estimated exponent carries its own ~1e-12 misfit, so a free fit reproduces
    # the exact family one order above the fixed-exponent case; all of these are
    # still ten orders below the 1e-2 data uncertainty, which is the point.
    scale = float(np.max(np.abs(CAM_FREQUENCY)))
    assert 0.0 <= fit.rmse <= 1.0e-10 * scale
    assert 0.0 <= fit.max_abs_residual <= 1.0e-9 * scale
    assert 0.0 <= fit.loo_rmse <= 1.0e-8 * scale
    assert 0.0 < fit.exponent_sem <= 1.0e-9
    assert fit.aicc == pytest.approx(
        _aicc_of(fit.objective, fit.point_count, fit.parameter_count), rel=1.0e-12
    )
    assert fit.weighted_reduced_chi2 == pytest.approx(
        float(np.sum((fit.residual / UNCERTAINTY) ** 2)) / fit.degrees_of_freedom,
        rel=1.0e-12,
    )


def test_free_exponent_fit_with_free_offset_recovers_the_weighted_family():
    """The weighted free-offset fit recovers the same family without an anchor."""
    fit = free_exponent_fit(
        X,
        G1_FREQUENCY,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=UNCERTAINTY,
        weighted=True,
    )
    assert fit.parameter_count == 4
    assert fit.anchor is None
    assert fit.exponent == pytest.approx(0.33333333332161447, **EXACT)
    assert fit.offset == pytest.approx(0.7499999999919966, **EXACT)
    assert fit.coefficients == pytest.approx(
        (1.1603972083777778, -0.17235477516852948), **EXACT
    )
    assert fit.condition == pytest.approx(90.15703475465155, **EXACT)
    assert fit.weighted is True
    scale = float(np.max(np.abs(G1_FREQUENCY)))
    assert 0.0 <= fit.rmse <= 1.0e-10 * scale
    assert 0.0 <= fit.loo_rmse <= 1.0e-8 * scale
    assert 0.0 < fit.exponent_sem <= 1.0e-9
    assert fit.aicc == pytest.approx(
        _aicc_of(fit.objective, fit.point_count, fit.parameter_count), rel=1.0e-12
    )


def test_fixed_exponent_fit_recovers_the_generating_family():
    """A fixed one-third fit recovers the polynomial it was given.

    The data are exactly in the fitted span, so the residual is rounding noise:
    the coefficients and offset are the recovery facts, and leverage LOO can only
    amplify that noise, not measure a model error.
    """
    fit = fixed_exponent_fit(
        X, G1_FREQUENCY, exponent=THIRD, order=2, uncertainty=UNCERTAINTY
    )
    assert fit.model == "fixed_exponent"
    assert fit.parameter_count == 3
    assert fit.degrees_of_freedom == 6
    assert fit.exponent == pytest.approx(THIRD, **EXACT)
    assert fit.exponent_sem is None
    assert fit.exponent_bounds is None
    assert fit.profile_points is None
    assert fit.offset == pytest.approx(0.7500000000000004, **EXACT)
    assert fit.coefficients == pytest.approx(
        (1.160397208403194, -0.17235477520255027), **EXACT
    )
    assert fit.condition == pytest.approx(22.57297189010064, **EXACT)
    assert fit.metadata["loo_method"] == "leverage"
    scale = float(np.max(np.abs(G1_FREQUENCY)))
    assert 0.0 <= fit.rmse <= ROUNDING * scale
    assert 0.0 <= fit.loo_rmse <= 100.0 * ROUNDING * scale
    assert fit.aicc == pytest.approx(
        _aicc_of(fit.objective, fit.point_count, fit.parameter_count), rel=1.0e-12
    )
    assert fit.weighted_reduced_chi2 == pytest.approx(
        float(np.sum((fit.residual / UNCERTAINTY) ** 2)) / fit.degrees_of_freedom,
        rel=1.0e-12,
    )


def test_fixed_exponent_fit_with_anchor_reproduces_reference():
    fit = fixed_exponent_fit(
        X,
        G1_FREQUENCY,
        exponent=THIRD,
        order=2,
        uncertainty=UNCERTAINTY,
        offset=False,
        anchor=CRITICAL,
    )
    assert fit.parameter_count == 2
    assert fit.offset == pytest.approx(CRITICAL, **EXACT)
    assert fit.anchor == CRITICAL
    assert fit.coefficients == pytest.approx(
        (-0.9796026468810815, 1.604998177364615), **EXACT
    )
    assert fit.rmse == pytest.approx(0.16057662565209904, **STABLE)
    assert fit.max_abs_residual == pytest.approx(0.2877735439972441, **STABLE)
    assert fit.loo_rmse == pytest.approx(0.29249630186206493, **STABLE)
    assert fit.aicc == pytest.approx(-26.921712575042903, **STABLE)
    assert fit.weighted_reduced_chi2 == pytest.approx(331.51953478904164, **STABLE)
    assert fit.condition == pytest.approx(6.292330236666059, **EXACT)


def test_allan_relative_fit_recovers_the_generating_intensity():
    """The Allan fit recovers the three-term intensity it was given.

    The relative residual is cancellation noise, so it is bounded relative to 1
    and AICc/BIC are checked against their definitions; the exponent, the leading
    power, the coefficients and the condition are the recovery facts.
    """
    fit = allan_relative_fit(X, ALLAN_INTENSITY, order=2, bounds=(0.25, 0.45))
    assert fit.model == "allan_relative"
    assert fit.order == 2
    assert fit.parameter_count == 4
    assert fit.exponent == pytest.approx(0.33333333335783366, **EXACT)
    assert fit.metadata["leading_power"] == pytest.approx(-1.3333333332843327, **EXACT)
    assert fit.coefficients == pytest.approx(
        (21.54434690901204, 4.99999998368522, 1.1603972162620213), **EXACT
    )
    assert fit.condition == pytest.approx(86.77341857057084, **EXACT)
    assert fit.at_exponent_bound is False
    assert fit.loo_rmse is None
    assert fit.weighted_reduced_chi2 is None
    assert 0.0 <= fit.relative_rmse <= 1.0e-8
    assert 0.0 <= float(np.max(np.abs(fit.residual / ALLAN_INTENSITY))) <= 1.0e-7
    assert 0.0 <= fit.loo_relative_rmse <= 1.0e-6
    assert 0.0 < fit.exponent_sem <= 1.0e-8
    assert fit.aicc == pytest.approx(
        _aicc_of(fit.objective, fit.point_count, fit.parameter_count), rel=1.0e-12
    )
    assert fit.bic == pytest.approx(
        _bic_of(fit.objective, fit.point_count, fit.parameter_count), rel=1.0e-12
    )


def test_allan_window_diagnostics_recover_the_generating_exponent():
    """Every window recovers the same exponent on an exact intensity.

    ``span`` is the spread of the recovered exponents, so it is checked against
    that definition and bounded: on an intensity that is one Puiseux family in
    every window the span can only reflect the recovery precision. The
    extrapolation errors are of the same order, so they are bounded as relative
    quantities; a window geometry with a genuine exponent drift gets a known
    answer in the planted-residual section below.
    """
    diagnostics = allan_window_diagnostics(
        X,
        ALLAN_INTENSITY,
        order=2,
        bounds=(0.25, 0.45),
        window_maxima=(1.0e-2, 10.0**-1.5, 1.0e-1),
        inner_max=1.0e-2,
    )
    assert diagnostics.window_maxima == (1.0e-2, 10.0**-1.5, 1.0e-1)
    assert diagnostics.inner_max == 1.0e-2
    assert diagnostics.window_exponents[0] == pytest.approx(0.3333333333617728, **EXACT)
    assert diagnostics.window_exponents[1] == pytest.approx(
        0.33333333336056764, **EXACT
    )
    assert diagnostics.span == pytest.approx(
        max(diagnostics.window_exponents) - min(diagnostics.window_exponents),
        rel=1.0e-12,
    )
    assert 0.0 <= diagnostics.span <= 1.0e-9
    assert 0.0 <= diagnostics.outer_extrapolation_relative_rmse <= 1.0e-7
    assert 0.0 <= diagnostics.outer_extrapolation_max_abs_relative_error <= 1.0e-6


def test_two_stage_frequency_fit_reproduces_reference():
    fit = two_stage_frequency_fit(
        X, G1_FREQUENCY, lower=1.0e-4, upper=1.0e-1, exponent_bounds=(0.25, 0.45)
    )
    assert fit.stage1_exponent == pytest.approx(THIRD, **EXACT)
    assert fit.stage1_point_count == 9
    assert fit.stage1_offset == pytest.approx(0.7813297950191174, **EXACT)
    assert fit.stage1_amplitude == pytest.approx(0.9782777876699669, **EXACT)
    assert fit.stage1_rmse == pytest.approx(0.012722230022209065, **STABLE)
    assert fit.stage2_exponent == pytest.approx(0.3269426390146774, **PROFILE)
    assert fit.stage2_amplitude == pytest.approx(0.971603457009521, **PROFILE)
    assert fit.stage2_full_range_rmse == pytest.approx(0.011948045472374755, **STABLE)
    assert fit.stage2_max_abs_residual == pytest.approx(0.018559227609663553, **PROFILE)
    assert fit.exponent_bounds == (0.25, 0.45)
    assert fit.lower == 1.0e-4
    assert fit.upper == 1.0e-1


def test_two_stage_fit_limits_stage_one_to_the_requested_window():
    fit = two_stage_frequency_fit(
        X, G1_FREQUENCY, lower=1.0e-4, upper=1.0e-3, exponent_bounds=(0.25, 0.45)
    )
    assert fit.stage1_point_count == int(np.count_nonzero(X <= 1.0e-3))
    assert fit.stage2_exponent >= 0.25
    assert fit.stage2_full_range_rmse == pytest.approx(
        np.sqrt(np.mean(fit.stage2_residual**2)), **STABLE
    )


def test_free_exponent_fit_reports_an_exponent_pinned_to_the_bound():
    fit = free_exponent_fit(
        X, X**0.9, order=1, bounds=(0.25, 0.45), calculate_loo=False
    )
    assert fit.exponent == pytest.approx(0.45, **EXACT)
    assert fit.at_exponent_bound is True


def test_wide_bounds_do_not_move_the_exponent_off_the_reference():
    narrow = free_exponent_fit(
        X,
        CAM_FREQUENCY,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=UNCERTAINTY,
        anchor=CRITICAL,
        calculate_loo=False,
    )
    wide = free_exponent_fit(
        X,
        CAM_FREQUENCY,
        order=2,
        bounds=(0.10, 0.70),
        uncertainty=UNCERTAINTY,
        anchor=CRITICAL,
        calculate_loo=False,
    )
    assert wide.exponent == pytest.approx(narrow.exponent, rel=1.0e-6, abs=0.0)


def test_leave_one_out_can_be_skipped():
    fit = free_exponent_fit(
        X,
        CAM_FREQUENCY,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=UNCERTAINTY,
        anchor=CRITICAL,
        calculate_loo=False,
    )
    assert fit.loo_rmse is None
    assert fit.metadata["loo_method"] == "disabled"


# --- known-answer statistics on a planted residual -------------------------
#
# Freezing a rmse/LOO/AICc/SEM value is only honest when the residual that feeds it
# is a number the platform cannot move. The family below carries a planted residual
# of 1e-3 of the response scale — four orders above the rounding level — so each
# statistic has either a closed form (fixed exponent, orthogonal residual) or the
# independent reference computed by the small oracles in this file.

SAMPLE_X = np.asarray([1.0e-4, 3.0e-4, 1.0e-3, 3.0e-3, 1.0e-2, 3.0e-2, 6.0e-2, 1.0e-1])
SAMPLE_SCALE = float(np.max(SAMPLE_X))
SAMPLE_COORDINATE = SAMPLE_X / SAMPLE_SCALE
SAMPLE_EXPONENT = THIRD
SAMPLE_COEFFICIENTS = np.asarray([1.5, 2.0, -0.5])
SAMPLE_RESIDUAL = 1.0e-3
SAMPLE_UNCERTAINTY = 0.01


def _puiseux_design(coordinate, order, exponent, *, offset=True):
    """Build the Puiseux design here instead of borrowing the module's."""
    columns = [coordinate ** (power * exponent) for power in range(1, order + 1)]
    if offset:
        return np.column_stack((np.ones(coordinate.size), *columns))
    return np.column_stack(columns)


def _unit_orthogonal_direction(design):
    """Return a unit vector orthogonal to every design column."""
    raw = np.arange(1.0, design.shape[0] + 1.0)
    coefficients, *_ = np.linalg.lstsq(design, raw, rcond=None)
    direction = raw - design @ coefficients
    return direction / np.linalg.norm(direction)


def _oracle_minimise(objective, bounds, points=401):
    """Independent bounded search: a coarse grid, then a refinement of its best."""
    grid = np.linspace(bounds[0], bounds[1], points)
    values = np.asarray([objective(float(value)) for value in grid])
    best = int(np.argmin(values))
    low = bounds[0] if best == 0 else float(grid[best - 1])
    high = bounds[1] if best == len(grid) - 1 else float(grid[best + 1])
    optimum = minimize_scalar(
        objective, bounds=(low, high), method="bounded", options={"xatol": 1.0e-13}
    )
    return float(optimum.x)


def _puiseux_reference(response, *, order, bounds, anchor=None):
    """Least-squares Puiseux fit and its statistics, computed in this file."""
    coordinate = SAMPLE_COORDINATE
    offset = anchor is None
    target = response if offset else response - float(anchor)
    parameter_count = order + (2 if offset else 1)
    size = coordinate.size

    def projected(exponent):
        design = _puiseux_design(coordinate, order, exponent, offset=offset)
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        residual = target - design @ coefficients
        return float(residual @ residual), coefficients, design

    exponent = _oracle_minimise(lambda value: projected(float(value))[0], bounds)
    rss, coefficients, design = projected(exponent)
    puiseux = coefficients[1:] if offset else coefficients
    derivative = np.sum(
        [
            puiseux[power - 1]
            * power
            * np.log(coordinate)
            * coordinate ** (power * exponent)
            for power in range(1, order + 1)
        ],
        axis=0,
    )
    jacobian = np.column_stack((design, derivative))
    covariance = np.linalg.pinv(jacobian.T @ jacobian) * (
        rss / (size - parameter_count)
    )
    prediction = design @ coefficients + (0.0 if offset else float(anchor))
    loo = []
    for omitted in range(size):
        retained = np.arange(size) != omitted
        local_coordinate = coordinate[retained]
        local_target = target[retained]

        def local_projected(exponent, coordinate=local_coordinate, target=local_target):
            local_design = _puiseux_design(coordinate, order, exponent, offset=offset)
            coefficients, *_ = np.linalg.lstsq(local_design, target, rcond=None)
            residual = target - local_design @ coefficients
            return float(residual @ residual), coefficients

        local_exponent = _oracle_minimise(
            lambda value: local_projected(float(value))[0], bounds
        )
        _, local_coefficients = local_projected(local_exponent)
        single = _puiseux_design(
            coordinate[[omitted]], order, local_exponent, offset=offset
        )
        omitted_prediction = float((single @ local_coefficients)[0]) + (
            0.0 if offset else float(anchor)
        )
        loo.append(float(response[omitted] - omitted_prediction))
    return {
        "exponent": exponent,
        "coefficients": coefficients,
        "rss": rss,
        "rmse": float(np.sqrt(np.mean((response - prediction) ** 2))),
        "loo_rmse": float(np.sqrt(np.mean(np.square(loo)))),
        "exponent_sem": float(np.sqrt(max(float(covariance[-1, -1]), 0.0))),
        "aicc": _aicc_of(rss, size, parameter_count),
    }


def _allan_design(coordinate, order, exponent):
    """Build the Allan Puiseux design here instead of borrowing the module's."""
    powers = -2.0 * (1.0 - exponent) + exponent * np.arange(order + 1)
    return coordinate[:, None] ** powers[None, :]


def _allan_reference(values, *, order, bounds, profile_points=801, loo_points=401):
    """Relative-residual Allan fit and its statistics, computed in this file."""
    coordinate = SAMPLE_COORDINATE
    size = values.size
    parameter_count = order + 2
    degrees_of_freedom = size - parameter_count

    def projected(exponent, coordinate=coordinate, values=values):
        design = _allan_design(coordinate, order, exponent)
        coefficients, *_ = np.linalg.lstsq(
            design / values[:, None], np.ones(values.size), rcond=None
        )
        relative = (values - design @ coefficients) / values
        return float(relative @ relative), coefficients

    exponent = _oracle_minimise(
        lambda value: projected(float(value))[0], bounds, points=profile_points
    )
    objective, coefficients = projected(exponent)
    step = 1.0e-4
    curvature = (
        projected(exponent + step)[0] - 2.0 * objective + projected(exponent - step)[0]
    ) / step**2
    relative = (
        values - _allan_design(coordinate, order, exponent) @ coefficients
    ) / values
    loo = []
    for omitted in range(size):
        retained = np.arange(size) != omitted
        local_coordinate = coordinate[retained]
        local_values = values[retained]

        def local_projected(exponent, coordinate=local_coordinate, values=local_values):
            design = _allan_design(coordinate, order, exponent)
            coefficients, *_ = np.linalg.lstsq(
                design / values[:, None], np.ones(values.size), rcond=None
            )
            relative = (values - design @ coefficients) / values
            return float(relative @ relative), coefficients

        local_exponent = _oracle_minimise(
            lambda value: local_projected(float(value))[0], bounds, points=loo_points
        )
        _, local_coefficients = local_projected(local_exponent)
        prediction = float(
            (
                _allan_design(coordinate[[omitted]], order, local_exponent)
                @ local_coefficients
            )[0]
        )
        loo.append(float((values[omitted] - prediction) / values[omitted]))
    return {
        "exponent": exponent,
        "objective": objective,
        "relative_rmse": float(np.sqrt(np.mean(relative**2))),
        "max_abs_relative": float(np.max(np.abs(relative))),
        "loo_relative_rmse": float(np.sqrt(np.mean(np.square(loo)))),
        "exponent_sem": float(
            np.sqrt(2.0 * (objective / degrees_of_freedom) / curvature)
        ),
        "aicc": _aicc_of(objective, size, parameter_count),
        "bic": _bic_of(objective, size, parameter_count),
        "leading_power": float(-2.0 * (1.0 - exponent)),
    }


def _allan_window_reference(
    values, *, order, bounds, window_maxima, inner_max, profile_points=801
):
    """Per-window Allan exponents, span and extrapolation, computed in this file."""

    def exponent_and_coefficients(selection):
        coordinate = SAMPLE_COORDINATE[selection]
        local_values = values[selection]

        def projected(exponent, coordinate=coordinate, values=local_values):
            design = _allan_design(coordinate, order, exponent)
            coefficients, *_ = np.linalg.lstsq(
                design / values[:, None], np.ones(values.size), rcond=None
            )
            relative = (values - design @ coefficients) / values
            return float(relative @ relative), coefficients

        exponent = _oracle_minimise(
            lambda value: projected(float(value))[0], bounds, points=profile_points
        )
        return exponent, projected(exponent)[1]

    exponents = [
        exponent_and_coefficients(SAMPLE_X <= maximum * (1.0 + 1.0e-12))[0]
        for maximum in window_maxima
    ]
    inner = SAMPLE_X <= inner_max * (1.0 + 1.0e-12)
    inner_exponent, inner_coefficients = exponent_and_coefficients(inner)
    prediction = (
        _allan_design(SAMPLE_COORDINATE[~inner], order, inner_exponent)
        @ inner_coefficients
    )
    outer_relative = (values[~inner] - prediction) / values[~inner]
    return {
        "window_exponents": exponents,
        "span": float(max(exponents) - min(exponents)),
        "outer_relative_rmse": float(np.sqrt(np.mean(outer_relative**2))),
        "outer_max_abs_relative": float(np.max(np.abs(outer_relative))),
    }


def _planted_puiseux_response(exponent=SAMPLE_EXPONENT):
    """Return the generating family plus a unit orthogonal residual."""
    design = _puiseux_design(SAMPLE_COORDINATE, 2, exponent)
    direction = _unit_orthogonal_direction(design)
    return design @ SAMPLE_COEFFICIENTS + SAMPLE_RESIDUAL * direction, direction


def _planted_allan_intensity():
    """Return a positive three-term intensity with a 1e-3 relative residual."""
    model = SAMPLE_COORDINATE ** (-4.0 / 3.0) * (
        1.0 + 0.5 * SAMPLE_COORDINATE ** (1.0 / 3.0)
    )
    direction = np.where(np.arange(SAMPLE_X.size) % 2 == 0, 1.0, -1.0)
    direction = direction / np.linalg.norm(direction)
    return model * (1.0 + SAMPLE_RESIDUAL * np.sqrt(SAMPLE_X.size) * direction)


def test_fixed_exponent_fit_recovers_a_planted_residual_with_closed_forms():
    """Every statistic here has a closed form from the planted residual.

    The residual direction is orthogonal to the design, so the least-squares
    solution is the generating coefficient vector and the residual is the planted
    one: rmse = |r|/sqrt(n), leverage LOO = r/(1-h), and AICc follows from the RSS.
    """
    response, direction = _planted_puiseux_response()
    design = _puiseux_design(SAMPLE_COORDINATE, 2, SAMPLE_EXPONENT)
    fit = fixed_exponent_fit(
        SAMPLE_X,
        response,
        exponent=SAMPLE_EXPONENT,
        order=2,
        uncertainty=np.full(SAMPLE_X.size, SAMPLE_UNCERTAINTY),
    )
    assert fit.coefficients == pytest.approx(SAMPLE_COEFFICIENTS[1:], rel=1.0e-12)
    assert fit.offset == pytest.approx(SAMPLE_COEFFICIENTS[0], rel=1.0e-12)
    assert fit.residual == pytest.approx(
        SAMPLE_RESIDUAL * direction, rel=1.0e-10, abs=1.0e-18
    )
    assert fit.rmse == pytest.approx(
        SAMPLE_RESIDUAL / np.sqrt(SAMPLE_X.size), rel=1.0e-10
    )
    assert fit.max_abs_residual == pytest.approx(
        SAMPLE_RESIDUAL * float(np.max(np.abs(direction))), rel=1.0e-10
    )
    gram_inverse = np.linalg.inv(design.T @ design)
    leverage = np.einsum("ij,jk,ik->i", design, gram_inverse, design)
    loo = SAMPLE_RESIDUAL * direction / (1.0 - leverage)
    assert fit.loo_rmse == pytest.approx(float(np.sqrt(np.mean(loo**2))), rel=1.0e-10)
    rss = float(np.sum((SAMPLE_RESIDUAL * direction) ** 2))
    assert fit.objective == pytest.approx(rss, rel=1.0e-12)
    assert fit.aicc == pytest.approx(_aicc_of(rss, SAMPLE_X.size, 3), rel=1.0e-12)
    assert fit.weighted_reduced_chi2 == pytest.approx(
        rss / SAMPLE_UNCERTAINTY**2 / (SAMPLE_X.size - 3), rel=1.0e-12
    )
    assert fit.condition == pytest.approx(float(np.linalg.cond(design)), rel=1.0e-12)


def test_free_exponent_fit_statistics_match_an_independent_reference():
    """A free exponent on a planted residual: exponent, SEM, rmse, LOO and AICc.

    The reference solves the same least-squares problem in this file with its own
    one-dimensional search; nothing is regenerated from the module under test.
    """
    response, _ = _planted_puiseux_response(0.35)
    reference = _puiseux_reference(response, order=2, bounds=(0.25, 0.45))
    fit = free_exponent_fit(
        SAMPLE_X,
        response,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=np.full(SAMPLE_X.size, SAMPLE_UNCERTAINTY),
    )
    assert fit.exponent == pytest.approx(reference["exponent"], rel=1.0e-9)
    # The planted residual biases a free exponent by a few times its own size (it
    # is orthogonal at the planted exponent, not at its neighbours); the strict
    # check is the reference comparison above, this one keeps the bias bounded.
    assert abs(fit.exponent - 0.35) <= 3.0 * SAMPLE_RESIDUAL
    assert fit.coefficients == pytest.approx(reference["coefficients"][1:], rel=1.0e-8)
    assert fit.exponent_sem == pytest.approx(reference["exponent_sem"], rel=1.0e-6)
    assert fit.rmse == pytest.approx(reference["rmse"], rel=1.0e-8)
    assert fit.loo_rmse == pytest.approx(reference["loo_rmse"], rel=1.0e-6)
    assert fit.aicc == pytest.approx(reference["aicc"], rel=1.0e-9)
    assert fit.metadata["loo_method"] == "refit"


def test_allan_relative_fit_statistics_match_an_independent_reference():
    """The relative-residual statistics on a planted 1e-3 relative residual.

    The SEM comes from the curvature of the profiled objective, so it needs a
    residual that is not rounding noise; the reference differentiates the same
    profile in this file.
    """
    values = _planted_allan_intensity()
    reference = _allan_reference(values, order=2, bounds=(0.25, 0.45))
    fit = allan_relative_fit(SAMPLE_X, values, order=2, bounds=(0.25, 0.45))
    assert fit.exponent == pytest.approx(reference["exponent"], rel=1.0e-9)
    assert fit.relative_rmse == pytest.approx(reference["relative_rmse"], rel=1.0e-8)
    assert float(np.max(np.abs(fit.residual / values))) == pytest.approx(
        reference["max_abs_relative"], rel=1.0e-8
    )
    assert fit.loo_relative_rmse == pytest.approx(
        reference["loo_relative_rmse"], rel=1.0e-6
    )
    assert fit.exponent_sem == pytest.approx(reference["exponent_sem"], rel=1.0e-6)
    assert fit.metadata["leading_power"] == pytest.approx(
        reference["leading_power"], rel=1.0e-9
    )
    assert fit.aicc == pytest.approx(reference["aicc"], rel=1.0e-9)
    assert fit.bic == pytest.approx(reference["bic"], rel=1.0e-9)
    # The planted residual is a real signal: the fit cannot reach the family.
    assert fit.relative_rmse > 1.0e-4


def test_allan_window_diagnostics_span_has_a_known_answer_when_it_drifts():
    """A drifting effective exponent gives a span far above the recovery level.

    The intensity is not one Puiseux family, so the windows genuinely disagree;
    the reference repeats the per-window fit in this file and the test requires a
    span that the platform noise cannot produce.
    """
    values = SAMPLE_COORDINATE ** (-4.0 / 3.0) * (
        1.0 + 0.5 * SAMPLE_COORDINATE ** (1.0 / 3.0)
    ) ** (-1.0)
    maxima = (1.0e-2, 10.0**-1.5, 1.0e-1)
    reference = _allan_window_reference(
        values, order=2, bounds=(0.25, 0.45), window_maxima=maxima, inner_max=1.0e-2
    )
    diagnostics = allan_window_diagnostics(
        SAMPLE_X,
        values,
        order=2,
        bounds=(0.25, 0.45),
        window_maxima=maxima,
        inner_max=1.0e-2,
    )
    for observed, expected in zip(
        diagnostics.window_exponents, reference["window_exponents"], strict=True
    ):
        assert observed == pytest.approx(expected, rel=1.0e-9)
    assert diagnostics.span == pytest.approx(reference["span"], rel=1.0e-8)
    assert diagnostics.span > 1.0e-4
    assert diagnostics.outer_extrapolation_relative_rmse == pytest.approx(
        reference["outer_relative_rmse"], rel=1.0e-6
    )
    assert diagnostics.outer_extrapolation_max_abs_relative_error == pytest.approx(
        reference["outer_max_abs_relative"], rel=1.0e-6
    )


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(
            lambda: fixed_exponent_fit(
                X, G1_FREQUENCY, exponent=THIRD, order=2, offset=False
            ),
            id="missing-anchor",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(
                X, G1_FREQUENCY, exponent=THIRD, order=2, anchor=CRITICAL
            ),
            id="offset-and-anchor",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(-X, G1_FREQUENCY, exponent=THIRD, order=2),
            id="non-positive-coordinate",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(X, G1_FREQUENCY, exponent=0.0, order=2),
            id="non-positive-exponent",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(X, G1_FREQUENCY, exponent=THIRD, order=2.5),
            id="non-integer-order",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(X, G1_FREQUENCY, exponent=THIRD, order=0),
            id="zero-order",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(X, G1_FREQUENCY, exponent=THIRD, order=8),
            id="no-degrees-of-freedom",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(
                X, G1_FREQUENCY, exponent=THIRD, order=2, uncertainty=-UNCERTAINTY
            ),
            id="non-positive-uncertainty",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(X, G1_FREQUENCY[:3], exponent=THIRD, order=2),
            id="response-length-mismatch",
        ),
        pytest.param(
            lambda: fixed_exponent_fit(
                np.where(X > 1e-2, np.inf, X), G1_FREQUENCY, exponent=THIRD, order=2
            ),
            id="non-finite-coordinate",
        ),
        pytest.param(
            lambda: free_exponent_fit(X, G1_FREQUENCY, order=2, bounds=(0.45, 0.25)),
            id="bounds-not-increasing",
        ),
        pytest.param(
            lambda: free_exponent_fit(X, G1_FREQUENCY, order=2, bounds=(0.0, 0.45)),
            id="bounds-not-positive",
        ),
        pytest.param(
            lambda: allan_relative_fit(
                X, -ALLAN_INTENSITY, order=2, bounds=(0.25, 0.45)
            ),
            id="non-positive-intensity",
        ),
        pytest.param(
            lambda: allan_relative_fit(
                X, ALLAN_INTENSITY, order=2, bounds=(0.25, 0.45), profile_points=2
            ),
            id="profile-too-coarse",
        ),
        pytest.param(
            lambda: allan_window_diagnostics(
                X,
                ALLAN_INTENSITY,
                order=2,
                bounds=(0.25, 0.45),
                window_maxima=(1.0e-6,),
                inner_max=1.0e-2,
            ),
            id="window-too-narrow",
        ),
        pytest.param(
            lambda: allan_window_diagnostics(
                X,
                ALLAN_INTENSITY,
                order=2,
                bounds=(0.25, 0.45),
                window_maxima=(1.0e-1,),
                inner_max=1.0e-1,
            ),
            id="inner-window-covers-everything",
        ),
        pytest.param(
            lambda: allan_window_diagnostics(
                X,
                ALLAN_INTENSITY,
                order=2,
                bounds=(0.25, 0.45),
                window_maxima=(),
                inner_max=1.0e-2,
            ),
            id="no-windows",
        ),
        pytest.param(
            lambda: two_stage_frequency_fit(
                X,
                G1_FREQUENCY,
                lower=1.0e-1,
                upper=1.0e-4,
                exponent_bounds=(0.25, 0.45),
            ),
            id="stage-one-window-inverted",
        ),
        pytest.param(
            lambda: two_stage_frequency_fit(
                X,
                G1_FREQUENCY,
                lower=1.0e-4,
                upper=2.0e-4,
                exponent_bounds=(0.25, 0.45),
            ),
            id="stage-one-window-too-narrow",
        ),
        pytest.param(
            lambda: two_stage_frequency_fit(
                X,
                G1_FREQUENCY,
                lower=1.0e-4,
                upper=1.0e-1,
                exponent_bounds=(0.25, 0.25),
            ),
            id="degenerate-exponent-bounds",
        ),
    ],
)
def test_fit_inputs_are_rejected_instead_of_coerced(call):
    with pytest.raises(ValueError):
        call()


def test_classifier_exposes_the_scaling_layer():
    classifier = ScalingSignatureClassifier(ScalingSignatureConfig())
    anchored = classifier.fit_puiseux(
        X,
        CAM_FREQUENCY,
        order=2,
        bounds=(0.25, 0.45),
        uncertainty=UNCERTAINTY,
        anchor=CRITICAL,
    )
    assert anchored.exponent == pytest.approx(0.3333333333236878, **EXACT)
    fixed = classifier.fit_puiseux(
        X, G1_FREQUENCY, exponent=THIRD, order=2, uncertainty=UNCERTAINTY
    )
    assert fixed.offset == pytest.approx(0.7500000000000004, **EXACT)
    assert fixed.parameter_count == 3
    allan = classifier.fit_allan(
        X, ALLAN_INTENSITY, order=2, bounds=(0.25, 0.45), calculate_loo=False
    )
    assert allan.exponent == pytest.approx(0.33333333335783366, **EXACT)
    assert allan.loo_relative_rmse is None
    two_stage = classifier.fit_two_stage(
        X, G1_FREQUENCY, lower=1.0e-4, upper=1.0e-1, exponent_bounds=(0.25, 0.45)
    )
    assert two_stage.stage2_exponent == pytest.approx(0.3269426390146774, **PROFILE)


def test_classifier_rejects_ambiguous_puiseux_requests():
    classifier = ScalingSignatureClassifier(ScalingSignatureConfig())
    with pytest.raises(ValueError):
        classifier.fit_puiseux(X, G1_FREQUENCY, order=2)
    with pytest.raises(ValueError):
        classifier.fit_puiseux(
            X, G1_FREQUENCY, order=2, exponent=THIRD, bounds=(0.25, 0.45)
        )
