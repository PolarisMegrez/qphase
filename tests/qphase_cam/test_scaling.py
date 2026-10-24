"""Scaling-signature classifier and scaling-fit numeric tests.

The frozen numbers in the second half come from running the historical evaluation
scripts (``scripts/tools/evaluate_*puiseux*`` and ``evaluate_*two_stage*``) on the
synthetic inputs below, so the extracted :mod:`qphase_cam.solver.scaling` layer can
be checked against the code it replaces without importing those scripts.
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

#: Bookkeeping quantities round-trip at machine precision; least-squares
#: objectives that sit near a cancellation are only reliable to a few ulp.
EXACT = {"rel": 1.0e-9, "abs": 0.0}
OBJECTIVE = {"rel": 1.0e-6, "abs": 0.0}
#: Profile-search resolution on the two-stage objective (401-point grid).
PROFILE = {"rel": 1.0e-8, "abs": 0.0}


def test_free_exponent_fit_with_anchor_reproduces_cam_reference():
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
    assert fit.exponent_sem == pytest.approx(3.9376550743893484e-12, **OBJECTIVE)
    assert fit.at_exponent_bound is False
    assert fit.offset == pytest.approx(CRITICAL, **EXACT)
    assert fit.anchor == CRITICAL
    assert fit.coefficients == pytest.approx(
        (1.1603972083531455, -0.17235477515035624), **EXACT
    )
    assert fit.leading_coefficient == pytest.approx(1.1603972083531455, **EXACT)
    assert fit.rmse == pytest.approx(2.2654005537415463e-12, **OBJECTIVE)
    assert fit.max_abs_residual == pytest.approx(3.0189184485607257e-12, **OBJECTIVE)
    assert fit.loo_rmse == pytest.approx(6.2441621539073665e-12, **OBJECTIVE)
    assert fit.aicc == pytest.approx(-471.8386348074195, **OBJECTIVE)
    assert fit.weighted_reduced_chi2 == pytest.approx(
        7.698059503338756e-20, **OBJECTIVE
    )
    assert fit.condition == pytest.approx(33.20959999861707, **EXACT)
    assert fit.profile_points == 401


def test_free_exponent_fit_with_free_offset_reproduces_weighted_reference():
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
    assert fit.exponent_sem == pytest.approx(5.240833019496452e-12, **OBJECTIVE)
    assert fit.offset == pytest.approx(0.7499999999919966, **EXACT)
    assert fit.coefficients == pytest.approx(
        (1.1603972083777778, -0.17235477516852948), **EXACT
    )
    assert fit.rmse == pytest.approx(7.241803901220422e-13, **OBJECTIVE)
    assert fit.loo_rmse == pytest.approx(2.5691103583782195e-12, **OBJECTIVE)
    assert fit.aicc == pytest.approx(-402.2741824189984, **OBJECTIVE)
    assert fit.condition == pytest.approx(90.15703475465155, **EXACT)
    assert fit.weighted is True


def test_fixed_exponent_fit_reproduces_leverage_loo_reference():
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
    assert fit.rmse == pytest.approx(3.1619165601783647e-16, **OBJECTIVE)
    assert fit.loo_rmse == pytest.approx(4.436478767723345e-16, **OBJECTIVE)
    assert fit.aicc == pytest.approx(-631.6232964798783, **OBJECTIVE)
    assert fit.condition == pytest.approx(22.57297189010064, **EXACT)
    assert fit.metadata["loo_method"] == "leverage"


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
    assert fit.rmse == pytest.approx(0.16057662565209904, **OBJECTIVE)
    assert fit.max_abs_residual == pytest.approx(0.2877735439972441, **OBJECTIVE)
    assert fit.loo_rmse == pytest.approx(0.29249630186206493, **OBJECTIVE)
    assert fit.aicc == pytest.approx(-26.921712575042903, **OBJECTIVE)
    assert fit.weighted_reduced_chi2 == pytest.approx(331.51953478904164, **OBJECTIVE)
    assert fit.condition == pytest.approx(6.292330236666059, **EXACT)


def test_allan_relative_fit_reproduces_reference():
    fit = allan_relative_fit(X, ALLAN_INTENSITY, order=2, bounds=(0.25, 0.45))
    assert fit.model == "allan_relative"
    assert fit.order == 2
    assert fit.parameter_count == 4
    assert fit.exponent == pytest.approx(0.33333333335783366, **EXACT)
    assert fit.exponent_sem == pytest.approx(1.0956875544733504e-11, **OBJECTIVE)
    assert fit.metadata["leading_power"] == pytest.approx(-1.3333333332843327, **EXACT)
    assert fit.coefficients == pytest.approx(
        (21.54434690901204, 4.99999998368522, 1.1603972162620213), **EXACT
    )
    assert fit.relative_rmse == pytest.approx(1.079796797616877e-11, **OBJECTIVE)
    assert np.max(np.abs(fit.residual / ALLAN_INTENSITY)) == pytest.approx(
        1.8762221429906396e-11, **OBJECTIVE
    )
    assert fit.loo_relative_rmse == pytest.approx(2.2182698478151948e-10, **OBJECTIVE)
    assert fit.loo_rmse is None
    assert fit.weighted_reduced_chi2 is None
    assert fit.aicc == pytest.approx(-436.5299366974011, **OBJECTIVE)
    assert fit.bic == pytest.approx(-445.74103838805627, **OBJECTIVE)
    assert fit.condition == pytest.approx(86.77341857057084, **EXACT)
    assert fit.at_exponent_bound is False


def test_allan_window_diagnostics_reproduce_reference():
    diagnostics = allan_window_diagnostics(
        X,
        ALLAN_INTENSITY,
        order=2,
        bounds=(0.25, 0.45),
        window_maxima=(1.0e-2, 10.0**-1.5, 1.0e-1),
        inner_max=1.0e-2,
    )
    assert diagnostics.window_exponents[0] == pytest.approx(0.3333333333617728, **EXACT)
    assert diagnostics.window_exponents[1] == pytest.approx(
        0.33333333336056764, **EXACT
    )
    assert diagnostics.span == pytest.approx(3.939126802521287e-12, **OBJECTIVE)
    assert diagnostics.outer_extrapolation_relative_rmse == pytest.approx(
        2.628083850090857e-10, **OBJECTIVE
    )
    assert diagnostics.outer_extrapolation_max_abs_relative_error == pytest.approx(
        4.2278119264947655e-10, **OBJECTIVE
    )


def test_two_stage_frequency_fit_reproduces_reference():
    fit = two_stage_frequency_fit(
        X, G1_FREQUENCY, lower=1.0e-4, upper=1.0e-1, exponent_bounds=(0.25, 0.45)
    )
    assert fit.stage1_exponent == pytest.approx(THIRD, **EXACT)
    assert fit.stage1_point_count == 9
    assert fit.stage1_offset == pytest.approx(0.7813297950191174, **EXACT)
    assert fit.stage1_amplitude == pytest.approx(0.9782777876699669, **EXACT)
    assert fit.stage1_rmse == pytest.approx(0.012722230022209065, **OBJECTIVE)
    assert fit.stage2_exponent == pytest.approx(0.3269426390146774, **PROFILE)
    assert fit.stage2_amplitude == pytest.approx(0.971603457009521, **PROFILE)
    assert fit.stage2_full_range_rmse == pytest.approx(
        0.011948045472374755, **OBJECTIVE
    )
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
        np.sqrt(np.mean(fit.stage2_residual**2)), **OBJECTIVE
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
