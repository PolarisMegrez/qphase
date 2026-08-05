"""Scaling-signature classifier tests on canonical normal forms."""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.reduction import LocalScalingSeries
from qphase_cam.solver.bifurcation_classifier import (
    ScalingSignatureClassifier,
    ScalingSignatureConfig,
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
