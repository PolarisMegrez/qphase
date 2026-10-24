"""10G2a probe 2: the local effective-drift correction.

Real two-mode / four-canonical-coordinate toy built through the public
``vector_to_matrix(x, 2)`` and ``canonical_sample_matrix_covariance`` paths:

* state ``[1, 0.5, 0, 0]`` (``R00 = 1``, ``R11 = 0.5``),
  ``J = diag(-1, -4 - 6x1, -3, -5)`` so the critical mode is the first coordinate
  and the stable basis is ``e1, e2, e3``;
* ``cam_diffusion`` returns ``diag(q/(2 R00), 1/R00)`` with ``q = 1 + 4 R11²``, so
  the canonical covariance is verified numerically below as
  ``diag(2, 1, 0.75, 0.75)`` at the reference, giving ``Σ_s[0,0] = 1/14``;
* the projected noise is ``q(x) = 1 + 4x1²``, so its stable curvature is 8 and the
  diffusion correction is ``4 Σ_s = 2/7``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.coordinates import vector_to_matrix
from qphase_cam.core.noise import canonical_sample_matrix_covariance
from qphase_cam.effective_drift import effective_drift_correction
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus

IDENTITY = ObservableIdentity(
    kind=ObservableKind.STATE,
    sector=ChargeSector.CHARGE_ZERO,
    name="x",
    measurement_mapping="x0 + x1^2 / 2",
)
STATE = np.asarray([1.0, 0.5, 0.0, 0.0])


class _Adapter:
    """Four canonical coordinates through ``vector_to_matrix(·, 2)``."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector_to_matrix(np.asarray(vector, dtype=float), 2))

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[1])
        return np.diag([-1.0, -4.0 - 6.0 * value, -3.0, -5.0])


class _SymmetricAdapter:
    """Symmetric 2D toy: ``J = [[-1, -600 x1], [-600 x1, -4]]``.

    The matrix is symmetric, so the critical left/right eigenvector stays ``e0``
    while ``B(e1,e1) = (∂_1 J)e1 = -600 e0``; the stable covariance is
    ``D11/(2·4) = 1/8``.
    """

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.diag(np.asarray(vector, dtype=float).reshape(-1).astype(complex))

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[1])
        return np.asarray([[-1.0, -600.0 * value], [-600.0 * value, -4.0]])


class _Model:
    """``D = diag(q/(2 R00), 1/R00)`` in the matrix basis."""

    name = "effective_toy"
    n_modes = 2
    params = {"mu": 0.0}
    scale = 1.0

    @classmethod
    def cam_diffusion(cls, matrix, params=None):
        del params
        array = np.asarray(matrix)
        r00 = (
            float(np.real(array.reshape(-1)[0]))
            if array.ndim
            else float(np.real(array))
        )
        r11 = float(np.real(np.asarray(array)[..., 1, 1])) if array.ndim >= 2 else r00
        q = 1.0 + 4.0 * r11**2
        return cls.scale * np.asarray(
            [[q / (2.0 * r00), 0.0], [0.0, 1.0 / r00]], dtype=complex
        )

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


class _ScaledModel(_Model):
    scale = 100.0


def _observable():
    return FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0]
        )
        + 0.5 * float(np.asarray(state, dtype=float).reshape(-1)[1]) ** 2,
        gradient_fn=lambda adapter, state, params: [
            1.0,
            float(np.asarray(state, dtype=float).reshape(-1)[1]),
            0.0,
            0.0,
        ],
    )


def test_the_real_canonical_covariance_is_the_expected_diagonal():
    matrix = np.asarray(vector_to_matrix(STATE, 2))
    r00, r11 = float(np.real(matrix[0, 0])), float(np.real(matrix[1, 1]))
    q = 1.0 + 4.0 * r11**2
    noise = np.asarray(np.diag([q / (2.0 * r00), 1.0 / r00]), dtype=complex)
    covariance = canonical_sample_matrix_covariance(matrix, noise)
    np.testing.assert_allclose(np.diag(covariance), [2.0, 1.0, 0.75, 0.75], atol=1e-12)


def test_the_state_dependent_diffusion_correction_matches_the_closed_form():
    assessment = effective_drift_correction(
        _Adapter(), _Model(), _observable(), state=STATE, params={}
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    covariance = np.asarray(assessment.center.stable_covariance, dtype=float)
    assert covariance[0, 0] == pytest.approx(1.0 / 14.0, rel=1e-10)
    # q = 1 + 4x1² has stable curvature 8, so the correction is 4 Σ_s = 2/7.
    assert assessment.effective_diffusion_correction == pytest.approx(
        2.0 / 7.0, rel=1e-6
    )
    record = assessment.diffusion_derivative
    assert record is not None
    assert len(record.gradient) == 3
    np.testing.assert_allclose(record.gradient, 2.0 / 7.0, rtol=1e-6)
    assert record.repeat_relative_error is not None
    assert record.repeat_relative_error < 1e-6
    assert record.metadata["quantity"] == "effective_diffusion_correction"


def test_the_drift_is_the_centre_correction_and_the_readout_cross_checks():
    assessment = effective_drift_correction(
        _Adapter(), _Model(), _observable(), state=STATE, params={}
    )
    np.testing.assert_allclose(
        np.asarray(assessment.effective_drift_correction, dtype=float),
        [0.0, -3.0 / 14.0, 0.0, 0.0],
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        np.asarray(assessment.effective_drift_correction, dtype=float),
        np.asarray(assessment.center.drift_correction, dtype=float),
        atol=0.0,
    )
    assert assessment.drift_derivative.method == "analytic"
    assert "identical estimator" in assessment.drift_derivative.metadata["source"]
    assert assessment.effective_readout_correction == pytest.approx(
        1.0 / 28.0, rel=1e-6
    )
    assert assessment.effective_readout_correction == pytest.approx(
        assessment.center.readout_hessian_correction, rel=1e-6
    )


def test_a_negative_noise_covariance_stops_before_any_correction():
    assessment = effective_drift_correction(
        _Adapter(),
        _Model(),
        _observable(),
        state=STATE,
        params={},
        diffusion=-np.eye(4),
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "noise_not_psd"
    assert assessment.effective_drift_correction == ()
    assert assessment.effective_diffusion_correction is None
    assert assessment.effective_readout_correction is None
    assert assessment.diffusion_derivative is None
    assert assessment.correction_dominates_linear is None


def test_a_single_mode_has_no_stable_basis():
    class ScalarAdapter:
        moment_layout = "normal"

        def state_matrix(self, vector, params=None):
            del params
            return np.asarray(
                [[float(np.asarray(vector).reshape(-1)[0])]], dtype=complex
            )

        def jacobian(self, vector, params=None):
            del vector, params
            return np.asarray([[-1.0]])

    assessment = effective_drift_correction(
        ScalarAdapter(),
        _Model(),
        _observable(),
        state=[1.0],
        params={},
        diffusion=[[1.0]],
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert assessment.reason_code in {"stable_basis_degenerate", "mode_unresolved"}
    assert assessment.effective_drift_correction == ()
    assert assessment.effective_diffusion_correction is None
    assert assessment.effective_readout_correction is None


def _two_coordinate_observable():
    return FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0]
        )
        + 0.5 * float(np.asarray(state, dtype=float).reshape(-1)[1]) ** 2,
        gradient_fn=lambda adapter, state, params: [
            1.0,
            float(np.asarray(state, dtype=float).reshape(-1)[1]),
        ],
    )


def test_the_dominance_ratio_follows_the_centre_probe_definition():
    assessment = effective_drift_correction(
        _SymmetricAdapter(),
        _Model(),
        _two_coordinate_observable(),
        state=[0.0, 0.0],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    np.testing.assert_allclose(
        np.asarray(assessment.center.left, dtype=float), [1.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(assessment.center.drift_correction, dtype=float),
        [-37.5, 0.0],
        rtol=1e-10,
        atol=1e-12,
    )
    assert assessment.critical_rate == pytest.approx(1.0, rel=1e-12)
    assert assessment.projected_noise == pytest.approx(1.0, rel=1e-12)
    assert assessment.sigma_x == pytest.approx(np.sqrt(0.5), rel=1e-12)
    assert assessment.dominance_ratio == pytest.approx(75.0 / np.sqrt(2.0), rel=1e-10)
    assert assessment.correction_dominates_linear is True


def test_an_invalid_step_factor_is_rejected():
    from pydantic import ValidationError
    from qphase_cam.effective_drift import EffectiveCorrectionConfig

    for factors in ((0.0,), (-1e-3,), (1e-3, float("nan")), (float("inf"),)):
        with pytest.raises(ValidationError):
            EffectiveCorrectionConfig(step_factors=factors)


def test_a_non_finite_readout_fails_instead_of_reporting_nan():
    def value(adapter, state, params):
        # Finite at the reference point, non-finite at every perturbed one, so the
        # failure comes from the second average rather than from the centre value.
        point = np.asarray(state, dtype=float).reshape(-1)
        if float(np.linalg.norm(point)) > 0.0:
            return float("nan")
        return 0.0

    broken = FunctionalObservable(
        identity=IDENTITY,
        value_fn=value,
        gradient_fn=lambda adapter, state, params: [
            1.0,
            float(np.asarray(state, dtype=float).reshape(-1)[1]),
        ],
    )
    assessment = effective_drift_correction(
        _SymmetricAdapter(),
        _Model(),
        broken,
        state=[0.0, 0.0],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "correction_not_finite"
    assert assessment.effective_drift_correction == ()
    assert assessment.effective_diffusion_correction is None
    assert assessment.effective_readout_correction is None
    assert assessment.diffusion_derivative is None
    assert assessment.readout_derivative is None
