"""10G2a probe 1: the centre-manifold covariance correction.

The fixture is a two-mode toy with an exact solution. ``J = diag(-1, -4 - 6x1)``
makes the critical mode the first coordinate (``left = e0``), so the stable basis
is ``e1`` and every correction has a closed form:

* ``Σ_s = D11 / 8``, ``Σ_full = Σ_s e1e1ᵀ``,
* ``B(e1, e1) = -6 e1``, ``drift_correction = -3 Σ_s e1``,
* ``stable_mean_shift = -Σ_s·3/4 e1`` (from ``-J_s⁻¹`` times the projected drift),
* ``critical_drift = 0`` (``left`` sees no stable component) and ``critical_shift = 0``.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.center_manifold import (
    CenterManifoldConfig,
    center_manifold_correction,
)
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


class _Adapter:
    """Two-mode toy: ``J = diag(-1, -4 - 6x1)``, state matrix is the diagonal."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        values = np.asarray(vector, dtype=float).reshape(-1)
        return np.diag(values.astype(complex))

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[1])
        return np.asarray([[-1.0, 0.0], [0.0, -4.0 - 6.0 * value]])


class _Model:
    name = "center_toy"
    n_modes = 2
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.eye(np.asarray(matrix).shape[-1], dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


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
        ],
    )


def test_the_two_mode_toy_matches_the_closed_forms():
    assessment = center_manifold_correction(
        _Adapter(),
        _Model(),
        _observable(),
        state=[0.0, 0.5],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason is None
    noise = np.asarray(assessment.stable_noise, dtype=float)
    covariance = np.asarray(assessment.stable_covariance, dtype=float)
    # The stable basis is the second coordinate, whose noise is D11 = 1.
    np.testing.assert_allclose(noise, [[1.0]], atol=1e-15)
    np.testing.assert_allclose(
        np.asarray(assessment.stable_jacobian), [[-7.0]], atol=1e-14
    )
    np.testing.assert_allclose(covariance, [[1.0 / 14.0]], rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(assessment.full_stable_covariance),
        [[0.0, 0.0], [0.0, 1.0 / 14.0]],
        atol=1e-15,
    )
    correction = np.asarray(assessment.drift_correction, dtype=float)
    np.testing.assert_allclose(correction, [0.0, -3.0 / 14.0], rtol=1e-12)
    np.testing.assert_allclose(
        np.asarray(assessment.stable_mean_shift), [0.0, -3.0 / 98.0], rtol=1e-12
    )
    assert assessment.critical_drift_correction == pytest.approx(0.0, abs=1e-15)
    assert assessment.critical_linear_shift == pytest.approx(0.0, abs=1e-15)
    assert assessment.lyapunov_relative_residual < 1e-14
    # The readout is quadratic along the stable direction, so the curvature term is
    # exact and step independent: 1/2 Σ_s H_s with H_s = 1.
    assert assessment.readout_hessian_correction == pytest.approx(1.0 / 28.0, rel=1e-9)
    record = assessment.readout_derivative
    assert record is not None
    assert record.method == "central_difference"
    assert record.repeat_relative_error is not None
    assert record.repeat_relative_error < 1e-6
    assert len(record.gradient) == 3
    assert record.metadata["quantity"] == "readout_curvature"


def test_a_negative_noise_covariance_is_reported_not_used():
    assessment = center_manifold_correction(
        _Adapter(),
        _Model(),
        _observable(),
        state=[0.0, 0.5],
        params={},
        diffusion=-np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.reason_code == "noise_not_psd"
    assert np.isnan(assessment.critical_drift_correction)
    assert assessment.drift_correction == ()


def test_a_non_finite_state_is_not_converged():
    assessment = center_manifold_correction(
        _Adapter(),
        _Model(),
        _observable(),
        state=[np.nan, 0.5],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.NOT_CONVERGED
    assert assessment.reason_code == "state_not_finite"


def test_an_unresolved_mode_is_unidentifiable():
    class Rotating(_Adapter):
        def jacobian(self, vector, params=None):
            del vector, params
            return np.asarray([[-0.5, -1.0], [1.0, -0.5]])

    assessment = center_manifold_correction(
        Rotating(),
        _Model(),
        _observable(),
        state=[0.0, 0.5],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert assessment.reason_code == "mode_unresolved"


def test_a_model_without_a_diffusion_is_a_configuration_error():
    class NoDiffusion:
        params = {"mu": 0.0}

    with pytest.raises(ValueError, match="cam_diffusion"):
        center_manifold_correction(
            _Adapter(),
            NoDiffusion(),
            _observable(),
            state=[0.0, 0.5],
            params={},
        )


def test_the_readout_curvature_can_be_aggregated_from_several_steps():
    assessment = center_manifold_correction(
        _Adapter(),
        _Model(),
        _observable(),
        state=[0.0, 0.5],
        params={},
        diffusion=np.eye(2),
        config=CenterManifoldConfig(hessian_step_factors=(1e-2, 1e-3)),
    )
    assert len(assessment.readout_derivative.gradient) == 2
    assert assessment.readout_hessian_correction == pytest.approx(1.0 / 28.0, rel=1e-8)
