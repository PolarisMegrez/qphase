"""10G2a probe 2: the local effective-drift correction and the full curve.

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

The full-curve tests use a second toy on the same coordinates, chosen so the
reconstruction it feeds is ``valid`` and both of its boundaries are physical:

* ``_CurveAdapter`` keeps ``J = diag(-1, -3 - 3x1, -4, -5)`` and adds the matching
  ``rhs`` (the quadratic in ``x1`` is what makes ``B(e1, e1) = -3``) plus an
  identity ``parameter_jacobian``;
* ``_CurveModel`` returns ``diag(½(1 - R00/2)(1 + 0.2 R11²), 0.05)``, so the
  projected noise is ``q = R00 (1 - R00/2)(1 + 0.2 R11²)``: it vanishes at
  ``R00 = 0`` and ``R00 = 2``, i.e. at ``x = ±1``, which is why the caller-supplied
  confining manifold gets two physical boundaries. Its dependence on ``R11`` is
  what gives the noise a stable curvature, so the eigendirection average of the
  curve is not identically zero.

The source is reconstructed with ``grid_points = CURVE_SIDE = 11`` and the curve with
``coarse_points_per_side = 11``: the curve counts points per side, so its coarse grid
is the reconstruction's own 21-point grid and the two are compared point for point.

The postprocessor tests drive the same toy through the public annotation path: the
real ``NonlinearReconstruction`` plugin, a real ``build_state_product`` state product
and the real ``process`` entry point, with only ``QPSGenDynamicsAdapter.from_model``
patched to return ``_CurveAdapter``. The plugin resolves its own manifold with
``slow_manifold``, so the row it produces is an uncontrolled approximation: that is
the diagnostic curve the reference probe reported, and it is compared field by field
against a curve computed directly along the same path.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError
from qphase.data import load_bundle
from qphase_cam.core.coordinates import vector_to_matrix
from qphase_cam.core.noise import canonical_sample_matrix_covariance
from qphase_cam.data import build_state_product, cam_bundle
from qphase_cam.effective_drift import (
    EffectiveDriftConfig,
    effective_drift_correction,
    effective_drift_curve,
)
from qphase_cam.nonlinear import (
    NonlinearManifold,
    NonlinearManifoldConfig,
    ScalarReconstructionConfig,
    scalar_reconstruction,
    slow_manifold,
)
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.nonlinear_reconstruction import (
    NonlinearReconstruction,
)

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


# --- the full curve -------------------------------------------------------

PARAMETER_DIRECTION = np.asarray([1.0, 0.0, 0.0, 0.0])


class _CurveAdapter(_Adapter):
    """``_Adapter``'s canonical state matrix with a matching ``J`` and ``rhs``.

    ``J = diag(-1, -3 - 3x1, -4, -5)`` keeps the critical mode on the first
    coordinate, and ``rhs`` is the antiderivative of that Jacobian, so
    ``B(e1, e1) = (∂_1 J) e1 = -3 e1`` and the stable mean has the closed form used
    below.
    """

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[1])
        return np.diag([-1.0, -3.0 - 3.0 * value, -4.0, -5.0])

    def rhs(self, vector, params=None):
        del params
        value = np.asarray(vector, dtype=float).reshape(-1)
        return np.asarray(
            [
                -value[0],
                -3.0 * value[1] - 1.5 * value[1] ** 2,
                -4.0 * value[2],
                -5.0 * value[3],
            ]
        )

    def parameter_jacobian(self, vector, params=None):
        del vector, params
        return np.eye(4)


class _UnstableCurveAdapter(_CurveAdapter):
    """The same toy with a stable direction that does not decay."""

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[1])
        return np.diag([-1.0, 3.0 + 3.0 * value, -4.0, -5.0])


class _BandCurveAdapter(_CurveAdapter):
    """A stable direction that only decays away from the critical point.

    ``J11 = 4 - 8x²`` is positive for ``|x| < 1/√2`` and negative beyond it, so the
    coarse grid keeps its two ends and drops the middle: the stable interval is real
    but disconnected.
    """

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.diag([-1.0, 4.0 - 8.0 * (value - 1.0) ** 2, -4.0, -5.0])


class _CurveModel:
    """``D = diag(½(1 - R00/2)(1 + 0.2 R11²), 0.05)`` in the matrix basis."""

    name = "effective_curve_toy"
    n_modes = 2
    params = {"mu": 0.0}

    @classmethod
    def cam_diffusion(cls, matrix, params=None):
        del params
        array = np.asarray(matrix)
        r00 = float(np.real(array[..., 0, 0]))
        r11 = float(np.real(array[..., 1, 1]))
        diffusion = [[0.5 * (1.0 - r00 / 2.0) * (1.0 + 0.2 * r11**2), 0.0], [0.0, 0.05]]
        return np.asarray(diffusion, dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


class _ZeroDiffusionModel(_CurveModel):
    """A model whose diffusion vanishes everywhere: no projected noise at all."""

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.zeros(np.asarray(matrix).shape, dtype=complex)


def _zero_readout() -> FunctionalObservable:
    """Return a readout that is identically zero, so the intensity is exactly zero."""
    return FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: 0.0,
        gradient_fn=lambda adapter, state, params: [0.0, 0.0, 0.0, 0.0],
    )


def _curve_manifold() -> NonlinearManifold:
    """Return a confining manifold whose embedding is exactly ``R00 = 1 + x``."""
    return NonlinearManifold(
        order=3,
        soft_rate=1.0,
        critical_eigenvalue_real=-1.0,
        critical_eigenvalue_imag=0.0,
        noncritical_spectral_gap=2.0,
        right=(1.0, 0.0, 0.0, 0.0),
        left=(1.0, 0.0, 0.0, 0.0),
        embeddings=((1.0, 0.0, 0.0, 0.0),),
        drift=(0.0, -1.0, 0.0, -0.5),
        maximum_bordered_condition=1.0,
        confining=True,
        method={"service": "test fixture"},
    )


CURVE_SIDE = 11


def _curve_reconstruction(*, grid_points: int = CURVE_SIDE) -> Any:
    reconstruction = scalar_reconstruction(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        manifold=_curve_manifold(),
        parameter_direction=PARAMETER_DIRECTION,
        config=ScalarReconstructionConfig(grid_points=grid_points),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    assert reconstruction.grid_points == 2 * grid_points - 1
    return reconstruction


def _curve_config(**updates: Any) -> EffectiveDriftConfig:
    settings: dict[str, Any] = {
        "coarse_points_per_side": CURVE_SIDE,
        "dense_points": 41,
        "derivative_step_factor": 1.0e-3,
    }
    settings.update(updates)
    return EffectiveDriftConfig(**settings)


def _curve_centre_forms() -> tuple[float, float, float]:
    """Return ``(q, omega, trace Σ_s)`` of the curve at the reference point.

    With ``R00 = 1`` and ``R11 = 0.5`` the canonical covariance of the toy is
    ``diag(2 D00 R00, 2 D11 R11, ½·2(D00 R11 + D11 R00), ·)``, the stable basis is
    ``e1, e2, e3`` and the stable Jacobian is ``diag(-3 - 3R11, -4, -5)``, so the
    Lyapunov covariance is diagonal and known in closed form. ``B(e1, e1) = -3``
    shifts ``R11`` by ``½(-3)Σ11/4.5`` along the stable mean, ``q = R00(1 - R00/2)
    (1 + 0.2 R11²)`` has stable curvature ``0.4 R00 (1 - R00/2)`` and the readout
    ``R00 + R11²/2`` has curvature ``1``.
    """
    diffusion_00 = 0.5 * (1.0 - 0.5) * (1.0 + 0.2 * 0.5**2)
    diffusion_11 = 0.05
    noise_11 = 2.0 * diffusion_11 * 0.5
    noise_22 = 0.5 * (diffusion_00 * 0.5 + diffusion_11 * 1.0)
    covariance_11 = noise_11 / (2.0 * 4.5)
    covariance_22 = noise_22 / (2.0 * 4.0)
    covariance_33 = noise_22 / (2.0 * 5.0)
    shift = 0.5 * (-3.0) * covariance_11 / 4.5
    q_centre = 0.5 * (1.0 + 0.2 * (0.5 + shift) ** 2) + 0.5 * covariance_11 * 0.2
    omega_centre = 1.0 + 0.5 * (0.5 + shift) ** 2 + 0.5 * covariance_11
    return q_centre, omega_centre, covariance_11 + covariance_22 + covariance_33


def test_the_curve_interpolates_the_interval_and_evaluates_the_generator():
    reconstruction = _curve_reconstruction()
    curve = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=PARAMETER_DIRECTION,
        config=_curve_config(),
    )
    assert curve.status is CAMOutcomeStatus.VALID
    assert curve.reason is None and curve.reason_code is None
    assert curve.source_status is CAMOutcomeStatus.VALID
    assert curve.manifold == reconstruction.manifold

    # The coarse grid is the reconstruction's own grid: one point per side counting the
    # zero, deduplicated in the middle, so it holds 2 * side - 1 points.
    x_coarse = np.asarray(curve.coarse["x"], dtype=float)
    zero = int(np.argmin(np.abs(x_coarse)))
    assert x_coarse.size == 2 * CURVE_SIDE - 1 == 21
    assert x_coarse.size == reconstruction.grid_points
    assert x_coarse[0] == reconstruction.x_min
    assert x_coarse[-1] == reconstruction.x_max
    assert x_coarse[zero] == 0.0
    assert zero == CURVE_SIDE - 1
    # The left side is one point per side from the boundary to zero, the right side the
    # mirror, so the spacing is uniform on each half but not across zero.
    np.testing.assert_allclose(
        np.diff(x_coarse[: zero + 1]),
        reconstruction.x_max / (CURVE_SIDE - 1),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.diff(x_coarse[zero:]),
        reconstruction.x_max / (CURVE_SIDE - 1),
        atol=1e-12,
    )
    for name in (
        "f_deterministic",
        "f_effective",
        "q_effective",
        "omega_effective",
        "forcing",
        "stable_covariance_trace",
        "stable_margin",
        "physical_minimum",
        "lyapunov_residual",
        "valid",
    ):
        assert np.asarray(curve.coarse[name]).size == 21, name
    assert np.all(np.asarray(curve.coarse["valid"], dtype=bool))
    # The curve re-uses the reconstruction's own grid point for point, not just its
    # boundaries: the source exposes it as its generator table.
    source_grid = reconstruction.generator_table
    assert source_grid is not None
    np.testing.assert_allclose(x_coarse, source_grid.x, atol=0.0)
    # The readout is along the critical coordinate, so the drift correction has no
    # critical component and the curve keeps the manifold's own drift.
    np.testing.assert_allclose(
        curve.coarse["f_effective"], curve.coarse["f_deterministic"], atol=0.0
    )
    assert float(curve.coarse["f_effective"][zero]) == 0.0
    assert float(curve.coarse["stable_margin"][zero]) == pytest.approx(4.0, rel=1e-12)
    assert float(curve.coarse["physical_minimum"][zero]) == pytest.approx(
        0.5, rel=1e-12
    )
    assert np.all(np.asarray(curve.coarse["q_effective"], dtype=float) > 0.0)

    # Centre-point closed form (see ``_curve_centre_forms``).
    q_centre, omega_centre, trace = _curve_centre_forms()
    assert float(curve.coarse["stable_covariance_trace"][zero]) == pytest.approx(
        trace, rel=1e-12
    )
    assert float(curve.coarse["q_effective"][zero]) == pytest.approx(q_centre, rel=1e-9)
    assert float(curve.coarse["omega_effective"][zero]) == pytest.approx(
        omega_centre, rel=1e-9
    )
    np.testing.assert_allclose(curve.coarse["forcing"], 1.0, atol=0.0)

    # The dense grid is the PCHIP interpolation of the coarse interval, and the
    # exported table is that interpolation, not a second sampling.
    assert curve.generator is not None
    x_dense = np.asarray(curve.dense["x"], dtype=float)
    assert x_dense.size == 41
    assert x_dense[0] == x_coarse[0] and x_dense[-1] == x_coarse[-1]
    assert np.all(np.diff(x_dense) > 0.0)
    np.testing.assert_allclose(curve.generator.x, curve.dense["x"], atol=0.0)
    np.testing.assert_allclose(curve.generator.drift, curve.dense["f_effective"])
    np.testing.assert_allclose(curve.generator.diffusion, curve.dense["q_effective"])
    np.testing.assert_allclose(curve.generator.readout, curve.dense["omega_effective"])
    assert np.all(np.asarray(curve.dense["q_effective"], dtype=float) > 0.0)
    # Between two coarse points the interpolation is monotone, so the centre value
    # is reproduced exactly.
    dense_zero = int(np.argmin(np.abs(x_dense)))
    assert float(curve.dense["q_effective"][dense_zero]) == pytest.approx(
        q_centre, rel=1e-9
    )

    evaluation = curve.evaluation
    assert evaluation is not None
    assert evaluation.normalizable is True
    assert len(evaluation.density) == 41
    assert evaluation.core_point_count == curve.method["core_point_count"]
    assert evaluation.edge_point_count == curve.method["edge_point_count"]
    np.testing.assert_allclose(curve.dense["density"], evaluation.density, atol=0.0)

    summary = curve.summary
    for name in (
        "stationary_mean",
        "stationary_mean_x",
        "stationary_response",
        "allan_intensity",
        "information_rate",
        "path_information_rate",
        "information_over_path",
        "edge_mass_fraction",
        "endpoint_density_ratio",
        "minimum_q_effective",
        "minimum_stable_margin",
        "lyapunov_residual_max",
    ):
        assert summary[name] is not None and np.isfinite(summary[name]), name
    assert summary["allan_intensity"] == evaluation.scalar_intensity
    assert summary["stationary_response"] == evaluation.stationary_response
    assert summary["path_information_rate"] == evaluation.path_information_rate
    assert summary["information_rate"] == pytest.approx(
        summary["stationary_response"] ** 2 / summary["allan_intensity"], rel=1e-12
    )
    assert summary["information_over_path"] == pytest.approx(
        summary["information_rate"] / summary["path_information_rate"], rel=1e-12
    )
    assert summary["minimum_stable_margin"] == pytest.approx(4.0, rel=1e-12)
    assert summary["lyapunov_residual_max"] < 1.0e-10
    assert summary["coarse_interval"] == [reconstruction.x_min, reconstruction.x_max]
    assert summary["negative_boundary"] == reconstruction.x_min
    assert summary["positive_boundary"] == reconstruction.x_max
    assert summary["sigma_x"] == reconstruction.sigma
    assert summary["valid_coarse_count"] == 21
    assert summary["coarse_point_count"] == 21
    assert summary["dense_point_count"] == 41
    assert summary["negative_physical_boundary"] is True
    assert summary["positive_physical_boundary"] is True
    # The stable margin is 4 on the whole interval and the soft rate is 1, so the
    # mass below 3x the soft rate is empty and the mass below 5x is everything.
    assert summary["probability_stable_margin_le_1x_soft_rate"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert summary["probability_stable_margin_le_3x_soft_rate"] == pytest.approx(
        0.0, abs=1e-12
    )
    assert summary["probability_stable_margin_le_5x_soft_rate"] == pytest.approx(
        1.0, rel=1e-12
    )
    assert summary["probability_stable_margin_le_10x_soft_rate"] == pytest.approx(
        1.0, rel=1e-12
    )

    method = curve.method
    assert method["service"] == "qphase_cam.effective_drift.effective_drift_curve"
    assert method["integration_split"] == "nearest_zero"
    assert method["coarse_points_per_side"] == CURVE_SIDE
    assert method["dense_points"] == 41
    assert method["derivative_step_factor"] == pytest.approx(1.0e-3)
    assert method["parameter_direction"] is True
    assert method["physical_tolerance"] == pytest.approx(1.0e-9)
    assert method["stable_covariance_tolerance"] == pytest.approx(1.0e-7)
    assert method["density_normalization_raw"] == evaluation.density_normalization
    assert method["source_valid"] is True
    assert method["source_status"] == CAMOutcomeStatus.VALID.value


def test_a_curve_without_a_parameter_direction_reports_no_forcing_numbers():
    reconstruction = _curve_reconstruction()
    # The chart defaults are the historical ones, so this is also the widest-step
    # path: the single 3e-7 step resolves the quadratic readout to about 1e-4.
    curve = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=EffectiveDriftConfig(coarse_points_per_side=CURVE_SIDE, dense_points=41),
    )
    assert curve.status is CAMOutcomeStatus.VALID
    assert curve.method["parameter_direction"] is False
    assert curve.method["derivative_step_factor"] == pytest.approx(3.0e-7)
    q_centre, omega_centre, _ = _curve_centre_forms()
    zero = int(np.argmin(np.abs(np.asarray(curve.coarse["x"], dtype=float))))
    assert float(curve.coarse["q_effective"][zero]) == pytest.approx(q_centre, rel=1e-4)
    assert float(curve.coarse["omega_effective"][zero]) == pytest.approx(
        omega_centre, rel=1e-4
    )
    # No forcing is not a zero forcing: the per-point array keeps its NaN and the
    # two forced numbers stay unavailable instead of being faked.
    assert np.all(np.isnan(np.asarray(curve.coarse["forcing"], dtype=float)))
    assert curve.dense["forcing"] is None
    assert curve.summary["stationary_response"] is None
    assert curve.summary["path_information_rate"] is None
    assert curve.summary["information_rate"] is None
    assert curve.summary["information_over_path"] is None
    assert curve.summary["stationary_mean"] is not None
    assert curve.summary["allan_intensity"] is not None


def test_an_uncontrolled_source_still_yields_a_diagnostic_curve():
    """A non-valid source with every field kept is evaluated and stays non-valid."""
    reconstruction = _curve_reconstruction()
    uncontrolled = reconstruction.model_copy(
        update={
            "status": CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION,
            "reason": "the truncated normal form is not confining",
            "reason_code": "manifold_not_confining",
            "validity_domain": "not_confining",
        }
    )
    curve = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=uncontrolled,
        parameter_direction=PARAMETER_DIRECTION,
        config=_curve_config(),
    )
    # The arrays are the diagnostic the reference probe reported...
    assert curve.generator is not None
    assert curve.evaluation is not None
    assert np.asarray(curve.coarse["x"]).size == 2 * CURVE_SIDE - 1
    assert np.asarray(curve.dense["x"]).size == 41
    assert curve.summary["allan_intensity"] is not None
    # ... and the verdict is the source's, not a fresh one.
    assert curve.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert curve.source_status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert curve.reason_code == "manifold_not_confining"
    assert "not confining" in curve.reason
    assert "not valid" in curve.reason
    assert curve.manifold == reconstruction.manifold
    assert curve.method["source_valid"] is False
    assert curve.method["source_reason_code"] == "manifold_not_confining"
    assert curve.method["source_validity_domain"] == "not_confining"
    # An internal failure still wins over the source's verdict, and the source status is
    # kept separately so the two are never confused.
    stale = effective_drift_curve(
        _UnstableCurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=uncontrolled,
        config=_curve_config(),
    )
    assert stale.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert stale.reason_code == "no_stable_interval"
    assert stale.source_status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION


def test_a_source_without_boundaries_is_not_turned_into_a_curve():
    """A source that never reached its grid reports its own verdict, not a curve."""
    reconstruction = scalar_reconstruction(
        _CurveAdapter(),
        _ZeroDiffusionModel(),
        _observable(),
        state=STATE,
        params={},
        manifold=_curve_manifold(),
        config=ScalarReconstructionConfig(grid_points=CURVE_SIDE),
    )
    # The failure stops before the boundaries, so there is nothing to place.
    assert reconstruction.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert reconstruction.x_min is None and reconstruction.x_max is None
    assert reconstruction.sigma is None
    assert reconstruction.grid_points is None
    curve = effective_drift_curve(
        _CurveAdapter(),
        _ZeroDiffusionModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=_curve_config(),
    )
    assert curve.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert curve.source_status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert curve.reason_code == "source_reconstruction_incomplete"
    assert reconstruction.reason in curve.reason
    assert curve.generator is None
    assert curve.evaluation is None
    assert curve.coarse == {}
    assert curve.dense == {}
    assert curve.summary["missing_source_fields"] == [
        "x_min",
        "x_max",
        "sigma",
        "grid_points",
    ]


def test_an_unknown_physical_boundary_stays_unknown():
    reconstruction = _curve_reconstruction().model_copy(
        update={"negative_physical_boundary": None}
    )
    curve = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=_curve_config(),
    )
    assert curve.status is CAMOutcomeStatus.VALID
    assert curve.summary["negative_physical_boundary"] is None
    assert curve.summary["positive_physical_boundary"] is True


def test_a_stable_direction_that_does_not_decay_leaves_no_interval():
    reconstruction = _curve_reconstruction()
    curve = effective_drift_curve(
        _UnstableCurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=_curve_config(),
    )
    assert curve.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert curve.reason_code == "no_stable_interval"
    assert curve.source_status is CAMOutcomeStatus.VALID
    assert curve.generator is None
    assert curve.evaluation is None
    assert curve.coarse == {}
    # The failure carries the drift of the solve that refused it: no point passed.
    assert curve.summary["valid_coarse_count"] == 0
    assert curve.summary["stable_margin_range"] == pytest.approx([-4.5, -4.5])


def test_a_stable_interval_that_is_disconnected_is_refused():
    reconstruction = _curve_reconstruction()
    curve = effective_drift_curve(
        _BandCurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=_curve_config(),
    )
    assert curve.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert curve.reason_code == "disconnected_interval"
    assert curve.source_status is CAMOutcomeStatus.VALID
    assert curve.generator is None
    assert curve.dense == {}
    # The two ends are stable and the fifteen points between them are not: the curve
    # refuses to interpolate across the gap instead of dropping the ends.
    assert curve.summary["valid_coarse_count"] == 6
    assert curve.summary["invalid_inside_interval_count"] == 15
    assert curve.summary["stable_margin_range"] == pytest.approx([-4.0, 4.0], abs=1e-9)


def test_a_reconstruction_without_a_two_sided_interval_raises():
    reconstruction = _curve_reconstruction()
    one_sided = reconstruction.model_copy(update={"x_min": 0.0})
    with pytest.raises(ValueError, match="does not straddle zero"):
        effective_drift_curve(
            _CurveAdapter(),
            _CurveModel(),
            _observable(),
            state=STATE,
            params={},
            reconstruction=one_sided,
            config=_curve_config(),
        )
    without_sigma = reconstruction.model_copy(update={"sigma": None})
    with pytest.raises(ValueError, match="missing sigma"):
        effective_drift_curve(
            _CurveAdapter(),
            _CurveModel(),
            _observable(),
            state=STATE,
            params={},
            reconstruction=without_sigma,
            config=_curve_config(),
        )


def test_the_curve_coarse_grid_is_the_reconstruction_grid():
    """``coarse_points_per_side`` counts one side; the default therefore holds 401."""
    settings = EffectiveDriftConfig()
    assert settings.coarse_points_per_side == 201
    assert 2 * settings.coarse_points_per_side - 1 == 401
    # A reconstruction on a different grid is a structural mismatch, not a rescale: the
    # default 201-per-side curve refuses this 21-point reconstruction instead of quietly
    # sampling its own grid.
    reconstruction = _curve_reconstruction()
    assert reconstruction.grid_points == 21
    with pytest.raises(ValueError, match="would hold 401 points"):
        effective_drift_curve(
            _CurveAdapter(),
            _CurveModel(),
            _observable(),
            state=STATE,
            params={},
            reconstruction=reconstruction,
            config=EffectiveDriftConfig(),
        )
    with pytest.raises(ValueError, match="coarse_points_per_side=11"):
        effective_drift_curve(
            _CurveAdapter(),
            _CurveModel(),
            _observable(),
            state=STATE,
            params={},
            reconstruction=reconstruction,
            config=EffectiveDriftConfig(coarse_points_per_side=31, dense_points=41),
        )


def test_a_forced_curve_with_zero_intensity_is_not_reported_as_valid():
    """A constant readout has no information: 0/0 must not read as a valid curve."""
    reconstruction = scalar_reconstruction(
        _CurveAdapter(),
        _CurveModel(),
        _zero_readout(),
        state=STATE,
        params={},
        manifold=_curve_manifold(),
        parameter_direction=PARAMETER_DIRECTION,
        config=ScalarReconstructionConfig(grid_points=CURVE_SIDE),
    )
    assert reconstruction.status is CAMOutcomeStatus.VALID
    assert reconstruction.scalar_intensity == 0.0
    forced = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _zero_readout(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        parameter_direction=PARAMETER_DIRECTION,
        config=_curve_config(),
    )
    assert forced.status is not CAMOutcomeStatus.VALID
    assert forced.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert forced.reason_code == "intensity_not_positive"
    assert forced.generator is None
    assert forced.dense == {}
    assert forced.summary["scalar_intensity"] == 0.0
    assert "information rate is undefined" in forced.reason
    # Without forcing the same zero is a real zero: no information rate is requested, so
    # there is nothing to divide by.
    unforced = effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _zero_readout(),
        state=STATE,
        params={},
        reconstruction=reconstruction,
        config=_curve_config(),
    )
    assert unforced.status is CAMOutcomeStatus.VALID
    assert unforced.summary["allan_intensity"] == 0.0
    assert unforced.summary["information_rate"] is None
    assert unforced.summary["path_information_rate"] is None
    assert unforced.summary["information_over_path"] is None


def test_the_curve_configuration_is_validated():
    for updates in (
        {"coarse_points_per_side": 2},
        {"dense_points": 10},
        {"derivative_step_factor": 0.0},
        {"derivative_step_factor": float("nan")},
        {"derivative_step_factor": float("inf")},
        {"physical_tolerance": -1.0},
        {"physical_tolerance": float("nan")},
        {"physical_tolerance": float("inf")},
        {"stable_covariance_tolerance": -1.0},
        {"stable_covariance_tolerance": float("nan")},
        {"stable_covariance_tolerance": float("-inf")},
    ):
        with pytest.raises(ValidationError):
            EffectiveDriftConfig(**updates)
    settings = EffectiveDriftConfig()
    assert settings.dense_points == 6001
    assert settings.derivative_step_factor == pytest.approx(3.0e-7)
    # The default quadrature is the historical dense-table convention.
    assert settings.quadrature.integration_split == "nearest_zero"
    assert ScalarReconstructionConfig().integration_split == "grid_zero"


# --- the postprocessor: the optional effective field ----------------------


class _Patch:
    """Patch the adapter factory of the postprocessor module."""

    def __init__(self, monkeypatch: Any, adapter: Any) -> None:
        monkeypatch.setattr(
            "qphase_cam.postprocessor.nonlinear_reconstruction."
            "QPSGenDynamicsAdapter.from_model",
            lambda model: adapter,
        )


def _state_product(*, success: bool = True) -> Any:
    matrix = np.asarray(vector_to_matrix(STATE, 2), dtype=complex)
    return build_state_product(
        states=matrix.reshape(1, 1, 2, 2),
        residuals=np.zeros((1, 1)),
        success=np.asarray([[success]], dtype=bool),
        valid_mask=np.ones((1, 1), dtype=bool),
        solution_count=np.ones(1, dtype=np.int64),
        parameters={"mu": np.asarray([0.0])},
    )


def _context(product: Any) -> CAMAnnotationContext:
    return CAMAnnotationContext(
        model=_CurveModel(), products={"state": product}, source_role="state"
    )


def _plugin(orders: tuple[int, ...] = (3,), **updates: Any) -> Any:
    settings: dict[str, Any] = {
        "orders": orders,
        "reconstruction": ScalarReconstructionConfig(grid_points=CURVE_SIDE),
        "observable": _observable(),
    }
    settings.update(updates)
    return NonlinearReconstruction(**settings)


def _reference_curve(order: int = 3) -> Any:
    """Return the curve the plugin row must reproduce, along the plugin's own path."""
    manifold = slow_manifold(
        _CurveAdapter(),
        STATE,
        {"mu": 0.0},
        order=order,
        config=NonlinearManifoldConfig(order=order),
    )
    reconstruction = scalar_reconstruction(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={"mu": 0.0},
        order=order,
        manifold=manifold,
        config=ScalarReconstructionConfig(grid_points=CURVE_SIDE),
    )
    return effective_drift_curve(
        _CurveAdapter(),
        _CurveModel(),
        _observable(),
        state=STATE,
        params={"mu": 0.0},
        reconstruction=reconstruction,
        config=_curve_config(),
    )


def _field(annotation: Any, name: str) -> np.ndarray:
    return np.asarray(annotation.handle(name).materialize())


def test_the_default_annotation_has_no_effective_fields(monkeypatch):
    _Patch(monkeypatch, _CurveAdapter())
    product = _state_product()
    processor = _plugin()
    annotation = processor.process(product, _context(product))
    names = [variable.name for variable in annotation.variables]
    assert not [name for name in names if name.startswith("effective")]
    assert [axis.name for axis in annotation.axes] == ["annotation", "order"]
    assert not [
        name
        for name in annotation.attributes["field_axes"]
        if name.startswith("effective")
    ]
    # Nothing is added to either metadata block, so the default annotation is what it
    # was before the option existed.
    assert "effective" not in processor.result_metadata
    assert "effective" not in str(_field(annotation, "method_metadata")[0])
    assert _field(annotation, "order_status").tolist() == [
        ["uncontrolled_approximation"]
    ]
    assert np.isfinite(_field(annotation, "scalar_intensity")).all()


def test_the_enabled_curve_is_emitted_per_order(monkeypatch):
    _Patch(monkeypatch, _CurveAdapter())
    product = _state_product()
    processor = _plugin(orders=(3, 5), effective=_curve_config())
    annotation = processor.process(product, _context(product))
    assert _field(annotation, "order").tolist() == [[3, 5]]
    # The plugin resolves its own manifold, which does not confine this toy: the row is
    # an uncontrolled approximation and the curve repeats that instead of upgrading it.
    assert _field(annotation, "status").tolist() == ["uncontrolled_approximation"]
    assert _field(annotation, "order_status").tolist() == [
        ["uncontrolled_approximation", "unidentifiable"]
    ]
    assert _field(annotation, "order_reason_code").tolist() == [
        ["manifold_not_confining", "manifold_unresolved"]
    ]
    assert _field(annotation, "effective_status").tolist() == [
        ["uncontrolled_approximation", "unidentifiable"]
    ]
    assert _field(annotation, "effective_reason_code").tolist() == [
        ["manifold_not_confining", "manifold_unresolved"]
    ]
    assert "not valid" in _field(annotation, "effective_reason")[0][0]

    # Axes: one coarse point per side counting zero, and the dense table's own count.
    shared = annotation.attributes["shared_axes"]
    assert shared["order"] == 2
    assert shared["effective_coarse_point"] == 2 * CURVE_SIDE - 1 == 21
    assert shared["effective_dense_point"] == 41
    assert {axis.name: axis.size for axis in annotation.axes} == {
        "annotation": 1,
        "order": 2,
        "effective_coarse_point": 21,
        "effective_dense_point": 41,
    }
    field_axes = annotation.attributes["field_axes"]
    for name in ("effective_x", "effective_forcing", "effective_valid"):
        assert field_axes[name] == ["order", "effective_coarse_point"], name
    for name in ("effective_dense_x", "effective_density"):
        assert field_axes[name] == ["order", "effective_dense_point"], name
    assert annotation.attributes["ragged"] is False
    # The typed schema names the same axes, so a consumer reads them off the product.
    dims = {variable.name: variable.dims for variable in annotation.variables}
    assert dims["effective_x"] == (
        "annotation",
        "order",
        "effective_coarse_point",
    )
    assert dims["effective_dense_x"] == (
        "annotation",
        "order",
        "effective_dense_point",
    )
    assert dims["effective_status"] == ("annotation", "order")
    assert dims["effective_valid_coarse_count"] == ("annotation", "order")

    # The values are the curve's own: coarse and dense, point for point.
    curve = _reference_curve(3)
    assert curve.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    coarse_columns = {
        "effective_x": "x",
        "effective_f_deterministic": "f_deterministic",
        "effective_f_effective": "f_effective",
        "effective_q_effective": "q_effective",
        "effective_omega_effective": "omega_effective",
        "effective_stable_covariance_trace": "stable_covariance_trace",
        "effective_stable_margin": "stable_margin",
        "effective_physical_minimum": "physical_minimum",
        "effective_lyapunov_residual": "lyapunov_residual",
    }
    for name, key in coarse_columns.items():
        np.testing.assert_allclose(_field(annotation, name)[0, 0], curve.coarse[key])
    dense_columns = {
        "effective_dense_x": "x",
        "effective_density": "density",
        "effective_f_effective_dense": "f_effective",
        "effective_q_effective_dense": "q_effective",
        "effective_omega_effective_dense": "omega_effective",
        "effective_stable_margin_dense": "stable_margin",
    }
    for name, key in dense_columns.items():
        np.testing.assert_allclose(_field(annotation, name)[0, 0], curve.dense[key])
    assert _field(annotation, "effective_valid")[0, 0].all()
    # The coarse grid spans the reconstruction's own boundaries.
    coarse_x = _field(annotation, "effective_x")[0, 0]
    assert coarse_x[0] == _field(annotation, "x_min")[0, 0]
    assert coarse_x[-1] == _field(annotation, "x_max")[0, 0]

    # Summary scalars, with NaN for what the curve did not produce.
    assert int(_field(annotation, "effective_valid_coarse_count")[0, 0]) == 21
    for name, key in (
        ("effective_stationary_mean", "stationary_mean"),
        ("effective_stationary_mean_x", "stationary_mean_x"),
        ("effective_allan_intensity", "allan_intensity"),
        ("effective_edge_mass_fraction", "edge_mass_fraction"),
        ("effective_endpoint_density_ratio", "endpoint_density_ratio"),
        ("effective_minimum_q_effective", "minimum_q_effective"),
        ("effective_minimum_stable_margin", "minimum_stable_margin"),
        ("effective_lyapunov_residual_max", "lyapunov_residual_max"),
        (
            "effective_probability_stable_margin_le_1x_soft_rate",
            "probability_stable_margin_le_1x_soft_rate",
        ),
        (
            "effective_probability_stable_margin_le_3x_soft_rate",
            "probability_stable_margin_le_3x_soft_rate",
        ),
        (
            "effective_probability_stable_margin_le_5x_soft_rate",
            "probability_stable_margin_le_5x_soft_rate",
        ),
        (
            "effective_probability_stable_margin_le_10x_soft_rate",
            "probability_stable_margin_le_10x_soft_rate",
        ),
    ):
        assert float(_field(annotation, name)[0, 0]) == pytest.approx(
            curve.summary[key], rel=1e-12
        ), name
    assert _field(annotation, "effective_interval_min")[0, 0] == pytest.approx(
        curve.summary["coarse_interval"][0], rel=0.0, abs=0.0
    )
    assert _field(annotation, "effective_interval_max")[0, 0] == pytest.approx(
        curve.summary["coarse_interval"][1], rel=0.0, abs=0.0
    )
    # No forcing was requested, so the forced numbers and the dense forcing vector are
    # unavailable rather than zero.
    assert np.isnan(_field(annotation, "effective_information_rate")[0, 0])
    assert np.isnan(_field(annotation, "effective_path_information_rate")[0, 0])
    assert np.isnan(_field(annotation, "effective_forcing")[0, 0]).all()
    assert np.isnan(_field(annotation, "effective_forcing_dense")[0, 0]).all()

    # Order 5 has no curve at all: unavailable values, not a shorter array.
    assert int(_field(annotation, "effective_valid_coarse_count")[0, 1]) == -1
    assert np.isnan(_field(annotation, "effective_x")[0, 1]).all()
    assert np.isnan(_field(annotation, "effective_dense_x")[0, 1]).all()
    assert not _field(annotation, "effective_valid")[0, 1].any()
    assert np.isnan(_field(annotation, "effective_stationary_mean")[0, 1])

    # Method and result metadata name the configuration, the curve's own method and the
    # source verdict.
    metadata = str(_field(annotation, "method_metadata")[0])
    for needle in (
        "nearest_zero",
        "stable_projection",
        "source_status",
        "source_valid",
        "uncontrolled_approximation",
        "coarse_points_per_side",
    ):
        assert needle in metadata, needle
    curve_metadata = processor.result_metadata["effective"]
    assert curve_metadata["enabled"] is True
    assert curve_metadata["coarse_points_per_side"] == CURVE_SIDE
    assert curve_metadata["coarse_point_count"] == 21
    assert curve_metadata["dense_points"] == 41
    assert curve_metadata["integration_split"] == "nearest_zero"
    assert curve_metadata["config"]["quadrature"]["integration_split"] == "nearest_zero"


def test_the_effective_columns_round_trip_through_an_artifact(monkeypatch, tmp_path):
    _Patch(monkeypatch, _CurveAdapter())
    product = _state_product()
    annotation = _plugin(effective=_curve_config()).process(product, _context(product))
    cam_bundle({"nonlinear_reconstruction": annotation}).save_dataset(
        tmp_path, layout="single", shard_target_bytes=4096
    )
    restored = load_bundle(tmp_path).require("nonlinear_reconstruction")
    for name in (
        "effective_x",
        "effective_q_effective",
        "effective_valid",
        "effective_dense_x",
        "effective_density",
        "effective_valid_coarse_count",
        "effective_minimum_stable_margin",
    ):
        np.testing.assert_allclose(
            np.asarray(restored.handle(name).materialize(), dtype=float),
            np.asarray(annotation.handle(name).materialize(), dtype=float),
            equal_nan=True,
        )
    # The status column is text, so it is compared as one.
    assert (
        _field(restored, "effective_status").tolist()
        == _field(annotation, "effective_status").tolist()
    )
    assert [axis.name for axis in restored.axes] == [
        "annotation",
        "order",
        "effective_coarse_point",
        "effective_dense_point",
    ]


def test_a_curve_failure_caps_the_row_status(monkeypatch):
    _Patch(monkeypatch, _UnstableCurveAdapter())
    product = _state_product()
    annotation = _plugin(effective=_curve_config()).process(product, _context(product))
    # The curve's internal failure is not hidden by the source's own verdict...
    assert _field(annotation, "status").tolist() == ["stochastic_invalid"]
    assert _field(annotation, "order_status").tolist() == [["stochastic_invalid"]]
    assert _field(annotation, "order_reason_code").tolist() == [["no_stable_interval"]]
    assert _field(annotation, "effective_status").tolist() == [["stochastic_invalid"]]
    assert _field(annotation, "effective_reason_code").tolist() == [
        ["no_stable_interval"]
    ]
    assert "reconstruction:" in _field(annotation, "order_reason")[0][0]
    # The base reconstruction's verdict stays visible in the method block, so capping
    # the row status does not erase what the source said.
    metadata = str(_field(annotation, "method_metadata")[0])
    assert "uncontrolled_approximation" in metadata
    assert "no_stable_interval" in metadata
    assert "source_valid" in metadata
    # ... and every array is unavailable instead of being an older success.
    assert np.isnan(_field(annotation, "effective_x")).all()
    assert np.isnan(_field(annotation, "effective_dense_x")).all()
    assert not _field(annotation, "effective_valid").any()
    assert int(_field(annotation, "effective_valid_coarse_count")[0, 0]) == -1


def test_an_unconverged_target_has_no_curve(monkeypatch):
    _Patch(monkeypatch, _CurveAdapter())
    product = _state_product(success=False)
    annotation = _plugin(effective=_curve_config()).process(product, _context(product))
    assert _field(annotation, "status").tolist() == ["not_converged"]
    assert _field(annotation, "effective_status").tolist() == [["not_converged"]]
    assert _field(annotation, "effective_reason_code").tolist() == [["not_converged"]]
    assert np.isnan(_field(annotation, "effective_x")).all()
    assert np.isnan(_field(annotation, "effective_dense_x")).all()
    assert not _field(annotation, "effective_valid").any()
    assert int(_field(annotation, "effective_valid_coarse_count")[0, 0]) == -1
    assert np.isnan(_field(annotation, "effective_stationary_mean")).all()


def test_the_effective_grid_must_match_the_reconstruction():
    with pytest.raises(ValueError, match="coarse_points_per_side"):
        NonlinearReconstruction(
            orders=(3,),
            reconstruction=ScalarReconstructionConfig(grid_points=CURVE_SIDE),
            effective=EffectiveDriftConfig(
                coarse_points_per_side=CURVE_SIDE + 2, dense_points=41
            ),
        )
    # The matching pair is accepted, and the mismatch says what the coarse grid is.
    NonlinearReconstruction(
        orders=(3,),
        reconstruction=ScalarReconstructionConfig(grid_points=CURVE_SIDE),
        effective=EffectiveDriftConfig(
            coarse_points_per_side=CURVE_SIDE, dense_points=41
        ),
    )


def test_the_effective_grid_is_checked_before_any_computation(monkeypatch):
    def explode(model: Any) -> Any:
        raise AssertionError("the adapter factory must not be reached")

    monkeypatch.setattr(
        "qphase_cam.postprocessor.nonlinear_reconstruction."
        "QPSGenDynamicsAdapter.from_model",
        explode,
    )
    product = _state_product()
    processor = _plugin(effective=_curve_config())
    # A config assigned after construction skips the validator, so ``process`` must
    # refuse the mismatch itself, before it asks for an adapter.
    processor.config.reconstruction = ScalarReconstructionConfig(grid_points=5)
    with pytest.raises(ValueError, match="coarse_points_per_side"):
        processor.process(product, _context(product))
