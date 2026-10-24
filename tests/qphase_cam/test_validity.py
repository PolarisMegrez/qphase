"""Phase 10F1: observables, the derivative record and the unified validity service.

The numeric benchmark is the scalar Ornstein-Uhlenbeck process: ``J = -1``,
``D = 1``, observable ``x``. With the Itô convention ``J Σ + Σ Jᵀ + D = 0`` that
gives ``Σ = 0.5``, a zero-frequency response of ``1``, an integrated noise
intensity of ``1`` and a correlation time of ``1``. Every number below is an
exact analytic value, not a recorded run, so the service cannot drift silently.
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from qphase.data import load_bundle
from qphase_cam.data import cam_bundle
from qphase_cam.observable import (
    CAMObservable,
    ChargeSector,
    DerivativeRecord,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
    RayleighObservable,
    coordinate_scale,
    finite_difference_gradient,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.stochastic_validity import (
    StochasticValidity,
    StochasticValidityConfig,
)
from qphase_cam.validity import (
    StochasticAssessmentConfig,
    assess_stochastic_state,
    build_stochastic_assessment_annotation,
    critical_mode,
    lyapunov_covariance,
    minimum_eigenvalue,
    noise_timescales,
    projected_intensity,
    zero_frequency_gain,
)

SCALAR_IDENTITY = ObservableIdentity(
    kind=ObservableKind.STATE,
    sector=ChargeSector.CHARGE_ZERO,
    name="x",
    measurement_mapping="x",
)


class _ScalarAdapter:
    """One-coordinate adapter: state matrix is the scalar, Jacobian is given."""

    moment_layout = "normal"

    def __init__(self, jacobian=-1.0) -> None:
        self._jacobian = np.atleast_2d(np.asarray(jacobian, dtype=float))

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector, dtype=float).reshape(1, 1)

    def jacobian(self, vector, params=None):
        del vector, params
        return self._jacobian

    def closure_provenance(self):
        return {"representation": "wigner"}


def _scalar_observable(*, analytic_gradient: bool = True):
    def value(adapter, state, params):
        del adapter, params
        return float(np.asarray(state, dtype=float).reshape(-1)[0])

    def gradient(adapter, state, params):
        del adapter, state, params
        return [1.0]

    return FunctionalObservable(
        identity=SCALAR_IDENTITY,
        value_fn=value,
        gradient_fn=gradient if analytic_gradient else None,
    )


def _scalar_assessment(*, jacobian=-1.0, diffusion=1.0, analytic_gradient=True):
    return assess_stochastic_state(
        _ScalarAdapter(jacobian),
        _scalar_observable(analytic_gradient=analytic_gradient),
        state=[1.0],
        params={},
        diffusion=np.asarray([[diffusion]], dtype=float),
    )


# --- scalar OU benchmark ---------------------------------------------------


def test_scalar_ou_benchmark_is_exact():
    """cov=0.5, zero response=1, integrated noise=1, correlation time=1."""
    assessment = _scalar_assessment()
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason is None
    assert assessment.lyapunov_covariance_trace == pytest.approx(0.5, rel=1e-12)
    assert assessment.lyapunov_covariance_minimum_eigenvalue == pytest.approx(
        0.5, rel=1e-12
    )
    assert assessment.lyapunov_relative_residual < 1e-12
    assert assessment.zero_frequency_response_norm == pytest.approx(1.0, rel=1e-12)
    assert assessment.integrated_noise_intensity == pytest.approx(1.0, rel=1e-12)
    assert assessment.correlation_time == pytest.approx(1.0, rel=1e-12)
    assert assessment.projected_variance == pytest.approx(0.5, rel=1e-12)
    assert assessment.projected_variance_is_positive is True
    assert assessment.diffusion_is_psd is True
    assert assessment.lyapunov_is_psd is True
    assert assessment.critical_mode_resolved is True
    assert assessment.critical_eigenvalue_real == pytest.approx(-1.0, rel=1e-12)
    assert assessment.representation_supported is True
    assert assessment.derivative is not None
    assert assessment.derivative.method == "analytic"
    assert assessment.method["lyapunov_convention"] == "J Sigma + Sigma J^T + D = 0"


def test_scalar_ou_benchmark_holds_for_the_finite_difference_gradient():
    assessment = _scalar_assessment(analytic_gradient=False)
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.lyapunov_covariance_trace == pytest.approx(0.5, rel=1e-12)
    assert assessment.zero_frequency_response_norm == pytest.approx(1.0, rel=1e-9)
    assert assessment.integrated_noise_intensity == pytest.approx(1.0, rel=1e-9)
    assert assessment.correlation_time == pytest.approx(1.0, rel=1e-9)
    record = assessment.derivative
    assert isinstance(record, DerivativeRecord)
    assert record.method == "central_difference"
    assert record.order == 2
    assert record.step == (pytest.approx(1e-5),)
    assert record.scale == (1.0,)
    assert record.repeat_relative_error is not None
    assert record.repeat_relative_error < 1e-8


def test_the_service_uses_the_plugin_covariance_conventions():
    """The plugin and the service cannot drift apart on tolerances."""
    plugin = StochasticValidityConfig()
    service = StochasticAssessmentConfig()
    assert plugin.critical_eigenvalue_tolerance == service.critical_eigenvalue_tolerance
    assert plugin.covariance_tolerance == service.covariance_tolerance
    assert plugin.overlap_tolerance == service.overlap_tolerance


def test_the_plugin_reuses_the_shared_covariance_and_projection():
    import qphase_cam.core.noise as noise
    import qphase_cam.postprocessor.stochastic_validity as plugin

    # The plugin carries no private copy: every covariance and mode helper is the
    # shared object the service uses.
    assert plugin.canonical_sample_matrix_covariance is (
        noise.canonical_sample_matrix_covariance
    )
    assert plugin.minimum_eigenvalue is minimum_eigenvalue
    assert plugin.projected_intensity is projected_intensity
    assert plugin.critical_mode is critical_mode


# --- failure modes ---------------------------------------------------------


def test_negative_diffusion_is_reported_not_clamped():
    assessment = _scalar_assessment(diffusion=-1.0)
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert "positive semidefinite" in (assessment.reason or "")
    assert assessment.diffusion_is_psd is False
    # The negative eigenvalue survives as it is; it is not clipped to zero.
    assert assessment.diffusion_minimum_eigenvalue == pytest.approx(-1.0, rel=1e-12)
    # The linearisation diagnostics computed before the failing check stay.
    assert assessment.critical_mode_resolved is True
    assert assessment.critical_eigenvalue_real == pytest.approx(-1.0, rel=1e-12)
    assert assessment.lyapunov_covariance_trace is None
    assert assessment.projected_variance is None


def test_singular_jacobian_has_no_stationary_covariance():
    assessment = _scalar_assessment(jacobian=0.0)
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert "Lyapunov" in (assessment.reason or "")
    assert assessment.lyapunov_solution_finite is True
    assert assessment.lyapunov_relative_residual is not None
    assert assessment.lyapunov_relative_residual > 0.5
    assert assessment.integrated_noise_intensity is None


def test_zero_projected_variance_is_unidentifiable():
    jacobian = np.diag([-1.0, -1.0])
    diffusion = np.diag([1.0, 0.0])

    def value(adapter, state, params):
        del adapter, params
        return float(np.asarray(state, dtype=float).reshape(-1)[1])

    observable = FunctionalObservable(
        identity=SCALAR_IDENTITY,
        value_fn=value,
        # A non-zero gradient that is orthogonal to the noise: the readout sees
        # nothing, which is unidentifiable rather than a zero number.
        gradient_fn=lambda adapter, state, params: [0.0, 1.0],
    )
    assessment = assess_stochastic_state(
        _ScalarAdapter(jacobian),
        observable,
        state=[0.0, 0.0],
        params={},
        diffusion=diffusion,
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert "projected variance" in (assessment.reason or "")
    assert assessment.projected_variance == pytest.approx(0.0, abs=1e-15)
    assert assessment.projected_variance_is_positive is False
    assert assessment.zero_frequency_response_norm is None


def test_a_zero_gradient_is_unidentifiable():
    observable = FunctionalObservable(
        identity=SCALAR_IDENTITY,
        value_fn=lambda adapter, state, params: 1.0,
        gradient_fn=lambda adapter, state, params: [0.0],
    )
    assessment = assess_stochastic_state(
        _ScalarAdapter(-1.0),
        observable,
        state=[1.0],
        params={},
        diffusion=np.asarray([[1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert "gradient" in (assessment.reason or "")
    assert assessment.diffusion_is_psd is True


def test_anomalous_diffusion_is_an_uncontrolled_approximation():
    class Model:
        params = {"mu": 0.0}

        @staticmethod
        def cam_diffusion(state, params):
            del state, params
            return np.asarray([[1.0]])

        @staticmethod
        def cam_anomalous_diffusion(state, params):
            del state, params
            return np.asarray([[0.5]])

    assessment = assess_stochastic_state(
        _ScalarAdapter(-1.0),
        _scalar_observable(),
        state=[1.0],
        params={},
        model=Model(),
    )
    assert assessment.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert "anomalous diffusion" in (assessment.reason or "")
    assert assessment.anomalous_diffusion_norm == pytest.approx(0.5, rel=1e-12)
    assert assessment.representation_supported is True


def test_an_unsupported_moment_layout_is_uncontrolled():
    adapter = _ScalarAdapter(-1.0)
    adapter.moment_layout = "classical"
    assessment = assess_stochastic_state(
        adapter,
        _scalar_observable(),
        state=[1.0],
        params={},
        diffusion=np.asarray([[1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert "moment layout" in (assessment.reason or "")
    assert assessment.representation_supported is False
    assert assessment.critical_mode_resolved is False


def test_a_growing_mode_is_unstable():
    assessment = _scalar_assessment(jacobian=0.5)
    assert assessment.status is CAMOutcomeStatus.UNSTABLE
    assert "grows" in (assessment.reason or "")
    assert assessment.critical_eigenvalue_real == pytest.approx(0.5, rel=1e-12)
    assert assessment.diffusion_is_psd is None
    assert assessment.critical_mode_resolved is True


def test_a_rotating_jacobian_has_no_real_critical_mode():
    jacobian = np.asarray([[0.0, -1.0], [1.0, 0.0]])
    assert critical_mode(jacobian) is None
    observable = FunctionalObservable(
        identity=SCALAR_IDENTITY,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0]
        ),
    )
    assessment = assess_stochastic_state(
        _ScalarAdapter(jacobian),
        observable,
        state=[1.0, 0.0],
        params={},
        diffusion=np.eye(2),
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert "critical mode" in (assessment.reason or "")
    assert assessment.critical_mode_resolved is False


def test_a_non_finite_lyapunov_solution_fails_instead_of_looking_large():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with np.errstate(over="ignore", invalid="ignore"):
            assessment = _scalar_assessment(jacobian=-1e-300, diffusion=1e300)
            covariance, residual = lyapunov_covariance(
                np.asarray([[-1e-300]]), np.asarray([[1e300]])
            )
    assert not np.all(np.isfinite(covariance))
    assert not np.isfinite(residual)
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert assessment.lyapunov_solution_finite is False
    assert "not finite" in (assessment.reason or "")


def test_the_observable_must_be_evaluable():
    def explode(adapter, state, params):
        del adapter, state, params
        raise ValueError("trace vanished")

    observable = FunctionalObservable(
        identity=SCALAR_IDENTITY,
        value_fn=explode,
        gradient_fn=lambda adapter, state, params: [1.0],
    )
    assessment = assess_stochastic_state(
        _ScalarAdapter(-1.0),
        observable,
        state=[1.0],
        params={},
        diffusion=np.asarray([[1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert "trace vanished" in (assessment.reason or "")
    assert assessment.lyapunov_covariance_trace == pytest.approx(0.5, rel=1e-12)


def test_the_service_derives_the_canonical_diffusion_from_the_model():
    """Without an explicit diffusion the model's is transformed, not invented."""

    class Model:
        params = {"mu": 0.0}

        @staticmethod
        def cam_diffusion(state, params):
            del params
            # A unit diffusion matrix on a state whose trace is one maps to the
            # canonical covariance D = 2 (the sample-matrix transform is
            # R-dependent), so Sigma = D / (2|J|) = 1.
            assert np.asarray(state).shape == (1, 1)
            return np.asarray([[1.0]])

    assessment = assess_stochastic_state(
        _ScalarAdapter(-1.0),
        _scalar_observable(),
        state=[1.0],
        params={},
        model=Model(),
    )
    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.diffusion_minimum_eigenvalue == pytest.approx(2.0, rel=1e-12)
    assert assessment.lyapunov_covariance_trace == pytest.approx(1.0, rel=1e-12)
    assert assessment.integrated_noise_intensity == pytest.approx(2.0, rel=1e-12)
    assert assessment.correlation_time == pytest.approx(1.0, rel=1e-12)


def test_missing_diffusion_and_model_is_an_error():
    with pytest.raises(ValueError, match="diffusion covariance"):
        assess_stochastic_state(
            _ScalarAdapter(-1.0),
            _scalar_observable(),
            state=[1.0],
            params={},
        )


# --- low-level helpers -----------------------------------------------------


def test_noise_timescales_match_the_analytic_integrals():
    jacobian = np.asarray([[-2.0]])
    covariance = np.asarray([[0.25]])  # D = 1 gives Sigma = D / (2|J|)
    integrated, correlation_time = noise_timescales(
        jacobian, covariance, np.asarray([1.0])
    )
    assert integrated == pytest.approx(2.0 * 0.25 / 2.0, rel=1e-12)
    assert correlation_time == pytest.approx(0.5, rel=1e-12)
    response, norm = zero_frequency_gain(jacobian, np.asarray([1.0]))
    assert norm == pytest.approx(0.5, rel=1e-12)
    assert response == pytest.approx(np.asarray([0.5]), rel=1e-12)


# --- observables -----------------------------------------------------------


class _MatrixAdapter:
    """Two-coordinate adapter whose state matrix is ``diag(x0, x1)``."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        values = np.asarray(vector, dtype=float).reshape(-1)
        return np.diag(values.astype(complex))

    def jacobian(self, vector, params=None):
        del vector, params
        return np.diag([-1.0, -1.0])


class _HamiltonianModel:
    params = {"mu": 0.0}

    @staticmethod
    def cam_hamiltonian(state, params):
        del state, params
        return np.asarray([[1.0, 0.0], [0.0, 3.0]], dtype=complex)


def test_rayleigh_observable_value_and_gradient():
    observable = RayleighObservable(_HamiltonianModel())
    adapter = _MatrixAdapter()
    assert isinstance(observable, CAMObservable)
    assert observable.identity.kind is ObservableKind.CARRIER
    assert observable.identity.sector is ChargeSector.CHARGE_ZERO
    assert observable.identity.measurement_mapping == "Re Tr(H R) / Tr(R)"
    # Tr(H R) / Tr(R) = (2 + 3) / 3 for R = diag(2, 1).
    assert observable.value(adapter, [2.0, 1.0], {}) == pytest.approx(5.0 / 3.0)
    record = observable.gradient(adapter, [2.0, 1.0], {})
    assert record.method == "central_difference"
    # d/dx0 of (x0 + 3 x1)/(x0 + x1) = -2/9, d/dx1 = 4/9.
    assert record.gradient[0] == pytest.approx(-2.0 / 9.0, rel=1e-8)
    assert record.gradient[1] == pytest.approx(4.0 / 9.0, rel=1e-8)
    assert record.repeat_relative_error < 1e-8


def test_rayleigh_observable_rejects_a_model_without_a_hamiltonian():
    class NoHamiltonian:
        params = {}

    with pytest.raises(TypeError, match="cam_hamiltonian"):
        RayleighObservable(NoHamiltonian())


def test_coordinate_scale_and_finite_difference_validation():
    np.testing.assert_allclose(coordinate_scale([0.0, -3.0, 0.5]), [1.0, 3.0, 1.0])
    record = finite_difference_gradient(
        lambda vector: float(vector[0] ** 2 + 3.0 * vector[1]), [2.0, 1.0]
    )
    assert record.gradient == (
        pytest.approx(4.0, rel=1e-8),
        pytest.approx(3.0, rel=1e-8),
    )
    assert record.condition == pytest.approx(4.0 / 3.0, rel=1e-8)
    with pytest.raises(ValueError, match="finite state"):
        finite_difference_gradient(lambda v: 0.0, [np.nan])
    with pytest.raises(ValueError, match="relative_step"):
        finite_difference_gradient(lambda v: 0.0, [1.0], relative_step=0.0)
    with pytest.raises(ValueError, match="scale has shape"):
        finite_difference_gradient(lambda v: 0.0, [1.0, 2.0], scale=[1.0])


# --- annotation builder ----------------------------------------------------


def test_assessment_annotation_carries_status_fields_and_reasons():
    valid = _scalar_assessment()
    invalid = _scalar_assessment(diffusion=-1.0)
    product = build_stochastic_assessment_annotation(
        [valid, invalid], target_identity=["cam:solution:a", "cam:solution:b"]
    )
    assert list(np.asarray(product.handle("status").materialize())) == [
        "valid",
        "stochastic_invalid",
    ]
    np.testing.assert_allclose(
        np.asarray(product.handle("zero_frequency_response_norm").materialize())[:1],
        [1.0],
        rtol=1e-12,
    )
    # An unavailable number stays NaN instead of becoming zero.
    assert np.isnan(
        np.asarray(product.handle("zero_frequency_response_norm").materialize())[1]
    )
    assert np.isnan(
        np.asarray(product.handle("lyapunov_covariance_trace").materialize())[1]
    )
    # A boolean the assessment could not decide is -1, not False.
    np.testing.assert_array_equal(
        np.asarray(product.handle("diffusion_is_psd").materialize()), [1, 0]
    )
    reasons = np.asarray(product.handle("reason").materialize())
    assert reasons[0] == ""
    assert "positive semidefinite" in str(reasons[1])
    metadata = np.asarray(product.handle("method_metadata").materialize())
    assert "tristate_semantics" in str(metadata[0])


def test_assessment_annotation_roundtrips_and_rejects_bad_shapes():
    product = build_stochastic_assessment_annotation(
        [_scalar_assessment()], target_identity=["cam:solution:a"]
    )
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cam_bundle({"validity": product}).save_dataset(
            root, layout="single", shard_target_bytes=4096
        )
        restored = load_bundle(root).require("validity")
        np.testing.assert_allclose(
            np.asarray(restored.handle("correlation_time").materialize()), [1.0]
        )
    with pytest.raises(ValueError, match="at least one assessment"):
        build_stochastic_assessment_annotation([], target_identity=[])
    with pytest.raises(ValueError, match="target identities"):
        build_stochastic_assessment_annotation(
            [_scalar_assessment()], target_identity=["a", "b"]
        )


def test_the_existing_plugin_still_produces_its_cubic_branch_fields():
    """The plugin keeps its row schema; only the maths moved to the service."""
    template = StochasticValidity._base_row(
        0, branch_index=0, epsilon_side=0, closure={}, status=""
    )
    assert "projected_noise_intensity" in template
    assert "critical_mode_condition_number" in template
