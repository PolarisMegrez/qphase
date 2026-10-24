"""Phase 10F2a: the closure-risk service and its postprocessor.

The numeric benchmark is a scalar mode with a cubic nonlinearity,
``dx/dt = -x - x³``, a unit diffusion and the readout ``f = x + a x²/2``. Every
closure quantity has a closed form:

* ``J = -1 - 3x²``, ``Σ = D / (2|J|)``,
* contraction ``b = ½ (∂J/∂x) Σ = -3x Σ``, displacement ``δx = b / |J|``,
* drift bias ``(1 + a x) δx``, Jensen bias ``½ a Σ``,

so the service is checked against algebra rather than against a recorded run.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.closure import (
    ClosureAssessmentConfig,
    assess_closure,
    build_closure_assessment_annotation,
    closure_contraction,
    closure_shape_fraction,
    observable_jensen_bias,
)
from qphase_cam.data import build_candidate_product, build_state_product
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.closure_risk import ClosureRisk

X0 = 0.5
CURVATURE = 2.0
IDENTITY = ObservableIdentity(
    kind=ObservableKind.STATE,
    sector=ChargeSector.CHARGE_ZERO,
    name="x",
    measurement_mapping="x + a x^2 / 2",
)


class _ScalarAdapter:
    """One-coordinate adapter with ``J(x) = -1 - 3x²`` and a physical spectrum."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray([[np.asarray(vector, dtype=float).reshape(-1)[0]]])

    def jacobian(self, vector, params=None):
        del params
        x = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * x**2]])

    def physical_eigenvalues(self, vector, params=None):
        del params
        x = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([x])


def _scalar_observable(*, curvature: float = CURVATURE, gradient_scale: float = 1.0):
    def value(adapter, state, params):
        del adapter, params
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        return x + 0.5 * curvature * x**2

    def gradient(adapter, state, params):
        del adapter, params
        x = float(np.asarray(state, dtype=float).reshape(-1)[0])
        return [gradient_scale * (1.0 + curvature * x)]

    return FunctionalObservable(identity=IDENTITY, value_fn=value, gradient_fn=gradient)


def _scalar_assessment(*, x0: float = X0, curvature: float = CURVATURE, **kwargs):
    return assess_closure(
        _ScalarAdapter(),
        _scalar_observable(curvature=curvature),
        state=[x0],
        params={},
        diffusion=np.asarray([[1.0]]),
        **kwargs,
    )


# --- scalar analytic benchmark --------------------------------------------


def test_scalar_closure_benchmark_matches_the_closed_forms():
    assessment = _scalar_assessment()
    jacobian = -1.0 - 3.0 * X0**2
    covariance = 1.0 / (2.0 * abs(jacobian))
    contraction = -3.0 * X0 * covariance
    displacement = contraction / abs(jacobian)
    drift = (1.0 + CURVATURE * X0) * displacement
    jensen = 0.5 * CURVATURE * covariance

    assert assessment.status is CAMOutcomeStatus.VALID
    assert assessment.reason is None
    assert assessment.stochastic.lyapunov_covariance_trace == pytest.approx(
        covariance, rel=1e-12
    )
    assert assessment.closure_contraction_norm == pytest.approx(
        abs(contraction), rel=1e-8
    )
    assert assessment.closure_displacement == (pytest.approx(displacement, rel=1e-8),)
    assert assessment.drift_bias == pytest.approx(drift, rel=1e-8)
    assert assessment.jensen_bias == pytest.approx(jensen, rel=1e-8)
    assert assessment.closure_bias_total == pytest.approx(drift + jensen, rel=1e-8)
    # A one-mode state can only move radially.
    assert assessment.closure_shape_fraction == pytest.approx(0.0, abs=1e-14)
    assert assessment.soft_mode_rate == pytest.approx(abs(jacobian), rel=1e-12)
    assert assessment.validity_domain == "weak_noise"
    assert assessment.approximation_order == 2
    assert assessment.physical_minimum_eigenvalue == pytest.approx(X0, rel=1e-12)
    assert assessment.physical_is_psd is True


def test_scalar_closure_records_its_finite_difference_provenance():
    record = _scalar_assessment().closure_derivative
    assert record is not None
    assert record.method == "central_difference"
    assert record.order == 2
    assert record.step == (pytest.approx(1e-5),)
    assert record.scale == (1.0,)
    assert record.repeat_relative_error is not None
    assert record.repeat_relative_error < 1e-8
    assert record.metadata["quantity"] == "closure_drift_contraction"
    # The Jensen curvature is evaluated at its own (script) step and checked.
    assessment = _scalar_assessment()
    assert assessment.jensen_step == pytest.approx(1e-4 * max(1.0, X0), rel=1e-12)
    assert assessment.jensen_repeat_relative_error is not None
    assert assessment.jensen_repeat_relative_error < 1e-6
    assert assessment.jensen_eigenvalue_count == 1


def test_kubo_numbers_agree_in_the_single_mode_limit():
    """One mode means the soft and full noise estimates must coincide."""
    assessment = _scalar_assessment()
    assert assessment.soft_kubo == pytest.approx(assessment.full_kubo, rel=1e-10)
    jacobian = abs(-1.0 - 3.0 * X0**2)
    covariance = 1.0 / (2.0 * jacobian)
    gradient = 1.0 + CURVATURE * X0
    # sigma_full * tau_full with tau = 1/|J| for a single relaxation rate.
    assert assessment.full_kubo == pytest.approx(
        np.sqrt(gradient**2 * covariance) / jacobian, rel=1e-8
    )


def test_the_bias_total_is_unavailable_when_the_curvature_is_skipped():
    assessment = _scalar_assessment(
        config=ClosureAssessmentConfig(evaluate_jensen=False)
    )
    assert assessment.jensen_bias is None
    assert assessment.closure_bias_total is None
    assert assessment.drift_bias is not None
    assert "Jensen" in assessment.diagnostics["closure_bias_total_status"]


def test_the_shape_fraction_can_be_skipped():
    assessment = _scalar_assessment(
        config=ClosureAssessmentConfig(evaluate_shape=False)
    )
    assert assessment.closure_shape_fraction is None
    assert assessment.closure_displacement is not None


# --- helpers ---------------------------------------------------------------


def test_closure_shape_fraction_separates_amplitude_from_shape():
    radial = closure_shape_fraction(np.diag([1.0, 1.0]), np.diag([1.0, 1.0]))
    pure_shape = closure_shape_fraction(np.diag([1.0, 1.0]), np.diag([1.0, -1.0]))
    mixed = closure_shape_fraction(np.diag([1.0, 0.0]), np.diag([1.0, 1.0]))
    assert radial == pytest.approx(0.0, abs=1e-15)
    assert pure_shape == pytest.approx(1.0, rel=1e-12)
    assert mixed == pytest.approx(np.sqrt(0.5), rel=1e-12)
    # Undefined, not zero, when either side has no norm.
    assert closure_shape_fraction(np.diag([1.0, 1.0]), np.zeros((2, 2))) is None
    assert closure_shape_fraction(np.zeros((2, 2)), np.diag([1.0, 1.0])) is None


def test_closure_contraction_reports_an_unresolved_derivative():
    # A Jacobian that oscillates with the step is not differentiable; the record
    # must say the derivative did not converge instead of averaging it away.
    def jacobian_at(vector):
        x = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 1e-3 * np.sin(1e9 * x)]])

    contraction, record = closure_contraction(
        jacobian_at, [1.0], np.asarray([[1.0]]), relative_step=1e-5
    )
    assert contraction.shape == (1,)
    assert record.repeat_relative_error is None or record.repeat_relative_error > 1e-4


def test_jensen_bias_only_expands_positive_covariance_directions():
    covariance = np.diag([1.0, 0.0])

    def value(vector):
        point = np.asarray(vector, dtype=float)
        return float(point[0] ** 2 + point[1] ** 2)

    bias, step, repeat, expanded = observable_jensen_bias(
        value, [0.0, 0.0], covariance, relative_step=1e-4
    )
    # Only the first direction carries noise: 1/2 * 1 * 2 = 1.
    assert bias == pytest.approx(1.0, rel=1e-6)
    assert step == pytest.approx(1e-4, rel=1e-12)
    assert expanded == 1
    assert repeat is not None


# --- inherited statuses ----------------------------------------------------


def test_a_failed_linear_noise_assessment_is_inherited_without_invented_numbers():
    assessment = assess_closure(
        _ScalarAdapter(),
        _scalar_observable(),
        state=[X0],
        params={},
        diffusion=np.asarray([[-1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert "positive semidefinite" in (assessment.reason or "")
    assert assessment.stochastic.diffusion_is_psd is False
    # No closure number is produced on top of a failed assessment.
    assert assessment.closure_contraction_norm is None
    assert assessment.closure_displacement is None
    assert assessment.drift_bias is None
    assert assessment.jensen_bias is None
    assert assessment.diagnostics["stochastic_status"] == "stochastic_invalid"


def test_an_unsupported_layout_is_uncontrolled():
    adapter = _ScalarAdapter()
    adapter.moment_layout = "classical"
    assessment = assess_closure(
        adapter,
        _scalar_observable(),
        state=[X0],
        params={},
        diffusion=np.asarray([[1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.UNCONTROLLED_APPROXIMATION
    assert assessment.closure_contraction_norm is None


def test_a_nonphysical_state_is_reported_with_its_numbers_kept():
    # x < 0 makes the state matrix indefinite; the readout must stay resolved
    # (a zero gradient would fail earlier, as unidentifiable).
    assessment = _scalar_assessment(x0=-0.25)
    assert assessment.status is CAMOutcomeStatus.NONPHYSICAL
    assert "positive semidefinite" in (assessment.reason or "")
    assert assessment.physical_is_psd is False
    assert assessment.physical_minimum_eigenvalue == pytest.approx(-0.25, rel=1e-12)
    # The numbers are still reported: physicality is a verdict, not an erasure.
    assert assessment.closure_displacement is not None
    assert assessment.drift_bias is not None


def test_an_observable_without_projection_is_unidentifiable():
    observable = FunctionalObservable(
        identity=IDENTITY,
        value_fn=lambda adapter, state, params: 1.0,
        gradient_fn=lambda adapter, state, params: [0.0],
    )
    assessment = assess_closure(
        _ScalarAdapter(),
        observable,
        state=[X0],
        params={},
        diffusion=np.asarray([[1.0]]),
    )
    assert assessment.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert "gradient" in (assessment.reason or "")


# --- annotation builder ----------------------------------------------------


def test_the_annotation_carries_statuses_fields_and_method_metadata():
    valid = _scalar_assessment()
    invalid = assess_closure(
        _ScalarAdapter(),
        _scalar_observable(),
        state=[X0],
        params={},
        diffusion=np.asarray([[-1.0]]),
    )
    product = build_closure_assessment_annotation(
        [valid, invalid], target_identity=["cam:solution:a", "cam:solution:b"]
    )
    assert list(np.asarray(product.handle("status").materialize())) == [
        "valid",
        "stochastic_invalid",
    ]
    np.testing.assert_allclose(
        np.asarray(product.handle("closure_contraction_norm").materialize())[:1],
        [abs(-3.0 * X0 / (2.0 * 1.75))],
        rtol=1e-8,
    )
    # The failed row keeps NaN numbers and its own reason.
    assert np.isnan(
        np.asarray(product.handle("closure_contraction_norm").materialize())[1]
    )
    assert "positive semidefinite" in str(
        np.asarray(product.handle("reason").materialize())[1]
    )
    assert list(np.asarray(product.handle("validity_domain").materialize()))[0] == (
        "weak_noise"
    )
    metadata = np.asarray(product.handle("method_metadata").materialize())
    assert "ito_taylor_second_order" in str(metadata[0])
    assert list(np.asarray(product.handle("stochastic_status").materialize())) == [
        "valid",
        "stochastic_invalid",
    ]


def test_an_empty_product_keeps_the_field_schema():
    product = build_closure_assessment_annotation([], target_identity=[])
    assert product.axis("annotation").size == 0
    fields = {item["name"] for item in product.attributes["field_variables"]}
    assert "closure_contraction_norm" in fields
    assert "validity_domain" in fields
    assert "physical_is_psd" in fields


# --- postprocessor ---------------------------------------------------------


class _ClosureAdapter(_ScalarAdapter):
    """Adapter double used by the postprocessor wiring tests."""

    def jacobian(self, vector, params=None):
        del params
        x = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * x**2]])


class _ClosureModel:
    """Minimal model: a unit diffusion and the parameters a case needs."""

    name = "closure_toy"
    n_modes = 1
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(state, params=None):
        del state, params
        return np.asarray([[1.0]])


def _state_product(values, *, success=None):
    states = np.asarray(values, dtype=complex).reshape(-1, 1, 1, 1)
    scan, capacity = states.shape[0], states.shape[1]
    return build_state_product(
        states=states,
        residuals=np.zeros((scan, capacity)),
        success=(
            np.ones((scan, capacity), dtype=bool)
            if success is None
            else np.asarray(success, dtype=bool).reshape(scan, capacity)
        ),
        valid_mask=np.ones((scan, capacity), dtype=bool),
        solution_count=np.full(scan, capacity, dtype=np.int64),
        parameters={"mu": np.zeros(scan)},
    )


def _candidate_product(values, *, success=True):
    count = len(values)
    return build_candidate_product(
        states=np.asarray(values, dtype=float).reshape(count, 1, 1),
        state_vectors=np.asarray(values, dtype=float).reshape(count, 1),
        control_values=np.empty((count, 0)),
        control_names=(),
        full_residual_norm=np.zeros(count),
        search_residual_norm=np.zeros(count),
        success=np.full(count, success, dtype=bool),
        status=["verified"] * count,
        method=["test"] * count,
        verification_digits=np.full(count, 50, dtype=np.int64),
        verification_status=["verified"] * count,
        candidate_identity=[f"cam:candidate:test{index}" for index in range(count)],
    )


def _context(model, product, role):
    return CAMAnnotationContext(
        model=model,
        metadata={},
        products={role: product},
        source_role=role,
    )


def _patch_adapter(monkeypatch):
    monkeypatch.setattr(
        "qphase_cam.postprocessor.closure_risk.QPSGenDynamicsAdapter.from_model",
        lambda model: _ClosureAdapter(),
    )


def test_the_postprocessor_annotates_a_state_product(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _state_product([X0, 1.0])
    annotation = ClosureRisk(observable=_scalar_observable()).process(
        product, _context(_ClosureModel(), product, "state")
    )
    assert annotation.attributes["annotation_kind"] == "closure_risk"
    np.testing.assert_array_equal(
        np.asarray(annotation.handle("target_identity").materialize()),
        np.asarray(product.handle("solution_identity").materialize()).reshape(-1),
    )
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "valid",
        "valid",
    ]
    # An empty case scope: a state product has no search case.
    assert list(np.asarray(annotation.handle("case_identity").materialize())) == [
        "",
        "",
    ]
    expected = _scalar_assessment()
    np.testing.assert_allclose(
        np.asarray(annotation.handle("drift_bias").materialize())[0],
        expected.drift_bias,
        rtol=1e-8,
    )
    metadata = np.asarray(annotation.handle("method_metadata").materialize())
    assert "assess_closure" in str(metadata[0])
    assert metadata.size == 2


def test_the_postprocessor_annotates_candidates_with_their_case_scope(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _candidate_product([1.0])
    annotation = ClosureRisk(observable=_scalar_observable()).process(
        product, _context(_ClosureModel(), product, "candidates")
    )
    assert annotation.attributes["target_kind"] == "candidate"
    np.testing.assert_array_equal(
        np.asarray(annotation.handle("target_identity").materialize()),
        ["cam:candidate:test0"],
    )
    assert list(np.asarray(annotation.handle("case_identity").materialize())) == [
        "legacy_case:0"
    ]
    assert list(np.asarray(annotation.handle("status").materialize())) == ["valid"]
    # The postprocessor derives the canonical diffusion from the model, so the
    # direct comparison must take the same path.
    direct = assess_closure(
        _ClosureAdapter(),
        _scalar_observable(),
        state=[1.0],
        params={"mu": 0.0},
        model=_ClosureModel(),
    )
    np.testing.assert_allclose(
        np.asarray(annotation.handle("closure_contraction_norm").materialize()),
        [direct.closure_contraction_norm],
        rtol=1e-8,
    )


def test_an_unconverged_target_is_marked_not_converged(monkeypatch):
    _patch_adapter(monkeypatch)
    product = _candidate_product([1.0], success=False)
    annotation = ClosureRisk(observable=_scalar_observable()).process(
        product, _context(_ClosureModel(), product, "candidates")
    )
    assert list(np.asarray(annotation.handle("status").materialize())) == [
        "not_converged"
    ]
    assert "did not converge" in str(
        np.asarray(annotation.handle("reason").materialize())[0]
    )


def test_an_empty_state_product_keeps_the_schema(monkeypatch):
    _patch_adapter(monkeypatch)
    states = np.zeros((1, 1, 1, 1), dtype=complex)
    product = build_state_product(
        states=states,
        residuals=np.zeros((1, 1)),
        success=np.zeros((1, 1), dtype=bool),
        valid_mask=np.zeros((1, 1), dtype=bool),
        solution_count=np.zeros(1, dtype=np.int64),
        parameters={"mu": np.zeros(1)},
    )
    annotation = ClosureRisk(observable=_scalar_observable()).process(
        product, _context(_ClosureModel(), product, "state")
    )
    assert annotation.axis("annotation").size == 0
    fields = {item["name"] for item in annotation.attributes["field_variables"]}
    assert "closure_bias_total" in fields
    assert "validity_domain" in fields
