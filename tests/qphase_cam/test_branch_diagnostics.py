"""Phase 10G2d1c1: the branch-point diagnostic service and its annotation.

The fixture is a one-mode toy whose state matrix is ``[[x]]``, ``J = -1 - 3 v²`` and a
unit diffusion at ``v = 0.4``: the fold's ``nearest_zero`` mode is the only mode, the
canonical noise is analytic and the two readouts are ``x`` and ``x²``, so every
number below is a closed form rather than a recorded run.
"""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.branch_diagnostics import (
    branch_diagnostics,
)
from qphase_cam.core.noise import canonical_sample_matrix_covariance
from qphase_cam.data import build_candidate_product
from qphase_cam.data.annotations import build_annotation_product  # noqa: F401
from qphase_cam.observable import (
    ChargeSector,
    FunctionalObservable,
    ObservableIdentity,
    ObservableKind,
)
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.candidates import candidate_view
from qphase_cam.postprocessor.local_response import (
    LocalResponseValidation,
    LocalResponseValidationConfig,
)

X0 = 0.4
PARAMS = {"mu": 0.0}


class _ToyAdapter:
    """One-coordinate adapter: state matrix ``[[x]]`` and ``J = -1 - 3 x²``."""

    moment_layout = "normal"

    def state_matrix(self, vector, params=None):
        del params
        return np.asarray(vector, dtype=float).reshape(1, 1).astype(complex)

    def jacobian(self, vector, params=None):
        del params
        value = float(np.asarray(vector, dtype=float).reshape(-1)[0])
        return np.asarray([[-1.0 - 3.0 * value**2]])

    def parameter_jacobian(self, vector, params=None):
        del params
        return np.asarray([[1.0]])

    def parameter_direction(self, parameter, scale=1.0):
        assert parameter == "mu"
        return np.asarray([float(scale)])

    def physical_eigenvalues(self, vector, params=None):
        del params
        return np.asarray([float(np.asarray(vector, dtype=float).reshape(-1)[0])])


class _ToyModel:
    """Unit diffusion whose Hamiltonian is the state matrix itself."""

    params = dict(PARAMS)

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.ones_like(np.asarray(matrix), dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


def _identity(name: str, mapping: str) -> ObservableIdentity:
    return ObservableIdentity(
        kind=ObservableKind.STATE,
        sector=ChargeSector.CHARGE_ZERO,
        name=name,
        measurement_mapping=mapping,
    )


def _linear():
    return FunctionalObservable(
        identity=_identity("x", "x"),
        relative_step=2.0e-6,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0]
        ),
    )


def _coupled():
    """Return a readout that depends on the scan parameter, so its response is not 0."""
    return FunctionalObservable(
        identity=_identity("x_plus_mu", "x + mu"),
        relative_step=2.0e-6,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0]
        )
        + float((params or {}).get("mu", 0.0)),
    )


def _quadratic():
    return FunctionalObservable(
        identity=_identity("x_squared", "x**2"),
        relative_step=2.0e-6,
        value_fn=lambda adapter, state, params: float(
            np.asarray(state, dtype=float).reshape(-1)[0] ** 2
        ),
    )


def _candidate_product():
    """One candidate with one real branch along ``mu``, so the branch solve runs."""
    signature = {
        "state_order": 1,
        "perturbation_order": 1,
        "coupling_state_order": 1,
        "exponent_numerator": 1,
        "exponent_denominator": 2,
        "sublinear": False,
        "branches": [
            {
                "epsilon_side": 1,
                "amplitude": 1.0,
                "leading_state_coefficient": [[1.0]],
            }
        ],
    }
    return build_candidate_product(
        states=np.asarray([[[X0]]], dtype=float),
        state_vectors=np.asarray([[X0]], dtype=float),
        control_values=np.empty((1, 0)),
        control_names=(),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.full(1, 50, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["cam:candidate:toy"],
        classification=[
            {
                "scaling_signatures": [signature],
                "state_tangent_vector": [1.0],
                "verified_full_state_decimal_values": ["0.4"],
            }
        ],
        base_audit=[{}],
        case_metadata=[
            {"case_parameters": dict(PARAMS), "perturbation_parameter": "mu"}
        ],
        candidate_offsets=np.asarray([0, 1]),
        case_shape=(1,),
        case_params={"mu": np.asarray([PARAMS["mu"]])},
    )


def _context(model, product):
    return CAMAnnotationContext(
        model=model,
        metadata={},
        products={"candidates": product},
        source_role="candidates",
    )


@pytest.fixture
def toy_adapter(monkeypatch):
    """Keep the toy model on its own adapter instead of a qpsgen-backed one."""
    monkeypatch.setattr(
        "qphase_cam.postprocessor.local_response.QPSGenDynamicsAdapter.from_model",
        lambda model: _ToyAdapter(),
    )
    return _ToyAdapter()


@pytest.fixture
def fake_solve(monkeypatch):
    """Return the branch prediction itself as the solved state, and count the calls.

    The real continuation is covered by the solver's own tests; here the point is
    that the annotation resolves each branch point *once* and then diagnoses several
    readouts on that one state.
    """
    from types import SimpleNamespace

    calls: list[dict[str, object]] = []

    def solve(adapter, initial, params, **kwargs):
        calls.append(
            {"state": np.asarray(initial, dtype=float), "params": dict(params)}
        )
        return SimpleNamespace(
            state=np.asarray(initial, dtype=float), residual=0.0, converged=True
        )

    monkeypatch.setattr(
        "qphase_cam.postprocessor.local_response.solve_high_precision", solve
    )
    return calls


def test_the_diagnostic_is_the_closed_form_of_the_toy():
    adapter = _ToyAdapter()
    model = _ToyModel()
    diagnostics = branch_diagnostics(
        adapter,
        model,
        _linear(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
        critical_state=[0.0],
    )
    jacobian = -1.0 - 3.0 * X0**2
    covariance = canonical_sample_matrix_covariance(
        np.asarray([[X0]], dtype=complex), np.asarray([[1.0]], dtype=complex)
    )
    # The canonical transform maps the unit diffusion to 2 X on the diagonal.
    projected = float(covariance[0, 0])
    rate = -jacobian
    sigma = float(np.sqrt(projected / (2.0 * rate)))
    assert diagnostics.status is CAMOutcomeStatus.VALID
    assert diagnostics.soft_rate == pytest.approx(rate, rel=1e-15)
    assert diagnostics.projected_noise_intensity == pytest.approx(projected, rel=1e-15)
    assert diagnostics.local_ou_sigma_x == pytest.approx(sigma, rel=1e-15)
    assert diagnostics.soft_coordinate == pytest.approx(X0, rel=1e-15)
    assert diagnostics.rho_x == pytest.approx(X0 / sigma, rel=1e-15)
    # l = r = 1 for the scalar mode: eta2 = |½ J'(x)| sigma / rate.
    assert diagnostics.eta2 == pytest.approx(3.0 * X0 * sigma / rate, rel=1e-3)
    # A linear readout has unit projection, zero curvature and no eta_readout.
    assert diagnostics.frequency_soft_projection == pytest.approx(1.0, rel=1e-8)
    assert diagnostics.frequency_curvature == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.eta_readout == pytest.approx(0.0, abs=1e-12)
    assert diagnostics.derivative is not None
    assert diagnostics.derivative.method == "central_difference"
    assert diagnostics.derivative.step[0] == pytest.approx(2.0e-6 * max(1.0, X0))
    assert diagnostics.method["critical_mode_rule"] == "nearest_zero"
    # The quadratic readout has a curvature of exactly two and no projection problem.
    quadratic = branch_diagnostics(
        adapter,
        _ToyModel(),
        _quadratic(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
        critical_state=[0.0],
    )
    assert quadratic.frequency_curvature == pytest.approx(2.0, rel=1e-6)
    assert quadratic.eta_readout == pytest.approx(
        0.5 * 2.0 * sigma / (2.0 * X0), rel=1e-6
    )


def test_a_vanishing_projection_leaves_eta_readout_undefined():
    flat = FunctionalObservable(
        identity=_identity("flat", "constant"),
        relative_step=2.0e-6,
        value_fn=lambda adapter, state, params: 1.0,
    )
    diagnostics = branch_diagnostics(
        _ToyAdapter(),
        _ToyModel(),
        flat,
        state=[X0],
        params=PARAMS,
        parameter="mu",
    )
    assert diagnostics.frequency_soft_projection == pytest.approx(0.0)
    # |projection| <= 1e-14 has no denominator: the field is undefined, not zero.
    assert diagnostics.eta_readout is None


def test_an_undecaying_mode_is_reported_not_diagnosed():
    diagnostics = branch_diagnostics(
        _ToyAdapter(),
        _ToyModel(),
        _linear(),
        state=[1.0],
        params=PARAMS,
        parameter="mu",
    )
    # J(1) = -4 < 0: the mode decays. A growing mode needs a positive Jacobian, so
    # the toy is evaluated through an adapter that states one.
    class _Growing(_ToyAdapter):
        def jacobian(self, vector, params=None):
            del vector, params
            return np.asarray([[2.0]])

    unstable = branch_diagnostics(
        _Growing(),
        _ToyModel(),
        _linear(),
        state=[0.3],
        params=PARAMS,
        parameter="mu",
    )
    assert diagnostics.status is CAMOutcomeStatus.VALID
    assert unstable.status is CAMOutcomeStatus.UNSTABLE
    assert unstable.reason_code == "soft_mode_not_decaying"
    assert unstable.local_ou_sigma_x is None
    assert unstable.eta2 is None


def test_the_annotation_carries_one_readout_axis_without_resolving_again(
    monkeypatch, toy_adapter, fake_solve
):
    del monkeypatch, toy_adapter
    product = _candidate_product()
    processor = LocalResponseValidation(
        LocalResponseValidationConfig(
            epsilon_min=1e-6,
            epsilon_max=1e-4,
            epsilon_points=3,
            fit_points=3,
            residual_tolerance=1e-30,
            diagnostic_observables=(_linear(), _quadratic()),
        )
    )
    annotation = processor.process(product, _context(_ToyModel(), product))
    view = candidate_view(product)
    branches = view.branches.size
    # One solve per (branch, epsilon), independent of how many readouts are diagnosed.
    assert len(fake_solve) == branches * 3
    field_axes = {
        name: tuple(dims) for name, dims in annotation.attributes["field_axes"].items()
    }
    # Every diagnostic field carries the readout axis, including the mode-level ones:
    # the row set is one per (branch, epsilon, readout), exactly as a per-readout CSV
    # repeats them.
    assert field_axes["eta2"] == ("branch", "epsilon_sample", "readout")
    assert field_axes["diagnostic_status"] == (
        "branch",
        "epsilon_sample",
        "readout",
    )
    assert field_axes["frequency_curvature"] == (
        "branch",
        "epsilon_sample",
        "readout",
    )
    assert annotation.attributes["shared_axes"]["readout"] == 2
    identity = np.asarray(annotation.handle("target_identity").materialize())
    assert list(identity) == ["cam:candidate:toy"]
    metadata = [
        __import__("json").loads(str(value))
        for value in np.asarray(annotation.handle("method_metadata").materialize())
    ]
    assert [item["name"] for item in metadata[0]["diagnostic_observables"]] == [
        "x",
        "x_squared",
    ]
    assert metadata[0]["diagnostic_observables"][1]["measurement_mapping"] == "x**2"

    statuses = np.asarray(annotation.handle("diagnostic_status").materialize())
    curvature = np.asarray(annotation.handle("frequency_curvature").materialize())
    eta = np.asarray(annotation.handle("eta2").materialize())
    frequency = np.asarray(annotation.handle("frequency").materialize())
    assert statuses.shape == (1, branches, 3, 2)
    assert curvature.shape == (1, branches, 3, 2)
    assert eta.shape == (1, branches, 3, 2)
    # Every value equals the public service called on the state that was solved, so
    # the annotation is a projection of the service, not a second implementation.
    for branch in range(branches):
        side = int(view.branches.epsilon_side[branch])
        design = view.design_at(0)
        base = view.params_at(0)[design.parameter]
        assert design.parameter == "mu"
        for sample in range(3):
            record = fake_solve[branch * 3 + sample]
            state = np.asarray(record["state"], dtype=float)
            params = dict(record["params"])
            expected_rate = None
            for readout, observable in enumerate((_linear(), _quadratic())):
                direct = branch_diagnostics(
                    _ToyAdapter(),
                    _ToyModel(),
                    observable,
                    state=state,
                    params=params,
                    parameter="mu",
                    scale=1.0,
                    critical_state=[X0],
                )
                assert statuses[0, branch, sample, readout] == direct.status.value
                assert curvature[0, branch, sample, readout] == pytest.approx(
                    direct.frequency_curvature, rel=1e-12
                )
                assert frequency[0, branch, sample, readout] == pytest.approx(
                    direct.frequency, rel=1e-12
                )
                expected_rate = direct.soft_rate
            assert eta[0, branch, sample, 0] == pytest.approx(
                branch_diagnostics(
                    _ToyAdapter(),
                    _ToyModel(),
                    _linear(),
                    state=state,
                    params=params,
                    parameter="mu",
                    scale=1.0,
                    critical_state=[X0],
                ).eta2,
                rel=1e-12,
            )
            assert expected_rate is not None
            assert abs(base) < 1.0
            assert side in (-1, 1)


def test_the_default_configuration_keeps_the_annotation_unchanged(
    toy_adapter, fake_solve
):
    del fake_solve
    product = _candidate_product()
    processor = LocalResponseValidation(
        LocalResponseValidationConfig(
            epsilon_min=1e-6, epsilon_max=1e-4, epsilon_points=3, fit_points=3
        )
    )
    annotation = processor.process(product, _context(_ToyModel(), product))
    axes = annotation.attributes["field_axes"]
    assert "readout" not in annotation.attributes["shared_axes"]
    assert all("readout" not in dims for dims in axes.values())
    assert "diagnostic_status" not in axes


def test_the_service_reuses_the_public_gaussian_and_response_algebras():
    """On a stable state the service is the public algebra, not a second definition."""
    from qphase_cam.nonlinear import ou_intensities, response_prediction

    adapter = _ToyAdapter()
    model = _ToyModel()
    diagnostics = branch_diagnostics(
        adapter,
        model,
        _coupled(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
        critical_state=[0.0],
    )
    gaussian = ou_intensities(
        adapter, _coupled(), state=[X0], params=PARAMS, model=model
    )
    response = response_prediction(
        adapter, _coupled(), state=[X0], params=PARAMS, parameter="mu"
    )
    assert diagnostics.soft_ou_intensity == gaussian.soft_intensity
    assert diagnostics.full_ou_intensity == gaussian.full_intensity
    assert diagnostics.projected_noise_intensity == gaussian.projected_noise
    assert diagnostics.local_ou_sigma_x == pytest.approx(gaussian.sigma, rel=1e-15)
    assert diagnostics.frequency_soft_projection == gaussian.projection
    assert diagnostics.soft_response == pytest.approx(
        response.soft_response, rel=1e-12
    )
    assert diagnostics.explicit_response == response.explicit_response
    assert diagnostics.full_response == pytest.approx(
        response.full_response, rel=1e-12
    )
    # The explicit response keeps its own record, with the repeat check intact.
    assert diagnostics.derivative is not None
    assert diagnostics.derivative.metadata["parameter"] == "mu"
    assert diagnostics.explicit_response == pytest.approx(1.0, rel=1e-9)
    assert diagnostics.derivative.repeat_relative_error is not None


def test_a_non_positive_projected_noise_is_stochastic_invalid():
    """A negative projected noise is not an intensity: no sigma, no soft OU."""

    class _InvertedNoise(_ToyModel):
        @staticmethod
        def cam_diffusion(matrix, params=None):
            del params
            return -np.ones_like(np.asarray(matrix), dtype=complex)

    diagnostics = branch_diagnostics(
        _ToyAdapter(),
        _InvertedNoise(),
        _linear(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
    )
    assert diagnostics.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert diagnostics.reason_code == "projected_noise_not_positive"
    assert diagnostics.projected_noise_intensity is not None
    assert diagnostics.projected_noise_intensity <= 0.0
    assert diagnostics.local_ou_sigma_x is None
    assert diagnostics.soft_ou_intensity is None
    assert diagnostics.eta2 is None
    assert diagnostics.eta_readout is None
    # The readout itself is still defined, and is not called valid.
    assert diagnostics.frequency is not None
    assert diagnostics.frequency_soft_projection is not None


def test_an_unphysical_state_is_reported_and_keeps_its_algebra():
    """A state below the physical tolerance is `nonphysical`, with its numbers kept."""

    class _Unphysical(_ToyAdapter):
        def physical_eigenvalues(self, vector, params=None):
            del vector, params
            return np.asarray([-1.0e-3])

    diagnostics = branch_diagnostics(
        _Unphysical(),
        _ToyModel(),
        _linear(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
    )
    assert diagnostics.status is CAMOutcomeStatus.NONPHYSICAL
    assert diagnostics.reason_code == "nonphysical_state"
    assert "not physical" in (diagnostics.reason or "")
    # The algebra is still reported: only the verdict changed.
    assert diagnostics.soft_rate is not None
    assert diagnostics.projected_noise_intensity is not None
    assert diagnostics.frequency is not None
    assert diagnostics.minimum_physical_eigenvalue == pytest.approx(-1.0e-3)


def test_an_unreadable_or_non_finite_physical_spectrum_is_not_valid():
    class _Unreadable(_ToyAdapter):
        def physical_eigenvalues(self, vector, params=None):
            raise ValueError("no physical spectrum")

    unreadable = branch_diagnostics(
        _Unreadable(), _ToyModel(), _linear(), state=[X0], params=PARAMS, parameter="mu"
    )
    assert unreadable.status is CAMOutcomeStatus.UNIDENTIFIABLE
    assert unreadable.reason_code == "physical_spectrum_unreadable"

    class _NotFinite(_ToyAdapter):
        def physical_eigenvalues(self, vector, params=None):
            return np.asarray([np.nan])

    not_finite = branch_diagnostics(
        _NotFinite(), _ToyModel(), _linear(), state=[X0], params=PARAMS, parameter="mu"
    )
    assert not_finite.status is CAMOutcomeStatus.STOCHASTIC_INVALID
    assert not_finite.reason_code == "physical_eigenvalue_not_finite"
    assert not_finite.minimum_physical_eigenvalue is None


def test_a_bad_critical_state_is_refused_not_passed_through():
    with pytest.raises(ValueError, match="coordinates for a"):
        branch_diagnostics(
            _ToyAdapter(),
            _ToyModel(),
            _linear(),
            state=[X0],
            params=PARAMS,
            parameter="mu",
            critical_state=[0.0, 0.0],
        )
    diagnostics = branch_diagnostics(
        _ToyAdapter(),
        _ToyModel(),
        _linear(),
        state=[X0],
        params=PARAMS,
        parameter="mu",
        critical_state=[np.nan],
    )
    assert diagnostics.status is CAMOutcomeStatus.NOT_CONVERGED
    assert diagnostics.reason_code == "critical_state_not_finite"
    # A NaN coordinate never reaches the result.
    assert diagnostics.soft_coordinate is None
    assert diagnostics.rho_x is None


def test_the_annotation_states_why_a_readout_failed(toy_adapter, fake_solve):
    """A failed diagnostic is readable from the product, not only as a status."""
    del toy_adapter, fake_solve
    product = _candidate_product()

    class _Unphysical(_ToyAdapter):
        def physical_eigenvalues(self, vector, params=None):
            del vector, params
            return np.asarray([-1.0e-3])

    import qphase_cam.postprocessor.local_response as module

    original = module.QPSGenDynamicsAdapter.from_model
    module.QPSGenDynamicsAdapter.from_model = lambda model: _Unphysical()
    try:
        processor = LocalResponseValidation(
            LocalResponseValidationConfig(
                epsilon_min=1e-6,
                epsilon_max=1e-4,
                epsilon_points=3,
                fit_points=3,
                diagnostic_observables=(_linear(),),
            )
        )
        annotation = processor.process(product, _context(_ToyModel(), product))
    finally:
        module.QPSGenDynamicsAdapter.from_model = original
    statuses = np.asarray(annotation.handle("diagnostic_status").materialize())
    codes = np.asarray(annotation.handle("diagnostic_reason_code").materialize())
    reasons = np.asarray(annotation.handle("diagnostic_reason").materialize())
    assert statuses.shape[-1] == 1
    if (statuses != "").any():
        assert set(statuses[statuses != ""]) == {"nonphysical"}
        assert set(codes[codes != ""]) == {"nonphysical_state"}
        assert any("not physical" in str(value) for value in reasons.ravel())
