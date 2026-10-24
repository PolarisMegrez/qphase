"""Phase 10F3c: the Petermann series postprocessor and the standalone audit.

Two layers: the series itself (tracking, per-family statuses, optional scaling fit)
and the audit that every postprocessor runs directly on its typed input product
without a search engine in the loop.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.data import (
    build_annotation_product,
    build_candidate_product,
    build_state_product,
)
from qphase_cam.nonlinear import ScalarReconstructionConfig
from qphase_cam.outcomes import CAMOutcomeStatus
from qphase_cam.postprocessor import (
    CAMFiniteDelayCarrier,
    ClosureRisk,
    CoherencePoleSpectrum,
    HamiltonianSpectrum,
    JacobianSpectrum,
    NonlinearPrediction,
    NonlinearReconstruction,
    ObservableScaling,
    PetermannSeries,
    PetermannSpectrum,
    Physicality,
    RayleighFrequency,
    StochasticValidity,
)
from qphase_cam.postprocessor.base import CAMAnnotationContext
from qphase_cam.postprocessor.local_response import LocalResponseValidation
from qphase_cam.postprocessor.nonlinear_prediction import (
    NonlinearPredictionConfig,
)
from qphase_cam.postprocessor.petermann_series import track_soft_index

from models.kerr_2mode import Kerr2ModeModel

# --- spectrum fixture ------------------------------------------------------


def _spectrum_product(*, rows=5, shift=0.0):
    """Build one petermann_spectrum annotation with analytic factors."""
    spectrum = np.asarray(
        [
            [3.0e-3 * (index + 1) + 0.1j * index, -1.0 - 2.0 * index]
            for index in range(rows)
        ],
        dtype=complex,
    )
    factors = np.asarray(
        [[2.0 * (index + 1) + shift, 1.0 + index] for index in range(rows)], dtype=float
    )
    values = spectrum
    value_factors = factors
    parameters = json.dumps({"omega": 0.5})
    return build_annotation_product(
        annotation_kind="petermann_spectrum",
        target_kind="state",
        target_identity=[f"cam:solution:test{index}" for index in range(rows)],
        status=["valid"] * rows,
        fields={
            "hamiltonian_eigenvalues": values,
            "hamiltonian_petermann_factors": value_factors,
            "bogoliubov_eigenvalues": values,
            "bogoliubov_petermann_factors": value_factors,
            "monodromy_eigenvalues": values,
            "monodromy_petermann_factors": value_factors,
            "cam_jacobian_eigenvalues": values,
            "cam_jacobian_petermann_factors": value_factors,
            "parameters": np.asarray(
                [json.dumps({"omega": 0.5 + 0.25 * index}) for index in range(rows)],
                dtype=str,
            ),
        },
        field_axes={
            name: ("mode",)
            for name in (
                "hamiltonian_eigenvalues",
                "hamiltonian_petermann_factors",
                "bogoliubov_eigenvalues",
                "bogoliubov_petermann_factors",
                "monodromy_eigenvalues",
                "monodromy_petermann_factors",
                "cam_jacobian_eigenvalues",
                "cam_jacobian_petermann_factors",
            )
        },
        axes={"mode": 2},
        method_metadata=[{"parameters_field": parameters} for _ in range(rows)],
    )


def _spectrum_context(product):
    return CAMAnnotationContext(
        products={"petermann_spectrum": product},
        source_role="petermann_spectrum",
    )


def test_the_series_tracks_the_soft_mode_per_family():
    product = _spectrum_product()
    series = PetermannSeries().process(product, _spectrum_context(product))
    assert series.attributes["annotation_kind"] == "petermann_series"
    assert series.attributes["target_kind"] == "case"
    identity = str(np.asarray(series.handle("target_identity").materialize())[0])
    assert identity.startswith("cam:series:")
    np.testing.assert_array_equal(
        np.asarray(series.handle("hamiltonian_soft_index").materialize()),
        [[0, 0, 0, 0, 0]],
    )
    np.testing.assert_allclose(
        np.asarray(series.handle("hamiltonian_soft_petermann").materialize()),
        [[2.0, 4.0, 6.0, 8.0, 10.0]],
    )
    np.testing.assert_allclose(
        np.asarray(series.handle("hamiltonian_soft_eigenvalue").materialize())[0].real,
        [3.0e-3, 6.0e-3, 9.0e-3, 1.2e-2, 1.5e-2],
    )
    for family in ("hamiltonian", "bogoliubov", "monodromy", "cam_jacobian"):
        assert np.asarray(series.handle(f"{family}_status").materialize()).tolist() == [
            ["valid"] * 5
        ]
    # No fit configuration: tracking only, every fit field unavailable.
    assert np.isnan(
        np.asarray(series.handle("hamiltonian_fit_exponent").materialize())[0]
    )
    assert np.asarray(series.handle("parameter_value").materialize()).shape == (1, 5)


def test_a_damped_spectrum_is_unidentifiable():
    values = np.asarray([[-1.0, -2.0]])
    product = build_annotation_product(
        annotation_kind="petermann_spectrum",
        target_kind="state",
        target_identity=["cam:solution:damped"],
        status=["valid"],
        fields={
            "hamiltonian_eigenvalues": values,
            "hamiltonian_petermann_factors": np.asarray([[1.0, 2.0]]),
        },
        field_axes={
            "hamiltonian_eigenvalues": ("mode",),
            "hamiltonian_petermann_factors": ("mode",),
        },
        axes={"mode": 2},
    )
    series = PetermannSeries().process(product, _spectrum_context(product))
    assert np.asarray(series.handle("hamiltonian_soft_index").materialize())[0, 0] == 0
    assert str(np.asarray(series.handle("hamiltonian_status").materialize())[0, 0]) == (
        "unidentifiable"
    )
    assert str(np.asarray(series.handle("status").materialize())[0]) in {
        "unidentifiable",
        "valid",
    }
    assert np.asarray(series.handle("bogoliubov_status").materialize()).tolist() == [
        ["unidentifiable"]
    ]


def test_the_scaling_fit_uses_the_parameter_distance():
    rows = 6
    distances = np.geomspace(1.0e-3, 1.0e-1, rows)
    factors = 0.5 * distances**-0.25
    # The soft seed needs a positive real part, as in the reference scan.
    spectrum = np.stack([np.asarray([[1.0e-3 + 0.0j, -2.0]]) for _ in range(rows)])
    factor_field = np.stack([np.asarray([[1.0, 2.0]]) for _ in range(rows)])
    factor_field[:, 0, 0] = factors
    product = build_annotation_product(
        annotation_kind="petermann_spectrum",
        target_kind="state",
        target_identity=[f"cam:solution:s{index}" for index in range(rows)],
        status=["valid"] * rows,
        fields={
            "hamiltonian_eigenvalues": spectrum.reshape(rows, 2),
            "hamiltonian_petermann_factors": factor_field.reshape(rows, 2),
            "parameters": np.asarray(
                [json.dumps({"omega": 0.5 + float(value)}) for value in distances],
                dtype=str,
            ),
        },
        field_axes={
            "hamiltonian_eigenvalues": ("mode",),
            "hamiltonian_petermann_factors": ("mode",),
        },
        axes={"mode": 2},
    )
    series = PetermannSeries(
        parameter="omega", critical_value=0.5, inner_max=1.0
    ).process(product, _spectrum_context(product))
    exponent = np.asarray(series.handle("hamiltonian_fit_exponent").materialize())[0]
    r_squared = np.asarray(series.handle("hamiltonian_fit_r_squared").materialize())[0]
    assert exponent == pytest.approx(-0.25, rel=1e-9)
    assert r_squared == pytest.approx(1.0, rel=1e-9)
    assert (
        int(np.asarray(series.handle("hamiltonian_fit_point_count").materialize())[0])
        == rows
    )
    # The row order follows the parameter, not the stored order.
    parameter_values = np.asarray(series.handle("parameter_value").materialize())[0]
    np.testing.assert_allclose(parameter_values, [0.5 + value for value in distances])


def test_track_soft_index_seeds_and_tracks():
    spectrum = np.asarray([-3.0, 0.2j, 0.1 + 0.0j])
    assert track_soft_index(spectrum, None) == 2
    assert track_soft_index(spectrum, 0.11 + 0.0j) == 2
    assert track_soft_index(spectrum, 0.0 + 0.21j) == 1
    damped = np.asarray([-5.0, -1.0])
    # No positive real part: the whole spectrum is searched, nearest to zero wins.
    assert track_soft_index(damped, None) == 1
    assert track_soft_index(np.asarray([-2.0, -1.0]), None) == 1
    with pytest.raises(ValueError, match="empty spectrum"):
        track_soft_index(np.asarray([]), None)


# --- standalone audit ------------------------------------------------------

KERR = Kerr2ModeModel(
    omega_a=0.0, omega_b=-0.1, chi=0.01, gamma_a=0.5, gamma_b=1.0, g=0.5
)


def _state_product():
    states = np.asarray([[[2.0, 0.1j], [-0.1j, 1.0]]], dtype=complex).reshape(
        1, 1, 2, 2
    )
    return build_state_product(
        states=states,
        residuals=np.zeros((1, 1)),
        success=np.ones((1, 1), dtype=bool),
        valid_mask=np.ones((1, 1), dtype=bool),
        solution_count=np.ones(1, dtype=np.int64),
        parameters=dict(KERR.params),
    )


def _fold_candidates():
    """Build a real two-mode fold candidate product for the local-response audit."""
    from qphase_cam.solver.bifurcation import ControlRange
    from qphase_cam.solver.execution import candidate_product_from_output

    from tests.qphase_cam.test_bifurcation import TwoModeFoldModel, _solver

    model = TwoModeFoldModel()
    output = _solver(2, {"gamma": ControlRange(min=-1.0, max=0.0)}).search(
        model, NumpyBackend()
    )
    return candidate_product_from_output(model, output)


def _candidate_product():
    """Build a one-candidate product with the diagnostics local_response needs."""
    from tests.qphase_cam.test_bifurcation import _branch_signature

    return build_candidate_product(
        states=np.asarray([[[1.0 + 0.0j]]]),
        state_vectors=np.asarray([[1.0]]),
        control_values=np.empty((1, 0)),
        control_names=(),
        full_residual_norm=np.zeros(1),
        search_residual_norm=np.zeros(1),
        success=np.ones(1, dtype=bool),
        status=["verified"],
        method=["test"],
        verification_digits=np.full(1, 50, dtype=np.int64),
        verification_status=["verified"],
        candidate_identity=["cam:candidate:standalone"],
        classification=[
            {
                "scaling_signatures": [_branch_signature()],
                "state_tangent_vector": [1.0],
            }
        ],
    )


def _state_context(products=None):
    return CAMAnnotationContext(
        model=KERR,
        backend=NumpyBackend(),
        products=products or {"state": _state_product()},
        source_role="state",
    )


def _candidate_context(product=None):
    product = product if product is not None else _candidate_product()
    return CAMAnnotationContext(
        model=KERR,
        metadata={"input_artifact_id": "artifact-standalone"},
        products={"candidates": product},
        source_role="candidates",
    )


@pytest.mark.parametrize(
    "processor",
    [
        RayleighFrequency(),
        HamiltonianSpectrum(),
        JacobianSpectrum(),
        Physicality(),
        PetermannSpectrum(),
        CoherencePoleSpectrum(),
        CAMFiniteDelayCarrier(),
    ],
    ids=lambda item: item.name,
)
def test_state_postprocessors_run_standalone(processor):
    product = _state_product()
    annotation = processor.process(product, _state_context({"state": product}))
    assert annotation.attributes["annotation_kind"] == processor.annotation_kind
    assert annotation.axis("annotation").size == 1


class _CandidateAdapter:
    moment_layout = "normal"

    @staticmethod
    def state_matrix(state, params=None):
        del params
        return np.asarray([[state[0]]], dtype=complex)

    @staticmethod
    def jacobian(state, params=None):
        del state, params
        return np.asarray([[0.0]])

    @staticmethod
    def parameter_jacobian(state, params=None):
        del state, params
        return np.asarray([[1.0]])

    @staticmethod
    def parameter_direction(name, scale=1.0):
        del name
        return np.asarray([scale])

    @staticmethod
    def closure_provenance():
        return {"representation": "wigner", "fpe_is_exact": False}


class _CandidateModel:
    name = "candidate_toy"
    n_modes = 1
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(state, params=None):
        del state, params
        return np.asarray([[1.0]])

    @staticmethod
    def cam_hamiltonian(state, params=None):
        del params
        return np.asarray(state, dtype=complex)


def _fold_context(product):
    from tests.qphase_cam.test_bifurcation import TwoModeFoldModel

    return CAMAnnotationContext(
        model=TwoModeFoldModel(),
        backend=NumpyBackend(),
        metadata={
            "perturbation_parameter": "gamma",
            "perturbation_scale": 1.0,
            "fixed_params": {},
            "input_artifact_id": "artifact-standalone",
        },
        products={"candidates": product},
        source_role="candidates",
    )


def test_local_response_runs_standalone_on_a_real_candidate_product():
    product = _fold_candidates()
    processor = LocalResponseValidation(
        epsilon_min=1.0e-10,
        epsilon_max=1.0e-6,
        epsilon_points=4,
        fit_points=3,
        residual_tolerance=1.0e-25,
    )
    annotation = processor.process(product, _fold_context(product))
    assert annotation.attributes["annotation_kind"] == "local_response_validation"
    assert annotation.axis("annotation").size >= 1


def test_stochastic_validity_runs_standalone_on_a_one_mode_candidate(monkeypatch):
    monkeypatch.setattr(
        "qphase_cam.postprocessor.stochastic_validity.QPSGenDynamicsAdapter.from_model",
        lambda model: _CandidateAdapter(),
    )
    product = _candidate_product()
    processor = StochasticValidity()
    annotation = processor.process(
        product,
        CAMAnnotationContext(
            model=_CandidateModel(),
            metadata={
                "perturbation_parameter": "mu",
                "perturbation_scale": 1.0,
                "input_artifact_id": "artifact-standalone",
            },
            products={"candidates": product},
            source_role="candidates",
        ),
    )
    assert annotation.attributes["annotation_kind"] == "stochastic_validity"
    assert annotation.axis("annotation").size == 1


class _ToyAdapter:
    """One-mode adapter the reconstruction-based processors can use standalone."""

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
        return np.asarray([[1.7]])

    def parameter_direction(self, name, scale=1.0):
        del name
        return np.asarray([scale])


class _ToyModel:
    name = "toy"
    n_modes = 1
    params = {"mu": 0.0}

    @staticmethod
    def cam_diffusion(matrix, params=None):
        del params
        return np.ones_like(np.asarray(matrix), dtype=complex)

    @staticmethod
    def cam_hamiltonian(matrix, params=None):
        del params
        return np.asarray(matrix, dtype=complex)


def _toy_product():
    states = np.zeros((3, 1, 1, 1), dtype=complex)
    states[:, 0, 0, 0] = [0.4, 0.5, 0.6]
    return build_state_product(
        states=states,
        residuals=np.zeros((3, 1)),
        success=np.ones((3, 1), dtype=bool),
        valid_mask=np.ones((3, 1), dtype=bool),
        solution_count=np.ones(3, dtype=np.int64),
        parameters={"mu": np.asarray([0.4, 0.5, 0.6])},
    )


@pytest.mark.parametrize(
    "processor",
    [
        ClosureRisk(),
        NonlinearReconstruction(
            orders=(3,),
            reconstruction=ScalarReconstructionConfig(grid_points=201),
        ),
        NonlinearPrediction(
            NonlinearPredictionConfig(
                orders=(3,),
                perturbation_parameter="mu",
                reconstruction=ScalarReconstructionConfig(grid_points=201),
            )
        ),
    ],
    ids=lambda item: f"{item.name}-state",
)
def test_new_postprocessors_run_standalone_on_state(processor, monkeypatch):
    for module in (
        "closure_risk",
        "observable_scaling",
        "nonlinear_reconstruction",
        "nonlinear_prediction",
    ):
        monkeypatch.setattr(
            f"qphase_cam.postprocessor.{module}.QPSGenDynamicsAdapter.from_model",
            lambda model: _ToyAdapter(),
        )
    product = _toy_product()
    context = CAMAnnotationContext(
        model=_ToyModel(),
        metadata={"input_artifact_id": "artifact-standalone"},
        products={"state": product},
        source_role="state",
    )
    annotation = processor.process(product, context)
    if processor.name == "nonlinear_prediction":
        assert annotation.attributes["product"] == "prediction"
        return
    assert annotation.attributes["annotation_kind"] == processor.annotation_kind


def test_the_series_runs_standalone_on_a_spectrum_product():
    spectrum = PetermannSpectrum().process(_state_product(), _state_context())
    series = PetermannSeries().process(
        spectrum,
        CAMAnnotationContext(
            products={"petermann_spectrum": spectrum},
            source_role="petermann_spectrum",
        ),
    )
    assert series.attributes["annotation_kind"] == "petermann_series"
    # The real Jacobian family is tracked too, and the fit is not requested.
    assert np.asarray(series.handle("cam_jacobian_soft_index").materialize()).shape == (
        1,
        1,
    )


@pytest.mark.parametrize(
    ("processor", "context"),
    [
        (PetermannSeries(), _state_context()),
        (PetermannSpectrum(), _candidate_context()),
        (RayleighFrequency(), _candidate_context()),
    ],
    ids=["series-needs-spectrum", "spectrum-needs-state", "frequency-needs-state"],
)
def test_an_unaccepted_role_is_rejected_by_the_engine_contract(processor, context):
    """A role the processor does not accept has no source product."""
    accepted = tuple(processor.accepted_product_roles)
    available = tuple(context.products)
    assert not set(accepted) & set(available)


def test_requirements_are_not_relaxed_for_standalone_use():
    assert PetermannSeries.requires_model is False
    assert PetermannSeries.requires_backend is False
    for processor in (
        RayleighFrequency(),
        HamiltonianSpectrum(),
        JacobianSpectrum(),
        Physicality(),
        PetermannSpectrum(),
        CoherencePoleSpectrum(),
        CAMFiniteDelayCarrier(),
        LocalResponseValidation(),
        StochasticValidity(),
        ClosureRisk(),
        ObservableScaling(parameter="mu"),
        NonlinearReconstruction(),
        NonlinearPrediction(perturbation_parameter="mu"),
    ):
        assert processor.accepted_product_roles, processor.name
    # The summarize task forbids a model: only providers that need neither a model
    # nor a backend can annotate there, which is a task rule, not a plugin one.
    assert CAMOutcomeStatus.VALID.value == "valid"
