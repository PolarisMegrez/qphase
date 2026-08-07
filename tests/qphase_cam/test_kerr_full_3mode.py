"""CAM consistency tests for the fully connected three-mode Kerr model."""

from __future__ import annotations

import numpy as np
from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.core.jacobian import SymbolicJacobian, central_difference_jacobian
from qphase_cam.core.liouvillian import model_liouvillian, residual_vector

from models.kerr_3mode import Kerr3ModeModel
from models.kerr_full_3mode import KerrFull3ModeModel


def _params(g_bc: float) -> dict[str, float]:
    return {
        "omega_a": 0.0,
        "omega_b": -0.7,
        "omega_c": 0.4,
        "chi": 0.003,
        "gamma_a": 1.0,
        "gamma_b": 0.8,
        "gamma_c": 0.6,
        "g_ab": 0.5,
        "g_ac": 1.2,
        "g_bc": g_bc,
    }


def _state() -> np.ndarray:
    rng = np.random.default_rng(73)
    raw = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    return raw @ raw.conj().T


def test_vector_fast_paths_match_matrix_equations():
    model = KerrFull3ModeModel(**_params(0.35))
    state = _state()
    states = np.stack((state, 2.5 * state))
    vectors = matrix_to_vector(states)

    np.testing.assert_allclose(
        model.cam_residual_vector(vectors, model.params),
        matrix_to_vector(model_liouvillian(model, states, model.params)),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        model.cam_jacobian_vector(vectors, model.params),
        model.cam_jacobian(states, model.params),
        atol=1e-12,
    )


def test_analytic_symbolic_and_finite_difference_jacobians_agree():
    model = KerrFull3ModeModel(**_params(0.35))
    state = _state()
    vector = matrix_to_vector(state)
    analytic = model.cam_jacobian(state, model.params)
    symbolic = SymbolicJacobian(model.cam_symbolic_matrices(), 3, "numpy")(
        state, model.params
    )
    numerical = central_difference_jacobian(
        lambda value: residual_vector(model, vector_to_matrix(value, 3), model.params),
        vector,
        1e-6,
    )

    np.testing.assert_allclose(analytic, symbolic, atol=1e-11)
    np.testing.assert_allclose(symbolic, numerical, atol=3e-8)


def test_zero_bc_coupling_reduces_to_kerr_3mode():
    full_params = _params(0.0)
    old_params = {name: value for name, value in full_params.items() if name != "g_bc"}
    full = KerrFull3ModeModel(**full_params)
    old = Kerr3ModeModel(**old_params)
    state = _state()
    vector = matrix_to_vector(state)
    amplitudes = np.asarray([[1.0 + 0.2j, -0.4 + 0.5j, 0.3 - 0.8j]])

    np.testing.assert_allclose(
        full.drift(amplitudes, 0.0, full.params),
        old.drift(amplitudes, 0.0, old.params),
    )
    np.testing.assert_allclose(
        full.cam_residual_vector(vector, full.params),
        old.cam_residual_vector(vector, old.params),
    )
    np.testing.assert_allclose(
        full.cam_jacobian_vector(vector, full.params),
        old.cam_jacobian_vector(vector, old.params),
    )


def test_fpgen_dynamics_match_model_fast_paths():
    model = KerrFull3ModeModel(**_params(0.35))
    adapter = FPGenDynamicsAdapter.from_model(model)
    vector = matrix_to_vector(_state())

    assert adapter.parameter_domains["g_bc"] == "nonnegative"
    np.testing.assert_allclose(
        adapter.rhs(vector), model.cam_residual_vector(vector, model.params), atol=1e-12
    )
    np.testing.assert_allclose(
        adapter.jacobian(vector),
        model.cam_jacobian_vector(vector, model.params),
        atol=1e-12,
    )
