"""CAM Jacobian source and accuracy tests."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase_cam.core.coordinates import matrix_to_vector, vector_to_matrix
from qphase_cam.core.jacobian import (
    JacobianResolver,
    SymbolicJacobian,
    central_difference_jacobian,
)
from qphase_cam.core.liouvillian import model_liouvillian, residual_vector
from qphase_cam.errors import JacobianUnavailableError

from models.kerr_2mode import Kerr2ModeModel
from models.kerr_3mode import Kerr3ModeModel
from models.vdp_2mode import VDP2ModeModel


def _state(n_modes: int):
    rng = np.random.default_rng(n_modes + 20)
    raw = rng.normal(size=(n_modes, n_modes)) + 1j * rng.normal(
        size=(n_modes, n_modes)
    )
    return raw @ raw.conj().T


@pytest.mark.parametrize(
    "model",
    [
        VDP2ModeModel(
            omega_a=0.2,
            omega_b=-0.1,
            gamma_a=2.0,
            gamma_b=1.0,
            Gamma=0.01,
            g=0.5,
        ),
        Kerr2ModeModel(
            omega_a=0.0,
            omega_b=-0.01,
            chi=0.01,
            gamma_a=0.5,
            gamma_b=1.8728,
            g=0.5,
        ),
    ],
)
def test_analytic_symbolic_and_finite_difference_agree(model):
    state = _state(model.n_modes)
    symbolic = SymbolicJacobian(
        model.cam_symbolic_matrices(), model.n_modes, "numpy"
    )(state, model.params)
    analytic = model.cam_jacobian(state, model.params)
    vector = matrix_to_vector(state)
    numerical = central_difference_jacobian(
        lambda value: residual_vector(
            model, vector_to_matrix(value, model.n_modes), model.params
        ),
        vector,
        1e-6,
    )
    np.testing.assert_allclose(analytic, symbolic, atol=1e-11)
    np.testing.assert_allclose(symbolic, numerical, atol=2e-8)


def test_three_mode_symbolic_matches_finite_difference():
    model = Kerr3ModeModel(
        omega_a=0.0,
        omega_b=-0.1,
        omega_c=0.2,
        chi=0.01,
        gamma_a=0.5,
        gamma_b=1.0,
        gamma_c=0.4,
        g_ab=0.5,
        g_ac=0.3,
    )
    state = _state(3)
    symbolic = JacobianResolver().resolve(
        model, state, model.params, NumpyBackend()
    )
    vector = matrix_to_vector(state)
    numerical = central_difference_jacobian(
        lambda value: residual_vector(model, vector_to_matrix(value, 3), model.params),
        vector,
        1e-6,
    )
    np.testing.assert_allclose(symbolic, numerical, atol=2e-8)


def test_vdp_vector_fast_path_matches_matrix_equations():
    model = VDP2ModeModel(
        omega_a=0.2,
        omega_b=-0.1,
        gamma_a=2.0,
        gamma_b=1.0,
        Gamma=0.01,
        g=0.5,
    )
    states = np.stack([_state(2), _state(2) * 3.0])
    vectors = matrix_to_vector(states)

    np.testing.assert_allclose(
        model.cam_residual_vector(vectors, model.params),
        matrix_to_vector(model_liouvillian(model, states, model.params)),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        model.cam_jacobian_vector(vectors, model.params),
        model.cam_jacobian(states, model.params),
        atol=1e-12,
    )


def test_missing_jacobian_requires_explicit_fallback(no_jacobian_model):
    state = np.eye(2, dtype=complex)
    with pytest.raises(JacobianUnavailableError):
        JacobianResolver().resolve(
            no_jacobian_model, state, {}, NumpyBackend()
        )
    resolver = JacobianResolver(allow_finite_difference=True)
    jacobian = resolver.resolve(no_jacobian_model, state, {}, NumpyBackend())
    np.testing.assert_allclose(jacobian, -np.eye(4), atol=1e-8)
    assert resolver.last_source == "finite_difference"
