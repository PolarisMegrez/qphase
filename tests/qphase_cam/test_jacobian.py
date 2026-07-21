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

from models.crosskerr_2mode import CrossKerr2ModeModel
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
        CrossKerr2ModeModel(
            omega_a=0.5,
            omega_b=0.01,
            chi=0.01,
            gamma_a=0.5,
            gamma_b=0.517926,
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
    symbolic = SymbolicJacobian(
        model.cam_symbolic_matrices(), model.n_modes, "numpy"
    )(state, model.params)
    resolver = JacobianResolver()
    analytic = resolver.resolve(model, state, model.params, NumpyBackend())
    vector = matrix_to_vector(state)
    (
        r_aa,
        _,
        _,
        r_ab_real,
        r_ac_real,
        _,
        r_ab_imag,
        r_ac_imag,
        _,
    ) = vector
    params = model.params
    detuning_ab = (
        params["omega_a"] - params["omega_b"] + 2.0 * params["chi"] * (r_aa - 1.0)
    )
    detuning_ac = (
        params["omega_a"] - params["omega_c"] + 2.0 * params["chi"] * (r_aa - 1.0)
    )
    detuning_bc = params["omega_b"] - params["omega_c"]
    decay_ab = -(params["gamma_a"] + params["gamma_b"]) / 2.0
    decay_ac = (-params["gamma_a"] + params["gamma_c"]) / 2.0
    decay_bc = (-params["gamma_b"] + params["gamma_c"]) / 2.0
    expected = np.zeros((9, 9))
    expected[0, (0, 6, 7)] = (
        -params["gamma_a"],
        -2.0 * params["g_ab"],
        -2.0 * params["g_ac"],
    )
    expected[1, (1, 6)] = (-params["gamma_b"], 2.0 * params["g_ab"])
    expected[2, (2, 7)] = (params["gamma_c"], 2.0 * params["g_ac"])
    expected[3, (0, 3, 6, 8)] = (
        2.0 * params["chi"] * r_ab_imag,
        decay_ab,
        detuning_ab,
        -params["g_ac"],
    )
    expected[4, (0, 4, 7, 8)] = (
        2.0 * params["chi"] * r_ac_imag,
        decay_ac,
        detuning_ac,
        params["g_ab"],
    )
    expected[5, (5, 6, 7, 8)] = (
        decay_bc,
        params["g_ac"],
        params["g_ab"],
        detuning_bc,
    )
    expected[6, (0, 1, 3, 5, 6)] = (
        params["g_ab"] - 2.0 * params["chi"] * r_ab_real,
        -params["g_ab"],
        -detuning_ab,
        -params["g_ac"],
        decay_ab,
    )
    expected[7, (0, 2, 4, 5, 7)] = (
        params["g_ac"] - 2.0 * params["chi"] * r_ac_real,
        -params["g_ac"],
        -detuning_ac,
        -params["g_ab"],
        decay_ac,
    )
    expected[8, (3, 4, 5, 8)] = (
        params["g_ac"],
        -params["g_ab"],
        -detuning_bc,
        decay_bc,
    )
    numerical = central_difference_jacobian(
        lambda value: residual_vector(model, vector_to_matrix(value, 3), model.params),
        vector,
        1e-6,
    )
    assert resolver.last_source == "analytic"
    np.testing.assert_allclose(analytic, expected, atol=1e-12)
    np.testing.assert_allclose(analytic, symbolic, atol=1e-11)
    np.testing.assert_allclose(symbolic, numerical, atol=2e-8)


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
        CrossKerr2ModeModel(
            omega_a=0.5,
            omega_b=0.01,
            chi=0.01,
            gamma_a=0.5,
            gamma_b=0.517926,
            g=0.5,
        ),
        Kerr3ModeModel(
            omega_a=0.0,
            omega_b=-0.1,
            omega_c=0.2,
            chi=0.01,
            gamma_a=0.5,
            gamma_b=1.0,
            gamma_c=0.4,
            g_ab=0.5,
            g_ac=0.3,
        ),
    ],
)
def test_vector_fast_path_matches_matrix_equations(model):
    state = _state(model.n_modes)
    states = np.stack([state, state * 3.0])
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
