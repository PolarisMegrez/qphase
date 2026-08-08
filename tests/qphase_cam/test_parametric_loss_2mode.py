"""Contracts for the augmented parametric-loss model."""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from models.parametric_loss_2mode import ParametricLoss2ModeModel


@pytest.fixture
def model() -> ParametricLoss2ModeModel:
    return ParametricLoss2ModeModel(
        omega_a=0.2,
        omega_b=-0.1,
        epsilon=0.3,
        g=0.4,
        kappa_a=0.6,
        kappa_b=1.0,
        kappa_2=0.05,
    )


def test_fpgen_selects_augmented_moment_layout(model):
    adapter = FPGenDynamicsAdapter.from_model(model)

    assert adapter.moment_layout == "augmented"
    assert adapter.state_size == 10
    assert adapter.state_matrix_shape == (4, 4)
    assert adapter.state_ids == (
        "r_diag_0",
        "r_diag_1",
        "r_re_0_1",
        "r_im_0_1",
        "c_re_0_0",
        "c_re_0_1",
        "c_re_1_1",
        "c_im_0_0",
        "c_im_0_1",
        "c_im_1_1",
    )


def test_augmented_matrix_round_trip_matches_fpgen(model):
    adapter = FPGenDynamicsAdapter.from_model(model)
    vector = np.arange(1, 11, dtype=float) / 13.0
    state = adapter.state_matrix(vector)

    np.testing.assert_allclose(model._augmented_vector(state, np), vector)
    np.testing.assert_allclose(
        model.cam_residual_vector(vector, model.params), adapter.rhs(vector)
    )
    np.testing.assert_allclose(
        model.cam_jacobian(state, model.params), adapter.jacobian(vector)
    )


def test_augmented_matrices_are_fpgen_outputs(model):
    adapter = FPGenDynamicsAdapter.from_model(model)
    vector = np.arange(1, 11, dtype=float) / 11.0
    state = adapter.state_matrix(vector)
    substitutions = dict(
        zip(
            adapter.state_symbols + adapter.parameter_symbols,
            tuple(vector) + tuple(adapter.parameter_vector()),
            strict=True,
        )
    )

    expected_h = np.asarray(
        adapter.symbolic_hamiltonian.subs(substitutions), dtype=complex
    )
    expected_d = np.asarray(
        model.cam_fpgen_dynamics().diffusion.subs(substitutions), dtype=complex
    )
    np.testing.assert_allclose(model.cam_hamiltonian(state, model.params), expected_h)
    np.testing.assert_allclose(model.cam_diffusion(state, model.params), expected_d)


def test_sde_drift_and_diffusion_match_wigner_equations(model):
    state = np.asarray([[0.2 + 0.3j, -0.4 + 0.1j]])
    alpha, beta = state[0]
    params = model.params
    expected = np.asarray(
        [
            [
                (
                    -1j * params["omega_a"]
                    - params["kappa_a"] / 2
                    + params["kappa_2"] * (1 - abs(alpha) ** 2)
                )
                * alpha
                - 1j * params["g"] * beta
                - 1j * params["epsilon"] * alpha.conjugate(),
                (-1j * params["omega_b"] - params["kappa_b"] / 2) * beta
                - 1j * params["g"] * alpha,
            ]
        ]
    )
    factor = model.diffusion(state, 0.0, params)
    expected_covariance = np.diag(
        [
            2 * params["kappa_2"] * abs(alpha) ** 2
            - params["kappa_2"]
            + params["kappa_a"] / 2,
            params["kappa_b"] / 2,
        ]
    )

    np.testing.assert_allclose(model.drift(state, 0.0, params), expected)
    np.testing.assert_allclose(
        factor @ factor.conj().transpose(0, 2, 1), expected_covariance[None, ...]
    )


def test_config_rejects_negative_truncated_wigner_diffusion():
    with pytest.raises(ValueError, match="at least 2\\*kappa_2"):
        ParametricLoss2ModeModel(
            omega_a=0.0,
            omega_b=0.0,
            epsilon=0.1,
            g=0.2,
            kappa_a=0.1,
            kappa_b=1.0,
            kappa_2=0.1,
        )


def test_generic_cayley_path_fails_explicitly(model):
    state = np.ones((1, 2), dtype=complex)
    with pytest.raises(NotImplementedError, match="conjugate drift"):
        model.drift_matrix(state, 0.0, model.params)
