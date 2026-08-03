"""Cross-capability equation tests for project models."""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from models.crosskerr_2mode import CrossKerr2ModeModel
from models.kerr_2mode import Kerr2ModeModel
from models.kerr_3mode import Kerr3ModeModel
from models.vdp_2mode import VDP2ModeModel


@pytest.fixture(
    params=[
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
    ]
)
def model(request):
    return request.param


def test_cam_hamiltonian_matches_sde_drift_matrix(model):
    rng = np.random.default_rng(model.n_modes)
    state = (
        2.0
        + rng.normal(size=(8, model.n_modes))
        + 1j * rng.normal(size=(8, model.n_modes))
    )
    density = np.einsum("...i,...j->...ij", state, state.conj())
    np.testing.assert_allclose(
        model.drift_matrix(state, 0.0, model.params),
        -1j * model.cam_hamiltonian(density, model.params),
    )


def test_cam_diffusion_matches_sde_factor(model):
    rng = np.random.default_rng(10 + model.n_modes)
    state = (
        5.0
        + rng.normal(size=(8, model.n_modes))
        + 1j * rng.normal(size=(8, model.n_modes))
    )
    density = np.einsum("...i,...j->...ij", state, state.conj())
    factor = model.diffusion(state, 0.0, model.params)
    covariance = factor @ factor.conj().transpose(0, 2, 1)
    np.testing.assert_allclose(
        covariance, model.cam_diffusion(density, model.params), rtol=1e-12
    )


def test_declared_solution_capacities():
    assert VDP2ModeModel.steady_state_capacity == 4
    assert Kerr2ModeModel.steady_state_capacity == 3
    assert Kerr3ModeModel.steady_state_capacity == 8
    assert CrossKerr2ModeModel.steady_state_capacity == 3


def test_fpgen_dynamics_match_cam_vector_fast_paths(model):
    adapter = FPGenDynamicsAdapter.from_model(model)
    rng = np.random.default_rng(100 + model.n_modes)
    raw = rng.normal(size=(model.n_modes, model.n_modes)) + 1j * rng.normal(
        size=(model.n_modes, model.n_modes)
    )
    vector = matrix_to_vector(raw @ raw.conj().T)

    np.testing.assert_allclose(
        adapter.rhs(vector),
        model.cam_residual_vector(vector, model.params),
        atol=2e-12,
    )
    np.testing.assert_allclose(
        adapter.jacobian(vector),
        model.cam_jacobian_vector(vector, model.params),
        atol=2e-12,
    )
    assert adapter.state_ids[0] == "r_diag_0"
    assert adapter.provenance()["fingerprint"]
