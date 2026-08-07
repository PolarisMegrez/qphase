"""Cross-capability equation tests for project models."""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.coordinates import matrix_to_vector
from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from models.crosskerr_2mode import CrossKerr2ModeModel
from models.kerr_2mode import Kerr2ModeModel
from models.kerr_3mode import Kerr3ModeModel
from models.pair_hopping_2mode import PairHopping2ModeModel
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
    assert PairHopping2ModeModel.steady_state_capacity == 5


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
    provenance = adapter.provenance()
    assert provenance["fingerprint"]
    assert provenance["model_schema"] == "2.0"
    assert provenance["moment_api"] == "1.0"
    assert provenance["reduction_api"] == "1.1"
    assert provenance["matrix_semantics"] == "normal_second_moment"
    assert provenance["physical_domain_hint"] == "hermitian_psd"
    reconstructed = adapter.state_matrix(vector)
    assert reconstructed.shape == (model.n_modes, model.n_modes)
    np.testing.assert_allclose(reconstructed, reconstructed.conj().T)
    for name in model.params:
        expected = "real" if name.startswith("omega") else "nonnegative"
        assert adapter.parameter_domains[name] == expected


def test_pair_hopping_fpgen_contract():
    model = PairHopping2ModeModel(
        omega_a=0.0,
        omega_b=0.0,
        g=1.0,
        k=0.1,
        gamma_a=1.0,
        gamma_b=1.0,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    assert adapter.state_matrix_shape == (2, 2)
    assert adapter.parameter_domains == {
        "omega_a": "real",
        "omega_b": "real",
        "g": "nonnegative",
        "k": "nonnegative",
        "gamma_a": "nonnegative",
        "gamma_b": "nonnegative",
    }
    vector = np.asarray([2.0, 1.0, 0.2, -0.1])
    np.testing.assert_allclose(
        model.cam_residual_vector(vector, model.params), adapter.rhs(vector)
    )
    np.testing.assert_allclose(
        model.cam_jacobian_vector(vector, model.params), adapter.jacobian(vector)
    )
    step = 1e-6
    finite_difference = np.column_stack(
        [
            (
                model.cam_residual_vector(
                    vector + step * np.eye(4)[index], model.params
                )
                - model.cam_residual_vector(
                    vector - step * np.eye(4)[index], model.params
                )
            )
            / (2.0 * step)
            for index in range(4)
        ]
    )
    np.testing.assert_allclose(
        model.cam_jacobian_vector(vector, model.params),
        finite_difference,
        rtol=1e-9,
        atol=1e-9,
    )
    state = adapter.state_matrix(vector)
    assert model.cam_solution_sort_key(state, model.params) == pytest.approx(2.0)
    hamiltonian = model.cam_hamiltonian(state, model.params)
    diffusion = model.cam_diffusion(state, model.params)
    symbols = adapter.state_symbols + adapter.parameter_symbols
    values = tuple(vector) + tuple(adapter.parameter_vector())
    np.testing.assert_allclose(
        hamiltonian,
        np.asarray(
            adapter.symbolic_hamiltonian.subs(dict(zip(symbols, values, strict=True))),
            dtype=complex,
        ),
    )
    np.testing.assert_allclose(diffusion, np.diag([0.5, 0.5]))


def test_fpgen_mixed_state_parameter_directional_matches_difference():
    model = VDP2ModeModel(
        omega_a=0.2,
        omega_b=-0.1,
        gamma_a=2.0,
        gamma_b=1.0,
        Gamma=0.01,
        g=0.5,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    state = np.asarray([1.2, 0.8, 0.1, -0.2])
    state_direction = np.asarray([0.7, -0.3, 0.2, 0.4])
    parameter_direction = adapter.parameter_direction("Gamma")
    exact = adapter.mixed_directional(
        2,
        state,
        model.params,
        state_directions=(state_direction,),
        parameter_directions=(parameter_direction,),
    )
    step = 1e-6
    plus = {**model.params, "Gamma": model.params["Gamma"] + step}
    minus = {**model.params, "Gamma": model.params["Gamma"] - step}
    finite_difference = (
        adapter.directional(1, state, plus, state_direction)
        - adapter.directional(1, state, minus, state_direction)
    ) / (2.0 * step)
    np.testing.assert_allclose(exact, finite_difference, rtol=1e-7, atol=1e-8)
