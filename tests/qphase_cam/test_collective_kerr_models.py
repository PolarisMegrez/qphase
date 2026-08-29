"""fpgen-backed collective-reservoir Kerr model contracts."""

from __future__ import annotations

import numpy as np
import pytest
from qphase_cam.core.fpgen import FPGenDynamicsAdapter
from qphase_cam.model import CAMBogoliubovModel

from models.collective_kerr_2mode import CollectiveKerr2ModeModel
from models.collective_loss_kerr_3mode import CollectiveLossKerr3ModeModel
from models.reservoir_kerr_3mode import ReservoirKerr3ModeModel


@pytest.fixture(
    params=(
        CollectiveKerr2ModeModel(
            omega_0=0.0,
            delta=0.1,
            chi=0.01,
            g=0.5,
            kappa_bright=1.2,
            pump_bright=0.4,
            kappa_dark=0.02,
        ),
        CollectiveLossKerr3ModeModel(
            omega_a=0.0,
            omega_b=0.2,
            omega_c=-0.1,
            chi=0.01,
            g_ab=0.4,
            g_ac=0.3,
            g_bc=0.05,
            pump_a=0.2,
            kappa_bright=1.0,
            kappa_dark=0.02,
        ),
        ReservoirKerr3ModeModel(
            omega_r=0.0,
            omega_0=0.0,
            delta=0.1,
            chi=0.01,
            g=0.5,
            g_r=0.8,
            kappa_r=4.0,
            pump_r=1.0,
            kappa_local=0.02,
        ),
    ),
    ids=("collective", "collective_loss", "explicit_reservoir"),
)
def fpgen_model(request):
    return request.param


def test_numerical_fast_paths_are_compiled_from_fpgen(fpgen_model):
    adapter = FPGenDynamicsAdapter.from_model(fpgen_model)
    vector = np.arange(1, adapter.state_size + 1, dtype=float) / 10.0

    np.testing.assert_allclose(
        fpgen_model.cam_residual_vector(vector, fpgen_model.params),
        adapter.rhs(vector),
        atol=1e-13,
    )
    np.testing.assert_allclose(
        fpgen_model.cam_jacobian_vector(vector, fpgen_model.params),
        adapter.jacobian(vector),
        atol=1e-13,
    )


def test_hamiltonian_and_diffusion_are_fpgen_outputs(fpgen_model):
    adapter = FPGenDynamicsAdapter.from_model(fpgen_model)
    vector = np.arange(1, adapter.state_size + 1, dtype=float) / 7.0
    state = adapter.state_matrix(vector)
    substitutions = dict(
        zip(
            adapter.state_symbols + adapter.parameter_symbols,
            tuple(vector) + tuple(adapter.parameter_vector()),
            strict=True,
        )
    )

    expected_hamiltonian = np.asarray(
        adapter.symbolic_hamiltonian.subs(substitutions), dtype=complex
    )
    expected_diffusion = np.asarray(
        fpgen_model.cam_fpgen_dynamics().diffusion.subs(substitutions),
        dtype=complex,
    )
    np.testing.assert_allclose(
        fpgen_model.cam_hamiltonian(state, fpgen_model.params),
        expected_hamiltonian,
    )
    np.testing.assert_allclose(
        fpgen_model.cam_diffusion(state, fpgen_model.params), expected_diffusion
    )


def test_sde_factor_reconstructs_fpgen_normal_diffusion(fpgen_model):
    rng = np.random.default_rng(100 + fpgen_model.n_modes)
    state = rng.normal(size=(4, fpgen_model.n_modes)) + 1j * rng.normal(
        size=(4, fpgen_model.n_modes)
    )
    density = np.einsum("...i,...j->...ij", state, state.conj())
    factor = fpgen_model.diffusion(state, 0.0, fpgen_model.params)

    np.testing.assert_allclose(
        factor @ factor.conj().transpose(0, 2, 1),
        fpgen_model.cam_diffusion(density, fpgen_model.params),
        atol=1e-13,
    )


def test_collective_diffusion_factor_supports_semidefinite_dark_mode():
    model = CollectiveKerr2ModeModel(
        omega_0=0.0,
        delta=0.0,
        chi=0.01,
        g=0.5,
        kappa_bright=1.0,
        pump_bright=0.0,
        kappa_dark=0.0,
    )
    state = np.asarray([[0.2 + 0.1j, -0.3 + 0.4j]])
    density = np.einsum("...i,...j->...ij", state, state.conj())
    factor = model.diffusion(state, 0.0, model.params)

    np.testing.assert_allclose(
        factor @ factor.conj().transpose(0, 2, 1),
        model.cam_diffusion(density, model.params),
        atol=1e-13,
    )


def test_collective_model_has_exchange_symmetry_at_zero_detuning():
    model = CollectiveKerr2ModeModel(
        omega_0=0.2,
        delta=0.0,
        chi=0.03,
        g=0.4,
        kappa_bright=1.1,
        pump_bright=0.3,
        kappa_dark=0.04,
    )
    state = np.asarray([[1.4, 0.2 + 0.1j], [0.2 - 0.1j, 0.8]])
    swap = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    swapped = swap @ state @ swap

    np.testing.assert_allclose(
        model.cam_hamiltonian(swapped, model.params),
        swap @ model.cam_hamiltonian(state, model.params) @ swap,
    )
    np.testing.assert_allclose(
        model.cam_diffusion(swapped, model.params),
        swap @ model.cam_diffusion(state, model.params) @ swap,
    )


def test_collective_loss_model_has_b_c_exchange_symmetry():
    model = CollectiveLossKerr3ModeModel(
        omega_a=0.0,
        omega_b=0.2,
        omega_c=0.2,
        chi=0.03,
        g_ab=0.4,
        g_ac=0.4,
        g_bc=0.05,
        pump_a=0.3,
        kappa_bright=1.1,
        kappa_dark=0.04,
    )
    state = np.asarray(
        [
            [1.0, 0.2 + 0.1j, -0.1 + 0.3j],
            [0.2 - 0.1j, 1.4, 0.15 + 0.05j],
            [-0.1 - 0.3j, 0.15 - 0.05j, 0.8],
        ]
    )
    swap = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    swapped = swap @ state @ swap

    np.testing.assert_allclose(
        model.cam_hamiltonian(swapped, model.params),
        swap @ model.cam_hamiltonian(state, model.params) @ swap,
    )
    np.testing.assert_allclose(
        model.cam_diffusion(swapped, model.params),
        swap @ model.cam_diffusion(state, model.params) @ swap,
    )


@pytest.mark.parametrize("g_ac,g_bc", ((-0.3, 0.05), (0.3, -0.05), (-0.3, -0.05)))
def test_collective_loss_model_accepts_real_coupling_sectors(g_ac, g_bc):
    model = CollectiveLossKerr3ModeModel(
        omega_a=0.0,
        omega_b=0.2,
        omega_c=-0.1,
        chi=0.01,
        g_ab=0.4,
        g_ac=g_ac,
        g_bc=g_bc,
        pump_a=0.2,
        kappa_bright=1.0,
        kappa_dark=0.02,
    )
    adapter = FPGenDynamicsAdapter.from_model(model)
    vector = np.arange(1, adapter.state_size + 1, dtype=float) / 10.0

    assert adapter.parameter_domains["g_ac"] == "real"
    assert adapter.parameter_domains["g_bc"] == "real"
    np.testing.assert_allclose(
        model.cam_residual_vector(vector, model.params),
        adapter.rhs(vector),
        atol=1e-13,
    )


def test_collective_loss_bogoliubov_block_matches_wirtinger_linearization():
    model = CollectiveLossKerr3ModeModel(
        omega_a=0.1,
        omega_b=0.4,
        omega_c=-0.2,
        chi=0.03,
        g_ab=0.35,
        g_ac=-0.12,
        g_bc=0.07,
        pump_a=0.18,
        kappa_bright=1.1,
        kappa_dark=0.06,
    )
    amplitude = np.asarray([1.4, 0.3 + 0.2j, -0.15 + 0.25j], dtype=complex)
    state = np.outer(amplitude, amplitude.conj())
    interaction = model.cam_bogoliubov_interaction(state, model.params)

    assert isinstance(model, CAMBogoliubovModel)
    np.testing.assert_allclose(
        interaction,
        np.diag([2.0 * model.params["chi"] * state[0, 0], 0.0, 0.0]),
    )

    step = 1e-6
    derivative_real = np.empty((model.n_modes, model.n_modes), dtype=complex)
    derivative_imag = np.empty_like(derivative_real)

    def drift(value):
        return model.drift(value[None, :], 0.0, model.params)[0]

    for column in range(model.n_modes):
        offset = np.zeros(model.n_modes, dtype=complex)
        offset[column] = step
        derivative_real[:, column] = (
            drift(amplitude + offset) - drift(amplitude - offset)
        ) / (2.0 * step)
        offset[column] = 1j * step
        derivative_imag[:, column] = (
            drift(amplitude + offset) - drift(amplitude - offset)
        ) / (2.0 * step)

    normal_block = 0.5j * (derivative_real - 1j * derivative_imag)
    anomalous_block = 0.5j * (derivative_real + 1j * derivative_imag)
    hamiltonian = model.cam_hamiltonian(state, model.params)
    np.testing.assert_allclose(
        normal_block, hamiltonian + interaction, rtol=2e-9, atol=2e-9
    )
    np.testing.assert_allclose(anomalous_block, interaction, rtol=2e-9, atol=2e-9)


def test_explicit_reservoir_preserves_main_mode_exchange_symmetry():
    model = ReservoirKerr3ModeModel(
        omega_r=-0.1,
        omega_0=0.2,
        delta=0.0,
        chi=0.03,
        g=0.4,
        g_r=0.7,
        kappa_r=3.0,
        pump_r=0.5,
        kappa_local=0.04,
    )
    state = np.asarray(
        [
            [1.0, 0.2 + 0.1j, -0.1 + 0.3j],
            [0.2 - 0.1j, 1.4, 0.15 + 0.05j],
            [-0.1 - 0.3j, 0.15 - 0.05j, 0.8],
        ]
    )
    swap = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    swapped = swap @ state @ swap

    np.testing.assert_allclose(
        model.cam_hamiltonian(swapped, model.params),
        swap @ model.cam_hamiltonian(state, model.params) @ swap,
    )
    np.testing.assert_allclose(
        model.cam_diffusion(swapped, model.params),
        swap @ model.cam_diffusion(state, model.params) @ swap,
    )
