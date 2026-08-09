"""fpgen-backed collective-loss VDP model contracts."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError
from qphase_cam.core.fpgen import FPGenDynamicsAdapter

from models.collective_vdp_2mode import CollectiveVDP2ModeModel


@pytest.fixture
def model() -> CollectiveVDP2ModeModel:
    return CollectiveVDP2ModeModel(
        omega_a=0.2,
        omega_b=-0.1,
        Gamma=0.03,
        g=0.4,
        pump_a=0.5,
        kappa_bright=1.0,
        kappa_dark=0.04,
    )


def test_fpgen_contract_and_regular_reduction(model):
    adapter = FPGenDynamicsAdapter.from_model(model)

    assert adapter.state_size == 4
    assert adapter.moment_layout == "normal"
    assert adapter.parameter_domains == {
        "omega_a": "real",
        "omega_b": "real",
        "Gamma": "nonnegative",
        "g": "nonnegative",
        "pump_a": "nonnegative",
        "kappa_bright": "nonnegative",
        "kappa_dark": "nonnegative",
    }
    search = adapter.search_linear_reductions(retained_dimension=1)
    assert len(search.candidates) == 4
    assert {item.retained_ids for item in search.candidates} == {("r_diag_0",)}
    assert {item.reduced_degree for item in search.candidates} == {4}


def test_fast_paths_match_fpgen(model):
    adapter = FPGenDynamicsAdapter.from_model(model)
    vector = np.asarray([1.3, 0.7, 0.2, -0.1])

    np.testing.assert_allclose(
        model.cam_residual_vector(vector, model.params),
        adapter.rhs(vector),
        atol=1.0e-13,
    )
    np.testing.assert_allclose(
        model.cam_jacobian_vector(vector, model.params),
        adapter.jacobian(vector),
        atol=1.0e-13,
    )


def test_sde_factor_reconstructs_state_dependent_diffusion(model):
    amplitudes = np.asarray([[0.8 + 0.2j, -0.3 + 0.1j], [1.1 - 0.4j, 0.2 + 0.5j]])
    state = np.einsum("...i,...j->...ij", amplitudes, amplitudes.conj())
    factor = model.diffusion(amplitudes, 0.0, model.params)

    np.testing.assert_allclose(
        factor @ factor.conj().transpose(0, 2, 1),
        model.cam_diffusion(state, model.params),
        atol=1.0e-13,
    )


@pytest.mark.parametrize("name", ("Gamma", "g", "pump_a", "kappa_dark"))
def test_negative_rates_are_rejected(name):
    params = {
        "omega_a": 0.0,
        "omega_b": 0.0,
        "Gamma": 0.03,
        "g": 0.4,
        "pump_a": 0.5,
        "kappa_bright": 1.0,
        "kappa_dark": 0.04,
    }
    params[name] = -0.01
    with pytest.raises(ValidationError, match="must be nonnegative"):
        CollectiveVDP2ModeModel(**params)


def test_non_psd_wigner_diffusion_is_rejected():
    with pytest.raises(ValidationError, match="non-PSD"):
        CollectiveVDP2ModeModel(
            omega_a=0.0,
            omega_b=0.0,
            Gamma=0.2,
            g=0.4,
            pump_a=0.01,
            kappa_bright=1.0,
            kappa_dark=0.0,
        )


def test_fused_cayley_kernel_is_registered(model):
    class CuPyBackendName:
        @staticmethod
        def backend_name():
            return "cupy"

    backend = CuPyBackendName()
    assert model.supports_fused_step("cayley_maruyama", backend)
    assert model.supports_fused_chunk("cayley_maruyama", backend)
