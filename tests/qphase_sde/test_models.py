"""Equation and contract tests for local model plugins."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from models.base import SDEModelPlugin
from models.kerr_2mode import Kerr2ModeModel
from models.kerr_3mode import Kerr3ModeModel
from models.vdp_2mode import VDP2ModeModel


@pytest.mark.parametrize(
    ("model", "name", "n_modes"),
    [
        (
            VDP2ModeModel(
                omega_a=0.2, omega_b=0.3, gamma_a=0.4, gamma_b=0.6,
                Gamma=0.1, g=0.5,
            ),
            "vdp_2mode",
            2,
        ),
        (
            Kerr2ModeModel(
                omega_a=0.2, omega_b=0.3, chi=0.1, gamma_a=0.4,
                gamma_b=0.6, g=0.5,
            ),
            "kerr_2mode",
            2,
        ),
        (
            Kerr3ModeModel(
                omega_a=0.2, omega_b=0.3, omega_c=0.4, chi=0.1,
                gamma_a=0.4, gamma_b=0.6, gamma_c=0.8, g_ab=0.5,
                g_ac=0.7,
            ),
            "kerr_3mode",
            3,
        ),
    ],
)
def test_model_plugin_contract(model, name, n_modes):
    assert isinstance(model, SDEModelPlugin)
    assert model.name == name
    assert model.n_modes == n_modes
    assert model.noise_basis == "complex"
    assert model.noise_dim == 2 * n_modes
    assert model.to_diffusive_sde_model().name == name


def test_model_schemas_reject_unknown_parameters():
    with pytest.raises(ValidationError, match="extra"):
        VDP2ModeModel(
            omega_a=0.2, omega_b=0.3, gamma_a=0.4, gamma_b=0.6,
            Gamma=0.1, g=0.5, D=1.0,
        )


def test_vdp_equations():
    model = VDP2ModeModel(
        omega_a=0.2, omega_b=0.3, gamma_a=0.4, gamma_b=0.6,
        Gamma=0.1, g=0.5,
    )
    y = np.array([[1.0 + 2.0j, 3.0 - 1.0j]])
    alpha, beta = y[0]
    expected = np.array([[(-0.2j + 0.2 + 0.1 * (1 - abs(alpha) ** 2)) * alpha
                          - 0.5j * beta,
                          (-0.3j - 0.3) * beta - 0.5j * alpha]])
    np.testing.assert_allclose(model.drift(y, 0.0, model.params), expected)
    diffusion = model.diffusion(y, 0.0, model.params)
    np.testing.assert_allclose(np.diagonal(diffusion, axis1=1, axis2=2) ** 2,
                               [[1.1, 0.3]])


def test_kerr_2mode_equations():
    model = Kerr2ModeModel(
        omega_a=0.2, omega_b=0.3, chi=0.1, gamma_a=0.4,
        gamma_b=0.6, g=0.5,
    )
    y = np.array([[1.0 + 2.0j, 3.0 - 1.0j]])
    matrix = model.drift_matrix(y, 0.0, model.params)
    assert matrix[0, 0, 0] == pytest.approx(0.2 - 1.0j)
    assert matrix[0, 1, 1] == pytest.approx(-0.3 - 0.3j)
    np.testing.assert_allclose(
        model.drift(y, 0.0, model.params), (matrix[0] @ y[0])[None, :]
    )


def test_kerr_3mode_gain_loss_signs_and_diffusion():
    model = Kerr3ModeModel(
        omega_a=0.2, omega_b=0.3, omega_c=0.4, chi=0.1,
        gamma_a=0.4, gamma_b=0.6, gamma_c=0.8, g_ab=0.5, g_ac=0.7,
    )
    y = np.array([[1.0 + 0.0j, 2.0 + 0.0j, 3.0 + 0.0j]])
    matrix = model.drift_matrix(y, 0.0, model.params)
    assert matrix[0, 0, 0] == pytest.approx(-0.2 - 0.2j)
    assert matrix[0, 1, 1] == pytest.approx(-0.3 - 0.3j)
    assert matrix[0, 2, 2] == pytest.approx(0.4 - 0.4j)
    diffusion = model.diffusion(y, 0.0, model.params)
    np.testing.assert_allclose(np.diagonal(diffusion, axis1=1, axis2=2) ** 2,
                               [[0.2, 0.3, 0.4]])
