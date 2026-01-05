"""Test consistency between Level 2 (FPE) and Level 3 (SDE) models."""

import numpy as np
import pytest
from qphase_sde.model import fpe_to_sde

from models.vdp_level2 import VDPLevel2Config, VDPLevel2Model
from models.vdp_level3 import VDPLevel3Config, VDPLevel3Model


@pytest.fixture
def vdp_params():
    return {
        "omega_a": 1.0,
        "omega_b": 1.1,
        "gamma_a": 0.1,
        "gamma_b": 0.2,
        "Gamma": 0.05,
        "g": 0.01,
        "D": 1.0,
    }


def test_vdp_level2_to_level3_consistency(vdp_params):
    """Verify that converting Level 2 FPE to Level 3 SDE yields
    drift and diffusion terms consistent with the manual Level 3 implementation.
    """
    # 1. Create models (Plugins)
    fpe_plugin = VDPLevel2Model(VDPLevel2Config(**vdp_params))
    manual_sde_plugin = VDPLevel3Model(VDPLevel3Config(**vdp_params))

    # Get underlying dataclasses/protocols
    fpe_model = fpe_plugin.to_phase_space_model()
    manual_sde_model = manual_sde_plugin  # It implements SDEModel protocol directly

    # 2. Convert FPE to SDE automatically
    # Note: fpe_to_sde uses a generic sqrt wrapper for diffusion
    auto_sde_model = fpe_to_sde(fpe_model)

    # 3. Define a test state
    n_traj = 5
    # Random complex state
    rng = np.random.default_rng(42)
    y = rng.standard_normal((n_traj, 2)) + 1j * rng.standard_normal((n_traj, 2))
    t = 0.0

    # 4. Check Drift
    drift_manual = manual_sde_model.drift(y, t, vdp_params)
    drift_auto = auto_sde_model.drift(y, t, vdp_params)

    np.testing.assert_allclose(
        drift_auto,
        drift_manual,
        rtol=1e-10,
        err_msg="Drift terms do not match between manual SDE and FPE->SDE conversion",
    )

    # 5. Check Diffusion
    # Note: The manual implementation returns a matrix (n, 2, 2)
    # The FPE implementation returns D2 coefficients (n, 2)
    # The auto converter takes sqrt(D2).
    # Since D2 is diagonal in our FPE definition, sqrt(D2) should match
    # the diagonal of B.

    diff_manual = manual_sde_model.diffusion(y, t, vdp_params)  # (n, 2, 2)
    diff_auto = auto_sde_model.diffusion(y, t, vdp_params)  # (n, 2) from sqrt(D2)

    # The auto converter in model.py currently does: return xp.sqrt(d2)
    # In vdp_level2.py, vdp_diffusion_d2 returns shape (n, 2)
    # So diff_auto has shape (n, 2)

    # We need to verify that diff_auto corresponds to the diagonal of diff_manual
    # diff_manual is diagonal, so we extract diagonal elements
    diff_manual_diag = np.stack([diff_manual[:, 0, 0], diff_manual[:, 1, 1]], axis=1)

    np.testing.assert_allclose(
        diff_auto,
        diff_manual_diag,
        rtol=1e-10,
        err_msg="Diffusion terms do not match. Auto SDE should be sqrt(D2).",
    )


def test_fpe_to_sde_structure(vdp_params):
    """Verify the structure of the converted model."""
    fpe_plugin = VDPLevel2Model(VDPLevel2Config(**vdp_params))
    fpe_model = fpe_plugin.to_phase_space_model()
    sde_model = fpe_to_sde(fpe_model)

    assert sde_model.n_modes == 2
    assert sde_model.noise_basis == "complex"
    assert sde_model.noise_dim == 2
    assert sde_model.params == vdp_params
