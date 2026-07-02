"""Lightweight import/instantiation tests for the current model plugins."""

from __future__ import annotations


def test_kerr_3pa_model():
    """The Kerr 3PA model can be imported and instantiated."""
    from models.kerr_3pa import Kerr3PAModel

    model = Kerr3PAModel(
        omega0=1.0,
        chi=0.01,
        kappa3=0.001,
        beta=1.0,
        epsilon=0.1,
        kappa1=1.0,
    )
    assert model.name == "kerr_3pa"
    assert model.n_modes == 1
    assert model.noise_dim == 2
    sde = model.to_diffusive_sde_model()
    assert sde.name == "kerr_3pa"


def test_kerr_3mode_model():
    """The Kerr three-mode model can be imported and instantiated."""
    from models.kerr_3mode import Kerr3ModeModel

    model = Kerr3ModeModel(
        omega_a=1.0,
        omega_b=1.0,
        omega_c=1.0,
        chi=0.01,
        kappa_a=0.1,
        kappa_b=0.1,
        kappa_c=0.1,
        g_ab=0.1,
        g_ac=0.1,
    )
    assert model.name == "kerr_3mode"
    assert model.n_modes == 3
    assert model.noise_dim == 6
    sde = model.to_diffusive_sde_model()
    assert sde.name == "kerr_3mode"


def test_vdp_level3_model():
    """The Van der Pol level-3 model can be imported and instantiated."""
    from models.vdp_level3 import VDPLevel3Model

    model = VDPLevel3Model(
        omega_a=1.0,
        omega_b=1.0,
        gamma_a=0.1,
        gamma_b=0.1,
        Gamma=1.0,
        g=0.5,
        D=1.0,
    )
    assert model.name == "vdp_level3"
    assert model.n_modes == 2
    assert model.noise_dim == 4
    sde = model.to_diffusive_sde_model()
    assert sde.name == "vdp_level3"
