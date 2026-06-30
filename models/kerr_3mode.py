"""Kerr Three-Mode System (Itô SDE).

This module defines a model plugin for a 3-mode system (a, b, c) with:
- Kerr nonlinearity in mode a
- Linear coupling between a-b and a-c
- Loss in a, b
- Linear gain in c
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel

logger = logging.getLogger(__name__)


class Kerr3ModeConfig(BaseModel):
    """Configuration schema for the Kerr 3-mode system."""

    omega_a: float = Field(
        description="Frequency of mode a", json_schema_extra={"scanable": True}
    )
    omega_b: float = Field(
        description="Frequency of mode b", json_schema_extra={"scanable": True}
    )
    omega_c: float = Field(
        description="Frequency of mode c", json_schema_extra={"scanable": True}
    )

    chi: float = Field(
        description="Kerr nonlinearity strength for mode a",
        json_schema_extra={"scanable": True},
    )

    kappa_a: float = Field(
        description="Loss rate for central cavity a",
        json_schema_extra={"scanable": True},
    )
    kappa_b: float = Field(
        description="Gain rate for active cavity b",
        json_schema_extra={"scanable": True},
    )
    kappa_c: float = Field(
        description="Loss rate for lossy cavity c", json_schema_extra={"scanable": True}
    )

    g_ab: float = Field(
        description="Coupling strength between a and b",
        json_schema_extra={"scanable": True},
    )
    g_ac: float = Field(
        description="Coupling strength between a and c",
        json_schema_extra={"scanable": True},
    )


class Kerr3ModeModel:
    """Kerr Three-Mode System.

    Hamiltonian:
    H = ω_a a†a + ω_b b†b + ω_c c†c + g_ab(a†b + b†a) + g_ac(a†c + c†a) + χ a†²a²

    Dissipation/Gain:
    a: Loss κ_a
    b: Gain κ_b (Active medium)
    c: Loss κ_c

    Ref SDEs (C-number Langevin):
    dα = [-i(ω_a α + 2χ|α|²α + g_ab β + g_ac γ) - κ_a/2 α] dt + dW_α
    dβ = [-i(ω_b β + g_ab α) + κ_b/2 β] dt + dW_β
    dγ = [-i(ω_c γ + g_ac α) - κ_c/2 γ] dt + dW_γ

    Noise properties handled via real basis covariance construction.
    D = diag(κ_a, κ_b, κ_c)
    M = diag(-2iχ α², 0, 0)
    """

    name: ClassVar[str] = "kerr_3mode"
    description: ClassVar[str] = (
        "Three-mode system with Kerr nonlinearity and Gain/Loss"
    )
    config_schema: ClassVar[type[Kerr3ModeConfig]] = Kerr3ModeConfig

    def __init__(self, config: Kerr3ModeConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = Kerr3ModeConfig(**kwargs)
        self.config = config
        self._params = config.model_dump()

    @property
    def n_modes(self) -> int:
        return 3

    @property
    def noise_basis(self) -> str:
        # Use real basis to handle general covariance structures (squeezing)
        return "real"

    @property
    def noise_dim(self) -> int:
        # 6 Real noise sources (2 per mode)
        return 6

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Drift Vector."""
        xp = get_xp(y)
        alpha = y[:, 0]
        beta = y[:, 1]
        gamma = y[:, 2]

        omega_a = p["omega_a"]
        omega_b = p["omega_b"]
        omega_c = p["omega_c"]
        chi = p["chi"]
        kappa_a = p["kappa_a"]
        kappa_b = p["kappa_b"]
        kappa_c = p["kappa_c"]
        g_ab = p["g_ab"]
        g_ac = p["g_ac"]

        # Mode a
        # Nonlinear term 2*chi*|alpha|^2 * alpha comes from [a, a+ a+ a a] ~ 2 a+ a a
        if chi != 0:
            term_kerr = 2.0 * chi * (xp.abs(alpha) ** 2)
        else:
            term_kerr = 0.0

        dalpha = (
            -(1j * omega_a + kappa_a / 2.0) * alpha
            - 1j * term_kerr * alpha
            - 1j * g_ab * beta
            - 1j * g_ac * gamma
        )

        # Mode b (Gain: +kappa_b / 2)
        # dβ/dt = -(iω_b - κ_b/2)β - ig_ab α
        dbeta = -(1j * omega_b - kappa_b / 2.0) * beta - 1j * g_ab * alpha

        # Mode c (Loss: -kappa_c / 2)
        # dγ/dt = -(iω_c + κ_c/2)γ - ig_ac α
        dgamma = -(1j * omega_c + kappa_c / 2.0) * gamma - 1j * g_ac * alpha

        out = xp.empty_like(y)
        out[:, 0] = dalpha
        out[:, 1] = dbeta
        out[:, 2] = dgamma
        return out

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix (Block Diagonal in Real Basis)."""
        xp = get_xp(y)
        alpha = y[:, 0]

        chi = p["chi"]
        kappa_a = p["kappa_a"]
        kappa_b = p["kappa_b"]
        kappa_c = p["kappa_c"]

        n = y.shape[0]

        # --- Block A (alpha) ---
        # D_aa = -2i * chi * alpha^2
        # D_aa* = kappa_a/2 ? No, text says D = diag(kappa_a, kappa_b,
        # kappa_c) implies E[|dW|^2] ~ kappa dt. Standard P-rep diffusion
        # for loss is zero unless text implies thermal or additive noise.
        # Assuming Text convention: D_aa* = kappa_a

        M_a = -2j * chi * (alpha**2)
        # Normal diffusion for cavity a comes from its loss rate kappa_a
        D_a_val = kappa_a

        # Stability fix: enforce D > |M| if D is used to regularize.
        # If D_a_val is small/zero, this simulation likely unstable for
        # P-rep without Positive-P. User is responsible for parameters,
        # but we can apply scaling if D_a > 0
        if D_a_val > 1e-9:
            abs_M = xp.abs(M_a)
            violation_mask = abs_M > D_a_val
            if xp.any(violation_mask):
                safe_factor = 0.999
                eps_div = 1e-16
                scale = xp.where(
                    violation_mask, (safe_factor * D_a_val) / (abs_M + eps_div), 1.0
                )
                M_a = M_a * scale

        # Construct 2x2 covariance for Real/Imag
        # Cov_xx = 0.5 * (D + Re(M))
        # Cov_yy = 0.5 * (D - Re(M))
        # Cov_xy = 0.5 * Im(M)

        ReM = xp.real(M_a)
        ImM = xp.imag(M_a)

        Sig_xx = 0.5 * (D_a_val + ReM)
        Sig_yy = 0.5 * (D_a_val - ReM)
        Sig_xy = 0.5 * ImM

        # Cholesky of [[Sig_xx, Sig_xy], [Sig_xy, Sig_yy]]
        # L11 = sqrt(Sig_xx)
        # L21 = Sig_xy / L11
        # L22 = sqrt(Sig_yy - L21^2)

        # Ensure positivity (clip)
        Sig_xx = xp.clip(Sig_xx, 0.0, None)

        L11 = xp.sqrt(Sig_xx)
        # Avoid division by zero
        L11_safe = L11 + 1e-16
        L21 = Sig_xy / L11_safe
        L21 = xp.where(L11 < 1e-9, 0.0, L21)  # If Sig_xx ~ 0, L21 should be 0

        term_sq = Sig_yy - L21**2
        term_sq = xp.clip(term_sq, 0.0, None)
        L22 = xp.sqrt(term_sq)

        # --- Block B (beta) ---
        # Active Gain cavity b: D_bb* = kappa_b
        noise_b = xp.sqrt(0.5 * kappa_b)
        if not hasattr(noise_b, "shape") or noise_b.shape != (n,):
            noise_b = xp.full((n,), float(noise_b))

        # --- Block C (gamma) ---
        # Lossy cavity c: D_cc* = kappa_c
        noise_c = xp.sqrt(0.5 * kappa_c)
        if not hasattr(noise_c, "shape") or noise_c.shape != (n,):
            noise_c = xp.full((n,), float(noise_c))

        # Output shape: (n_traj, n_modes, noise_dim) -> (n, 3, 6)
        diffusion_matrix = xp.zeros((n, 3, 6), dtype=y.dtype)

        # Mode a (index 0) uses noise channels 0 and 1
        # L1 = L11 + i*L21
        # L2 = i*L22
        L1_a = L11 + 1j * L21
        L2_a = 1j * L22
        diffusion_matrix[:, 0, 0] = L1_a
        diffusion_matrix[:, 0, 1] = L2_a

        # Mode b (index 1) uses noise channels 2 and 3
        # Isotropic noise: L1 = noise_b, L2 = i*noise_b
        diffusion_matrix[:, 1, 2] = noise_b
        diffusion_matrix[:, 1, 3] = 1j * noise_b

        # Mode c (index 2) uses noise channels 4 and 5
        # Isotropic noise: L1 = noise_c, L2 = i*noise_c
        diffusion_matrix[:, 2, 4] = noise_c
        diffusion_matrix[:, 2, 5] = 1j * noise_c

        return diffusion_matrix

    def to_diffusive_sde_model(self) -> FunctionalSDEModel:
        return FunctionalSDEModel(
            name=self.name,
            n_modes=self.n_modes,
            noise_basis=self.noise_basis,
            noise_dim=self.noise_dim,
            params=self.params,
            drift=self.drift,
            diffusion=self.diffusion,
        )
