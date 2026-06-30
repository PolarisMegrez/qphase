"""Kerr 3PA Model (Itô SDE).

This module defines a model plugin for simulating a single-mode Kerr nonlinear cavity
with 3-photon absorption (Quintic dissipation).
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel

logger = logging.getLogger(__name__)


class Kerr3PAConfig(BaseModel):
    """Configuration schema for the Kerr 3PA model."""

    omega0: float = Field(
        description="Mode frequency",
        json_schema_extra={"scanable": True},
    )
    chi: float = Field(
        description="Kerr nonlinearity",
        json_schema_extra={"scanable": True},
    )
    kappa3: float = Field(
        description="3-photon absorption rate",
        json_schema_extra={"scanable": True},
    )
    beta: float = Field(
        description="Criticality scaling factor",
        json_schema_extra={"scanable": True},
    )
    epsilon: float = Field(
        description="Criticality parameter",
        json_schema_extra={"scanable": True},
    )
    kappa1: float = Field(
        default=5.0,
        description="Linear gain coefficient (kappa1)",
        json_schema_extra={"scanable": True},
    )


class Kerr3PAModel:
    """Single-mode Kerr nonlinear cavity with 3PA (Itô SDE).

    Dynamics:
    dα = (μ - iω₀)α dt - 2iχ|α|²α dt - 1.5κ₃|α|⁴α dt + ξ(t)

    where μ = βε.
    Noise correlations:
    <ξξ†> = D = κ₁ + 2βε
    <ξξ> = M = -(2iχ + 3κ₃|α|²)α²
    """

    name: ClassVar[str] = "kerr_3pa"
    description: ClassVar[str] = "Kerr cavity with 3-photon absorption"
    config_schema: ClassVar[type[Kerr3PAConfig]] = Kerr3PAConfig

    def __init__(self, config: Kerr3PAConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = Kerr3PAConfig(**kwargs)
        self.config = config
        self._params = config.model_dump()

    @property
    def n_modes(self) -> int:
        return 1

    @property
    def noise_basis(self) -> str:
        # We handle complex noise explicitly via 2 real channels to support anomalous diffusion M
        return "real"

    @property
    def noise_dim(self) -> int:
        return 2

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Drift Vector."""
        xp = get_xp(y)
        alpha = y[:, 0]

        omega0 = p["omega0"]
        chi = p["chi"]
        kappa3 = p["kappa3"]
        beta = p["beta"]
        epsilon = p["epsilon"]

        mu = beta * epsilon
        abs_alpha_sq = xp.abs(alpha) ** 2

        # dα = (μ - iω₀)α - 2iχ|α|²α - 1.5κ₃|α|⁴α
        #    = [ (μ - iω₀) - 2iχ|α|² - 1.5κ₃|α|⁴ ] α

        term_linear = mu - 1j * omega0
        term_kerr = -2j * chi * abs_alpha_sq
        term_3pa = -1.5 * kappa3 * (abs_alpha_sq ** 2)

        dalpha = (term_linear + term_kerr + term_3pa) * alpha

        out = xp.empty_like(y)
        out[:, 0] = dalpha
        return out

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix.
        
        Uses soft-clipping of anomalous diffusion M to ensure positive semi-definiteness.
        """
        xp = get_xp(y)
        alpha = y[:, 0]

        chi = p["chi"]
        kappa3 = p["kappa3"]
        kappa1 = p["kappa1"]
        beta = p["beta"]
        epsilon = p["epsilon"]

        # D = kappa1 + 2*beta*epsilon
        D = kappa1 + 2.0 * beta * epsilon
        
        # M = -(2iχ + 3κ₃|α|²)α²
        # Calculate M parameters
        abs_alpha_sq = xp.abs(alpha) ** 2
        term_M_pre = -(2j * chi + 3.0 * kappa3 * abs_alpha_sq)
        M = term_M_pre * (alpha ** 2)

        # --- Stability Fix: Enforce D > |M| ---
        # The P-representation Langevin equation is only valid when the diffusion matrix
        # is positive semi-definite, which requires D >= |M|.
        # In numerical simulations, fluctuations can push the state into invalid regions.
        # We regularize M to enforce PSD: if |M| > D, scale M down.
        
        abs_M = xp.abs(M)
        
        # Check violation
        violation_mask = abs_M > D
        if xp.any(violation_mask):
            # Log debug warning (once per call to avoid flooding, or checking first element)
            # To be efficient on GPU, we sync only if needed? 
            # For debugging, we can just print/log. But pulling from GPU is slow.
            # We'll just assume user accepts slow-down for safety/debug.
            try:
                # Check max violation
                max_diff = xp.max(abs_M - D)
                if hasattr(max_diff, "get"):
                     max_diff_val = float(max_diff.get())
                else:
                     max_diff_val = float(max_diff)

                if max_diff_val > 1e-6:
                     logger.debug(f"[kerr_3pa] PSD violation: max(|M|-D)={max_diff_val:.4e}. Scaling applied.")
            except Exception:
                pass

        # Apply scaling ONLY where violation occurs (or min logic handles it effectively)
        # Using min(1.0, D/|M|) is cleaner and differentiable-ish.
        # To be STRICT about "no modification if satisfied":
        # scale = 1.0 where |M| <= D
        # scale = D/|M| where |M| > D
        
        eps_div = 1e-16
        scale = xp.ones_like(abs_M)
        
        # Only Modify where violation_mask is True
        # Note: logical indexing might be slow or return copy.
        # xp.minimum(1.0, D / |M|) is numerically equivalent and vector-friendly.
        # But user asked: "ensure ... not any modification".
        # 1.0 vs 0.999999999 (numerical noise). 
        # Using 'where' is safest for "exact" constraint.
        # We also scale to 0.999 * D to be safely inside the PSD region when correcting.
        
        safe_factor = 0.999
        scale = xp.where(
            violation_mask, 
            (safe_factor * D) / (abs_M + eps_div),  # Scale factor when violated
            1.0                                     # Keep as 1.0 when valid
        )
        
        # Apply scaling to M
        M = M * scale

        # --- Cholesky Decomposition of Real Covariance Matrix ---
        # Sigma = [[A, B], [B, C]]
        # A = 0.5 * (D + Re(M))
        # C = 0.5 * (D - Re(M))
        # B = 0.5 * Im(M)

        ReM = xp.real(M)
        ImM = xp.imag(M)

        A = 0.5 * (D + ReM)
        C = 0.5 * (D - ReM)
        B = 0.5 * ImM

        # Cholesky L:
        # l11 = sqrt(A)
        # l21 = B / l11
        # l22 = sqrt(C - l21^2)

        # With scaling guaranteed D > |M| -> D > |ReM| -> A > 0 and C > 0.
        # Also Det = A*C - B^2 = (D^2 - |M|^2)/4 > 0.
        
        eps = 1e-12
        l11 = xp.sqrt(xp.maximum(A, eps))
        l21 = B / l11
        
        # term_l22_sq should be positive theoretically, but numerical noise might affect it
        term_l22_sq = xp.maximum(C - l21**2, 0.0)
        l22 = xp.sqrt(term_l22_sq)

        # Construct diffusion coefficients for complex update:
        # L1 = l11 + i*l21
        # L2 = i*l22

        L1 = l11 + 1j * l21
        L2 = 1j * l22

        # Output shape: (n_traj, n_modes, noise_dim) -> (n, 1, 2)
        n = y.shape[0]
        diffusion_matrix = xp.empty((n, 1, 2), dtype=y.dtype)

        diffusion_matrix[:, 0, 0] = L1
        diffusion_matrix[:, 0, 1] = L2

        return diffusion_matrix

    def to_diffusive_sde_model(self) -> FunctionalSDEModel:
        """Convert to the standard FunctionalSDEModel dataclass."""
        return FunctionalSDEModel(
            name=self.name,
            n_modes=self.n_modes,
            noise_basis=self.noise_basis,
            noise_dim=self.noise_dim,
            params=self.params,
            drift=self.drift,
            diffusion=self.diffusion,
        )
