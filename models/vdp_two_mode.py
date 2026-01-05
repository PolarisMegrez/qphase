"""Two-mode van der Pol coupled cavities (Itô SDE) with complex noise basis.

This module defines a model plugin for simulating two coupled optical cavities
with van der Pol-type nonlinearity.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel


class VDPTwoModeConfig(BaseModel):
    """Configuration schema for the VDP two-mode model."""

    omega_a: float = Field(description="Frequency of mode a")
    omega_b: float = Field(description="Frequency of mode b")
    gamma_a: float = Field(description="Damping rate of mode a")
    gamma_b: float = Field(description="Damping rate of mode b")
    Gamma: float = Field(description="Nonlinear gain coefficient")
    g: float = Field(description="Coupling strength between modes")
    D: float = Field(default=1.0, description="Diffusion coefficient")


class VDPTwoModeModel:
    """Two-mode van der Pol coupled cavities (Itô SDE).

    This class implements the Plugin protocol and the SDEModel protocol.
    """

    name: ClassVar[str] = "vdp_two_mode"
    description: ClassVar[str] = (
        "Two-mode van der Pol coupled cavities (Ito SDE) with complex noise basis"
    )
    config_schema: ClassVar[type[VDPTwoModeConfig]] = VDPTwoModeConfig

    def __init__(self, config: VDPTwoModeConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = VDPTwoModeConfig(**kwargs)
        self.config = config
        self._params = config.model_dump()

    @property
    def n_modes(self) -> int:
        return 2

    @property
    def noise_basis(self) -> str:
        return "complex"

    @property
    def noise_dim(self) -> int:
        return 4

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Drift Vector."""
        xp = get_xp(y)
        alpha = y[:, 0]
        beta = y[:, 1]

        omega_a = p["omega_a"]
        omega_b = p["omega_b"]
        gamma_a = p["gamma_a"]
        gamma_b = p["gamma_b"]
        Gamma = p["Gamma"]
        g = p["g"]

        dalpha = (
            (-1j * omega_a) + (gamma_a / 2.0) + Gamma * (1.0 - xp.abs(alpha) ** 2)
        ) * alpha - 1j * g * beta
        dbeta = ((-1j * omega_b) - (gamma_b / 2.0)) * beta - 1j * g * alpha

        out = xp.empty_like(y)
        out[:, 0] = dalpha
        out[:, 1] = dbeta
        return out

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix."""
        xp = get_xp(y)
        alpha = y[:, 0]
        gamma_a = p["gamma_a"]
        gamma_b = p["gamma_b"]
        Gamma = p["Gamma"]
        D = p["D"]

        D_alpha = D * (gamma_a / 2.0 + Gamma * (2.0 * xp.abs(alpha) ** 2 - 1.0))
        D_beta = D * (gamma_b / 2.0)

        n = y.shape[0]
        if not hasattr(D_alpha, "shape") or D_alpha.shape != (n,):
            D_alpha = xp.full((n,), float(D_alpha))
        if not hasattr(D_beta, "shape") or D_beta.shape != (n,):
            D_beta = xp.full((n,), float(D_beta))

        D_alpha = xp.clip(D_alpha, 0.0, None)
        D_beta = xp.clip(D_beta, 0.0, None)

        Lc = xp.zeros((n, 2, 2), dtype=y.dtype)
        Lc[:, 0, 0] = xp.sqrt(D_alpha)
        Lc[:, 1, 1] = xp.sqrt(D_beta)

        return Lc

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
