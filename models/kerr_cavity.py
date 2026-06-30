"""Kerr Cavity Model (Itô SDE).

This module defines a model plugin for simulating a single-mode Kerr nonlinear cavity.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel


class KerrCavityConfig(BaseModel):
    """Configuration schema for the Kerr cavity model."""

    delta: float = Field(description="Detuning")
    chi: float = Field(description="Kerr nonlinearity")
    epsilon: float = Field(description="Drive strength")
    gamma: float = Field(description="Damping rate")


class KerrCavityModel:
    """Single-mode Kerr nonlinear cavity (Itô SDE).

    This class implements the Plugin protocol and the SDEModel protocol.
    """

    name: ClassVar[str] = "kerr_cavity"
    description: ClassVar[str] = "Single-mode Kerr nonlinear cavity"
    config_schema: ClassVar[type[KerrCavityConfig]] = KerrCavityConfig

    def __init__(self, config: KerrCavityConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = KerrCavityConfig(**kwargs)
        self.config = config
        self._params = config.model_dump()

    @property
    def n_modes(self) -> int:
        return 1

    @property
    def noise_basis(self) -> str:
        return "complex"

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
        delta = p["delta"]
        chi = p["chi"]
        epsilon = p["epsilon"]
        gamma = p["gamma"]

        # dα = (-iΔ - γ/2 - iχ|α|^2)α dt + ε dt
        dalpha = (
            -1j * delta - gamma / 2.0 - 1j * chi * xp.abs(alpha) ** 2
        ) * alpha + epsilon

        out = xp.empty_like(y)
        out[:, 0] = dalpha
        return out

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix."""
        xp = get_xp(y)
        gamma = p["gamma"]

        # Additive noise for simple damping: sqrt(gamma/2)
        n = y.shape[0]
        Lc = xp.zeros((n, 1, 1), dtype=y.dtype)
        Lc[:, 0, 0] = (gamma / 2.0) ** 0.5
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
