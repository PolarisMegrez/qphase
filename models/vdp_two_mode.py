"""Two-mode van der Pol coupled cavities (Itô SDE) with complex noise basis.

This module defines a model plugin for simulating two coupled optical cavities
with van der Pol-type nonlinearity.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase_sde.core.xputil import get_xp


class VDPTwoModeConfig(BaseModel):
    """Configuration schema for the VDP two-mode model.

    Parameters
    ----------
    omega_a : float
        Frequency of mode a.
    omega_b : float
        Frequency of mode b.
    gamma_a : float
        Damping rate of mode a.
    gamma_b : float
        Damping rate of mode b.
    Gamma : float
        Nonlinear gain coefficient.
    g : float
        Coupling strength between modes.
    D : float
        Diffusion coefficient.

    """

    omega_a: float = Field(description="Frequency of mode a")
    omega_b: float = Field(description="Frequency of mode b")
    gamma_a: float = Field(description="Damping rate of mode a")
    gamma_b: float = Field(description="Damping rate of mode b")
    Gamma: float = Field(description="Nonlinear gain coefficient")
    g: float = Field(description="Coupling strength between modes")
    D: float = Field(default=1.0, description="Diffusion coefficient")


class VDPTwoMode:
    """Two-mode van der Pol coupled cavities (Itô SDE) with complex noise basis.

    dα/dt = [-i ω_a + γ_a/2 + Γ(1 - |α|^2)] α - i g β + ξ_α
    dβ/dt = [-i ω_b - γ_b/2] β - i g α + ξ_β

    Diffusion (complex basis, diagonal):
      D_α^c = D [ γ_a/2 + Γ(2|α|^2 - 1) ]
      D_β^c = D γ_b/2
    We set L_c = diag( sqrt(max(D_α^c, 0)), sqrt(max(D_β^c, 0)) ).
    """

    name: ClassVar[str] = "vdp_two_mode"
    description: ClassVar[str] = (
        "Two-mode van der Pol coupled cavities (Ito SDE) with complex noise basis"
    )
    config_schema: ClassVar[type[Any]] = VDPTwoModeConfig

    def __init__(
        self, config: VDPTwoModeConfig | dict | None = None, **kwargs: Any
    ) -> None:
        """Initialize the VDP two-mode model.

        Parameters
        ----------
        config : VDPTwoModeConfig | dict | None
            Configuration for the model. Can be a VDPTwoModeConfig instance,
            a dictionary, or None (uses defaults).
        **kwargs : Any
            Additional keyword arguments for future extensibility.

        """
        if config is None:
            config = VDPTwoModeConfig()
        elif isinstance(config, dict):
            config = VDPTwoModeConfig.model_validate(config)
        self.config = config

        # Store parameters as dict for drift/diffusion functions
        self._params = config.model_dump()

    @property
    def n_modes(self) -> int:
        """Number of modes in the system."""
        return 2

    @property
    def noise_basis(self) -> str:
        """Noise basis type."""
        return "complex"

    @property
    def noise_dim(self) -> int:
        """Dimension of the noise."""
        return 2

    @property
    def params(self) -> dict:
        """Model parameters as dictionary."""
        return self._params

    def drift(self, y, t: float, p: dict):
        """Compute drift term.

        Parameters
        ----------
        y : array
            State vector of shape (n_traj, 2) complex, y[:,0]=α, y[:,1]=β
        t : float
            Current time
        p : dict
            Parameters dictionary

        Returns
        -------
        array
            Drift vector of same shape as y

        """
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

    def diffusion(self, y, t: float, p: dict):
        """Compute diffusion term.

        Parameters
        ----------
        y : array
            State vector of shape (n_traj, 2) complex
        t : float
            Current time
        p : dict
            Parameters dictionary

        Returns
        -------
        array
            Diffusion matrix of shape (n_traj, 2, 2)

        """
        xp = get_xp(y)
        alpha = y[:, 0]
        gamma_a = p["gamma_a"]
        gamma_b = p["gamma_b"]
        Gamma = p["Gamma"]
        D = p["D"]

        D_alpha = D * (gamma_a / 2.0 + Gamma * (2.0 * xp.abs(alpha) ** 2 - 1.0))
        D_beta = D * (gamma_b / 2.0)

        # Ensure diffusion arrays are backend arrays with shape (n_traj,)
        n = y.shape[0]
        if not hasattr(D_alpha, "clip"):
            D_alpha = xp.full((n,), float(D_alpha))
        if not hasattr(D_beta, "clip"):
            D_beta = xp.full((n,), float(D_beta))

        # Clip to non-negative for numerical stability
        D_alpha = xp.clip(D_alpha, 0.0, None)
        D_beta = xp.clip(D_beta, 0.0, None)

        Lc = xp.zeros((y.shape[0], 2, 2), dtype=y.dtype)
        Lc[:, 0, 0] = xp.sqrt(D_alpha)
        Lc[:, 1, 1] = xp.sqrt(D_beta)
        return Lc

    @property
    def diffusion_jacobian(self):
        """Jacobian of diffusion (not implemented)."""
        return None
