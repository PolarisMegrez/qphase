"""Van der Pol Oscillator - Level 2 (Phase Space Model) Plugin
------------------------------------------------------------
Defines the VDP model using Kramers-Moyal coefficients (Drift D1 and Diffusion D2).
Implemented as a QPhase Plugin.
"""

from typing import Any, ClassVar

from pydantic import BaseModel, Field
from qphase.backend.xputil import get_xp
from qphase_sde.model import PhaseSpaceModel


class VDPLevel2Config(BaseModel):
    """Configuration for VDP Level 2 Model."""

    omega_a: float = Field(description="Frequency of mode a")
    omega_b: float = Field(description="Frequency of mode b")
    gamma_a: float = Field(description="Damping rate of mode a")
    gamma_b: float = Field(description="Damping rate of mode b")
    Gamma: float = Field(description="Nonlinear gain coefficient")
    g: float = Field(description="Coupling strength between modes")
    D: float = Field(default=1.0, description="Diffusion coefficient")


class VDPLevel2Model:
    """Van der Pol Oscillator - Level 2 (Phase Space Model).

    This class implements the Plugin protocol and acts as a factory/container
    for the PhaseSpaceModel data.
    """

    name: ClassVar[str] = "vdp_level2"
    description: ClassVar[str] = "Van der Pol Oscillator (Phase Space Model)"
    config_schema: ClassVar[type[VDPLevel2Config]] = VDPLevel2Config

    def __init__(self, config: VDPLevel2Config | None = None, **kwargs: Any) -> None:
        if config is None:
            config = VDPLevel2Config(**kwargs)
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

    @property
    def terms(self) -> dict[int, Any]:
        return {1: self._drift, 2: self._diffusion_d2}

    def drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        return self._drift(y, t, p)

    def diffusion(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Matrix L from D2."""
        # D2 is (n, 2) vector of diffusion coefficients (diagonal)
        # We assume D2 = diag(L L^dagger) / 2 ?
        # Or D2 is the diffusion coefficient in FPE.
        # If dx = A dt + B dW, then D2 = B B^T / 2.
        # So B = sqrt(2 * D2).
        # Let's use sqrt(D2) to match vdp_two_mode.py which uses sqrt(D_alpha).
        # vdp_two_mode.py: D_alpha = D * (gamma/2 + ...).
        # Lc = sqrt(D_alpha).
        # So it seems D2 in vdp_two_mode corresponds to D_alpha.
        # I will use sqrt(D2).
        
        d2 = self._diffusion_d2(y, t, p) # (n, 2)
        xp = get_xp(y)
        
        n = y.shape[0]
        Lc = xp.zeros((n, 2, 2), dtype=y.dtype)
        Lc[:, 0, 0] = xp.sqrt(d2[:, 0])
        Lc[:, 1, 1] = xp.sqrt(d2[:, 1])
        return Lc

    def _drift(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Drift Vector D1(alpha)."""
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

    def _diffusion_d2(self, y: Any, t: float, p: dict[str, Any]) -> Any:
        """Compute Diffusion Coefficient D2(alpha)."""
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

        return xp.stack([D_alpha, D_beta], axis=1)

    def to_phase_space_model(self) -> PhaseSpaceModel:
        """Convert to the standard PhaseSpaceModel dataclass."""
        return PhaseSpaceModel(
            name=self.name, n_modes=self.n_modes, terms=self.terms, params=self.params
        )
