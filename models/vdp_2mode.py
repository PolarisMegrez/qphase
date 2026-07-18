"""Two-mode van der Pol Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig, SDEModelPlugin
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import VDP2ModeCayleyCuPyKernel
from .kernels.euler_maruyama import VDP2ModeEulerCuPyKernel


class VDP2ModeConfig(ModelConfig):
    """Configuration for the two-mode van der Pol model."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    gamma_a: Any = Field(json_schema_extra={"scanable": True})
    gamma_b: Any = Field(json_schema_extra={"scanable": True})
    Gamma: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})


class VDP2ModeModel(SDEModelPlugin):
    """Noisy two-mode van der Pol oscillator."""

    name: ClassVar[str] = "vdp_2mode"
    description: ClassVar[str] = "Two-mode van der Pol oscillator"
    config_schema: ClassVar[type[VDP2ModeConfig]] = VDP2ModeConfig
    mode_count: ClassVar[int] = 2

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (VDP2ModeEulerCuPyKernel(), VDP2ModeCayleyCuPyKernel())

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        alpha, beta = y[:, 0], y[:, 1]
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        coupling = self.parameter(params, "g", xp)

        out = xp.empty_like(y)
        out[:, 0] = (
            -1j * omega_a
            + gamma_a / 2.0
            + nonlinear_gain * (1.0 - xp.abs(alpha) ** 2)
        ) * alpha - 1j * coupling * beta
        out[:, 1] = (-1j * omega_b - gamma_b / 2.0) * beta - 1j * coupling * alpha
        return out

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        alpha = y[:, 0]
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        coupling = self.parameter(params, "g", xp)

        matrix = xp.zeros((y.shape[0], 2, 2), dtype=y.dtype)
        matrix[:, 0, 0] = (
            -1j * omega_a
            + gamma_a / 2.0
            + nonlinear_gain * (1.0 - xp.abs(alpha) ** 2)
        )
        matrix[:, 0, 1] = -1j * coupling
        matrix[:, 1, 0] = -1j * coupling
        matrix[:, 1, 1] = -1j * omega_b - gamma_b / 2.0
        return matrix

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        alpha = y[:, 0]
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        return self.diagonal_complex_diffusion(
            y,
            (
                gamma_a / 2.0
                + nonlinear_gain * (2.0 * xp.abs(alpha) ** 2 - 1.0),
                gamma_b / 2.0,
            ),
        )
