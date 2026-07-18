"""Two-mode Kerr Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig, SDEModelPlugin
from .kernels.base import ModelKernelPlugin
from .kernels.euler_maruyama import Kerr2ModeEulerCuPyKernel


class Kerr2ModeConfig(ModelConfig):
    """Configuration for the two-mode Kerr model."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    chi: Any = Field(json_schema_extra={"scanable": True})
    gamma_a: Any = Field(json_schema_extra={"scanable": True})
    gamma_b: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})


class Kerr2ModeModel(SDEModelPlugin):
    """Two coupled modes with Kerr nonlinearity in mode a."""

    name: ClassVar[str] = "kerr_2mode"
    description: ClassVar[str] = "Two-mode Kerr oscillator"
    config_schema: ClassVar[type[Kerr2ModeConfig]] = Kerr2ModeConfig
    mode_count: ClassVar[int] = 2

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (Kerr2ModeEulerCuPyKernel(),)

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        alpha, beta = y[:, 0], y[:, 1]
        matrix = self.drift_matrix(y, 0.0, params)
        out = xp.empty_like(y)
        out[:, 0] = matrix[:, 0, 0] * alpha + matrix[:, 0, 1] * beta
        out[:, 1] = matrix[:, 1, 0] * alpha + matrix[:, 1, 1] * beta
        return out

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        coupling = self.parameter(params, "g", xp)

        matrix = xp.zeros((y.shape[0], 2, 2), dtype=y.dtype)
        matrix[:, 0, 0] = (
            gamma_a / 2.0
            - 1j * omega_a
            - 2j * chi * (xp.abs(y[:, 0]) ** 2 - 1.0)
        )
        matrix[:, 0, 1] = -1j * coupling
        matrix[:, 1, 0] = -1j * coupling
        matrix[:, 1, 1] = -gamma_b / 2.0 - 1j * omega_b
        return matrix

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        return self.diagonal_complex_diffusion(y, (gamma_a / 2.0, gamma_b / 2.0))
