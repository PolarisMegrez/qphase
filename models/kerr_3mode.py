"""Three-mode Kerr Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig, SDEModelPlugin
from .kernels.base import ModelKernelPlugin
from .kernels.euler_maruyama import Kerr3ModeEulerCuPyKernel


class Kerr3ModeConfig(ModelConfig):
    """Configuration for the three-mode Kerr model."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    omega_c: Any = Field(json_schema_extra={"scanable": True})
    chi: Any = Field(json_schema_extra={"scanable": True})
    gamma_a: Any = Field(json_schema_extra={"scanable": True})
    gamma_b: Any = Field(json_schema_extra={"scanable": True})
    gamma_c: Any = Field(json_schema_extra={"scanable": True})
    g_ab: Any = Field(json_schema_extra={"scanable": True})
    g_ac: Any = Field(json_schema_extra={"scanable": True})


class Kerr3ModeModel(SDEModelPlugin):
    """Three coupled modes with Kerr nonlinearity in mode a."""

    name: ClassVar[str] = "kerr_3mode"
    description: ClassVar[str] = "Three-mode Kerr oscillator"
    config_schema: ClassVar[type[Kerr3ModeConfig]] = Kerr3ModeConfig
    mode_count: ClassVar[int] = 3
    steady_state_capacity: ClassVar[int] = 8

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (Kerr3ModeEulerCuPyKernel(),)

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        matrix = self.drift_matrix(y, 0.0, params)
        return xp.einsum("...ij,...j->...i", matrix, y)

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        omega_c = self.parameter(params, "omega_c", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        coupling_ab = self.parameter(params, "g_ab", xp)
        coupling_ac = self.parameter(params, "g_ac", xp)

        matrix = xp.zeros((y.shape[0], 3, 3), dtype=y.dtype)
        matrix[:, 0, 0] = (
            -gamma_a / 2.0
            - 1j * omega_a
            - 2j * chi * (xp.abs(y[:, 0]) ** 2 - 1.0)
        )
        matrix[:, 0, 1] = -1j * coupling_ab
        matrix[:, 0, 2] = -1j * coupling_ac
        matrix[:, 1, 0] = -1j * coupling_ab
        matrix[:, 1, 1] = -gamma_b / 2.0 - 1j * omega_b
        matrix[:, 2, 0] = -1j * coupling_ac
        matrix[:, 2, 2] = gamma_c / 2.0 - 1j * omega_c
        return matrix

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        return self.diagonal_complex_diffusion(
            y, (gamma_a / 2.0, gamma_b / 2.0, gamma_c / 2.0)
        )

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        omega_c = self.parameter(params, "omega_c", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        coupling_ab = self.parameter(params, "g_ab", xp)
        coupling_ac = self.parameter(params, "g_ac", xp)
        r_aa = xp.real(state[..., 0, 0])
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = (
            omega_a + 2.0 * chi * (r_aa - 1.0) - 1j * gamma_a / 2.0
        )
        matrix[..., 0, 1] = coupling_ab
        matrix[..., 0, 2] = coupling_ac
        matrix[..., 1, 0] = coupling_ab
        matrix[..., 1, 1] = omega_b - 1j * gamma_b / 2.0
        matrix[..., 2, 0] = coupling_ac
        matrix[..., 2, 2] = omega_c + 1j * gamma_c / 2.0
        return matrix

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = gamma_a / 2.0
        matrix[..., 1, 1] = gamma_b / 2.0
        matrix[..., 2, 2] = gamma_c / 2.0
        return matrix

    @classmethod
    @lru_cache(maxsize=1)
    def cam_symbolic_matrices(cls) -> Any:
        import sympy as sp
        from qphase_cam.core.coordinates import symbolic_hermitian_matrix
        from qphase_cam.model import CAMSymbolicSpec

        state, symbols = symbolic_hermitian_matrix(3)
        omega_a, omega_b, omega_c, chi = sp.symbols(
            "omega_a omega_b omega_c chi", real=True
        )
        gamma_a, gamma_b, gamma_c = sp.symbols(
            "gamma_a gamma_b gamma_c", real=True
        )
        coupling_ab, coupling_ac = sp.symbols("g_ab g_ac", real=True)
        r_aa = state[0, 0]
        hamiltonian = sp.Matrix(
            [
                [
                    omega_a + 2 * chi * (r_aa - 1) - sp.I * gamma_a / 2,
                    coupling_ab,
                    coupling_ac,
                ],
                [coupling_ab, omega_b - sp.I * gamma_b / 2, 0],
                [coupling_ac, 0, omega_c + sp.I * gamma_c / 2],
            ]
        )
        diffusion = sp.diag(gamma_a / 2, gamma_b / 2, gamma_c / 2)
        return CAMSymbolicSpec(
            hamiltonian,
            diffusion,
            state,
            symbols,
            (
                omega_a,
                omega_b,
                omega_c,
                chi,
                gamma_a,
                gamma_b,
                gamma_c,
                coupling_ab,
                coupling_ac,
            ),
        )
