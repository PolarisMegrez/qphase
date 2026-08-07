"""Fully connected three-mode Kerr Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import KerrFull3ModeCayleyCuPyKernel
from .kernels.euler_maruyama import KerrFull3ModeEulerCuPyKernel
from .kerr_3mode import Kerr3ModeConfig, Kerr3ModeModel


class KerrFull3ModeConfig(Kerr3ModeConfig):
    """Configuration for the fully connected three-mode Kerr model."""

    g_bc: Any = Field(json_schema_extra={"scanable": True})


class KerrFull3ModeModel(Kerr3ModeModel):
    """Three Kerr-network modes with all pairwise linear couplings."""

    name: ClassVar[str] = "kerr_full_3mode"
    description: ClassVar[str] = "Fully connected three-mode Kerr oscillator"
    config_schema: ClassVar[type[KerrFull3ModeConfig]] = KerrFull3ModeConfig

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (
            KerrFull3ModeEulerCuPyKernel(),
            KerrFull3ModeCayleyCuPyKernel(),
        )

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        matrix = super().drift_matrix(y, t, params)
        xp = get_xp(y)
        coupling_bc = self.parameter(params, "g_bc", xp)
        matrix[:, 1, 2] = -1j * coupling_bc
        matrix[:, 2, 1] = -1j * coupling_bc
        return matrix

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        matrix = super().cam_hamiltonian(state, params)
        xp = get_xp(state)
        coupling_bc = self.parameter(params, "g_bc", xp)
        matrix[..., 1, 2] = coupling_bc
        matrix[..., 2, 1] = coupling_bc
        return matrix

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        residual = super().cam_residual_vector(vector, params)
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        coupling_bc = self.parameter(params, "g_bc", xp)
        r_ab_real = vector[..., 3]
        r_ac_real = vector[..., 4]
        r_ab_imag = vector[..., 6]
        r_ac_imag = vector[..., 7]
        r_bc_imag = vector[..., 8]

        residual[..., 1] -= 2.0 * coupling_bc * r_bc_imag
        residual[..., 2] += 2.0 * coupling_bc * r_bc_imag
        residual[..., 3] -= coupling_bc * r_ac_imag
        residual[..., 4] -= coupling_bc * r_ab_imag
        residual[..., 6] += coupling_bc * r_ac_real
        residual[..., 7] += coupling_bc * r_ab_real
        residual[..., 8] += coupling_bc * (vector[..., 1] - vector[..., 2])
        return residual

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        jacobian = super().cam_jacobian_vector(vector, params)
        xp = get_xp(vector)
        coupling_bc = self.parameter(params, "g_bc", xp)

        jacobian[..., 1, 8] -= 2.0 * coupling_bc
        jacobian[..., 2, 8] += 2.0 * coupling_bc
        jacobian[..., 3, 7] -= coupling_bc
        jacobian[..., 4, 6] -= coupling_bc
        jacobian[..., 6, 4] += coupling_bc
        jacobian[..., 7, 3] += coupling_bc
        jacobian[..., 8, 1] += coupling_bc
        jacobian[..., 8, 2] -= coupling_bc
        return jacobian

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        """Return exact factorized normal-moment dynamics for bifurcation work."""
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b, c = boson_modes("a", "b", "c")
        parameters = sp.symbols(
            "omega_a omega_b omega_c chi gamma_a gamma_b gamma_c "
            "g_ab g_ac g_bc",
            real=True,
        )
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
            coupling_bc,
        ) = parameters
        master = MasterEquation(
            modes=(a, b, c),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + omega_c * c.dag * c
                + coupling_ab * (a.dag * b + a * b.dag)
                + coupling_ac * (a.dag * c + a * c.dag)
                + coupling_bc * (b.dag * c + b * c.dag)
                + chi * a.dag**2 * a**2
            ),
            channels=(
                LindbladChannel(a, gamma_a),
                LindbladChannel(b, gamma_b),
                LindbladChannel(c.dag, gamma_c),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=parameters,
                parameter_domains={
                    chi: "nonnegative",
                    coupling_ab: "nonnegative",
                    coupling_ac: "nonnegative",
                    coupling_bc: "nonnegative",
                    gamma_a: "nonnegative",
                    gamma_b: "nonnegative",
                    gamma_c: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )

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
        gamma_a, gamma_b, gamma_c = sp.symbols("gamma_a gamma_b gamma_c", real=True)
        coupling_ab, coupling_ac, coupling_bc = sp.symbols(
            "g_ab g_ac g_bc", real=True
        )
        r_aa = state[0, 0]
        hamiltonian = sp.Matrix(
            [
                [
                    omega_a + 2 * chi * (r_aa - 1) - sp.I * gamma_a / 2,
                    coupling_ab,
                    coupling_ac,
                ],
                [coupling_ab, omega_b - sp.I * gamma_b / 2, coupling_bc],
                [coupling_ac, coupling_bc, omega_c + sp.I * gamma_c / 2],
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
                coupling_bc,
            ),
        )
