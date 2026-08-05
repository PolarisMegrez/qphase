"""Three-mode Kerr Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig, SDEModelPlugin
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import Kerr3ModeCayleyCuPyKernel
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
        return (Kerr3ModeEulerCuPyKernel(), Kerr3ModeCayleyCuPyKernel())

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
            -gamma_a / 2.0 - 1j * omega_a - 2j * chi * (xp.abs(y[:, 0]) ** 2 - 1.0)
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
        matrix[..., 0, 0] = omega_a + 2.0 * chi * (r_aa - 1.0) - 1j * gamma_a / 2.0
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

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the CAM residual directly in canonical real coordinates."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        (
            r_aa,
            r_bb,
            r_cc,
            r_ab_real,
            r_ac_real,
            r_bc_real,
            r_ab_imag,
            r_ac_imag,
            r_bc_imag,
        ) = (vector[..., index] for index in range(9))
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        omega_c = self.parameter(params, "omega_c", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        coupling_ab = self.parameter(params, "g_ab", xp)
        coupling_ac = self.parameter(params, "g_ac", xp)
        nonlinear_frequency = omega_a + 2.0 * chi * (r_aa - 1.0)
        detuning_ab = nonlinear_frequency - omega_b
        detuning_ac = nonlinear_frequency - omega_c
        detuning_bc = omega_b - omega_c
        decay_ab = -(gamma_a + gamma_b) / 2.0
        decay_ac = (gamma_c - gamma_a) / 2.0
        decay_bc = (gamma_c - gamma_b) / 2.0

        residual = xp.empty_like(vector)
        residual[..., 0] = (
            -gamma_a * r_aa
            - 2.0 * coupling_ab * r_ab_imag
            - 2.0 * coupling_ac * r_ac_imag
            + gamma_a / 2.0
        )
        residual[..., 1] = (
            -gamma_b * r_bb + 2.0 * coupling_ab * r_ab_imag + gamma_b / 2.0
        )
        residual[..., 2] = (
            gamma_c * r_cc + 2.0 * coupling_ac * r_ac_imag + gamma_c / 2.0
        )
        residual[..., 3] = (
            decay_ab * r_ab_real + detuning_ab * r_ab_imag - coupling_ac * r_bc_imag
        )
        residual[..., 4] = (
            decay_ac * r_ac_real + detuning_ac * r_ac_imag + coupling_ab * r_bc_imag
        )
        residual[..., 5] = (
            decay_bc * r_bc_real
            + detuning_bc * r_bc_imag
            + coupling_ab * r_ac_imag
            + coupling_ac * r_ab_imag
        )
        residual[..., 6] = (
            decay_ab * r_ab_imag
            - detuning_ab * r_ab_real
            + coupling_ab * (r_aa - r_bb)
            - coupling_ac * r_bc_real
        )
        residual[..., 7] = (
            decay_ac * r_ac_imag
            - detuning_ac * r_ac_real
            + coupling_ac * (r_aa - r_cc)
            - coupling_ab * r_bc_real
        )
        residual[..., 8] = (
            decay_bc * r_bc_imag
            - detuning_bc * r_bc_real
            - coupling_ab * r_ac_real
            + coupling_ac * r_ab_real
        )
        return residual

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the analytic CAM Jacobian without rebuilding a matrix."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        omega_c = self.parameter(params, "omega_c", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        gamma_c = self.parameter(params, "gamma_c", xp)
        coupling_ab = self.parameter(params, "g_ab", xp)
        coupling_ac = self.parameter(params, "g_ac", xp)
        r_aa = vector[..., 0]
        r_ab_real = vector[..., 3]
        r_ac_real = vector[..., 4]
        r_ab_imag = vector[..., 6]
        r_ac_imag = vector[..., 7]
        nonlinear_frequency = omega_a + 2.0 * chi * (r_aa - 1.0)
        detuning_ab = nonlinear_frequency - omega_b
        detuning_ac = nonlinear_frequency - omega_c
        detuning_bc = omega_b - omega_c
        decay_ab = -(gamma_a + gamma_b) / 2.0
        decay_ac = (gamma_c - gamma_a) / 2.0
        decay_bc = (gamma_c - gamma_b) / 2.0

        jacobian = xp.zeros(vector.shape[:-1] + (9, 9), dtype=vector.dtype)
        jacobian[..., 0, 0] = -gamma_a
        jacobian[..., 0, 6] = -2.0 * coupling_ab
        jacobian[..., 0, 7] = -2.0 * coupling_ac
        jacobian[..., 1, 1] = -gamma_b
        jacobian[..., 1, 6] = 2.0 * coupling_ab
        jacobian[..., 2, 2] = gamma_c
        jacobian[..., 2, 7] = 2.0 * coupling_ac

        jacobian[..., 3, 0] = 2.0 * chi * r_ab_imag
        jacobian[..., 3, 3] = decay_ab
        jacobian[..., 3, 6] = detuning_ab
        jacobian[..., 3, 8] = -coupling_ac
        jacobian[..., 4, 0] = 2.0 * chi * r_ac_imag
        jacobian[..., 4, 4] = decay_ac
        jacobian[..., 4, 7] = detuning_ac
        jacobian[..., 4, 8] = coupling_ab
        jacobian[..., 5, 5] = decay_bc
        jacobian[..., 5, 6] = coupling_ac
        jacobian[..., 5, 7] = coupling_ab
        jacobian[..., 5, 8] = detuning_bc

        jacobian[..., 6, 0] = coupling_ab - 2.0 * chi * r_ab_real
        jacobian[..., 6, 1] = -coupling_ab
        jacobian[..., 6, 3] = -detuning_ab
        jacobian[..., 6, 5] = -coupling_ac
        jacobian[..., 6, 6] = decay_ab
        jacobian[..., 7, 0] = coupling_ac - 2.0 * chi * r_ac_real
        jacobian[..., 7, 2] = -coupling_ac
        jacobian[..., 7, 4] = -detuning_ac
        jacobian[..., 7, 5] = -coupling_ab
        jacobian[..., 7, 7] = decay_ac
        jacobian[..., 8, 3] = coupling_ac
        jacobian[..., 8, 4] = -coupling_ab
        jacobian[..., 8, 5] = -detuning_bc
        jacobian[..., 8, 8] = decay_bc
        return jacobian

    def cam_jacobian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = xp.stack(
            (
                xp.real(state[..., 0, 0]),
                xp.real(state[..., 1, 1]),
                xp.real(state[..., 2, 2]),
                xp.real(state[..., 0, 1]),
                xp.real(state[..., 0, 2]),
                xp.real(state[..., 1, 2]),
                xp.imag(state[..., 0, 1]),
                xp.imag(state[..., 0, 2]),
                xp.imag(state[..., 1, 2]),
            ),
            axis=-1,
        )
        return self.cam_jacobian_vector(vector, params)

    def cam_bifurcation_scales(self, params: dict[str, Any]) -> dict[str, Any]:
        scale = max(1.0, 1.0 / max(abs(float(params["chi"])), 1.0e-12))
        return {"state": [scale] * 9, "source": "model:chi"}

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        """Return the exact fpgen normal-moment dynamics."""
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b, c = boson_modes("a", "b", "c")
        parameters = sp.symbols(
            "omega_a omega_b omega_c chi gamma_a gamma_b gamma_c g_ab g_ac",
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
        ) = parameters
        master = MasterEquation(
            modes=(a, b, c),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + omega_c * c.dag * c
                + coupling_ab * (a.dag * b + a * b.dag)
                + coupling_ac * (a.dag * c + a * c.dag)
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
