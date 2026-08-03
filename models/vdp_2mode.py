"""Two-mode van der Pol Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
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
    steady_state_capacity: ClassVar[int] = 4

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

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        coupling = self.parameter(params, "g", xp)
        r_aa = xp.real(state[..., 0, 0])
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = omega_a + 1j * (
            gamma_a / 2.0 + nonlinear_gain * (1.0 - r_aa)
        )
        matrix[..., 0, 1] = coupling
        matrix[..., 1, 0] = coupling
        matrix[..., 1, 1] = omega_b - 1j * gamma_b / 2.0
        return matrix

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        r_aa = xp.real(state[..., 0, 0])
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = (
            gamma_a / 2.0 + nonlinear_gain * (2.0 * r_aa - 1.0)
        )
        matrix[..., 1, 1] = gamma_b / 2.0
        return matrix

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the CAM residual directly in canonical real coordinates."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        r_aa, r_bb, r_ab_real, r_ab_imag = (
            vector[..., index] for index in range(4)
        )
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        coupling = self.parameter(params, "g", xp)
        common = (
            (gamma_a - gamma_b) / 2.0
            - nonlinear_gain * (r_aa - 1.0)
        )
        residual = xp.empty_like(vector)
        residual[..., 0] = (
            gamma_a * r_aa
            + 2.0 * nonlinear_gain * r_aa * (2.0 - r_aa)
            + gamma_a / 2.0
            - nonlinear_gain
            - 2.0 * coupling * r_ab_imag
        )
        residual[..., 1] = -gamma_b * r_bb + gamma_b / 2.0 + 2.0 * coupling * r_ab_imag
        residual[..., 2] = (
            common * r_ab_real + (omega_a - omega_b) * r_ab_imag
        )
        residual[..., 3] = (
            common * r_ab_imag
            + (omega_b - omega_a) * r_ab_real
            + coupling * (r_aa - r_bb)
        )
        return residual

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the analytic CAM Jacobian without rebuilding a matrix."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        nonlinear_gain = self.parameter(params, "Gamma", xp)
        coupling = self.parameter(params, "g", xp)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        r_aa = vector[..., 0]
        r_ab_real = vector[..., 2]
        r_ab_imag = vector[..., 3]
        common = (
            (gamma_a - gamma_b) / 2.0
            - nonlinear_gain * (r_aa - 1.0)
        )
        jacobian = xp.zeros(vector.shape[:-1] + (4, 4), dtype=vector.dtype)
        jacobian[..., 0, 0] = gamma_a - 4.0 * nonlinear_gain * (r_aa - 1.0)
        jacobian[..., 0, 3] = -2.0 * coupling
        jacobian[..., 1, 1] = -gamma_b
        jacobian[..., 1, 3] = 2.0 * coupling
        jacobian[..., 2, 0] = -nonlinear_gain * r_ab_real
        jacobian[..., 2, 2] = common
        jacobian[..., 2, 3] = omega_a - omega_b
        jacobian[..., 3, 0] = coupling - nonlinear_gain * r_ab_imag
        jacobian[..., 3, 1] = -coupling
        jacobian[..., 3, 2] = omega_b - omega_a
        jacobian[..., 3, 3] = common
        return jacobian

    def cam_jacobian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = xp.stack(
            (
                xp.real(state[..., 0, 0]),
                xp.real(state[..., 1, 1]),
                xp.real(state[..., 0, 1]),
                xp.imag(state[..., 0, 1]),
            ),
            axis=-1,
        )
        return self.cam_jacobian_vector(vector, params)

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

        a, b = boson_modes("a", "b")
        parameters = sp.symbols(
            "omega_a omega_b gamma_a gamma_b Gamma g", real=True
        )
        omega_a, omega_b, gamma_a, gamma_b, nonlinear_gain, coupling = parameters
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + coupling * (a.dag * b + a * b.dag)
            ),
            channels=(
                LindbladChannel(a.dag, gamma_a),
                LindbladChannel(a**2, nonlinear_gain),
                LindbladChannel(b, gamma_b),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=parameters,
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

        state, symbols = symbolic_hermitian_matrix(2)
        omega_a, omega_b, gamma_a, gamma_b, nonlinear_gain, coupling = sp.symbols(
            "omega_a omega_b gamma_a gamma_b Gamma g", real=True
        )
        r_aa = state[0, 0]
        hamiltonian = sp.Matrix(
            [
                [
                    omega_a
                    + sp.I * (gamma_a / 2 + nonlinear_gain * (1 - r_aa)),
                    coupling,
                ],
                [coupling, omega_b - sp.I * gamma_b / 2],
            ]
        )
        diffusion = sp.diag(
            gamma_a / 2 + nonlinear_gain * (2 * r_aa - 1), gamma_b / 2
        )
        return CAMSymbolicSpec(
            hamiltonian,
            diffusion,
            state,
            symbols,
            (omega_a, omega_b, gamma_a, gamma_b, nonlinear_gain, coupling),
        )
