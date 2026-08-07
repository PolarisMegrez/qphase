"""Two-mode Kerr Ito SDE model plugin."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig, SDEModelPlugin
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import Kerr2ModeCayleyCuPyKernel
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
    steady_state_capacity: ClassVar[int] = 3

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (Kerr2ModeEulerCuPyKernel(), Kerr2ModeCayleyCuPyKernel())

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
            gamma_a / 2.0 - 1j * omega_a - 2j * chi * (xp.abs(y[:, 0]) ** 2 - 1.0)
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

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        coupling = self.parameter(params, "g", xp)
        r_aa = xp.real(state[..., 0, 0])
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = omega_a + 2.0 * chi * (r_aa - 1.0) + 1j * gamma_a / 2.0
        matrix[..., 0, 1] = coupling
        matrix[..., 1, 0] = coupling
        matrix[..., 1, 1] = omega_b - 1j * gamma_b / 2.0
        return matrix

    def cam_bogoliubov_interaction(
        self, state: Any, params: dict[str, Any]
    ) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        interaction = xp.zeros_like(state, dtype=complex)
        interaction[..., 0, 0] = (
            2.0 * self.parameter(params, "chi", xp) * state[..., 0, 0]
        )
        return interaction

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = gamma_a / 2.0
        matrix[..., 1, 1] = gamma_b / 2.0
        return matrix

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the CAM residual directly in canonical real coordinates."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        r_aa, r_bb, r_ab_real, r_ab_imag = (vector[..., index] for index in range(4))
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        coupling = self.parameter(params, "g", xp)
        common = (gamma_a - gamma_b) / 2.0
        detuning = omega_a - omega_b + 2.0 * chi * (r_aa - 1.0)
        residual = xp.empty_like(vector)
        residual[..., 0] = gamma_a * r_aa - 2.0 * coupling * r_ab_imag + gamma_a / 2.0
        residual[..., 1] = -gamma_b * r_bb + 2.0 * coupling * r_ab_imag + gamma_b / 2.0
        residual[..., 2] = common * r_ab_real + detuning * r_ab_imag
        residual[..., 3] = (
            common * r_ab_imag - detuning * r_ab_real + coupling * (r_aa - r_bb)
        )
        return residual

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        """Evaluate the analytic CAM Jacobian without rebuilding a matrix."""
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        chi = self.parameter(params, "chi", xp)
        gamma_a = self.parameter(params, "gamma_a", xp)
        gamma_b = self.parameter(params, "gamma_b", xp)
        coupling = self.parameter(params, "g", xp)
        r_aa = vector[..., 0]
        r_ab_real = vector[..., 2]
        r_ab_imag = vector[..., 3]
        common = (gamma_a - gamma_b) / 2.0
        detuning = omega_a - omega_b + 2.0 * chi * (r_aa - 1.0)
        jacobian = xp.zeros(vector.shape[:-1] + (4, 4), dtype=vector.dtype)
        jacobian[..., 0, 0] = gamma_a
        jacobian[..., 0, 3] = -2.0 * coupling
        jacobian[..., 1, 1] = -gamma_b
        jacobian[..., 1, 3] = 2.0 * coupling
        jacobian[..., 2, 0] = 2.0 * chi * r_ab_imag
        jacobian[..., 2, 2] = common
        jacobian[..., 2, 3] = detuning
        jacobian[..., 3, 0] = coupling - 2.0 * chi * r_ab_real
        jacobian[..., 3, 1] = -coupling
        jacobian[..., 3, 2] = -detuning
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

    def cam_bifurcation_scales(self, params: dict[str, Any]) -> dict[str, Any]:
        scale = max(1.0, 1.0 / max(abs(float(params["chi"])), 1.0e-12))
        return {"state": [scale] * 4, "source": "model:chi"}

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
        parameters = sp.symbols("omega_a omega_b chi gamma_a gamma_b g", real=True)
        omega_a, omega_b, chi, gamma_a, gamma_b, coupling = parameters
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + coupling * (a.dag * b + a * b.dag)
                + chi * a.dag**2 * a**2
            ),
            channels=(
                LindbladChannel(a.dag, gamma_a),
                LindbladChannel(b, gamma_b),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=parameters,
                parameter_domains={
                    coupling: "nonnegative",
                    chi: "nonnegative",
                    gamma_a: "nonnegative",
                    gamma_b: "nonnegative",
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

        state, symbols = symbolic_hermitian_matrix(2)
        omega_a, omega_b, chi, gamma_a, gamma_b, coupling = sp.symbols(
            "omega_a omega_b chi gamma_a gamma_b g", real=True
        )
        r_aa = state[0, 0]
        hamiltonian = sp.Matrix(
            [
                [
                    omega_a + 2 * chi * (r_aa - 1) + sp.I * gamma_a / 2,
                    coupling,
                ],
                [coupling, omega_b - sp.I * gamma_b / 2],
            ]
        )
        diffusion = sp.diag(gamma_a / 2, gamma_b / 2)
        return CAMSymbolicSpec(
            hamiltonian,
            diffusion,
            state,
            symbols,
            (omega_a, omega_b, chi, gamma_a, gamma_b, coupling),
        )
