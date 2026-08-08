"""Parametrically driven dimer with engineered two-photon loss."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field, model_validator
from qphase.backend.xputil import get_xp

from .base import FPGenBackedSDEModel, ModelConfig


class ParametricLoss2ModeConfig(ModelConfig):
    """Configuration generated from the driven dissipative master equation."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    epsilon: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})
    kappa_a: Any = Field(json_schema_extra={"scanable": True})
    kappa_b: Any = Field(json_schema_extra={"scanable": True})
    kappa_2: Any = Field(json_schema_extra={"scanable": True})

    @model_validator(mode="after")
    def validate_scalar_physical_domain(self) -> ParametricLoss2ModeConfig:
        nonnegative = ("epsilon", "g", "kappa_a", "kappa_b", "kappa_2")
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, (int, float)) and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")
        if isinstance(self.kappa_a, (int, float)) and isinstance(
            self.kappa_2, (int, float)
        ):
            if self.kappa_a < 2.0 * self.kappa_2:
                raise ValueError(
                    "kappa_a must be at least 2*kappa_2 so the truncated-Wigner "
                    "diffusion remains nonnegative"
                )
        return self


class ParametricLoss2ModeModel(FPGenBackedSDEModel):
    """Two-mode parametric oscillator saturated by two-photon dissipation."""

    name: ClassVar[str] = "parametric_loss_2mode"
    description: ClassVar[str] = (
        "Parametrically driven two-mode oscillator with two-photon loss"
    )
    config_schema: ClassVar[type[ParametricLoss2ModeConfig]] = ParametricLoss2ModeConfig
    mode_count: ClassVar[int] = 2
    steady_state_capacity: ClassVar[int] = 16

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        """Derive augmented Wigner CAM dynamics from the master equation."""
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b = boson_modes("a", "b")
        parameters = sp.symbols(
            "omega_a omega_b epsilon g kappa_a kappa_b kappa_2", real=True
        )
        omega_a, omega_b, epsilon, coupling, kappa_a, kappa_b, kappa_2 = parameters
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + coupling * (a.dag * b + a * b.dag)
                + sp.Rational(1, 2) * epsilon * (a.dag**2 + a**2)
            ),
            channels=(
                LindbladChannel(a, kappa_a),
                LindbladChannel(b, kappa_b),
                LindbladChannel(a**2, kappa_2),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=parameters,
                parameter_domains={
                    omega_a: "real",
                    omega_b: "real",
                    epsilon: "nonnegative",
                    coupling: "nonnegative",
                    kappa_a: "nonnegative",
                    kappa_b: "nonnegative",
                    kappa_2: "nonnegative",
                },
                layout="auto",
                closure="factorized_bilinear",
            )
        )

    @staticmethod
    def _augmented_vector(state: Any, xp: Any) -> Any:
        """Map fpgen's augmented state matrix back to its declared coordinates."""
        if state.shape[-2:] != (4, 4):
            raise ValueError("augmented CAM state must have shape (..., 4, 4)")
        return xp.stack(
            (
                xp.real(state[..., 0, 0]),
                xp.real(state[..., 1, 1]),
                xp.real(state[..., 0, 1]),
                xp.imag(state[..., 0, 1]),
                xp.real(state[..., 0, 2]),
                xp.real(state[..., 0, 3]),
                xp.real(state[..., 1, 3]),
                xp.imag(state[..., 0, 2]),
                xp.imag(state[..., 0, 3]),
                xp.imag(state[..., 1, 3]),
            ),
            axis=-1,
        )

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        omega_a = self.parameter(params, "omega_a", xp)
        omega_b = self.parameter(params, "omega_b", xp)
        epsilon = self.parameter(params, "epsilon", xp)
        coupling = self.parameter(params, "g", xp)
        kappa_a = self.parameter(params, "kappa_a", xp)
        kappa_b = self.parameter(params, "kappa_b", xp)
        kappa_2 = self.parameter(params, "kappa_2", xp)
        alpha = y[..., 0]
        beta = y[..., 1]
        result = xp.empty_like(y)
        result[..., 0] = (
            (-1j * omega_a - kappa_a / 2 + kappa_2 * (1 - xp.abs(alpha) ** 2)) * alpha
            - 1j * coupling * beta
            - 1j * epsilon * xp.conj(alpha)
        )
        result[..., 1] = (-1j * omega_b - kappa_b / 2) * beta - 1j * coupling * alpha
        return result

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del y, t, params
        raise NotImplementedError(
            "parametric_loss_2mode has conjugate drift and requires an augmented "
            "integrator; use Euler-Maruyama until its fused Cayley kernel is enabled"
        )

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        kappa_a = self.parameter(params, "kappa_a", xp)
        kappa_b = self.parameter(params, "kappa_b", xp)
        kappa_2 = self.parameter(params, "kappa_2", xp)
        diagonal_a = 2 * kappa_2 * xp.abs(y[..., 0]) ** 2 - kappa_2 + kappa_a / 2
        return self.diagonal_complex_diffusion(y, (diagonal_a, kappa_b / 2))

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = self._augmented_vector(state, xp)
        value = self._evaluate_fpgen_matrix("hamiltonian", vector, params, xp)
        return xp.asarray(value, dtype=state.dtype)

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = self._augmented_vector(state, xp)
        value = self._evaluate_fpgen_matrix("diffusion", vector, params, xp)
        return xp.asarray(value, dtype=state.dtype)

    def cam_jacobian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        return self.cam_jacobian_vector(self._augmented_vector(state, xp), params)
