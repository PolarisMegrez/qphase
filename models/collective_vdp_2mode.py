"""Collectively damped two-mode van der Pol oscillator."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from numbers import Real
from typing import Any, ClassVar

from pydantic import Field, model_validator

from .base import FPGenBackedSDEModel, ModelConfig
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import CollectiveVDP2ModeCayleyCuPyKernel


class CollectiveVDP2ModeConfig(ModelConfig):
    """Configuration generated from the collective-loss VDP master equation."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    Gamma: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})
    pump_a: Any = Field(json_schema_extra={"scanable": True})
    kappa_bright: Any = Field(json_schema_extra={"scanable": True})
    kappa_dark: Any = Field(json_schema_extra={"scanable": True})

    @model_validator(mode="after")
    def validate_scalar_physical_domain(self) -> CollectiveVDP2ModeConfig:
        nonnegative = (
            "Gamma",
            "g",
            "pump_a",
            "kappa_bright",
            "kappa_dark",
        )
        for name in nonnegative:
            value = getattr(self, name)
            if isinstance(value, Real) and value < 0.0:
                raise ValueError(f"{name} must be nonnegative")

        values = (
            self.Gamma,
            self.pump_a,
            self.kappa_bright,
            self.kappa_dark,
        )
        if all(isinstance(value, Real) for value in values):
            nonlinear_loss, pump, bright_loss, dark_loss = map(float, values)
            mean_loss = (bright_loss + dark_loss) / 4.0
            determinant_at_vacuum = bright_loss * dark_loss / 4.0 + mean_loss * (
                pump / 2.0 - nonlinear_loss
            )
            if mean_loss < 0.0 or determinant_at_vacuum < -1.0e-12:
                raise ValueError(
                    "parameters make the truncated-Wigner diffusion non-PSD at R_aa=0"
                )
        return self


class CollectiveVDP2ModeModel(FPGenBackedSDEModel):
    """VDP gain saturation coupled to collective bright and dark reservoirs."""

    name: ClassVar[str] = "collective_vdp_2mode"
    description: ClassVar[str] = (
        "Two-mode van der Pol oscillator with collective bright/dark loss"
    )
    config_schema: ClassVar[type[CollectiveVDP2ModeConfig]] = CollectiveVDP2ModeConfig
    mode_count: ClassVar[int] = 2
    steady_state_capacity: ClassVar[int] = 8

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (CollectiveVDP2ModeCayleyCuPyKernel(),)

    def cam_bifurcation_scales(self, params: dict[str, Any]) -> dict[str, Any]:
        scale = max(1.0, 1.0 / max(abs(float(params["Gamma"])), 1.0e-12))
        return {"state": [scale] * 4, "source": "model:Gamma"}

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        """Derive the authoritative Wigner CAM dynamics with fpgen."""
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b = boson_modes("a", "b")
        parameters = sp.symbols(
            "omega_a omega_b Gamma g pump_a kappa_bright kappa_dark",
            real=True,
        )
        (
            omega_a,
            omega_b,
            nonlinear_loss,
            coupling,
            pump_a,
            kappa_bright,
            kappa_dark,
        ) = parameters
        bright = (a + b) * (1 / sp.sqrt(2))
        dark = (a - b) * (1 / sp.sqrt(2))
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + coupling * (a.dag * b + a * b.dag)
            ),
            channels=(
                LindbladChannel(a.dag, pump_a),
                LindbladChannel(bright, kappa_bright),
                LindbladChannel(dark, kappa_dark),
                LindbladChannel(a**2, nonlinear_loss),
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
                    nonlinear_loss: "nonnegative",
                    coupling: "nonnegative",
                    pump_a: "nonnegative",
                    kappa_bright: "nonnegative",
                    kappa_dark: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )
