"""Kerr trimer with correlated bright/dark loss channels."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field

from .base import FPGenBackedSDEModel, ModelConfig
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import CollectiveLossKerr3ModeCayleyCuPyKernel


class CollectiveLossKerr3ModeConfig(ModelConfig):
    """Configuration generated from the collective-loss master equation."""

    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    omega_c: Any = Field(json_schema_extra={"scanable": True})
    chi: Any = Field(json_schema_extra={"scanable": True})
    g_ab: Any = Field(json_schema_extra={"scanable": True})
    g_ac: Any = Field(json_schema_extra={"scanable": True})
    g_bc: Any = Field(json_schema_extra={"scanable": True})
    pump_a: Any = Field(json_schema_extra={"scanable": True})
    kappa_bright: Any = Field(json_schema_extra={"scanable": True})
    kappa_dark: Any = Field(json_schema_extra={"scanable": True})


class CollectiveLossKerr3ModeModel(FPGenBackedSDEModel):
    """Pumped Kerr mode coupled to collectively damped bright/dark modes."""

    name: ClassVar[str] = "collective_loss_kerr_3mode"
    description: ClassVar[str] = "Three-mode Kerr oscillator with collective loss"
    config_schema: ClassVar[type[CollectiveLossKerr3ModeConfig]] = (
        CollectiveLossKerr3ModeConfig
    )
    mode_count: ClassVar[int] = 3
    steady_state_capacity: ClassVar[int] = 16

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (CollectiveLossKerr3ModeCayleyCuPyKernel(self),)

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        """Derive Wigner CAM and Langevin dynamics from one master equation."""
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b, c = boson_modes("a", "b", "c")
        parameters = sp.symbols(
            "omega_a omega_b omega_c chi g_ab g_ac g_bc "
            "pump_a kappa_bright kappa_dark",
            real=True,
        )
        (
            omega_a,
            omega_b,
            omega_c,
            chi,
            coupling_ab,
            coupling_ac,
            coupling_bc,
            pump_a,
            kappa_bright,
            kappa_dark,
        ) = parameters
        bright = (b + c) * (1 / sp.sqrt(2))
        dark = (b - c) * (1 / sp.sqrt(2))
        master = MasterEquation(
            modes=(a, b, c),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + omega_c * c.dag * c
                + chi * a.dag**2 * a**2
                + coupling_ab * (a.dag * b + a * b.dag)
                + coupling_ac * (a.dag * c + a * c.dag)
                + coupling_bc * (b.dag * c + b * c.dag)
            ),
            channels=(
                LindbladChannel(a.dag, pump_a),
                LindbladChannel(bright, kappa_bright),
                LindbladChannel(dark, kappa_dark),
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
                    omega_c: "real",
                    chi: "nonnegative",
                    coupling_ab: "nonnegative",
                    coupling_ac: "nonnegative",
                    coupling_bc: "nonnegative",
                    pump_a: "nonnegative",
                    kappa_bright: "nonnegative",
                    kappa_dark: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )
