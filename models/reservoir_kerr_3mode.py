"""Kerr dimer with an explicit lossy bright-mode reservoir."""

from __future__ import annotations

from collections.abc import Iterable
from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field

from .base import FPGenBackedSDEModel, ModelConfig
from .kernels.base import ModelKernelPlugin
from .kernels.cayley_maruyama import ReservoirKerr3ModeCayleyCuPyKernel


class ReservoirKerr3ModeConfig(ModelConfig):
    """Configuration generated from the explicit-reservoir master equation."""

    omega_r: Any = Field(json_schema_extra={"scanable": True})
    omega_0: Any = Field(json_schema_extra={"scanable": True})
    delta: Any = Field(json_schema_extra={"scanable": True})
    chi: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})
    g_r: Any = Field(json_schema_extra={"scanable": True})
    kappa_r: Any = Field(json_schema_extra={"scanable": True})
    pump_r: Any = Field(json_schema_extra={"scanable": True})
    kappa_local: Any = Field(json_schema_extra={"scanable": True})


class ReservoirKerr3ModeModel(FPGenBackedSDEModel):
    """Symmetric Kerr dimer coupled through a damped auxiliary mode."""

    name: ClassVar[str] = "reservoir_kerr_3mode"
    description: ClassVar[str] = "Explicit-reservoir three-mode Kerr oscillator"
    config_schema: ClassVar[type[ReservoirKerr3ModeConfig]] = ReservoirKerr3ModeConfig
    mode_count: ClassVar[int] = 3
    steady_state_capacity: ClassVar[int] = 16

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        return (ReservoirKerr3ModeCayleyCuPyKernel(self),)

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

        reservoir, a, b = boson_modes("r", "a", "b")
        parameters = sp.symbols(
            "omega_r omega_0 delta chi g g_r kappa_r pump_r kappa_local",
            real=True,
        )
        (
            omega_r,
            omega_0,
            delta,
            chi,
            coupling,
            reservoir_coupling,
            kappa_r,
            pump_r,
            kappa_local,
        ) = parameters
        master = MasterEquation(
            modes=(reservoir, a, b),
            hamiltonian=(
                omega_r * reservoir.dag * reservoir
                + (omega_0 + delta) * a.dag * a
                + (omega_0 - delta) * b.dag * b
                - coupling * (a.dag * b + a * b.dag)
                + reservoir_coupling
                * (1 / sp.sqrt(2))
                * (reservoir.dag * (a + b) + (a.dag + b.dag) * reservoir)
                + chi * (a.dag**2 * a**2 + b.dag**2 * b**2)
            ),
            channels=(
                LindbladChannel(reservoir, kappa_r),
                LindbladChannel(reservoir.dag, pump_r),
                LindbladChannel(a, kappa_local),
                LindbladChannel(b, kappa_local),
            ),
        )
        return (
            derive_kramers_moyal(master, "wigner")
            .truncate(2)
            .to_langevin()
            .to_second_moment_dynamics(
                parameters=parameters,
                parameter_domains={
                    omega_r: "real",
                    omega_0: "real",
                    delta: "real",
                    chi: "nonnegative",
                    coupling: "nonnegative",
                    reservoir_coupling: "nonnegative",
                    kappa_r: "nonnegative",
                    pump_r: "nonnegative",
                    kappa_local: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )
