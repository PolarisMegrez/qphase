"""Exchange-symmetric Kerr dimer with collective bright/dark reservoirs."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field

from .base import FPGenBackedSDEModel, ModelConfig


class CollectiveKerr2ModeConfig(ModelConfig):
    """Configuration generated from the collective Kerr master equation."""

    omega_0: Any = Field(json_schema_extra={"scanable": True})
    delta: Any = Field(json_schema_extra={"scanable": True})
    chi: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})
    kappa_bright: Any = Field(json_schema_extra={"scanable": True})
    pump_bright: Any = Field(json_schema_extra={"scanable": True})
    kappa_dark: Any = Field(json_schema_extra={"scanable": True})


class CollectiveKerr2ModeModel(FPGenBackedSDEModel):
    """Two Kerr modes coupled to bright gain/loss and weak dark loss channels."""

    name: ClassVar[str] = "collective_kerr_2mode"
    description: ClassVar[str] = "Collectively damped two-mode Kerr oscillator"
    config_schema: ClassVar[type[CollectiveKerr2ModeConfig]] = (
        CollectiveKerr2ModeConfig
    )
    mode_count: ClassVar[int] = 2
    steady_state_capacity: ClassVar[int] = 8

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
            "omega_0 delta chi g kappa_bright pump_bright kappa_dark",
            real=True,
        )
        (
            omega_0,
            delta,
            chi,
            coupling,
            kappa_bright,
            pump_bright,
            kappa_dark,
        ) = parameters
        bright = (a + b) * (1 / sp.sqrt(2))
        dark = (a - b) * (1 / sp.sqrt(2))
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                (omega_0 + delta) * a.dag * a
                + (omega_0 - delta) * b.dag * b
                - coupling * (a.dag * b + a * b.dag)
                + chi * (a.dag**2 * a**2 + b.dag**2 * b**2)
            ),
            channels=(
                LindbladChannel(bright, kappa_bright),
                LindbladChannel(bright.dag, pump_bright),
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
                    omega_0: "real",
                    delta: "real",
                    chi: "nonnegative",
                    coupling: "nonnegative",
                    kappa_bright: "nonnegative",
                    pump_bright: "nonnegative",
                    kappa_dark: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )
