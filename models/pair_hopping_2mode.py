"""Two-mode pair-hopping CAM model plugin."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, ClassVar

from pydantic import Field
from qphase.backend.xputil import get_xp

from .base import ModelConfig


class PairHopping2ModeConfig(ModelConfig):
    omega_a: Any = Field(json_schema_extra={"scanable": True})
    omega_b: Any = Field(json_schema_extra={"scanable": True})
    g: Any = Field(json_schema_extra={"scanable": True})
    k: Any = Field(json_schema_extra={"scanable": True})
    gamma_a: Any = Field(json_schema_extra={"scanable": True})
    gamma_b: Any = Field(json_schema_extra={"scanable": True})


class PairHopping2ModeModel:
    """CAM-only pair-hopping model derived by fpgen."""

    name: ClassVar[str] = "pair_hopping_2mode"
    description: ClassVar[str] = "Two-mode pair-hopping oscillator"
    config_schema: ClassVar[type[PairHopping2ModeConfig]] = PairHopping2ModeConfig
    mode_count: ClassVar[int] = 2
    steady_state_capacity: ClassVar[int] = 5

    def __init__(
        self, config: PairHopping2ModeConfig | None = None, **kwargs: Any
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword parameters, not both")
        source: Any = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)
        self._params = self.config.model_dump()

    @property
    def n_modes(self) -> int:
        return self.mode_count

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    def cam_solution_sort_key(self, state: Any, params: dict[str, Any]) -> float:
        del params
        xp = get_xp(state)
        value = xp.real(state[..., 0, 0])
        return float(value.item() if hasattr(value, "item") else value)

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        omega_a = params["omega_a"]
        omega_b = params["omega_b"]
        coupling = params["g"]
        pair_coupling = params["k"]
        gamma_a = params["gamma_a"]
        gamma_b = params["gamma_b"]
        r_ab = state[..., 0, 1]
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = omega_a + 0.5j * gamma_a
        matrix[..., 0, 1] = coupling + 2.0 * pair_coupling * xp.conj(r_ab)
        matrix[..., 1, 0] = coupling + 2.0 * pair_coupling * r_ab
        matrix[..., 1, 1] = omega_b - 0.5j * gamma_b
        return matrix

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        matrix = xp.zeros(state.shape, dtype=state.dtype)
        matrix[..., 0, 0] = params["gamma_a"] / 2.0
        matrix[..., 1, 1] = params["gamma_b"] / 2.0
        return matrix

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        r_aa, r_bb, r_ab_real, r_ab_imag = (vector[..., index] for index in range(4))
        omega_a = params["omega_a"]
        omega_b = params["omega_b"]
        coupling = params["g"]
        pair_coupling = params["k"]
        gamma_a = params["gamma_a"]
        gamma_b = params["gamma_b"]
        population_difference = r_aa - r_bb
        effective_coupling = coupling + 2.0 * pair_coupling * r_ab_real
        common = (gamma_a - gamma_b) / 2.0
        residual = xp.empty_like(vector)
        residual[..., 0] = (
            gamma_a * r_aa
            + gamma_a / 2.0
            - 2.0 * coupling * r_ab_imag
            - 8.0 * pair_coupling * r_ab_real * r_ab_imag
        )
        residual[..., 1] = (
            -gamma_b * r_bb
            + gamma_b / 2.0
            + 2.0 * coupling * r_ab_imag
            + 8.0 * pair_coupling * r_ab_real * r_ab_imag
        )
        residual[..., 2] = (
            common * r_ab_real
            + (omega_a - omega_b + 2.0 * pair_coupling * population_difference)
            * r_ab_imag
        )
        residual[..., 3] = (
            common * r_ab_imag
            + effective_coupling * population_difference
            + (omega_b - omega_a) * r_ab_real
        )
        return residual

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        r_aa, r_bb, r_ab_real, r_ab_imag = (vector[..., index] for index in range(4))
        omega_a = params["omega_a"]
        omega_b = params["omega_b"]
        coupling = params["g"]
        pair_coupling = params["k"]
        gamma_a = params["gamma_a"]
        gamma_b = params["gamma_b"]
        population_difference = r_aa - r_bb
        common = (gamma_a - gamma_b) / 2.0
        jacobian = xp.zeros(vector.shape[:-1] + (4, 4), dtype=vector.dtype)
        jacobian[..., 0, 0] = gamma_a
        jacobian[..., 0, 2] = -8.0 * pair_coupling * r_ab_imag
        jacobian[..., 0, 3] = -8.0 * pair_coupling * r_ab_real - 2.0 * coupling
        jacobian[..., 1, 1] = -gamma_b
        jacobian[..., 1, 2] = 8.0 * pair_coupling * r_ab_imag
        jacobian[..., 1, 3] = 8.0 * pair_coupling * r_ab_real + 2.0 * coupling
        jacobian[..., 2, 0] = 2.0 * pair_coupling * r_ab_imag
        jacobian[..., 2, 1] = -2.0 * pair_coupling * r_ab_imag
        jacobian[..., 2, 2] = common
        jacobian[..., 2, 3] = (
            omega_a - omega_b + 2.0 * pair_coupling * population_difference
        )
        jacobian[..., 3, 0] = 2.0 * pair_coupling * r_ab_real + coupling
        jacobian[..., 3, 1] = -2.0 * pair_coupling * r_ab_real - coupling
        jacobian[..., 3, 2] = (
            omega_b - omega_a + 2.0 * pair_coupling * population_difference
        )
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
        scale = max(1.0, 1.0 / max(abs(float(params["k"])), 1.0e-12))
        return {"state": [scale] * 4, "source": "model:k"}

    @classmethod
    @lru_cache(maxsize=1)
    def cam_fpgen_dynamics(cls) -> Any:
        import sympy as sp
        from fpgen import (
            LindbladChannel,
            MasterEquation,
            boson_modes,
            derive_kramers_moyal,
        )

        a, b = boson_modes("a", "b")
        omega_a, omega_b, coupling, pair_coupling, gamma_a, gamma_b = sp.symbols(
            "omega_a omega_b g k gamma_a gamma_b", real=True
        )
        parameters = (
            omega_a,
            omega_b,
            coupling,
            pair_coupling,
            gamma_a,
            gamma_b,
        )
        master = MasterEquation(
            modes=(a, b),
            hamiltonian=(
                omega_a * a.dag * a
                + omega_b * b.dag * b
                + coupling * (a.dag * b + a * b.dag)
                + pair_coupling * (a.dag**2 * b**2 + b.dag**2 * a**2)
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
                    pair_coupling: "nonnegative",
                    gamma_a: "nonnegative",
                    gamma_b: "nonnegative",
                },
                layout="normal",
                closure="factorized_bilinear",
            )
        )
