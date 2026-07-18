"""Cayley-Maruyama integrator for matrix-drift Ito SDEs."""

from typing import Any, ClassVar, Literal

import numpy as np
from pydantic import Field
from qphase.backend.base import BackendBase
from qphase.core.protocols import PluginConfigBase

from qphase_sde import ops
from qphase_sde.integrator.base import ChunkStepResult, Integrator
from qphase_sde.model import SDEModel

__all__ = ["CayleyMaruyama", "CayleyMaruyamaConfig"]


class CayleyMaruyamaConfig(PluginConfigBase):
    """Configuration for the fixed-step Cayley-Maruyama scheme."""

    fused: Literal["auto", "required", "off"] = Field(
        "auto", description="Use a model-provided fused implementation when available"
    )
    chunk_steps: int = Field(
        1,
        ge=1,
        description="Number of fixed steps per optional fused chunk",
    )
    max_modes: int = Field(
        16,
        ge=1,
        le=64,
        description="Maximum supported matrix-drift dimension",
    )


class CayleyMaruyama(Integrator):
    """Linearly implicit midpoint drift with left-point Ito diffusion."""

    name: ClassVar[str] = "cayley_maruyama"
    description: ClassVar[str] = (
        "Cayley-Maruyama integrator for Ito SDEs with matrix-valued drift"
    )
    config_schema: ClassVar[type[CayleyMaruyamaConfig]] = CayleyMaruyamaConfig

    def __init__(
        self, config: CayleyMaruyamaConfig | None = None, **kwargs: Any
    ) -> None:
        self.config = config or CayleyMaruyamaConfig(**kwargs)

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        model: SDEModel,
        noise: Any,
        backend: BackendBase,
    ) -> Any:
        """Return one Cayley-Maruyama increment without mutating ``y``."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")

        fused = self._fused_step(y, t, dt, model, noise, backend)
        if fused is not None:
            return fused

        drift_matrix = getattr(model, "drift_matrix", None)
        if not callable(drift_matrix):
            raise TypeError("Cayley-Maruyama requires model.drift_matrix(y, t, params)")

        n_modes = int(getattr(model, "n_modes", y.shape[-1]))
        if n_modes < 1 or n_modes > self.config.max_modes:
            raise ValueError(
                f"Cayley-Maruyama supports 1..{self.config.max_modes} modes; "
                f"received {n_modes}"
            )
        if int(y.shape[-1]) != n_modes:
            raise ValueError(
                f"state has {y.shape[-1]} modes but model declares {n_modes}"
            )

        matrix = drift_matrix(y, t, model.params)
        if tuple(matrix.shape[-2:]) != (n_modes, n_modes):
            raise ValueError(
                "drift_matrix must end with shape "
                f"({n_modes}, {n_modes}); received {matrix.shape}"
            )

        diffusion = model.diffusion(y, t, model.params)
        if getattr(model, "noise_basis", "real") == "complex":
            diffusion = ops.expand_complex_noise(diffusion, backend)
        stochastic = ops.contract_noise(diffusion, noise, backend)

        identity = backend.asarray(np.eye(n_modes), dtype=y.dtype)
        half_dt = 0.5 * dt
        lhs = identity - half_dt * matrix
        rhs_matrix = identity + half_dt * matrix
        rhs = backend.einsum("...ij,...j->...i", rhs_matrix, y) + stochastic
        y_next = backend.solve(lhs, backend.expand_dims(rhs, axis=-1))[..., 0]
        return y_next - y

    def _fused_step(
        self,
        y: Any,
        t: float,
        dt: float,
        model: SDEModel,
        noise: Any,
        backend: BackendBase,
    ) -> Any | None:
        if self.config.fused == "off":
            return None

        supports = getattr(model, "supports_fused_step", None)
        fused_step = getattr(model, "fused_step", None)
        if (
            callable(supports)
            and callable(fused_step)
            and bool(supports(self.name, backend))
        ):
            return fused_step(self.name, y, t, dt, model.params, noise, backend)
        if self.config.fused == "required":
            raise RuntimeError(
                f"model {getattr(model, 'name', type(model).__name__)!r} does not "
                f"provide a fused {self.name!r} step for "
                f"backend {backend.backend_name()!r}"
            )
        return None

    def supports_adaptive_step(self) -> bool:
        return False

    def step_adaptive(
        self,
        y: Any,
        t: float,
        dt: float,
        tol: float,
        model: Any,
        noise: Any,
        backend: BackendBase,
        rng: Any = None,
    ) -> tuple[Any, float, float, float]:
        """Adaptive stepping not supported by Cayley-Maruyama."""
        raise NotImplementedError("Cayley-Maruyama does not support adaptive stepping")

    def supports_chunk_step(self, model: Any, backend: BackendBase) -> bool:
        """Return whether the model/backend pair provides a fused chunk path."""
        if self.config.fused == "off" or self.config.chunk_steps <= 1:
            return False
        supports = getattr(model, "supports_fused_chunk", None)
        return callable(supports) and bool(supports(self.name, backend))

    def step_chunk(
        self,
        y: Any,
        t: float,
        dt: float,
        model: Any,
        noise: Any,
        backend: BackendBase,
        *,
        n_steps: int,
        save_offsets: tuple[int, ...],
        record_modes: tuple[int, ...],
    ) -> ChunkStepResult:
        """Advance a model-provided fused chunk."""
        if not self.supports_chunk_step(model, backend):
            raise RuntimeError("fused Cayley-Maruyama chunk path is unavailable")
        final_state, saved_states = model.fused_step_chunk(
            self.name,
            y,
            t,
            dt,
            model.params,
            noise,
            backend,
            n_steps=n_steps,
            save_offsets=save_offsets,
            record_modes=record_modes,
        )
        return ChunkStepResult(final_state=final_state, saved_states=saved_states)

    def reset(self) -> None:
        pass

    def supports_strided_state(self) -> bool:
        return False
