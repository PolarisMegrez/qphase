"""Public contracts and shared utilities for local SDE model plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel

from .kernels.base import ModelKernelPlugin, ModelKernelRegistry

__all__ = ["ModelConfig", "SDEModelPlugin"]


class ModelConfig(BaseModel):
    """Strict base schema for model plugin configuration."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class SDEModelPlugin(ABC):
    """Base class for local diffusive SDE model plugins."""

    name: ClassVar[str]
    description: ClassVar[str]
    config_schema: ClassVar[type[ModelConfig]]
    mode_count: ClassVar[int]
    noise_basis_name: ClassVar[str] = "complex"

    def __init__(self, config: ModelConfig | None = None, **kwargs: Any) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword parameters, not both")
        source: Any = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)
        self._params = self.config.model_dump()
        self._kernel_registry = ModelKernelRegistry()
        for provider in self.kernel_plugins():
            self._kernel_registry.register(provider)

    def kernel_plugins(self) -> Iterable[ModelKernelPlugin]:
        """Return accelerated implementations owned by this model."""
        return ()

    @property
    def n_modes(self) -> int:
        return self.mode_count

    @property
    def noise_basis(self) -> str:
        return self.noise_basis_name

    @property
    def noise_dim(self) -> int:
        return 2 * self.mode_count if self.noise_basis == "complex" else self.mode_count

    @property
    def params(self) -> dict[str, Any]:
        return self._params

    @staticmethod
    def parameter(params: dict[str, Any], name: str, xp: Any) -> Any:
        """Move an array-valued scan parameter to the active backend."""
        value = params[name]
        if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
            return xp.asarray(value)
        return value

    def diagonal_complex_diffusion(
        self, y: Any, diagonal: Iterable[Any]
    ) -> Any:
        """Factor a non-negative diagonal complex covariance matrix."""
        xp = get_xp(y)
        diffusion = xp.zeros(
            (y.shape[0], self.n_modes, self.n_modes), dtype=y.dtype
        )
        for mode, value in enumerate(diagonal):
            value = xp.asarray(value, dtype=y.real.dtype)
            diffusion[:, mode, mode] = xp.sqrt(xp.clip(value, 0.0, None))
        return diffusion

    @abstractmethod
    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        """Return the Ito drift vector."""

    @abstractmethod
    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        """Return ``A(y,t)`` satisfying ``drift=A@y``."""

    @abstractmethod
    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        """Return a factor of the complex diffusion covariance."""

    def has_kernelized_terms(self, backend: Any) -> bool:
        return self._kernel_registry.supports(
            "euler_maruyama", backend, operation="terms"
        )

    def kernelized_terms(
        self, y: Any, t: float, params: dict[str, Any], backend: Any
    ) -> tuple[Any, Any]:
        del t
        terms = self._kernel_registry.resolve(
            "euler_maruyama", backend, operation="terms"
        )
        return terms(y, params, backend)

    def supports_fused_step(self, scheme: str, backend: Any) -> bool:
        return self._kernel_registry.supports(scheme, backend, operation="step")

    def fused_step(
        self,
        scheme: str,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: Any,
    ) -> Any:
        step = self._kernel_registry.resolve(scheme, backend, operation="step")
        return step(y, t, dt, params, noise, backend)

    def supports_fused_chunk(self, scheme: str, backend: Any) -> bool:
        return self._kernel_registry.supports(
            scheme, backend, operation="step_chunk"
        )

    def fused_step_chunk(
        self,
        scheme: str,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: Any,
        *,
        n_steps: int,
        save_offsets: tuple[int, ...],
        record_modes: tuple[int, ...],
    ) -> tuple[Any, Any]:
        step_chunk = self._kernel_registry.resolve(
            scheme, backend, operation="step_chunk"
        )
        return step_chunk(
            y,
            t,
            dt,
            params,
            noise,
            backend,
            n_steps=n_steps,
            save_offsets=save_offsets,
            record_modes=record_modes,
        )

    def to_diffusive_sde_model(self) -> FunctionalSDEModel:
        return FunctionalSDEModel(
            name=self.name,
            n_modes=self.n_modes,
            noise_basis=self.noise_basis,
            noise_dim=self.noise_dim,
            params=self.params,
            drift=self.drift,
            diffusion=self.diffusion,
            drift_matrix=self.drift_matrix,
        )
