"""Public contracts and shared utilities for local SDE model plugins."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from functools import cache, lru_cache
from typing import Any, ClassVar

import numpy as np
from pydantic import BaseModel, ConfigDict
from qphase.backend.xputil import get_xp
from qphase_sde.model import FunctionalSDEModel

from .kernels.base import ModelKernelPlugin, ModelKernelRegistry

__all__ = ["FPGenBackedSDEModel", "ModelConfig", "SDEModelPlugin"]


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

    def diagonal_complex_diffusion(self, y: Any, diagonal: Iterable[Any]) -> Any:
        """Factor a non-negative diagonal complex covariance matrix."""
        xp = get_xp(y)
        diffusion = xp.zeros((y.shape[0], self.n_modes, self.n_modes), dtype=y.dtype)
        for mode, value in enumerate(diagonal):
            value = xp.asarray(value, dtype=y.real.dtype)
            diffusion[:, mode, mode] = xp.sqrt(xp.clip(value, 0.0, None))
        return diffusion

    def cam_solution_sort_key(self, state: Any, params: dict[str, Any]) -> float:
        """Return the default per-point ordering key for CAM steady states."""
        del params
        value = get_xp(state).real(state[..., 0, 0])
        return float(value.item() if hasattr(value, "item") else value)

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
        return self._kernel_registry.supports(scheme, backend, operation="step_chunk")

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


def _canonical_hermitian_vector(state: Any, xp: Any) -> Any:
    n_modes = int(state.shape[-1])
    pairs = tuple(
        (i, j) for i in range(n_modes) for j in range(i + 1, n_modes)
    )
    values = [xp.real(state[..., index, index]) for index in range(n_modes)]
    values.extend(xp.real(state[..., i, j]) for i, j in pairs)
    values.extend(xp.imag(state[..., i, j]) for i, j in pairs)
    return xp.stack(values, axis=-1)


class _CompiledFPGenMatrix:
    """Backend-aware scalar compilation of one fpgen symbolic matrix."""

    def __init__(self, expression: Any, arguments: tuple[Any, ...], modules: Any):
        import sympy as sp

        matrix = sp.Matrix(expression)
        self.shape = matrix.shape
        self._functions = tuple(
            sp.lambdify(arguments, value, modules=modules) for value in matrix
        )

    def __call__(
        self,
        state_vector: Any,
        params: dict[str, Any],
        parameter_names: tuple[str, ...],
        xp: Any,
    ) -> Any:
        arguments = [
            state_vector[..., index] for index in range(state_vector.shape[-1])
        ]
        arguments.extend(xp.asarray(params[name]) for name in parameter_names)
        entries = [xp.asarray(function(*arguments)) for function in self._functions]
        batch_shape = tuple(state_vector.shape[:-1])
        if entries:
            batch_shape = np.broadcast_shapes(
                batch_shape, *(tuple(entry.shape) for entry in entries)
            )
        broadcast = [xp.broadcast_to(entry, batch_shape) for entry in entries]
        stacked = xp.stack(broadcast, axis=-1)
        return stacked.reshape(batch_shape + self.shape)


class FPGenBackedSDEModel(SDEModelPlugin):
    """Model whose numerical equations are compiled from fpgen output."""

    @classmethod
    def cam_fpgen_dynamics(cls) -> Any:
        """Return the authoritative fpgen second-moment dynamics."""
        raise NotImplementedError

    @classmethod
    @cache
    def _fpgen_expression(cls, name: str) -> Any:
        dynamics = cls.cam_fpgen_dynamics()
        if name == "jacobian":
            return dynamics.jacobian()
        return getattr(dynamics, name)

    @classmethod
    @cache
    def _fpgen_parameter_names(cls) -> tuple[str, ...]:
        dynamics = cls.cam_fpgen_dynamics()
        return tuple(item.symbol.name for item in dynamics.parameter_spec)

    @classmethod
    @cache
    def _compiled_fpgen_matrix(
        cls, name: str, backend_name: str
    ) -> _CompiledFPGenMatrix:
        dynamics = cls.cam_fpgen_dynamics()
        if backend_name == "cupy":
            import cupy as modules
        else:
            modules = "numpy"
        arguments = tuple(dynamics.coordinates) + tuple(
            item.symbol for item in dynamics.parameter_spec
        )
        return _CompiledFPGenMatrix(
            cls._fpgen_expression(name), arguments, modules
        )

    @staticmethod
    def _backend_name(xp: Any) -> str:
        return "cupy" if xp.__name__.split(".", 1)[0] == "cupy" else "numpy"

    def _evaluate_fpgen_matrix(
        self, name: str, state_vector: Any, params: dict[str, Any], xp: Any
    ) -> Any:
        compiled = self._compiled_fpgen_matrix(name, self._backend_name(xp))
        return compiled(state_vector, params, self._fpgen_parameter_names(), xp)

    def drift(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        return xp.einsum("...ij,...j->...i", self.drift_matrix(y, 0.0, params), y)

    def drift_matrix(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        state = xp.einsum("...i,...j->...ij", y, xp.conj(y))
        return -1j * self.cam_hamiltonian(state, params)

    def diffusion(self, y: Any, t: float, params: dict[str, Any]) -> Any:
        del t
        xp = get_xp(y)
        state = xp.einsum("...i,...j->...ij", y, xp.conj(y))
        covariance = self.cam_diffusion(state, params)
        covariance = (covariance + xp.swapaxes(xp.conj(covariance), -1, -2)) / 2.0
        eigenvalues, eigenvectors = xp.linalg.eigh(covariance)
        eigenvalues = xp.clip(eigenvalues, 0.0, None)
        return eigenvectors * xp.sqrt(eigenvalues)[..., None, :]

    def cam_hamiltonian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = _canonical_hermitian_vector(state, xp)
        value = self._evaluate_fpgen_matrix("hamiltonian", vector, params, xp)
        return xp.asarray(value, dtype=state.dtype)

    def cam_diffusion(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        vector = _canonical_hermitian_vector(state, xp)
        value = self._evaluate_fpgen_matrix("diffusion", vector, params, xp)
        return xp.asarray(value, dtype=state.dtype)

    def cam_residual_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        value = self._evaluate_fpgen_matrix("rhs", vector, params, xp)
        return xp.asarray(value[..., 0], dtype=vector.dtype)

    def cam_jacobian_vector(self, vector: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(vector)
        vector = xp.asarray(vector)
        value = self._evaluate_fpgen_matrix("jacobian", vector, params, xp)
        return xp.asarray(value, dtype=vector.dtype)

    def cam_jacobian(self, state: Any, params: dict[str, Any]) -> Any:
        xp = get_xp(state)
        state = xp.asarray(state)
        return self.cam_jacobian_vector(
            _canonical_hermitian_vector(state, xp), params
        )

    @classmethod
    @lru_cache(maxsize=1)
    def cam_symbolic_matrices(cls) -> Any:
        from qphase_cam.model import CAMSymbolicSpec

        dynamics = cls.cam_fpgen_dynamics()
        spec = dynamics.to_model_spec(name=cls.name)
        return CAMSymbolicSpec(
            dynamics.hamiltonian,
            dynamics.diffusion,
            dynamics.covariance,
            tuple(dynamics.coordinates),
            tuple(item.symbol for item in dynamics.parameter_spec),
            version=f"fpgen:{spec.fingerprint}",
        )
