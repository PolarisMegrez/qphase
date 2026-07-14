"""Protocols and registry for model-owned accelerated kernels."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ChunkKernelPlugin",
    "ModelKernelPlugin",
    "ModelKernelRegistry",
    "StepKernelPlugin",
    "TermsKernelPlugin",
]


@runtime_checkable
class ModelKernelPlugin(Protocol):
    """Common metadata for a model-local numerical kernel."""

    scheme: str
    backend_name: str


@runtime_checkable
class TermsKernelPlugin(ModelKernelPlugin, Protocol):
    """Kernel that evaluates drift and diffusion terms."""

    def terms(
        self, y: Any, params: dict[str, Any], backend: Any
    ) -> tuple[Any, Any]: ...


@runtime_checkable
class StepKernelPlugin(ModelKernelPlugin, Protocol):
    """Kernel that evaluates one complete integrator step."""

    def step(
        self,
        y: Any,
        t: float,
        dt: float,
        params: dict[str, Any],
        noise: Any,
        backend: Any,
    ) -> Any: ...


@runtime_checkable
class ChunkKernelPlugin(ModelKernelPlugin, Protocol):
    """Kernel that advances multiple fixed steps in one launch."""

    def step_chunk(self, *args: Any, **kwargs: Any) -> Any: ...


class ModelKernelRegistry:
    """Resolve model-local kernels by scheme, backend, and operation."""

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str], ModelKernelPlugin] = {}

    def register(self, provider: ModelKernelPlugin) -> None:
        key = (provider.scheme.lower(), provider.backend_name.lower())
        if key in self._providers:
            raise ValueError(f"duplicate model kernel provider for {key}")
        self._providers[key] = provider

    def resolve(self, scheme: str, backend: Any, operation: str) -> Any:
        backend_name = str(backend.backend_name()).lower()
        provider = self._providers.get((scheme.lower(), backend_name))
        method = getattr(provider, operation, None) if provider is not None else None
        if not callable(method):
            raise LookupError(
                f"no model kernel for scheme={scheme!r}, backend={backend_name!r}, "
                f"operation={operation!r}"
            )
        return method

    def supports(self, scheme: str, backend: Any, operation: str) -> bool:
        try:
            self.resolve(scheme, backend, operation)
        except (AttributeError, LookupError):
            return False
        return True
