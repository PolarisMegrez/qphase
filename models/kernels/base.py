"""Public base classes and registry for model-owned kernels."""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

__all__ = ["ModelKernelConfig", "ModelKernelPlugin", "ModelKernelRegistry"]


class ModelKernelConfig(BaseModel):
    """Strict base schema for an accelerated model implementation."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")


class ModelKernelPlugin(ABC):
    """Base class for kernels specialized by integration scheme and backend."""

    scheme: ClassVar[str]
    backend_name: ClassVar[str]
    operations: ClassVar[frozenset[str]]
    config_schema: ClassVar[type[ModelKernelConfig]] = ModelKernelConfig

    def __init__(
        self, config: ModelKernelConfig | None = None, **kwargs: Any
    ) -> None:
        if config is not None and kwargs:
            raise TypeError("provide either config or keyword parameters, not both")
        source: Any = kwargs if config is None else config.model_dump()
        self.config = self.config_schema.model_validate(source)
        if not self.scheme or not self.backend_name or not self.operations:
            raise TypeError(
                "kernel plugins must declare scheme, backend, and operations"
            )


class ModelKernelRegistry:
    """Resolve model-local kernels by scheme, backend, and operation."""

    def __init__(self) -> None:
        self._providers: dict[tuple[str, str], ModelKernelPlugin] = {}

    def register(self, provider: ModelKernelPlugin) -> None:
        if not isinstance(provider, ModelKernelPlugin):
            raise TypeError("model kernels must inherit ModelKernelPlugin")
        key = (provider.scheme.lower(), provider.backend_name.lower())
        if key in self._providers:
            raise ValueError(f"duplicate model kernel provider for {key}")
        self._providers[key] = provider

    def resolve(self, scheme: str, backend: Any, operation: str) -> Any:
        backend_name = str(backend.backend_name()).lower()
        provider = self._providers.get((scheme.lower(), backend_name))
        if provider is None or operation not in provider.operations:
            raise LookupError(
                f"no model kernel for scheme={scheme!r}, backend={backend_name!r}, "
                f"operation={operation!r}"
            )
        method = getattr(provider, operation, None)
        if not callable(method):
            raise TypeError(
                f"kernel {type(provider).__name__} declares operation "
                f"{operation!r} without implementing it"
            )
        return method

    def supports(self, scheme: str, backend: Any, operation: str) -> bool:
        try:
            self.resolve(scheme, backend, operation)
        except (AttributeError, LookupError, TypeError):
            return False
        return True
