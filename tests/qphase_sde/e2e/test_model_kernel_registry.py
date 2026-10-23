"""Tests for model-local accelerated kernel registration."""

from __future__ import annotations

import pytest

from models.kernels.base import ModelKernelPlugin, ModelKernelRegistry
from models.kernels.cupy_utils import get_lru_buffer


class Backend:
    def backend_name(self):
        return "dummy"


class Provider(ModelKernelPlugin):
    scheme = "scheme"
    backend_name = "dummy"
    operations = frozenset({"terms", "step"})

    def terms(self, y, params, backend):
        return y, params

    def step(self, y, t, dt, params, noise, backend):
        return noise


def test_registry_resolves_declared_operations():
    registry = ModelKernelRegistry()
    provider = Provider()
    registry.register(provider)

    assert registry.supports("scheme", Backend(), "terms")
    assert registry.supports("scheme", Backend(), "step")
    assert not registry.supports("scheme", Backend(), "step_chunk")
    assert registry.resolve("scheme", Backend(), "terms").__self__ is provider


def test_registry_rejects_duplicate_provider():
    registry = ModelKernelRegistry()
    registry.register(Provider())
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(Provider())


def test_registry_rejects_non_plugin_provider():
    registry = ModelKernelRegistry()
    with pytest.raises(TypeError, match="inherit"):
        registry.register(object())


def test_kernel_schema_rejects_unknown_options():
    with pytest.raises(ValueError, match="extra"):
        Provider(unknown=True)


def test_device_buffer_lru_bounds_retained_shapes():
    cache = {}
    keys = []

    first = get_lru_buffer(cache, keys, "a", lambda: object())
    get_lru_buffer(cache, keys, "b", lambda: object())
    assert get_lru_buffer(cache, keys, "a", lambda: object()) is first
    get_lru_buffer(cache, keys, "c", lambda: object())

    assert keys == ["a", "c"]
    assert set(cache) == {"a", "c"}
