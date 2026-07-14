"""Tests for model-local accelerated kernel registration."""

from __future__ import annotations

import pytest

from models.kernels.base import ModelKernelRegistry


class Backend:
    def backend_name(self):
        return "dummy"


class Provider:
    scheme = "scheme"
    backend_name = "dummy"

    def terms(self, y, params, backend):
        return y, params

    def step(self, y, t, dt, params, noise, backend):
        return noise


def test_registry_resolves_supported_operations():
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


def test_registry_reports_missing_kernel():
    registry = ModelKernelRegistry()
    with pytest.raises(LookupError, match="no model kernel"):
        registry.resolve("missing", Backend(), "step")
