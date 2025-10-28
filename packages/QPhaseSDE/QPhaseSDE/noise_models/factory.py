from __future__ import annotations

"""Noise model factory: map config to concrete noise model instance via registry."""

from typing import Any

from ..core.protocols import BackendBase as Backend
from ..core.protocols import NoiseSpec
from .protocols import NoiseModel
from ..core.registry import registry


def make_noise_model(spec: NoiseSpec, backend: Backend) -> NoiseModel:
    # Currently only Gaussian noise is supported; independent/correlated via spec
    return registry.create("noise_model:gaussian", spec=spec, backend=backend)
