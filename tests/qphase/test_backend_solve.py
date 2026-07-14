"""Backend contract tests for batched linear solves."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from qphase.backend.numba_backend import NumbaBackend
from qphase.backend.numpy_backend import NumpyBackend


def _system() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    a = np.array(
        [
            [[2.0 + 0.5j, 0.25j], [-0.1j, 1.5 - 0.2j]],
            [[1.2 - 0.1j, 0.3], [0.2, 1.8 + 0.4j]],
        ]
    )
    expected = np.array([[0.5 + 0.2j, -0.1j], [0.3j, 0.7 - 0.2j]])
    b = np.matmul(a, expected[..., None])
    return a, b, expected


@pytest.mark.parametrize("backend", [NumpyBackend(), NumbaBackend()])
def test_cpu_backend_batched_solve(backend: Any):
    a, b, expected = _system()
    actual = backend.solve(backend.asarray(a), backend.asarray(b))[..., 0]
    np.testing.assert_allclose(actual, expected)


def test_torch_backend_batched_solve():
    torch = pytest.importorskip("torch")
    from qphase.backend.torch_backend import TorchBackend, TorchConfig

    a, b, expected = _system()
    backend = TorchBackend(TorchConfig(device="cpu", float_dtype="float64"))
    actual = backend.solve(backend.asarray(a), backend.asarray(b))[..., 0]
    np.testing.assert_allclose(actual.detach().cpu().numpy(), expected)
    assert actual.dtype == torch.complex128


def test_cupy_backend_batched_solve():
    cp = pytest.importorskip("cupy")
    try:
        cp.cuda.runtime.getDevice()
    except Exception:
        pytest.skip("CUDA device unavailable")

    from qphase.backend.cupy_backend import CuPyBackend

    a, b, expected = _system()
    backend = CuPyBackend()
    actual = backend.solve(backend.asarray(a), backend.asarray(b))[..., 0]
    cp.testing.assert_allclose(actual, cp.asarray(expected))
