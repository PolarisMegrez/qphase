"""Tests for the VDP Level 3 kernelized terms path."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.cupy_backend import CuPyBackend
from qphase.backend.numpy_backend import NumpyBackend
from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

pytestmark = pytest.mark.gpu


# Runtime CuPy availability check.
def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


@pytest.fixture
def model():
    from models.vdp_2mode import VDP2ModeConfig, VDP2ModeModel

    return VDP2ModeModel(
        VDP2ModeConfig(
            omega_a=0.005,
            omega_b=0.0,
            gamma_a=2.0,
            gamma_b=1.0,
            Gamma=0.01,
            g=0.5,
        )
    )


def test_drift_matrix_matches_drift(model):
    """The matrix-drift capability reproduces the public drift function."""
    rng = np.random.default_rng(41)
    y = (rng.standard_normal((32, 2)) + 1j * rng.standard_normal((32, 2))).astype(
        np.complex128
    )

    matrix = model.drift_matrix(y, 0.0, model.params)
    actual = np.einsum("...ij,...j->...i", matrix, y)

    np.testing.assert_allclose(actual, model.drift(y, 0.0, model.params))


def test_cayley_generic_vdp_step(model):
    """The VDP model supports the backend-generic Cayley path."""
    backend = NumpyBackend()
    y = np.array([[2.0 + 0.5j, -0.3j], [1.0 - 0.2j, 0.4 + 0.1j]])
    d_w = np.zeros((2, 4))

    dy = CayleyMaruyama(fused="off").step(y, 0.0, 0.1, model, d_w, backend)

    assert dy.shape == y.shape
    assert np.all(np.isfinite(dy))


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_kernelized_terms_match_python(model):
    """CuPy kernelized drift/diffusion matches the Python implementation."""
    import cupy as cp

    backend = CuPyBackend()
    n = 100
    rng = np.random.default_rng(42)
    y_np = (rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))).astype(
        np.complex64
    )
    y = cp.asarray(y_np)

    a_py = cp.asarray(model.drift(y_np, 0.0, model.params))
    L_py = cp.asarray(model.diffusion(y_np, 0.0, model.params))

    a_k, L_k = model.kernelized_terms(y, 0.0, model.params, backend)

    cp.testing.assert_allclose(a_k, a_py, rtol=1e-5, atol=1e-6)
    cp.testing.assert_allclose(L_k, L_py, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_kernelized_vectorized_params(model):
    """Kernel handles per-trajectory arrays of scanned parameters."""
    import cupy as cp

    backend = CuPyBackend()
    n = 30
    rng = np.random.default_rng(43)
    y = cp.asarray(
        (rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))).astype(
            np.complex64
        )
    )

    params = dict(model.params)
    params["omega_a"] = cp.asarray(
        np.repeat([0.001, 0.002, 0.003], 10), dtype=cp.float32
    )

    a, L = model.kernelized_terms(y, 0.0, params, backend)
    assert a.shape == (n, 2)
    assert L.shape == (n, 2, 2)


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_cayley_fused_step_matches_generic(model):
    """The VDP Cayley RawKernel matches the generic batched solve."""
    import cupy as cp

    backend = CuPyBackend()
    rng = np.random.default_rng(44)
    y = cp.asarray(
        (rng.standard_normal((64, 2)) + 1j * rng.standard_normal((64, 2))).astype(
            np.complex64
        )
    )
    d_w = cp.asarray(rng.standard_normal((64, 4)).astype(np.float32) * np.sqrt(0.1))

    generic = CayleyMaruyama(fused="off").step(y, 0.0, 0.1, model, d_w, backend)
    fused = CayleyMaruyama(fused="required").step(y, 0.0, 0.1, model, d_w, backend)

    cp.testing.assert_allclose(fused, generic, rtol=2e-5, atol=2e-6)


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_cayley_chunk_matches_repeated_fused_steps(model):
    """A fused chunk matches repeated single-step kernels for fixed noise."""
    import cupy as cp

    backend = CuPyBackend()
    integrator = CayleyMaruyama(fused="required", chunk_steps=8)
    rng = np.random.default_rng(45)
    y = cp.asarray(
        (rng.standard_normal((32, 2)) + 1j * rng.standard_normal((32, 2))).astype(
            np.complex64
        )
    )
    noise = cp.asarray(
        rng.standard_normal((8, 32, 4)).astype(np.float32) * np.sqrt(0.1)
    )
    save_offsets = (2, 5, 8)

    chunk = integrator.step_chunk(
        y,
        0.0,
        0.1,
        model,
        noise,
        backend,
        n_steps=8,
        save_offsets=save_offsets,
        record_modes=(0, 1),
    )

    current = y.copy()
    expected_saved = []
    for index in range(8):
        current = current + integrator.step(
            current, index * 0.1, 0.1, model, noise[index], backend
        )
        if index + 1 in save_offsets:
            expected_saved.append(current.copy())
    expected = cp.stack(expected_saved, axis=1)

    cp.testing.assert_allclose(chunk.final_state, current, rtol=2e-5, atol=2e-6)
    cp.testing.assert_allclose(chunk.saved_states, expected, rtol=2e-5, atol=2e-6)

    alpha_only = integrator.step_chunk(
        y,
        0.0,
        0.1,
        model,
        noise,
        backend,
        n_steps=8,
        save_offsets=save_offsets,
        record_modes=(0,),
    )
    cp.testing.assert_allclose(
        alpha_only.saved_states, expected[:, :, :1], rtol=2e-5, atol=2e-6
    )
