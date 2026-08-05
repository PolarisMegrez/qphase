"""GPU contract tests for pair-hopping Cayley-Maruyama kernels."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.backend.cupy_backend import CuPyBackend
from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

from models.pair_hopping_2mode import PairHopping2ModeModel

pytestmark = pytest.mark.gpu


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


@pytest.fixture
def model() -> PairHopping2ModeModel:
    return PairHopping2ModeModel(
        omega_a=0.0,
        omega_b=0.1,
        g=1.0,
        k=0.001,
        gamma_a=1.2941252717,
        gamma_b=1.5490301811,
    )


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_fused_step_matches_generic(model):
    import cupy as cp

    rng = np.random.default_rng(71)
    y = cp.asarray(
        (30.0 * rng.standard_normal((32, 2)) + 30.0j * rng.standard_normal((32, 2)))
        .astype(np.complex128)
    )
    noise = cp.asarray(
        (rng.standard_normal((32, 4)) * np.sqrt(0.02)).astype(np.float64)
    )
    backend = CuPyBackend()
    generic = CayleyMaruyama(fused="off").step(
        y, 0.0, 0.02, model, noise, backend
    )
    fused = CayleyMaruyama(fused="required").step(
        y, 0.0, 0.02, model, noise, backend
    )
    cp.testing.assert_allclose(fused, generic, rtol=2e-12, atol=2e-12)


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_fused_chunk_matches_repeated_steps(model):
    import cupy as cp

    rng = np.random.default_rng(72)
    y = cp.asarray(
        (20.0 * rng.standard_normal((16, 2)) + 20.0j * rng.standard_normal((16, 2)))
        .astype(np.complex128)
    )
    noise = cp.asarray(
        (rng.standard_normal((8, 16, 4)) * np.sqrt(0.02)).astype(np.float64)
    )
    backend = CuPyBackend()
    integrator = CayleyMaruyama(fused="required", chunk_steps=8)
    chunk = integrator.step_chunk(
        y,
        0.0,
        0.02,
        model,
        noise,
        backend,
        n_steps=8,
        save_offsets=(2, 5, 8),
        record_modes=(0, 1),
    )
    chunk_final = chunk.final_state.copy()
    chunk_saved = chunk.saved_states.copy()
    current = y.copy()
    saved = []
    for step in range(8):
        current += integrator.step(
            current, step * 0.02, 0.02, model, noise[step], backend
        )
        if step + 1 in (2, 5, 8):
            saved.append(current.copy())
    cp.testing.assert_allclose(chunk_final, current, rtol=2e-12, atol=2e-12)
    cp.testing.assert_allclose(
        chunk_saved, cp.stack(saved, axis=1), rtol=2e-12, atol=2e-12
    )
