"""GPU contracts for the collective-loss VDP Cayley kernel."""

from __future__ import annotations

import multiprocessing
import traceback

import numpy as np
import pytest

from models.collective_vdp_2mode import CollectiveVDP2ModeModel

pytestmark = pytest.mark.gpu


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


def _model() -> CollectiveVDP2ModeModel:
    return CollectiveVDP2ModeModel(
        omega_a=0.493,
        omega_b=0.01,
        Gamma=2.8e-4,
        g=0.213,
        pump_a=2.57,
        kappa_bright=1.0,
        kappa_dark=0.0016,
    )


def _step_worker(dtype_name: str, queue) -> None:
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        dtype = np.complex64 if dtype_name == "complex64" else np.complex128
        real_dtype = np.float32 if dtype_name == "complex64" else np.float64
        rng = np.random.default_rng(20260809)
        n, dt = 24, 0.01
        state = cp.asarray(
            (rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))).astype(
                dtype
            )
        )
        noise = cp.asarray(
            (rng.standard_normal((n, 4)) * np.sqrt(dt)).astype(real_dtype)
        )
        model = _model()
        model.params["omega_b"] = cp.linspace(-0.02, 0.02, n, dtype=real_dtype)
        backend = CuPyBackend()
        generic = (
            CayleyMaruyama(fused="off")
            .step(state, 0.0, dt, model, noise, backend)
            .copy()
        )
        fused = (
            CayleyMaruyama(fused="required")
            .step(state, 0.0, dt, model, noise, backend)
            .copy()
        )
        cp.cuda.Stream.null.synchronize()
        queue.put((True, float(cp.max(cp.abs(fused - generic)).get())))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _chunk_worker(queue) -> None:
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        rng = np.random.default_rng(20260810)
        n, n_steps, dt = 16, 7, 0.01
        initial = cp.asarray(
            (rng.standard_normal((n, 2)) + 1j * rng.standard_normal((n, 2))).astype(
                np.complex64
            )
        )
        noise = cp.asarray(
            (rng.standard_normal((n_steps, n, 4)) * np.sqrt(dt)).astype(np.float32)
        )
        model = _model()
        backend = CuPyBackend()
        integrator = CayleyMaruyama(fused="required", chunk_steps=n_steps)
        save_offsets = (2, 5, 7)
        record_modes = (1, 0)
        chunk = integrator.step_chunk(
            initial,
            0.0,
            dt,
            model,
            noise,
            backend,
            n_steps=n_steps,
            save_offsets=save_offsets,
            record_modes=record_modes,
        )
        final_state = chunk.final_state.copy()
        saved_states = chunk.saved_states.copy()
        current = initial.copy()
        expected_saved = []
        for index in range(n_steps):
            current += integrator.step(
                current, index * dt, dt, model, noise[index], backend
            )
            if index + 1 in save_offsets:
                expected_saved.append(current[:, record_modes])
        expected = cp.stack(expected_saved, axis=1)
        cp.cuda.Stream.null.synchronize()
        queue.put(
            (
                True,
                {
                    "final": float(cp.max(cp.abs(final_state - current)).get()),
                    "saved": float(cp.max(cp.abs(saved_states - expected)).get()),
                },
            )
        )
    except Exception:
        queue.put((False, traceback.format_exc()))


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
@pytest.mark.parametrize("dtype_name", ("complex64", "complex128"))
def test_fused_step_matches_generic_path(dtype_name):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_step_worker, args=(dtype_name, queue))
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "collective VDP step worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    tolerance = 3.0e-5 if dtype_name == "complex64" else 3.0e-12
    assert payload < tolerance


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_fused_chunk_matches_repeated_steps():
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_chunk_worker, args=(queue,))
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "collective VDP chunk worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    assert payload["final"] < 3.0e-5
    assert payload["saved"] < 3.0e-5
