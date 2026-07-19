"""GPU consistency tests for Kerr model kernels."""

from __future__ import annotations

import multiprocessing
import traceback

import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


def _make_model(model_name):
    if model_name == "kerr_2mode":
        from models.kerr_2mode import Kerr2ModeModel

        return Kerr2ModeModel(
            omega_a=0.0,
            omega_b=-0.01,
            chi=0.01,
            gamma_a=0.5,
            gamma_b=1.8728,
            g=0.5,
        )
    from models.kerr_3mode import Kerr3ModeModel

    return Kerr3ModeModel(
        omega_a=0.5, omega_b=0.3, omega_c=0.2, chi=0.01,
        gamma_a=1.0, gamma_b=0.5, gamma_c=1.0, g_ab=0.1, g_ac=0.05,
    )


@pytest.fixture(params=["kerr_2mode", "kerr_3mode"])
def model(request):
    return _make_model(request.param)


class _CuPyBackendName:
    @staticmethod
    def backend_name():
        return "cupy"


def test_cayley_kernels_are_registered(model):
    backend = _CuPyBackendName()

    assert model.supports_fused_step("cayley_maruyama", backend)
    assert model.supports_fused_chunk("cayley_maruyama", backend)


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
def test_kernelized_terms_match_python(model):
    import cupy as cp
    from qphase.backend.cupy_backend import CuPyBackend

    n = 64
    rng = np.random.default_rng(42)
    y_np = (
        rng.standard_normal((n, model.n_modes))
        + 1j * rng.standard_normal((n, model.n_modes))
    ).astype(np.complex64)
    y = cp.asarray(y_np)

    drift, diffusion = model.kernelized_terms(y, 0.0, model.params, CuPyBackend())
    cp.testing.assert_allclose(
        drift, cp.asarray(model.drift(y_np, 0.0, model.params)), rtol=1e-4, atol=1e-5
    )
    cp.testing.assert_allclose(
        diffusion,
        cp.asarray(model.diffusion(y_np, 0.0, model.params)),
        rtol=1e-4,
        atol=1e-5,
    )


def _cayley_step_worker(model_name, dtype_name, queue):
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase.backend.numpy_backend import NumpyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        dtype = np.complex64 if dtype_name == "complex64" else np.complex128
        real_dtype = np.float32 if dtype_name == "complex64" else np.float64
        n = 24
        dt = 0.02
        rng = np.random.default_rng(45)
        model = _make_model(model_name)
        y_numpy = (
            rng.standard_normal((n, model.n_modes))
            + 1j * rng.standard_normal((n, model.n_modes))
        ).astype(dtype)
        noise_numpy = (
            rng.standard_normal((n, model.noise_dim)).astype(real_dtype) * np.sqrt(dt)
        )
        omega_a = np.linspace(0.1, 0.5, n, dtype=real_dtype)
        model.params["omega_a"] = omega_a
        expected = CayleyMaruyama(fused="off").step(
            y_numpy, 0.0, dt, model, noise_numpy, NumpyBackend()
        )
        model.params["omega_a"] = cp.asarray(omega_a)
        fused = CayleyMaruyama(fused="required").step(
            cp.asarray(y_numpy),
            0.0,
            dt,
            model,
            cp.asarray(noise_numpy),
            CuPyBackend(),
        )
        error = float(np.max(np.abs(cp.asnumpy(fused) - expected)))
        queue.put((True, error))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _cayley_chunk_worker(model_name, queue):
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        model = _make_model(model_name)
        n_steps = 6
        n = 16
        dt = 0.02
        rng = np.random.default_rng(46)
        initial = cp.asarray(
            (
                rng.standard_normal((n, model.n_modes))
                + 1j * rng.standard_normal((n, model.n_modes))
            ).astype(np.complex64)
        )
        noise = cp.asarray(
            rng.standard_normal((n_steps, n, model.noise_dim)).astype(np.float32)
            * np.sqrt(dt)
        )
        save_offsets = (2, 4, 6)
        record_modes = tuple(range(model.n_modes - 1, -1, -1))
        backend = CuPyBackend()
        integrator = CayleyMaruyama(fused="required", chunk_steps=n_steps)
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
        saved = []
        for index in range(n_steps):
            current = current + integrator.step(
                current, index * dt, dt, model, noise[index], backend
            )
            if index + 1 in save_offsets:
                saved.append(current[:, record_modes])
        expected_saved = cp.stack(saved, axis=1)
        errors = {
            "final": float(cp.max(cp.abs(final_state - current)).get()),
            "saved": float(cp.max(cp.abs(saved_states - expected_saved)).get()),
        }
        queue.put((True, errors))
    except Exception:
        queue.put((False, traceback.format_exc()))


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
@pytest.mark.parametrize("model_name", ["kerr_2mode", "kerr_3mode"])
@pytest.mark.parametrize("dtype_name", ["complex64", "complex128"])
def test_cayley_fused_step(model_name, dtype_name):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_cayley_step_worker, args=(model_name, dtype_name, queue)
    )
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "Cayley CUDA worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    tolerance = 3e-5 if dtype_name == "complex64" else 2e-12
    assert payload < tolerance


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
@pytest.mark.parametrize("model_name", ["kerr_2mode", "kerr_3mode"])
def test_cayley_fused_chunk(model_name):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(
        target=_cayley_chunk_worker, args=(model_name, queue)
    )
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "Cayley chunk CUDA worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    assert payload["final"] < 3e-5
    assert payload["saved"] < 3e-5
