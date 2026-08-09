"""GPU contracts for the fpgen-generated collective Kerr kernels."""

from __future__ import annotations

import multiprocessing
import traceback

import numpy as np
import pytest

from models.collective_kerr_2mode import CollectiveKerr2ModeModel
from models.collective_loss_kerr_3mode import CollectiveLossKerr3ModeModel
from models.reservoir_kerr_3mode import ReservoirKerr3ModeModel

pytestmark = pytest.mark.gpu


def _cupy_available() -> bool:
    try:
        import cupy as cp

        cp.cuda.runtime.getDevice()
        return True
    except Exception:
        return False


def _make_model(model_name: str):
    if model_name == "collective_kerr_2mode":
        return CollectiveKerr2ModeModel(
            omega_0=0.0,
            delta=0.1,
            chi=0.01,
            g=0.5,
            kappa_bright=1.2,
            pump_bright=0.4,
            kappa_dark=0.02,
        )
    if model_name.startswith("collective_loss_kerr_3mode"):
        signed = model_name.endswith("_signed")
        return CollectiveLossKerr3ModeModel(
            omega_a=0.0,
            omega_b=0.2,
            omega_c=-0.1,
            chi=0.01,
            g_ab=0.4,
            g_ac=-0.3 if signed else 0.3,
            g_bc=-0.05 if signed else 0.05,
            pump_a=0.2,
            kappa_bright=1.0,
            kappa_dark=0.02,
        )
    return ReservoirKerr3ModeModel(
        omega_r=0.0,
        omega_0=0.0,
        delta=0.1,
        chi=0.01,
        g=0.5,
        g_r=0.8,
        kappa_r=4.0,
        pump_r=1.0,
        kappa_local=0.02,
    )


@pytest.fixture(
    params=(
        "collective_kerr_2mode",
        "collective_loss_kerr_3mode",
        "reservoir_kerr_3mode",
    )
)
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


def test_generated_source_records_fpgen_fingerprint(model):
    provider = tuple(model.kernel_plugins())[0]
    fingerprint = model.cam_fpgen_dynamics().to_model_spec(name=model.name).fingerprint

    assert fingerprint in provider._generated[0]
    assert fingerprint in provider._generated[1]


def _step_worker(model_name: str, dtype_name: str, queue) -> None:
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        model = _make_model(model_name)
        dtype = np.complex64 if dtype_name == "complex64" else np.complex128
        real_dtype = np.float32 if dtype_name == "complex64" else np.float64
        rng = np.random.default_rng(81 + model.n_modes)
        n = 24
        dt = 0.01
        state = cp.asarray(
            (
                rng.standard_normal((n, model.n_modes))
                + 1j * rng.standard_normal((n, model.n_modes))
            ).astype(dtype)
        )
        noise = cp.asarray(
            (rng.standard_normal((n, model.noise_dim)) * np.sqrt(dt)).astype(real_dtype)
        )
        scan_parameter = (
            "omega_c"
            if model_name.startswith("collective_loss_kerr_3mode")
            else "delta"
        )
        model.params[scan_parameter] = cp.linspace(-0.15, 0.15, n, dtype=real_dtype)
        backend = CuPyBackend()
        generic = (
            CayleyMaruyama(fused="off")
            .step(state, 0.0, dt, model, noise, backend)
            .copy()
        )
        cp.cuda.Stream.null.synchronize()
        fused = (
            CayleyMaruyama(fused="required")
            .step(state, 0.0, dt, model, noise, backend)
            .copy()
        )
        cp.cuda.Stream.null.synchronize()
        queue.put((True, float(cp.max(cp.abs(fused - generic)).get())))
    except Exception:
        queue.put((False, traceback.format_exc()))


def _chunk_worker(model_name: str, queue) -> None:
    try:
        import cupy as cp
        from qphase.backend.cupy_backend import CuPyBackend
        from qphase_sde.integrator.cayley_maruyama import CayleyMaruyama

        model = _make_model(model_name)
        rng = np.random.default_rng(91 + model.n_modes)
        n = 16
        n_steps = 7
        dt = 0.01
        initial = cp.asarray(
            (
                rng.standard_normal((n, model.n_modes))
                + 1j * rng.standard_normal((n, model.n_modes))
            ).astype(np.complex64)
        )
        noise = cp.asarray(
            (rng.standard_normal((n_steps, n, model.noise_dim)) * np.sqrt(dt)).astype(
                np.float32
            )
        )
        save_offsets = (2, 5, 7)
        record_modes = tuple(reversed(range(model.n_modes)))
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
        cp.cuda.Stream.null.synchronize()
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
@pytest.mark.parametrize(
    "model_name",
    (
        "collective_kerr_2mode",
        "collective_loss_kerr_3mode",
        "collective_loss_kerr_3mode_signed",
        "reservoir_kerr_3mode",
    ),
)
@pytest.mark.parametrize("dtype_name", ("complex64", "complex128"))
def test_fused_step_matches_fpgen_generic_path(model_name, dtype_name):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_step_worker, args=(model_name, dtype_name, queue))
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "generated Cayley step worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    tolerance = 3e-5 if dtype_name == "complex64" else 3e-12
    assert payload < tolerance


@pytest.mark.skipif(not _cupy_available(), reason="CuPy not available")
@pytest.mark.parametrize(
    "model_name",
    (
        "collective_kerr_2mode",
        "collective_loss_kerr_3mode",
        "collective_loss_kerr_3mode_signed",
        "reservoir_kerr_3mode",
    ),
)
def test_fused_chunk_matches_repeated_steps(model_name):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=_chunk_worker, args=(model_name, queue))
    process.start()
    process.join(timeout=90)

    assert not process.is_alive(), "generated Cayley chunk worker timed out"
    assert process.exitcode == 0
    success, payload = queue.get(timeout=5)
    assert success, payload
    assert payload["final"] < 3e-5
    assert payload["saved"] < 3e-5
