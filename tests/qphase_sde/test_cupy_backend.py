"""qphase_sde: CuPy backend smoke tests.

These tests are skipped automatically when ``cupy`` is not installed or when
no CUDA device is available. They verify that the GPU backend produces
numerically consistent results with the NumPy backend and that GPU memory
optimizations (forced trajectory drop) are active.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend

pytestmark = [pytest.mark.gpu, pytest.mark.integration]

cupy = pytest.importorskip("cupy")

try:
    cupy.cuda.Device().use()
    _CUPY_AVAILABLE = True
except Exception:  # pragma: no cover
    _CUPY_AVAILABLE = False

if not _CUPY_AVAILABLE:
    pytest.skip("No CUDA device available", allow_module_level=True)

from qphase.backend.cupy_backend import CuPyBackend  # noqa: E402
from qphase_sde.engine import Engine, EngineConfig  # noqa: E402
from qphase_sde.integrator.srk import GenericSRK  # noqa: E402
from qphase_sde.model import FunctionalSDEModel  # noqa: E402


def _dummy_model():
    """Return a tiny complex-noise SDE model for smoke tests."""

    def drift(y, t, p):
        return -0.01 * y

    def diffusion(y, t, p):
        n = y.shape[0]
        L = np.zeros((n, 2, 2), dtype=y.dtype)
        L[:, 0, 0] = 0.1
        L[:, 1, 1] = 0.1
        return L

    return FunctionalSDEModel(
        name="dummy",
        n_modes=2,
        noise_basis="complex",
        noise_dim=4,
        params={},
        drift=drift,
        diffusion=diffusion,
    )


def test_cupy_backend_primitives():
    """CuPy backend implements the expanded primitive set."""
    be = CuPyBackend()
    x = be.asarray([1.0, 2.0, 3.0])
    assert hasattr(be, "fftshift")
    assert hasattr(be, "histogram")
    assert hasattr(be, "histogram2d")
    assert hasattr(be, "std")
    assert hasattr(be, "matmul")

    hist, edges = be.histogram(x, bins=3)
    assert isinstance(hist, cupy.ndarray)
    assert isinstance(edges, cupy.ndarray)


def test_cupy_contract_noise_matches_numpy():
    """SDE noise contraction on cupy matches numpy reference."""
    from qphase_sde.math import ops

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_modes, m = 4, 3, 6
    L_np = np.random.randn(n_traj, n_modes, m) + 1j * np.random.randn(
        n_traj, n_modes, m
    )
    dW_np = np.random.randn(n_traj, m)

    L_cp = cp_be.asarray(L_np)
    dW_cp = cp_be.asarray(dW_np)

    out_np = ops.contract_noise(L_np, dW_np, np_be)
    out_cp = ops.contract_noise(L_cp, dW_cp, cp_be)

    np.testing.assert_allclose(out_np, cupy.asnumpy(out_cp), rtol=1e-5)


def test_cupy_engine_honours_explicit_trajectory_retention():
    """CuPy analysis runs on device, then an explicit record moves to host."""
    be = CuPyBackend()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=10.0,
            dt=0.1,
            n_traj=5,
            seed=42,
            ic=["1.0+0.0j", "0.0+0.0j"],
            keep_traj=True,
        ),
        plugins={
            "backend": be,
            "model": _dummy_model(),
            "integrator": GenericSRK(),
            # GenericSRK defaults to method="heun"
            "analyser": {
                "psd": type(
                    "PsdStub",
                    (),
                    {
                        "name": "psd",
                        "analyze": lambda self, data, backend: type(
                            "Res",
                            (),
                            {"data_dict": {"ok": True}},
                        )(),
                    },
                )()
            },
        },
    )

    bundle = engine.run()
    assert "trajectories" in bundle.products
    alpha = bundle.products["trajectories"].handle("alpha").materialize()
    assert isinstance(alpha, np.ndarray)
    assert bundle.metadata["trajectory_storage"] == "host"
    ok = bundle.products["psd"].handle("ok").materialize()
    assert bool(ok) is True


def test_cupy_vs_numpy_psd_periodogram():
    """CuPy periodogram matches numpy periodogram for simple signal."""
    from qphase_sde.analyser.psd import PsdAnalyzer, PsdAnalyzerConfig

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_time = 4, 256
    t = np.arange(n_time) * 0.1
    signal = (
        np.exp(1j * 2.0 * t)[None, :]
        + 0.1 * np.random.randn(n_traj, n_time)
        + 0.1j * np.random.randn(n_traj, n_time)
    )

    analyzer = PsdAnalyzer(
        config=PsdAnalyzerConfig(modes=[0], kind="complex", convention="symmetric")
    )

    axis_np, P_np = analyzer._compute_single(signal, 0.1, backend=np_be)
    axis_cp, P_cp = analyzer._compute_single(cp_be.asarray(signal), 0.1, backend=cp_be)

    np.testing.assert_allclose(axis_np, axis_cp, rtol=1e-6)
    np.testing.assert_allclose(P_np, P_cp, rtol=1e-4)

    estimate_np = analyzer._estimate_single(signal, 0.1, backend=np_be)
    estimate_cp = analyzer._estimate_single(cp_be.asarray(signal), 0.1, backend=cp_be)
    np.testing.assert_allclose(estimate_np.std, estimate_cp.std, rtol=1e-4)
    np.testing.assert_allclose(estimate_np.sem, estimate_cp.sem, rtol=1e-4)


def test_cupy_dist_matches_numpy():
    """CuPy dist analyzer matches numpy reference."""
    from qphase_sde.analyser.dist import DistAnalyzer, DistAnalyzerConfig

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_time = 4, 256
    data = np.random.randn(n_traj, n_time, 2) + 1j * np.random.randn(n_traj, n_time, 2)

    dist_np = DistAnalyzer(
        config=DistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).analyze(data, np_be)
    data_cp = cp_be.asarray(data)
    dist_cp = DistAnalyzer(
        config=DistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).analyze(data_cp, cp_be)

    np.testing.assert_allclose(
        dist_np.data_dict["distributions"][0]["hist"],
        dist_cp.data_dict["distributions"][0]["hist"],
        rtol=1e-5,
    )


def test_cupy_polar_dist_matches_numpy():
    """CuPy polar_dist analyzer matches numpy reference."""
    from qphase_sde.analyser.polar_dist import (
        PolarDistAnalyzer,
        PolarDistAnalyzerConfig,
    )

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_time = 4, 256
    data = np.random.randn(n_traj, n_time, 2) + 1j * np.random.randn(n_traj, n_time, 2)

    pdist_np = PolarDistAnalyzer(
        config=PolarDistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).analyze(data, np_be)
    pdist_cp = PolarDistAnalyzer(
        config=PolarDistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).analyze(cp_be.asarray(data), cp_be)

    np.testing.assert_allclose(
        pdist_np.data_dict["distributions"][0]["hist"],
        pdist_cp.data_dict["distributions"][0]["hist"],
        rtol=1e-5,
    )


def test_cupy_vdp_end_to_end_smoke():
    """End-to-end smoke test of the VDP model on CuPy with PSD analysis."""
    from qphase_sde.analyser.psd import PsdAnalyzer, PsdAnalyzerConfig

    from models.vdp_2mode import VDP2ModeModel

    cp_be = CuPyBackend()
    model = VDP2ModeModel(
        omega_a=0.00251189,
        omega_b=0.0,
        gamma_a=2.0,
        gamma_b=1.0,
        Gamma=0.00001,
        g=0.5,
    ).to_diffusive_sde_model()

    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=100.0,
            dt=0.1,
            n_traj=10,
            seed=42,
            ic=["800.0+0.0j", "0.0-800.0j"],
            save_stride=10,
            keep_traj=True,
        ),
        plugins={
            "backend": cp_be,
            "model": model,
            "integrator": GenericSRK(),
            "analyser": {
                "psd": PsdAnalyzer(
                    config=PsdAnalyzerConfig(
                        modes=[0], kind="complex", find_peaks=False
                    )
                )
            },
        },
    )

    bundle = engine.run()
    assert "trajectories" in bundle.products
    alpha = bundle.products["trajectories"].handle("alpha").materialize()
    assert isinstance(alpha, np.ndarray)
    assert "psd" in bundle.products
    psd_data = bundle.products["psd"]
    variables = {variable.name for variable in psd_data.variables}
    assert "axis" in variables
    assert "psd" in variables
    psd_values = psd_data.handle("psd").materialize()
    assert psd_values.shape[1] == 1  # one mode requested


def test_cupy_scheduler_vdp_smoke(tmp_path, temp_project):
    """Run a minimal vdp_2mode job via the scheduler on CuPy."""
    from qphase.core.registry import discovery
    from qphase.core.scheduler import Scheduler
    from qphase.core.system_config import SystemConfig
    from qphase.core.workflow import load_workflow

    discovery.discover_plugins()
    discovery.discover_local_plugins()

    config_path = tmp_path / "cupy_vdp.yaml"
    config_path.write_text(
        """
schema: qphase.workflow/2
id: cupy_vdp_smoke
title: CuPy VDP smoke test
jobs:
  - name: cupy_vdp_smoke
    save: true
    engine:
      sde:
        t0: 0.0
        t1: 100.0
        dt: 0.1
        n_traj: 10
        seed: 42
        ic:
          - ["800.0+0.0j", "0.0-800.0j"]
        adaptive: false
        save_stride: 10
        keep_traj: true
    backend:
      cupy:
        float_dtype: float32
    integrator:
      srk:
        method: heun
    model:
      vdp_2mode:
        omega_a: 0.00251189
        omega_b: 0.0
        gamma_a: 2.0
        gamma_b: 1.0
        Gamma: 0.00001
        g: 0.5
    analyser:
      psd:
        modes: [0]
        kind: complex
        find_peaks: false
"""
    )

    system_config = SystemConfig()

    job_list = load_workflow(config_path)
    captured_job_dir = None

    def _on_job_dir(job_dir: Path) -> None:
        nonlocal captured_job_dir
        captured_job_dir = job_dir

    scheduler = Scheduler(
        system_config=system_config,
        project=temp_project,
        on_job_dir=_on_job_dir,
    )
    results = scheduler.run(job_list)

    assert len(results) == 1
    assert results[0].success
    assert captured_job_dir is not None

    # Typed bundles persist as a v4 artifact directory, not an npz result.
    from qphase.data import load_products
    from qphase_sde.result import legacy_view_from_products

    manifest_path = captured_job_dir / "artifact_manifest.json"
    assert manifest_path.exists()
    products = {
        name: dataset.materialize()
        for name, dataset in load_products(captured_job_dir).items()
    }
    result = legacy_view_from_products(products)
    assert result.trajectory is not None
    assert isinstance(result.trajectory.data, np.ndarray)
    assert "psd" in result.analysis
