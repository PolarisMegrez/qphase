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
from qphase.core.scan import ScanSpec
from qphase.data import ExternalInputBinding, ProductGraph, ProductNode

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
from qphase_sde.analyser.psd import PsdAnalyzer  # noqa: E402
from qphase_sde.engine import Engine, EngineConfig  # noqa: E402
from qphase_sde.graph import compile_product_graph  # noqa: E402
from qphase_sde.integrator.euler_maruyama import EulerMaruyama  # noqa: E402
from qphase_sde.integrator.srk import GenericSRK  # noqa: E402
from qphase_sde.model import FunctionalSDEModel  # noqa: E402
from qphase_sde.state import TrajectorySet  # noqa: E402


def _graph_for(label, analyzer):
    node = ProductNode(
        node_id=f"analyser.{label}",
        producer=f"analyser:{label}",
        requirements=list(analyzer.input_requirements()),
        outputs=list(analyzer.output_specs(label=label)),
    )
    return ProductGraph(
        nodes=[node],
        external_inputs=[
            ExternalInputBinding(
                consumer=node.node_id,
                requirement=requirement.name,
                source="input",
                product="trajectories",
            )
            for requirement in node.requirements
        ],
    )


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


def test_cupy_backend_construction_does_not_replace_global_allocator():
    """Multiple backend instances must keep existing device arrays valid."""
    allocator = cupy.cuda.get_allocator()
    existing = cupy.arange(16, dtype=cupy.float64)

    backends = (CuPyBackend(), CuPyBackend())

    assert cupy.cuda.get_allocator() is allocator
    assert [backend.backend_name() for backend in backends] == ["cupy", "cupy"]
    backends[0].free_all_blocks()
    np.testing.assert_array_equal(cupy.asnumpy(existing), np.arange(16))


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
    """Explicitly retained typed trajectories stay device-resident in memory."""
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
        },
    )

    bundle = engine.run()
    assert "trajectories" in bundle.products
    alpha = bundle.products["trajectories"].handle("alpha").materialize()
    assert isinstance(alpha, cupy.ndarray)


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
        config=PsdAnalyzerConfig(
            measurement={
                "complex_field": {"channels": [0], "representation": "complex"}
            },
            convention="symmetric",
        )
    )

    from qphase_sde.result import _trajectory_product

    np_input = _trajectory_product(
        TrajectorySet(signal[:, :, None], t0=0.0, dt=0.1),
        scan_size=None,
        n_traj_per_point=None,
    )
    cp_input = _trajectory_product(
        TrajectorySet(cp_be.asarray(signal[:, :, None]), t0=0.0, dt=0.1),
        scan_size=None,
        n_traj_per_point=None,
    )
    product_np = analyzer.execute({"signal": np_input}, backend=np_be, label="psd")[
        "psd"
    ]
    product_cp = analyzer.execute({"signal": cp_input}, backend=cp_be, label="psd")[
        "psd"
    ]
    axis_np = product_np.handle("axis").materialize()
    axis_cp = product_cp.handle("axis").materialize()
    P_np = product_np.handle("psd").materialize()
    P_cp = product_cp.handle("psd").materialize()

    np.testing.assert_allclose(axis_np, axis_cp, rtol=1e-6)
    np.testing.assert_allclose(P_np, P_cp, rtol=1e-4)

    np.testing.assert_allclose(
        product_np.handle("psd_std").materialize(),
        product_cp.handle("psd_std").materialize(),
        rtol=1e-4,
    )
    np.testing.assert_allclose(
        product_np.handle("psd_sem").materialize(),
        product_cp.handle("psd_sem").materialize(),
        rtol=1e-4,
    )


def test_cupy_typed_scan_and_trajectory_batching_roundtrip(tmp_path):
    from qphase.core.registry import discovery, registry

    def model():
        def drift(y, t, params):
            del t
            xp = cupy.get_array_module(y)
            rate = xp.asarray(params["rate"])
            return -rate[..., None] * y

        def diffusion(y, t, params):
            del t, params
            xp = cupy.get_array_module(y)
            return xp.zeros(y.shape + (1,), dtype=y.real.dtype)

        return FunctionalSDEModel(
            name="cupy_typed_scan",
            n_modes=1,
            noise_basis="real",
            noise_dim=1,
            params={"rate": 0.1},
            drift=drift,
            diffusion=diffusion,
        )

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={
            "analyser": {
                "psd": {
                    "measurement": {
                        "complex_field": {
                            "channels": [0],
                            "representation": "complex",
                        }
                    }
                }
            }
        },
        registry=registry,
        input_bindings={"psd": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.cupy_typed_scan.rate",
                    "values": [0.1, 0.2, 0.3, 0.4],
                }
            }
        }
    ).compile()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.08,
            dt=0.01,
            n_traj=64,
            trajectory_batching="required",
            trajectory_batch_size=32,
            rng_group_size=32,
            seed=53,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": CuPyBackend(),
            "model": model(),
            "integrator": EulerMaruyama(),
            "analyser": {
                "psd": PsdAnalyzer(
                    measurement={
                        "complex_field": {
                            "channels": [0],
                            "representation": "complex",
                        }
                    }
                )
            },
        },
        product_graph=graph,
    )
    context = type(
        "Context",
        (),
        {
            "parameter_grid": grid,
            "product_graph": graph,
            "progress": None,
            "cancellation": None,
            "metadata": {},
            "resources": None,
        },
    )()

    result = engine.run(context=context)

    assert result.shape == (4,)
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2
    assert np.isfinite(result.products["psd"].handle("psd").materialize()).all()
    result.save_dataset(
        tmp_path,
        layout="single",
        shard_target_bytes=1024 * 1024,
    )
    from qphase.data import load_bundle

    restored = load_bundle(tmp_path)
    assert restored.shape == (4,)


def test_cupy_dist_matches_numpy():
    """CuPy dist analyzer matches numpy reference."""
    from qphase_sde.analyser.dist import DistAnalyzer, DistAnalyzerConfig
    from qphase_sde.result import _trajectory_product

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_time = 4, 256
    data = np.random.randn(n_traj, n_time, 2) + 1j * np.random.randn(n_traj, n_time, 2)

    dist_np = DistAnalyzer(
        config=DistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).execute(
        {
            "trajectories": _trajectory_product(
                TrajectorySet(data, t0=0.0, dt=1.0),
                scan_size=None,
                n_traj_per_point=None,
            )
        },
        backend=np_be,
        label="dist",
    )["dist"]
    data_cp = cp_be.asarray(data)
    dist_cp = DistAnalyzer(
        config=DistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).execute(
        {
            "trajectories": _trajectory_product(
                TrajectorySet(data_cp, t0=0.0, dt=1.0),
                scan_size=None,
                n_traj_per_point=None,
            )
        },
        backend=cp_be,
        label="dist",
    )["dist"]

    np.testing.assert_allclose(
        dist_np.handle("distributions.0.hist").materialize(),
        dist_cp.handle("distributions.0.hist").materialize(),
        rtol=1e-5,
    )


def test_cupy_polar_dist_matches_numpy():
    """CuPy polar_dist analyzer matches numpy reference."""
    from qphase_sde.analyser.polar_dist import (
        PolarDistAnalyzer,
        PolarDistAnalyzerConfig,
    )
    from qphase_sde.result import _trajectory_product

    np_be = NumpyBackend()
    cp_be = CuPyBackend()

    n_traj, n_time = 4, 256
    data = np.random.randn(n_traj, n_time, 2) + 1j * np.random.randn(n_traj, n_time, 2)

    pdist_np = PolarDistAnalyzer(
        config=PolarDistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).execute(
        {
            "trajectories": _trajectory_product(
                TrajectorySet(data, t0=0.0, dt=1.0),
                scan_size=None,
                n_traj_per_point=None,
            )
        },
        backend=np_be,
        label="polar_dist",
    )["polar_dist"]
    pdist_cp = PolarDistAnalyzer(
        config=PolarDistAnalyzerConfig(modes=[0], bins=20, density=True)
    ).execute(
        {
            "trajectories": _trajectory_product(
                TrajectorySet(cp_be.asarray(data), t0=0.0, dt=1.0),
                scan_size=None,
                n_traj_per_point=None,
            )
        },
        backend=cp_be,
        label="polar_dist",
    )["polar_dist"]

    np.testing.assert_allclose(
        pdist_np.handle("distributions.0.hist").materialize(),
        pdist_cp.handle("distributions.0.hist").materialize(),
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

    analyzer = PsdAnalyzer(
        config=PsdAnalyzerConfig(
            measurement={
                "complex_field": {"channels": [0], "representation": "complex"}
            }
        )
    )
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
            "analyser": {"psd": analyzer},
        },
        product_graph=_graph_for("psd", analyzer),
    )

    bundle = engine.run()
    assert "trajectories" in bundle.products
    alpha = bundle.products["trajectories"].handle("alpha").materialize()
    assert isinstance(alpha, cupy.ndarray)
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
        measurement:
          complex_field:
            channels: [0]
            representation: complex
    input_bindings:
      psd:
        signal: input.trajectories
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

    manifest_path = captured_job_dir / "artifact_manifest.json"
    assert manifest_path.exists()
    products = {
        name: dataset.materialize()
        for name, dataset in load_products(captured_job_dir).items()
    }
    assert "trajectories" in products
    assert isinstance(
        products["trajectories"].handle("alpha").materialize(), np.ndarray
    )
    assert "psd" in products
