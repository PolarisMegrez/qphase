"""Planner-facing workspace honesty of the spectral analysers (Gate 6D/6E).

The scalar Allan estimator is host-only today: every selected channel is
copied to the host and reduced with a NumPy cumsum, so the declared
capability and workspace must reflect the host path instead of inheriting
the phase estimator's device estimate. PSD retained output scales with the
configured channel count, never with the physical record-mode count.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.contracts.analyser import AnalyserWorkspaceRequest
from qphase_sde.engine import EngineConfig
from qphase_sde.planning import build_execution_plan


class FakeBackend:
    config = SimpleNamespace(float_dtype="float64")

    def backend_name(self):
        return "cupy"


class FakeNumpyBackend(FakeBackend):
    def backend_name(self):
        return "numpy"


class FakeModel:
    name = "fake"
    n_modes = 3
    noise_dim = 6


class FakeIntegrator:
    config = SimpleNamespace(chunk_steps=128)


def _request(
    n_traj: int,
    saved_samples: int,
    n_modes: int,
    *,
    itemsize: int = 8,
    backend: str = "cupy",
    total_n_traj: int | None = None,
) -> AnalyserWorkspaceRequest:
    return AnalyserWorkspaceRequest(
        trajectory_bytes=n_traj * saved_samples * n_modes * 2 * itemsize,
        n_traj=n_traj,
        total_n_traj=total_n_traj,
        saved_samples=saved_samples,
        n_record_modes=n_modes,
        real_itemsize=itemsize,
        backend_name=backend,
    )


def _scalar_allan(n_channels: int = 1) -> AllanVarianceAnalyzer:
    return AllanVarianceAnalyzer(
        measurement={
            "sampled_scalar": {
                "channels": [f"n{index}" for index in range(n_channels)]
            }
        },
        points=8,
    )


def _phase_allan() -> AllanVarianceAnalyzer:
    return AllanVarianceAnalyzer(
        measurement={"phase_frequency": {"channels": [0]}}, points=8
    )


def _scalar_psd(n_channels: int) -> PsdAnalyzer:
    return PsdAnalyzer(
        measurement={
            "sampled_scalar": {
                "channels": [f"n{index}" for index in range(n_channels)],
                "detrend": "constant",
            }
        }
    )


def _field_psd(modes: list[int]) -> PsdAnalyzer:
    return PsdAnalyzer(
        measurement={
            "complex_field": {"channels": modes, "representation": "complex"}
        }
    )


def _trajectory_graph(*names: str):
    """Minimal typed graph: each named analyser node consumes trajectories."""
    from qphase.data import (
        DataKind,
        ExternalInputBinding,
        ProductDeclaration,
        ProductGraph,
        ProductNode,
        ProductRequirement,
    )

    nodes = [
        ProductNode(
            node_id=f"analyser.{name}",
            producer=f"analyser:{name}",
            requirements=[
                ProductRequirement(name="trajectories", kind=DataKind.TIME_SERIES)
            ],
            outputs=[ProductDeclaration(name=name, kind=DataKind.STATISTICS)],
        )
        for name in names
    ]
    bindings = [
        ExternalInputBinding(
            consumer=f"analyser.{name}",
            requirement="trajectories",
            source="input",
            product="trajectories",
        )
        for name in names
    ]
    return ProductGraph(nodes=nodes, external_inputs=bindings)


def test_scalar_allan_declares_host_execution():
    assert _scalar_allan().execution_capabilities().preferred_location == "host"
    assert _phase_allan().execution_capabilities().preferred_location == "backend"


def test_scalar_allan_workspace_covers_full_host_channel():
    # Audit probe: 64e6 float64 samples of one channel need at least the
    # 512 MB host copy; cumsum and difference temporaries add to that.
    n_traj, saved_samples = 64, 1_000_000
    estimate = _scalar_allan().workspace(_request(n_traj, saved_samples, 3))
    total_samples = n_traj * saved_samples
    assert estimate.device_bytes == 0
    assert estimate.retained_device_bytes == 0
    assert estimate.host_bytes >= total_samples * (8 + 8 + 8)
    assert estimate.host_bytes <= total_samples * 64
    assert estimate.retained_host_bytes > 0


def test_scalar_allan_workspace_scales_with_batch_not_total():
    batched = _scalar_allan().workspace(_request(16, 1000, 3, total_n_traj=128))
    full = _scalar_allan().workspace(_request(128, 1000, 3, total_n_traj=128))
    assert batched.host_bytes < full.host_bytes
    assert batched.retained_host_bytes == full.retained_host_bytes


def test_scalar_allan_float32_input_still_counts_float64_reduction():
    estimate = _scalar_allan().workspace(_request(64, 1000, 3, itemsize=4))
    assert estimate.host_bytes >= 64 * 1000 * (4 + 8 + 8)


def test_phase_allan_keeps_device_workspace_estimate():
    estimate = _phase_allan().workspace(_request(64, 1000, 3))
    assert estimate.device_bytes > 0
    assert estimate.host_bytes >= 0


@pytest.mark.parametrize(("n_modes", "n_channels"), [(3, 5), (3, 3), (3, 2)])
def test_psd_retained_tracks_selected_channels(n_modes, n_channels):
    estimate = _scalar_psd(n_channels).workspace(_request(64, 1000, n_modes))
    assert estimate.retained_host_bytes == 1000 * n_channels * 8 * 5


def test_psd_retained_ignores_record_mode_count():
    few_modes = _scalar_psd(2).workspace(_request(64, 1000, 3))
    many_modes = _scalar_psd(2).workspace(_request(64, 1000, 9))
    assert few_modes.retained_host_bytes == many_modes.retained_host_bytes


def test_field_psd_retained_tracks_selected_modes_not_recorded_modes():
    estimate = _field_psd([0, 2]).workspace(_request(64, 1000, 3))
    assert estimate.retained_host_bytes == 1000 * 2 * 8 * 5


def test_scalar_allan_plan_counts_host_workspace_and_no_device_workspace():
    analyser = _scalar_allan()
    n_traj = 32
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.1,
        n_traj=n_traj,
        trajectory_batching="off",
        keep_traj=False,
    )
    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={"allan_variance": analyser},
        resources=None,
        device_memory=None,
        product_graph=_trajectory_graph("allan_variance"),
    )
    estimate = analyser.workspace(
        _request(n_traj, plan.saved_samples, FakeModel.n_modes)
    )
    assert plan.memory.analyzer_device_workspace_bytes == 0
    assert plan.memory.analyzer_host_workspace_bytes >= (
        estimate.host_bytes + estimate.retained_host_bytes
    )


def test_scalar_allan_host_budget_forces_trajectory_batches():
    analyser = _scalar_allan()
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=1000,
        save_stride=1,
        record_modes=[0],
        keep_traj=False,
    )
    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={"allan_variance": analyser},
        resources=SimpleNamespace(memory_limit_mib=64),
        product_graph=_trajectory_graph("allan_variance"),
    )
    assert plan.memory.analyzer_host_workspace_bytes > 0
    assert plan.trajectory_batch_size < config.n_traj
    assert plan.trajectory_batch_count > 1
    assert plan.budget_bytes is not None
    assert plan.memory.estimated_peak_bytes <= plan.budget_bytes


def test_phase_allan_still_uses_device_workspace_under_cupy():
    analyser = _phase_allan()
    n_traj = 32
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.1,
        n_traj=n_traj,
        trajectory_batching="off",
        keep_traj=False,
    )
    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={"allan_variance": analyser},
        resources=None,
        device_memory=None,
        product_graph=_trajectory_graph("allan_variance"),
    )
    assert plan.memory.analyzer_device_workspace_bytes > 0
