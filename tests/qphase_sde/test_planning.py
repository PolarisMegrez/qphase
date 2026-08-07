from types import SimpleNamespace

import pytest
from qphase.core.scan import ScanSpec
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
    n_modes = 2
    noise_dim = 4


class FakeIntegrator:
    config = SimpleNamespace(chunk_steps=128)


def _grid(size=31):
    return ScanSpec.model_validate(
        {
            "axes": {
                "omega": {
                    "target": "model.fake.omega",
                    "values": list(range(size)),
                }
            }
        }
    ).compile()


def test_planner_rejects_invalid_psd_bandwidth_before_allocation():
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.08,
        n_traj=1,
        save_stride=1000,
        record_modes=[0],
    )
    psd = SimpleNamespace(
        config=SimpleNamespace(
            expected_freq_max=0.34,
            convention="symmetric",
        )
    )

    with pytest.raises(ValueError, match="save_stride <= 115"):
        build_execution_plan(
            config=config,
            grid=_grid(2),
            model=FakeModel(),
            backend=FakeBackend(),
            integrator=FakeIntegrator(),
            analysers={"psd": psd},
            resources=SimpleNamespace(gpu_memory_fraction=None),
            device_memory=(4 * 1024**3, 4 * 1024**3),
        )


def test_planner_tiles_production_shape_from_resource_object():
    config = EngineConfig(
        t0=0.0,
        t1=400000.0,
        dt=0.08,
        n_traj=1000,
        save_stride=100,
        record_modes=[0],
        keep_traj=False,
    )
    psd = SimpleNamespace(
        config=SimpleNamespace(
            expected_freq_max=0.34,
            convention="symmetric",
        )
    )
    resources = SimpleNamespace(gpu_memory_fraction=0.75)

    plan = build_execution_plan(
        config=config,
        grid=_grid(),
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={"psd": psd},
        resources=resources,
        device_memory=(4 * 1024**3, 4 * 1024**3),
    )

    assert plan.scan_tile_size == 1
    assert plan.tile_count == 31
    assert plan.stream_analysis is True
    assert plan.gpu_memory_fraction == 0.75
    assert plan.memory.full_scan_trajectory_bytes > 20 * 1024**3
    assert plan.budget_bytes == 3 * 1024**3


def test_planner_batches_trajectories_when_one_point_exceeds_host_budget():
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=1000,
        save_stride=1,
        record_modes=[0],
        keep_traj=False,
    )
    psd = SimpleNamespace(
        name="psd",
        config=SimpleNamespace(expected_freq_max=None),
        create_result_accumulator=lambda: None,
    )

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={"psd": psd},
        resources=SimpleNamespace(memory_limit_mib=4),
    )

    assert plan.trajectory_batch_size < config.n_traj
    assert plan.trajectory_batch_count > 1
    assert plan.scan_tile_size == 1
    assert plan.budget_bytes is not None
    assert plan.memory.estimated_peak_bytes <= plan.budget_bytes


def test_planner_uses_dynamic_host_memory_when_limit_is_unset():
    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=2,
        save_stride=10,
    )
    resources = SimpleNamespace(
        memory_limit_mib=None,
        hardware=SimpleNamespace(
            available_memory_mib=8000,
            total_memory_mib=32000,
        ),
    )

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={},
        resources=resources,
    )

    assert plan.budget_bytes == 6000 * 1024**2
    assert plan.available_device_bytes == 8000 * 1024**2
    assert plan.device_total_bytes == 32000 * 1024**2


def test_planner_counts_warmup_as_work_but_not_as_saved_samples():
    config = EngineConfig(
        t0=2.0,
        t1=10.0,
        dt=0.1,
        n_traj=2,
        save_stride=5,
    )

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={},
        resources=SimpleNamespace(memory_limit_mib=None, hardware=None),
    )

    assert plan.steps == 100
    assert plan.warmup_steps == 20
    assert plan.observation_steps == 80
    assert plan.saved_samples == 17


def test_planner_rejects_observation_boundary_off_the_integration_grid():
    config = EngineConfig(t0=0.15, t1=1.0, dt=0.1)

    with pytest.raises(ValueError, match="t0=.*integer multiple"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={},
            resources=None,
        )
