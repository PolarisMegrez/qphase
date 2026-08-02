from types import SimpleNamespace

import pytest
from qphase.core.scan import ScanSpec
from qphase_sde.engine import EngineConfig
from qphase_sde.planning import build_execution_plan


class FakeBackend:
    config = SimpleNamespace(float_dtype="float64")

    def backend_name(self):
        return "cupy"


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
