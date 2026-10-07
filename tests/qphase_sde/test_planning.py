from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from qphase.core.scan import ScanSpec
from qphase_sde.analyser.trajectory_diagnostics import TrajectoryDiagnostics
from qphase_sde.contracts.analyser import AnalyserExecutionCapabilities
from qphase_sde.engine import EngineConfig
from qphase_sde.planning import SDEMemoryPlanningError, build_execution_plan
from qphase_sde.spectral_estimator.base import SpectralEstimatorCapabilities


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


def _trajectory_graph(*names):
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
        resources=SimpleNamespace(memory_limit_mib=32),
    )

    assert plan.trajectory_batch_size < config.n_traj
    assert plan.trajectory_batch_size % plan.logical_rng_group_size == 0
    assert plan.trajectory_batch_count > 1
    assert plan.scan_tile_size == 1
    assert plan.budget_bytes is not None
    assert plan.memory.estimated_peak_bytes <= plan.budget_bytes


def test_planner_rejects_budget_below_one_logical_rng_group():
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

    with pytest.raises(SDEMemoryPlanningError, match="rng_group_size"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={"psd": psd},
            resources=SimpleNamespace(memory_limit_mib=4),
        )


def test_planner_rejects_requested_batch_misaligned_with_rng_group():
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.01,
        n_traj=256,
        trajectory_batching="required",
        trajectory_batch_size=48,
        keep_traj=False,
    )
    psd = SimpleNamespace(
        name="psd",
        config=SimpleNamespace(expected_freq_max=None),
        create_result_accumulator=lambda: None,
    )

    # 48 is below the default logical RNG group of 64: a hard config error.
    with pytest.raises(ValueError, match="rng_group_size"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={"psd": psd},
            resources=None,
        )

    aligned = config.model_copy(update={"rng_group_size": 16})
    plan = build_execution_plan(
        config=aligned,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={"psd": psd},
        resources=None,
    )
    assert plan.logical_rng_group_size == 16
    assert plan.trajectory_batch_size == 48
    assert plan.trajectory_batch_count == 6

    # A requested size that is not a whole multiple of the group is a hard
    # config error: silently rounding it would change the random streams the
    # user asked for. Budget-derived sizes still round down (tested above).
    capped = config.model_copy(update={"trajectory_batch_size": 100})
    with pytest.raises(ValueError, match="rng_group_size"):
        build_execution_plan(
            config=capped,
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={"psd": psd},
            resources=None,
        )


def test_typed_planner_rejects_non_tileable_products_before_running():
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=1,
        save_stride=1,
        keep_traj=False,
    )
    typed_ragged = SimpleNamespace(
        name="typed_ragged",
        config=SimpleNamespace(),
        output_specs=lambda **kwargs: (),
        execution_capabilities=lambda: AnalyserExecutionCapabilities(),
    )

    with pytest.raises(SDEMemoryPlanningError, match="scan tiling"):
        build_execution_plan(
            config=config,
            grid=_grid(31),
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={"typed_ragged": typed_ragged},
            resources=SimpleNamespace(memory_limit_mib=2),
            typed_execution=True,
            product_graph=_trajectory_graph("typed_ragged"),
        )


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


def _psd_analyser_with_estimator(estimator):
    return SimpleNamespace(
        name="psd",
        config=SimpleNamespace(expected_freq_max=None),
        estimator=estimator,
        create_result_accumulator=lambda: None,
    )


def _workspace_plan(estimator, backend):
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=8,
        save_stride=10,
        record_modes=[0],
        keep_traj=False,
    )
    resources = (
        SimpleNamespace(gpu_memory_fraction=0.75)
        if backend.backend_name() == "cupy"
        else SimpleNamespace(memory_limit_mib=4096)
    )
    return build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=backend,
        integrator=FakeIntegrator(),
        analysers={"psd": _psd_analyser_with_estimator(estimator)},
        resources=resources,
        device_memory=(4 * 1024**3, 4 * 1024**3),
    )


def _fake_estimator(*, backend_native, fft_chunk_trajectories=None):
    return SimpleNamespace(
        capabilities=lambda: SpectralEstimatorCapabilities(
            backend_native=backend_native
        ),
        config=SimpleNamespace(fft_chunk_trajectories=fft_chunk_trajectories),
    )


def test_planner_workspace_relieves_device_for_host_side_estimator():
    native = _workspace_plan(_fake_estimator(backend_native=True), FakeBackend()).memory
    host_side = _workspace_plan(
        _fake_estimator(backend_native=False), FakeBackend()
    ).memory

    assert native.analyzer_device_workspace_bytes > 0
    assert native.analyzer_host_workspace_bytes == 0
    assert host_side.analyzer_device_workspace_bytes == 0
    assert host_side.analyzer_host_workspace_bytes > 0


def test_planner_workspace_scales_with_periodogram_fft_chunk():
    full = _workspace_plan(
        _fake_estimator(backend_native=True), FakeNumpyBackend()
    ).memory.analyzer_workspace_bytes
    chunked = _workspace_plan(
        _fake_estimator(backend_native=True, fft_chunk_trajectories=2),
        FakeNumpyBackend(),
    ).memory.analyzer_workspace_bytes

    assert full == 4 * chunked


def test_planner_workspace_without_estimator_metadata_is_unchanged():
    analysers = {
        "psd": SimpleNamespace(
            name="psd",
            config=SimpleNamespace(expected_freq_max=None),
            create_result_accumulator=lambda: None,
        )
    }
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=8,
        save_stride=10,
        record_modes=[0],
        keep_traj=False,
    )
    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers=analysers,
        resources=SimpleNamespace(memory_limit_mib=4096),
    )

    n_time = plan.steps // config.save_stride + 1
    trajectory_bytes = 8 * n_time * 16
    assert plan.memory.analyzer_workspace_bytes == 2 * trajectory_bytes


def test_planner_accounts_for_host_diagnostics_on_cupy():
    config = EngineConfig(
        t0=0.0,
        t1=100.0,
        dt=0.01,
        n_traj=8,
        save_stride=10,
        record_modes=[0, 1],
        keep_traj=False,
    )
    diagnostics = TrajectoryDiagnostics(modes=[0, 1])
    resources = SimpleNamespace(
        gpu_memory_fraction=0.75,
        memory_limit_mib=4096,
        hardware=SimpleNamespace(
            available_memory_mib=8192,
            total_memory_mib=16384,
        ),
    )

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={"trajectory_diagnostics": diagnostics},
        resources=resources,
        device_memory=(4 * 1024**3, 4 * 1024**3),
    )

    assert plan.memory.analyzer_device_workspace_bytes == 0
    assert plan.memory.analyzer_host_workspace_bytes > 0
    assert plan.host_budget_bytes == 4096 * 1024**2
    assert plan.memory.estimated_host_peak_bytes <= plan.host_budget_bytes


class _FakeModel3Mode(FakeModel):
    n_modes = 3


class _FakeIncrementalAnalyser:
    """Minimal analyser with the typed-reducer interface batching requires."""

    def execution_capabilities(self):
        return AnalyserExecutionCapabilities(supports_trajectory_batching=True)

    def create_result_accumulator(self):
        return object()


_REPO_ROOT = Path(__file__).resolve().parents[2]

# Tracked production workflows whose pre-2.0 runs used a physical batch below
# the old default RNG group (64); each sim job pins rng_group_size to its
# original batch so the legacy (master, point, group) stream is reproduced.
_MIGRATED_RNG_GROUP_JOBS = [
    (
        "configs/workflows/collective_loss_kerr_3mode/reference/acn_bcn/collective_loss_kerr_3mode_figure1_full.yaml",
        "collective_loss_kerr_3mode_figure1_full_sim",
        60,
        60,
    ),
    (
        "configs/workflows/collective_loss_kerr_3mode/reference/refined_dark/collective_loss_kerr_3mode_refined_dark_signed_figure1.yaml",
        "collective_loss_kerr_3mode_refined_dark_signed_figure1_sim",
        32,
        32,
    ),
    (
        "configs/workflows/collective_loss_kerr_3mode/reference/refined_dark/collective_loss_kerr_3mode_refined_dark_signed_rerun.yaml",
        "collective_loss_kerr_3mode_refined_dark_signed_rerun_sim",
        48,
        48,
    ),
    (
        "configs/workflows/collective_loss_kerr_3mode/reference/refined_dark/collective_loss_kerr_3mode_refined_dark_signed_supplement.yaml",
        "collective_loss_kerr_3mode_refined_dark_signed_supplement_sim",
        48,
        48,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0.yaml",
        "kerr_full_3mode_q0_kf3f_019_omega_c_positive",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0.yaml",
        "kerr_full_3mode_q0_kf3f_036_omega_c_negative",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0_branch_followup.yaml",
        "kerr_full_3mode_q0b_kf3f_019_inner",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0_branch_followup.yaml",
        "kerr_full_3mode_q0b_kf3f_019_outer",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0_branch_followup.yaml",
        "kerr_full_3mode_q0b_kf3f_036_crossover",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q0_fallback.yaml",
        "kerr_full_3mode_q0f_kf3f_032_outer",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_frequency_q1_3decade.yaml",
        "kerr_full_3mode_q1_kf3f_032_sim",
        4,
        4,
    ),
    (
        "configs/workflows/kerr_full_3mode/kerr_full_3mode_kf3f_001_batch_benchmark.yaml",
        "kerr_full_3mode_kf3f_001_bench_batch32",
        32,
        32,
    ),
]


@pytest.mark.parametrize(
    ("workflow_path", "job_name", "batch", "group"),
    _MIGRATED_RNG_GROUP_JOBS,
    ids=[f"{Path(p).stem}:{j}" for p, j, _, _ in _MIGRATED_RNG_GROUP_JOBS],
)
def test_migrated_workflow_passes_real_planner_with_pinned_rng_group(
    workflow_path, job_name, batch, group
):
    """Migrated workflows must load into EngineConfig and pass the real SDE
    planner with the explicit batch aligned to the pinned RNG group.
    """
    doc = yaml.safe_load((_REPO_ROOT / workflow_path).read_text(encoding="utf-8"))
    job = next(j for j in doc["jobs"] if j["name"] == job_name)
    sde = job["engine"]["sde"]
    assert sde["trajectory_batch_size"] == batch
    assert sde["rng_group_size"] == group

    config = EngineConfig(**sde)
    assert config.rng_group_size == group

    grid = ScanSpec.model_validate(job["scan"]).compile() if "scan" in job else None
    integrator = SimpleNamespace(
        config=SimpleNamespace(
            chunk_steps=job["integrator"]["cayley_maruyama"]["chunk_steps"]
        )
    )

    plan = build_execution_plan(
        config=config,
        grid=grid,
        model=_FakeModel3Mode(),
        backend=FakeBackend(),
        integrator=integrator,
        analysers={"psd": _FakeIncrementalAnalyser()},
        resources=SimpleNamespace(gpu_memory_fraction=None),
        device_memory=(80 * 1024**3, 80 * 1024**3),
    )

    assert plan.trajectory_batch_size == batch
    assert plan.logical_rng_group_size == group
    assert plan.trajectory_batch_size % plan.logical_rng_group_size == 0
    assert plan.rng_strategy == "logical_trajectory_group_seedsequence_v1"


@pytest.mark.parametrize("n_traj", [65, 100, 127])
def test_required_batching_falls_back_to_one_full_rng_group(n_traj):
    """Forced batching without an explicit batch picks one full RNG group
    plus the tail instead of halving below the group and failing.
    """
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.01,
        n_traj=n_traj,
        save_stride=10,
        record_modes=[0],
        keep_traj=False,
        trajectory_batching="required",
    )

    plan = build_execution_plan(
        config=config,
        grid=None,
        model=FakeModel(),
        backend=FakeNumpyBackend(),
        integrator=FakeIntegrator(),
        analysers={"psd": _FakeIncrementalAnalyser()},
        resources=SimpleNamespace(gpu_memory_fraction=None),
        device_memory=(80 * 1024**3, 80 * 1024**3),
    )

    assert plan.trajectory_batch_size == plan.logical_rng_group_size == 64
    assert plan.trajectory_batch_count == 2


@pytest.mark.parametrize("n_traj", [2, 32, 64])
def test_required_batching_rejects_single_group_ensemble(n_traj):
    """An ensemble covered by one logical RNG group cannot be force-batched."""
    config = EngineConfig(
        t0=0.0,
        t1=10.0,
        dt=0.01,
        n_traj=n_traj,
        save_stride=10,
        record_modes=[0],
        keep_traj=False,
        trajectory_batching="required",
    )

    with pytest.raises(SDEMemoryPlanningError, match="rng_group_size"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers={"psd": _FakeIncrementalAnalyser()},
            resources=SimpleNamespace(gpu_memory_fraction=None),
            device_memory=(80 * 1024**3, 80 * 1024**3),
        )


class _FakeAdaptiveIntegrator(FakeIntegrator):
    def supports_adaptive_step(self):
        return True


def _adaptive_config(**overrides):
    base = {
        "t0": 0.0,
        "t1": 10.0,
        "dt": 0.01,
        "n_traj": 128,
        "save_stride": 10,
        "record_modes": [0],
        "keep_traj": False,
        "adaptive": True,
    }
    base.update(overrides)
    return EngineConfig(**base)


def _adaptive_plan(config, **overrides):
    kwargs = {
        "config": config,
        "grid": None,
        "model": FakeModel(),
        "backend": FakeNumpyBackend(),
        "integrator": _FakeAdaptiveIntegrator(),
        "analysers": {"psd": _FakeIncrementalAnalyser()},
        "resources": SimpleNamespace(gpu_memory_fraction=None),
        "device_memory": (80 * 1024**3, 80 * 1024**3),
    }
    kwargs.update(overrides)
    return build_execution_plan(**kwargs)


def test_planner_rejects_adaptive_with_required_batching():
    with pytest.raises(ValueError, match="adaptive stepping"):
        _adaptive_plan(_adaptive_config(trajectory_batching="required"))


def test_planner_rejects_adaptive_with_explicit_batch_below_ensemble():
    with pytest.raises(ValueError, match="adaptive stepping"):
        _adaptive_plan(_adaptive_config(trajectory_batch_size=64))


def test_planner_rejects_adaptive_when_budget_forces_batching():
    config = _adaptive_config(
        n_traj=1000,
        t1=100.0,
        save_stride=1,
    )
    with pytest.raises(SDEMemoryPlanningError, match="adaptive stepping"):
        _adaptive_plan(
            config,
            resources=SimpleNamespace(memory_limit_mib=32),
            device_memory=None,
        )


def test_planner_rejects_adaptive_when_budget_forces_scan_tiling():
    config = _adaptive_config(
        n_traj=1000,
        t1=400000.0,
        dt=0.08,
        save_stride=100,
        trajectory_batching="off",
    )
    with pytest.raises(SDEMemoryPlanningError, match="adaptive stepping"):
        _adaptive_plan(
            config,
            grid=_grid(31),
            backend=FakeBackend(),
            resources=SimpleNamespace(gpu_memory_fraction=0.75),
            device_memory=(4 * 1024**3, 4 * 1024**3),
        )


def test_planner_allows_adaptive_single_batch_single_tile():
    config = _adaptive_config(n_traj=8)
    plan = _adaptive_plan(config, grid=_grid(3))
    assert plan.trajectory_batch_size == 8
    assert plan.trajectory_batch_count == 1
    assert plan.tile_count == 1
