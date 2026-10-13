from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from qphase.core.scan import ScanSpec
from qphase_sde.analyser.trajectory_diagnostics import TrajectoryDiagnostics
from qphase_sde.contracts.analyser import (
    AnalyserExecutionCapabilities,
    AnalyserWorkspaceEstimate,
    AnalyserWorkspaceRequest,
)
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


class _DeclaredWorkspaceAnalyser:
    def __init__(
        self,
        *,
        device_bytes=0,
        host_bytes=0,
        retained_device_bytes=0,
        retained_host_bytes=0,
        reducer=False,
        scan_tiling=False,
    ):
        self.device_bytes = device_bytes
        self.host_bytes = host_bytes
        self.retained_device_bytes = retained_device_bytes
        self.retained_host_bytes = retained_host_bytes
        self.has_reducer = reducer
        self.scan_tiling = scan_tiling

    def execution_capabilities(self):
        return AnalyserExecutionCapabilities(
            supports_trajectory_batching=True,
            supports_scan_tiling=self.scan_tiling,
        )

    def workspace(self, request):
        del request
        return AnalyserWorkspaceEstimate(
            device_bytes=self.device_bytes,
            host_bytes=self.host_bytes,
            retained_device_bytes=self.retained_device_bytes,
            retained_host_bytes=self.retained_host_bytes,
        )

    def reducer(self):
        return object() if self.has_reducer else None


def _batch_local_workspace_graph(reducer_names=("reducer",)):
    from qphase.data import (
        DataKind,
        ExternalInputBinding,
        GraphEdge,
        ProductDeclaration,
        ProductGraph,
        ProductNode,
        ProductRequirement,
    )

    transform = ProductNode(
        node_id="analyser.transform",
        producer="analyser:transform",
        requirements=[
            ProductRequirement(name="trajectories", kind=DataKind.TIME_SERIES)
        ],
        outputs=[ProductDeclaration(name="signal", kind=DataKind.TIME_SERIES)],
    )
    reducers = [
        ProductNode(
            node_id=f"analyser.{name}",
            producer="analyser:reducer",
            requirements=[
                ProductRequirement(name="signal", kind=DataKind.TIME_SERIES)
            ],
            outputs=[ProductDeclaration(name=name, kind=DataKind.STATISTICS)],
        )
        for name in reducer_names
    ]
    return ProductGraph(
        nodes=[transform, *reducers],
        edges=[
            GraphEdge(
                producer="analyser.transform",
                product="signal",
                consumer=node.node_id,
                requirement="signal",
            )
            for node in reducers
        ],
        external_inputs=[
            ExternalInputBinding(
                consumer="analyser.transform",
                requirement="trajectories",
                source="input",
                product="trajectories",
            )
        ],
    )


@pytest.mark.parametrize(
    ("transform_bytes", "retained_bytes"),
    [
        (10 * 1024**3, [0]),
        (0, [10 * 1024**3]),
        (400 * 1024**2, [400 * 1024**2, 400 * 1024**2]),
    ],
    ids=["oversized-transform", "oversized-reducer", "fan-out"],
)
def test_batch_local_declared_workspace_must_fit_budget(
    transform_bytes, retained_bytes
):
    names = tuple(f"reducer_{index}" for index in range(len(retained_bytes)))
    graph = _batch_local_workspace_graph(names)
    analysers = {
        "transform": _DeclaredWorkspaceAnalyser(device_bytes=transform_bytes),
        **{
            name: _DeclaredWorkspaceAnalyser(
                device_bytes=value,
                retained_host_bytes=value,
                reducer=True,
            )
            for name, value in zip(names, retained_bytes, strict=True)
        },
    }
    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=8,
        trajectory_batching="required",
        trajectory_batch_size=4,
        rng_group_size=4,
        keep_traj=False,
    )

    with pytest.raises(SDEMemoryPlanningError, match="working-set budget"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeBackend(),
            integrator=FakeIntegrator(),
            analysers=analysers,
            resources=SimpleNamespace(gpu_memory_fraction=0.75),
            device_memory=(1024**3, 1024**3),
            product_graph=graph,
        )


def test_multi_point_reducer_state_forces_scan_tiling_or_rejection():
    graph = _batch_local_workspace_graph()
    analysers = {
        "transform": _DeclaredWorkspaceAnalyser(),
        "reducer": _DeclaredWorkspaceAnalyser(
            retained_host_bytes=100 * 1024**2,
            reducer=True,
        ),
    }
    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=8,
        trajectory_batching="required",
        trajectory_batch_size=4,
        rng_group_size=4,
        keep_traj=False,
    )

    with pytest.raises(SDEMemoryPlanningError, match="scan tiling is required"):
        build_execution_plan(
            config=config,
            grid=_grid(10),
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers=analysers,
            resources=SimpleNamespace(memory_limit_mib=512),
            product_graph=graph,
        )


def test_per_point_analyser_memory_scales_monotonically_with_tile_size():
    graph = _batch_local_workspace_graph()
    analysers = {
        "transform": _DeclaredWorkspaceAnalyser(
            device_bytes=40 * 1024**2,
            scan_tiling=True,
        ),
        "reducer": _DeclaredWorkspaceAnalyser(
            retained_device_bytes=10 * 1024**2,
            reducer=True,
            scan_tiling=True,
        ),
    }
    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=8,
        trajectory_batching="required",
        trajectory_batch_size=4,
        rng_group_size=4,
        keep_traj=False,
    )

    small = build_execution_plan(
        config=config,
        grid=_grid(10),
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers=analysers,
        resources=SimpleNamespace(gpu_memory_fraction=0.75),
        device_memory=(512 * 1024**2, 512 * 1024**2),
        product_graph=graph,
    )
    large = build_execution_plan(
        config=config,
        grid=_grid(10),
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers=analysers,
        resources=SimpleNamespace(gpu_memory_fraction=0.75),
        device_memory=(1024**3, 1024**3),
        product_graph=graph,
    )

    assert small.scan_tile_size < large.scan_tile_size == 10
    assert small.memory.analyzer_device_workspace_bytes == (
        50 * 1024**2 * small.scan_tile_size
    )
    assert large.memory.analyzer_device_workspace_bytes == 500 * 1024**2
    assert small.memory.estimated_device_peak_bytes < (
        large.memory.estimated_device_peak_bytes
    )


class _BatchScaledWorkspaceAnalyser(_DeclaredWorkspaceAnalyser):
    def __init__(self, *, location):
        super().__init__(reducer=True, scan_tiling=True)
        self.location = location

    def workspace(self, request):
        scaled = request.n_traj * 2 * 1024**2
        kwargs = {f"{self.location}_bytes": scaled}
        return AnalyserWorkspaceEstimate(**kwargs)


@pytest.mark.parametrize("location", ["device", "host"])
def test_auto_batch_matches_largest_explicit_batch_for_each_memory_location(location):
    graph = _trajectory_graph("reducer")
    analyser = _BatchScaledWorkspaceAnalyser(location=location)
    common = dict(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=256,
        rng_group_size=64,
        keep_traj=False,
    )
    resources = SimpleNamespace(
        gpu_memory_fraction=0.75,
        memory_limit_mib=192 if location == "host" else None,
    )
    kwargs = dict(
        grid=None,
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={"reducer": analyser},
        resources=resources,
        device_memory=(256 * 1024**2, 256 * 1024**2),
        product_graph=graph,
    )

    automatic = build_execution_plan(config=EngineConfig(**common), **kwargs)
    explicit = build_execution_plan(
        config=EngineConfig(**common, trajectory_batch_size=64),
        **kwargs,
    )

    assert automatic.trajectory_batch_size == 64
    assert automatic.trajectory_batch_size == explicit.trajectory_batch_size
    if location == "host":
        assert automatic.memory.estimated_host_peak_bytes <= 192 * 1024**2
    else:
        assert automatic.memory.estimated_device_peak_bytes <= 192 * 1024**2


def _production_reducer_case(name):
    from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
    from qphase_sde.analyser.coherence_carrier import CoherenceCarrierAnalyzer
    from qphase_sde.analyser.coherence_matrix import CoherenceMatrixAnalyzer
    from qphase_sde.analyser.moment_statistics import MomentStatisticsAnalyzer
    from qphase_sde.analyser.psd import PsdAnalyzer
    from qphase_sde.analyser.quadratic_moments import QuadraticMomentAnalyzer

    modes = list(range(32 if name != "moment_statistics" else 64))
    if name == "moment_statistics":
        analyser = MomentStatisticsAnalyzer(modes=modes)
        n_traj = 2048
    elif name == "quadratic_moments":
        matrix = [
            [1.0 if row == column else 0.0 for column in range(16)]
            for row in range(16)
        ]
        analyser = QuadraticMomentAnalyzer(
            modes=list(range(16)),
            observables={f"q{index}": {"matrix": matrix} for index in range(64)},
        )
        n_traj = 16384
    elif name == "coherence_matrix":
        analyser = CoherenceMatrixAnalyzer(modes=modes)
        n_traj = 2048
    elif name == "coherence_carrier":
        analyser = CoherenceCarrierAnalyzer(
            modes=modes,
            maximum_lag_points=256,
        )
        n_traj = 1024
    elif name == "psd":
        analyser = PsdAnalyzer(kind="complex", modes=[0])
        n_traj = 128
    else:
        assert name == "allan_variance"
        analyser = AllanVarianceAnalyzer(modes=[0], points=5)
        n_traj = 128
    return analyser, n_traj


@pytest.mark.parametrize(
    "name",
    [
        "moment_statistics",
        "quadratic_moments",
        "coherence_matrix",
        "coherence_carrier",
    ],
)
def test_production_reducer_retained_host_state_limits_scan_tile(name):
    analyser, n_traj = _production_reducer_case(name)
    config = EngineConfig(
        t0=0.0,
        t1=0.1,
        dt=0.01,
        n_traj=n_traj,
        trajectory_batching="required",
        trajectory_batch_size=64,
        rng_group_size=64,
        keep_traj=False,
    )

    plan = build_execution_plan(
        config=config,
        grid=_grid(8),
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={name: analyser},
        resources=SimpleNamespace(
            gpu_memory_fraction=0.75,
            memory_limit_mib=512,
        ),
        device_memory=(4 * 1024**3, 4 * 1024**3),
        product_graph=_trajectory_graph(name),
    )

    assert plan.trajectory_batch_count > 1
    assert plan.scan_tile_size < plan.scan_size
    assert plan.memory.analyzer_host_workspace_bytes > 0
    assert plan.memory.estimated_host_peak_bytes <= 512 * 1024**2


@pytest.mark.parametrize(
    "name",
    ["moment_statistics", "quadratic_moments", "coherence_matrix"],
)
def test_finalize_peak_has_exact_single_tile_host_budget_boundary(name):
    analyser, n_traj = _production_reducer_case(name)
    batch_size = 64
    estimate = analyser.workspace(
        AnalyserWorkspaceRequest(
            trajectory_bytes=1,
            n_traj=batch_size,
            total_n_traj=n_traj,
            saved_samples=11,
            n_record_modes=FakeModel.n_modes,
            real_itemsize=8,
            backend_name="cupy",
        )
    )
    declared_host_peak = estimate.host_bytes + estimate.retained_host_bytes
    mib = 1024**2

    def fits_one_point(limit_mib):
        budget = limit_mib * mib
        reserve = max(mib, min(256 * mib, budget // 16))
        return reserve + declared_host_peak <= budget

    high_limit = next(
        limit for limit in range(1, 4097) if fits_one_point(limit)
    )
    low_limit = high_limit - 1
    assert not fits_one_point(low_limit)
    config = EngineConfig(
        t0=0.0,
        t1=0.1,
        dt=0.01,
        n_traj=n_traj,
        trajectory_batching="required",
        trajectory_batch_size=batch_size,
        rng_group_size=batch_size,
        keep_traj=False,
    )
    common = dict(
        config=config,
        grid=_grid(2),
        model=FakeModel(),
        backend=FakeBackend(),
        integrator=FakeIntegrator(),
        analysers={name: analyser},
        device_memory=(4 * 1024**3, 4 * 1024**3),
        product_graph=_trajectory_graph(name),
    )

    with pytest.raises(SDEMemoryPlanningError, match="working-set budget"):
        build_execution_plan(
            **common,
            resources=SimpleNamespace(
                gpu_memory_fraction=0.75,
                memory_limit_mib=low_limit,
            ),
        )

    plan = build_execution_plan(
        **common,
        resources=SimpleNamespace(
            gpu_memory_fraction=0.75,
            memory_limit_mib=high_limit,
        ),
    )
    assert plan.scan_tile_size == 1
    assert plan.memory.estimated_host_peak_bytes <= high_limit * mib


@pytest.mark.parametrize(
    "name",
    [
        "psd",
        "allan_variance",
        "moment_statistics",
        "quadratic_moments",
        "coherence_matrix",
        "coherence_carrier",
    ],
)
def test_single_batch_plan_counts_production_retained_memory(name):
    analyser, n_traj = _production_reducer_case(name)
    config = EngineConfig(
        t0=0.0,
        t1=0.1,
        dt=0.01,
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
        analysers={name: analyser},
        resources=None,
        device_memory=None,
        product_graph=_trajectory_graph(name),
    )
    estimate = analyser.workspace(
        AnalyserWorkspaceRequest(
            trajectory_bytes=n_traj * plan.saved_samples * FakeModel.n_modes * 16,
            n_traj=n_traj,
            total_n_traj=n_traj,
            saved_samples=plan.saved_samples,
            n_record_modes=FakeModel.n_modes,
            real_itemsize=8,
            backend_name="cupy",
        )
    )

    assert plan.trajectory_batch_count == 1
    assert plan.memory.analyzer_host_workspace_bytes >= (
        estimate.host_bytes + estimate.retained_host_bytes
    )
    assert plan.memory.analyzer_device_workspace_bytes >= (
        estimate.device_bytes + estimate.retained_device_bytes
    )


def test_single_batch_quadratic_finalize_rejects_insufficient_host_budget():
    name = "quadratic_moments"
    analyser, n_traj = _production_reducer_case(name)
    config = EngineConfig(
        t0=0.0,
        t1=0.1,
        dt=0.01,
        n_traj=n_traj,
        trajectory_batching="off",
        keep_traj=False,
    )

    with pytest.raises(SDEMemoryPlanningError, match="working-set budget"):
        build_execution_plan(
            config=config,
            grid=None,
            model=FakeModel(),
            backend=FakeBackend(),
            integrator=FakeIntegrator(),
            analysers={name: analyser},
            resources=SimpleNamespace(
                gpu_memory_fraction=0.75,
                memory_limit_mib=64,
            ),
            device_memory=(4 * 1024**3, 4 * 1024**3),
            product_graph=_trajectory_graph(name),
        )


def test_batch_local_multi_output_reducer_is_rejected():
    from qphase.data import DataKind, ProductDeclaration

    graph = _batch_local_workspace_graph(("reducer",))
    reducer_node = next(
        node for node in graph.nodes if node.node_id == "analyser.reducer"
    )
    graph = graph.model_copy(
        update={
            "nodes": [
                reducer_node.model_copy(
                    update={
                        "outputs": [
                            ProductDeclaration(
                                name="first", kind=DataKind.STATISTICS
                            ),
                            ProductDeclaration(
                                name="second", kind=DataKind.STATISTICS
                            ),
                        ]
                    }
                )
                if node.node_id == reducer_node.node_id
                else node
                for node in graph.nodes
            ]
        }
    )
    analysers = {
        "transform": _DeclaredWorkspaceAnalyser(),
        "reducer": _DeclaredWorkspaceAnalyser(reducer=True),
    }

    with pytest.raises(SDEMemoryPlanningError, match="one Dataset"):
        build_execution_plan(
            config=EngineConfig(
                t0=0.0,
                t1=1.0,
                dt=0.01,
                n_traj=8,
                trajectory_batching="required",
                trajectory_batch_size=4,
                rng_group_size=4,
                keep_traj=False,
            ),
            grid=None,
            model=FakeModel(),
            backend=FakeNumpyBackend(),
            integrator=FakeIntegrator(),
            analysers=analysers,
            resources=None,
            product_graph=graph,
        )


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
        ),
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
        ),
        workspace=lambda request: AnalyserWorkspaceEstimate(
            device_bytes=2 * request.trajectory_bytes
        ),
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
        workspace=lambda request: AnalyserWorkspaceEstimate(
            retained_host_bytes=(
                request.saved_samples
                * request.n_record_modes
                * request.real_itemsize
                * 5
            )
        ),
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
            workspace=lambda request: AnalyserWorkspaceEstimate(
                host_bytes=2 * request.trajectory_bytes
            ),
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

    def workspace(self, request):
        return AnalyserWorkspaceEstimate(
            retained_host_bytes=(
                request.saved_samples
                * request.n_record_modes
                * request.real_itemsize
                * 5
            )
        )


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
