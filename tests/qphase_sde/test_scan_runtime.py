import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.dataset import DatasetResultProtocol
from qphase.core.progress import ProgressReporter
from qphase.core.scan import ScanSpec
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ExternalInputBinding,
    GraphEdge,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
    ProductSchema,
    SpectralQuantity,
    StatisticsDataset,
    VariableSchema,
    concat_datasets,
    load_bundle,
)
from qphase_sde.analyser.allan_variance import AllanVarianceAnalyzer
from qphase_sde.analyser.base import (
    Analyzer,
    map_trajectory_product,
    trajectory_arrays,
    trajectory_requirement,
)
from qphase_sde.analyser.coherence_carrier import CoherenceCarrierAnalyzer
from qphase_sde.analyser.coherence_matrix import CoherenceMatrixAnalyzer
from qphase_sde.analyser.moment_statistics import MomentStatisticsAnalyzer
from qphase_sde.analyser.peaks import PeaksAnalyzer, PeaksAnalyzerConfig
from qphase_sde.analyser.psd import PsdAnalyzer
from qphase_sde.analyser.quadratic_moments import QuadraticMomentAnalyzer
from qphase_sde.contracts.bundle import SDEProvenance
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.peak_finder import ScipyPeakFinder
from qphase_sde.result import SDEDataBundle, _trajectory_product

pytestmark = pytest.mark.integration

#: Shared complex-field measurement declaration of the psd call sites.
_PSD_MEASUREMENT = {"complex_field": {"channels": [0], "representation": "complex"}}
_PSD_CONFIG = {"measurement": _PSD_MEASUREMENT}
#: Shared phase-frequency measurement declaration of the allan call sites.
_ALLAN_MEASUREMENT = {
    "phase_frequency": {"channels": [0], "min_independent_windows": 1}
}


def _psd_analyzer() -> PsdAnalyzer:
    return PsdAnalyzer(measurement=_PSD_MEASUREMENT)


def _allan_analyzer() -> AllanVarianceAnalyzer:
    return AllanVarianceAnalyzer(
        measurement=_ALLAN_MEASUREMENT, points=5, min_windows=2
    )


def test_typed_scan_refuses_ragged_table_concatenation():
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=1),
            AxisSchema(name="group", role=AxisRole.INDEX, size=1),
        ],
        variables=[
            VariableSchema(
                name="value",
                dtype="float64",
                value_domain="real",
                dims=("scan", "group"),
            )
        ],
        attributes={"table": "peak_candidates"},
    )
    dataset = StatisticsDataset.from_arrays(
        schema, {"value": np.asarray([[1.0]])}, owner="test"
    )
    with pytest.raises(RuntimeError, match="ragged/table products"):
        concat_datasets([dataset, dataset], axis="scan")


def test_typed_scan_concat_rejects_inconsistent_tile_metadata():
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="scan", role=AxisRole.PARAMETER, size=1)],
        variables=[
            VariableSchema(
                name="value",
                dtype="float64",
                value_domain="real",
                dims=("scan",),
            )
        ],
        attributes={"source": "tile"},
    )
    first = StatisticsDataset.from_arrays(
        schema, {"value": np.asarray([1.0])}, owner="test"
    )
    changed_schema = schema.model_copy(update={"attributes": {"source": "other"}})
    second = StatisticsDataset.from_arrays(
        changed_schema, {"value": np.asarray([2.0])}, owner="test"
    )
    with pytest.raises(ValueError, match="inconsistent attributes"):
        concat_datasets([first, second], axis="scan")


class ScannedDummyModel:
    name = "scanned_dummy"
    n_modes = 1
    noise_basis = "real"
    noise_dim = 1

    def __init__(self):
        self._params = {"rate": 1.0}

    @property
    def params(self):
        return self._params

    def drift(self, y, t, params):
        del t
        return -np.asarray(params["rate"])[..., None] * y

    def diffusion(self, y, t, params):
        del t, params
        return np.zeros(y.shape + (1,))


class MeanAnalyzer(Analyzer):
    name = "mean"
    config = SimpleNamespace(expected_freq_max=None)

    def input_requirements(self):
        return [trajectory_requirement(value_domain=None)]

    def execution_capabilities(self):
        from qphase_sde.contracts.analyser import AnalyserExecutionCapabilities

        return AnalyserExecutionCapabilities(supports_scan_tiling=True)

    def output_specs(self, *, label):
        return (
            ProductDeclaration(name=label, kind=DataKind.STATISTICS, fields=["mean"]),
        )

    def execute(
        self, inputs, *, backend, label, metadata=None, context=None
    ):
        del backend, metadata
        trajectory = inputs["trajectories"]

        def compute(point):
            values, _, _, _ = trajectory_arrays(point)
            schema = ProductSchema(
                kind=DataKind.STATISTICS,
                variables=[
                    VariableSchema(
                        name="mean", dtype="<f8", value_domain="real", dims=()
                    )
                ],
                attributes={"graph_ready": True},
            )
            return {
                label: StatisticsDataset.from_arrays(
                    schema,
                    {"mean": np.asarray(float(np.mean(np.asarray(values))))},
                    owner="test.mean",
                )
            }

        return map_trajectory_product(trajectory, compute, context=context)


class IdentityBatchAnalyzer(Analyzer):
    """Batch-local transform probe with no reducer."""

    name = "identity_batch"

    def __init__(self):
        self.calls = 0

    def input_requirements(self):
        return [trajectory_requirement(value_domain=None)]

    def output_specs(self, *, label):
        return (
            ProductDeclaration(
                name=label,
                kind=DataKind.TIME_SERIES,
                quantity="field_amplitude",
                fields=["alpha", "valid_length"],
            ),
        )

    def execution_capabilities(self):
        from qphase_sde.contracts.analyser import AnalyserExecutionCapabilities

        return AnalyserExecutionCapabilities(
            supports_trajectory_batching=True,
            supports_scan_tiling=True,
        )

    def execute(self, inputs, *, backend, label, metadata=None, context=None):
        del backend, metadata, context
        self.calls += 1
        return {label: inputs["trajectories"]}


class FinalSpectrumSummaryAnalyzer(Analyzer):
    """Final-scope probe consuming a merged spectral scan."""

    name = "final_spectrum_summary"

    def input_requirements(self):
        return [
            ProductRequirement(
                name="psd_a",
                kind=DataKind.SPECTRAL,
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                required_fields=["psd"],
            )
        ]

    def output_specs(self, *, label):
        return (
            ProductDeclaration(
                name=label, kind=DataKind.STATISTICS, fields=["mean_power"]
            ),
        )

    def execute(self, inputs, *, backend, label, metadata=None, context=None):
        del backend, metadata, context
        values = np.asarray(inputs["psd_a"].handle("psd").materialize())
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(
                    name="mean_power",
                    dtype="<f8",
                    value_domain="real",
                    dims=(),
                )
            ],
            attributes={"graph_ready": True},
        )
        return {
            label: StatisticsDataset.from_arrays(
                schema,
                {"mean_power": np.asarray(float(np.mean(values)))},
                owner="test.final_summary",
            )
        }


def _branched_batch_graph(*, publish_identity=False):
    identity = IdentityBatchAnalyzer()
    psd_a = _psd_analyzer()
    psd_b = _psd_analyzer()
    nodes = [
        ProductNode(
            node_id="analyser.identity",
            producer="analyser:identity_batch",
            requirements=identity.input_requirements(),
            outputs=list(identity.output_specs(label="identity")),
            publish_outputs=["identity"] if publish_identity else [],
        ),
        ProductNode(
            node_id="analyser.psd_a",
            producer="analyser:psd",
            requirements=psd_a.input_requirements(),
            outputs=list(psd_a.output_specs(label="psd_a")),
        ),
        ProductNode(
            node_id="analyser.psd_b",
            producer="analyser:psd",
            requirements=psd_b.input_requirements(),
            outputs=list(psd_b.output_specs(label="psd_b")),
        ),
    ]
    graph = ProductGraph(
        nodes=nodes,
        edges=[
            GraphEdge(
                producer="analyser.identity",
                product="identity",
                consumer=node_id,
                requirement="signal",
            )
            for node_id in ("analyser.psd_a", "analyser.psd_b")
        ],
        external_inputs=[
            ExternalInputBinding(
                consumer="analyser.identity",
                requirement="trajectories",
                source="input",
                product="trajectories",
            )
        ],
    )
    return graph, identity, psd_a, psd_b


def _partitioned_batch_graph():
    graph, identity, psd_a, psd_b = _branched_batch_graph()
    summary = FinalSpectrumSummaryAnalyzer()
    nodes = [
        *graph.nodes,
        ProductNode(
            node_id="analyser.summary",
            producer="analyser:final_spectrum_summary",
            requirements=summary.input_requirements(),
            outputs=list(summary.output_specs(label="summary")),
        ),
    ]
    edges = [
        *graph.edges,
        GraphEdge(
            producer="analyser.psd_a",
            product="psd_a",
            consumer="analyser.summary",
            requirement="psd_a",
        ),
    ]
    return (
        ProductGraph(
            nodes=nodes,
            edges=edges,
            external_inputs=graph.external_inputs,
        ),
        identity,
        psd_a,
        psd_b,
        summary,
    )


def _mean_graph() -> ProductGraph:
    return ProductGraph(
        nodes=[
            ProductNode(
                node_id="analyser.mean",
                producer="analyser:mean",
                requirements=[trajectory_requirement(value_domain=None)],
                outputs=list(MeanAnalyzer().output_specs(label="mean")),
                publish_outputs=["mean"],
            )
        ],
        external_inputs=[
            ExternalInputBinding(
                consumer="analyser.mean",
                requirement="trajectories",
                source="input",
                product="trajectories",
            )
        ],
    )


def _graph(analyser_config, *, signal_bindings=()):
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": analyser_config},
        registry=registry,
        input_bindings={
            label: {"signal": "input.trajectories"} for label in signal_bindings
        }
        or None,
    )
    assert graph is not None
    return graph


class StochasticScannedModel(ScannedDummyModel):
    def diffusion(self, y, t, params):
        del t, params
        return np.ones(y.shape + (1,))


def _product_arrays(dataset):
    """Materialize every variable of a bundle product as a name→array dict."""
    return {
        variable.name: dataset.handle(variable.name).materialize()
        for variable in dataset.variables
    }


def _grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0],
                }
            }
        }
    ).compile()


def _four_point_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0, 3.0, 4.0],
                }
            }
        }
    ).compile()


def _single_point_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [2.0],
                }
            }
        }
    ).compile()


def test_sde_engine_adapts_parameter_grid_to_existing_fused_path():
    model = ScannedDummyModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.02,
            dt=0.01,
            n_traj=2,
            seed=7,
            ic=[[1.0]],
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
        },
    )

    result = engine.run(context=SimpleNamespace(parameter_grid=_grid(), progress=None))

    assert isinstance(result, DatasetResultProtocol)
    assert result.shape == (2,)
    # The fused (scan*n_traj) leading dimension is now (scan, trajectory).
    alpha = result.products["trajectories"].handle("alpha").materialize()
    assert alpha.shape[0] * alpha.shape[1] == 4
    point = result.point_view((0,))
    point_alpha = point.products["trajectories"].handle("alpha").materialize()
    assert point_alpha.shape[0] == 2
    assert result.point_view((1,)).metadata["scan_point"]["rate"] == 2.0
    assert engine.config.n_traj == 2
    assert model.params == {"rate": 1.0}


def test_sde_engine_analyzes_resource_limited_scan_tiles():
    model = ScannedDummyModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=1.0,
            dt=0.01,
            n_traj=1000,
            seed=7,
            ic=[[1.0]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"mean": MeanAnalyzer()},
        },
        product_graph=_mean_graph(),
    )
    context = SimpleNamespace(
        parameter_grid=_grid(),
        progress=None,
        resources=SimpleNamespace(
            memory_limit_mib=4,
            gpu_memory_fraction=None,
        ),
        cancellation=None,
    )

    result = engine.run(context=context)

    assert "trajectories" not in result.products
    mean = result.products["mean"].handle("mean").materialize()
    assert len(mean) == 2
    assert result.metadata["execution_plan"]["scan_tile_size"] == 1
    assert (
        result.metadata["execution_plan"]["rng_strategy"]
        == "logical_trajectory_group_seedsequence_v1"
    )
    assert engine.config.n_traj == 1000
    assert model.params == {"rate": 1.0}


def test_sde_scan_rng_is_independent_of_tile_size():
    def run(memory_limit_mib):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.1,
                dt=0.01,
                n_traj=5000,
                seed=19,
                ic=[[1.0]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"mean": MeanAnalyzer()},
            },
            product_graph=_mean_graph(),
        )
        context = SimpleNamespace(
            parameter_grid=_four_point_grid(),
            progress=None,
            resources=SimpleNamespace(
                memory_limit_mib=memory_limit_mib,
                gpu_memory_fraction=None,
            ),
            cancellation=None,
        )
        result = engine.run(context=context)
        means = list(result.products["mean"].handle("mean").materialize())
        return (
            result.metadata["execution_plan"]["scan_tile_size"],
            result.metadata["execution_plan"]["tile_count"],
            result.metadata["execution_plan"]["rng_strategy"],
            means,
        )

    tile_unlimited, count_unlimited, strategy_unlimited, means_unlimited = run(None)
    tile_one, _, _, means_one = run(3)
    tile_two, _, _, means_two = run(5)

    assert count_unlimited == 1
    assert strategy_unlimited == "logical_trajectory_group_seedsequence_v1"
    assert tile_one == 1
    assert tile_two == 2
    # A single fused tile and any budget-driven tiling must draw the same
    # per-point streams, bit for bit.
    np.testing.assert_array_equal(means_unlimited, means_one)
    np.testing.assert_array_equal(means_one, means_two)


def test_scan_bundle_saves_finite_shards(tmp_path):
    trajectory = SimpleNamespace(
            data=np.zeros((8, 3, 1)),
            t0=0.0,
            dt=0.1,
            meta={},
    )
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": [1.0, 2.0, 3.0, 4.0],
                }
            }
        }
    ).compile()
    bundle = SDEDataBundle(
        {
            "trajectories": _trajectory_product(
                trajectory, scan_size=4, n_traj_per_point=2
            )
        },
        SDEProvenance(dt=0.1),
        scan_axes=dict(grid.axes),
        scan_shape=grid.shape,
        n_traj_per_point=2,
    )

    report = bundle.save_dataset(
        tmp_path / "sde",
        layout="sharded",
        shard_target_bytes=1,
    )

    assert report.layout == "sharded"
    assert len(report.files) > 1
    loaded = load_bundle(tmp_path / "sde")
    assert loaded.shape == (4,)
    assert loaded.point_view((3,)).metadata["scan_point"]["rate"] == 4.0


def test_trajectory_batch_size_does_not_change_psd_random_streams():
    def run(batch_size, rng_group_size=64):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.08,
                dt=0.01,
                n_traj=256,
                trajectory_batching="required",
                trajectory_batch_size=batch_size,
                rng_group_size=rng_group_size,
                seed=23,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
        )
        result = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
        psd = _product_arrays(result.products["psd"])
        return psd, result.metadata["execution_plan"]

    batch_64, plan_64 = run(64)
    batch_128, plan_128 = run(128)

    assert plan_64["trajectory_batch_count"] == 4
    assert plan_128["trajectory_batch_count"] == 2
    np.testing.assert_allclose(batch_64["psd"], batch_128["psd"], rtol=1e-12)
    np.testing.assert_allclose(batch_64["psd_std"], batch_128["psd_std"], rtol=1e-12)

    # Below one logical group the physical batch cannot cover whole groups:
    # with the default group of 64 a batch of 32 is a config error, not a
    # silently different random stream.
    with pytest.raises(ValueError, match="rng_group_size"):
        run(32)

    # With a smaller provenance group, batches 16/32/64 all reproduce the
    # same per-point streams; aggregation-order float noise stays far below
    # the solver tolerance.
    ref_psd, ref_plan = run(64, rng_group_size=16)
    assert ref_plan["logical_rng_group_size"] == 16
    for batch_size in (16, 32):
        small_psd, small_plan = run(batch_size, rng_group_size=16)
        assert small_plan["trajectory_batch_size"] == batch_size
        np.testing.assert_allclose(small_psd["psd"], ref_psd["psd"], rtol=1e-12)
        np.testing.assert_allclose(small_psd["psd_std"], ref_psd["psd_std"], rtol=1e-12)


def test_off_auto_and_required_batching_share_random_streams():
    """Every batching mode derives streams per (seed, point, logical group)."""

    def run(batching, batch_size=None):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.08,
                dt=0.01,
                n_traj=256,
                trajectory_batching=batching,
                trajectory_batch_size=batch_size,
                seed=31,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
        )
        result = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
        return _product_arrays(result.products["psd"]), result.metadata[
            "execution_plan"
        ]

    off_psd, off_plan = run("off")
    auto_psd, auto_plan = run("auto")
    batched_psd, batched_plan = run("required", 64)

    assert off_plan["trajectory_batch_count"] == 1
    assert auto_plan["trajectory_batch_count"] == 1
    assert batched_plan["trajectory_batch_count"] == 4
    for name in ("psd", "psd_std"):
        np.testing.assert_allclose(off_psd[name], batched_psd[name], rtol=1e-12)
        np.testing.assert_allclose(auto_psd[name], batched_psd[name], rtol=1e-12)


def test_tail_rng_group_keeps_batching_invariance():
    """A short tail group (n_traj % group != 0) stays batching-invariant."""

    def run(batching, batch_size=None):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.08,
                dt=0.01,
                n_traj=250,
                trajectory_batching=batching,
                trajectory_batch_size=batch_size,
                seed=37,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
        )
        result = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
        return _product_arrays(result.products["psd"]), result.metadata[
            "execution_plan"
        ]

    ref_psd, ref_plan = run("off")
    batched_64, plan_64 = run("required", 64)
    batched_128, plan_128 = run("required", 128)

    assert ref_plan["trajectory_batch_count"] == 1
    assert plan_64["trajectory_batch_count"] == 4
    assert plan_128["trajectory_batch_count"] == 2
    for name in ("psd", "psd_std"):
        np.testing.assert_allclose(ref_psd[name], batched_64[name], rtol=1e-12)
        np.testing.assert_allclose(ref_psd[name], batched_128[name], rtol=1e-12)


def test_required_batching_falls_back_to_full_group_plus_tail():
    """n_traj=65 with required batching and no explicit batch resolves to one
    full RNG group plus a tail batch, reproducing the non-batched streams.
    """

    def run(batching, batch_size=None):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.08,
                dt=0.01,
                n_traj=65,
                trajectory_batching=batching,
                trajectory_batch_size=batch_size,
                seed=47,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
        )
        result = engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))
        return _product_arrays(result.products["psd"]), result.metadata[
            "execution_plan"
        ]

    ref_psd, ref_plan = run("off")
    batched_psd, batched_plan = run("required")

    assert ref_plan["trajectory_batch_count"] == 1
    assert batched_plan["trajectory_batch_size"] == 64
    assert batched_plan["trajectory_batch_count"] == 2
    for name in ("psd", "psd_std"):
        np.testing.assert_allclose(ref_psd[name], batched_psd[name], rtol=1e-12)


def test_scan_fused_and_batched_paths_share_point_streams():
    """Fused whole-scan and per-point batched runs draw identical streams."""

    def run(batching, batch_size=None):
        model = StochasticScannedModel()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=0.08,
                dt=0.01,
                n_traj=250,
                trajectory_batching=batching,
                trajectory_batch_size=batch_size,
                seed=43,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": model,
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
        )
        result = engine.run(
            context=SimpleNamespace(
                parameter_grid=_grid(),
                progress=None,
                cancellation=None,
            )
        )
        return _product_arrays(result.products["psd"]), result.metadata[
            "execution_plan"
        ]

    fused_psd, fused_plan = run("off")
    batched_psd, batched_plan = run("required", 128)

    # The fused run covers per-point tail groups (64, 64, 64, 58) in the
    # middle of the fused ensemble; the batched run covers them in the final
    # batch of each point. Both must draw the same per-point streams.
    assert fused_plan["trajectory_batch_count"] == 1
    assert batched_plan["trajectory_batch_count"] == 2
    assert fused_psd["psd"].shape[0] == 2
    for name in ("psd", "psd_std"):
        np.testing.assert_allclose(fused_psd[name], batched_psd[name], rtol=1e-12)


def test_scan_runs_trajectory_batches_inside_each_parameter_point():
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=1.0,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=29,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"psd": _psd_analyzer()},
        },
        product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=_grid(),
            progress=None,
            cancellation=None,
        )
    )

    assert "trajectories" not in result.products
    psd = _product_arrays(result.products["psd"])
    assert psd["psd"].shape[0] == 2
    # The bridge flattens the nested "uncertainty" mapping into dotted
    # variables; the per-point independent trajectory count is pinned
    # directly, exactly as the 1.x per-point payload asserted.
    np.testing.assert_array_equal(psd["uncertainty.n_independent"], [128, 128])
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2
    assert engine.config.n_traj == 128
    assert model.params == {"rate": 1.0}


def test_typed_dense_scan_supports_tiles_and_trajectory_batches(tmp_path):
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": {"psd": _PSD_CONFIG}},
        registry=registry,
        input_bindings={"psd": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=1.0,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=41,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"psd": _psd_analyzer()},
        },
        product_graph=graph,
    )

    def run(memory_limit_mib):
        events = []
        reporter = ProgressReporter(events.append)
        run_engine = Engine(
            config=engine.config.model_copy(deep=True),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": StochasticScannedModel(),
                "analyser": {"psd": _psd_analyzer()},
            },
            product_graph=graph,
        )
        run_context = SimpleNamespace(
            parameter_grid=ScanSpec.model_validate(
                {
                    "axes": {
                        "rate": {
                            "target": "model.scanned_dummy.rate",
                            "values": list(np.arange(16, dtype=float) + 1.0),
                        }
                    }
                }
            ).compile(),
            product_graph=graph,
            progress=reporter,
            cancellation=None,
            metadata={},
            resources=SimpleNamespace(
                memory_limit_mib=memory_limit_mib,
                gpu_memory_fraction=None,
            ),
        )
        run_result = run_engine.run(context=run_context)
        return run_result, _product_arrays(run_result.products["psd"]), events

    result, wide_tile, wide_events = run(6)
    narrow_result, narrow_tile, narrow_events = run(2)

    assert result.shape == (16,)
    assert "trajectories" not in result.products
    assert result.products["psd"].axis("scan").size == 16
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2
    assert result.metadata["execution_plan"]["scan_tile_size"] == 16
    assert narrow_result.metadata["execution_plan"]["scan_tile_size"] < 16
    assert narrow_result.metadata["execution_plan"]["tile_count"] > 1
    sampling = [
        event
        for event in wide_events
        if event.kind == "progress" and event.stage == "sampling"
    ]
    completed = [int(event.completed) for event in sampling]
    assert completed == sorted(completed)
    assert completed[-1] == 16 * 128 * 100
    narrow_completed = [
        int(event.completed)
        for event in narrow_events
        if event.kind == "progress" and event.stage == "sampling"
    ]
    assert narrow_completed == sorted(narrow_completed)
    assert narrow_completed[-1] == 16 * 128 * 100
    for name in ("psd", "psd_std", "psd_sem", "uncertainty.n_independent"):
        np.testing.assert_allclose(wide_tile[name], narrow_tile[name], rtol=1e-12)
    artifact_dir = narrow_result.save_dataset(
        tmp_path / "typed_scan",
        layout="single",
        shard_target_bytes=1024 * 1024,
    )
    restored = load_bundle(tmp_path / "typed_scan")
    assert artifact_dir.schema_version
    assert restored.shape == narrow_result.shape
    np.testing.assert_allclose(
        _product_arrays(restored.products["psd"])["psd"],
        narrow_tile["psd"],
        rtol=1e-12,
    )
    probe = (
        "import numpy as np; import qphase_sde; "
        "from qphase.data import load_bundle; "
        f"bundle=load_bundle(r'{tmp_path / 'typed_scan'}'); "
        "assert bundle.shape == (16,); "
        "assert bundle.products['psd'].axis('scan').size == 16; "
        "assert np.isfinite(bundle.products['psd'].handle('psd')"
        ".materialize()).all()"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


def test_allan_variance_runs_through_engine_trajectory_batches():
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.8,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=37,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"allan_variance": _allan_analyzer()},
        },
        product_graph=_graph(
            {
                "allan_variance": {
                    "measurement": _ALLAN_MEASUREMENT,
                    "points": 5,
                    "min_windows": 2,
                }
            },
            signal_bindings=("allan_variance",),
        ),
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            progress=None,
            cancellation=None,
        )
    )

    payload = _product_arrays(result.products["allan_variance"])
    assert "trajectories" not in result.products
    assert payload["n_traj"] == 128
    # The bridge flattens the nested "mode_results" mapping into dotted
    # variables; the merged per-trajectory Allan rows number n_traj.
    assert payload["mode_results.0.allan.per_trajectory"].shape[0] == 128
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2


def test_coherence_matrix_runs_through_engine_trajectory_batches():
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.8,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=41,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {
                "coherence_matrix": CoherenceMatrixAnalyzer(
                    modes=[0],
                    time_blocks=2,
                    min_block_samples=2,
                )
            },
        },
        product_graph=_graph(
            {
                "coherence_matrix": {
                    "modes": [0],
                    "time_blocks": 2,
                    "min_block_samples": 2,
                }
            }
        ),
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            progress=None,
            cancellation=None,
        )
    )

    payload = _product_arrays(result.products["coherence_matrix"])
    assert "trajectories" not in result.products
    assert payload["n_traj"] == 128
    assert payload["per_trajectory_matrix"].shape == (128, 1, 1)
    assert payload["purity"] == 1.0
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2


def test_quadratic_moments_run_through_engine_trajectory_batches():
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.8,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=43,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {
                "quadratic_moments": QuadraticMomentAnalyzer(
                    observables={"occupation": {"matrix": [[1.0]], "center": 0.25}},
                    time_blocks=2,
                    min_block_samples=2,
                )
            },
        },
        product_graph=_graph(
            {
                "quadratic_moments": {
                    "observables": {
                        "occupation": {"matrix": [[1.0]], "center": 0.25}
                    },
                    "time_blocks": 2,
                    "min_block_samples": 2,
                }
            }
        ),
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            progress=None,
            cancellation=None,
        )
    )

    payload = _product_arrays(result.products["quadratic_moments"])
    assert "trajectories" not in result.products
    assert payload["n_traj"] == 128
    assert payload["per_trajectory_raw_moments"].shape == (128, 1, 4)
    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2


def test_typed_graph_batches_all_statistical_reducers():
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    analyser_config = {
        "coherence_matrix": {
            "modes": [0],
            "time_blocks": 2,
            "min_block_samples": 2,
        },
        "coherence_carrier": {
            "modes": [0],
            "include_trace": False,
            "minimum_lag_points": 3,
            "maximum_lag_points": 4,
        },
        "moment_statistics": {
            "modes": [0],
            "time_blocks": 2,
            "min_block_samples": 2,
        },
        "quadratic_moments": {
            "observables": {"occupation": {"matrix": [[1.0]]}},
            "modes": [0],
            "time_blocks": 2,
            "min_block_samples": 2,
        },
    }
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": analyser_config}, registry=registry
    )
    assert graph is not None
    analysers = {
        "coherence_matrix": CoherenceMatrixAnalyzer(
            modes=[0], time_blocks=2, min_block_samples=2
        ),
        "coherence_carrier": CoherenceCarrierAnalyzer(
            modes=[0],
            include_trace=False,
            minimum_lag_points=3,
            maximum_lag_points=4,
        ),
        "moment_statistics": MomentStatisticsAnalyzer(
            modes=[0], time_blocks=2, min_block_samples=2
        ),
        "quadratic_moments": QuadraticMomentAnalyzer(
            observables={"occupation": {"matrix": [[1.0]]}},
            modes=[0],
            time_blocks=2,
            min_block_samples=2,
        ),
    }
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=47,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": analysers,
        },
        product_graph=graph,
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
        )
    )

    assert set(result.products) == set(analysers)
    for product in result.products.values():
        assert int(product.handle("n_traj").materialize()) == 8


def test_batch_local_transform_fans_out_once_per_batch():
    graph, identity, psd_a, psd_b = _branched_batch_graph()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=51,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": {
                "identity": identity,
                "psd_a": psd_a,
                "psd_b": psd_b,
            },
        },
        product_graph=graph,
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
        )
    )

    assert identity.calls == 2
    assert set(result.products) == {"psd_a", "psd_b"}
    np.testing.assert_allclose(
        result.products["psd_a"].handle("psd").materialize(),
        result.products["psd_b"].handle("psd").materialize(),
    )


def test_batch_local_non_reducer_cannot_be_published():
    from qphase_sde.planning import SDEMemoryPlanningError

    graph, identity, psd_a, psd_b = _branched_batch_graph(publish_identity=True)
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=51,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": {
                "identity": identity,
                "psd_a": psd_a,
                "psd_b": psd_b,
            },
        },
        product_graph=graph,
    )

    with pytest.raises(SDEMemoryPlanningError, match="identity.*published"):
        engine.run(
            context=SimpleNamespace(
                parameter_grid=None,
                product_graph=graph,
                progress=None,
                cancellation=None,
                metadata={},
            )
        )


def test_batch_local_non_reducer_cannot_cross_final_boundary():
    from qphase_sde.planning import SDEMemoryPlanningError

    identity = IdentityBatchAnalyzer()
    final = MeanAnalyzer()
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="analyser.identity",
                producer="analyser:identity_batch",
                requirements=identity.input_requirements(),
                outputs=list(identity.output_specs(label="identity")),
            ),
            ProductNode(
                node_id="analyser.final",
                producer="analyser:mean",
                requirements=final.input_requirements(),
                outputs=list(final.output_specs(label="final")),
            ),
        ],
        edges=[
            GraphEdge(
                producer="analyser.identity",
                product="identity",
                consumer="analyser.final",
                requirement="trajectories",
            )
        ],
        external_inputs=[
            ExternalInputBinding(
                consumer="analyser.identity",
                requirement="trajectories",
                source="input",
                product="trajectories",
            )
        ],
    )
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=51,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": {"identity": identity, "final": final},
        },
        product_graph=graph,
    )

    with pytest.raises(SDEMemoryPlanningError, match="identity.*boundary"):
        engine.run(
            context=SimpleNamespace(
                parameter_grid=None,
                product_graph=graph,
                progress=None,
                cancellation=None,
                metadata={},
            )
        )


def test_scan_tiles_and_trajectory_batches_preserve_transient_graph_results():
    grid = ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": list(np.arange(32, dtype=float) + 1.0),
                }
            }
        }
    ).compile()

    def run(*, batching, memory_limit_mib=None):
        graph, identity, psd_a, psd_b, summary = _partitioned_batch_graph()
        engine = Engine(
            config=EngineConfig(
                t0=0.0,
                t1=1.0,
                dt=0.01,
                n_traj=128,
                trajectory_batching=batching,
                trajectory_batch_size=64 if batching == "required" else None,
                rng_group_size=64,
                seed=73,
                ic=[[1.0 + 0.0j]],
                keep_traj=False,
            ),
            plugins={
                "backend": NumpyBackend(),
                "integrator": EulerMaruyama(),
                "model": StochasticScannedModel(),
                "analyser": {
                    "identity": identity,
                    "psd_a": psd_a,
                    "psd_b": psd_b,
                    "summary": summary,
                },
            },
            product_graph=graph,
        )
        result = engine.run(
            context=SimpleNamespace(
                parameter_grid=grid,
                product_graph=graph,
                progress=None,
                cancellation=None,
                metadata={},
                resources=SimpleNamespace(
                    memory_limit_mib=memory_limit_mib,
                    gpu_memory_fraction=None,
                ),
            )
        )
        return result, identity.calls

    reference, reference_calls = run(batching="off")
    partitioned, partitioned_calls = run(
        batching="required", memory_limit_mib=4
    )

    plan = partitioned.metadata["execution_plan"]
    assert plan["tile_count"] >= 2
    assert plan["trajectory_batch_count"] == 2
    assert reference_calls == 1
    assert partitioned_calls == plan["tile_count"] * 2
    assert set(partitioned.products) == {"psd_b", "summary"}
    np.testing.assert_array_equal(
        partitioned.products["psd_b"].coordinate("rate"),
        np.arange(1.0, 33.0),
    )
    np.testing.assert_allclose(
        partitioned.products["psd_b"].handle("psd").materialize(),
        reference.products["psd_b"].handle("psd").materialize(),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        partitioned.products["summary"].handle("mean_power").materialize(),
        reference.products["summary"].handle("mean_power").materialize(),
        rtol=1e-12,
    )


def test_trajectory_batching_reduces_psd_before_running_peak_finder():
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    config = {
        "psd": _PSD_CONFIG,
        "peaks": {
            "scan_param": "scan",
            "finder": {"scipy": {"noise_threshold": None}},
        },
    }
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": config},
        registry=registry,
        input_bindings={"psd": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=53,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": {
                "psd": _psd_analyzer(),
                "peaks": PeaksAnalyzer(
                    PeaksAnalyzerConfig(scan_param="scan"),
                    subplugins={
                        "finder": ScipyPeakFinder(noise_threshold=None),
                    },
                ),
            },
        },
        product_graph=graph,
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
        )
    )

    assert set(result.products) == {"peaks"}
    assert result.products["peaks"].attributes["source_product"] == "psd"


def test_scanned_trajectory_batching_runs_peak_finder_after_reduction():
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    config = {
        "psd": _PSD_CONFIG,
        "peaks": {
            "scan_param": "rate",
            "finder": {"scipy": {"noise_threshold": None}},
        },
    }
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": config},
        registry=registry,
        input_bindings={"psd": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.2,
            dt=0.01,
            n_traj=8,
            trajectory_batching="required",
            trajectory_batch_size=4,
            rng_group_size=4,
            seed=59,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": {
                "psd": _psd_analyzer(),
                "peaks": PeaksAnalyzer(
                    PeaksAnalyzerConfig(scan_param="rate"),
                    subplugins={
                        "finder": ScipyPeakFinder(noise_threshold=None),
                    },
                ),
            },
        },
        product_graph=graph,
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=_grid(),
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
            resources=None,
        )
    )

    peaks = result.products["peaks"]
    assert peaks.axis("scan").size == 2
    np.testing.assert_array_equal(peaks.coordinate("rate"), [1.0, 2.0])


def test_sde_progress_uses_stable_trajectory_step_units():
    events = []
    reporter = ProgressReporter(events.append)
    model = StochasticScannedModel()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.04,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=31,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": model,
            "analyser": {"psd": _psd_analyzer()},
        },
        product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
    )

    engine.run(
        context=SimpleNamespace(
            parameter_grid=_grid(),
            progress=reporter,
            cancellation=None,
            metadata={},
        )
    )

    sampling = [
        event
        for event in events
        if event.kind == "progress" and event.stage == "sampling"
    ]
    assert sampling
    assert {event.unit for event in sampling} == {"trajectory-step"}
    assert {event.total for event in sampling} == {2 * 128 * 4}
    assert sampling[-1].completed == 2 * 128 * 4
    assert any(event.stage == "analyser.psd" for event in events)
    plan_status = next(
        event
        for event in events
        if event.kind == "status" and event.stage == "planning"
    )
    assert plan_status.importance == "normal"
    assert plan_status.metadata["execution_plan"]["trajectory_batch_count"] == 2


def _psd_engine(**config_overrides):
    """PSD-analysed engine over the deterministic scanned dummy model."""
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.08,
            dt=0.01,
            n_traj=128,
            seed=23,
            ic=[[1.0 + 0.0j]],
            keep_traj=config_overrides.pop("keep_traj", False),
            **config_overrides,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": ScannedDummyModel(),
            "analyser": {"psd": _psd_analyzer()},
        },
        product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
    )
    return engine


def test_singleton_scan_with_trajectory_batching_keeps_products():
    """P0 regression: a 1-point scan routed through trajectory batching must
    still produce every analyser product, with a size-1 scan axis.
    """

    def run(**config_overrides):
        return _psd_engine(**config_overrides).run(
            context=SimpleNamespace(
                parameter_grid=_single_point_grid(),
                progress=None,
                cancellation=None,
            )
        )

    batched = run(trajectory_batching="required", trajectory_batch_size=64)
    reference = run()  # equivalent non-batched 1-point scan (fused path)

    plan = batched.metadata["execution_plan"]
    assert plan["trajectory_batch_count"] == 2
    assert batched.shape == (1,)
    assert "dropped_products" not in batched.metadata

    product = batched.products["psd"]
    assert product.axis("scan").size == 1
    np.testing.assert_array_equal(product.coordinate("rate"), [2.0])
    np.testing.assert_array_equal(
        product.handle("uncertainty.n_independent").materialize(), [128]
    )

    # The deterministic model makes the batched and fused runs comparable:
    # values agree to accumulator floating-point merge noise.
    reference_product = reference.products["psd"]
    assert reference_product.axis("scan").size == 1
    np.testing.assert_allclose(
        product.handle("psd").materialize(),
        reference_product.handle("psd").materialize(),
        rtol=1e-12,
    )
    for variable in ("psd_std", "psd_sem"):
        np.testing.assert_allclose(
            product.handle(variable).materialize(),
            reference_product.handle(variable).materialize(),
            atol=1e-15,
        )

    point_product = batched.point_view((0,)).products["psd"]
    axis = point_product.handle("axis").materialize()
    values = point_product.handle("psd").materialize()
    assert axis.shape == values.shape[:1]
    assert values.shape == axis.shape + (1,)


def test_singleton_scan_without_batching_keeps_scan_axis():
    """A fused 1-point scan keeps a size-1 scan axis on the analyser product
    and on the retained trajectory product.
    """
    result = _psd_engine(keep_traj=True).run(
        context=SimpleNamespace(
            parameter_grid=_single_point_grid(),
            progress=None,
            cancellation=None,
        )
    )

    assert result.shape == (1,)
    assert result.products["psd"].axis("scan").size == 1
    trajectories = result.products["trajectories"]
    assert trajectories.axis("scan").size == 1
    alpha = trajectories.handle("alpha").materialize()
    assert alpha.shape[0] == 1
    assert alpha.shape[1] == 128
    point = result.point_view((0,))
    assert point.metadata["scan_point"]["rate"] == pytest.approx(2.0)
    assert point.products["trajectories"].handle("alpha").materialize().ndim == 3


def test_no_grid_run_products_stay_non_scan():
    """Without a parameter grid the analyser product carries no scan axis;
    the trajectory product keeps its single fused scan point (as before).
    """
    result = _psd_engine(keep_traj=True).run(
        context=SimpleNamespace(parameter_grid=None, progress=None)
    )

    assert result.shape == ()
    assert "scan" not in {axis.name for axis in result.products["psd"].axes}
    trajectories = result.products["trajectories"]
    assert trajectories.axis("scan").size == 1
    assert "dropped_products" not in result.metadata


def test_two_instance_labels_of_same_analyser_run_independently():
    graph = _graph(
        {
            "raw_psd": {
                "plugin": "psd",
                "config": _PSD_CONFIG,
            },
            "other_psd": {
                "plugin": "psd",
                "config": _PSD_CONFIG,
            },
        },
        signal_bindings=("raw_psd", "other_psd"),
    )
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.08,
            dt=0.01,
            n_traj=8,
            seed=67,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": ScannedDummyModel(),
            "analyser": {
                "raw_psd": _psd_analyzer(),
                "other_psd": _psd_analyzer(),
            },
        },
        product_graph=graph,
    )

    result = engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
        )
    )

    assert set(result.products) == {"raw_psd", "other_psd"}
    np.testing.assert_allclose(
        result.products["raw_psd"].handle("psd").materialize(),
        result.products["other_psd"].handle("psd").materialize(),
    )


def test_engine_fails_when_analyser_execute_builds_no_product(monkeypatch):
    """A registered analyser whose typed execution returns nothing fails."""
    analyser = _psd_analyzer()
    monkeypatch.setattr(
        analyser,
        "execute",
        lambda inputs, *, backend, label, metadata=None, context=None: {},
    )
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=0.08,
            dt=0.01,
            n_traj=16,
            seed=23,
            ic=[[1.0 + 0.0j]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": ScannedDummyModel(),
            "analyser": {"psd": analyser},
        },
        product_graph=_graph({"psd": _PSD_CONFIG}, signal_bindings=("psd",)),
    )

    with pytest.raises(ValueError, match="returned products"):
        engine.run(context=SimpleNamespace(parameter_grid=None, progress=None))


_ALLAN_SCALING_CONFIG = {
    "scan_param": "rate",
    "critical_value": 1.0,
    "side": "positive",
    "mode": 0,
    "min_scaling_points": 3,
    "bootstrap_samples": 0,
    # Relaxed white-window gates: the wiring tests exercise graph routing and
    # coordinate flow on a tiny dummy dataset, not the physics acceptance.
    "min_averaging_samples": 1,
    "min_independent_windows": 1,
    "max_relative_sem": 100.0,
    "min_local_r2": 0.0,
    "min_tau_decades": 0.0,
    "white_slope_min": -3.0,
    "white_slope_max": 1.0,
    "local_window_points": 3,
}


def _allan_chain_setup(publish_allan=False, **engine_overrides):
    """Compiled allan_variance -> allan_scaling graph plus matching instances."""
    from qphase.core.registry import discovery, registry
    from qphase_sde.analyser.allan_scaling import AllanScalingAnalyzer
    from qphase_sde.graph import compile_product_graph

    analyser_config = {
        "allan_variance": {
            "measurement": _ALLAN_MEASUREMENT,
            "points": 5,
            "min_windows": 2,
        },
        "allan_scaling": dict(_ALLAN_SCALING_CONFIG),
    }
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": analyser_config},
        registry=registry,
        publish_outputs=(
            {"allan_variance": ["allan_variance"]} if publish_allan else None
        ),
        input_bindings={"allan_variance": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    analysers = {
        "allan_variance": _allan_analyzer(),
        "allan_scaling": AllanScalingAnalyzer(
            AllanScalingAnalyzer.config_schema(**_ALLAN_SCALING_CONFIG)
        ),
    }
    config = EngineConfig(
        t0=0.0,
        t1=0.8,
        dt=0.01,
        n_traj=128,
        seed=37,
        ic=[["1.0+0.0j"]],
        keep_traj=False,
        **engine_overrides,
    )
    return graph, analysers, config


def _run_allan_chain(graph, analysers, config, grid, resources=None):
    engine = Engine(
        config=config,
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": analysers,
        },
        product_graph=graph,
    )
    return engine.run(
        context=SimpleNamespace(
            parameter_grid=grid,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
            resources=resources,
        )
    )


def test_typed_allan_chain_scan_carries_parameter_coordinates():
    """Reducer-completed frontier products must carry real scan coordinates.

    Regression probe for the Gate 5C blocker: allan_variance -> allan_scaling
    on a 4-point scan with 2 trajectory batches failed with
    "typed Allan product is missing parameter.rate" because the completed
    reducer products entered the graph executor without scan coordinates.
    """
    graph, analysers, config = _allan_chain_setup(
        publish_allan=True, trajectory_batching="required", trajectory_batch_size=64
    )

    result = _run_allan_chain(graph, analysers, config, _four_point_grid())

    assert result.metadata["execution_plan"]["trajectory_batch_count"] == 2
    assert set(result.products) == {"allan_variance", "allan_scaling"}
    allan = result.products["allan_variance"]
    assert allan.axis("scan").size == 4
    np.testing.assert_array_equal(allan.coordinate("rate"), [1.0, 2.0, 3.0, 4.0])
    scaling = _product_arrays(result.products["allan_scaling"])
    # Points below the critical value are gated out; the three supercritical
    # points feed the scaling summary.
    assert scaling["summary.accepted_points"] >= 3


def _sixteen_point_grid():
    return ScanSpec.model_validate(
        {
            "axes": {
                "rate": {
                    "target": "model.scanned_dummy.rate",
                    "values": list(np.arange(16, dtype=float) + 1.0),
                }
            }
        }
    ).compile()


def _assert_products_equal(reference, candidate):
    assert set(reference.products) == set(candidate.products)
    for name, product in reference.products.items():
        other = candidate.products[name]
        assert product.schema == other.schema, name
        for variable in product.variables:
            expected = product.handle(variable.name).materialize()
            actual = other.handle(variable.name).materialize()
            if np.dtype(expected.dtype).kind in "fc":
                np.testing.assert_allclose(actual, expected, rtol=1e-12, err_msg=name)
            else:
                np.testing.assert_array_equal(actual, expected, err_msg=name)


def test_typed_allan_chain_multi_tile_matches_single_tile_batched():
    """Tile/final split: a pointwise -> cross-scan chain is tiling-invariant.

    allan_variance executes per scan tile; allan_scaling is deferred to one
    post-merge execution on the full-scan products. The terminal products of
    a two-tile run must equal the single-tile run exactly (RNG included).
    """
    graph, _, _ = _allan_chain_setup(
        publish_allan=True, trajectory_batching="required", trajectory_batch_size=64
    )

    def run(memory_limit_mib):
        _, analysers, run_config = _allan_chain_setup(
            publish_allan=True,
            trajectory_batching="required",
            trajectory_batch_size=64,
        )
        run_config = run_config.model_copy(update={"t1": 1.0})
        return _run_allan_chain(
            graph,
            analysers,
            run_config,
            _sixteen_point_grid(),
            resources=SimpleNamespace(
                memory_limit_mib=memory_limit_mib, gpu_memory_fraction=None
            ),
        )

    single = run(12)
    multi = run(3)

    assert single.metadata["execution_plan"]["tile_count"] == 1
    multi_plan = multi.metadata["execution_plan"]
    assert multi_plan["tile_count"] > 1
    assert multi_plan["trajectory_batch_count"] == 2
    assert multi_plan["deferred_analysers"] == ["allan_scaling"]
    _assert_products_equal(single, multi)


def test_typed_allan_chain_tiling_is_invariant_unbatched():
    """The non-batched tiled branch defers cross-scan analysers identically.

    An unbatched single-tile scan runs the legacy RNG regime while tiled
    scans draw per-point seeds (planner-frozen rng_strategy), so the
    invariance pin compares two different multi-tile splittings, exactly
    like the existing tile-size RNG independence test.
    """
    graph, _, _ = _allan_chain_setup(publish_allan=True, trajectory_batching="off")
    grid = _sixteen_point_grid()

    def run(memory_limit_mib):
        _, analysers, run_config = _allan_chain_setup(
            publish_allan=True, trajectory_batching="off"
        )
        run_config = run_config.model_copy(update={"t1": 1.0})
        return _run_allan_chain(
            graph,
            analysers,
            run_config,
            grid,
            resources=SimpleNamespace(
                memory_limit_mib=memory_limit_mib, gpu_memory_fraction=None
            ),
        )

    wide = run(5)
    narrow = run(4)

    for result in (wide, narrow):
        plan = result.metadata["execution_plan"]
        assert plan["tile_count"] > 1
        assert plan["trajectory_batch_count"] == 1
        assert plan["deferred_analysers"] == ["allan_scaling"]
    assert (
        wide.metadata["execution_plan"]["scan_tile_size"]
        != (narrow.metadata["execution_plan"]["scan_tile_size"])
    )
    _assert_products_equal(wide, narrow)


def _peaks_chain_setup():
    """Compiled psd -> peaks graph plus matching instances (table terminal)."""
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    analyser_config = {
        "psd": _PSD_CONFIG,
        "peaks": {
            "scan_param": "rate",
            "finder": {"scipy": {"noise_threshold": None}},
        },
    }
    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={"analyser": analyser_config},
        registry=registry,
        input_bindings={"psd": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    analysers = {
        "psd": _psd_analyzer(),
        "peaks": PeaksAnalyzer(
            PeaksAnalyzerConfig(scan_param="rate"),
            subplugins={"finder": ScipyPeakFinder(noise_threshold=None)},
        ),
    }
    return graph, analysers


def _peaks_plan(graph, analysers, memory_limit_mib):
    from qphase_sde.planning import build_execution_plan

    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=128,
        trajectory_batching="required",
        trajectory_batch_size=64,
        seed=59,
        ic=[[1.0]],
        keep_traj=False,
    )
    return build_execution_plan(
        config=config,
        grid=_sixteen_point_grid(),
        model=StochasticScannedModel(),
        backend=NumpyBackend(),
        integrator=EulerMaruyama(),
        analysers=analysers,
        observers={},
        resources=SimpleNamespace(
            memory_limit_mib=memory_limit_mib, gpu_memory_fraction=None
        ),
        product_graph=graph,
    )


def test_typed_multi_tile_plan_rejects_table_terminal():
    """A multi-tile plan over a ragged/table terminal fails at PLAN time."""
    from qphase_sde.planning import SDEMemoryPlanningError

    graph, analysers = _peaks_chain_setup()

    with pytest.raises(
        SDEMemoryPlanningError, match=r"'peaks'.*domain-specific reducer"
    ):
        _peaks_plan(graph, analysers, 2)


def test_typed_single_tile_plan_allows_table_terminal():
    """Single-tile plans over the same table terminal still pass planning."""
    graph, analysers = _peaks_chain_setup()

    plan = _peaks_plan(graph, analysers, 6)

    assert plan.tile_count == 1


def test_typed_multi_tile_scan_rejects_table_terminal_at_plan_time():
    """The engine surfaces the table-terminal rejection before simulating."""
    from qphase_sde.planning import SDEMemoryPlanningError

    graph, analysers = _peaks_chain_setup()
    engine = Engine(
        config=EngineConfig(
            t0=0.0,
            t1=1.0,
            dt=0.01,
            n_traj=128,
            trajectory_batching="required",
            trajectory_batch_size=64,
            seed=59,
            ic=[[1.0]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": StochasticScannedModel(),
            "analyser": analysers,
        },
        product_graph=graph,
    )

    with pytest.raises(SDEMemoryPlanningError, match="domain-specific reducer"):
        engine.run(
            context=SimpleNamespace(
                parameter_grid=_sixteen_point_grid(),
                product_graph=graph,
                progress=None,
                cancellation=None,
                metadata={},
                resources=SimpleNamespace(memory_limit_mib=2, gpu_memory_fraction=None),
            )
        )
