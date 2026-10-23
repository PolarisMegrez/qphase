"""qphase_sde: engine-level ProductGraph tests for observable_transform.

The transform runs as a batch-local transient node in front of a reducer
frontier. These tests drive the real engine runtime (normal and
trajectory-batched paths) so the source provenance comes from the frozen
ProductGraph, never from hand-injected metadata.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from qphase.backend.numpy_backend import NumpyBackend
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
    StatisticsDataset,
    VariableSchema,
)
from qphase_sde.analyser.base import Analyser, requirement_source
from qphase_sde.analyser.observable_transform import ObservableTransformAnalyser
from qphase_sde.contracts.analyser import AnalyserExecutionCapabilities
from qphase_sde.contracts.quantities import SDEQuantity
from qphase_sde.engine import Engine, EngineConfig
from qphase_sde.integrator.euler_maruyama import EulerMaruyama

T0 = 0.0
DT = 0.01
T1 = 0.2


class _DummyModel:
    name = "observable_transform_dummy"
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
        return np.ones(y.shape + (1,))


def _engine(transform, probe=None, *, graph, batching="off", batch_size=None):
    analysers = {"obs": transform}
    if probe is not None:
        analysers["probe"] = probe
    return Engine(
        config=EngineConfig(
            t0=T0,
            t1=T1,
            dt=DT,
            n_traj=8,
            trajectory_batching=batching,
            trajectory_batch_size=batch_size,
            rng_group_size=4,
            seed=17,
            ic=[["1.0+0.0j"]],
            keep_traj=False,
        ),
        plugins={
            "backend": NumpyBackend(),
            "integrator": EulerMaruyama(),
            "model": _DummyModel(),
            "analyser": analysers,
        },
        product_graph=graph,
    )


def _run(engine, graph):
    return engine.run(
        context=SimpleNamespace(
            parameter_grid=None,
            product_graph=graph,
            progress=None,
            cancellation=None,
            metadata={},
        )
    )


def _compiled_graph(publish_outputs=None):
    from qphase.core.registry import discovery, registry
    from qphase_sde.graph import compile_product_graph

    discovery.discover_plugins()
    graph = compile_product_graph(
        plugin_config={
            "analyser": {
                "obs": {
                    "plugin": "observable_transform",
                    "config": {
                        "units": "occupation",
                        "observables": {"n": {"matrix": [[1.0]]}},
                    },
                }
            }
        },
        registry=registry,
        publish_outputs=publish_outputs,
        input_bindings={"obs": {"signal": "input.trajectories"}},
    )
    assert graph is not None
    return graph


def test_normal_graph_source_provenance_from_frozen_graph():
    """Normal execution records the real external binding, not an injection."""
    graph = _compiled_graph(publish_outputs={"obs": ["obs"]})
    transform = ObservableTransformAnalyser(
        units="occupation", observables={"n": {"matrix": [[1.0]]}}
    )
    result = _run(_engine(transform, graph=graph), graph)

    product = result.products["obs"]
    assert product.attributes["source_product"] == "input.trajectories"
    assert product.attributes["producer_analyser"] == "obs"
    assert product.provenance["source_product"] == "input.trajectories"
    assert product.provenance["producer_analyser"] == "obs"
    value = product.handle("value").materialize()
    assert value.shape == (8, 21, 1)
    assert np.all(np.isfinite(value))


class _CountingTransform(ObservableTransformAnalyser):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.calls = 0

    def execute(self, inputs, **kwargs):
        self.calls += 1
        return super().execute(inputs, **kwargs)


class _ScalarProbeAccumulator:
    """Per-point fold of batch probe outputs; checkpoint-friendly."""

    def __init__(self):
        self.payloads: list[np.ndarray] = []

    def consume_product(self, product):
        self.payloads.append(
            np.asarray(product.handle("trajectory_mean").materialize())
        )

    def merge(self, other):
        self.payloads.extend(other.payloads)

    def finalize_product(self, label):
        data = np.concatenate(self.payloads, axis=0)
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            axes=[
                AxisSchema(
                    name="trajectory", role=AxisRole.REALIZATION, size=data.shape[0]
                ),
                AxisSchema(
                    name="channel", role=AxisRole.COMPONENT, size=data.shape[1]
                ),
            ],
            variables=[
                VariableSchema(
                    name="trajectory_mean",
                    dtype="<f8",
                    value_domain="real",
                    dims=("trajectory", "channel"),
                )
            ],
            attributes={"graph_ready": True},
        )
        return {
            label: StatisticsDataset.from_arrays(
                schema, {"trajectory_mean": data}, owner="test.scalar_probe"
            )
        }

    def checkpoint(self):
        return {"payloads": [payload.tolist() for payload in self.payloads]}

    def restore(self, checkpoint):
        self.payloads = [np.asarray(payload) for payload in checkpoint["payloads"]]


class _ScalarProbe(Analyser):
    """Test-only reducer frontier consuming sampled-observable series."""

    name = "scalar_probe"

    def __init__(self):
        self.seen_source_products: list[str] = []
        self.seen_requirement_sources: list[str] = []

    def input_requirements(self):
        return [
            ProductRequirement(
                name="observable",
                kind=DataKind.TIME_SERIES,
                quantity=SDEQuantity.SAMPLED_OBSERVABLE.value,
                value_domain="real",
                required_fields=["value", "valid_length"],
            )
        ]

    def output_specs(self, *, label):
        return (
            ProductDeclaration(
                name=label, kind=DataKind.STATISTICS, fields=["trajectory_mean"]
            ),
        )

    def execution_capabilities(self):
        return AnalyserExecutionCapabilities(supports_trajectory_batching=True)

    def create_result_accumulator(self):
        return _ScalarProbeAccumulator()

    def execute(self, inputs, *, backend, label, metadata=None, context=None):
        del backend, context
        product = inputs["observable"]
        self.seen_source_products.append(
            str(product.attributes.get("source_product", ""))
        )
        self.seen_requirement_sources.append(
            requirement_source(metadata, "observable") or ""
        )
        value = np.asarray(product.handle("value").materialize())
        scan_size, n_traj, _, n_channel = value.shape
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            axes=[
                AxisSchema(name="scan", role=AxisRole.PARAMETER, size=scan_size),
                AxisSchema(
                    name="trajectory", role=AxisRole.REALIZATION, size=n_traj
                ),
                AxisSchema(
                    name="channel", role=AxisRole.COMPONENT, size=n_channel
                ),
            ],
            variables=[
                VariableSchema(
                    name="trajectory_mean",
                    dtype="<f8",
                    value_domain="real",
                    dims=("scan", "trajectory", "channel"),
                )
            ],
            attributes={"graph_ready": True},
        )
        return {
            label: StatisticsDataset.from_arrays(
                schema,
                {"trajectory_mean": value.mean(axis=2)},
                owner="test.scalar_probe",
            )
        }


def _probe_graph(transform, probe, *, publish_transform=False):
    return ProductGraph(
        nodes=[
            ProductNode(
                node_id="analyser.obs",
                producer="analyser:observable_transform",
                requirements=transform.input_requirements(),
                outputs=list(transform.output_specs(label="obs")),
                publish_outputs=["obs"] if publish_transform else [],
            ),
            ProductNode(
                node_id="analyser.probe",
                producer="analyser:scalar_probe",
                requirements=probe.input_requirements(),
                outputs=list(probe.output_specs(label="probe")),
            ),
        ],
        edges=[
            GraphEdge(
                producer="analyser.obs",
                product="obs",
                consumer="analyser.probe",
                requirement="observable",
            )
        ],
        external_inputs=[
            ExternalInputBinding(
                consumer="analyser.obs",
                requirement="signal",
                source="input",
                product="trajectories",
            )
        ],
    )


def test_batched_graph_transform_runs_once_per_batch_and_stays_transient():
    transform = _CountingTransform(
        units="occupation", observables={"n": {"matrix": [[1.0]]}}
    )
    probe = _ScalarProbe()
    graph = _probe_graph(transform, probe)
    engine = _engine(
        transform, probe, graph=graph, batching="required", batch_size=4
    )
    result = _run(engine, graph)

    # 8 trajectories in batches of 4: the transform executes exactly once
    # per physical batch and its output never enters the terminal bundle.
    assert transform.calls == 2
    assert set(result.products) == {"probe"}
    # Provenance through the real batch runner: the transform saw the
    # external binding; the probe saw the internal edge identity.
    assert probe.seen_source_products == ["input.trajectories"] * 2
    assert probe.seen_requirement_sources == ["obs"] * 2

    reduced = result.products["probe"].handle("trajectory_mean").materialize()
    # Reference: the same seed without batching, transform output published
    # through the compiled-graph path (transform-only node set).
    reference_transform = ObservableTransformAnalyser(
        units="occupation", observables={"n": {"matrix": [[1.0]]}}
    )
    reference_graph = _compiled_graph(publish_outputs={"obs": ["obs"]})
    reference = _run(
        _engine(reference_transform, graph=reference_graph), reference_graph
    )
    published = reference.products["obs"].handle("value").materialize()
    np.testing.assert_allclose(
        reduced,
        published.mean(axis=1),
        rtol=1e-12,
        atol=1e-14,
    )
