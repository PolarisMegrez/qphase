"""Analyser-config expansion: nested ridge configs become sibling graph nodes."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from qphase.backend.numpy_backend import NumpyBackend
from qphase.core.registry import discovery, registry
from qphase_sde.analyser_config import RIDGE_LABEL_SUFFIX, expand_analyser_config
from qphase_sde.engine import EngineConfig
from qphase_sde.graph import compile_product_graph
from qphase_sde.integrator.euler_maruyama import EulerMaruyama
from qphase_sde.planning import build_execution_plan

pytestmark = pytest.mark.integration

RIDGE_LABEL = f"carrier{RIDGE_LABEL_SUFFIX}"


def _carrier_config(**overrides):
    config = {
        "scan_param": "rate",
        "psd_key": "psd",
        "readout": 0,
        "maximum_lag": 80.0,
        "center": {
            "spectral_ridge": {
                "scan_param": "placeholder",
                "readouts": [0],
                "freq_min": -1.2,
                "freq_max": 1.2,
            }
        },
    }
    config.update(overrides)
    return config


def _analyser_section(**overrides):
    return {
        "carrier": {
            "plugin": "band_limited_carrier",
            "config": _carrier_config(**overrides),
        }
    }


def test_expander_promotes_the_nested_ridge_config():
    """The nested ridge config becomes a sibling analyser with resolved knobs."""
    raw = _analyser_section(freq_min=-0.5, freq_max=0.7)

    expanded = expand_analyser_config(raw)

    assert set(expanded) == {"carrier", RIDGE_LABEL}
    ridge = expanded[RIDGE_LABEL]
    assert ridge["plugin"] == "spectral_ridge"
    # The carrier drives the ridge over its own scan axis, PSD and readout;
    # its explicit frequency bounds win over the nested ones.
    assert ridge["config"]["scan_param"] == "rate"
    assert ridge["config"]["psd_key"] == "psd"
    assert ridge["config"]["readouts"] == [0]
    assert ridge["config"]["freq_min"] == -0.5
    assert ridge["config"]["freq_max"] == 0.7
    # The carrier keeps only the label of the analyser it now consumes.
    carrier_config = expanded["carrier"]["config"]
    assert carrier_config["center"] == {"ridge_label": RIDGE_LABEL}
    assert "spectral_ridge" not in carrier_config["center"]


def test_expander_is_idempotent_and_does_not_mutate_its_input():
    """A second expansion is a no-op and the caller's mapping stays intact."""
    raw = _analyser_section()

    once = expand_analyser_config(raw)
    twice = expand_analyser_config(once)

    assert once == twice
    assert "spectral_ridge" in raw["carrier"]["config"]["center"]


def test_expander_leaves_plain_carriers_and_other_analysers_alone():
    """A carrier without a ridge-conditioned center is not rewritten."""
    section = {
        "carrier": {
            "plugin": "band_limited_carrier",
            "config": {"scan_param": "rate", "readout": "trace"},
        },
        "peaks": {"plugin": "peaks", "config": {"scan_param": "rate"}},
    }

    assert expand_analyser_config(section) == section


def test_expanded_graph_carries_a_transient_ridge_node():
    """Both ridge products are consumed, so the node stays graph-local."""
    discovery.discover_plugins()
    expanded = expand_analyser_config(_analyser_section())

    graph = compile_product_graph(
        plugin_config={"analyser": expanded}, registry=registry
    )

    assert graph is not None
    node_ids = {node.node_id for node in graph.nodes}
    assert f"analyser.{RIDGE_LABEL}" in node_ids
    ridge_node = next(
        node for node in graph.nodes if node.node_id.endswith(RIDGE_LABEL)
    )
    assert ridge_node.publish_outputs == []
    # Nothing the synthetic node produces is terminal: the executor would
    # otherwise publish (and persist) it.
    assert graph.terminal_outputs()[ridge_node.node_id] == set()
    consumed = {
        (edge.producer, edge.product, edge.requirement)
        for edge in graph.edges
        if edge.producer == ridge_node.node_id
    }
    assert consumed == {
        (ridge_node.node_id, RIDGE_LABEL, RIDGE_LABEL),
        (
            ridge_node.node_id,
            f"{RIDGE_LABEL}.candidates",
            f"{RIDGE_LABEL}.candidates",
        ),
    }


def test_multi_tile_plan_defers_the_expanded_ridge_node():
    """A tiled carrier job plans the synthesized node as a deferred analyser."""
    from tests.qphase_sde.runtime.test_scan_runtime import (
        ScannedDummyModel,
        _sixteen_point_grid,
    )

    discovery.discover_plugins()
    expanded = expand_analyser_config(_analyser_section())
    analysers = {
        label: registry.create_plugin_instance(
            "analyser", {"name": spec["plugin"], **spec["config"]}
        )
        for label, spec in expanded.items()
    }
    graph = compile_product_graph(
        plugin_config={"analyser": expanded}, registry=registry
    )
    assert graph is not None
    config = EngineConfig(
        t0=0.0,
        t1=1.0,
        dt=0.01,
        n_traj=128,
        trajectory_batching="off",
        seed=59,
        ic=[[1.0]],
        keep_traj=False,
    )

    plan = build_execution_plan(
        config=config,
        grid=_sixteen_point_grid(),
        model=ScannedDummyModel(),
        backend=NumpyBackend(),
        integrator=EulerMaruyama(),
        analysers=analysers,
        observers={},
        resources=SimpleNamespace(memory_limit_mib=2, gpu_memory_fraction=None),
        product_graph=graph,
    )

    assert plan.tile_count > 1
    # The synthesized analyser is planned like any other one: counted, not hidden.
    assert RIDGE_LABEL in plan.deferred_analysers
    assert "carrier" in plan.deferred_analysers
