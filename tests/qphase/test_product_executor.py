"""Tests for the resource-independent ProductGraph executor."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.core import ProductGraphExecutor
from qphase.data import (
    AxisRole,
    AxisSchema,
    CoordinateSchema,
    DataKind,
    GraphEdge,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
    ProductSchema,
    SamplingBasisSchema,
    StatisticsDataset,
    TimeSeriesDataset,
    VariableSchema,
)


def _product(name: str = "value") -> StatisticsDataset:
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="sample", role=AxisRole.INDEX, size=1)],
        variables=[
            VariableSchema(
                name=name,
                dtype="float64",
                value_domain="real",
                dims=("sample",),
            )
        ],
    )
    return StatisticsDataset.from_arrays(
        schema, {name: np.asarray([1.0])}, owner="test"
    )


def _aligned_product(
    values: list[float], *, sampling_count: int = 4
) -> StatisticsDataset:
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="scan", role=AxisRole.PARAMETER, size=len(values))],
        sampling_bases=[SamplingBasisSchema(name="trajectory", count=sampling_count)],
        variables=[
            VariableSchema(
                name="value",
                dtype="float64",
                value_domain="real",
                dims=("scan",),
            ),
            VariableSchema(
                name="parameter.x",
                dtype="float64",
                value_domain="real",
                dims=("scan",),
            ),
        ],
        coordinates=[
            CoordinateSchema(
                name="x",
                variable="parameter.x",
                dims=("scan",),
                role="parameter",
            )
        ],
    )
    array = np.asarray(values)
    return StatisticsDataset.from_arrays(
        schema,
        {"value": array, "parameter.x": array},
        owner="test",
    )


def _time_series_product() -> TimeSeriesDataset:
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=2),
            AxisSchema(name="time", role=AxisRole.COORDINATE, size=3),
        ],
        variables=[
            VariableSchema(
                name="x",
                dtype="float64",
                value_domain="real",
                dims=("trajectory", "time"),
            )
        ],
    )
    return TimeSeriesDataset.from_arrays(
        schema, {"x": np.arange(6.0).reshape(2, 3)}, owner="test"
    )


def _shared_downstream_graph(*, publish: bool) -> ProductGraph:
    """One time_series -> time_series node feeding two downstream consumers."""
    return ProductGraph(
        nodes=[
            ProductNode(
                node_id="transform",
                producer="test.transform",
                requirements=[
                    ProductRequirement(name="input", kind=DataKind.TIME_SERIES)
                ],
                outputs=[ProductDeclaration(name="series", kind=DataKind.TIME_SERIES)],
                publish_outputs=["series"] if publish else [],
            ),
            ProductNode(
                node_id="left",
                producer="test.left",
                requirements=[
                    ProductRequirement(name="series", kind=DataKind.TIME_SERIES)
                ],
                outputs=[ProductDeclaration(name="result", kind=DataKind.TIME_SERIES)],
            ),
            ProductNode(
                node_id="right",
                producer="test.right",
                requirements=[
                    ProductRequirement(name="series", kind=DataKind.TIME_SERIES)
                ],
                outputs=[ProductDeclaration(name="result", kind=DataKind.TIME_SERIES)],
            ),
        ],
        edges=[
            {
                "producer": "transform",
                "product": "series",
                "consumer": "left",
                "requirement": "series",
            },
            {
                "producer": "transform",
                "product": "series",
                "consumer": "right",
                "requirement": "series",
            },
        ],
        external_inputs=[
            {
                "consumer": "transform",
                "requirement": "input",
                "source": "external",
                "product": "raw",
            }
        ],
    )


def _run_shared_downstream(graph: ProductGraph):
    calls: list[str] = []
    received: dict[str, TimeSeriesDataset] = {}

    def transform(node, inputs):
        calls.append(node.node_id)
        return {"series": inputs["input"]}

    def consumer(node, inputs):
        calls.append(node.node_id)
        received[node.node_id] = inputs["series"]
        return {"result": _time_series_product()}

    result = ProductGraphExecutor(
        graph,
        {"transform": transform, "left": consumer, "right": consumer},
    ).run({"external": {"raw": _time_series_product()}})
    return result, calls, received


def test_executor_shares_one_time_series_product_between_consumers():
    result, calls, received = _run_shared_downstream(
        _shared_downstream_graph(publish=True)
    )

    assert calls == ["transform", "left", "right"]
    assert received["left"] is received["right"]
    assert set(result) == {"transform", "left", "right"}
    assert result["transform"]["series"] is received["left"]
    assert set(result["left"]) == {"result"}
    assert set(result["right"]) == {"result"}


def test_executor_releases_shared_product_only_after_last_consumer():
    result, calls, received = _run_shared_downstream(
        _shared_downstream_graph(publish=False)
    )

    # Both consumers still receive the live shared product, so the executor
    # must not release it before the last consumer has completed.
    assert calls == ["transform", "left", "right"]
    assert received["left"] is received["right"]
    # Once the last consumer completed, the unpublished intermediate is
    # released: only the terminal consumers appear in the outputs.
    assert set(result) == {"left", "right"}
    assert set(result["left"]) == {"result"}
    assert set(result["right"]) == {"result"}


def test_executor_resolves_external_input_and_checks_runtime_schema():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="consumer",
                producer="test.consumer",
                requirements=[
                    ProductRequirement(
                        name="input",
                        kind=DataKind.STATISTICS,
                        required_fields=["value"],
                    )
                ],
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            )
        ],
        external_inputs=[
            {
                "consumer": "consumer",
                "requirement": "input",
                "source": "external",
                "product": "value",
            }
        ],
    )

    seen: list[str] = []

    def runner(node, inputs):
        seen.append(node.node_id)
        assert inputs["input"].schema.variable("value").name == "value"
        return {"output": _product()}

    result = ProductGraphExecutor(graph, {"consumer": runner}).run(
        {"external": {"value": _product()}}
    )

    assert seen == ["consumer"]
    assert result["consumer"]["output"].kind is DataKind.STATISTICS


def test_executor_rejects_non_atomic_node_outputs():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="consumer",
                producer="test.consumer",
                requirements=[ProductRequirement(name="input")],
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            )
        ],
        external_inputs=[
            {
                "consumer": "consumer",
                "requirement": "input",
                "source": "external",
                "product": "value",
            }
        ],
    )

    with pytest.raises(ValueError, match="returned products"):
        ProductGraphExecutor(
            graph,
            {"consumer": lambda node, inputs: {}},
        ).run({"external": {"value": _product()}})


def test_executor_checks_output_quantity_and_fields():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                requirements=[],
                outputs=[
                    ProductDeclaration(
                        name="output",
                        kind=DataKind.STATISTICS,
                        quantity="required_quantity",
                        fields=["required_field"],
                    )
                ],
            )
        ]
    )
    with pytest.raises(TypeError, match="declared quantity"):
        ProductGraphExecutor(
            graph,
            {"producer": lambda node, inputs: {"output": _product()}},
        ).run({})

    def valid_quantity_product() -> StatisticsDataset:
        schema = ProductSchema(
            kind=DataKind.STATISTICS,
            axes=[AxisSchema(name="sample", role=AxisRole.INDEX, size=1)],
            variables=[
                VariableSchema(
                    name="other",
                    dtype="float64",
                    value_domain="real",
                    quantity="required_quantity",
                    dims=("sample",),
                )
            ],
        )
        return StatisticsDataset.from_arrays(
            schema, {"other": np.asarray([1.0])}, owner="test"
        )

    with pytest.raises(TypeError, match="missing declared fields"):
        ProductGraphExecutor(
            graph,
            {"producer": lambda node, inputs: {"output": valid_quantity_product()}},
        ).run({})


def test_executor_routes_internal_edges_and_releases_intermediates():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                outputs=[ProductDeclaration(name="a", kind=DataKind.STATISTICS)],
            ),
            ProductNode(
                node_id="consumer",
                producer="test.consumer",
                requirements=[ProductRequirement(name="a")],
                outputs=[ProductDeclaration(name="b", kind=DataKind.STATISTICS)],
            ),
        ],
        edges=[
            {
                "producer": "producer",
                "product": "a",
                "consumer": "consumer",
                "requirement": "a",
            }
        ],
    )
    seen: list[str] = []

    def producer(node, inputs):
        del node, inputs
        return {"a": _product("a")}

    def consumer(node, inputs):
        del node
        seen.append(inputs["a"].variable("a").name)
        return {"b": _product("b")}

    result = ProductGraphExecutor(
        graph, {"producer": producer, "consumer": consumer}
    ).run({})

    assert seen == ["a"]
    assert set(result) == {"consumer"}
    assert set(result["consumer"]) == {"b"}


def test_executor_reports_nodes_and_checks_cancellation():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            )
        ]
    )

    class Token:
        def __init__(self):
            self.calls = 0

        def raise_if_cancelled(self):
            self.calls += 1

    class Progress:
        def __init__(self):
            self.messages = []

        def status(self, message, *, stage):
            self.messages.append((message, stage))

    token = Token()
    progress = Progress()
    context = type("Context", (), {"cancellation": token, "progress": progress})()
    ProductGraphExecutor(
        graph,
        {"producer": lambda node, inputs: {"output": _product()}},
    ).run({}, context=context)

    assert token.calls == 1
    assert [item[1] for item in progress.messages] == [
        "producer",
        "producer",
    ]


def test_executor_reports_output_validation_failure():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                outputs=[
                    ProductDeclaration(
                        name="output", kind=DataKind.STATISTICS, fields=["missing"]
                    )
                ],
            )
        ]
    )

    class Progress:
        def __init__(self):
            self.messages = []

        def status(self, message, *, stage):
            self.messages.append((message, stage))

    progress = Progress()
    context = type("Context", (), {"progress": progress})()
    with pytest.raises(TypeError, match="missing declared fields"):
        ProductGraphExecutor(
            graph,
            {"producer": lambda node, inputs: {"output": _product()}},
        ).run({}, context=context)
    assert [item[1] for item in progress.messages] == ["producer", "producer"]


def test_executor_reports_non_mapping_output_failure():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            )
        ]
    )

    class Progress:
        def __init__(self):
            self.messages = []

        def status(self, message, *, stage):
            self.messages.append((message, stage))

    progress = Progress()
    context = type("Context", (), {"progress": progress})()
    with pytest.raises(TypeError, match="product mapping"):
        ProductGraphExecutor(
            graph,
            {"producer": lambda node, inputs: []},
        ).run({}, context=context)
    assert [item[1] for item in progress.messages] == ["producer", "producer"]


@pytest.mark.parametrize(
    ("left", "other", "message"),
    [
        (
            _aligned_product([1.0, 2.0]),
            _aligned_product([1.0, 3.0]),
            "disagree on coordinate",
        ),
        (
            _aligned_product([1.0, 2.0]),
            _aligned_product([1.0, 2.0], sampling_count=8),
            "sampling bases",
        ),
        (
            _aligned_product([1.0, 2.0]),
            _aligned_product([1.0, 1.0]),
            "duplicate values in coordinate 'x' of input 'right'",
        ),
        (
            _aligned_product([1.0, 2.0]),
            _aligned_product([np.nan, np.nan]),
            "duplicate values in coordinate 'x' of input 'right'",
        ),
        (
            _aligned_product([1.0, 1.0]),
            _aligned_product([1.0, 2.0]),
            "duplicate values in coordinate 'x' of input 'left'",
        ),
        (
            _aligned_product([1.0, 1.0]),
            _aligned_product([1.0, 1.0]),
            "duplicate values in coordinate 'x' of input 'right'",
        ),
    ],
)
def test_executor_enforces_align_with(left, other, message):
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="consumer",
                producer="test.consumer",
                requirements=[
                    ProductRequirement(name="left", kind=DataKind.STATISTICS),
                    ProductRequirement(
                        name="right",
                        kind=DataKind.STATISTICS,
                        align_with="left",
                    ),
                ],
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            )
        ],
        external_inputs=[
            {
                "consumer": "consumer",
                "requirement": "left",
                "source": "external",
                "product": "left",
            },
            {
                "consumer": "consumer",
                "requirement": "right",
                "source": "external",
                "product": "right",
            },
        ],
    )
    with pytest.raises(ValueError, match=message):
        ProductGraphExecutor(
            graph,
            {"consumer": lambda node, inputs: {"output": _product()}},
        ).run(
            {
                "external": {
                    "left": left,
                    "right": other,
                }
            }
        )


def test_executor_routes_completed_products_through_remaining_graph():
    graph = ProductGraph(
        nodes=[
            ProductNode(
                node_id="producer",
                producer="test.producer",
                outputs=[ProductDeclaration(name="value", kind=DataKind.STATISTICS)],
            ),
            ProductNode(
                node_id="consumer",
                producer="test.consumer",
                requirements=[
                    ProductRequirement(name="value", kind=DataKind.STATISTICS)
                ],
                outputs=[ProductDeclaration(name="output", kind=DataKind.STATISTICS)],
            ),
        ],
        edges=[
            GraphEdge(
                producer="producer",
                product="value",
                consumer="consumer",
                requirement="value",
            )
        ],
    )
    value = _product()
    seen = []

    def producer(node, inputs):
        del node, inputs
        raise AssertionError("a completed producer must not run")

    def consumer(node, inputs):
        del node
        seen.append(inputs["value"])
        return {"output": inputs["value"]}

    result = ProductGraphExecutor(
        graph, {"producer": producer, "consumer": consumer}
    ).run({}, completed_nodes={"producer": {"value": value}})

    assert seen == [value]
    assert result == {"consumer": {"output": value}}
