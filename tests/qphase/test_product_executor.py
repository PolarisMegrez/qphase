"""Tests for the resource-independent ProductGraph executor."""

from __future__ import annotations

import numpy as np
import pytest
from qphase.core import ProductGraphExecutor
from qphase.data import (
    AxisRole,
    AxisSchema,
    DataKind,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
    ProductSchema,
    StatisticsDataset,
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
