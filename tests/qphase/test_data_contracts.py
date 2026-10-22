"""Contract tests for the approved Phase 0 data product schema protocols."""

import json
import math

import pytest
from pydantic import ValidationError
from qphase.core.task_profile import (
    EngineTaskProfile,
    InputProductRequirement,
    OutputProductDeclaration,
    PluginRequirementSet,
)
from qphase.data import (
    PRODUCT_SCHEMA_VERSION,
    ArtifactRef,
    AxisRole,
    AxisSchema,
    DataHandleProtocol,
    DataKind,
    DataLeaseProtocol,
    ExternalInputBinding,
    GraphEdge,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
    ProductSchema,
    RuntimeProductBacking,
    SamplingBasisSchema,
    SpectralQuantity,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
    requirement_matches,
    validate_backing,
)


def _time_series_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(
                name="time",
                role=AxisRole.COORDINATE,
                size=1024,
                coordinate="regular",
                start=0.0,
                step=0.01,
                units="s",
            ),
            AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=64),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=2),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype="complex128",
                value_domain="complex",
                dims=("trajectory", "time", "channel"),
                quantity="field_amplitude",
            )
        ],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", source_axis="trajectory")
        ],
        uncertainties=[
            UncertaintySchema(
                target="alpha",
                kind="sample_std",
                sampling_basis="trajectory",
                covariance="real_imag",
                count=64,
            )
        ],
    )


def _spectral_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER),
            AxisSchema(
                name="frequency",
                role=AxisRole.COORDINATE,
                size=513,
                units="Hz",
            ),
            AxisSchema(name="channel", role=AxisRole.COMPONENT, size=2),
        ],
        sampling_bases=[SamplingBasisSchema(name="trajectory", count=64)],
        variables=[
            VariableSchema(
                name="power",
                dtype="float64",
                value_domain="real",
                dims=("scan", "frequency", "channel"),
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                constraints=VariableConstraints(nonnegative=True),
            )
        ],
        uncertainties=[
            UncertaintySchema(
                target="power",
                kind="sample_std",
                sampling_basis="trajectory",
                covariance="real",
            )
        ],
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "one_sided",
            "normalization": "density",
            "window": "hann",
            "estimator": "periodogram",
            "effective_degrees_of_freedom": 128.0,
        },
    )


def test_product_schema_json_roundtrip_and_stable_fingerprint():
    """Schemas round-trip through JSON with a stable fingerprint."""
    schema = _time_series_schema()
    payload = json.loads(json.dumps(schema.model_dump(mode="json")))
    reparsed = ProductSchema.model_validate(payload)
    assert reparsed == schema
    assert reparsed.fingerprint() == schema.fingerprint()
    assert len(schema.fingerprint()) == 64


def test_product_schema_fingerprint_tracks_mutation():
    """Fingerprinting reflects mutable schema metadata instead of caching it."""
    schema = _time_series_schema()
    before = schema.fingerprint()
    schema.attributes["revision"] = "changed"

    assert schema.fingerprint() != before


def test_schema_fingerprint_golden():
    """Freeze canonical serialization with a golden digest."""
    assert _spectral_schema().fingerprint() == GOLDEN_SPECTRAL_FINGERPRINT


def test_schema_is_extra_forbid():
    """Unknown fields are rejected at every level."""
    with pytest.raises(ValidationError):
        ProductSchema.model_validate({"kind": "spectral", "variables": [], "bogus": 1})
    with pytest.raises(ValidationError):
        AxisSchema.model_validate({"name": "t", "bogus": 1})


def test_variable_dtype_validation():
    """Object dtypes are forbidden; dtype names are normalized."""
    with pytest.raises(ValidationError, match="object dtype"):
        VariableSchema(name="bad", dtype="object", value_domain="real", dims=())
    variable = VariableSchema(name="x", dtype="float64", value_domain="real", dims=())
    assert variable.dtype == "<f8"


def test_variable_domain_dtype_bidirectional():
    """Real forbids complex dtypes; complex requires complex dtypes."""
    with pytest.raises(ValidationError, match="value_domain"):
        VariableSchema(name="x", dtype="complex128", value_domain="real", dims=())
    with pytest.raises(ValidationError, match="value_domain"):
        VariableSchema(name="x", dtype="float64", value_domain="complex", dims=())


def test_nonnegative_requires_real_numeric():
    """Nonnegative constraints only apply to real numeric variables."""
    with pytest.raises(ValidationError, match="nonnegative"):
        VariableSchema(
            name="z",
            dtype="complex128",
            value_domain="complex",
            dims=(),
            constraints=VariableConstraints(nonnegative=True),
        )
    ok = VariableSchema(
        name="n",
        dtype="int64",
        value_domain="real",
        dims=(),
        constraints=VariableConstraints(nonnegative=True),
    )
    assert ok.constraints.nonnegative


def test_variable_dims_unique_and_known():
    """Variables reference declared axes exactly once each."""
    with pytest.raises(ValidationError, match="unknown axes"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[AxisSchema(name="time", size=8)],
            variables=[
                VariableSchema(
                    name="x",
                    dtype="float64",
                    value_domain="real",
                    dims=("time", "ghost"),
                )
            ],
        )
    with pytest.raises(ValidationError, match="repeat"):
        VariableSchema(
            name="x",
            dtype="float64",
            value_domain="real",
            dims=("time", "time"),
        )


def test_hermitian_requires_two_tensor_dims():
    """A Hermitian layout needs at least two component/index dims."""
    with pytest.raises(ValidationError, match="Hermitian"):
        ProductSchema(
            kind=DataKind.SPECTRAL,
            axes=[AxisSchema(name="frequency", role=AxisRole.COORDINATE)],
            variables=[
                VariableSchema(
                    name="cross",
                    dtype="complex128",
                    value_domain="complex",
                    dims=("frequency",),
                    constraints=VariableConstraints(symmetry="hermitian"),
                )
            ],
            attributes={
                "frequency_units": "Hz",
                "orientation": "phase_decreasing",
                "sidedness": "two_sided",
                "normalization": "density",
                "window": "rectangular",
                "estimator": "periodogram",
            },
        )
    ok = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="frequency", role=AxisRole.COORDINATE),
            AxisSchema(name="channel_i", role=AxisRole.COMPONENT),
            AxisSchema(name="channel_j", role=AxisRole.COMPONENT),
        ],
        variables=[
            VariableSchema(
                name="cross",
                dtype="complex128",
                value_domain="complex",
                dims=("frequency", "channel_i", "channel_j"),
                constraints=VariableConstraints(symmetry="hermitian"),
            )
        ],
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "density",
            "window": "rectangular",
            "estimator": "periodogram",
        },
    )
    assert ok.variable("cross").constraints.symmetry == "hermitian"


def test_complex_uncertainty_requires_covariance_representation():
    """Complex targets must use real_imag or magnitude_phase, never a bare std."""
    base = dict(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="time", role=AxisRole.COORDINATE, size=8),
        ],
        variables=[
            VariableSchema(
                name="a",
                dtype="complex128",
                value_domain="complex",
                dims=("time",),
            )
        ],
    )
    with pytest.raises(ValidationError, match="covariance representation"):
        ProductSchema(
            **base,
            uncertainties=[UncertaintySchema(target="a", kind="sample_std")],
        )
    with pytest.raises(ValidationError, match="covariance representation"):
        ProductSchema(
            **base,
            uncertainties=[
                UncertaintySchema(target="a", kind="sample_std", covariance="real")
            ],
        )


def test_real_uncertainty_uses_real_covariance():
    """Real targets may only use the 'real' covariance representation."""
    with pytest.raises(ValidationError, match="'real' covariance"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(name="m1", dtype="float64", value_domain="real", dims=())
            ],
            uncertainties=[
                UncertaintySchema(
                    target="m1", kind="covariance", covariance="real_imag"
                )
            ],
        )


def test_custom_covariance_representation_not_frozen():
    """The 'custom' covariance representation is not part of the contract."""
    with pytest.raises(ValidationError):
        UncertaintySchema(target="x", kind="covariance", covariance="custom")


def test_uncertainty_sampling_basis_separates_retained_and_reduced_axes():
    """Sampling uncertainty references a basis, not a discarded payload axis."""
    axes = [AxisSchema(name="time", role=AxisRole.COORDINATE)]
    variable = VariableSchema(
        name="x", dtype="float64", value_domain="real", dims=("time",)
    )
    with pytest.raises(ValidationError, match="unknown sampling basis"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            variables=[variable],
            uncertainties=[
                UncertaintySchema(target="x", kind="sem", sampling_basis="trajectory")
            ],
        )
    with pytest.raises(ValidationError, match="must reference a sampling basis"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=axes,
            variables=[variable],
            uncertainties=[UncertaintySchema(target="x", kind="sem", scope="sampling")],
        )
    ok = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=axes,
        variables=[variable],
        sampling_bases=[SamplingBasisSchema(name="trajectory", count=32)],
        uncertainties=[
            UncertaintySchema(
                target="x",
                kind="sem",
                sampling_basis="trajectory",
                scope="sampling",
                count=32,
            )
        ],
    )
    assert ok.uncertainties[0].sampling_basis == "trajectory"


def test_sampling_basis_source_and_count_contracts():
    """Sampling bases use a retained realization or a concrete sample count."""
    with pytest.raises(ValidationError, match="either count or count_variable"):
        SamplingBasisSchema(name="trajectory", count=4, count_variable="n")

    with pytest.raises(ValidationError, match="reduced realizations"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[AxisSchema(name="trajectory", role=AxisRole.REALIZATION, size=4)],
            variables=[
                VariableSchema(
                    name="mean", dtype="float64", value_domain="real", dims=()
                )
            ],
        )

    with pytest.raises(ValidationError, match="not a realization axis"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            axes=[AxisSchema(name="scan", role=AxisRole.PARAMETER, size=2)],
            sampling_bases=[SamplingBasisSchema(name="trajectory", source_axis="scan")],
            variables=[
                VariableSchema(
                    name="mean", dtype="float64", value_domain="real", dims=("scan",)
                )
            ],
        )

    reduced = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="scan", role=AxisRole.PARAMETER, size=2)],
        sampling_bases=[
            SamplingBasisSchema(name="trajectory", count_variable="independent_count")
        ],
        variables=[
            VariableSchema(
                name="mean", dtype="float64", value_domain="real", dims=("scan",)
            ),
            VariableSchema(
                name="independent_count",
                dtype="int64",
                value_domain="real",
                dims=("scan",),
            ),
        ],
    )
    assert reduced.is_closed


def test_uncertainty_confidence_and_count_bounds():
    """Confidence must lie in (0, 1); count must be a positive integer."""
    with pytest.raises(ValidationError, match="confidence"):
        UncertaintySchema(target="x", kind="confidence_interval", confidence=1.5)
    with pytest.raises(ValidationError, match="confidence"):
        UncertaintySchema(target="x", kind="confidence_interval", confidence=0.0)
    with pytest.raises(ValidationError, match="positive"):
        UncertaintySchema(target="x", kind="sem", count=0)


def test_uncertainty_data_variable_must_be_typed_variable():
    """Covariance payloads reference typed variables, not metadata dicts."""
    with pytest.raises(ValidationError, match="data variable"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(name="m1", dtype="float64", value_domain="real", dims=())
            ],
            uncertainties=[
                UncertaintySchema(
                    target="m1",
                    kind="covariance",
                    covariance="real",
                    data_variable="ghost",
                )
            ],
        )


def test_uncertainty_target_must_exist():
    """Uncertainties must reference declared variables."""
    with pytest.raises(ValidationError, match="unknown variable"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(name="m1", dtype="float64", value_domain="real", dims=())
            ],
            uncertainties=[UncertaintySchema(target="nope", kind="sem")],
        )


def test_axis_shape_closure():
    """Schemas may be partially open at plan time and closed before use."""
    open_schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time")],
        variables=[
            VariableSchema(
                name="x", dtype="float64", value_domain="real", dims=("time",)
            )
        ],
    )
    assert not open_schema.is_closed
    closed = open_schema.model_copy(update={"axes": [AxisSchema(name="time", size=4)]})
    assert closed.is_closed


def test_regular_axis_closure_requires_finite_nonzero_step():
    """Regular coordinates are closed only with a finite, non-zero step."""
    axis = AxisSchema(name="time", coordinate="regular", size=8)
    assert not axis.is_closed
    assert not axis.model_copy(update={"step": 0.0}).is_closed
    assert not axis.model_copy(update={"step": float("inf")}).is_closed
    assert axis.model_copy(update={"step": 0.1}).is_closed


def test_axis_size_must_be_nonnegative():
    """Axis sizes are non-negative integers."""
    with pytest.raises(ValidationError):
        AxisSchema(name="time", size=-1)


def test_spectral_products_require_spectral_attributes():
    """Spectral products must carry the mandatory attribute set."""
    with pytest.raises(ValidationError):
        ProductSchema(
            kind=DataKind.SPECTRAL,
            axes=[AxisSchema(name="frequency", size=4)],
            variables=[
                VariableSchema(
                    name="power",
                    dtype="float64",
                    value_domain="real",
                    dims=("frequency",),
                )
            ],
            attributes={"estimator": "periodogram"},
        )


def test_spectral_attributes_reject_empty_required_fields():
    """Mandatory spectral attributes must not bypass validation via ''."""
    with pytest.raises(ValidationError):
        ProductSchema(
            kind=DataKind.SPECTRAL,
            axes=[AxisSchema(name="frequency", size=4)],
            variables=[
                VariableSchema(
                    name="power",
                    dtype="float64",
                    value_domain="real",
                    dims=("frequency",),
                )
            ],
            attributes={
                "frequency_units": "",
                "orientation": "phase_decreasing",
                "sidedness": "one_sided",
                "normalization": "density",
                "window": "rectangular",
                "estimator": "periodogram",
            },
        )


def test_moment_family_removed_from_core():
    """Moment families are SDE-private; the core schema has no such field."""
    import qphase.data

    assert not hasattr(qphase.data, "MomentFamilySchema")
    assert "moment_family" not in ProductSchema.model_fields
    with pytest.raises(ValidationError):
        ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(name="m", dtype="float64", value_domain="real", dims=())
            ],
            moment_family={"family_id": "x"},
        )


def test_artifact_ref_is_minimal_and_typed():
    """Artifact refs carry identity only: no provenance, arrays or extras."""
    schema = _spectral_schema()
    ref = ArtifactRef(
        artifact_id="art-123",
        product_name="psd",
        product_schema=schema,
        storage_adapter="npz/3",
    )
    payload = json.loads(json.dumps(ref.model_dump(mode="json")))
    assert ArtifactRef.model_validate(payload) == ref

    with pytest.raises(ValidationError):
        ArtifactRef(
            artifact_id="a",
            product_name="psd",
            product_schema=schema,
            storage_adapter="npz/3",
            provenance={"engine": "sde"},
        )
    with pytest.raises(ValidationError, match="storage_adapter"):
        ArtifactRef(
            artifact_id="a",
            product_name="psd",
            product_schema=schema,
            # Adapter ids are 'name/version' registry keys, not dotted paths.
            storage_adapter="fixture.fictional_adapter:load",
        )
    with pytest.raises(ValidationError, match="extra"):
        ArtifactRef(
            artifact_id="a",
            product_name="psd",
            product_schema=schema,
            storage_adapter="npz/3",
            content_hash="removed",
        )


def _node(producer: str, *products, requires=(), publish=()) -> ProductNode:
    """Build a node; products are names, (name, kind) pairs or declarations."""
    outputs = []
    for product in products:
        if isinstance(product, ProductDeclaration):
            outputs.append(product)
        elif isinstance(product, tuple):
            outputs.append(ProductDeclaration(name=product[0], kind=product[1]))
        else:
            outputs.append(ProductDeclaration(name=product, kind=DataKind.SPECTRAL))
    return ProductNode(
        node_id=f"{producer}:{':'.join(output.name for output in outputs)}",
        producer=producer,
        outputs=outputs,
        requirements=list(requires),
        publish_outputs=list(publish),
    )


def _edge(
    source: ProductNode, product: str, sink: ProductNode, requirement: str
) -> GraphEdge:
    return GraphEdge(
        producer=source.node_id,
        product=product,
        consumer=sink.node_id,
        requirement=requirement,
    )


def _binding(
    sink: ProductNode,
    requirement: str,
    source: str = "job.input",
    product: str = "input",
) -> ExternalInputBinding:
    return ExternalInputBinding(
        consumer=sink.node_id,
        requirement=requirement,
        source=source,
        product=product,
    )


def test_product_graph_validates_and_orders():
    """Graphs validate references, reject cycles and order topologically."""
    source = _node("engine.sde", ("trajectories", DataKind.TIME_SERIES))
    mid = _node(
        "analyser.spectrum",
        "psd",
        requires=[ProductRequirement(name="traces", kind=DataKind.TIME_SERIES)],
    )
    sink = _node(
        "analyser.spectral_peaks",
        "peaks",
        requires=[ProductRequirement(name="spectrum", kind=DataKind.SPECTRAL)],
    )
    edges = [
        _edge(source, "trajectories", mid, "traces"),
        _edge(mid, "psd", sink, "spectrum"),
    ]
    graph = ProductGraph(nodes=[sink, source, mid], edges=edges)
    order = [n.producer for n in graph.topological_order()]
    assert order == ["engine.sde", "analyser.spectrum", "analyser.spectral_peaks"]
    assert graph.terminal_outputs() == {
        source.node_id: set(),
        mid.node_id: set(),
        sink.node_id: {"peaks"},
    }

    with pytest.raises(ValidationError, match="acyclic"):
        looped = _node(
            "engine.looped",
            ("series", DataKind.TIME_SERIES),
            requires=[ProductRequirement(name="in", kind=DataKind.SPECTRAL)],
        )
        ProductGraph(
            nodes=[looped, mid],
            edges=[
                _edge(looped, "series", mid, "traces"),
                _edge(mid, "psd", looped, "in"),
            ],
        )

    with pytest.raises(ValidationError, match="unknown producer"):
        ProductGraph(
            nodes=[source],
            edges=[
                GraphEdge(
                    producer="missing",
                    product="trajectories",
                    consumer=source.node_id,
                    requirement="traces",
                )
            ],
        )

    with pytest.raises(ValidationError, match="undeclared output"):
        ProductGraph(
            nodes=[source, mid],
            edges=[_edge(source, "not-an-output", mid, "traces")],
        )

    with pytest.raises(ValidationError, match="undeclared requirement"):
        ProductGraph(
            nodes=[source, mid],
            edges=[_edge(source, "trajectories", mid, "not-a-requirement")],
        )


def test_product_node_multi_output_and_publish():
    """Nodes declare >=1 uniquely named outputs; publishing is validated."""
    node = _node("analyser.peaks", "candidates", "fit_table", publish=["candidates"])
    assert [o.name for o in node.outputs] == ["candidates", "fit_table"]
    assert node.publish_outputs == ["candidates"]

    with pytest.raises(ValidationError, match="duplicate output"):
        _node("analyser.peaks", "candidates", "candidates")
    with pytest.raises(ValidationError, match="publishes undeclared output"):
        _node("analyser.peaks", "candidates", publish=["missing"])
    with pytest.raises(ValidationError, match="at least 1"):
        ProductNode(node_id="n", producer="p", outputs=[])


def test_requirement_align_with_must_reference_sibling():
    """``align_with`` names an existing sibling requirement of the same node."""
    ok = _node(
        "analyser.scaling",
        "fit",
        requires=[
            ProductRequirement(name="series", kind=DataKind.TIME_SERIES),
            ProductRequirement(
                name="initial_state",
                kind=DataKind.TIME_SERIES,
                align_with="series",
                optional=True,
            ),
        ],
    )
    assert ok.requirements[1].align_with == "series"

    with pytest.raises(ValidationError, match="unknown sibling requirement"):
        _node(
            "analyser.scaling",
            "fit",
            requires=[
                ProductRequirement(name="aux", align_with="missing"),
            ],
        )
    with pytest.raises(ValidationError, match="aligns with itself"):
        _node(
            "analyser.scaling",
            "fit",
            requires=[
                ProductRequirement(name="aux", align_with="aux"),
            ],
        )


def test_product_node_rejects_duplicate_requirement_names():
    """Requirement names are unique within a node."""
    with pytest.raises(ValidationError, match="duplicate requirement names"):
        _node(
            "analyser.scaling",
            "fit",
            requires=[
                ProductRequirement(name="series", kind=DataKind.TIME_SERIES),
                ProductRequirement(name="series", kind=DataKind.SPECTRAL),
            ],
        )


def test_product_graph_rejects_conflicting_edges():
    """Edges whose endpoints declare conflicting facets are rejected."""
    stats = _node("analyser.moments", ("stats", DataKind.STATISTICS))
    series_sink = _node(
        "analyser.consumer",
        "out",
        requires=[ProductRequirement(name="series", kind=DataKind.TIME_SERIES)],
    )
    with pytest.raises(ValidationError, match="kind"):
        ProductGraph(
            nodes=[stats, series_sink],
            edges=[_edge(stats, "stats", series_sink, "series")],
        )

    mean_source = _node(
        "analyser.reducer",
        ProductDeclaration(name="mean", kind=DataKind.STATISTICS, quantity="mean"),
    )
    variance_sink = _node(
        "analyser.plot",
        "fig",
        requires=[
            ProductRequirement(
                name="input", kind=DataKind.STATISTICS, quantity="variance"
            )
        ],
    )
    with pytest.raises(ValidationError, match="quantity"):
        ProductGraph(
            nodes=[mean_source, variance_sink],
            edges=[_edge(mean_source, "mean", variance_sink, "input")],
        )

    fields_source = _node(
        "analyser.histogram",
        ProductDeclaration(name="hist", kind=DataKind.STATISTICS, fields=["counts"]),
    )
    fields_sink = _node(
        "analyser.report",
        "table",
        requires=[
            ProductRequirement(
                name="hist",
                kind=DataKind.STATISTICS,
                required_fields=["counts", "edges"],
            )
        ],
    )
    with pytest.raises(ValidationError, match="fields"):
        ProductGraph(
            nodes=[fields_source, fields_sink],
            edges=[_edge(fields_source, "hist", fields_sink, "hist")],
        )


def test_product_graph_rejects_multiple_edges_per_slot():
    """One (consumer, requirement) slot binds at most one edge: no fan-in."""
    source_a = _node("engine.sde_a", ("traces", DataKind.TIME_SERIES))
    source_b = _node("engine.sde_b", ("traces", DataKind.TIME_SERIES))
    sink = _node(
        "analyser.spectrum",
        "psd",
        requires=[ProductRequirement(name="series", kind=DataKind.TIME_SERIES)],
    )
    with pytest.raises(ValidationError, match="more than one edge"):
        ProductGraph(
            nodes=[source_a, source_b, sink],
            edges=[
                _edge(source_a, "traces", sink, "series"),
                _edge(source_b, "traces", sink, "series"),
            ],
        )


def test_product_graph_accepts_compatible_and_open_facets():
    """Compatible statistics edges pass; absent facets never conflict."""
    source = _node(
        "analyser.moments",
        ProductDeclaration(
            name="stats",
            kind=DataKind.STATISTICS,
            quantity="variance",
            fields=["value", "stderr"],
        ),
    )
    sink = _node(
        "analyser.report",
        "table",
        requires=[
            ProductRequirement(
                name="stats",
                kind=DataKind.STATISTICS,
                quantity="variance",
                required_fields=["value"],
            )
        ],
    )
    graph = ProductGraph(
        nodes=[source, sink], edges=[_edge(source, "stats", sink, "stats")]
    )
    assert len(graph.edges) == 1

    # Facets absent on either side (kind/quantity/fields unset) cannot conflict.
    open_source = _node("engine.any", "data")
    open_sink = _node("analyser.any", "out", requires=[ProductRequirement(name="in")])
    ProductGraph(
        nodes=[open_source, open_sink],
        edges=[_edge(open_source, "data", open_sink, "in")],
    )

    # A requirement facet with no declared counterpart is not a conflict either.
    partial_source = _node("engine.partial", ("series", DataKind.TIME_SERIES))
    partial_sink = _node(
        "analyser.partial",
        "out",
        requires=[
            ProductRequirement(
                name="in", kind=DataKind.TIME_SERIES, quantity="field_amplitude"
            )
        ],
    )
    ProductGraph(
        nodes=[partial_source, partial_sink],
        edges=[_edge(partial_source, "series", partial_sink, "in")],
    )


def test_external_input_binding_contract():
    """External bindings name a consumer slot and a non-empty source label."""
    binding = ExternalInputBinding(
        consumer="n", requirement="in", source="job.input", product="input"
    )
    assert binding.source == "job.input"
    assert binding.product == "input"
    with pytest.raises(ValidationError, match="source"):
        ExternalInputBinding(consumer="n", requirement="in", source="")
    with pytest.raises(ValidationError, match="extra"):
        ExternalInputBinding(
            consumer="n",
            requirement="in",
            source="job.input",
            product="input",
            provenance="x",
        )


def test_product_graph_required_slots_need_exactly_one_binding():
    """Required requirements bind exactly one edge or one external input."""
    source = _node("engine.sde", ("trajectories", DataKind.TIME_SERIES))
    sink = _node(
        "analyser.spectrum",
        "psd",
        requires=[
            ProductRequirement(name="traces", kind=DataKind.TIME_SERIES),
            ProductRequirement(name="reference", kind=DataKind.SPECTRAL, optional=True),
        ],
    )

    # Required slot bound externally; optional slot left unbound.
    graph = ProductGraph(
        nodes=[source, sink],
        external_inputs=[_binding(sink, "traces", source="job.trajectories")],
    )
    assert graph.external_inputs[0].source == "job.trajectories"

    with pytest.raises(ValidationError, match="has no binding"):
        ProductGraph(nodes=[source, sink])

    # An edge and an external binding for the same slot both count.
    with pytest.raises(ValidationError, match="more than one binding"):
        ProductGraph(
            nodes=[source, sink],
            edges=[_edge(source, "trajectories", sink, "traces")],
            external_inputs=[_binding(sink, "traces")],
        )

    # Two external bindings for the same slot count too.
    with pytest.raises(ValidationError, match="more than one binding"):
        ProductGraph(
            nodes=[source, sink],
            external_inputs=[
                _binding(sink, "traces", source="job.trajectories"),
                _binding(sink, "traces", source="artifact.ref"),
            ],
        )


def test_external_input_binding_resolves_one_schema_compatible_product():
    """Materialization validates the explicitly named external product."""
    sink = _node(
        "analyser.spectrum",
        "out",
        requires=[ProductRequirement(name="input", kind=DataKind.SPECTRAL)],
    )
    graph = ProductGraph(
        nodes=[sink],
        external_inputs=[
            _binding(sink, "input", source="artifact", product="spectrum")
        ],
    )
    schema = ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[AxisSchema(name="frequency", role=AxisRole.COORDINATE, size=4)],
        variables=[
            VariableSchema(
                name="spectrum",
                dtype="float64",
                value_domain="real",
                dims=("frequency",),
            )
        ],
        attributes={
            "frequency_units": "Hz",
            "orientation": "phase_decreasing",
            "sidedness": "two_sided",
            "normalization": "test",
            "window": "rectangular",
            "estimator": "test",
        },
    )
    graph.validate_external_products({"artifact": {"spectrum": schema}})
    with pytest.raises(ValueError, match="does not resolve"):
        graph.validate_external_products({"artifact": {"other": schema}})
    with pytest.raises(ValueError, match="does not satisfy"):
        graph.validate_external_products(
            {
                "artifact": {
                    "spectrum": ProductSchema(
                        kind=DataKind.TIME_SERIES,
                        axes=[
                            AxisSchema(
                                name="frequency", role=AxisRole.COORDINATE, size=4
                            )
                        ],
                        variables=schema.variables,
                    )
                }
            }
        )


def test_product_graph_optional_slots_allow_at_most_one_binding():
    """Optional requirements may be unbound but never bound twice."""
    ref_source = _node("analyser.reference", ("ref", DataKind.SPECTRAL))
    sink = _node(
        "analyser.compare",
        "delta",
        requires=[
            ProductRequirement(name="reference", kind=DataKind.SPECTRAL, optional=True)
        ],
    )
    with pytest.raises(ValidationError, match="more than one binding"):
        ProductGraph(
            nodes=[ref_source, sink],
            edges=[_edge(ref_source, "ref", sink, "reference")],
            external_inputs=[_binding(sink, "reference")],
        )


def test_product_graph_external_bindings_must_reference_declared_slots():
    """External bindings reference existing nodes and requirements."""
    sink = _node(
        "analyser.spectrum",
        "psd",
        requires=[ProductRequirement(name="traces", kind=DataKind.TIME_SERIES)],
    )
    with pytest.raises(ValidationError, match="unknown consumer"):
        ProductGraph(
            nodes=[sink],
            external_inputs=[
                ExternalInputBinding(
                    consumer="missing",
                    requirement="traces",
                    source="job.input",
                    product="trajectories",
                )
            ],
        )
    with pytest.raises(ValidationError, match="undeclared requirement"):
        ProductGraph(
            nodes=[sink],
            external_inputs=[_binding(sink, "not-a-requirement")],
        )


def test_requirement_matches_schema():
    """Requirements match schemas by kind/quantity/domain/fields/axes."""
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(name="scan", role=AxisRole.PARAMETER, size=1),
            AxisSchema(name="time", role=AxisRole.COORDINATE, size=8),
        ],
        variables=[
            VariableSchema(
                name="alpha",
                dtype="complex128",
                value_domain="complex",
                dims=("scan", "time"),
                quantity="field_amplitude",
            ),
            VariableSchema(
                name="valid_length",
                dtype="int64",
                value_domain="real",
                dims=("scan",),
                quantity="valid_length",
            ),
        ],
    )
    assert requirement_matches(ProductRequirement(name="in"), schema)
    assert requirement_matches(
        ProductRequirement(name="in", kind=DataKind.TIME_SERIES), schema
    )
    assert not requirement_matches(
        ProductRequirement(name="in", kind=DataKind.SPECTRAL), schema
    )
    # quantity and value_domain must hold for the same variable.
    assert requirement_matches(
        ProductRequirement(
            name="in", quantity="field_amplitude", value_domain="complex"
        ),
        schema,
    )
    assert not requirement_matches(
        ProductRequirement(name="in", quantity="field_amplitude", value_domain="real"),
        schema,
    )
    assert not requirement_matches(
        ProductRequirement(name="in", value_domain="complex", quantity="missing"),
        schema,
    )
    assert requirement_matches(
        ProductRequirement(name="in", required_fields=["valid_length"]), schema
    )
    assert not requirement_matches(
        ProductRequirement(name="in", required_fields=["missing"]), schema
    )
    assert requirement_matches(
        ProductRequirement(name="in", accepted_axes=["scan", "time", "extra"]), schema
    )
    assert not requirement_matches(
        ProductRequirement(name="in", accepted_axes=["time"]), schema
    )


def test_task_profile_roundtrip_and_resolver_validation():
    """Task profiles stay JSON-serializable; resolvers are dotted paths."""
    profile = EngineTaskProfile(
        id="analyze",
        requirements=PluginRequirementSet(
            required=["analyser"], forbidden=["model", "integrator"]
        ),
        inputs=[
            InputProductRequirement(
                name="spectrum",
                requirement=ProductRequirement(name="spectrum", kind=DataKind.SPECTRAL),
            )
        ],
        outputs=[
            OutputProductDeclaration(
                name="peaks",
                declaration=ProductDeclaration(name="peaks", kind=DataKind.STATISTICS),
            )
        ],
        # SDE profiles declare no resolver; a real test-owned callable keeps
        # this round-trip fixture pointing at an existing symbol.
        resolver="tests.plugins.profiled_plugins:resolve_base_profile",
    )
    payload = json.loads(json.dumps(profile.model_dump(mode="json")))
    assert EngineTaskProfile.model_validate(payload) == profile

    with pytest.raises(ValidationError, match="dotted path"):
        EngineTaskProfile(id="bad", resolver="not a path!")


def test_product_schema_version_is_frozen():
    """The product schema version is frozen for Phase 1 implementation."""
    assert PRODUCT_SCHEMA_VERSION == "qphase.product/1"
    assert _time_series_schema().schema_version == "qphase.product/1"


GOLDEN_SPECTRAL_FINGERPRINT = (
    "b0a12c58d8c92491a84a1cfd577c07f2d88189cec47ebb487deda696a3df6bb0"
)


class _FakeHandle:
    """Minimal runtime-conformant data handle."""

    def __init__(self, variable_schema, dtype, shape, device="cpu"):
        self._variable_schema = variable_schema
        self._dtype = dtype
        self._shape = tuple(shape)
        self._device = device

    @property
    def variable_schema(self):
        return self._variable_schema

    @property
    def device(self):
        return self._device

    @property
    def dtype(self):
        return self._dtype

    @property
    def shape(self):
        return self._shape

    @property
    def nbytes(self):
        return 8 * math.prod(self._shape or (1,))

    @property
    def read_only(self):
        return True

    @property
    def owner(self):
        return "engine.fake"

    def acquire(self, consumer, scope="execution"):
        return _FakeLease(self, consumer, scope)

    def materialize(self, target_device=None, copy_policy="allow"):
        if copy_policy == "never" and target_device not in (None, self._device):
            raise RuntimeError("payload not on target device")
        return None


class _FakeLease:
    """Minimal runtime-conformant lease with idempotent release."""

    def __init__(self, handle, consumer, scope):
        self._handle = handle
        self._consumer = consumer
        self._scope = scope
        self.released = False

    @property
    def handle(self):
        return self._handle

    @property
    def consumer(self):
        return self._consumer

    @property
    def scope(self):
        return self._scope

    def release(self):
        self.released = True


class _FakeBacking:
    """Runtime product backing over a fixed variable mapping."""

    def __init__(self, variables):
        self._variables = dict(variables)

    @property
    def variables(self):
        return self._variables


def _alpha_variable() -> VariableSchema:
    return VariableSchema(
        name="alpha",
        dtype="complex128",
        value_domain="complex",
        dims=("time",),
    )


def _handle_for(variable: VariableSchema, size: int = 8) -> _FakeHandle:
    return _FakeHandle(variable, variable.dtype, (size,))


def test_handle_and_lease_protocol_runtime_conformance():
    """Fake handles satisfy the frozen protocols; deferred APIs are absent."""
    handle = _handle_for(_alpha_variable())
    assert isinstance(handle, DataHandleProtocol)

    lease = handle.acquire("analyser.spectrum", scope="session")
    assert isinstance(lease, DataLeaseProtocol)
    assert lease.handle is handle
    assert lease.consumer == "analyser.spectrum"
    assert lease.scope == "session"
    lease.release()
    lease.release()  # idempotent
    assert lease.released

    # Deferred capabilities are deliberately absent from the frozen contract.
    assert not hasattr(DataHandleProtocol, "export_interface")
    assert not hasattr(DataHandleProtocol, "release")
    assert not hasattr(DataLeaseProtocol, "pin")
    assert not hasattr(DataLeaseProtocol, "pinned")


def test_materialize_never_copy_policy():
    """copy_policy='never' raises instead of copying across devices."""
    handle = _handle_for(_alpha_variable())
    handle.materialize(target_device="cpu", copy_policy="never")
    handle.materialize(copy_policy="never")  # no target: already resident
    with pytest.raises(RuntimeError, match="target device"):
        handle.materialize(target_device="cuda:0", copy_policy="never")


def test_validate_backing_accepts_exact_match():
    """A backing with one matching handle per variable validates."""
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=8)],
        variables=[
            _alpha_variable(),
            VariableSchema(
                name="count", dtype="int64", value_domain="real", dims=("time",)
            ),
        ],
    )
    backing = _FakeBacking(
        {
            "alpha": _handle_for(schema.variable("alpha")),
            "count": _handle_for(schema.variable("count")),
        }
    )
    assert isinstance(backing, RuntimeProductBacking)
    validate_backing(schema, backing)


def test_validate_backing_rejects_mismatches():
    """Missing/extra variables, dtype and shape mismatches raise."""
    schema = ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[AxisSchema(name="time", role=AxisRole.COORDINATE, size=8)],
        variables=[_alpha_variable()],
    )
    alpha = schema.variable("alpha")

    with pytest.raises(ValueError, match="misses variables"):
        validate_backing(schema, _FakeBacking({}))
    with pytest.raises(ValueError, match="unknown variables"):
        validate_backing(
            schema,
            _FakeBacking({"alpha": _handle_for(alpha), "ghost": _handle_for(alpha)}),
        )
    with pytest.raises(ValueError, match="dtype"):
        validate_backing(
            schema, _FakeBacking({"alpha": _FakeHandle(alpha, "float64", (8,))})
        )
    with pytest.raises(ValueError, match="closed axis size"):
        validate_backing(schema, _FakeBacking({"alpha": _handle_for(alpha, 4)}))
    with pytest.raises(ValueError, match="dims"):
        validate_backing(
            schema,
            _FakeBacking({"alpha": _FakeHandle(alpha, "complex128", (8, 1))}),
        )
    wrong_schema = alpha.model_copy(update={"quantity": "wrong_quantity"})
    with pytest.raises(ValueError, match="variable_schema"):
        validate_backing(
            schema,
            _FakeBacking({"alpha": _FakeHandle(wrong_schema, "complex128", (8,))}),
        )


def test_runtime_and_artifact_backings_are_distinct_types():
    """Artifact refs are not runtime backings; the two are never conflated."""
    ref = ArtifactRef(
        artifact_id="art-1",
        product_name="psd",
        product_schema=_spectral_schema(),
        storage_adapter="npz/3",
    )
    assert isinstance(ref, ArtifactRef)
    assert not isinstance(ref, RuntimeProductBacking)
