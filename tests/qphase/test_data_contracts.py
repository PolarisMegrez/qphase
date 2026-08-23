"""Contract tests for the experimental data product schema protocols."""

import json

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
    AxisSchema,
    DataKind,
    MomentFamilySchema,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
    ProductSchema,
    SpectralQuantity,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)


def _time_series_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.TIME_SERIES,
        axes=[
            AxisSchema(
                name="time", size=1024, coordinate="regular", start=0.0,
                step=0.01, units="s",
            ),
            AxisSchema(name="trajectory", size=64, independent=True),
            AxisSchema(name="channel", size=2),
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
        uncertainties=[
            UncertaintySchema(
                target="alpha",
                kind="sample_std",
                independent_unit="trajectory",
                covariance="real_imag",
                count=64,
            )
        ],
    )


def _spectral_schema() -> ProductSchema:
    return ProductSchema(
        kind=DataKind.SPECTRAL,
        axes=[
            AxisSchema(name="frequency", size=513, units="Hz"),
            AxisSchema(name="channel", size=2),
        ],
        variables=[
            VariableSchema(
                name="power",
                dtype="float64",
                value_domain="real",
                dims=("frequency", "channel"),
                quantity=SpectralQuantity.POWER_SPECTRAL_DENSITY.value,
                constraints=VariableConstraints(nonnegative=True),
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


def test_schema_fingerprint_golden():
    """Freeze canonical serialization with a golden digest."""
    assert _spectral_schema().fingerprint() == GOLDEN_SPECTRAL_FINGERPRINT


def test_schema_is_extra_forbid():
    """Unknown fields are rejected at every level."""
    with pytest.raises(ValidationError):
        ProductSchema.model_validate(
            {"kind": "spectral", "variables": [], "bogus": 1}
        )
    with pytest.raises(ValidationError):
        AxisSchema.model_validate({"name": "t", "bogus": 1})


def test_variable_dtype_validation():
    """Object dtypes are forbidden; dtype names are normalized."""
    with pytest.raises(ValidationError, match="object dtype"):
        VariableSchema(
            name="bad", dtype="object", value_domain="real", dims=()
        )
    variable = VariableSchema(
        name="x", dtype="float64", value_domain="real", dims=()
    )
    assert variable.dtype == "<f8"


def test_variable_domain_must_match_dtype():
    """A real-valued variable cannot declare a complex dtype."""
    with pytest.raises(ValidationError, match="value_domain"):
        VariableSchema(
            name="x", dtype="complex128", value_domain="real", dims=()
        )


def test_variable_dims_must_reference_axes():
    """Variables may only span declared axes."""
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


def test_complex_uncertainty_requires_covariance_representation():
    """Complex variables must not carry a bare complex std."""
    with pytest.raises(ValidationError, match="covariance representation"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[AxisSchema(name="time", size=8)],
            variables=[
                VariableSchema(
                    name="a",
                    dtype="complex128",
                    value_domain="complex",
                    dims=("time",),
                )
            ],
            uncertainties=[UncertaintySchema(target="a", kind="sample_std")],
        )


def test_uncertainty_target_must_exist():
    """Uncertainties must reference declared variables and axes."""
    with pytest.raises(ValidationError, match="unknown variable"):
        ProductSchema(
            kind=DataKind.STATISTICS,
            variables=[
                VariableSchema(
                    name="m1", dtype="float64", value_domain="real", dims=()
                )
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


def test_regular_axis_requires_step():
    """Regular coordinates must declare a step."""
    with pytest.raises(ValidationError, match="step"):
        AxisSchema(name="time", coordinate="regular")


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


def test_moment_family_roundtrip_and_kind_restriction():
    """Moment families belong to statistics products only."""
    family = MomentFamilySchema(
        family_id="alpha-moments",
        moment_kind="raw",
        ordering="c_number",
        maximum_order=4,
        symmetry="symmetric",
    )
    schema = ProductSchema(
        kind=DataKind.STATISTICS,
        axes=[AxisSchema(name="order", size=4)],
        variables=[
            VariableSchema(
                name="moment",
                dtype="complex128",
                value_domain="complex",
                dims=("order",),
                quantity="moments",
            )
        ],
        moment_family=family,
    )
    payload = json.loads(json.dumps(schema.model_dump(mode="json")))
    assert ProductSchema.model_validate(payload).moment_family == family

    with pytest.raises(ValidationError, match="statistics"):
        ProductSchema(
            kind=DataKind.TIME_SERIES,
            axes=[AxisSchema(name="time", size=4)],
            variables=[
                VariableSchema(
                    name="x",
                    dtype="float64",
                    value_domain="real",
                    dims=("time",),
                )
            ],
            moment_family=family,
        )


def test_protocol_runtime_conformance():
    """Minimal structural implementations satisfy the frozen protocols."""
    from qphase.data import DataHandleProtocol, DataLeaseProtocol, DataProduct

    product_schema = _time_series_schema()

    class FakeHandle:
        schema = product_schema
        device = "cpu"
        dtype = "<c16"
        shape = (64, 1024, 2)
        nbytes = 64 * 1024 * 2 * 16
        read_only = True
        owner = "engine.sde"

        def acquire(self, consumer, scope="execution"):
            return FakeLease(self, consumer, scope)

        def release(self):
            return None

        def materialize(self, *, device=None, copy=True):
            return None

        def export_interface(self):
            return {"type": "host", "copied": False}

    class FakeLease:
        def __init__(self, handle, consumer, scope):
            self.handle = handle
            self.consumer = consumer
            self.scope = scope
            self.pinned = False

        def pin(self):
            self.pinned = True

        def release(self):
            return None

    class FakeProduct:
        schema = product_schema
        provenance = {"engine": "sde"}

        @property
        def backing(self):
            return FakeHandle()

    handle = FakeHandle()
    lease = handle.acquire("analyser.psd")
    assert isinstance(handle, DataHandleProtocol)
    assert isinstance(lease, DataLeaseProtocol)
    assert isinstance(FakeProduct(), DataProduct)


def test_artifact_ref_is_json_serializable():
    """Artifact references carry identity only, no arrays."""
    ref = ArtifactRef(
        artifact_id="art-123",
        product_schema=_spectral_schema(),
        loader="qphase_sde.serialization.npz:load",
        content_hash="abc",
        provenance={"engine": "sde"},
    )
    payload = json.loads(json.dumps(ref.model_dump(mode="json")))
    assert ArtifactRef.model_validate(payload) == ref


def _node(producer: str, product: str, requires=()) -> ProductNode:
    return ProductNode(
        producer=producer,
        declaration=ProductDeclaration(name=product, kind=DataKind.SPECTRAL),
        requirements=list(requires),
    )


def test_product_graph_validates_and_orders():
    """Graphs validate references, reject cycles and order topologically."""
    source = _node("engine.sde", "trajectories")
    mid = _node(
        "analyser.spectrum",
        "psd",
        [ProductRequirement(name="traces", kind=DataKind.TIME_SERIES)],
    )
    sink = _node(
        "analyser.spectral_peaks",
        "peaks",
        [ProductRequirement(name="spectrum", kind=DataKind.SPECTRAL)],
    )
    edges = [
        {"producer": source.fingerprint(), "consumer": mid.fingerprint()},
        {"producer": mid.fingerprint(), "consumer": sink.fingerprint()},
    ]
    graph = ProductGraph(nodes=[sink, source, mid], edges=edges)
    order = [n.producer for n in graph.topological_order()]
    assert order == ["engine.sde", "analyser.spectrum", "analyser.spectral_peaks"]

    with pytest.raises(ValidationError, match="acyclic"):
        ProductGraph(
            nodes=[source, mid],
            edges=[
                {"producer": source.fingerprint(), "consumer": mid.fingerprint()},
                {"producer": mid.fingerprint(), "consumer": source.fingerprint()},
            ],
        )

    with pytest.raises(ValidationError, match="unknown producer"):
        ProductGraph(
            nodes=[source],
            edges=[{"producer": "0" * 64, "consumer": source.fingerprint()}],
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
                requirement=ProductRequirement(
                    name="spectrum", kind=DataKind.SPECTRAL
                ),
            )
        ],
        outputs=[
            OutputProductDeclaration(
                name="peaks",
                declaration=ProductDeclaration(
                    name="peaks", kind=DataKind.STATISTICS
                ),
            )
        ],
        resolver="qphase_sde.config:resolve_analyze_profile",
    )
    payload = json.loads(json.dumps(profile.model_dump(mode="json")))
    assert EngineTaskProfile.model_validate(payload) == profile

    with pytest.raises(ValidationError, match="dotted path"):
        EngineTaskProfile(id="bad", resolver="not a path!")


def test_product_schema_version_is_frozen():
    """The product schema version is part of the frozen contract."""
    assert PRODUCT_SCHEMA_VERSION == "qphase.product/1"
    assert _time_series_schema().schema_version == "qphase.product/1"


GOLDEN_SPECTRAL_FINGERPRINT = (
    "874979d31eb20d2690cc5025d75b701a344bf7ee802a22aa882519846c8c37d1"
)
