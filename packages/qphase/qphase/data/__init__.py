"""qphase: Typed Data Product Contracts
---------------------------------------------------------
Experimental contracts (schema ``qphase.product/1``) for typed data products:
the three public data kinds, the product schema language, runtime handles and
leases, durable artifact references and the product graph.

Public API
----------
kinds
    Data kinds and spectral quantities.
schema
    Axis/variable/uncertainty/product schema models.
product
    The data product protocol.
handles
    Runtime handle and lease protocols.
artifact
    Artifact references and materializers.
graph
    Typed product dependency graphs.
"""

from .artifact import ArtifactRef, DataMaterializerProtocol
from .graph import (
    GraphEdge,
    ProductDeclaration,
    ProductGraph,
    ProductNode,
    ProductRequirement,
)
from .handles import CopyPolicy, DataHandleProtocol, DataLeaseProtocol, LeaseScope
from .kinds import DataKind, SpectralQuantity
from .product import (
    DataBacking,
    DataProduct,
    RuntimeProductBacking,
    validate_backing,
)
from .runtime import (
    BackendArrayHandle,
    DataLease,
    HostArrayHandle,
    ReadOnlyArrayView,
)
from .schema import (
    PRODUCT_SCHEMA_VERSION,
    AxisRole,
    AxisSchema,
    ProductSchema,
    SamplingBasisSchema,
    SpectralAttributes,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)

__all__ = [
    "PRODUCT_SCHEMA_VERSION",
    "ArtifactRef",
    "AxisRole",
    "AxisSchema",
    "BackendArrayHandle",
    "CopyPolicy",
    "DataBacking",
    "DataHandleProtocol",
    "DataKind",
    "DataLease",
    "DataLeaseProtocol",
    "DataMaterializerProtocol",
    "DataProduct",
    "GraphEdge",
    "HostArrayHandle",
    "LeaseScope",
    "ProductDeclaration",
    "ProductGraph",
    "ProductNode",
    "ProductRequirement",
    "ProductSchema",
    "ReadOnlyArrayView",
    "RuntimeProductBacking",
    "SamplingBasisSchema",
    "SpectralAttributes",
    "SpectralQuantity",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
    "validate_backing",
]
