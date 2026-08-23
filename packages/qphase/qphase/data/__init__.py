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
from .handles import DataHandleProtocol, DataLeaseProtocol
from .kinds import DataKind, SpectralQuantity
from .product import DataBacking, DataProduct
from .schema import (
    PRODUCT_SCHEMA_VERSION,
    AxisSchema,
    MomentFamilySchema,
    ProductSchema,
    SpectralAttributes,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)

__all__ = [
    "PRODUCT_SCHEMA_VERSION",
    "ArtifactRef",
    "AxisSchema",
    "DataBacking",
    "DataHandleProtocol",
    "DataKind",
    "DataLeaseProtocol",
    "DataMaterializerProtocol",
    "DataProduct",
    "GraphEdge",
    "MomentFamilySchema",
    "ProductDeclaration",
    "ProductGraph",
    "ProductNode",
    "ProductRequirement",
    "ProductSchema",
    "SpectralAttributes",
    "SpectralQuantity",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
]
