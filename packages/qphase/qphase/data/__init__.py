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
runtime
    Concrete runtime handles, leases and backing containers.
datasets
    Typed dataset containers for the three data kinds.
artifact
    Artifact references and materializers.
store
    Artifact manifest v3 and the storage adapter contract.
npz
    NPZ 2.x storage adapter with lazy sharded handles.
graph
    Typed product dependency graphs.
"""

from .artifact import ArtifactRef, DataMaterializerProtocol
from .datasets import (
    Dataset,
    SpectralDataset,
    StatisticsDataset,
    TimeSeriesDataset,
)
from .errors import (
    ArtifactAdapterError,
    ArtifactChecksumError,
    ArtifactCorruptError,
    ArtifactError,
    ArtifactNotFoundError,
    ArtifactUnsupportedError,
)
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
    DictProductBacking,
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
from .store import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactManifestV3,
    ChunkRecord,
    ProductEntry,
    ProductStorage,
    StorageAdapterProtocol,
    load_products,
    register_adapter,
    save_products,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PRODUCT_SCHEMA_VERSION",
    "ArtifactAdapterError",
    "ArtifactChecksumError",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactManifestV3",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactUnsupportedError",
    "AxisRole",
    "AxisSchema",
    "BackendArrayHandle",
    "ChunkRecord",
    "CopyPolicy",
    "DataBacking",
    "DataHandleProtocol",
    "DataKind",
    "DataLease",
    "DataLeaseProtocol",
    "DataMaterializerProtocol",
    "DataProduct",
    "Dataset",
    "DictProductBacking",
    "GraphEdge",
    "HostArrayHandle",
    "LeaseScope",
    "ProductDeclaration",
    "ProductEntry",
    "ProductGraph",
    "ProductNode",
    "ProductRequirement",
    "ProductSchema",
    "ProductStorage",
    "ReadOnlyArrayView",
    "RuntimeProductBacking",
    "SamplingBasisSchema",
    "SpectralAttributes",
    "SpectralDataset",
    "SpectralQuantity",
    "StatisticsDataset",
    "StorageAdapterProtocol",
    "TimeSeriesDataset",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
    "load_products",
    "register_adapter",
    "save_products",
    "validate_backing",
]
