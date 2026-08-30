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
resolver
    Artifact location resolution contracts.
bundle
    The resource-independent generic data bundle.
store
    Artifact manifest v4 and the storage adapter contract.
npz
    NPZ 3.x storage adapter with lazy sharded handles.
graph
    Typed product dependency graphs.
"""

from .artifact import ArtifactRef, DataMaterializerProtocol
from .bundle import GenericDataBundle
from .datasets import (
    Dataset,
    SpectralDataset,
    StatisticsDataset,
    TimeSeriesDataset,
)
from .errors import (
    ArtifactAdapterError,
    ArtifactAmbiguousError,
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
from .resolver import (
    ArtifactResolverProtocol,
    DirectoryArtifactResolver,
    ProjectArtifactResolver,
    default_artifact_resolver,
    register_artifact_location,
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
    CoordinateSchema,
    ProductSchema,
    SamplingBasisSchema,
    SpectralAttributes,
    UncertaintySchema,
    VariableConstraints,
    VariableSchema,
)
from .store import (
    ARTIFACT_SCHEMA_VERSION,
    ArtifactAttachmentInfo,
    ArtifactManifest,
    BundleAdapterProtocol,
    BundleDescriptor,
    ProductEntry,
    ProductStorage,
    StorageAdapterProtocol,
    StorageVariableSummary,
    list_artifact_attachments,
    load_bundle,
    load_products,
    read_artifact_attachment,
    register_adapter,
    register_bundle_adapter,
    save_products,
    storage_adapter_available,
    storage_referenced_files,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PRODUCT_SCHEMA_VERSION",
    "ArtifactAdapterError",
    "ArtifactAmbiguousError",
    "ArtifactAttachmentInfo",
    "ArtifactCorruptError",
    "ArtifactError",
    "ArtifactManifest",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactResolverProtocol",
    "ArtifactUnsupportedError",
    "AxisRole",
    "AxisSchema",
    "BackendArrayHandle",
    "BundleAdapterProtocol",
    "BundleDescriptor",
    "CoordinateSchema",
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
    "DirectoryArtifactResolver",
    "GenericDataBundle",
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
    "ProjectArtifactResolver",
    "ReadOnlyArrayView",
    "RuntimeProductBacking",
    "SamplingBasisSchema",
    "SpectralAttributes",
    "SpectralDataset",
    "SpectralQuantity",
    "StatisticsDataset",
    "StorageAdapterProtocol",
    "StorageVariableSummary",
    "TimeSeriesDataset",
    "UncertaintySchema",
    "VariableConstraints",
    "VariableSchema",
    "default_artifact_resolver",
    "list_artifact_attachments",
    "load_bundle",
    "load_products",
    "read_artifact_attachment",
    "register_adapter",
    "register_artifact_location",
    "register_bundle_adapter",
    "save_products",
    "storage_adapter_available",
    "storage_referenced_files",
    "validate_backing",
]
