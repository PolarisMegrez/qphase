"""qphase: Resource Package Contracts
---------------------------------------------------------
Contracts (schema ``qphase.resource/1``) describing resource packages as
managed asset units: composable profiles, the resource package manifest,
asset declarations, the read-only resource package catalog and
development-time validators.

Public API
----------
profiles
    Resource profiles and their required assets.
manifest
    Resource package manifest models and fingerprinting.
assets
    Asset declarations and provenance.
catalog
    Read-only discovery/aggregation of installed resource packages.
validation
    Development-time manifest/source-layout/entry-point validators.
"""

from .assets import AssetOrigin, ResourceAssetDeclaration
from .catalog import (
    CATALOG_SCHEMA,
    CatalogAsset,
    ResourcePackageCatalog,
    ResourcePackageView,
    read_project_overlays,
)
from .manifest import (
    RESOURCE_ENTRY_POINT_PREFIX,
    RESOURCE_MANIFEST_SCHEMA,
    BackendCapabilities,
    CompatibilityRange,
    DataMaterializerDeclaration,
    DataProductDeclaration,
    EngineDeclaration,
    EntryPointDescriptor,
    OptionalDependencyDeclaration,
    PluginClassDeclaration,
    ResourcePackageManifest,
    ResourcePackageProtocol,
    load_manifest_object,
    manifest_fingerprint,
    resource_entry_point_name,
)
from .profiles import (
    ResourceProfile,
    profile_required_directories,
    profile_required_modules,
)
from .validation import (
    EntryPointPartition,
    ValidationIssue,
    classify_origin,
    partition_entry_points,
    validate_manifest,
    validate_overlay_entry_points,
    validate_package_entry_points,
    validate_source_layout,
)

__all__ = [
    "CATALOG_SCHEMA",
    "RESOURCE_ENTRY_POINT_PREFIX",
    "RESOURCE_MANIFEST_SCHEMA",
    "AssetOrigin",
    "BackendCapabilities",
    "CatalogAsset",
    "CompatibilityRange",
    "DataMaterializerDeclaration",
    "DataProductDeclaration",
    "EngineDeclaration",
    "EntryPointDescriptor",
    "EntryPointPartition",
    "OptionalDependencyDeclaration",
    "PluginClassDeclaration",
    "ResourceAssetDeclaration",
    "ResourcePackageCatalog",
    "ResourcePackageManifest",
    "ResourcePackageProtocol",
    "ResourcePackageView",
    "ResourceProfile",
    "ValidationIssue",
    "classify_origin",
    "load_manifest_object",
    "manifest_fingerprint",
    "partition_entry_points",
    "profile_required_directories",
    "profile_required_modules",
    "read_project_overlays",
    "resource_entry_point_name",
    "validate_manifest",
    "validate_overlay_entry_points",
    "validate_package_entry_points",
    "validate_source_layout",
]
