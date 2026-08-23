"""qphase: Resource Package Contracts
---------------------------------------------------------
Experimental contracts (schema ``qphase.resource/1``) describing resource
packages as managed asset units: composable profiles, the resource package
manifest, asset declarations and development-time validators.

Public API
----------
profiles
    Resource profiles and their required assets.
manifest
    Resource package manifest models and fingerprinting.
assets
    Asset declarations and provenance.
validation
    Development-time manifest/source-layout/entry-point validators.
"""

from .assets import AssetOrigin, ResourceAssetDeclaration
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
    ValidationIssue,
    validate_entry_points,
    validate_manifest,
    validate_source_layout,
)

__all__ = [
    "RESOURCE_ENTRY_POINT_PREFIX",
    "RESOURCE_MANIFEST_SCHEMA",
    "AssetOrigin",
    "BackendCapabilities",
    "CompatibilityRange",
    "DataMaterializerDeclaration",
    "DataProductDeclaration",
    "EngineDeclaration",
    "EntryPointDescriptor",
    "OptionalDependencyDeclaration",
    "PluginClassDeclaration",
    "ResourceAssetDeclaration",
    "ResourcePackageManifest",
    "ResourcePackageProtocol",
    "ResourceProfile",
    "ValidationIssue",
    "load_manifest_object",
    "manifest_fingerprint",
    "profile_required_directories",
    "profile_required_modules",
    "resource_entry_point_name",
    "validate_entry_points",
    "validate_manifest",
    "validate_source_layout",
]
