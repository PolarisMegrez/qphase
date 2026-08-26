"""qphase: Resource Package Manifest
---------------------------------------------------------
Defines the ``ResourcePackageManifest`` (schema ``qphase.resource/1``), the
managed-asset description of a resource package. The manifest freezes the
package's unique engine reference, composable resource profiles, plugin class
declarations, public data products, materializers, backend capabilities,
optional dependencies, compatibility range and a deterministic asset
fingerprint.

The manifest does not duplicate ``EngineManifest`` (engine task requirements)
or concrete ``PluginManifest`` (child slots and concrete configuration); it only
stores stable references which a resource catalog resolves and cross-validates.

Public API
----------
ResourcePackageManifest
    Top-level manifest model (schema ``qphase.resource/1``).
EngineDeclaration
    Unique engine reference of a resource package.
PluginClassDeclaration
    Declaration of one root-level plugin-class namespace.
DataProductDeclaration
    Reference to a public data product schema.
DataMaterializerDeclaration
    Reference to a runtime-handle/artifact materializer.
BackendCapabilities
    Declared backend/device capabilities.
OptionalDependencyDeclaration
    Declared optional dependency and its purpose.
CompatibilityRange
    Core compatibility version range.
EntryPointDescriptor
    Normalized description of one installed entry point.
ResourcePackageProtocol
    Structural protocol for manifest providers.
manifest_fingerprint
    Deterministic asset fingerprint.
load_manifest_object
    Resolve an entry-point target to a manifest.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ..core.utils import canonical_json
from .assets import ResourceAssetDeclaration
from .profiles import ResourceProfile

__all__ = [
    "RESOURCE_MANIFEST_SCHEMA",
    "RESOURCE_ENTRY_POINT_PREFIX",
    "BackendCapabilities",
    "CompatibilityRange",
    "DataMaterializerDeclaration",
    "DataProductDeclaration",
    "EngineDeclaration",
    "EntryPointDescriptor",
    "OptionalDependencyDeclaration",
    "PluginClassDeclaration",
    "ResourcePackageManifest",
    "ResourcePackageProtocol",
    "load_manifest_object",
    "manifest_fingerprint",
    "resource_entry_point_name",
]

#: Schema identifier frozen for the qphase 2.0 resource manifest contract.
RESOURCE_MANIFEST_SCHEMA = "qphase.resource/1"

#: Entry-point name prefix inside the existing ``qphase`` group.
RESOURCE_ENTRY_POINT_PREFIX = "resource."

_RESOURCE_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_DOTTED_TARGET_PATTERN = re.compile(r"^[A-Za-z_][\w.]*:[A-Za-z_][\w.]*$")


def resource_entry_point_name(resource_id: str) -> str:
    """Return the entry-point name registering a resource manifest."""
    return f"{RESOURCE_ENTRY_POINT_PREFIX}{resource_id}"


class EngineDeclaration(BaseModel):
    """Unique engine reference of a resource package.

    The referenced engine owns its ``EngineManifest`` (task requirements,
    defaults); the resource manifest only points at it.
    """

    model_config = ConfigDict(extra="forbid")

    entry_point: str = Field(
        description="Entry-point name inside the 'qphase' group, e.g. 'engine.sde'."
    )
    target: str = Field(
        description="Dotted import target of the engine class, e.g. "
        "'qphase_sde.engine:Engine'."
    )

    @field_validator("entry_point")
    @classmethod
    def _check_entry_point(cls, value: str) -> str:
        if not value.startswith("engine.") or len(value) <= len("engine."):
            raise ValueError(
                "engine entry point must use the 'engine.<id>' namespace"
            )
        return value

    @field_validator("target")
    @classmethod
    def _check_target(cls, value: str) -> str:
        if not _DOTTED_TARGET_PATTERN.match(value):
            raise ValueError(f"invalid dotted target: {value!r}")
        return value


class PluginClassDeclaration(BaseModel):
    """Declaration of one root-level plugin-class namespace.

    Directories, manifest namespaces and entry-point namespaces correspond
    one-to-one; ``directory`` and ``entry_point_namespace`` default to
    ``namespace`` and a mismatch is reported by the validator.
    """

    model_config = ConfigDict(extra="forbid")

    namespace: str = Field(
        description="Plugin-class namespace, e.g. 'analyser' or 'peak_finder'."
    )
    protocol: str = Field(
        description="Dotted path to the public plugin-class protocol, e.g. "
        "'qphase_sde.analyser.base:AnalyserProtocol'."
    )
    directory: str | None = Field(
        default=None,
        description="Root-level directory hosting the class; defaults to the "
        "namespace itself.",
    )
    schema_ref: str | None = Field(
        default=None,
        description="Dotted path to the base config schema of the class, if any.",
    )
    entry_point_namespace: str | None = Field(
        default=None,
        description="Entry-point namespace in the 'qphase' group; defaults to "
        "the namespace itself.",
    )
    description: str = ""

    @field_validator("namespace")
    @classmethod
    def _check_namespace(cls, value: str) -> str:
        if not _RESOURCE_ID_PATTERN.match(value):
            raise ValueError(f"invalid plugin-class namespace: {value!r}")
        return value

    @property
    def resolved_directory(self) -> str:
        """Return the declared directory, defaulting to the namespace."""
        return self.directory or self.namespace

    @property
    def resolved_entry_point_namespace(self) -> str:
        """Return the entry-point namespace, defaulting to the namespace."""
        return self.entry_point_namespace or self.namespace


class DataProductDeclaration(BaseModel):
    """Reference to one public data product of the resource package."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Stable product name, e.g. 'psd'.")
    kind: Literal["time_series", "spectral", "statistics"] = Field(
        description="Core data kind of the product."
    )
    schema_ref: str = Field(
        description="Dotted path to the product schema provider."
    )
    description: str = ""


class DataMaterializerDeclaration(BaseModel):
    """Reference to a materializer converting between runtime handles and
    artifact-backed products.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    target: str = Field(
        description="Dotted path implementing the data materializer protocol."
    )
    formats: list[str] = Field(
        default_factory=list,
        description="Artifact formats supported by this materializer.",
    )


class BackendCapabilities(BaseModel):
    """Declared backend and device capabilities of a resource package."""

    model_config = ConfigDict(extra="forbid")

    backends: list[str] = Field(default_factory=list)
    devices: list[str] = Field(default_factory=list)
    streaming: bool = Field(
        default=False,
        description="Whether the package supports time-streaming execution.",
    )


class OptionalDependencyDeclaration(BaseModel):
    """Declared optional dependency, the extra providing it and its purpose."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Distribution name, e.g. 'cupy-cuda12x'.")
    extra: str | None = Field(
        default=None, description="Package extra installing this dependency."
    )
    purpose: str = ""
    required_for: list[str] = Field(
        default_factory=list,
        description="Plugin classes or products that need this dependency.",
    )


class CompatibilityRange(BaseModel):
    """Core compatibility range of the resource package."""

    model_config = ConfigDict(extra="forbid")

    qphase_core: str = Field(
        description="Version specifier for the compatible qphase core, "
        "e.g. '>=2.0a0,<3.0'."
    )


@dataclass(frozen=True)
class EntryPointDescriptor:
    """Normalized description of one installed entry point.

    Only stable identifiers are recorded — never source paths, file mtimes or
    directory traversal results — so fingerprints are reproducible.
    """

    name: str
    value: str
    distribution: str | None = None
    version: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        """Return the JSON-serializable form used for fingerprinting."""
        return {
            "name": self.name,
            "value": self.value,
            "distribution": self.distribution,
            "version": self.version,
        }


class ResourcePackageManifest(BaseModel):
    """Managed-asset manifest of a resource package (schema qphase.resource/1).

    The manifest is the authoritative inventory used by the registry, CLI, GUI
    and scheduler; the source-tree skeleton is only a development convention
    checked by the development-time validator.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["qphase.resource/1"] = "qphase.resource/1"
    resource_id: str
    package_version: str
    engine: EngineDeclaration
    profiles: list[ResourceProfile]
    plugin_classes: list[PluginClassDeclaration] = Field(default_factory=list)
    data_products: list[DataProductDeclaration] = Field(default_factory=list)
    materializers: list[DataMaterializerDeclaration] = Field(default_factory=list)
    backend_capabilities: BackendCapabilities = Field(
        default_factory=BackendCapabilities
    )
    optional_dependencies: list[OptionalDependencyDeclaration] = Field(
        default_factory=list
    )
    compatibility: CompatibilityRange
    extra_assets: list[ResourceAssetDeclaration] = Field(default_factory=list)

    @field_validator("resource_id")
    @classmethod
    def _check_resource_id(cls, value: str) -> str:
        if not _RESOURCE_ID_PATTERN.match(value):
            raise ValueError(f"invalid resource id: {value!r}")
        return value

    @property
    def plugin_class_namespaces(self) -> tuple[str, ...]:
        """Return the declared plugin-class namespaces in declaration order."""
        return tuple(pc.namespace for pc in self.plugin_classes)

    def fingerprint(
        self, entry_points: Iterable[EntryPointDescriptor] = ()
    ) -> str:
        """Return the deterministic asset fingerprint of this manifest."""
        return manifest_fingerprint(self, entry_points)


@runtime_checkable
class ResourcePackageProtocol(Protocol):
    """Structural protocol for objects providing a resource manifest."""

    resource_manifest: ClassVar[Any]

    def get_resource_manifest(self) -> ResourcePackageManifest:
        """Return the package's resource manifest."""
        ...


def load_manifest_object(target: Any) -> ResourcePackageManifest:
    """Resolve an entry-point target to a ``ResourcePackageManifest``.

    Accepts a manifest instance, a JSON-compatible mapping, a zero-argument
    callable returning either, or an object exposing ``get_resource_manifest()``
    or a ``resource_manifest`` attribute.

    Raises
    ------
    TypeError
        If the target cannot provide a manifest.

    """
    if isinstance(target, ResourcePackageManifest):
        return target
    if isinstance(target, Mapping):
        return ResourcePackageManifest.model_validate(target)
    if hasattr(target, "get_resource_manifest"):
        return load_manifest_object(target.get_resource_manifest())
    if hasattr(target, "resource_manifest"):
        return load_manifest_object(target.resource_manifest)
    if callable(target):
        return load_manifest_object(target())
    raise TypeError(
        f"cannot resolve a ResourcePackageManifest from {type(target).__name__}"
    )


def manifest_fingerprint(
    manifest: ResourcePackageManifest,
    entry_points: Iterable[EntryPointDescriptor] = (),
) -> str:
    """Compute the deterministic asset fingerprint of a resource package.

    The fingerprint is derived from the canonicalized manifest (which carries
    the package version) and the sorted entry-point descriptors. It never
    depends on source absolute paths, file mtimes or directory traversal order.
    """
    descriptors = sorted(
        (ep.as_dict() for ep in entry_points),
        key=lambda item: canonical_json(item),
    )
    payload = {
        "manifest": manifest.model_dump(mode="json"),
        "entry_points": descriptors,
    }
    digest = hashlib.sha256(canonical_json(payload).encode("utf-8"))
    return digest.hexdigest()
