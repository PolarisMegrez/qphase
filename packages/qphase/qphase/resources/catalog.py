"""qphase: Resource Package Catalog
---------------------------------------------------------
Read-only resolved view of the installed resource packages. The catalog
discovers packages through the existing ``qphase`` entry-point group's
``resource.<id>`` entries — never by importing engines or walking source
trees — and cross-checks each package's manifest against the entry points
owned by its distribution. Project-local overlays are attributed to packages
by plugin-class namespace; third-party descriptors are provenance-labeled but
validated against their own distribution's manifest only.

The catalog is the single read model consumed by CLI/service/GUI; a resolved
job snapshot freezes the catalog view so the same package under different
projects still reproduces the exact asset set used.

Public API
----------
CatalogAsset
    One resolved asset (package-owned or overlay).
ResourcePackageView
    Resolved view of one resource package.
ResourcePackageCatalog
    Read-only catalog of all discovered resource packages.
read_project_overlays
    Read project-local plugin overlays from ``.qphase_plugins.yaml`` files.
"""

from __future__ import annotations

import importlib
import importlib.metadata
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ..core.utils import load_yaml
from .assets import AssetOrigin
from .manifest import (
    RESOURCE_ENTRY_POINT_PREFIX,
    EntryPointDescriptor,
    ResourcePackageManifest,
    load_manifest_object,
    manifest_fingerprint,
)
from .validation import (
    ValidationIssue,
    partition_entry_points,
    validate_overlay_entry_points,
    validate_package_entry_points,
)

if TYPE_CHECKING:
    from ..core.project import ProjectContext
    from ..core.protocols import EngineManifest

__all__ = [
    "CATALOG_SCHEMA",
    "CatalogAsset",
    "ResourcePackageCatalog",
    "ResourcePackageView",
    "read_project_overlays",
]

#: Schema identifier of catalog snapshots.
CATALOG_SCHEMA = "qphase.catalog/1"


@dataclass(frozen=True)
class CatalogAsset:
    """One resolved asset of the catalog.

    ``source`` records where the asset came from: ``"entry-point"`` for
    installed distributions, or the overlay file path for project-local
    plugins. Provenance is part of the asset identity — the same target
    provided by a package and by an overlay are distinct assets.
    """

    namespace: str
    name: str
    target: str
    origin: AssetOrigin
    distribution: str | None = None
    version: str | None = None
    source: str = "entry-point"

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-serializable form used for snapshots."""
        return {
            "namespace": self.namespace,
            "name": self.name,
            "target": self.target,
            "origin": self.origin.value,
            "distribution": self.distribution,
            "version": self.version,
            "source": self.source,
        }


def _asset_from_descriptor(
    descriptor: EntryPointDescriptor, origin: AssetOrigin
) -> CatalogAsset:
    namespace, _, name = descriptor.name.partition(".")
    return CatalogAsset(
        namespace=namespace,
        name=name,
        target=descriptor.value,
        origin=origin,
        distribution=descriptor.distribution,
        version=descriptor.version,
        source="entry-point",
    )


def _descriptor_from_entry_point(entry_point: Any) -> EntryPointDescriptor:
    """Normalize an importlib EntryPoint into a stable descriptor."""
    if isinstance(entry_point, EntryPointDescriptor):
        return entry_point
    distribution = getattr(entry_point, "dist", None)
    name = getattr(distribution, "metadata", {}).get("Name") if distribution else None
    version = getattr(distribution, "version", None) if distribution else None
    return EntryPointDescriptor(
        entry_point.name,
        entry_point.value,
        str(name) if name else None,
        str(version) if version else None,
    )


def read_project_overlays(project: ProjectContext) -> list[CatalogAsset]:
    """Read project-local plugin overlays from ``.qphase_plugins.yaml`` files.

    The same files the runtime registry consumes; parsing failures are skipped
    silently (mirroring discovery), never raising into catalog construction.
    """
    overlays: list[CatalogAsset] = []
    for plugin_dir in project.plugin_dirs:
        plugins_file = plugin_dir / ".qphase_plugins.yaml"
        if not plugins_file.is_file():
            continue
        try:
            data = load_yaml(plugins_file)
        except Exception:
            continue
        if not data or not isinstance(data, dict):
            continue
        plugins = data.get("plugins", [])
        if not isinstance(plugins, list):
            continue
        for entry in plugins:
            if not isinstance(entry, dict):
                continue
            plugin_type = entry.get("type", "")
            target = entry.get("target", "")
            if not plugin_type or not target:
                continue
            namespace, _, name = str(plugin_type).partition(".")
            if not name:
                namespace, name = "default", namespace
            overlays.append(
                CatalogAsset(
                    namespace=namespace,
                    name=name,
                    target=str(target),
                    origin=AssetOrigin.PROJECT_OVERLAY,
                    source=str(plugins_file),
                )
            )
    return overlays


def _import_target(dotted: str) -> Any:
    """Import a 'module:attr[.subattr]' reference."""
    module_name, _, attr = dotted.partition(":")
    target: Any = importlib.import_module(module_name)
    for part in attr.split("."):
        target = getattr(target, part)
    return target


def _core_version() -> str:
    """Best-effort installed qphase version for compatibility checks."""
    try:
        return importlib.metadata.version("qphase")
    except importlib.metadata.PackageNotFoundError:
        from .. import __version__

        return __version__


@dataclass(frozen=True)
class ResourcePackageView:
    """Resolved read-only view of one resource package.

    ``fingerprint`` covers the manifest plus the package-owned entry-point
    descriptors only; overlay assets are attributed views and never change
    the package fingerprint. The engine manifest is loaded lazily and on
    demand — discovery never imports the engine.
    """

    manifest: ResourcePackageManifest
    distribution: str | None
    package_version: str | None
    fingerprint: str
    package_assets: tuple[CatalogAsset, ...] = field(default=())
    overlay_assets: tuple[CatalogAsset, ...] = field(default=())

    @property
    def resource_id(self) -> str:
        """The package's resource identifier."""
        return self.manifest.resource_id

    def load_engine_manifest(self) -> EngineManifest:
        """Import the engine class and return its ``EngineManifest``.

        This is the single, explicit point where the catalog crosses from
        declarations into code; it is lazy so plain catalog discovery never
        imports engines or plugins.
        """
        engine_type = _import_target(self.manifest.engine.target)
        manifest = getattr(engine_type, "manifest", None)
        if manifest is None:
            raise ValueError(
                f"engine {self.manifest.engine.target!r} declares no manifest"
            )
        return manifest


class ResourcePackageCatalog:
    """Read-only catalog of discovered resource packages and overlays.

    Construct via :meth:`discover` (production) or :meth:`from_descriptors`
    (tests). Instances are immutable views; ``issues`` collects every
    validation finding instead of raising, so callers decide how to surface
    broken packages.
    """

    def __init__(
        self,
        packages: tuple[ResourcePackageView, ...],
        overlays: tuple[CatalogAsset, ...],
        issues: tuple[ValidationIssue, ...],
    ) -> None:
        self._packages = packages
        self._overlays = overlays
        self._issues = issues

    @classmethod
    def from_descriptors(
        cls,
        descriptors: list[EntryPointDescriptor],
        *,
        overlays: list[CatalogAsset] | None = None,
        manifest_loaders: dict[str, Any] | None = None,
        core_version: str | None = None,
    ) -> ResourcePackageCatalog:
        """Build a catalog from normalized descriptors (testable path).

        ``manifest_loaders`` maps resource ids to pre-loaded manifest objects
        (or loader callables) so tests never touch import machinery; entries
        absent from the mapping are imported from the descriptor target.
        """
        overlays = list(overlays or [])
        loaders = dict(manifest_loaders or {})
        version = core_version if core_version is not None else _core_version()

        issues: list[ValidationIssue] = []
        views: list[ResourcePackageView] = []

        resource_descriptors = [
            d for d in descriptors if d.name.startswith(RESOURCE_ENTRY_POINT_PREFIX)
        ]
        for resource_ep in resource_descriptors:
            resource_id = resource_ep.name[len(RESOURCE_ENTRY_POINT_PREFIX) :]
            distribution = resource_ep.distribution
            partition = partition_entry_points(descriptors, distribution)

            loader = loaders.get(resource_id)
            manifest: ResourcePackageManifest | None = None
            if loader is not None:
                try:
                    manifest = load_manifest_object(loader)
                except Exception as exc:  # noqa: BLE001 - collect, don't raise
                    issues.append(
                        ValidationIssue(
                            code="manifest-load-error",
                            message=f"manifest of {resource_id!r} failed to "
                            f"load: {exc}",
                            location=resource_ep.name,
                        )
                    )
                    continue
            else:
                try:
                    manifest = load_manifest_object(_import_target(resource_ep.value))
                except Exception as exc:  # noqa: BLE001
                    issues.append(
                        ValidationIssue(
                            code="manifest-load-error",
                            message=f"manifest of {resource_id!r} failed to "
                            f"load: {exc}",
                            location=resource_ep.name,
                        )
                    )
                    continue
            if manifest.resource_id != resource_id:
                issues.append(
                    ValidationIssue(
                        code="resource-id-mismatch",
                        message=(
                            f"entry point {resource_ep.name!r} serves a "
                            f"manifest for {manifest.resource_id!r}"
                        ),
                        location=resource_ep.name,
                    )
                )

            issues += validate_package_entry_points(manifest, partition)
            issues += validate_overlay_entry_points(manifest, partition)
            issues += _check_compatibility(manifest, version, resource_ep.name)
            issues += _check_package_version(
                manifest, resource_ep.version, resource_ep.name
            )

            owned = tuple(
                _asset_from_descriptor(d, AssetOrigin.PACKAGE)
                for d in partition.package_owned
            )
            attributed = _attribute_overlays(manifest, overlays)
            views.append(
                ResourcePackageView(
                    manifest=manifest,
                    distribution=distribution,
                    package_version=resource_ep.version,
                    fingerprint=manifest_fingerprint(manifest, partition.package_owned),
                    package_assets=owned,
                    overlay_assets=attributed,
                )
            )

        # Orphan overlays: namespace not declared by any discovered package.
        known_namespaces = {
            namespace
            for view in views
            for namespace in view.manifest.plugin_class_namespaces
        }
        for overlay in overlays:
            if overlay.namespace not in known_namespaces:
                issues.append(
                    ValidationIssue(
                        code="orphan-overlay-namespace",
                        message=(
                            f"project overlay {overlay.namespace}."
                            f"{overlay.name} declares a namespace no installed "
                            "resource package provides"
                        ),
                        location=f"{overlay.namespace}.{overlay.name}",
                    )
                )

        return cls(
            packages=tuple(sorted(views, key=lambda view: view.resource_id)),
            overlays=tuple(overlays),
            issues=tuple(issues),
        )

    @classmethod
    def discover(
        cls,
        group: str = "qphase",
        *,
        project: ProjectContext | None = None,
        entry_points: list[Any] | None = None,
    ) -> ResourcePackageCatalog:
        """Discover the installed resource packages and project overlays.

        Reads the ``qphase`` entry-point group (injectable for tests) and the
        project's ``.qphase_plugins.yaml`` overlays; engines and plugins are
        never imported.
        """
        if entry_points is None:
            entry_points = list(importlib.metadata.entry_points(group=group))
        descriptors = [_descriptor_from_entry_point(ep) for ep in entry_points]

        overlays: list[CatalogAsset] = []
        if project is None:
            try:
                from ..core.project import ProjectContext

                project = ProjectContext.discover()
            except Exception:  # noqa: BLE001 - overlays are optional
                project = None
        if project is not None:
            overlays = read_project_overlays(project)
        return cls.from_descriptors(descriptors, overlays=overlays)

    @property
    def packages(self) -> tuple[ResourcePackageView, ...]:
        """All discovered resource package views, sorted by resource id."""
        return self._packages

    @property
    def overlays(self) -> tuple[CatalogAsset, ...]:
        """All project overlay assets (attributed or orphan)."""
        return self._overlays

    @property
    def issues(self) -> tuple[ValidationIssue, ...]:
        """All validation findings collected during discovery."""
        return self._issues

    def package(self, resource_id: str) -> ResourcePackageView:
        """Return the view of one resource package."""
        for view in self._packages:
            if view.resource_id == resource_id:
                return view
        raise KeyError(f"unknown resource package {resource_id!r}")

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of the resolved catalog."""
        return {
            "schema": CATALOG_SCHEMA,
            "packages": [
                {
                    "resource_id": view.resource_id,
                    "distribution": view.distribution,
                    "package_version": view.package_version,
                    "fingerprint": view.fingerprint,
                    "plugin_classes": list(view.manifest.plugin_class_namespaces),
                    "package_assets": [
                        asset.as_dict() for asset in view.package_assets
                    ],
                    "overlay_assets": [
                        asset.as_dict() for asset in view.overlay_assets
                    ],
                }
                for view in self._packages
            ],
            "unattributed_overlays": [
                asset.as_dict()
                for asset in self._overlays
                if not any(asset in view.overlay_assets for view in self._packages)
            ],
            "issues": [
                {
                    "code": issue.code,
                    "message": issue.message,
                    "location": issue.location,
                }
                for issue in self._issues
            ],
        }


def _attribute_overlays(
    manifest: ResourcePackageManifest,
    overlays: list[CatalogAsset],
) -> tuple[CatalogAsset, ...]:
    """Attribute overlays to a package by declared plugin-class namespace."""
    declared = set(manifest.plugin_class_namespaces)
    return tuple(overlay for overlay in overlays if overlay.namespace in declared)


def _check_compatibility(
    manifest: ResourcePackageManifest, core_version: str, location: str
) -> list[ValidationIssue]:
    """Check the manifest's core compatibility range against the core version."""
    try:
        from packaging.specifiers import SpecifierSet
        from packaging.version import Version
    except ImportError:  # pragma: no cover - packaging is a core dependency
        return []
    specifier = manifest.compatibility.qphase_core
    try:
        if Version(core_version) in SpecifierSet(specifier):
            return []
    except Exception as exc:  # noqa: BLE001
        return [
            ValidationIssue(
                code="invalid-compatibility-range",
                message=f"unparseable compatibility range {specifier!r}: {exc}",
                location=location,
            )
        ]
    return [
        ValidationIssue(
            code="incompatible-core",
            message=(
                f"package {manifest.resource_id!r} requires qphase core "
                f"{specifier!r}, installed {core_version!r}"
            ),
            location=location,
        )
    ]


def _check_package_version(
    manifest: ResourcePackageManifest,
    distribution_version: str | None,
    location: str,
) -> list[ValidationIssue]:
    """Flag a manifest whose ``package_version`` drifts from its distribution.

    The manifest declaration and the installed distribution metadata are two
    views of the same release; a drift means one side was edited without
    re-releasing the other. Descriptors without distribution metadata
    (overlays) carry no version to compare against.
    """
    if distribution_version is None:
        return []
    if manifest.package_version == distribution_version:
        return []
    return [
        ValidationIssue(
            code="package-version-mismatch",
            message=(
                f"package {manifest.resource_id!r} declares package_version "
                f"{manifest.package_version!r} but the installed "
                f"distribution is {distribution_version!r}"
            ),
            location=location,
        )
    ]
