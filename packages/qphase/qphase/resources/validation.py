"""qphase: Resource Manifest Validation
---------------------------------------------------------
Development-time validators cross-checking a resource package's manifest, its
source-tree skeleton and its installed entry points. Runtime and wheel
installations treat the manifest and entry points as the only source of truth;
these validators exist for contract tests and release checks, never as a runtime
discovery mechanism.

Entry-point validation is ownership-scoped: the global ``qphase`` entry-point
group may mix many resource packages, backend plugins and project overlays, so
descriptors are partitioned by distribution ownership first
(:func:`partition_entry_points`). :func:`validate_package_entry_points` then
checks only the descriptors owned by one package's distribution, while
:func:`validate_overlay_entry_points` applies a separate, narrower policy to
project overlays. Third-party descriptors are provenance-labeled
(:func:`classify_origin`) but never namespace-checked against this package's
manifest — each distribution is validated against its own manifest.

Public API
----------
ValidationIssue
    One structured validation finding.
validate_manifest
    Check structural invariants of a manifest.
validate_source_layout
    Check a source tree against the manifest and profiles.
EntryPointPartition
    Installed entry points split by ownership.
partition_entry_points
    Partition descriptors by distribution ownership.
classify_origin
    Classify one descriptor's provenance.
validate_package_entry_points
    Check only the package-owned descriptors against the manifest.
validate_overlay_entry_points
    Check project-overlay descriptors against the overlay policy.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .assets import AssetOrigin
from .manifest import (
    RESOURCE_ENTRY_POINT_PREFIX,
    EntryPointDescriptor,
    resource_entry_point_name,
)
from .profiles import (
    ResourceProfile,
    profile_required_directories,
    profile_required_modules,
)

if TYPE_CHECKING:
    from .manifest import ResourcePackageManifest

__all__ = [
    "EntryPointPartition",
    "ValidationIssue",
    "classify_origin",
    "partition_entry_points",
    "validate_manifest",
    "validate_overlay_entry_points",
    "validate_package_entry_points",
    "validate_source_layout",
]

#: Root-level entries that are never treated as undeclared assets.
_IGNORED_ROOT_ENTRIES = frozenset({"__pycache__"})


@dataclass(frozen=True)
class ValidationIssue:
    """One structured validation finding."""

    code: str
    message: str
    location: str = ""


def _check_duplicates(
    values: list[str], code: str, location: str
) -> list[ValidationIssue]:
    seen: set[str] = set()
    issues: list[ValidationIssue] = []
    for value in values:
        if value in seen:
            issues.append(
                ValidationIssue(
                    code=code,
                    message=f"duplicate entry {value!r}",
                    location=location,
                )
            )
        seen.add(value)
    return issues


def validate_manifest(manifest: ResourcePackageManifest) -> list[ValidationIssue]:
    """Check structural invariants of a resource manifest.

    Returns a list of issues; an empty list means the manifest is valid.
    """
    issues: list[ValidationIssue] = []

    profiles = manifest.profiles
    if ResourceProfile.BASE not in profiles:
        issues.append(
            ValidationIssue(
                code="missing-base-profile",
                message="every resource package must declare the 'base' profile",
                location="profiles",
            )
        )
    issues += _check_duplicates(
        [str(p) for p in profiles], "duplicate-profile", "profiles"
    )

    namespaces = [pc.namespace for pc in manifest.plugin_classes]
    issues += _check_duplicates(
        namespaces, "duplicate-plugin-class", "plugin_classes"
    )
    for pc in manifest.plugin_classes:
        if pc.directory is not None and pc.directory != pc.namespace:
            issues.append(
                ValidationIssue(
                    code="plugin-class-directory-mismatch",
                    message=(
                        f"plugin class {pc.namespace!r} declares directory "
                        f"{pc.directory!r}; directories and namespaces must "
                        "correspond one-to-one"
                    ),
                    location=f"plugin_classes.{pc.namespace}",
                )
            )
        if (
            pc.entry_point_namespace is not None
            and pc.entry_point_namespace != pc.namespace
        ):
            issues.append(
                ValidationIssue(
                    code="plugin-class-entrypoint-mismatch",
                    message=(
                        f"plugin class {pc.namespace!r} declares entry-point "
                        f"namespace {pc.entry_point_namespace!r}; entry-point "
                        "namespaces and manifest namespaces must correspond "
                        "one-to-one"
                    ),
                    location=f"plugin_classes.{pc.namespace}",
                )
            )

    issues += _check_duplicates(
        [dp.name for dp in manifest.data_products],
        "duplicate-data-product",
        "data_products",
    )
    issues += _check_duplicates(
        [m.name for m in manifest.materializers],
        "duplicate-materializer",
        "materializers",
    )
    issues += _check_duplicates(
        [a.path for a in manifest.extra_assets],
        "duplicate-asset",
        "extra_assets",
    )
    return issues


def validate_source_layout(
    manifest: ResourcePackageManifest, package_root: Path
) -> list[ValidationIssue]:
    """Check a package source tree against its manifest and profiles.

    Detects missing common modules, concrete plugins living outside their
    declared plugin-class directory and undeclared root-level assets. This is a
    development-time audit helper; runtime discovery never walks source trees.
    """
    issues: list[ValidationIssue] = []
    root = Path(package_root)
    profiles = set(manifest.profiles)

    for module in profile_required_modules(profiles):
        if not (root / module).is_file():
            issues.append(
                ValidationIssue(
                    code="missing-module",
                    message=f"profile-required module {module!r} is missing",
                    location=module,
                )
            )
    for required_directory in profile_required_directories(profiles):
        if not (root / required_directory).is_dir():
            issues.append(
                ValidationIssue(
                    code="missing-directory",
                    message=(
                        f"profile-required directory {required_directory!r}/ is missing"
                    ),
                    location=required_directory,
                )
            )

    for pc in manifest.plugin_classes:
        directory = root / pc.resolved_directory
        if not directory.is_dir():
            issues.append(
                ValidationIssue(
                    code="missing-plugin-class-directory",
                    message=(
                        f"plugin class {pc.namespace!r} has no directory "
                        f"{pc.resolved_directory!r}/"
                    ),
                    location=pc.resolved_directory,
                )
            )
            continue
        for required in ("__init__.py", "base.py"):
            if not (directory / required).is_file():
                issues.append(
                    ValidationIssue(
                        code="missing-plugin-class-module",
                        message=(
                            f"plugin class {pc.namespace!r} is missing "
                            f"{pc.resolved_directory}/{required}"
                        ),
                        location=f"{pc.resolved_directory}/{required}",
                    )
                )

    allowed_modules = {"__init__.py"}
    allowed_modules.update(profile_required_modules(profiles))
    declared_asset_paths = {a.path for a in manifest.extra_assets}
    for asset in manifest.extra_assets:
        target = root / asset.path
        exists = (
            target.is_dir()
            if asset.kind == "directory"
            else (root / f"{asset.path}.py").is_file() or target.is_file()
        )
        if not exists:
            issues.append(
                ValidationIssue(
                    code="missing-asset",
                    message=f"declared asset {asset.path!r} does not exist",
                    location=asset.path,
                )
            )

    allowed_directories = set(profile_required_directories(profiles))
    allowed_directories.update(
        pc.resolved_directory for pc in manifest.plugin_classes
    )
    allowed_directories.update(
        path for path in declared_asset_paths if (root / path).is_dir()
    )

    if not root.is_dir():
        return issues
    for child in sorted(root.iterdir()):
        name = child.name
        if name.startswith(".") or name in _IGNORED_ROOT_ENTRIES:
            continue
        if child.is_file() and child.suffix == ".py":
            if name not in allowed_modules:
                issues.append(
                    ValidationIssue(
                        code="undeclared-module",
                        message=(
                            f"root-level module {name!r} is neither "
                            "profile-required nor declared in the manifest"
                        ),
                        location=name,
                    )
                )
        elif child.is_dir() and name not in allowed_directories:
            issues.append(
                ValidationIssue(
                    code="undeclared-directory",
                    message=(
                        f"root-level directory {name!r}/ is neither a declared "
                        "plugin class nor a declared asset"
                    ),
                    location=name,
                )
            )
    return issues


@dataclass(frozen=True)
class EntryPointPartition:
    """Installed entry-point descriptors split by ownership.

    ``package_owned`` ships with the resource package's own distribution;
    ``project_overlay`` has no distribution (project-local overlays);
    ``third_party`` belongs to any other installed distribution (peer resource
    packages, backend plugins or community extensions).
    """

    package_owned: tuple[EntryPointDescriptor, ...] = field(default=())
    project_overlay: tuple[EntryPointDescriptor, ...] = field(default=())
    third_party: tuple[EntryPointDescriptor, ...] = field(default=())


def classify_origin(
    descriptor: EntryPointDescriptor, package_distribution: str | None
) -> AssetOrigin:
    """Classify one descriptor's provenance relative to a distribution.

    A descriptor is package-owned only when its distribution name matches
    ``package_distribution`` (and that name is not ``None``); descriptors
    without a distribution are project overlays; anything else is third-party.
    """
    if (
        package_distribution is not None
        and descriptor.distribution == package_distribution
    ):
        return AssetOrigin.PACKAGE
    if descriptor.distribution is None:
        return AssetOrigin.PROJECT_OVERLAY
    return AssetOrigin.THIRD_PARTY


def partition_entry_points(
    descriptors: Iterable[EntryPointDescriptor],
    package_distribution: str | None,
) -> EntryPointPartition:
    """Partition descriptors by ownership relative to one distribution.

    Callers must pass the *unfiltered* global group listing; the returned
    partition is the explicit input of both entry-point validators, so no
    validator ever relies on callers pre-filtering descriptors.
    """
    owned: list[EntryPointDescriptor] = []
    overlay: list[EntryPointDescriptor] = []
    third_party: list[EntryPointDescriptor] = []
    for descriptor in descriptors:
        origin = classify_origin(descriptor, package_distribution)
        if origin is AssetOrigin.PACKAGE:
            owned.append(descriptor)
        elif origin is AssetOrigin.PROJECT_OVERLAY:
            overlay.append(descriptor)
        else:
            third_party.append(descriptor)
    return EntryPointPartition(
        package_owned=tuple(owned),
        project_overlay=tuple(overlay),
        third_party=tuple(third_party),
    )


def validate_package_entry_points(
    manifest: ResourcePackageManifest,
    partition: EntryPointPartition,
) -> list[ValidationIssue]:
    """Check only the package-owned descriptors against the manifest.

    Verifies that exactly one ``resource.<id>`` and exactly one ``engine.*``
    entry point are owned by the package's distribution, that the engine
    entry point matches the manifest declaration, and that no owned
    descriptor drifts into an undeclared namespace. Peer resource packages
    and overlays in the same global group are never counted here.
    """
    issues: list[ValidationIssue] = []
    descriptors = partition.package_owned

    resource_name = resource_entry_point_name(manifest.resource_id)
    resource_eps = [d for d in descriptors if d.name == resource_name]
    if not resource_eps:
        issues.append(
            ValidationIssue(
                code="missing-resource-entry-point",
                message=(
                    f"no {resource_name!r} entry point owned by this "
                    "package's distribution found in the 'qphase' group"
                ),
                location=resource_name,
            )
        )
    elif len(resource_eps) > 1:
        issues.append(
            ValidationIssue(
                code="duplicate-resource-entry-point",
                message=f"multiple {resource_name!r} entry points found",
                location=resource_name,
            )
        )

    engine_eps = [d for d in descriptors if d.name.startswith("engine.")]
    if len(engine_eps) != 1:
        issues.append(
            ValidationIssue(
                code="engine-count",
                message=(
                    f"expected exactly one package-owned engine entry point, "
                    f"found {len(engine_eps)}; a resource package declares "
                    "one engine"
                ),
                location="engine",
            )
        )
    else:
        engine_ep = engine_eps[0]
        if engine_ep.name != manifest.engine.entry_point:
            issues.append(
                ValidationIssue(
                    code="engine-entry-point-mismatch",
                    message=(
                        f"installed engine entry point {engine_ep.name!r} does "
                        f"not match manifest {manifest.engine.entry_point!r}"
                    ),
                    location=manifest.engine.entry_point,
                )
            )
        if engine_ep.value != manifest.engine.target:
            issues.append(
                ValidationIssue(
                    code="engine-target-mismatch",
                    message=(
                        f"installed engine target {engine_ep.value!r} does not "
                        f"match manifest {manifest.engine.target!r}"
                    ),
                    location=manifest.engine.entry_point,
                )
            )

    allowed_namespaces = {
        RESOURCE_ENTRY_POINT_PREFIX.rstrip("."),
        "engine",
        *manifest.plugin_class_namespaces,
    }
    for descriptor in descriptors:
        namespace = descriptor.name.split(".", 1)[0]
        if namespace not in allowed_namespaces:
            issues.append(
                ValidationIssue(
                    code="unknown-namespace",
                    message=(
                        f"package-owned entry point {descriptor.name!r} uses "
                        "a namespace not declared by the resource manifest"
                    ),
                    location=descriptor.name,
                )
            )
    return issues


def validate_overlay_entry_points(
    manifest: ResourcePackageManifest,
    partition: EntryPointPartition,
) -> list[ValidationIssue]:
    """Check project-overlay descriptors against the overlay policy.

    Project overlays (descriptors without a distribution) extend an installed
    resource package from the local project. Overlays are attributed to
    packages **by namespace**: an overlay whose namespace is declared by this
    manifest extends this package; an overlay in any other non-reserved
    namespace is attributed to another installed package and skipped here, so
    co-installed packages never flag each other's overlays. The reserved
    ``resource.*``/``engine.*`` namespaces are owned by distributions only —
    any project overlay found there is always rogue, regardless of which
    manifest is being validated. Third-party descriptors are deliberately out
    of scope: they are provenance-labeled via :func:`classify_origin` and
    validated against their own distribution's manifest.

    Detecting overlays in namespaces declared by *no* installed package (for
    example a typo'd namespace) is a catalog-level union check and left to the
    Phase 1 catalog implementation.
    """
    issues: list[ValidationIssue] = []
    reserved = (RESOURCE_ENTRY_POINT_PREFIX.rstrip("."), "engine")
    for descriptor in partition.project_overlay:
        namespace = descriptor.name.split(".", 1)[0]
        if namespace in reserved:
            issues.append(
                ValidationIssue(
                    code="overlay-reserved-namespace",
                    message=(
                        f"project overlay {descriptor.name!r} must not occupy "
                        f"the reserved {namespace!r} namespace; overlays may "
                        "only contribute plugin-class implementations"
                    ),
                    location=descriptor.name,
                )
            )
    return issues
