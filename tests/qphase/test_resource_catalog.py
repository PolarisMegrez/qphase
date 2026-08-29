"""Contract tests for the resource package catalog."""

import importlib
import json
from pathlib import Path

from qphase.resources import (
    AssetOrigin,
    CatalogAsset,
    EntryPointDescriptor,
    ResourcePackageCatalog,
    ResourcePackageManifest,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "resource_manifests"


def _load_manifest(name: str) -> ResourcePackageManifest:
    data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return ResourcePackageManifest.model_validate(data)


def _descriptors() -> list[EntryPointDescriptor]:
    """Build a mixed group with SDE, CAM, backend and third-party plugins."""
    descriptors = [
        EntryPointDescriptor(
            "resource.sde",
            "qphase_sde.manifest:RESOURCE_MANIFEST",
            "qphase-sde",
            "2.0.0",
        ),
        EntryPointDescriptor(
            "engine.sde", "qphase_sde.engine:Engine", "qphase-sde", "2.0.0"
        ),
        EntryPointDescriptor(
            "analyser.spectrum",
            "qphase_sde.analyser.spectrum:Spectrum",
            "qphase-sde",
            "2.0.0",
        ),
        EntryPointDescriptor(
            "resource.cam",
            "qphase_cam.manifest:RESOURCE_MANIFEST",
            "qphase-cam",
            "2.0.0",
        ),
        EntryPointDescriptor(
            "engine.cam", "qphase_cam.engine:Engine", "qphase-cam", "2.0.0"
        ),
        EntryPointDescriptor(
            "solver.homotopy",
            "qphase_cam.solver.homotopy:Homotopy",
            "qphase-cam",
            "2.0.0",
        ),
        # Backend plugin and third-party extension from other distributions.
        EntryPointDescriptor(
            "backend.cuda",
            "qphase_backend_cuda:Cuda",
            "qphase-backend-cuda",
            "0.3",
        ),
        EntryPointDescriptor(
            "analyser.exotic", "community:Exotic", "community-pkg", "1.0"
        ),
    ]
    return descriptors


def _catalog(
    overlays: list[CatalogAsset] | None = None,
    core_version: str = "2.0.0",
) -> ResourcePackageCatalog:
    return ResourcePackageCatalog.from_descriptors(
        _descriptors(),
        overlays=overlays,
        manifest_loaders={
            "sde": _load_manifest("sde.json"),
            "cam": _load_manifest("cam_minimal.json"),
        },
        core_version=core_version,
    )


def test_catalog_discovers_packages_without_importing_engines(monkeypatch):
    """Discovery resolves manifests and assets, never importing engines."""
    imported: list[str] = []
    real_import = importlib.import_module
    monkeypatch.setattr(
        "qphase.resources.catalog.importlib.import_module",
        lambda name: imported.append(name) or real_import(name),
    )

    catalog = _catalog()
    assert {view.resource_id for view in catalog.packages} == {"cam", "sde"}
    assert catalog.issues == ()
    # Manifest loaders were injected, so discovery must import nothing at all.
    assert imported == []


def test_catalog_attributes_assets_by_ownership():
    """Each package sees only its own engine; overlays are not package assets."""
    catalog = _catalog()
    sde = catalog.package("sde")
    cam = catalog.package("cam")

    sde_engines = [a for a in sde.package_assets if a.namespace == "engine"]
    assert [a.name for a in sde_engines] == ["sde"]
    cam_engines = [a for a in cam.package_assets if a.namespace == "engine"]
    assert [a.name for a in cam_engines] == ["cam"]

    # Backend/third-party descriptors are nobody's package asset.
    all_owned = [a for v in catalog.packages for a in v.package_assets]
    assert all(a.origin is AssetOrigin.PACKAGE for a in all_owned)
    assert {a.namespace for a in all_owned} <= {
        "resource",
        "engine",
        "analyser",
        "solver",
    }


def test_project_overlay_attribution_and_orphans():
    """Overlays are attributed by namespace; orphans are reported."""
    good = CatalogAsset(
        namespace="analyser",
        name="local",
        target="local_plugins:Local",
        origin=AssetOrigin.PROJECT_OVERLAY,
        source="proj/models/.qphase_plugins.yaml",
    )
    orphan = CatalogAsset(
        namespace="warpdrive",
        name="x",
        target="local_plugins:X",
        origin=AssetOrigin.PROJECT_OVERLAY,
        source="proj/models/.qphase_plugins.yaml",
    )
    catalog = _catalog(overlays=[good, orphan])
    sde = catalog.package("sde")
    assert good in sde.overlay_assets
    assert good.origin is AssetOrigin.PROJECT_OVERLAY
    assert {i.code for i in catalog.issues} == {"orphan-overlay-namespace"}
    assert catalog.package("cam").overlay_assets == ()


def test_compatibility_range_is_checked():
    """Manifests incompatible with the running core produce issues."""
    catalog = _catalog(core_version="0.9.0")
    codes = {issue.code for issue in catalog.issues}
    assert "incompatible-core" in codes


def test_package_version_mismatch_is_an_issue():
    """A manifest version drifting from its distribution is reported."""
    descriptors = [
        EntryPointDescriptor(
            "resource.sde",
            "qphase_sde.manifest:RESOURCE_MANIFEST",
            "qphase-sde",
            "9.9.9",
        ),
        EntryPointDescriptor(
            "engine.sde", "qphase_sde.engine:Engine", "qphase-sde", "9.9.9"
        ),
    ]
    catalog = ResourcePackageCatalog.from_descriptors(
        descriptors,
        manifest_loaders={"sde": _load_manifest("sde.json")},
        core_version="2.0.0",
    )
    codes = {issue.code for issue in catalog.issues}
    assert "package-version-mismatch" in codes


def test_manifest_load_failure_and_id_mismatch_are_issues():
    """Broken resource entry points degrade to issues, not exceptions."""
    descriptors = [
        EntryPointDescriptor(
            "resource.broken",
            "no_such_module:RESOURCE_MANIFEST",
            "broken-pkg",
            "0.1",
        ),
        EntryPointDescriptor(
            "engine.broken", "broken.engine:Engine", "broken-pkg", "0.1"
        ),
    ]
    catalog = ResourcePackageCatalog.from_descriptors(descriptors)
    assert {i.code for i in catalog.issues} == {"manifest-load-error"}
    assert catalog.packages == ()

    wrong_id = ResourcePackageCatalog.from_descriptors(
        [
            EntryPointDescriptor("resource.wrong", "unused:target", "wrong-pkg", "1.0"),
            EntryPointDescriptor(
                "engine.wrong", "wrong.engine:Engine", "wrong-pkg", "1.0"
            ),
        ],
        manifest_loaders={"wrong": _load_manifest("cam_minimal.json")},
        core_version="2.0.0",
    )
    codes = {i.code for i in wrong_id.issues}
    assert "resource-id-mismatch" in codes


def test_catalog_snapshot_is_json_serializable():
    """Catalog snapshots are JSON-safe without a global hash identity."""
    catalog = _catalog()
    snapshot = catalog.snapshot()
    assert snapshot["schema"] == "qphase.catalog/1"
    json.dumps(snapshot)  # must not raise

    overlay = CatalogAsset(
        namespace="analyser",
        name="local",
        target="local_plugins:Local",
        origin=AssetOrigin.PROJECT_OVERLAY,
        source="proj/.qphase_plugins.yaml",
    )
    with_overlay = _catalog(overlays=[overlay])
    assert with_overlay.snapshot() != snapshot


def test_overlay_assets_do_not_change_package_fingerprint():
    """Overlay provenance is separate from the package's own fingerprint."""
    plain = _catalog().package("sde")
    overlay = CatalogAsset(
        namespace="analyser",
        name="local",
        target="local_plugins:Local",
        origin=AssetOrigin.PROJECT_OVERLAY,
        source="proj/.qphase_plugins.yaml",
    )
    extended = _catalog(overlays=[overlay]).package("sde")
    assert plain.fingerprint == extended.fingerprint
