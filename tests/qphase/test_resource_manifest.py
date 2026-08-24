"""Contract tests for the experimental resource package manifest protocols."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from qphase.core.registry import DiscoveryService, RegistryCenter
from qphase.resources import (
    AssetOrigin,
    EntryPointDescriptor,
    ResourcePackageManifest,
    ResourceProfile,
    classify_origin,
    load_manifest_object,
    manifest_fingerprint,
    partition_entry_points,
    validate_manifest,
    validate_overlay_entry_points,
    validate_package_entry_points,
    validate_source_layout,
)

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "resource_manifests"

EXPECTED_SDE_NAMESPACES = (
    "model",
    "integrator",
    "observer",
    "analyser",
    "spectral_estimator",
    "peak_finder",
    "peak_tracker",
    "coherence_frequency",
)


def _load_fixture(name: str) -> ResourcePackageManifest:
    data = json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
    return ResourcePackageManifest.model_validate(data)


def _sde_entry_points() -> list[EntryPointDescriptor]:
    descriptors = [
        EntryPointDescriptor(
            "resource.sde", "qphase_sde.manifest:RESOURCE_MANIFEST",
            "qphase-sde", "2.0.0",
        ),
        EntryPointDescriptor(
            "engine.sde", "qphase_sde.engine:Engine", "qphase-sde", "2.0.0"
        ),
    ]
    for namespace in EXPECTED_SDE_NAMESPACES[3:]:
        descriptors.append(
            EntryPointDescriptor(
                f"{namespace}.example",
                f"qphase_sde.{namespace}.example:Example",
                "qphase-sde",
                "2.0.0",
            )
        )
    return descriptors


def _cam_owned_entry_points() -> list[EntryPointDescriptor]:
    return [
        EntryPointDescriptor(
            "resource.cam", "qphase_cam.manifest:RESOURCE_MANIFEST",
            "qphase-cam", "2.0.0",
        ),
        EntryPointDescriptor(
            "engine.cam", "qphase_cam.engine:Engine", "qphase-cam", "2.0.0"
        ),
        EntryPointDescriptor(
            "solver.homotopy", "qphase_cam.solver.homotopy:Homotopy",
            "qphase-cam", "2.0.0",
        ),
    ]


def _build_tree(root: Path, manifest: ResourcePackageManifest) -> None:
    from qphase.resources import (
        profile_required_directories,
        profile_required_modules,
    )

    profiles = set(manifest.profiles)
    for module in ("__init__.py", *profile_required_modules(profiles)):
        (root / module).write_text("", encoding="utf-8")
    for directory in profile_required_directories(profiles):
        (root / directory).mkdir(parents=True, exist_ok=True)
    for plugin_class in manifest.plugin_classes:
        directory = root / plugin_class.resolved_directory
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "__init__.py").write_text("", encoding="utf-8")
        (directory / "base.py").write_text("", encoding="utf-8")
    for asset in manifest.extra_assets:
        if asset.kind == "directory":
            (root / asset.path).mkdir(parents=True, exist_ok=True)
        else:
            (root / f"{asset.path}.py").write_text("", encoding="utf-8")


def test_sde_manifest_fixture_roundtrip_and_validation():
    """The full SDE target manifest round-trips through JSON and validates."""
    manifest = _load_fixture("sde.json")

    assert set(manifest.profiles) == {
        ResourceProfile.BASE,
        ResourceProfile.COMPUTE,
        ResourceProfile.SIMULATION,
    }
    assert manifest.plugin_class_namespaces == EXPECTED_SDE_NAMESPACES
    assert validate_manifest(manifest) == []

    dumped = manifest.model_dump(mode="json")
    reparsed = ResourcePackageManifest.model_validate(
        json.loads(json.dumps(dumped))
    )
    assert reparsed == manifest


def test_cam_minimal_manifest_fixture_is_valid():
    """A minimal CAM manifest proves the contract has no SDE-specific assumptions."""
    manifest = _load_fixture("cam_minimal.json")

    assert manifest.profiles == [ResourceProfile.BASE]
    assert manifest.plugin_class_namespaces == ("solver",)
    assert manifest.data_products[0].kind == "statistics"
    assert validate_manifest(manifest) == []


def test_manifest_rejects_unknown_fields():
    """The manifest schema is strictly extra-forbid."""
    data = json.loads(
        (FIXTURE_DIR / "cam_minimal.json").read_text(encoding="utf-8")
    )
    data["unexpected"] = True
    with pytest.raises(ValidationError):
        ResourcePackageManifest.model_validate(data)


def test_manifest_requires_valid_identifiers():
    """Resource ids and plugin namespaces are lowercase identifiers."""
    data = json.loads(
        (FIXTURE_DIR / "cam_minimal.json").read_text(encoding="utf-8")
    )
    data["resource_id"] = "CAM Upper"
    with pytest.raises(ValidationError):
        ResourcePackageManifest.model_validate(data)


def test_fingerprint_is_stable_and_order_independent():
    """The asset fingerprint does not depend on declaration or descriptor order."""
    manifest = _load_fixture("sde.json")
    entry_points = _sde_entry_points()

    first = manifest_fingerprint(manifest, entry_points)
    second = manifest_fingerprint(manifest, reversed(entry_points))
    assert first == second
    assert len(first) == 64  # sha256 hex digest

    shuffled = manifest.model_dump(mode="json")
    shuffled["plugin_classes"] = list(reversed(shuffled["plugin_classes"]))
    # Reordering declarations changes the canonical manifest; reordering the
    # fingerprint's entry-point descriptors must not.
    reordered = ResourcePackageManifest.model_validate(shuffled)
    assert manifest_fingerprint(reordered, entry_points) != first


def test_fingerprint_golden_value():
    """Freeze the canonicalization scheme with a golden digest."""
    manifest = _load_fixture("cam_minimal.json")
    descriptors = [
        EntryPointDescriptor(
            "resource.cam", "qphase_cam.manifest:RESOURCE_MANIFEST",
            "qphase-cam", "2.0.0",
        ),
        EntryPointDescriptor(
            "engine.cam", "qphase_cam.engine:Engine", "qphase-cam", "2.0.0"
        ),
    ]
    assert manifest_fingerprint(manifest, descriptors) == GOLDEN_CAM_FINGERPRINT


def test_fingerprint_distinguishes_overlay_provenance():
    """Package-owned and third-party descriptors produce different fingerprints."""
    manifest = _load_fixture("cam_minimal.json")
    package = EntryPointDescriptor(
        "solver.homotopy", "qphase_cam.solver.homotopy:Homotopy",
        "qphase-cam", "2.0.0",
    )
    third_party = EntryPointDescriptor(
        "solver.homotopy", "other_pkg.solvers:Homotopy", "other-pkg", "0.1"
    )
    assert manifest_fingerprint(manifest, [package]) != manifest_fingerprint(
        manifest, [third_party]
    )
    # Origin labels are explicit and serializable.
    assert AssetOrigin.THIRD_PARTY.value == "third_party"
    assert AssetOrigin.PROJECT_OVERLAY.value == "project_overlay"


def test_resource_entry_point_uses_existing_qphase_group(monkeypatch):
    """`resource.<id>` is discovered through the existing 'qphase' EP group."""

    class _FakeEntryPoint:
        def __init__(self, name: str, value: str) -> None:
            self.name = name
            self.value = value
            self.dist = None

    fake_eps = [
        _FakeEntryPoint("resource.cam", "qphase_cam.manifest:RESOURCE_MANIFEST"),
        _FakeEntryPoint("engine.cam", "qphase_cam.engine:Engine"),
    ]
    monkeypatch.setattr(
        "importlib.metadata.entry_points",
        lambda group: fake_eps if group == "qphase" else [],
    )

    center = RegistryCenter()
    DiscoveryService(center).discover_plugins()
    listing = center.list()
    assert "cam" in listing["resource"]
    assert "cam" in listing["engine"]


def test_classify_origin_and_partition_by_distribution():
    """Descriptors are classified and partitioned by distribution ownership."""
    sde = _sde_entry_points()
    cam = _cam_owned_entry_points()
    backend = EntryPointDescriptor(
        "backend.cuda", "qphase_backend_cuda:Cuda", "qphase-backend-cuda", "0.3"
    )
    overlay = EntryPointDescriptor("analyser.local", "local_plugins:Local")
    third_party = EntryPointDescriptor(
        "analyser.exotic", "community:Exotic", "community-pkg", "1.0"
    )
    mixed = [*sde, *cam, backend, overlay, third_party]

    assert classify_origin(sde[0], "qphase-sde") is AssetOrigin.PACKAGE
    assert classify_origin(overlay, "qphase-sde") is AssetOrigin.PROJECT_OVERLAY
    assert classify_origin(cam[0], "qphase-sde") is AssetOrigin.THIRD_PARTY
    # Without a distribution name, nothing is package-owned.
    assert classify_origin(overlay, None) is AssetOrigin.PROJECT_OVERLAY
    assert classify_origin(sde[0], None) is AssetOrigin.THIRD_PARTY

    partition = partition_entry_points(mixed, "qphase-sde")
    assert set(partition.package_owned) == set(sde)
    assert partition.project_overlay == (overlay,)
    assert set(partition.third_party) == {*cam, backend, third_party}


def test_validate_package_entry_points_detects_drift():
    """Package-owned validation catches missing resources, extra engines, drift."""
    manifest = _load_fixture("cam_minimal.json")
    good = _cam_owned_entry_points()
    partition = partition_entry_points(good, "qphase-cam")
    assert validate_package_entry_points(manifest, partition) == []

    missing = validate_package_entry_points(
        manifest, partition_entry_points(good[1:], "qphase-cam")
    )
    assert {i.code for i in missing} == {"missing-resource-entry-point"}

    two_engines = validate_package_entry_points(
        manifest,
        partition_entry_points(
            [
                *good,
                EntryPointDescriptor("engine.other", "x:Y", "qphase-cam", "2.0.0"),
            ],
            "qphase-cam",
        ),
    )
    assert "engine-count" in {i.code for i in two_engines}

    drifted = validate_package_entry_points(
        manifest,
        partition_entry_points(
            [
                *good,
                EntryPointDescriptor("analyser.psd", "x:Y", "qphase-cam", "2.0.0"),
            ],
            "qphase-cam",
        ),
    )
    assert "unknown-namespace" in {i.code for i in drifted}

    wrong_target = validate_package_entry_points(
        manifest,
        partition_entry_points(
            [
                good[0],
                EntryPointDescriptor(
                    "engine.cam", "qphase_cam.other:Engine", "qphase-cam", "2.0.0"
                ),
            ],
            "qphase-cam",
        ),
    )
    assert "engine-target-mismatch" in {i.code for i in wrong_target}


def test_mixed_global_group_validates_each_package_independently():
    """SDE+CAM+backend+overlay in one group cause no cross-talk.

    Each package recognizes only its own engine; backend and third-party
    descriptors never trigger engine-count or unknown-namespace issues, and
    overlay provenance stays separate from package-owned descriptors.
    """
    sde_manifest = _load_fixture("sde.json")
    cam_manifest = _load_fixture("cam_minimal.json")
    mixed = [
        *_sde_entry_points(),
        *_cam_owned_entry_points(),
        EntryPointDescriptor(
            "backend.cuda", "qphase_backend_cuda:Cuda", "qphase-backend-cuda", "0.3"
        ),
        # Project overlay extending SDE's declared analyser namespace.
        EntryPointDescriptor("analyser.local", "local_plugins:Local"),
    ]

    sde_partition = partition_entry_points(mixed, "qphase-sde")
    cam_partition = partition_entry_points(mixed, "qphase-cam")

    assert validate_package_entry_points(sde_manifest, sde_partition) == []
    assert validate_package_entry_points(cam_manifest, cam_partition) == []
    assert validate_overlay_entry_points(sde_manifest, sde_partition) == []
    # The SDE overlay is attributed to SDE by namespace; CAM's overlay
    # validation skips it instead of cross-flagging it.
    assert validate_overlay_entry_points(cam_manifest, cam_partition) == []


def test_validate_overlay_entry_points_enforces_overlay_policy():
    """Project overlays must not occupy reserved namespaces.

    Overlays in namespaces declared by the manifest are valid extensions;
    overlays in other non-reserved namespaces are attributed to other
    installed packages by namespace and skipped by this validator.
    """
    manifest = _load_fixture("sde.json")
    partition = partition_entry_points(
        [
            *_sde_entry_points(),
            EntryPointDescriptor("engine.rogue", "local_plugins:Rogue"),
            EntryPointDescriptor("resource.sde", "local_plugins:Fake"),
            # Declared SDE namespace: valid overlay.
            EntryPointDescriptor("analyser.local", "local_plugins:Local"),
            # Undeclared namespace: attributed to another package, skipped.
            EntryPointDescriptor("solver.local", "local_plugins:Solver"),
        ],
        "qphase-sde",
    )
    issues = validate_overlay_entry_points(manifest, partition)
    codes = [i.code for i in issues]
    assert codes.count("overlay-reserved-namespace") == 2
    assert "unknown-namespace" not in codes


def test_validate_source_layout_accepts_complete_tree(tmp_path):
    """A complete source tree passes the development-time validator."""
    manifest = _load_fixture("sde.json")
    root = tmp_path / "qphase_sde"
    root.mkdir()
    _build_tree(root, manifest)
    assert validate_source_layout(manifest, root) == []


def test_validate_source_layout_detects_problems(tmp_path):
    """The validator reports missing modules, stray files and misplaced plugins."""
    manifest = _load_fixture("sde.json")
    root = tmp_path / "qphase_sde"
    root.mkdir()
    _build_tree(root, manifest)

    (root / "config.py").unlink()
    (root / "utils.py").write_text("", encoding="utf-8")
    (root / "analyser" / "base.py").unlink()
    (root / "stray_dir").mkdir()

    codes = {issue.code for issue in validate_source_layout(manifest, root)}
    assert codes == {
        "missing-module",
        "undeclared-module",
        "missing-plugin-class-module",
        "undeclared-directory",
    }


def test_load_manifest_object_accepts_multiple_providers():
    """Manifests resolve from instances, mappings, callables and attributes."""
    manifest = _load_fixture("cam_minimal.json")

    assert load_manifest_object(manifest) is manifest
    assert load_manifest_object(manifest.model_dump(mode="json")) == manifest
    assert load_manifest_object(lambda: manifest) is manifest

    class _Provider:
        resource_manifest = manifest

    assert load_manifest_object(_Provider()) is manifest

    with pytest.raises(TypeError):
        load_manifest_object(object())


GOLDEN_CAM_FINGERPRINT = (
    "96cb4dc67ad5d8c80ab2674505c8a8a56f1faea2515240b15dceb20925d4a47d"
)
