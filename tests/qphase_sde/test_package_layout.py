"""Import-boundary and layout tests for the qphase_sde 2.0 package skeleton.

Freezes the standard resource-package root layout: the execution helpers live
under ``runtime/``, the backend-neutral numerics under ``numerics/``, and the
old root modules (``batch``/``buffers``/``scan``/``ops``/``coordinates``/
``utils``) plus the retired ``math/`` package are gone without aliases. The
manifest is a pure declaration: importing it must never pull the engine,
concrete plugins or CuPy, and ``contracts`` must never depend on ``runtime``.
"""

import ast
import importlib
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
from qphase.resources import ResourcePackageManifest, validate_manifest
from qphase_sde.manifest import MANIFEST

SDE_PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "packages" / "qphase_sde"
SDE_ROOT = SDE_PACKAGE_ROOT / "qphase_sde"
CONTRACTS_DIR = SDE_ROOT / "contracts"

REMOVED_ROOT_MODULES = ("batch", "buffers", "scan", "ops", "coordinates", "utils")
REMOVED_PACKAGES = ("math",)
RUNTIME_MODULES = ("buffers", "scan")
NUMERICS_MODULES = (
    "band_limited",
    "coordinates",
    "finite_delay",
    "observable",
    "ops",
    "short_delay",
    "spectral",
)

#: Entry-point namespaces reserved by the core registry, not plugin classes.
_RESERVED_NAMESPACES = frozenset({"engine", "resource"})


def test_old_root_modules_are_gone():
    """No importable trace of the pre-2.0 root modules may remain."""
    for name in REMOVED_ROOT_MODULES:
        assert importlib.util.find_spec(f"qphase_sde.{name}") is None, (
            f"qphase_sde.{name} must be deleted, not aliased"
        )
    for name in REMOVED_PACKAGES:
        assert importlib.util.find_spec(f"qphase_sde.{name}") is None, (
            f"qphase_sde.{name}/ must be merged into numerics/, not aliased"
        )


def test_runtime_and_numerics_modules_import_cleanly():
    """The migrated modules import from their canonical 2.0 locations."""
    for name in RUNTIME_MODULES:
        importlib.import_module(f"qphase_sde.runtime.{name}")
    for name in NUMERICS_MODULES:
        importlib.import_module(f"qphase_sde.numerics.{name}")


def test_manifest_import_is_declaration_only():
    """Importing the manifest must not import the engine, plugins or CuPy.

    Runs in a subprocess so the assertion is independent of whatever the
    current test process has already imported.
    """
    code = (
        "import sys\n"
        "import qphase_sde.manifest as manifest\n"
        "forbidden = [\n"
        "    name for name in sys.modules\n"
        "    if name == 'cupy'\n"
        "    or name == 'qphase_sde.engine'\n"
        "    or name.startswith('qphase_sde.analyser')\n"
        "    or name.startswith('qphase_sde.integrator')\n"
        "    or name.startswith('qphase_sde.observer')\n"
        "    or name.startswith('qphase_sde.runtime')\n"
        "]\n"
        "assert not forbidden, forbidden\n"
        "from qphase.resources import ResourcePackageManifest\n"
        "assert isinstance(manifest.MANIFEST, ResourcePackageManifest)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_contracts_do_not_depend_on_runtime():
    """The contracts package must never import the runtime package."""
    for module in CONTRACTS_DIR.glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not alias.name.startswith("qphase_sde.runtime"), (
                        f"{module.name} imports {alias.name}"
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.level == 0 and node.module:
                    assert not node.module.startswith("qphase_sde.runtime"), (
                        f"{module.name} imports {node.module}"
                    )
                elif node.level >= 2 and node.module:
                    assert not node.module.split(".")[0] == "runtime", (
                        f"{module.name} relatively imports {node.module}"
                    )


def _pyproject_entry_points() -> dict[str, str]:
    pyproject = tomllib.loads(
        (SDE_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    return dict(pyproject["project"]["entry-points"]["qphase"])


def test_manifest_is_valid_and_declares_one_engine():
    """The static MANIFEST validates and matches the installed engine EP."""
    assert isinstance(MANIFEST, ResourcePackageManifest)
    assert validate_manifest(MANIFEST) == []
    # No time-streaming reducer exists; trajectory batching is not
    # streaming, so the manifest must not claim the capability.
    assert MANIFEST.backend_capabilities.streaming is False

    entry_points = _pyproject_entry_points()
    engine_points = {
        name: target
        for name, target in entry_points.items()
        if name.startswith("engine.")
    }
    # Exactly one engine entry point, matching the manifest declaration.
    assert set(engine_points) == {MANIFEST.engine.entry_point}
    assert engine_points[MANIFEST.engine.entry_point] == MANIFEST.engine.target


def test_manifest_plugin_namespaces_match_entry_points():
    """Every pyproject EP namespace is declared; models are overlay-only."""
    entry_points = _pyproject_entry_points()
    ep_namespaces = {
        name.split(".", 1)[0] for name in entry_points
    } - _RESERVED_NAMESPACES
    declared = set(MANIFEST.plugin_class_namespaces)
    assert ep_namespaces <= declared
    # Concrete models never ship as package-owned entry points: they are
    # project-local/third-party overlays, so 'model' is declared without a
    # package entry point to keep overlay attribution working.
    assert "model" in declared
    assert "model" not in ep_namespaces
    # peak_tracker/coherence_frequency declare their plugin-class protocols
    # ahead of the Wave 5B concrete implementations and entry points.
    assert declared - ep_namespaces == {"model"}


def test_manifest_declared_targets_resolve():
    """Every protocol/schema_ref dotted target of the manifest exists."""
    from qphase.data import ProductSchema

    for plugin_class in MANIFEST.plugin_classes:
        module_name, _, attr = plugin_class.protocol.partition(":")
        module = importlib.import_module(module_name)
        assert hasattr(module, attr), plugin_class.namespace

    for product in MANIFEST.data_products:
        module_name, _, attr = product.schema_ref.partition(":")
        module = importlib.import_module(module_name)
        schema = getattr(module, attr)
        assert isinstance(schema, ProductSchema), product.name
        assert schema.kind.value == product.kind


def test_ridge_peak_tracker_is_registered_and_tracks_candidates():
    from qphase_sde.contracts.peaks import PeakStatus
    from qphase_sde.peak_tracker import materialize_path_product
    from qphase_sde.peak_tracker.ridge import RidgePeakTracker

    tracker = RidgePeakTracker(frequency_scale=0.5, path_count=2)
    paths = tracker.track(
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        {
            "location": np.asarray([0.0, 0.1, 0.2]),
            "intensity": np.asarray([1.0, 1.0, 1.0]),
        },
        scan_count=2,
        channel_names=["mode_0"],
    )
    assert len(paths) == 2
    assert paths[0].candidate_rows == [0, 1]
    assert paths[0].scan_positions == [0, 1]
    assert paths[1].candidate_rows == [0, 2]
    product = materialize_path_product(paths, candidate_count=3)
    assert product.attributes["graph_ready"] is True
    assert product.axis("path").size == 2
    assert product.axis("path_member").size == 4
    ambiguous = RidgePeakTracker(frequency_scale=0.5, path_count=1).track(
        np.asarray([0, 1, 3], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        {
            "location": np.asarray([0.0, 0.1, 0.2]),
            "intensity": np.asarray([1.0, 1.0, 1.0]),
        },
        scan_count=2,
        channel_names=["mode_0"],
    )
    assert not ambiguous[0].status & PeakStatus.PATH_UNIQUE


def test_ridge_peak_tracker_uses_global_huber_path_cost():
    from qphase_sde.peak_tracker.ridge import RidgePeakTracker

    tracker = RidgePeakTracker(
        frequency_scale=1.0, continuity_weight=10.0, path_count=1
    )
    paths = tracker.track(
        np.asarray([0, 1, 3, 4], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 0, 0], dtype=np.int64),
        {
            "location": np.asarray([0.0, 0.6, 0.01, 0.01]),
            "intensity": np.asarray([1.0, 1.0, 1.0, 1.0]),
            "score": np.asarray([1.0, 1.0, 0.5, 1.0]),
        },
        scan_count=3,
        channel_names=["mode_0"],
    )
    assert len(paths) == 1
    assert paths[0].candidate_rows == [0, 2, 3]


def test_ridge_peak_tracker_keeps_runner_up_before_output_truncation():
    from qphase_sde.contracts.peaks import PeakStatus
    from qphase_sde.peak_tracker.ridge import RidgePeakTracker

    paths = RidgePeakTracker(path_count=1).track(
        np.asarray([0, 2, 3], dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([0, 0], dtype=np.int64),
        {
            "location": np.asarray([-0.1, 0.0, 0.0]),
            "intensity": np.asarray([1.0, 1.0, 1.0]),
        },
        scan_count=2,
        channel_names=["mode_0"],
    )
    assert len(paths) == 1
    assert not paths[0].status & PeakStatus.PATH_UNIQUE


def test_ridge_peak_tracker_keeps_runner_up_when_paths_merge_at_terminal():
    from qphase_sde.contracts.peaks import PeakStatus
    from qphase_sde.peak_tracker.ridge import RidgePeakTracker

    paths = RidgePeakTracker(
        path_count=1,
        continuity_weight=0.0,
        ambiguity_cost_delta=0.5,
    ).track(
        np.asarray([0, 1, 3, 4], dtype=np.int64),
        np.asarray([0, 1, 2], dtype=np.int64),
        np.asarray([0, 0, 0], dtype=np.int64),
        {
            "location": np.asarray([0.0, -0.2, 0.2, 0.0]),
            "intensity": np.ones(4),
        },
        scan_count=3,
        channel_names=["mode_0"],
    )

    assert len(paths) == 1
    assert not paths[0].status & PeakStatus.PATH_UNIQUE


def test_coherence_frequency_children_return_common_estimate_rows():
    from qphase_sde.coherence_frequency import (
        BandLimitedCoherenceFrequency,
        FiniteDelayCoherenceFrequency,
        ShortDelayCoherenceFrequency,
    )

    tau = np.arange(20, dtype=float) * 0.1
    correlation = np.exp((-0.2 + 0.4j) * tau)[None, :].repeat(4, axis=0)
    short = ShortDelayCoherenceFrequency().estimate(
        {"per_trajectory_correlation": correlation, "dt": 0.1}
    )
    frequency = np.linspace(-2.0, 2.0, 4096, endpoint=False)
    spectrum = 0.02 + 2.0 * 0.08**2 / ((frequency - 0.3) ** 2 + 0.08**2)
    band = BandLimitedCoherenceFrequency(freq_min=-1.0, freq_max=1.0).estimate(
        {"axis": frequency, "spectrum": spectrum}
    )
    finite = FiniteDelayCoherenceFrequency().estimate(
        {"axis": frequency, "spectrum": spectrum}
    )
    assert {short.estimator, band.estimator, finite.estimator} == {
        "short_delay",
        "band_limited",
        "finite_delay",
    }
