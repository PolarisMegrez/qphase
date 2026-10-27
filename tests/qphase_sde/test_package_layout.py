"""Import-boundary and layout tests for the qphase_sde 2.0 package skeleton.

Freezes the standard resource-package root layout: the execution helpers live
under ``runtime/``, the backend-neutral numerics under ``numerics/``, and the
old root modules (``batch``/``buffers``/``scan``/``ops``/``coordinates``/
``utils``) plus the retired ``math/`` package are gone without aliases. The
manifest is a pure declaration: importing it must never pull the engine,
concrete plugins or CuPy, and ``contracts`` must never depend on ``runtime``.

It also freezes the layering direction between the package's own layers, as a
static AST scan (never by importing): ``contracts`` and ``numerics`` are leaf
layers, the plugin-class layers sit below ``analyser`` and consume only
``contracts``/``numerics``/``model`` without knowing each other, ``runtime``
may drive ``integrator``/``observer`` plugins but never imports ``analyser``,
``estimators`` or ``engine``, and no analyser imports or constructs another
analyser family's class. The cross-analyser scan resolves module-form imports
and constructor calls statically; dynamically resolved construction is out of
its scope.
"""

import ast
import importlib
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pytest
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

#: The root-level assets the manifest must declare besides its plugin classes.
#: ``prediction`` is the service Phase 10 added to validate frozen CAM
#: predictions against restored SDE observations.
EXPECTED_EXTRA_ASSETS = (
    ("analyser_config", "module"),
    ("estimators", "directory"),
    ("graph", "module"),
    ("numerics", "directory"),
    ("planning", "directory"),
    ("prediction", "module"),
    ("products", "module"),
)

#: Package roots every plugin-class layer may never import: the analyser
#: layer, the orchestration layers (``planning``/``products``/``graph``/
#: ``runtime``), the ``engine`` facade and the typed ``result`` bundle.
#: Plugins consume only ``contracts``, ``numerics`` and ``model``.
_PLUGIN_UPPER_FORBIDDEN = frozenset(
    {"analyser", "engine", "graph", "planning", "products", "result", "runtime"}
)

#: Plugins do not know each other: each plugin-class layer also forbids the
#: other two plugin layers. Relations inside one layer (its own ``base``
#: protocol and sibling implementations) stay legal.
_PLUGIN_PEER_FORBIDDEN = {
    "integrator": frozenset({"estimators", "observer"}),
    "observer": frozenset({"estimators", "integrator"}),
    "estimators": frozenset({"integrator", "observer"}),
}

#: Layer -> package roots it may never import. ``contracts`` is the bottom
#: layer; ``numerics`` sits next to it and may keep using ``contracts``;
#: ``analyser`` sits below the orchestration layers, so it may use
#: ``products``/``result`` but never ``planning``/``graph``/``runtime``.
#: ``runtime`` (orchestration) may drive ``integrator``/``observer`` plugins
#: and use ``planning``/``products``/``result``, but never imports
#: ``analyser``, ``estimators`` or the ``engine`` facade.
LAYER_FORBIDDEN_IMPORTS = {
    "contracts": frozenset(
        {
            "analyser",
            "engine",
            "estimators",
            "integrator",
            "numerics",
            "observer",
            "planning",
            "products",
            "result",
            "runtime",
        }
    ),
    "numerics": frozenset(
        {
            "analyser",
            "engine",
            "estimators",
            "integrator",
            "observer",
            "planning",
            "products",
            "result",
            "runtime",
        }
    ),
    "integrator": _PLUGIN_UPPER_FORBIDDEN | _PLUGIN_PEER_FORBIDDEN["integrator"],
    "observer": _PLUGIN_UPPER_FORBIDDEN | _PLUGIN_PEER_FORBIDDEN["observer"],
    "estimators": _PLUGIN_UPPER_FORBIDDEN | _PLUGIN_PEER_FORBIDDEN["estimators"],
    "analyser": frozenset({"engine", "graph", "planning", "runtime"}),
    "runtime": frozenset({"analyser", "engine", "estimators"}),
}

#: Submodules of ``analyser/`` that hold shared machinery rather than one
#: analyser family: importing the base class from ``base`` and helpers from
#: these modules is how every analyser is written.
ANALYSER_SHARED_MODULES = frozenset(
    {
        "_shared",
        "allan_statistics",
        "base",
        "frequency_orientation",
        "sampled_signal",
        "spectral_input",
        "typed_reducer",
    }
)

#: Entry-point namespaces reserved by the core registry, not plugin classes.
_RESERVED_NAMESPACES = frozenset({"engine", "resource"})

#: One scanned source file: ``(module, source, is_package)``.
_SourceUnit = tuple[str, str, bool]


def _module_name(path: Path) -> str:
    """Dotted module name of a source file inside the SDE package."""
    parts = path.relative_to(SDE_ROOT.parent).with_suffix("").parts
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _source_unit(path: Path) -> _SourceUnit:
    """Return the scanned unit for one source file."""
    return (
        _module_name(path),
        path.read_text(encoding="utf-8"),
        path.name == "__init__.py",
    )


def _imports_from_source(
    module: str, source: str, is_package: bool
) -> list[tuple[str, list[str]]]:
    """Every import of one unit as ``(absolute module, imported names)``."""
    base = module if is_package else module.rsplit(".", 1)[0]
    imported: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.extend((alias.name, [alias.name]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = base.split(".")
                parts = parts[: len(parts) - (node.level - 1)]
                target = ".".join(parts + ([node.module] if node.module else []))
            else:
                target = node.module or ""
            if target:
                imported.append((target, [alias.name for alias in node.names]))
    return imported


def _layer_modules(layer: str) -> list[Path]:
    return sorted((SDE_ROOT / layer).rglob("*.py"))


def _units_of(layer: str) -> list[_SourceUnit]:
    return [_source_unit(path) for path in _layer_modules(layer)]


def _layer_import_offenders(layer: str, units: list[_SourceUnit]) -> list[str]:
    """Cross-layer imports of *units* that the *layer* rule forbids."""
    forbidden = LAYER_FORBIDDEN_IMPORTS[layer]
    offenders = []
    for module, source, is_package in units:
        for target, _ in _imports_from_source(module, source, is_package):
            if not target.startswith("qphase_sde."):
                continue
            top = target.split(".")[1]
            if top in forbidden:
                offenders.append(f"{module} -> {target}")
    return offenders


def _analyser_family(module: str) -> str | None:
    """First component after ``qphase_sde.analyser``, or None if outside it."""
    parts = module.split(".")
    if len(parts) < 3 or parts[1] != "analyser":
        return None
    return parts[2]


def _foreign_analyser_family(origin: str, family: str | None) -> str | None:
    """Return the other analyser family *origin* belongs to, if any."""
    other = _analyser_family(origin)
    if other is None or other == family or other in ANALYSER_SHARED_MODULES:
        return None
    return other


def _dotted_name(node: ast.AST) -> str | None:
    """Reconstruct a dotted ``a.b.c`` expression, or None if not one."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


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


@pytest.mark.parametrize("layer", sorted(LAYER_FORBIDDEN_IMPORTS))
def test_layer_does_not_import_upper_layers(layer):
    """Each layer may only import from the layers beneath it.

    ``contracts`` is the bottom layer, ``numerics`` may use ``contracts``, and
    both are read-only to everyone else. The plugin-class layers
    (``integrator``/``observer``/``estimators``) consume only
    ``contracts``/``numerics``/``model``: they must not import the analyser
    layer, the orchestration layers, ``result`` or each other. ``runtime`` may
    drive integrator/observer plugins and use ``planning``/``products``/
    ``result``, but never the analyser layer, the estimators layer or the
    ``engine`` facade. The scan is static: it reads the import statements and
    never imports the modules under test.
    """
    offenders = _layer_import_offenders(layer, _units_of(layer))
    assert not offenders, f"{layer}/ imports upper layers: {offenders}"


@pytest.mark.parametrize(
    ("layer", "source"),
    [
        # The engine facade is the top layer: nothing may import it upwards.
        ("analyser", "from qphase_sde.engine import Engine\n"),
        ("runtime", "from qphase_sde.engine import EngineConfig\n"),
        # Plugin classes sit below the analyser layer and orchestration.
        ("integrator", "from qphase_sde.analyser.psd import PsdAnalyser\n"),
        ("observer", "import qphase_sde.products\n"),
        ("estimators", "from qphase_sde.planning import build_execution_plan\n"),
        # Plugins do not know each other.
        ("integrator", "from qphase_sde.observer.base import Observer\n"),
        # The typed result bundle is orchestration, not a plugin dependency.
        ("observer", "from qphase_sde.result import SDEDataBundle\n"),
        # Runtime drives integrator/observer, not the estimators layer.
        (
            "runtime",
            "from qphase_sde.estimators.spectral_estimator import PsdEstimate\n",
        ),
    ],
)
def test_layer_rule_flags_upper_layer_imports(layer, source):
    """The layer rule is known to fail on an upward or cross-plugin import."""
    unit = (f"qphase_sde.{layer}.trap", source, False)
    assert _layer_import_offenders(layer, [unit])


def _resolved_imports(
    module: str, source: str, is_package: bool
) -> list[tuple[ast.ImportFrom, str]]:
    """Every ``from`` import of one unit paired with its absolute target."""
    base = module if is_package else module.rsplit(".", 1)[0]
    resolved = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level:
            parts = base.split(".")
            parts = parts[: len(parts) - (node.level - 1)]
            target = ".".join(parts + ([node.module] if node.module else []))
        else:
            target = node.module or ""
        resolved.append((node, target))
    return resolved


def _is_analyser_module(name: str) -> bool:
    """Whether *name* is a root-level module or subpackage of ``analyser/``."""
    target = SDE_ROOT / "analyser" / name
    return target.with_suffix(".py").is_file() or target.is_dir()


def _analyser_import_offenders(units: list[_SourceUnit]) -> list[str]:
    """Cross-family analyser imports of *units* (module or class form)."""
    offenders = []
    for module, source, is_package in units:
        family = _analyser_family(module)
        if family is None:
            continue
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _foreign_analyser_family(alias.name, family):
                        offenders.append(f"{module} imports module {alias.name}")
        for node, target in _resolved_imports(module, source, is_package):
            if not target.startswith("qphase_sde.analyser"):
                continue
            for alias in node.names:
                if target == "qphase_sde.analyser":
                    # ``from qphase_sde.analyser import psd`` names a submodule;
                    # re-exported classes/configs are checked as symbols below.
                    if _is_analyser_module(alias.name):
                        candidate = f"{target}.{alias.name}"
                        if _foreign_analyser_family(candidate, family):
                            offenders.append(f"{module} imports module {candidate}")
                elif alias.name.endswith("Analyser") and alias.name != "Analyser":
                    if _foreign_analyser_family(target, family):
                        offenders.append(f"{module} -> {target}.{alias.name}")
    return offenders


def _analyser_construction_offenders(units: list[_SourceUnit]) -> list[str]:
    """Find calls in *units* that construct another analyser family's class.

    The scan resolves local bindings introduced by ``import ... as`` and
    ``from ... import ...`` and checks the constructed callee's family. It is
    static, so a constructor reached through ``getattr`` or an opaque factory
    is out of scope.
    """
    offenders = []
    for module, source, is_package in units:
        family = _analyser_family(module)
        if family is None:
            continue
        bindings: dict[str, str] = {}
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname:
                        bindings[alias.asname] = alias.name
        for node, target in _resolved_imports(module, source, is_package):
            for alias in node.names:
                if alias.name.endswith("Analyser") and alias.name != "Analyser":
                    bindings[alias.asname or alias.name] = f"{target}.{alias.name}"
                elif target == "qphase_sde.analyser" and _is_analyser_module(
                    alias.name
                ):
                    bindings[alias.asname or alias.name] = f"{target}.{alias.name}"
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.Call):
                continue
            dotted = _dotted_name(node.func)
            if dotted is None:
                continue
            head, _, rest = dotted.partition(".")
            origin = f"{bindings[head]}.{rest}" if head in bindings else dotted
            leaf = origin.rsplit(".", 1)[-1]
            if leaf.endswith("Analyser") and leaf != "Analyser":
                if _foreign_analyser_family(origin, family):
                    offenders.append(f"{module} constructs {origin}")
    return offenders


def test_analysers_do_not_import_each_others_classes():
    """An analyser may reuse another analyser's *config*, never its class.

    Sharing the class would make one plugin's behaviour depend on another's
    implementation; sharing a config model is the sanctioned form (a carrier
    that configures an inline ridge is the existing example). A whole-module
    import of another analyser family is the same dependency in module form.
    ``base``/``_shared`` and the other shared helper modules are excluded, and
    the scan is static.
    """
    offenders = _analyser_import_offenders(_units_of("analyser"))
    assert not offenders, f"analysers import each other's classes: {offenders}"


def test_analysers_do_not_construct_each_others_analysers():
    """No analyser constructs another analyser family's class directly.

    Cross-analyser data flow belongs on typed ProductGraph edges, not on a
    direct constructor call. The scan resolves module aliases and from-imported
    classes; dynamic construction through ``getattr`` is out of its scope.
    """
    offenders = _analyser_construction_offenders(_units_of("analyser"))
    assert not offenders, f"analysers construct each other: {offenders}"


@pytest.mark.parametrize(
    "source",
    [
        "import qphase_sde.analyser.spectral_ridge\n",
        (
            "import qphase_sde.analyser.spectral_ridge as ridge\n"
            "ridge.SpectralRidgeAnalyser()\n"
        ),
        (
            "from qphase_sde.analyser.spectral_ridge import SpectralRidgeAnalyser\n"
            "SpectralRidgeAnalyser()\n"
        ),
        (
            "import qphase_sde\n"
            "qphase_sde.analyser.spectral_ridge.SpectralRidgeAnalyser()\n"
        ),
        "from qphase_sde.analyser import spectral_ridge\n",
    ],
)
def test_cross_analyser_rule_flags_module_and_construction(source):
    """The cross-family rule is known to fail on module form and constructor use."""
    unit = ("qphase_sde.analyser.trap", source, False)
    assert _analyser_import_offenders([unit]) or _analyser_construction_offenders(
        [unit]
    )


def test_cross_analyser_rule_allows_shared_modules_and_configs():
    """Shared helpers, the base class and config models stay importable."""
    source = (
        "from qphase_sde.analyser.base import Analyser\n"
        "from qphase_sde.analyser.frequency_orientation import orientation_metadata\n"
        "from qphase_sde.analyser.spectral_ridge import SpectralRidgeConfig\n"
        "SpectralRidgeConfig()\n"
    )
    unit = ("qphase_sde.analyser.trap", source, False)
    assert _analyser_import_offenders([unit]) == []
    assert _analyser_construction_offenders([unit]) == []


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


def test_manifest_declares_existing_extra_assets():
    """Every declared extra asset exists and the retired ``math`` is gone.

    The manifest is the authoritative asset inventory, so the declaration and
    the tree must agree in both directions for the seven current assets, including
    the ``prediction`` validation service Phase 10 added.
    """
    declared = tuple(
        sorted((asset.path, asset.kind) for asset in MANIFEST.extra_assets)
    )
    assert declared == EXPECTED_EXTRA_ASSETS
    assert "math" not in {asset.path for asset in MANIFEST.extra_assets}
    assert all(asset.purpose for asset in MANIFEST.extra_assets)

    for asset in MANIFEST.extra_assets:
        target = SDE_ROOT / asset.path
        if asset.kind == "directory":
            assert target.is_dir(), asset.path
        else:
            assert (SDE_ROOT / f"{asset.path}.py").is_file(), asset.path


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
    from qphase_sde.estimators.peak_tracker import materialize_path_product
    from qphase_sde.estimators.peak_tracker.ridge import RidgePeakTracker

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
    from qphase_sde.estimators.peak_tracker.ridge import RidgePeakTracker

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
    from qphase_sde.estimators.peak_tracker.ridge import RidgePeakTracker

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
    from qphase_sde.estimators.peak_tracker.ridge import RidgePeakTracker

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
    from qphase_sde.estimators.coherence_frequency import (
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
