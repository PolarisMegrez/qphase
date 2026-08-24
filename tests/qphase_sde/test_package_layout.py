"""Import-boundary and layout tests for the qphase_sde 2.0 package skeleton.

Freezes the standard resource-package root layout: the execution helpers live
under ``runtime/``, the backend-neutral numerics under ``math/``, and the old
root modules (``batch``/``buffers``/``scan``/``ops``/``coordinates``/``utils``)
are gone without aliases. The manifest is a pure declaration: importing it must
never pull the engine, concrete plugins or CuPy, and ``contracts`` must never
depend on ``runtime``.
"""

import ast
import importlib
import importlib.util
import subprocess
import sys
import tomllib
from pathlib import Path

from qphase.resources import ResourcePackageManifest, validate_manifest
from qphase_sde.manifest import MANIFEST

SDE_PACKAGE_ROOT = (
    Path(__file__).resolve().parents[2] / "packages" / "qphase_sde"
)
SDE_ROOT = SDE_PACKAGE_ROOT / "qphase_sde"
CONTRACTS_DIR = SDE_ROOT / "contracts"

REMOVED_ROOT_MODULES = ("batch", "buffers", "scan", "ops", "coordinates", "utils")
RUNTIME_MODULES = ("batch", "buffers", "scan")
MATH_MODULES = ("ops", "coordinates")

#: Entry-point namespaces reserved by the core registry, not plugin classes.
_RESERVED_NAMESPACES = frozenset({"engine", "resource"})


def test_old_root_modules_are_gone():
    """No importable trace of the pre-2.0 root modules may remain."""
    for name in REMOVED_ROOT_MODULES:
        assert importlib.util.find_spec(f"qphase_sde.{name}") is None, (
            f"qphase_sde.{name} must be deleted, not aliased"
        )


def test_runtime_and_math_modules_import_cleanly():
    """The migrated modules import from their canonical 2.0 locations."""
    for name in RUNTIME_MODULES:
        importlib.import_module(f"qphase_sde.runtime.{name}")
    for name in MATH_MODULES:
        importlib.import_module(f"qphase_sde.math.{name}")


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
    """Declared plugin-class namespaces equal the pyproject EP namespaces."""
    entry_points = _pyproject_entry_points()
    ep_namespaces = {
        name.split(".", 1)[0] for name in entry_points
    } - _RESERVED_NAMESPACES
    assert set(MANIFEST.plugin_class_namespaces) == ep_namespaces


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
