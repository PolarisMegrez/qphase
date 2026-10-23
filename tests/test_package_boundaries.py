"""Package-boundary tests: the dependency direction of core and resource packages.

The dependency direction is frozen statically (an AST scan of ``import``
statements, nothing is imported from the scanned packages):

* the core package ``qphase`` never imports a resource package or the
  project-local ``models`` overlay;
* the resource packages ``qphase_sde``/``qphase_cam``/``qphase_viz`` never
  import each other;
* the ``qphase_viz``/``qphase_cam``/``qphase`` test suites never reach into a
  sibling resource package's internals. The core suite is included: core
  behaviours are tested with the core-owned fixtures in ``tests/plugins``, and
  the cases that drive a real resource package live in that package's suite.

The ``models/`` overlay is allowed to import resource packages: it is the
project-local plugin layer and depends downward, not the other way round.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGES = REPO_ROOT / "packages"

CORE_PACKAGE = "qphase"
RESOURCE_PACKAGES = ("qphase_sde", "qphase_cam", "qphase_viz")
OVERLAY_PACKAGE = "models"


def _python_files(root: Path) -> list[Path]:
    """Return every Python source file below ``root``, skipping bytecode dirs."""
    assert root.is_dir(), root
    return sorted(
        path for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def _imported_modules(path: Path) -> set[str]:
    """Return the absolute module names one file imports.

    Relative imports are skipped: they can only stay inside the scanned
    package, so they can never cross a package boundary.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def _top_level(module: str) -> str:
    """Return the top-level package name of a dotted module path."""
    return module.split(".", 1)[0]


def _crossing_imports(root: Path, forbidden: tuple[str, ...]) -> list[str]:
    """Return ``file imports module`` records for forbidden top-level packages."""
    found = []
    for path in _python_files(root):
        for module in sorted(_imported_modules(path)):
            if _top_level(module) in forbidden:
                found.append(f"{path.relative_to(REPO_ROOT)} imports {module}")
    return found


def test_core_does_not_import_resource_packages_or_models():
    """The core package stays below every resource package and the overlay."""
    forbidden = (*RESOURCE_PACKAGES, OVERLAY_PACKAGE)
    violations = _crossing_imports(PACKAGES / CORE_PACKAGE / CORE_PACKAGE, forbidden)
    assert not violations, violations


@pytest.mark.parametrize("package", RESOURCE_PACKAGES)
def test_resource_packages_import_no_sibling_package(package):
    """Resource packages are independent: no direction of import is allowed."""
    forbidden = tuple(name for name in RESOURCE_PACKAGES if name != package)
    violations = _crossing_imports(PACKAGES / package / package, forbidden)
    assert not violations, violations


@pytest.mark.parametrize("suite", ("qphase_viz", "qphase_cam"))
def test_resource_test_suites_do_not_import_sibling_packages(suite):
    """Resource test suites use public contracts, not a sibling's internals."""
    forbidden = tuple(name for name in RESOURCE_PACKAGES if name != suite)
    violations = _crossing_imports(REPO_ROOT / "tests" / suite, forbidden)
    assert not violations, violations


def test_core_test_suite_does_not_import_resource_packages():
    """Core tests exercise core behaviour through the core-owned fixtures.

    The scan sees function-local imports too: deferring an import into a
    fixture is still a dependency on the resource package. A case that really
    needs a resource package's engine or analyser belongs in that package's
    suite instead.
    """
    forbidden = (*RESOURCE_PACKAGES, OVERLAY_PACKAGE)
    violations = _crossing_imports(REPO_ROOT / "tests" / "qphase", forbidden)
    assert not violations, violations
