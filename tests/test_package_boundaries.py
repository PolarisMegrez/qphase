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
#: Workspace trees that only exist locally (``.gitignore``): a test that reads
#: them passes on the author's machine and fails in a clean checkout.
UNTRACKED_WORKSPACE_TREES = ("reports", "runs")
#: Trees whose *literal* path may never appear in a test. ``runs`` is left out on
#: purpose: negative cases pass ``"runs/..."`` as a rejected artifact id, and a
#: temporary project's own ``runs`` directory is a legitimate fixture, so only the
#: repository-rooted join form of ``runs`` is flagged.
_LITERAL_WORKSPACE_TREES = ("reports",)
#: Expressions whose value is the repository root rather than a temporary
#: workspace, so a ``runs`` segment below them is the local run store.
_REPO_ROOT_NAMES = frozenset({"REPO_ROOT", "ROOT", "REPO_DIR", "PROJECT_ROOT"})


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


def _path_literal_tree(value: str) -> str | None:
    """Return the untracked workspace tree a path literal names, if any.

    Constants that contain whitespace are prose (docstrings, error messages), not
    paths, so they are skipped; the remaining check asks whether the *first*
    segment of a repository-relative path is one of the locally-only trees that
    a test may not spell out in full (see ``_LITERAL_WORKSPACE_TREES``).
    """
    if not value or any(character.isspace() for character in value):
        return None
    segments = [segment for segment in value.replace("\\", "/").split("/") if segment]
    if segments and segments[0] in _LITERAL_WORKSPACE_TREES and len(segments) > 1:
        return segments[0]
    return None


def _is_repo_rooted(node: ast.AST) -> bool:
    """Return whether an expression is anchored at the repository root.

    ``__file__`` or a module-level ``REPO_ROOT``-style name anchors it; a bare
    ``workspace``/``tmp_path`` does not, which is how a temporary project's own
    ``runs`` directory stays legitimate.
    """
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and (
            child.id == "__file__" or child.id in _REPO_ROOT_NAMES
        ):
            return True
    return False


def _join_segment(node: ast.AST) -> str | None:
    """Return the untracked tree name a path join names, if any."""
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _workspace_segment(node.right, _is_repo_rooted(node.left))
    if isinstance(node, ast.Call) and node.args:
        callee = getattr(node.func, "attr", getattr(node.func, "id", ""))
        if callee != "join":
            return None
        rooted = _is_repo_rooted(node)
        for argument in node.args:
            found = _workspace_segment(argument, rooted)
            if found is not None:
                return found
    return None


def _workspace_segment(node: ast.AST, rooted: bool) -> str | None:
    """Return the tree name when a constant is a bare untracked segment."""
    if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
        return None
    if node.value not in UNTRACKED_WORKSPACE_TREES:
        return None
    if node.value == "runs" and not rooted:
        return None
    return node.value


def test_tests_read_no_untracked_workspace_evidence():
    """Tests must not read the git-ignored ``reports``/``runs`` workspace trees.

    A path into those trees resolves on the machine that produced the data and
    fails in every clean checkout, which is exactly how three search-evidence
    cases broke on CI. The scan flags two shapes, both at the exact segment
    level so that prose, report file names and temporary workspaces are ignored:

    * a string literal whose first segment is ``reports`` (a repository-relative
      evidence path written out in full), and
    * a ``/`` join or ``os.path.join`` with a bare ``reports``/``runs`` segment,
      where ``runs`` is only flagged when the expression is anchored at the
      repository root (``__file__``/``REPO_ROOT``) — a ``tmp_path`` project's
      own ``runs`` directory is a legitimate fixture, and ``"runs/..."`` stays a
      valid rejected artifact id in the negative cases.

    Real evidence belongs in a tracked fixture under ``tests/fixtures/``.
    """
    violations = []
    for path in _python_files(REPO_ROOT / "tests"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            relative = path.relative_to(REPO_ROOT)
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                name = _path_literal_tree(node.value)
                if name is not None:
                    violations.append(
                        f"{relative}:{node.lineno} names the untracked {name!r} tree"
                    )
                continue
            name = _join_segment(node)
            if name is not None:
                violations.append(
                    f"{relative}:{node.lineno} joins the untracked {name!r} tree"
                )
    assert not violations, violations
