"""Packaging contract: declared dependencies match what geotoolz imports.

Every third-party module imported at *module scope* anywhere under
``src/geotoolz`` must be a declared base dependency in ``pyproject.toml``
(so ``pip install geotoolz`` never relies on a transitive dependency),
and the optional extras must only ever be imported lazily (so a base
install can ``import geotoolz`` and fails only when an extra-backed
operator is used).

Imports inside functions, under ``if TYPE_CHECKING:``, or inside a
module-level ``try: ... except ImportError:`` guard are lazy / optional
and are not counted as module-scope imports.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from functools import cache
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src" / "geotoolz"
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

# Import name -> distribution name, for the cases where they differ. Used
# before falling back to the installed environment's metadata so the
# mapping does not depend on what happens to be installed.
IMPORT_TO_DIST = {
    "georeader": "georeader-spaceml",
    "skimage": "scikit-image",
    "sklearn": "scikit-learn",
    "yaml": "PyYAML",
    "netCDF4": "netCDF4",
    "pyhdf": "pyhdf",
    "hydra_zen": "hydra-zen",
    "geopatcher": "geotoolz-patcher",
    "geocatalog": "geotoolz-catalog",
    "typing_extensions": "typing-extensions",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
    assert match is not None, requirement
    return _normalize(match.group(1))


def _pyproject() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))


def _base_dependencies() -> set[str]:
    return {_requirement_name(r) for r in _pyproject()["project"]["dependencies"]}


def _extras() -> dict[str, set[str]]:
    optional = _pyproject()["project"].get("optional-dependencies", {})
    return {
        extra: {_requirement_name(r) for r in reqs} for extra, reqs in optional.items()
    }


@cache
def _installed_distributions() -> dict[str, list[str]]:
    return packages_distributions()


def _distribution(import_name: str) -> str:
    if import_name in IMPORT_TO_DIST:
        return _normalize(IMPORT_TO_DIST[import_name])
    installed = _installed_distributions().get(import_name)
    if installed:
        return _normalize(installed[0])
    return _normalize(import_name)


def _is_type_checking(test: ast.expr) -> bool:
    return "TYPE_CHECKING" in ast.unparse(test)


def _catches_import_error(node: ast.Try) -> bool:
    for handler in node.handlers:
        if handler.type is None:
            return True
        names = ast.unparse(handler.type)
        if "ImportError" in names or "ModuleNotFoundError" in names:
            return True
    return False


def _third_party_roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module is None:
            return []
        modules = [node.module]
    else:
        modules = [alias.name for alias in node.names]
    roots = []
    for module in modules:
        root = module.split(".")[0]
        if root in sys.stdlib_module_names or root in {"geotoolz", "__future__"}:
            continue
        roots.append(root)
    return roots


def _module_scope_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Third-party ``(root, lineno)`` imports executed at import time."""
    found: list[tuple[str, int]] = []

    def visit(statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Import | ast.ImportFrom):
                found.extend((root, stmt.lineno) for root in _third_party_roots(stmt))
            elif isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                continue  # lazy: runs only when called
            elif isinstance(stmt, ast.ClassDef):
                visit(stmt.body)  # class bodies execute at import time
            elif isinstance(stmt, ast.If):
                if not _is_type_checking(stmt.test):
                    visit(stmt.body)
                visit(stmt.orelse)
            elif isinstance(stmt, ast.Try):
                if not _catches_import_error(stmt):
                    visit(stmt.body)
                for handler in stmt.handlers:
                    visit(handler.body)
                visit(stmt.orelse)
                visit(stmt.finalbody)
            elif isinstance(stmt, ast.With):
                visit(stmt.body)

    visit(tree.body)
    return found


@cache
def _all_module_scope_imports() -> dict[str, list[str]]:
    """Map each third-party import root to the ``file:line`` sites using it."""
    sites: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        rel = path.relative_to(SRC.parent)
        for root, lineno in _module_scope_imports(tree):
            sites.setdefault(root, []).append(f"{rel}:{lineno}")
    return sites


def test_scanner_sees_known_top_level_imports() -> None:
    # Guard against a scanner bug silently making the tests vacuous.
    imports = _all_module_scope_imports()
    for expected in ("numpy", "rasterio", "affine", "pipekit", "georeader"):
        assert expected in imports, expected


def test_scanner_skips_lazy_and_guarded_imports() -> None:
    tree = ast.parse(
        "import numpy\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import pandas\n"
        "try:\n    import zarr\nexcept ImportError:\n    zarr = None\n"
        "def f():\n    import sklearn\n"
        "class C:\n    import scipy\n"
    )
    assert [root for root, _ in _module_scope_imports(tree)] == ["numpy", "scipy"]


def test_top_level_imports_are_declared() -> None:
    declared = _base_dependencies()
    undeclared = {
        root: sites
        for root, sites in _all_module_scope_imports().items()
        if _distribution(root) not in declared
    }
    assert not undeclared, (
        "Module-scope third-party imports whose distribution is not a base "
        "dependency in packages/geotoolz/pyproject.toml (declare it, or make "
        f"the import lazy behind an extra): {undeclared}"
    )


@pytest.mark.parametrize("extra", sorted(_extras()))
def test_extras_are_not_imported_at_module_top_level(extra: str) -> None:
    extra_dists = _extras()[extra] - _base_dependencies()
    leaked = {
        root: sites
        for root, sites in _all_module_scope_imports().items()
        if _distribution(root) in extra_dists
    }
    assert not leaked, (
        f"Distributions from the [{extra}] extra are imported at module top "
        f"level, so a base install cannot `import geotoolz`: {leaked}"
    )


def test_heavy_optional_backends_are_extras_not_base() -> None:
    # The base install stays slim: these are only reachable via extras.
    base = _base_dependencies()
    extras = _extras()
    for dist, extra in [
        ("matplotlib", "viz"),
        ("scikit-learn", "learn"),
        ("joblib", "learn"),
        ("zarr", "zarr"),
    ]:
        assert dist not in base, dist
        assert dist in extras.get(extra, set()), (dist, extra)


def test_missing_extra_error_names_the_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import importlib

    import numpy as np

    import geotoolz as gz
    from geotoolz.learn._src.estimators import GeoTensorEstimator

    real_import_module = importlib.import_module

    def fake_import_module(name: str, package: str | None = None):
        if name.split(".")[0] in {"matplotlib", "joblib"}:
            raise ImportError(f"simulated missing {name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match=r"pip install 'geotoolz\[viz\]'"):
        gz.ApplyColormap(name="viridis")(np.zeros((4, 4), dtype="float32"))
    estimator = GeoTensorEstimator.__new__(GeoTensorEstimator)
    with pytest.raises(ImportError, match=r"pip install 'geotoolz\[learn\]'"):
        estimator.save_state(tmp_path / "state.joblib")
