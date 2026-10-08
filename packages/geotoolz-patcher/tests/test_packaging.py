"""Packaging contract: declared dependencies match what geopatcher imports.

* Every third-party module imported at *module scope* under
  ``src/geopatcher`` is a declared base dependency, so ``pip install
  geotoolz-patcher`` never relies on a transitive dependency.
* No extra's distribution is imported at module scope (a base install can
  ``import geopatcher``; an adapter fails only when used).
* Every distribution an extra installs is imported somewhere (lazily is
  fine — that is what extras are for), no extra repeats a base dependency,
  and ``[patch-full]`` covers every substrate extra.
* ``geopatcher.__version__`` is the distribution version, kept in step by
  release-please through its ``x-release-please-version`` annotation.

Imports inside functions, under ``if TYPE_CHECKING:``, or inside a
``try: ... except ImportError:`` guard are lazy / optional and are not
module-scope imports.
"""

from __future__ import annotations

import ast
import re
import sys
import tomllib
from collections.abc import Callable
from functools import cache
from pathlib import Path

import pytest
from packaging.requirements import Requirement


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src" / "geopatcher"
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"

IMPORT_TO_DIST = {
    "geocloud": "geotoolz-cloud",
    "georeader": "georeader-spaceml",
    "typing_extensions": "typing-extensions",
}

# Extras that are deliberately not part of ``[patch-full]``, and why.
NOT_IN_PATCH_FULL = {
    "pipekit": "the operator-graph bridge, not a Field substrate",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _dist(import_root: str) -> str:
    return _normalize(IMPORT_TO_DIST.get(import_root, import_root))


@cache
def _project() -> dict:
    return tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]


def _base() -> set[str]:
    return {_normalize(Requirement(r).name) for r in _project()["dependencies"]}


def _extras() -> dict[str, set[str]]:
    return {
        extra: {_normalize(Requirement(r).name) for r in reqs}
        for extra, reqs in _project()["optional-dependencies"].items()
    }


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


def _roots(node: ast.Import | ast.ImportFrom) -> list[str]:
    if isinstance(node, ast.ImportFrom):
        if node.level or node.module is None:
            return []
        modules = [node.module]
    else:
        modules = [alias.name for alias in node.names]
    roots = []
    for module in modules:
        root = module.split(".")[0]
        if root in sys.stdlib_module_names or root in {"geopatcher", "__future__"}:
            continue
        roots.append(root)
    return roots


def _module_scope_imports(tree: ast.Module) -> list[tuple[str, int]]:
    """Third-party ``(root, lineno)`` imports executed at import time."""
    found: list[tuple[str, int]] = []

    def visit(statements: list[ast.stmt]) -> None:
        for stmt in statements:
            if isinstance(stmt, ast.Import | ast.ImportFrom):
                found.extend((root, stmt.lineno) for root in _roots(stmt))
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
def _trees() -> dict[str, ast.Module]:
    return {
        str(path.relative_to(SRC.parent)): ast.parse(
            path.read_text(encoding="utf-8"), filename=str(path)
        )
        for path in sorted(SRC.rglob("*.py"))
    }


@cache
def _module_scope_sites() -> dict[str, list[str]]:
    """Map each third-party import root to its module-scope ``file:line`` sites."""
    sites: dict[str, list[str]] = {}
    for rel, tree in _trees().items():
        for root, lineno in _module_scope_imports(tree):
            sites.setdefault(root, []).append(f"{rel}:{lineno}")
    return sites


@cache
def _used_distributions() -> set[str]:
    """Distributions imported anywhere (lazily included) by geopatcher."""
    roots: set[str] = set()
    for tree in _trees().values():
        for node in ast.walk(tree):
            if isinstance(node, ast.Import | ast.ImportFrom):
                roots.update(_roots(node))
    return {_dist(root) for root in roots}


def test_scanner_sees_known_imports() -> None:
    # Guard against a scanner bug silently making the tests vacuous.
    module_scope = _module_scope_sites()
    for expected in ("numpy", "georeader", "rasterio"):
        assert expected in module_scope, expected
    used = _used_distributions()
    for lazy in ("xarray", "zarr", "jax", "geotoolz-cloud", "pipekit", "pyproj"):
        assert lazy in used, lazy
        assert lazy not in {_dist(r) for r in module_scope}, lazy


def test_scanner_skips_lazy_and_guarded_imports() -> None:
    tree = ast.parse(
        "import numpy\n"
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    import pandas\n"
        "try:\n    import zarr\nexcept ImportError:\n    zarr = None\n"
        "def f():\n    import xarray\n"
        "class C:\n    import scipy\n"
    )
    assert [root for root, _ in _module_scope_imports(tree)] == ["numpy", "scipy"]


def test_top_level_imports_are_declared() -> None:
    base = _base()
    undeclared = {
        root: sites
        for root, sites in _module_scope_sites().items()
        if _dist(root) not in base
    }
    assert not undeclared, (
        "Module-scope third-party imports whose distribution is not a base "
        "dependency of geotoolz-patcher (declare it, or make the import lazy "
        f"behind an extra): {undeclared}"
    )


@pytest.mark.parametrize("extra", sorted(_extras()))
def test_extras_are_not_imported_at_module_top_level(extra: str) -> None:
    extra_dists = _extras()[extra] - _base()
    leaked = {
        root: sites
        for root, sites in _module_scope_sites().items()
        if _dist(root) in extra_dists
    }
    assert not leaked, (
        f"[{extra}] distributions imported at module top level, so a base "
        f"install cannot `import geopatcher`: {leaked}"
    )


def test_every_extra_dependency_is_used() -> None:
    used = _used_distributions()
    unused = {
        extra: sorted(dists - used)
        for extra, dists in _extras().items()
        if dists - used
    }
    assert not unused, f"extras install packages geopatcher never imports: {unused}"


def test_no_extra_repeats_a_base_dependency() -> None:
    base = _base()
    repeated = {
        extra: sorted(dists & base)
        for extra, dists in _extras().items()
        if dists & base
    }
    assert not repeated, f"extras re-list base dependencies: {repeated}"


def test_patch_full_covers_every_substrate_extra() -> None:
    extras = _extras()
    full = extras["patch-full"]
    for extra, dists in extras.items():
        if extra in {"patch-full", *NOT_IN_PATCH_FULL}:
            continue
        missing = dists - full
        assert not missing, f"[patch-full] lacks {sorted(missing)} from [{extra}]"


def test_version_is_the_distribution_version() -> None:
    import geopatcher

    version = _project()["version"]
    # release-please bumps `__version__` through this annotation (the
    # patcher's __init__ is an `extra-files` entry, i.e. a generic updater).
    init = (SRC / "__init__.py").read_text(encoding="utf-8")
    assert f'__version__ = "{version}"  # x-release-please-version' in init
    assert geopatcher.__version__ == version


def test_no_install_hint_names_the_import_package() -> None:
    # `geopatcher` is the import name; pip only knows `geotoolz-patcher`.
    stale = [
        f"{path.relative_to(SRC.parent)}:{lineno}"
        for path in sorted(SRC.rglob("*.py"))
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if re.search(r"(?<![\w-])geopatcher\[", line)
    ]
    assert not stale, f"install hints naming `geopatcher[...]`: {stale}"


def _blocked(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    """Make ``import <module>`` fail, as on an install without the extra."""
    for name in list(sys.modules):
        if name.split(".")[0] in modules:
            monkeypatch.delitem(sys.modules, name)
    for name in modules:
        monkeypatch.setitem(sys.modules, name, None)


def _xarray_field() -> None:
    from geopatcher.fields import XarrayField

    XarrayField(object())


# Adapters that guard their import at module scope (an import that may
# have run long before the test blocks the extra): the guarded name to
# unset alongside blocking the import.
_MODULE_GUARDS = {"grid": "geopatcher._src.fields.xarray.xr"}
# Adapter modules whose guard fails the import itself: dropped from
# ``sys.modules`` (restored afterwards) so the import re-runs.
_REIMPORT = {"cog": "geopatcher._src.fields.cog"}


def _dask_delayed() -> None:
    from geopatcher.run import to_delayed

    to_delayed(None, None)


def _streaming_writer() -> None:
    import numpy as np

    from geopatcher._src.spatial.aggregation import _open_zarr_array

    _open_zarr_array(
        "unused.zarr",
        shape=(1,),
        chunks=(1,),
        dtype=np.dtype("f4"),
        shard_shape=None,
        overwrite=False,
    )


def _cog_field() -> None:
    import importlib

    module = importlib.import_module("geopatcher._src.fields.cog")
    module.CogField.open("s3://bucket/scene.tif")


@pytest.mark.parametrize(
    ("extra", "blocked", "call"),
    [
        ("grid", ("xarray",), _xarray_field),
        ("dask", ("dask",), _dask_delayed),
        ("streaming", ("zarr",), _streaming_writer),
        ("cog", ("geocloud",), _cog_field),
    ],
)
def test_missing_extra_error_names_the_distribution(
    monkeypatch: pytest.MonkeyPatch,
    extra: str,
    blocked: tuple[str, ...],
    call: Callable[[], None],
) -> None:
    _blocked(monkeypatch, *blocked)
    if extra in _MODULE_GUARDS:
        monkeypatch.setattr(_MODULE_GUARDS[extra], None)
    if extra in _REIMPORT:
        monkeypatch.delitem(sys.modules, _REIMPORT[extra], raising=False)
    with pytest.raises(ImportError, match=re.escape(f"'geotoolz-patcher[{extra}]'")):
        call()
