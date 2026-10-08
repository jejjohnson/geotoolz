"""Packaging contract: every optional extra installs what geocatalog uses.

Each distribution an extra installs must be imported somewhere under
``src/geocatalog`` (lazily is fine — that is what extras are for) or be a
runtime plug-in a library we import loads by name (fsspec filesystems,
xarray engines). ``[full]`` must cover every other extra, and the
``geotoolz-patcher`` bridge is version-bounded rather than an unpinned or
direct-URL reference.
"""

from __future__ import annotations

import ast
import re
import tomllib
from functools import cache
from pathlib import Path

from packaging.requirements import Requirement


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC = PACKAGE_ROOT / "src" / "geocatalog"
PYPROJECT = PACKAGE_ROOT / "pyproject.toml"
SELF = "geotoolz-catalog"

IMPORT_TO_DIST = {
    "ee": "earthengine-api",
    "geocloud": "geotoolz-cloud",
    "geopatcher": "geotoolz-patcher",
    "planetary_computer": "planetary-computer",
    "pystac_client": "pystac-client",
}

# Distributions no geocatalog module imports by name, but that a library it
# does import loads at runtime. Keep this small and say why.
RUNTIME_PLUGINS = {
    "adlfs": "fsspec az:// / abfs:// filesystem",
    "gcsfs": "fsspec gs:// filesystem",
    "s3fs": "fsspec s3:// filesystem",
    "huggingface-hub": "fsspec hf:// filesystem",
    "zarr": "xarray's zarr engine for .zarr stores",
}


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


@cache
def _extras() -> dict[str, list[Requirement]]:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    return {
        extra: [Requirement(r) for r in reqs]
        for extra, reqs in project["optional-dependencies"].items()
    }


def _expand(extra: str) -> set[str]:
    """Distributions ``extra`` installs, following self-references."""
    dists: set[str] = set()
    for req in _extras()[extra]:
        if _normalize(req.name) == SELF:
            for sub in req.extras:
                dists |= _expand(sub)
        else:
            dists.add(_normalize(req.name))
    return dists


@cache
def _used_distributions() -> set[str]:
    """Distributions imported (anywhere, lazily included) by geocatalog."""
    roots: set[str] = set()
    for path in SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                roots.add(node.module.split(".")[0])
            elif (
                isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "require_extra"
                and node.args
                and isinstance(node.args[0], ast.Constant)
            ):
                roots.add(str(node.args[0].value).split(".")[0])
    return {_normalize(IMPORT_TO_DIST.get(root, root)) for root in roots}


def test_scanner_sees_lazy_extra_imports() -> None:
    used = _used_distributions()
    for dist in ("duckdb", "pystac-client", "earthengine-api", "geotoolz-patcher"):
        assert dist in used, dist


def test_every_extra_dependency_is_used() -> None:
    used = _used_distributions() | set(RUNTIME_PLUGINS)
    unused = {
        extra: sorted(_expand(extra) - used)
        for extra in _extras()
        if _expand(extra) - used
    }
    assert not unused, f"extras install packages geocatalog never uses: {unused}"


def test_runtime_plugins_are_still_declared() -> None:
    declared = set().union(*(_expand(extra) for extra in _extras()))
    assert set(RUNTIME_PLUGINS) <= declared


def test_full_covers_every_extra() -> None:
    full = _expand("full")
    for extra in _extras():
        missing = _expand(extra) - full
        assert not missing, f"[full] lacks {sorted(missing)} from [{extra}]"


def test_patcher_requirement_is_version_bounded() -> None:
    for extra, reqs in _extras().items():
        for req in reqs:
            if _normalize(req.name) == "geotoolz-patcher":
                assert req.url is None, extra
                assert req.specifier, f"[{extra}] pins no geotoolz-patcher version"


def test_no_direct_references_allowed() -> None:
    hatch = tomllib.loads(PYPROJECT.read_text(encoding="utf-8")).get("tool", {})
    metadata = hatch.get("hatch", {}).get("metadata", {})
    assert not metadata.get("allow-direct-references", False)
