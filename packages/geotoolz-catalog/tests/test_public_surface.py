"""The public namespaces: a small root, one home per name, organised by task.

Every public module declares a sorted, unique ``__all__``; every name in
it resolves; each public object has exactly one public home; the old
flat / thematic spellings are gone; and the extras-gated names resolve
lazily, so no namespace needs an optional dependency to import.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import types

import pytest

import geocatalog
from geocatalog._src._lazy import LAZY


NAMESPACES = (
    "backends",
    "build",
    "grid",
    "load",
    "matchup",
    "patch",
    "sources",
    "staging",
    "storage",
    "utils",
)
PUBLIC_MODULES = ("geocatalog", *(f"geocatalog.{n}" for n in NAMESPACES))
REMOVED_MODULES = (
    "geocatalog.bundle",
    "geocatalog.catalog",
    "geocatalog.io",
    "geocatalog.types",
)
OPTIONAL = (
    "duckdb", "xarray", "rioxarray", "pystac", "pystac_client", "planetary_computer",
    "earthaccess", "ee", "fsspec", "geopatcher", "zarr",
)  # fmt: skip


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_all_is_unique_sorted_and_resolves(name: str) -> None:
    module = importlib.import_module(name)
    exported = list(module.__all__)
    assert len(exported) == len(set(exported))
    # ruff's RUF022 order: SCREAMING_CASE, then CamelCase, then the rest.
    key = [(not n.isupper(), not n[:1].isupper(), n) for n in exported]
    assert key == sorted(key)
    assert [n for n in exported if not hasattr(module, n)] == []


def test_root_is_the_core_surface() -> None:
    assert set(geocatalog.__all__) == {
        "GeoCatalog",
        "GeoSlice",
        "__version__",
        "intersect",
        "open_catalog",
        "query",
        "union",
        *NAMESPACES,
    }


def test_one_home_per_public_name() -> None:
    homes: dict[int, list[str]] = {}
    for name in PUBLIC_MODULES:
        module = importlib.import_module(name)
        for attr in module.__all__:
            obj = getattr(module, attr)
            if isinstance(obj, types.ModuleType) or attr == "__version__":
                continue
            homes.setdefault(id(obj), []).append(f"{name}.{attr}")
    shared = sorted(paths for paths in homes.values() if len(paths) > 1)
    assert shared == [], f"names exported from more than one module: {shared}"


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_old_thematic_modules_are_gone(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_old_flat_names_are_gone() -> None:
    for old in (
        "InMemoryGeoCatalog",
        "load_raster",
        "to_geoparquet",
        "stage",
        "MatchupRow",
    ):
        assert not hasattr(geocatalog, old), old
    # Renamed so it no longer collides with geopatcher's stencil helper.
    assert not hasattr(geocatalog.grid, "divide_evenly")


def test_lazy_names_resolve_from_their_home() -> None:
    homes = {
        name: module
        for module in PUBLIC_MODULES[1:]
        for name in importlib.import_module(module).__all__
    }
    for name, defining in LAZY.items():
        obj = getattr(importlib.import_module(homes[name]), name)
        assert obj is getattr(importlib.import_module(defining), name), name


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_unknown_attribute_names_the_module(name: str) -> None:
    module = importlib.import_module(name)
    with pytest.raises(AttributeError, match=rf"module '{name}' has no attribute"):
        _ = module.no_such_name  # type: ignore[attr-defined]


def test_matchup_is_an_ordinary_module() -> None:
    from geocatalog._src.matchup import matchup as function

    assert isinstance(geocatalog.matchup, types.ModuleType)
    assert not callable(geocatalog.matchup)
    assert geocatalog.matchup.matchup is function


def test_star_imports_work_with_every_extra_missing() -> None:
    script = (
        "import sys\n"
        f"for m in {OPTIONAL!r}:\n"
        "    sys.modules[m] = None\n"
        f"for name in {PUBLIC_MODULES!r}:\n"
        "    ns = {}\n"
        "    exec(f'from {name} import *', ns)\n"
        "    module = sys.modules[name]\n"
        "    missing = [n for n in module.__all__ if n not in ns]\n"
        "    assert not missing, (name, missing)\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr


def test_helpers_have_public_paths() -> None:
    from geocatalog.storage import StreamingParquetWriter, sort_geoparquet
    from geocatalog.utils import parse_uri, retry_transient_io, to_utc_ts

    assert StreamingParquetWriter.__module__ == "geocatalog._src.storage.streaming"
    assert callable(sort_geoparquet) and callable(retry_transient_io)
    assert parse_uri("s3://b/k").is_remote
    assert to_utc_ts("2024-01-01").tzinfo is not None
