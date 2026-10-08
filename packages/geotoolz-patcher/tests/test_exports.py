"""The public namespaces: one home per name, organised by task.

Every public module declares ``__all__``; every name in it resolves; the
names a module defines itself are all exported; each public object has
exactly one public home; the old flat / prefixed spellings are gone; and
the extras-gated Field adapters resolve lazily from `geopatcher.fields`.
"""

from __future__ import annotations

import importlib
import inspect
import types

import pytest

import geopatcher
from geopatcher._src import fields as private_fields


PUBLIC_MODULES = [
    "geopatcher",
    "geopatcher.config",
    "geopatcher.fields",
    "geopatcher.matched",
    "geopatcher.observe",
    "geopatcher.run",
    "geopatcher.spatial",
    "geopatcher.spatial.aggregation",
    "geopatcher.spatial.geometry",
    "geopatcher.spatial.sampler",
    "geopatcher.spatial.window",
    "geopatcher.temporal",
    "geopatcher.temporal.aggregation",
    "geopatcher.temporal.geometry",
    "geopatcher.temporal.sampler",
    "geopatcher.temporal.stencils",
    "geopatcher.temporal.window",
]

# Modules and names that moved, with no alias left behind.
REMOVED_MODULES = [
    "geopatcher.cog",
    "geopatcher.dask",
    "geopatcher.hooks",
    "geopatcher.jax",
    "geopatcher.objstore",
    "geopatcher.runners",
    "geopatcher.time",
]


@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_facades_consistent(name: str) -> None:
    module = importlib.import_module(name)
    exported = getattr(module, "__all__", None)
    assert exported is not None, f"{name} has no __all__"
    assert len(exported) == len(set(exported)), f"{name}.__all__ has duplicates"
    missing = [n for n in exported if not hasattr(module, n)]
    assert not missing, f"{name}.__all__ names that do not resolve: {missing}"
    defined_here = {
        attr
        for attr, obj in vars(module).items()
        if not attr.startswith("_")
        and (inspect.isfunction(obj) or inspect.isclass(obj))
        and obj.__module__ == name
    }
    unexported = sorted(defined_here - set(exported))
    assert not unexported, f"{name} defines public names outside __all__: {unexported}"


def test_one_home_per_public_name() -> None:
    """No object is exported from two public modules (submodules aside)."""
    homes: dict[int, list[str]] = {}
    for name in PUBLIC_MODULES:
        module = importlib.import_module(name)
        for attr in module.__all__:
            obj = getattr(module, attr)
            if isinstance(obj, types.ModuleType) or attr == "__version__":
                continue
            homes.setdefault(id(obj), []).append(f"{name}.{attr}")
    shared = sorted(paths for paths in homes.values() if len(paths) > 1)
    assert not shared, f"names exported from more than one module: {shared}"


def test_root_is_the_core_surface() -> None:
    names = {n for n in geopatcher.__all__ if not n.startswith("_")}
    assert names == {
        "AsyncField",
        "AsyncSpatialPatcher",
        "Domain",
        "Field",
        "Patch",
        "RasterField",
        "SpatialPatcher",
        "SpatioTemporalPatch",
        "SpatioTemporalPatcher",
        "TemporalPatch",
        "TemporalPatcher",
        "config",
        "fields",
        "matched",
        "observe",
        "run",
        "spatial",
        "temporal",
    }


def test_axes_drop_their_family_prefix() -> None:
    for family in (geopatcher.spatial, geopatcher.temporal):
        for axis in family.__all__:
            for name in getattr(family, axis).__all__:
                assert not name.startswith(("Spatial", "Temporal")), (
                    f"{family.__name__}.{axis}.{name}"
                )
    assert (
        geopatcher.spatial.aggregation.Mean is not geopatcher.temporal.aggregation.Mean
    )


@pytest.mark.parametrize("name", REMOVED_MODULES)
def test_moved_modules_are_gone(name: str) -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(name)


def test_old_root_names_are_gone() -> None:
    for old in ("SpatialHann", "TemporalMean", "PatchCache", "ObstoreCogField"):
        assert not hasattr(geopatcher, old), old
    # Renamed so it no longer collides with geocatalog's grid helper.
    assert not hasattr(geopatcher.temporal.stencils, "divide_evenly")


def test_lazy_adapters_resolve_from_fields() -> None:
    lazy = set(private_fields.LAZY_ADAPTERS)
    assert lazy == {
        "CogField",
        "DaskField",
        "GeoPandasField",
        "RioXarrayField",
        "XarrayField",
        "XvecField",
    }
    assert lazy <= set(geopatcher.fields.__all__)
    for name in lazy:
        obj = getattr(geopatcher.fields, name)
        assert obj is getattr(private_fields, name)
        assert obj.__module__ == private_fields.LAZY_ADAPTERS[name]


def test_unknown_attribute_is_an_attribute_error() -> None:
    for module in (geopatcher.fields, private_fields):
        with pytest.raises(AttributeError, match="no attribute 'Nope'"):
            module.Nope  # noqa: B018


def test_star_import_needs_no_extra() -> None:
    namespace: dict[str, object] = {}
    exec("from geopatcher.fields import *", namespace)
    assert "XarrayField" in namespace and "CogField" in namespace
    namespace = {}
    exec("from geopatcher import *", namespace)
    assert "SpatialPatcher" in namespace and "spatial" in namespace
