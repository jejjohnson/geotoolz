"""The public facades agree with each other and with what they define.

Every public module declares ``__all__``; every name in it resolves; the
names a module defines itself (functions, classes) are all exported; and
the extras-gated Field adapters are listed — and lazily resolvable — the
same way at the root, in ``geopatcher.fields`` and in the private package.
"""

from __future__ import annotations

import importlib
import inspect

import pytest

import geopatcher
from geopatcher._src import fields as private_fields


PUBLIC_MODULES = [
    "geopatcher",
    "geopatcher.dask",
    "geopatcher.fields",
    "geopatcher.hooks",
    "geopatcher.jax",
    "geopatcher.matched",
    "geopatcher.objstore",
    "geopatcher.runners",
    "geopatcher.spatial",
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


def test_lazy_adapters_listed_everywhere() -> None:
    lazy = set(private_fields.LAZY_ADAPTERS)
    assert len(lazy) == 6
    for module in (geopatcher, geopatcher.fields, private_fields):
        assert lazy <= set(module.__all__), module.__name__
        for name in lazy:
            obj = getattr(module, name)
            assert obj is getattr(private_fields, name)
            assert obj.__module__ == private_fields.LAZY_ADAPTERS[name]


def test_fields_facade_matches_private_package() -> None:
    assert set(geopatcher.fields.__all__) == set(private_fields.__all__)
    # Was an AttributeError: the facade skipped it (#205).
    assert geopatcher.fields.ReprojectingRasterField is (
        geopatcher.ReprojectingRasterField
    )


def test_unknown_attribute_is_an_attribute_error() -> None:
    for module in (geopatcher, geopatcher.fields, private_fields):
        with pytest.raises(AttributeError, match="no attribute 'Nope'"):
            module.Nope  # noqa: B018


def test_hooks_and_runners_export_what_their_callers_receive() -> None:
    from geopatcher import hooks, runners
    from geopatcher._src.hooks import UNKNOWN_TOTAL
    from geopatcher._src.prefetch import prefetch_iterable

    assert hooks.UNKNOWN_TOTAL == UNKNOWN_TOTAL
    assert runners.prefetch_iterable is prefetch_iterable


def test_star_import_needs_no_extra() -> None:
    namespace: dict[str, object] = {}
    exec("from geopatcher import *", namespace)
    assert "XarrayField" in namespace and "SpatialPatcher" in namespace
