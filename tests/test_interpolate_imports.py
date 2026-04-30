"""Import-surface tests for :mod:`xr_toolz.interpolate` (Epic F3 / D12).

Pins the canonical re-export surface, the deep-import paths into
``_src.<sub>``, identity equivalence between the two, and verifies that
the legacy ``xr_toolz.geo`` re-exports for these names are gone (no
deprecation shim — the package is pre-1.0 and has no external users).
"""

from __future__ import annotations

import importlib

import pytest


CANONICAL_FUNCS = (
    ("Grid", "binning"),
    ("Period", "binning"),
    ("SpaceTimeGrid", "binning"),
    ("bin_2d", "binning"),
    ("histogram_2d", "binning"),
    ("fillnan_spatial", "gap_fill"),
    ("fillnan_temporal", "gap_fill"),
    ("fillnan_rbf", "gap_fill"),
    ("coarsen", "grid_to_grid"),
    ("refine", "grid_to_grid"),
    ("resample_time", "resample"),
    ("points_to_grid", "points_to_grid"),
)

CANONICAL_OPS = (
    "Bin2D",
    "Coarsen",
    "FillNaNRBF",
    "FillNaNSpatial",
    "FillNaNTemporal",
    "Histogram2D",
    "PointsToGrid",
    "Refine",
    "ResampleTime",
)

REMOVED_FROM_GEO = (
    "Grid",
    "Period",
    "SpaceTimeGrid",
    "bin_2d",
    "histogram_2d",
    "points_to_grid",
    "coarsen",
    "refine",
    "fillnan_spatial",
    "fillnan_temporal",
    "fillnan_rbf",
    "resample_time",
)


@pytest.mark.parametrize("name,submod", CANONICAL_FUNCS)
def test_canonical_function_surface(name: str, submod: str) -> None:
    package = importlib.import_module("xr_toolz.interpolate")
    deep = importlib.import_module(f"xr_toolz.interpolate._src.{submod}")
    assert hasattr(package, name)
    assert hasattr(deep, name)
    assert getattr(package, name) is getattr(deep, name)
    assert name in package.__all__


@pytest.mark.parametrize("name", CANONICAL_OPS)
def test_canonical_operator_surface(name: str) -> None:
    ops = importlib.import_module("xr_toolz.interpolate.operators")
    assert hasattr(ops, name)
    assert name in ops.__all__


@pytest.mark.parametrize("name", REMOVED_FROM_GEO)
def test_legacy_geo_names_are_gone(name: str) -> None:
    geo = importlib.import_module("xr_toolz.geo")
    with pytest.raises(AttributeError):
        getattr(geo, name)


def test_legacy_geo_operator_names_are_gone() -> None:
    ops = importlib.import_module("xr_toolz.geo.operators")
    for name in ("FillNaNSpatial", "FillNaNTemporal", "ResampleTime"):
        with pytest.raises(AttributeError):
            getattr(ops, name)


def test_placeholder_submodules_importable() -> None:
    for sub in ("coord_remap", "smooth", "downscale", "grid_to_points"):
        mod = importlib.import_module(f"xr_toolz.interpolate._src.{sub}")
        assert mod.__all__ == []
