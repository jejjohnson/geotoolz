"""Behavioral tests for :mod:`xr_toolz.interpolate.operators`.

Each new Layer-1 wrapper (``Bin2D``, ``Coarsen``, ``FillNaNRBF``,
``Histogram2D``, ``PointsToGrid``, ``Refine``) must:

* produce results equivalent to its underlying L0 function on small
  synthetic inputs, and
* return a JSON-serializable ``get_config()``.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xr_toolz.interpolate import (
    Grid,
    bin_2d,
    coarsen,
    fillnan_rbf,
    histogram_2d,
    points_to_grid,
    refine,
)
from xr_toolz.interpolate.operators import (
    Bin2D,
    Coarsen,
    FillNaNRBF,
    FillNaNSpatial,
    FillNaNTemporal,
    Histogram2D,
    PointsToGrid,
    Refine,
    ResampleTime,
)


# ---------- fixtures ------------------------------------------------------


@pytest.fixture
def da_grid() -> xr.DataArray:
    rng = np.random.default_rng(0)
    lon = np.linspace(-10.0, 10.0, 11)
    lat = np.linspace(-5.0, 5.0, 7)
    return xr.DataArray(
        rng.standard_normal((lat.size, lon.size)),
        dims=("lat", "lon"),
        coords={"lon": lon, "lat": lat},
        name="x",
    )


@pytest.fixture
def da_with_nans(da_grid: xr.DataArray) -> xr.DataArray:
    da = da_grid.copy()
    da.values[2:4, 3:6] = np.nan
    return da


@pytest.fixture
def ds_daily() -> xr.Dataset:
    time = pd.date_range("2020-01-01", "2020-03-31", freq="1D")
    rng = np.random.default_rng(1)
    return xr.Dataset(
        {"x": (("time",), rng.standard_normal(time.size))},
        coords={"time": time},
    )


@pytest.fixture
def scattered_da() -> xr.DataArray:
    rng = np.random.default_rng(2)
    n = 200
    lons = rng.uniform(-10, 10, size=n)
    lats = rng.uniform(-5, 5, size=n)
    vals = rng.standard_normal(n)
    return xr.DataArray(
        vals,
        dims=("obs",),
        coords={"lon": ("obs", lons), "lat": ("obs", lats)},
    )


@pytest.fixture
def grid() -> Grid:
    return Grid.from_bounds((-10.0, 10.0), (-5.0, 5.0), resolution=2.0)


# ---------- parity --------------------------------------------------------


def test_coarsen_matches_function(da_grid: xr.DataArray) -> None:
    op = Coarsen(factor={"lon": 2}, method="mean")
    expected = coarsen(da_grid, factor={"lon": 2}, method="mean")
    xr.testing.assert_allclose(op(da_grid), expected)


def test_refine_matches_function(da_grid: xr.DataArray) -> None:
    op = Refine(factor={"lon": 2}, method="linear")
    expected = refine(da_grid.isel(lat=0), factor={"lon": 2}, method="linear")
    xr.testing.assert_allclose(op(da_grid.isel(lat=0)), expected)


def test_fillnan_rbf_matches_function(da_with_nans: xr.DataArray) -> None:
    op = FillNaNRBF(kernel="thin_plate_spline", neighbors=16)
    expected = fillnan_rbf(da_with_nans, kernel="thin_plate_spline", neighbors=16)
    xr.testing.assert_allclose(op(da_with_nans), expected)


def test_bin_2d_matches_function(scattered_da: xr.DataArray, grid: Grid) -> None:
    op = Bin2D(grid=grid, statistic="mean")
    expected = bin_2d(scattered_da, grid=grid, statistic="mean")
    xr.testing.assert_allclose(op(scattered_da), expected)


def test_histogram_2d_matches_function(scattered_da: xr.DataArray, grid: Grid) -> None:
    op = Histogram2D(grid=grid)
    expected = histogram_2d(scattered_da, grid=grid)
    xr.testing.assert_allclose(op(scattered_da), expected)


def test_points_to_grid_matches_function(grid: Grid) -> None:
    rng = np.random.default_rng(3)
    lons = rng.uniform(-10, 10, size=150)
    lats = rng.uniform(-5, 5, size=150)
    vals = rng.standard_normal(150)
    op = PointsToGrid(grid=grid, statistic="mean")
    expected = points_to_grid(lons, lats, vals, grid=grid, statistic="mean")
    xr.testing.assert_allclose(op((lons, lats, vals)), expected)


# ---------- get_config JSON-serializable ----------------------------------


@pytest.mark.parametrize(
    "op",
    [
        FillNaNSpatial(method="linear"),
        FillNaNTemporal(method="linear", max_gap=None),
        FillNaNRBF(kernel="thin_plate_spline", neighbors=8),
        ResampleTime(freq="1D", method="mean"),
        Coarsen(factor={"lon": 2}, method="mean"),
        Refine(factor={"lon": 3}, method="linear"),
        Bin2D(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        Histogram2D(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        PointsToGrid(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
    ],
    ids=lambda op: type(op).__name__,
)
def test_get_config_json_round_trips(op) -> None:
    cfg = op.get_config()
    serialized = json.dumps(cfg)
    assert json.loads(serialized) == cfg
