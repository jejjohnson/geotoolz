"""Behavioral tests for :mod:`xrtoolz.interpolate.operators`.

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

from xrtoolz.interpolate import (
    Grid,
    bin_2d,
    coarsen,
    fillnan_climatology,
    fillnan_idw,
    fillnan_rbf,
    histogram_2d,
    idw_to_grid,
    idw_to_points,
    points_to_grid,
    refine,
)
from xrtoolz.interpolate.operators import (
    Bin2D,
    Coarsen,
    FillNaNClimatology,
    FillNaNIDW,
    FillNaNRBF,
    FillNaNSpatial,
    FillNaNTemporal,
    Histogram2D,
    IDWToGrid,
    IDWToPoints,
    KDEToGrid,
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


def test_fillnan_climatology_operator_matches_function() -> None:
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    da = xr.DataArray(
        np.tile(np.arange(12.0), 2),
        dims="time",
        coords={"time": time},
    )
    missing = da.copy()
    missing.loc[{"time": "2001-03-01"}] = np.nan
    op = FillNaNClimatology(group="month", residual="zero", min_count=1)

    expected = fillnan_climatology(missing, group="month", residual="zero", min_count=1)

    xr.testing.assert_allclose(op(missing), expected)


def test_resample_time_operator_passes_interp_method() -> None:
    time = pd.date_range("2020-01-01", periods=3, freq="1D")
    ds = xr.Dataset({"x": ("time", [0.0, 24.0, 48.0])}, coords={"time": time})

    linear = ResampleTime(freq="12h", method="interpolate", interp_method="linear")(ds)
    nearest = ResampleTime(freq="12h", method="interpolate", interp_method="nearest")(
        ds
    )

    assert float(linear["x"].sel(time="2020-01-01T12:00")) == pytest.approx(12.0)
    assert float(nearest["x"].sel(time="2020-01-01T12:00")) == pytest.approx(0.0)


def test_fillnan_idw_matches_function(da_with_nans: xr.DataArray) -> None:
    op = FillNaNIDW(k=6, power=1.5)
    expected = fillnan_idw(da_with_nans, k=6, power=1.5)
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


def test_idw_to_grid_matches_function(grid: Grid) -> None:
    rng = np.random.default_rng(4)
    lons = rng.uniform(-10, 10, size=150)
    lats = rng.uniform(-5, 5, size=150)
    vals = rng.standard_normal(150)
    op = IDWToGrid(grid=grid, k=5, power=1.0)
    expected = idw_to_grid(lons, lats, vals, grid=grid, k=5, power=1.0)
    xr.testing.assert_allclose(op((lons, lats, vals)), expected)


def test_idw_to_points_matches_function() -> None:
    rng = np.random.default_rng(5)
    lons = rng.uniform(-10, 10, size=150)
    lats = rng.uniform(-5, 5, size=150)
    vals = rng.standard_normal(150)
    dst_lons = np.array([-1.0, 0.0, 1.0])
    dst_lats = np.array([0.0, 1.0, 2.0])
    op = IDWToPoints(dst_lons, dst_lats, k=5, power=1.0)
    expected = idw_to_points(lons, lats, vals, dst_lons, dst_lats, k=5, power=1.0)
    np.testing.assert_allclose(op((lons, lats, vals)), expected)


# ---------- get_config JSON-serializable ----------------------------------


@pytest.mark.parametrize(
    "op",
    [
        FillNaNSpatial(method="linear"),
        FillNaNTemporal(method="linear", max_gap=None),
        FillNaNClimatology(group="month", residual="linear", min_count=2),
        FillNaNRBF(kernel="thin_plate_spline", neighbors=8),
        FillNaNIDW(k=4),
        ResampleTime(freq="1D", method="interpolate", interp_method="nearest"),
        Coarsen(factor={"lon": 2}, method="mean"),
        Refine(factor={"lon": 3}, method="linear"),
        Bin2D(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        Histogram2D(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        PointsToGrid(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        IDWToGrid(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
        IDWToPoints(np.array([0.0]), np.array([0.0])),
        KDEToGrid(grid=Grid.from_bounds((0, 1), (0, 1), 0.5)),
    ],
    ids=lambda op: type(op).__name__,
)
def test_get_config_json_round_trips(op) -> None:
    cfg = op.get_config()
    serialized = json.dumps(cfg)
    assert json.loads(serialized) == cfg
