"""Tests for `build_xarray_catalog` + `load_xarray`. Skipped without xarray."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest


xr = pytest.importorskip("xarray")

import numpy as np

from geotoolz.catalog import build_xarray_catalog, load_xarray
from geotoolz.types import GeoSlice


@pytest.fixture
def netcdf_file(tmp_path: Path) -> Path:
    """A small (time, y, x) NetCDF with one data variable."""
    ds = xr.Dataset(
        {
            "ndvi": (
                ("time", "y", "x"),
                np.linspace(0, 1, 5 * 16 * 16, dtype=np.float32).reshape(5, 16, 16),
            )
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=5, freq="D"),
            "y": np.linspace(40.5, 40.0, 16),
            "x": np.linspace(-3.5, -3.0, 16),
        },
    )
    path = tmp_path / "modis_2024.nc"
    ds.to_netcdf(path)
    return path


class TestBuildXarrayCatalog:
    def test_one_file(self, netcdf_file: Path) -> None:
        catalog = build_xarray_catalog(
            [netcdf_file], target_crs="EPSG:4326", data_vars=["ndvi"]
        )
        assert len(catalog) == 1
        assert catalog.backend == "xarray"
        assert catalog.gdf["n_timesteps"].iloc[0] == 5


class TestLoadXarray:
    def test_returns_dataset(self, netcdf_file: Path) -> None:
        catalog = build_xarray_catalog(
            [netcdf_file], target_crs="EPSG:4326", data_vars=["ndvi"]
        )
        sl = GeoSlice(
            bounds=(-3.4, 40.1, -3.1, 40.4),
            interval=pd.Interval(
                pd.Timestamp("2024-01-01"),
                pd.Timestamp("2024-01-10"),
                closed="both",
            ),
            resolution=(0.05, 0.05),
            crs="EPSG:4326",
        )
        ds = load_xarray(catalog, sl, data_vars=["ndvi"])
        assert isinstance(ds, xr.Dataset)
        assert "ndvi" in ds.data_vars

    def test_clips_by_time_interval(self, netcdf_file: Path) -> None:
        """Regression for the bug where a short-interval query against a
        multi-day file returned every timestep instead of just the
        requested window.
        """
        catalog = build_xarray_catalog(
            [netcdf_file], target_crs="EPSG:4326", data_vars=["ndvi"]
        )
        # File covers 2024-01-01 through 2024-01-05 (5 timesteps);
        # ask for the middle 2 days only.
        sl = GeoSlice(
            bounds=(-3.4, 40.1, -3.1, 40.4),
            interval=pd.Interval(
                pd.Timestamp("2024-01-02"),
                pd.Timestamp("2024-01-03"),
                closed="both",
            ),
            resolution=(0.05, 0.05),
            crs="EPSG:4326",
        )
        ds = load_xarray(catalog, sl, data_vars=["ndvi"])
        # Two timesteps in the window, not the file's full five.
        assert ds.sizes["time"] == 2

    def test_no_crs_raises_clear_error(self, tmp_path: Path) -> None:
        """Regression for the bug where build_xarray_catalog produced
        crs_value=None and the downstream gdf-construction error was
        cryptic. Now raised in the builder with a clear message.
        """
        ds = xr.Dataset(
            {"ndvi": (("y", "x"), np.zeros((4, 4), dtype=np.float32))},
            coords={
                "y": np.linspace(40.0, 40.4, 4),
                "x": np.linspace(-3.5, -3.1, 4),
            },
        )
        path = tmp_path / "no_crs.nc"
        ds.to_netcdf(path)
        with pytest.raises(ValueError, match="target_crs"):
            build_xarray_catalog([path], target_crs=None)
