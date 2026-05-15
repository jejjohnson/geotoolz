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
