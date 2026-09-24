"""Build twin xarray / ``georeader.GeoTensor`` rasters for oracle tests.

The vector ↔ raster bridge in :mod:`xrtoolz.geo` mirrors georeader's
``vectorize`` / ``rasterize`` / ``GeoTensor.footprint``; the tests build
the *same* raster in both carriers and check xrtoolz against georeader.
"""

from __future__ import annotations

import numpy as np
import pytest
import rioxarray  # noqa: F401
import xarray as xr
from affine import Affine


pytest.importorskip("geopandas")
georeader_geotensor = pytest.importorskip("georeader.geotensor")

# 10 m UTM 30N grid near Madrid — deliberately *not* EPSG:4326.
CRS = "EPSG:32630"
TRANSFORM = Affine(10.0, 0.0, 440_000.0, 0.0, -10.0, 4_475_000.0)


def as_dataarray(
    values: np.ndarray,
    *,
    transform: Affine = TRANSFORM,
    crs: str | None = CRS,
    nodata: float | None = None,
) -> xr.DataArray:
    """Wrap ``values`` (``(..., y, x)``) as a georeferenced DataArray."""
    height, width = values.shape[-2:]
    x = transform.c + transform.a * (np.arange(width) + 0.5)
    y = transform.f + transform.e * (np.arange(height) + 0.5)
    dims = (*(f"d{i}" for i in range(values.ndim - 2)), "y", "x")
    da = xr.DataArray(values, dims=dims, coords={"y": y, "x": x})
    if crs is not None:
        da = da.rio.write_crs(crs)
    if nodata is not None:
        da = da.rio.write_nodata(nodata)
    return da


def as_geotensor(
    values: np.ndarray,
    *,
    transform: Affine = TRANSFORM,
    crs: str = CRS,
    nodata: float = 0,
):
    """The same raster as a ``georeader.GeoTensor``."""
    return georeader_geotensor.GeoTensor(
        values, transform=transform, crs=crs, fill_value_default=nodata
    )
