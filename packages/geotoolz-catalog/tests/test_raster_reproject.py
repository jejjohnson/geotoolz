"""`load_raster` returns data on the slice grid in the slice CRS (#216).

Fixtures are gradients (value = row * width + col + base), so a wrong
window, a missing reprojection or a nodata mix-up changes the numbers,
not just the shape.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import rasterio
from rasterio.transform import from_bounds

from geocatalog import GeoSlice, build_raster_catalog, load_raster


REGEX = r"S2_(?P<name>[A-Za-z0-9]+)_(?P<date>\d{8})_.*\.tif"
IV = pd.Interval(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-07-01"), closed="both")
SIZE = 100  # pixels per side, 10 m each


def _tile(
    tmp_path: Path,
    name: str,
    crs: str,
    center: tuple[float, float],
    *,
    base: float = 0.0,
    dtype: str = "float32",
    nodata: float | None = -9999.0,
) -> Path:
    """A SIZE x SIZE gradient GeoTIFF centred on lon/lat ``center``."""
    x, y = pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform(*center)
    half = SIZE * 10 / 2
    bounds = (x - half, y - half, x + half, y + half)
    data = (np.arange(SIZE * SIZE).reshape(1, SIZE, SIZE) + base).astype(dtype)
    path = tmp_path / f"S2_{name}_20240601_x.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=SIZE,
        width=SIZE,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=from_bounds(*bounds, SIZE, SIZE),
        nodata=nodata,
    ) as dst:
        dst.write(data)
    return path


def _sample(path: Path, lon: float, lat: float) -> float:
    with rasterio.open(path) as src:
        x, y = pyproj.Transformer.from_crs(
            4326, src.crs.to_wkt(), always_xy=True
        ).transform(lon, lat)
        return float(next(src.sample([(x, y)]))[0])


def test_cross_crs_slice_returns_source_values(tmp_path: Path) -> None:
    lon, lat = -8.5, 38.0
    path = _tile(tmp_path, "A", "EPSG:32629", (lon, lat))
    cat = build_raster_catalog([path], target_crs="EPSG:4326", filename_regex=REGEX)
    slc = GeoSlice(
        (lon - 0.003, lat - 0.003, lon + 0.003, lat + 0.003),
        IV,
        (0.0001, 0.0001),
        "EPSG:4326",
    )

    gt = load_raster(cat, slc)
    values = np.asarray(gt)[0]

    assert gt.crs == pyproj.CRS.from_epsg(4326)
    assert values.shape == slc.shape
    assert not (values == gt.fill_value_default).any()
    # Pixel (30, 30)'s centre is (lon + 0.5 px, lat - 0.5 px). It matches the
    # source at that point within one source row of the gradient
    # (bilinear resampling of ~10 m pixels).
    centre = (lon + 0.00005, lat - 0.00005)
    assert abs(values[30, 30] - _sample(path, *centre)) < SIZE + 1


def test_mixed_crs_sources_mosaic(tmp_path: Path) -> None:
    """Zone 29 and zone 30 tiles either side of the 6°W boundary."""
    a = _tile(tmp_path, "A", "EPSG:32629", (-6.008, 38.0), base=0.0)
    b = _tile(tmp_path, "B", "EPSG:32630", (-5.992, 38.0), base=100_000.0)
    cat = build_raster_catalog([a, b], target_crs="EPSG:4326", filename_regex=REGEX)
    slc = GeoSlice((-6.012, 37.998, -5.988, 38.002), IV, (0.0001, 0.0001), "EPSG:4326")

    values = np.asarray(load_raster(cat, slc))[0]
    west, east = values[:, :40], values[:, -40:]
    assert ((west >= 0) & (west < 100_000)).mean() > 0.9
    assert (east >= 100_000).mean() > 0.9


def test_same_crs_path_is_exact(tmp_path: Path) -> None:
    path = _tile(tmp_path, "A", "EPSG:32629", (-8.5, 38.0))
    cat = build_raster_catalog([path], filename_regex=REGEX)
    with rasterio.open(path) as src:
        left, top = src.bounds.left, src.bounds.top
        expected = src.read(1)[10:20, 5:15]
    slc = GeoSlice(
        (left + 50, top - 200, left + 150, top - 100), IV, (10.0, 10.0), "EPSG:32629"
    )
    values = np.asarray(load_raster(cat, slc))[0]
    np.testing.assert_array_equal(values, expected)


def test_fill_value_default_is_the_nodata_used(tmp_path: Path) -> None:
    path = _tile(tmp_path, "A", "EPSG:32629", (-8.5, 38.0), nodata=-9999.0)
    cat = build_raster_catalog([path], filename_regex=REGEX)
    with rasterio.open(path) as src:
        b = src.bounds
    # Half the slice lies outside the tile.
    slc = GeoSlice(
        (b.left - 500, b.bottom, b.left + 500, b.top), IV, (10.0, 10.0), "EPSG:32629"
    )
    gt = load_raster(cat, slc)
    values = np.asarray(gt)[0]
    assert gt.fill_value_default == -9999.0
    assert (values[:, :50] == -9999.0).all()
    assert (values[:, 50:] != -9999.0).all()


def test_mixed_dtypes_promote(tmp_path: Path) -> None:
    a = _tile(tmp_path, "A", "EPSG:32629", (-8.5, 38.0), dtype="uint8", nodata=0)
    b = _tile(tmp_path, "B", "EPSG:32629", (-8.5, 38.0), base=0.5, dtype="float32")
    cat = build_raster_catalog([a, b], filename_regex=REGEX)
    with rasterio.open(a) as src:
        bb = src.bounds
    slc = GeoSlice(tuple(bb), IV, (10.0, 10.0), "EPSG:32629")
    assert np.asarray(load_raster(cat, slc)).dtype == np.float32
