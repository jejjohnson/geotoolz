"""`load_raster` resampling / nodata and mixed-CRS `build_raster_catalog` (#217)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj
import pytest
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

from geocatalog import GeoSlice, build_raster_catalog, load_raster


REGEX = r"S2_(?P<name>[A-Za-z0-9]+)_(?P<date>\d{8})_.*\.tif"
IV = pd.Interval(pd.Timestamp("2024-05-01"), pd.Timestamp("2024-07-01"), closed="both")


def _tile(
    tmp_path: Path,
    name: str,
    crs: str = "EPSG:32629",
    center: tuple[float, float] = (-8.5, 38.0),
    *,
    dtype: str = "uint16",
) -> Path:
    x, y = pyproj.Transformer.from_crs(4326, crs, always_xy=True).transform(*center)
    bounds = (x - 100, y - 100, x + 100, y + 100)  # 20 x 20 px at 10 m
    data = (np.arange(400).reshape(1, 20, 20) * 10).astype(dtype)
    path = tmp_path / f"S2_{name}_20240601_x.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=20,
        width=20,
        count=1,
        dtype=dtype,
        crs=crs,
        transform=from_bounds(*bounds, 20, 20),
    ) as dst:
        dst.write(data)
    return path


def _upsampling_slice(path: Path) -> GeoSlice:
    with rasterio.open(path) as src:
        b = src.bounds
    return GeoSlice((b.left, b.bottom, b.right, b.top), IV, (5.0, 5.0), "EPSG:32629")


def test_nearest_is_honoured(tmp_path: Path) -> None:
    path = _tile(tmp_path, "A")
    cat = build_raster_catalog([path], filename_regex=REGEX)
    slc = _upsampling_slice(path)
    source_values = set(np.arange(400) * 10)

    nearest = np.asarray(load_raster(cat, slc, resampling=Resampling.nearest))
    assert set(np.unique(nearest)) <= source_values

    bilinear = np.asarray(load_raster(cat, slc))
    assert not set(np.unique(bilinear)) <= source_values


@pytest.mark.parametrize("bad", [-1, 70_000, 1.5, float("nan")])
def test_unrepresentable_nodata_rejected(tmp_path: Path, bad: float) -> None:
    path = _tile(tmp_path, "A")
    cat = build_raster_catalog([path], filename_regex=REGEX)
    with pytest.raises(ValueError, match="cannot be represented"):
        load_raster(cat, _upsampling_slice(path), nodata=bad)


def test_representable_nodata_accepted(tmp_path: Path) -> None:
    path = _tile(tmp_path, "A")
    cat = build_raster_catalog([path], filename_regex=REGEX)
    gt = load_raster(cat, _upsampling_slice(path), nodata=65535)
    assert gt.fill_value_default == 65535


def test_build_without_target_crs_reprojects_foreign_rows(tmp_path: Path) -> None:
    a = _tile(tmp_path, "A", "EPSG:32629", (-6.01, 38.0))
    b = _tile(tmp_path, "B", "EPSG:32630", (-5.99, 38.0))
    cat = build_raster_catalog([a, b], filename_regex=REGEX)

    assert pyproj.CRS.from_user_input(cat.gdf.crs).to_epsg() == 32629
    # B's footprint is in zone 29 coordinates now: a query around B finds it.
    hits = cat.query(bounds=(-5.9905, 37.9995, -5.9895, 38.0005), crs="EPSG:4326")
    assert [r.filepath for r in hits.iter_rows()] == [str(b)]
