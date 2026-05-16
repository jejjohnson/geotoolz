"""Tests for geotoolz.io operators."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from georeader.geotensor import GeoTensor
from georeader.rasterio_reader import RasterioReader
from rasterio.transform import array_bounds, from_origin
from rasterio.windows import Window
from shapely.geometry import box

import geotoolz as gz
from geotoolz import io
from geotoolz.core import Identity, Sequential


def _sample_geotensor() -> GeoTensor:
    values = np.arange(2 * 4 * 5, dtype=np.int16).reshape(2, 4, 5)
    transform = from_origin(100.0, 200.0, 10.0, 10.0)
    return GeoTensor(
        values, transform=transform, crs="EPSG:32631", fill_value_default=-9999
    )


def _cog_test_geotensor() -> GeoTensor:
    values = np.arange(64 * 64, dtype=np.int16).reshape(1, 64, 64)
    transform = from_origin(100.0, 740.0, 10.0, 10.0)
    return GeoTensor(
        values, transform=transform, crs="EPSG:32631", fill_value_default=-9999
    )


def test_io_module_is_exported() -> None:
    assert gz.io is io
    assert io.ReadBounds is not None


def test_write_geotiff_then_read_bounds_roundtrips(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"

    assert io.WriteGeoTIFF(path=path)(gt) is None
    bounds = array_bounds(gt.shape[-2], gt.shape[-1], gt.transform)
    out = io.ReadBounds(src=path, bounds=bounds, crs="EPSG:32631", indexes=[2, 1])()

    np.testing.assert_array_equal(out.values, gt.values[[1, 0]])
    assert out.shape == (2, 4, 5)
    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.fill_value_default == -9999


def test_source_operator_can_start_sequential_without_input(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    out = Sequential(
        [
            io.ReadWindow(src=path, window=Window(1, 1, 2, 2), indexes=[1]),
            Identity(),
        ]
    )()

    np.testing.assert_array_equal(out.values, gt.values[:1, 1:3, 1:3])


def test_read_window_accepts_reader_source_and_rejects_indexed_objects(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)
    reader = RasterioReader(str(path))

    out = io.ReadWindow(src=reader, window=Window(0, 0, 2, 2), indexes=[2])()
    np.testing.assert_array_equal(out.values, gt.values[1:2, :2, :2])

    with pytest.raises(io.GeoToolzIOError, match="indexes are only supported"):
        io.ReadWindow(src=object(), window=Window(0, 0, 1, 1), indexes=[1])()


def test_read_window_outside_source_raises_clear_error(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    with pytest.raises(io.GeoToolzIOError, match="does not intersect"):
        io.ReadWindow(
            src=path,
            window=Window(100, 100, 2, 2),
            boundless=False,
        )()


def test_read_window_accepts_tuple_config(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    op = io.ReadWindow(src=path, window=(1, 1, 2, 2), indexes=[1])
    out = op()

    np.testing.assert_array_equal(out.values, gt.values[:1, 1:3, 1:3])
    assert op.get_config()["window"] == (1, 1, 2, 2)


def test_read_center_coords_and_polygon(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)

    centered = io.ReadCenterCoords(
        src=path,
        center=(120.0, 180.0),
        shape=(2, 2),
        crs="EPSG:32631",
        indexes=[1],
    )()
    polygon = io.ReadPolygon(
        src=path,
        polygon=box(110.0, 170.0, 130.0, 190.0),
        crs="EPSG:32631",
        indexes=[1],
    )()

    np.testing.assert_array_equal(centered.values, gt.values[:1, 1:3, 1:3])
    np.testing.assert_array_equal(polygon.values, gt.values[:1, 1:3, 1:3])


def test_reprojecting_readers_match_reference_grid(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "sample.tif"
    io.WriteGeoTIFF(path=path)(gt)
    bounds = array_bounds(gt.shape[-2], gt.shape[-1], gt.transform)

    like = io.ReadReprojectLike(src=path, like=gt, indexes=[1])()
    to_crs = io.ReadToCRS(
        src=path,
        dst_crs="EPSG:32631",
        bounds=bounds,
        resolution=(10.0, 10.0),
        indexes=[1],
    )()
    whole = io.ReadToCRS(src=path, dst_crs="EPSG:32631", indexes=[1])()

    assert like.shape == (1, 4, 5)
    assert like.transform == gt.transform
    assert like.crs == gt.crs
    assert to_crs.crs == gt.crs
    assert to_crs.shape[-2:] == gt.shape[-2:]
    assert whole.crs == gt.crs
    assert whole.shape[-2:] == gt.shape[-2:]


def test_write_cog_writes_readable_cog(tmp_path: Path) -> None:
    gt = _cog_test_geotensor()
    path = tmp_path / "sample_cog.tif"

    assert io.WriteCOG(path=path, compress="deflate")(gt) is None
    out = io.ReadBounds(
        src=path,
        bounds=array_bounds(gt.shape[-2], gt.shape[-1], gt.transform),
        crs="EPSG:32631",
    )()

    np.testing.assert_array_equal(out.values, gt.values)


def test_write_geotiff_handles_2d_data_profile_and_invalid_shapes(
    tmp_path: Path,
) -> None:
    gt = _sample_geotensor()
    two_dim = GeoTensor(
        gt.values[0],
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=None,
    )
    path = tmp_path / "two_dim.tif"

    io.WriteGeoTIFF(path=path, profile={"compress": "lzw"})(two_dim)
    out = io.ReadBounds(
        src=path,
        bounds=array_bounds(two_dim.shape[-2], two_dim.shape[-1], two_dim.transform),
        crs="EPSG:32631",
    )()
    np.testing.assert_array_equal(out.values, two_dim.values[np.newaxis, ...])

    invalid = SimpleNamespace(
        values=np.zeros((1, 1, 1, 1), dtype=np.uint8),
        crs=gt.crs,
        transform=gt.transform,
        fill_value_default=None,
    )
    with pytest.raises(io.GeoToolzIOError, match="expects 2D or 3D"):
        io.WriteGeoTIFF(path=tmp_path / "invalid.tif")(invalid)


def test_sink_operator_is_only_valid_at_end_of_sequential() -> None:
    assert io.WriteGeoTIFF(path="out.tif")._terminal is True
    with pytest.raises(TypeError, match="terminal operator"):
        Sequential([io.WriteGeoTIFF(path="out.tif"), Identity()])


def test_missing_source_raises_geotoolz_io_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.tif"

    with pytest.raises(io.GeoToolzIOError, match="Unable to read raster source"):
        io.ReadBounds(src=missing, bounds=(0.0, 0.0, 1.0, 1.0))()


def test_load_from_stac_reads_asset_href(tmp_path: Path) -> None:
    gt = _sample_geotensor()
    path = tmp_path / "asset.tif"
    io.WriteGeoTIFF(path=path)(gt)
    item = SimpleNamespace(assets={"visual": SimpleNamespace(href=str(path))})

    out = io.LoadFromSTAC(item=item, asset_key="visual")()

    np.testing.assert_array_equal(out.values, gt.values)
    assert out.transform == gt.transform


def test_operator_configs_are_serializable_for_common_values() -> None:
    polygon = box(0.0, 0.0, 1.0, 1.0)
    source_obj = object()
    ops = [
        io.ReadBounds(src="x.tif", bounds=(0.0, 0.0, 1.0, 1.0)),
        io.ReadCenterCoords(src="x.tif", center=(0.5, 0.5), shape=(2, 2)),
        io.ReadTile(src="x.tif", tile=(1, 0, 0)),
        io.ReadPolygon(src="x.tif", polygon=polygon),
        io.ReadReprojectLike(src="x.tif", like="grid"),
        io.ReadToCRS(src="x.tif", dst_crs="EPSG:4326"),
        io.WriteCOG(path="x.tif"),
        io.WriteGeoTIFF(path="x.tif"),
        io.WriteZarr(store="x.zarr", group="data", chunks={"y": 16, "x": 16}),
        io.LoadFromSTAC(item="item", asset_key="visual"),
        io.LoadFromEE(
            image_id="LANDSAT/LC08/C02/T1_L2/LC08_001001_20200101",
            bounds=(0.0, 0.0, 1.0, 1.0),
            crs="EPSG:4326",
            scale=30.0,
            bands=["B4"],
        ),
    ]

    assert (
        io.ReadBounds(src=source_obj, bounds=(0.0, 0.0, 1.0, 1.0)).get_config()["src"]
        is source_obj
    )
    for op in ops:
        cfg = op.get_config()
        assert isinstance(cfg, dict)
        assert cfg


def test_write_zarr_reports_missing_optional_dependency() -> None:
    gt = _sample_geotensor()

    with pytest.raises(io.GeoToolzIOError, match="optional zarr dependency"):
        io.WriteZarr(store="memory://out.zarr")(gt)


def test_load_from_ee_reports_missing_optional_dependencies() -> None:
    op = io.LoadFromEE(
        image_id="asset",
        bounds=(0.0, 0.0, 1.0, 1.0),
        crs="EPSG:4326",
        scale=30.0,
        bands=["B4"],
    )

    with pytest.raises(io.GeoToolzIOError, match="Earth Engine dependencies"):
        op()
