"""Geometry operator tests."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from shapely.geometry import box

import geotoolz as gz


def _gt(values: np.ndarray | None = None) -> GeoTensor:
    if values is None:
        values = np.arange(1 * 5 * 7, dtype=np.float32).reshape(1, 5, 7)
    return GeoTensor(
        values,
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )


def test_geom_module_is_public() -> None:
    assert gz.geom.Tile is gz.geom.Tile


def test_pad_to_then_crop_to_recovers_original_values() -> None:
    gt = _gt()

    padded = gz.geom.PadTo(shape=(9, 11), fill=-9999)(gt)
    cropped = gz.geom.CropTo(shape=gt.shape[-2:])(padded)

    assert cropped.shape == gt.shape
    np.testing.assert_array_equal(np.asarray(cropped), np.asarray(gt))
    assert cropped.same_extent(gt)


def test_tile_then_stitch_reconstructs_original_without_overlap() -> None:
    gt = _gt()

    tiles = gz.geom.Tile(size=(3, 4))(gt)
    stitched = gz.geom.Stitch(blend="average")(tiles)

    assert stitched.shape == gt.shape
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))
    assert stitched.same_extent(gt)


def test_reproject_like_matches_reference_grid() -> None:
    gt = _gt()
    like = GeoTensor(
        np.zeros((1, 3, 4), dtype=np.float32),
        transform=Affine(2, 0, 10, 0, -2, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )

    out = gz.geom.ReprojectLike(like=like, resampling="nearest")(gt)

    assert out.same_extent(like)


def test_resize_resample_and_crop_to_bounds() -> None:
    gt = _gt()

    resized = gz.geom.Resize(shape=(10, 14), anti_aliasing=False)(gt)
    resampled = gz.geom.Resample(resolution=(2, 2), anti_aliasing=False)(gt)
    cropped = gz.geom.CropToBounds(bounds=(12, 16, 15, 19), crs="EPSG:4326")(gt)

    assert resized.shape == (1, 10, 14)
    assert resampled.shape == (1, 2, 4)
    assert cropped.shape == (1, 3, 3)
    assert gz.geom.Resize(shape=(1, 1)).get_config()["shape"] == (1, 1)


def test_crop_to_validates_target_and_anchor() -> None:
    gt = _gt()

    upper_left = gz.geom.CropTo(shape=(2, 3), anchor="upper_left")(gt)

    np.testing.assert_array_equal(np.asarray(upper_left), np.asarray(gt)[..., :2, :3])
    with pytest.raises(ValueError, match="Cannot crop"):
        gz.geom.CropTo(shape=(6, 7))(gt)
    with pytest.raises(ValueError, match="anchor"):
        gz.geom.CropTo(shape=(2, 2), anchor="lower_right")(gt)


def test_sliding_window_and_tile_validate_stride() -> None:
    gt = _gt()

    tiles = gz.geom.SlidingWindow(size=(3, 3), overlap=1)(gt)

    assert tiles
    assert gz.geom.SlidingWindow(size=(3, 3), overlap=1).get_config() == {
        "size": (3, 3),
        "overlap": 1,
    }
    with pytest.raises(ValueError, match="stride"):
        gz.geom.Tile(size=(3, 3), stride=(4, 3))(gt)


def test_rasterize_and_vectorize_round_trip_geometry() -> None:
    gt = _gt(np.zeros((5, 7), dtype=np.uint8))
    geometry = box(12, 16, 15, 19)

    mask = gz.geom.Rasterize(geometries=[geometry], fill=0)(gt)
    polygons = gz.geom.Vectorize(min_area=0.5, simplify_tolerance=0.0)(mask)

    assert np.asarray(mask).sum() > 0
    assert polygons
    assert sum(polygon.area for polygon in polygons) == pytest.approx(geometry.area)


def test_rasterize_like_accepts_geodataframe() -> None:
    like = _gt(np.zeros((5, 7), dtype=np.uint8))
    dataframe = gpd.GeoDataFrame(
        {"value": [3]},
        geometry=[box(12, 16, 15, 19)],
        crs="EPSG:4326",
    )

    burned_values = gz.geom.RasterizeLike(
        like=like,
        geometries=dataframe,
        column="value",
    )()
    burned_default = gz.geom.RasterizeLike(like=like, geometries=dataframe)()

    assert np.asarray(burned_values).sum() == 27
    assert np.asarray(burned_default).sum() == 9


def test_stitch_blend_modes_and_errors() -> None:
    gt = _gt()
    tiles = gz.geom.Tile(size=(3, 4))(gt)

    for blend in ("first", "max", "feather"):
        stitched = gz.geom.Stitch(blend=blend, feather_width=1)(tiles)
        np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))

    with pytest.raises(ValueError, match="at least one"):
        gz.geom.Stitch()([])
    with pytest.raises(ValueError, match="blend"):
        gz.geom.Stitch(blend="unknown")(tiles)
    rotated = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(1, 0.1, 0, 0, -1, 2),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="north-up"):
        gz.geom.Stitch()([rotated])


def test_mosaic_methods_cover_adjacent_tiles() -> None:
    left = GeoTensor(
        np.ones((1, 3, 4), dtype=np.float32),
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )
    right = GeoTensor(
        np.full((1, 3, 4), 2, dtype=np.float32),
        transform=Affine(1, 0, 14, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )

    for method in ("first", "mean", "median", "max", "min"):
        out = gz.geom.Mosaic(method=method, resampling="nearest")([left, right])
        assert out.shape == (1, 3, 8)
        assert np.asarray(out).min() == 1
        assert np.asarray(out).max() == 2

    with pytest.raises(ValueError, match="method"):
        gz.geom.Mosaic(method="unknown")([left, right])
