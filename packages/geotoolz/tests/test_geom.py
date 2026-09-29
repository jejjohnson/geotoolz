"""Geometry operator tests.

Three tiers:

- **Tier-A**: pure-numpy primitives in
  :mod:`geotoolz.geom._src.array` (feather kernel, slicing math,
  validity mask, resampling resolution).
- **Tier-B**: ``GeoTensor`` round-trip checks for every Operator,
  with transform + CRS propagation asserted explicitly.
- **Tier-C**: hydra-zen ``builds()`` round-trip for the YAML-safe
  Operators.
"""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
from affine import Affine
from georeader.geotensor import GeoTensor
from rasterio.enums import Resampling
from shapely.geometry import box

import geotoolz as gz
from geotoolz._src.blending import triangular_weights
from geotoolz.geom._src import array as geom_array


def _gt(values: np.ndarray | None = None) -> GeoTensor:
    if values is None:
        values = np.arange(1 * 5 * 7, dtype=np.float32).reshape(1, 5, 7)
    return GeoTensor(
        values,
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )


# ----------------------------------------------------------------------------
# Tier-A: pure-numpy primitives
# ----------------------------------------------------------------------------


def test_array_feather_weights_ramps_from_centre_to_edge() -> None:
    weights = geom_array.feather_weights((5, 5), width=2)
    assert weights.dtype == np.float32
    # Centre pixel hits the saturated plateau (>= 1).
    assert weights[2, 2] == pytest.approx(1.0)
    # Corner pixel is ramped down.
    assert weights[0, 0] < weights[2, 2]
    # Symmetry along both axes.
    np.testing.assert_array_equal(weights, weights.T)


def test_array_feather_weights_zero_width_returns_ones() -> None:
    weights = geom_array.feather_weights((3, 4), width=0)
    np.testing.assert_array_equal(weights, np.ones((3, 4), dtype=np.float32))


def test_array_feather_weights_wrap_shared_triangular_kernel() -> None:
    np.testing.assert_array_equal(
        geom_array.feather_weights((5, 7), width=2),
        triangular_weights((5, 7), width=2),
    )


def test_array_resolve_resampling_handles_aliases_and_enum() -> None:
    assert geom_array.resolve_resampling("linear") == Resampling.bilinear
    assert geom_array.resolve_resampling("bicubic") == Resampling.cubic
    assert geom_array.resolve_resampling("nearest") == Resampling.nearest
    # Passthrough for enum members.
    assert geom_array.resolve_resampling(Resampling.mode) == Resampling.mode


def test_array_resolve_interpolation_maps_cubic_alias() -> None:
    assert geom_array.resolve_interpolation("cubic") == "bicubic"
    assert geom_array.resolve_interpolation("cubic_spline") == "bicubic"
    assert geom_array.resolve_interpolation("average") == "bilinear"
    # Passthrough for names that map identity.
    assert geom_array.resolve_interpolation("nearest") == "nearest"


def test_array_center_offsets_and_north_up() -> None:
    assert geom_array.center_offsets((10, 10), (4, 4)) == (3, 3)
    assert geom_array.center_offsets((5, 5), (5, 5)) == (0, 0)
    assert geom_array.is_north_up(Affine(1, 0, 0, 0, -1, 0)) is True
    assert geom_array.is_north_up(Affine(1, 0.1, 0, 0, -1, 0)) is False
    # South-up (``e > 0``) and east-to-west (``a < 0``) grids are flipped.
    assert geom_array.is_north_up(Affine(1, 0, 0, 0, 1, 0)) is False
    assert geom_array.is_north_up(Affine(-1, 0, 0, 0, -1, 0)) is False


def test_array_valid_pixel_mask_handles_dtypes_and_collapse() -> None:
    arr = np.array([[1.0, -9999.0], [3.0, 4.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        geom_array.valid_pixel_mask(arr, -9999),
        np.array([[True, False], [True, True]]),
    )
    # NaN sentinel path.
    nan_arr = np.array([[np.nan, 1.0]], dtype=np.float32)
    np.testing.assert_array_equal(
        geom_array.valid_pixel_mask(nan_arr, float("nan")),
        np.array([[False, True]]),
    )
    # None short-circuits to all-True.
    np.testing.assert_array_equal(
        geom_array.valid_pixel_mask(arr, None),
        np.ones_like(arr, dtype=bool),
    )
    # Multi-band collapse along the leading axis.
    multi = np.stack([arr, arr.copy()], axis=0)
    np.testing.assert_array_equal(
        geom_array.valid_pixel_mask(multi, -9999),
        np.array([[True, False], [True, True]]),
    )


def test_array_target_slices_clips_tile_outside_target() -> None:
    target = Affine(1, 0, 0, 0, -1, 0)
    tile = Affine(1, 0, 4, 0, -1, -4)  # tile lower-right corner
    out_slc, tile_slc = geom_array.target_slices(tile, (4, 4), target, (8, 8))
    assert out_slc == (slice(4, 8), slice(4, 8))
    assert tile_slc == (slice(0, 4), slice(0, 4))


def test_array_target_slices_empty_when_tile_fully_past_target() -> None:
    """Tiles fully off the target grid must return empty (non-negative) slices.

    Reproduces the bug where ``target_slices`` could return slices with
    negative starts/stops when a tile lies entirely outside the target
    window (e.g. when ``Stitch`` is given a custom ``target_shape``
    smaller than the union of tile bounds). Downstream blending would
    then write into wrapped-around indices.
    """
    target = Affine(1, 0, 0, 0, -1, 0)
    # Tile fully past the bottom-right of an 8x8 target.
    past_br = Affine(1, 0, 10, 0, -1, -10)
    out_slc, tile_slc = geom_array.target_slices(past_br, (4, 4), target, (8, 8))
    assert out_slc == (slice(0, 0), slice(0, 0))
    assert tile_slc == (slice(0, 0), slice(0, 0))
    # Tile fully past the top-left (negative row/col offset).
    past_tl = Affine(1, 0, -10, 0, -1, 10)
    out_slc, tile_slc = geom_array.target_slices(past_tl, (4, 4), target, (8, 8))
    assert out_slc == (slice(0, 0), slice(0, 0))
    assert tile_slc == (slice(0, 0), slice(0, 0))
    # All four slice bounds must be non-negative.
    for slc in (*out_slc, *tile_slc):
        assert slc.start >= 0 and slc.stop >= 0


# ----------------------------------------------------------------------------
# Tier-B: Operator round-trips
# ----------------------------------------------------------------------------


def test_geom_module_is_public() -> None:
    assert gz.geom.Tile is gz.geom.Tile


def test_pad_to_then_crop_to_recovers_original_values() -> None:
    gt = _gt()

    padded = gz.geom.PadTo(shape=(9, 11), fill=-9999)(gt)
    cropped = gz.geom.CropTo(shape=gt.shape[-2:])(padded)

    assert cropped.shape == gt.shape
    np.testing.assert_array_equal(np.asarray(cropped), np.asarray(gt))
    assert cropped.same_extent(gt)


def test_tile_then_stitch_reconstructs_original_with_crs_preserved() -> None:
    gt = _gt()

    tiles = gz.geom.Tile(size=(3, 4))(gt)
    stitched = gz.geom.Stitch(blend="average")(tiles)

    assert stitched.shape == gt.shape
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))
    assert stitched.same_extent(gt)
    # CRS propagation through the round-trip.
    assert str(stitched.crs) == str(gt.crs)
    # Affine origin matches.
    assert stitched.transform == gt.transform


def test_stitch_masks_fill_sentinels_from_padded_tiles() -> None:
    """Stitch must ignore the fill sentinel that boundless reads produce.

    Hand-build two tiles where the right tile has its rightmost column
    padded with the fill value; the blended output must reproduce the
    real values, not average in the sentinel.
    """
    crs = "EPSG:4326"
    fill = -9999.0
    real = np.ones((1, 3, 3), dtype=np.float32)
    left = GeoTensor(
        real,
        transform=Affine(1, 0, 0, 0, -1, 3),
        crs=crs,
        fill_value_default=fill,
    )
    padded = np.array(
        [[[1.0, 1.0, fill], [1.0, 1.0, fill], [1.0, 1.0, fill]]],
        dtype=np.float32,
    )
    right = GeoTensor(
        padded,
        transform=Affine(1, 0, 3, 0, -1, 3),
        crs=crs,
        fill_value_default=fill,
    )
    stitched = gz.geom.Stitch(blend="average")([left, right])
    out = np.asarray(stitched)
    # Sentinel pixels did not bleed, and the trailing all-sentinel column
    # (Tile's edge padding) is cropped off: only genuine values remain.
    assert out.shape == (1, 3, 5)
    np.testing.assert_array_equal(out, np.ones((1, 3, 5)))
    # Pinning the target grid keeps the padded column, filled with the fill.
    pinned = np.asarray(
        gz.geom.Stitch(blend="average", target_shape=(3, 6))([left, right])
    )
    assert pinned.shape == (1, 3, 6)
    np.testing.assert_array_equal(pinned[..., :5], np.ones((1, 3, 5)))
    np.testing.assert_array_equal(pinned[..., 5], np.full((1, 3), fill))


def test_reproject_like_matches_reference_grid_and_propagates_crs() -> None:
    gt = _gt()
    like = GeoTensor(
        np.zeros((1, 3, 4), dtype=np.float32),
        transform=Affine(2, 0, 10, 0, -2, 20),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )

    out = gz.geom.ReprojectLike(like=like, resampling="nearest")(gt)

    assert out.same_extent(like)
    assert str(out.crs) == str(like.crs)
    assert out.transform == like.transform


def test_resize_resample_and_crop_to_bounds() -> None:
    gt = _gt()

    resized = gz.geom.Resize(shape=(10, 14), anti_aliasing=False)(gt)
    resampled = gz.geom.Resample(resolution=(2, 2), anti_aliasing=False)(gt)
    cropped = gz.geom.CropToBounds(bounds=(12, 16, 15, 19), crs="EPSG:4326")(gt)

    assert resized.shape == (1, 10, 14)
    assert resampled.shape == (1, 2, 4)
    assert cropped.shape == (1, 3, 3)
    assert gz.geom.Resize(shape=(1, 1)).get_config()["shape"] == [1, 1]
    # CRS preserved through every resize / crop path.
    assert str(resized.crs) == str(gt.crs)
    assert str(resampled.crs) == str(gt.crs)
    assert str(cropped.crs) == str(gt.crs)


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
        "size": [3, 3],
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


def test_rasterize_like_accepts_geodataframe_and_default_column() -> None:
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


def test_stitch_feather_matches_spatial_overlap_add() -> None:
    pytest.importorskip(
        "geopatcher",
        reason="geotoolz.patch_ops bridge requires the [patch] extra (geopatcher)",
    )
    from geopatcher import (
        RasterField,
        SpatialOverlapAdd,
        SpatialPatcher,
        SpatialRectangular,
        SpatialRegularStride,
    )

    from geotoolz.patch_ops import SpatialTriangular

    gt = _gt()
    tiles = gz.geom.Tile(size=(3, 4), stride=(2, 3))(gt)

    stitched = gz.geom.Stitch(blend="feather", feather_width=2)(tiles)
    field = RasterField(gt)
    patcher = SpatialPatcher(
        geometry=SpatialRectangular(size=(3, 4)),
        sampler=SpatialRegularStride(step=(2, 3)),
        window=SpatialTriangular(width=2),
        aggregation=SpatialOverlapAdd(),
    )

    patches = list(patcher.split(field))
    patch_stitched = SpatialOverlapAdd().merge(patches, field.reader)

    assert patch_stitched.shape == gt.shape
    np.testing.assert_allclose(patch_stitched, np.asarray(gt), rtol=1e-6)
    np.testing.assert_allclose(patch_stitched, np.asarray(stitched), rtol=1e-6)


def test_stitch_feather_clips_valid_mask_for_custom_target_grid() -> None:
    fill = -9999.0
    tile = GeoTensor(
        np.array([[[1.0, fill], [fill, 4.0]]], dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 2),
        crs="EPSG:4326",
        fill_value_default=fill,
    )

    stitched = gz.geom.Stitch(
        blend="feather",
        feather_width=1,
        target_shape=(1, 1),
        target_transform=Affine(1, 0, 1, 0, -1, 1),
    )([tile])

    np.testing.assert_array_equal(
        np.asarray(stitched),
        np.array([[[4.0]]], dtype=np.float32),
    )


def test_stitch_validates_north_up_on_every_tile() -> None:
    """A rotated/sheared tile in any position must raise, not just the first.

    Regression: the north-up check previously only validated
    ``tiles[0]``. A rotated later tile would silently misalign because
    ``target_slices`` assumes axis-aligned transforms for every tile.
    """
    north_up = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 2),
        crs="EPSG:4326",
    )
    rotated = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(1, 0.1, 2, 0, -1, 2),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="tile 1"):
        gz.geom.Stitch()([north_up, rotated])


def test_stitch_handles_target_shape_smaller_than_tile_union() -> None:
    """A custom ``target_shape`` smaller than the tile union must not error.

    Regression: ``target_slices`` produced negative-bound slices for
    tiles fully past the target window, causing the blender to wrap
    around and write into wrong pixels. The fix clamps slices and
    skips empty intersections.
    """
    crs = "EPSG:4326"
    fill = -9999.0
    transform = Affine(1, 0, 0, 0, -1, 4)
    # Four 2x2 tiles tiling a 4x4 grid; we then ask for a 2x2 target.
    tiles = [
        GeoTensor(
            np.full((1, 2, 2), 1.0, dtype=np.float32),
            transform=Affine(1, 0, 0, 0, -1, 4),
            crs=crs,
            fill_value_default=fill,
        ),
        GeoTensor(
            np.full((1, 2, 2), 2.0, dtype=np.float32),
            transform=Affine(1, 0, 2, 0, -1, 4),
            crs=crs,
            fill_value_default=fill,
        ),
        GeoTensor(
            np.full((1, 2, 2), 3.0, dtype=np.float32),
            transform=Affine(1, 0, 0, 0, -1, 2),
            crs=crs,
            fill_value_default=fill,
        ),
        GeoTensor(
            np.full((1, 2, 2), 4.0, dtype=np.float32),
            transform=Affine(1, 0, 2, 0, -1, 2),
            crs=crs,
            fill_value_default=fill,
        ),
    ]
    stitched = gz.geom.Stitch(
        blend="first",
        target_shape=(2, 2),
        target_transform=transform,
    )(tiles)
    # Only the top-left tile (value 1) should contribute to a 2x2 window
    # anchored at (0, 0); the other three are fully outside and must be
    # silently dropped, not wrapped.
    out = np.asarray(stitched)
    assert out.shape == (1, 2, 2)
    np.testing.assert_array_equal(out, np.full((1, 2, 2), 1.0, dtype=np.float32))


def test_bowtie_correction_is_identity_for_zero_scan_angle() -> None:
    gt = _gt()

    out = gz.geom.BowtieCorrection(
        scan_angle_max_deg=0.0,
        pixels_per_scan=gt.shape[-1],
        scans_per_granule=gt.shape[-2],
    )(gt)

    assert out is gt
    np.testing.assert_array_equal(np.asarray(out), np.asarray(gt))
    assert out.transform == gt.transform
    assert str(out.crs) == str(gt.crs)


def test_bowtie_correction_resamples_edges_and_preserves_fill() -> None:
    values = np.tile(np.arange(7, dtype=np.float32), (5, 1))[None, ...]
    values[..., 2, 1] = -9999.0
    gt = _gt(values)

    out = gz.geom.BowtieCorrection(
        scan_angle_max_deg=60.0,
        pixels_per_scan=7,
        scans_per_granule=5,
        method="nearest",
    )(gt)

    assert out.shape == gt.shape
    assert out.transform == gt.transform
    assert str(out.crs) == str(gt.crs)
    assert np.asarray(out)[0, 2, 0] == -9999.0
    assert np.asarray(out)[0, 2, 1] == -9999.0
    assert np.asarray(out)[0, 0, 3] == np.asarray(gt)[0, 0, 3]
    assert np.asarray(out)[0, 0, 0] > np.asarray(gt)[0, 0, 0]


def test_antimeridian_split_uses_lon_attrs_and_returns_west_east() -> None:
    values = np.arange(1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4)
    lons = np.array([[170.0, 175.0, -179.0, -170.0]] * 2)
    gt = GeoTensor(
        values,
        transform=Affine(1, 0, 170, 0, -1, 10),
        crs="EPSG:4326",
        fill_value_default=-9999,
        attrs={"lons": lons},
    )

    west, east = gz.geom.AntimeridianSplit()(gt)

    np.testing.assert_array_equal(np.asarray(west), values[..., 2:])
    np.testing.assert_array_equal(np.asarray(east), values[..., :2])
    assert np.nanmax(west.attrs["lons"]) < 0
    assert np.nanmin(east.attrs["lons"]) > 0
    assert west.transform == gt.isel({"x": slice(2, 4)}).transform
    assert str(west.crs) == str(gt.crs)


def test_antimeridian_split_identity_when_no_crossing() -> None:
    gt = _gt()

    out = gz.geom.AntimeridianSplit()(gt)

    assert out == [gt]


def test_antimeridian_split_rejects_multiple_jumps() -> None:
    # Polar-pass-like swath that wraps the antimeridian twice in a row.
    values = np.arange(1 * 2 * 6, dtype=np.float32).reshape(1, 2, 6)
    lons = np.array([[170.0, -170.0, 170.0, -170.0, 170.0, -170.0]] * 2)
    gt = GeoTensor(
        values,
        transform=Affine(1, 0, 170, 0, -1, 10),
        crs="EPSG:4326",
        fill_value_default=-9999,
        attrs={"lons": lons},
    )

    with pytest.raises(ValueError, match="longitude jumps"):
        gz.geom.AntimeridianSplit()(gt)


def test_geostationary_parallax_zero_height_is_identity() -> None:
    gt = _gt()

    out = gz.geom.GeostationaryParallaxCorrect(
        satellite_lon_deg=gt.transform.c,
        target_height_m=0.0,
    )(gt)

    assert out is gt
    np.testing.assert_array_equal(np.asarray(out), np.asarray(gt))
    assert out.transform == gt.transform


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
def test_parallax_near_identity_small_height(method: str) -> None:
    # A 1 mm target height displaces the apparent position by far less than
    # a pixel, so sampling at pixel centres must reproduce the input (#119).
    values = np.arange(6 * 8, dtype=np.float64).reshape(1, 6, 8)
    gt = GeoTensor(
        values,
        transform=Affine(0.1, 0, -75.4, 0, -0.1, 0.3),
        crs="EPSG:4326",
        fill_value_default=np.nan,
    )

    out = gz.geom.GeostationaryParallaxCorrect(
        satellite_lon_deg=-75.0,
        target_height_m=1e-3,
        method=method,
    )(gt)

    arr = np.asarray(out)
    assert not np.isnan(arr).any()
    np.testing.assert_allclose(arr, values, atol=1e-6)


def test_geostationary_parallax_moves_elevated_point_toward_nadir() -> None:
    # Grid centred on (30N, 75W), due north of the GOES-East sub-satellite
    # point. A 40 km target is displaced ~0.25 deg (2 pixels) poleward in
    # the apparent view, so correction moves it 2 rows south, toward nadir.
    values = np.zeros((1, 9, 9), dtype=np.float32)
    values[0, 2, 4] = 1.0
    gt = GeoTensor(
        values,
        transform=Affine(0.125, 0, -75.5625, 0, -0.125, 30.5625),
        crs="EPSG:4326",
        fill_value_default=0.0,
    )

    out = gz.geom.GeostationaryParallaxCorrect(
        satellite_lon_deg=-75.0,
        target_height_m=40_000.0,
        method="nearest",
    )(gt)

    assert np.asarray(out)[0, 4, 4] == 1.0
    assert np.asarray(out)[0, 2, 4] == 0.0
    assert out.transform == gt.transform
    assert str(out.crs) == str(gt.crs)


def test_geostationary_parallax_fills_off_limb_pixels() -> None:
    # Grid that straddles the GOES-East sub-satellite point with pixels near
    # the limb; at a high target height some rays will miss the Earth.
    fill = -1.0
    values = np.ones((1, 5, 5), dtype=np.float32)
    gt = GeoTensor(
        values,
        transform=Affine(20.0, 0, -150.0, 0, -20.0, 50.0),
        crs="EPSG:4326",
        fill_value_default=fill,
    )

    out = gz.geom.GeostationaryParallaxCorrect(
        satellite_lon_deg=-75.0,
        target_height_m=200_000.0,
        method="nearest",
    )(gt)

    arr = np.asarray(out)
    # The corner pixels are far off-limb relative to the sub-satellite point.
    assert (arr == fill).any()
    assert out.transform == gt.transform


def test_segment_stitch_orders_segments_and_fills_missing_scan() -> None:
    fill = -1.0
    seg0 = GeoTensor(
        np.full((1, 2, 3), 1.0, dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 6),
        crs="EPSG:4326",
        fill_value_default=fill,
        attrs={"__geotoolz_segment_meta__": {"segment_index": 0, "n_segments": 3}},
    )
    seg2 = GeoTensor(
        np.full((1, 2, 3), 3.0, dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 2),
        crs="EPSG:4326",
        fill_value_default=fill,
        attrs={"__geotoolz_segment_meta__": {"segment_index": 2, "n_segments": 3}},
    )

    out = gz.geom.SegmentStitch(axis="scan", fill=fill)([seg2, seg0])

    expected = np.concatenate(
        [
            np.full((1, 2, 3), 1.0, dtype=np.float32),
            np.full((1, 2, 3), fill, dtype=np.float32),
            np.full((1, 2, 3), 3.0, dtype=np.float32),
        ],
        axis=-2,
    )
    np.testing.assert_array_equal(np.asarray(out), expected)
    assert out.transform == seg0.transform
    assert str(out.crs) == str(seg0.crs)


def test_segment_stitch_roundtrips_sample_segments() -> None:
    values = np.arange(1 * 3 * 6, dtype=np.float32).reshape(1, 3, 6)
    left = GeoTensor(
        values[..., :3],
        transform=Affine(1, 0, 0, 0, -1, 3),
        crs="EPSG:4326",
        attrs={"__geotoolz_segment_meta__": {"segment_index": 0, "n_segments": 2}},
    )
    right = GeoTensor(
        values[..., 3:],
        transform=Affine(1, 0, 3, 0, -1, 3),
        crs="EPSG:4326",
        attrs={"__geotoolz_segment_meta__": {"segment_index": 1, "n_segments": 2}},
    )

    out = gz.geom.SegmentStitch(axis="sample")([right, left])

    np.testing.assert_array_equal(np.asarray(out), values)
    assert out.transform == left.transform


def test_segment_stitch_integer_dtype_uses_existing_fill() -> None:
    # Integer sensor counts cannot hold NaN; stitching with the default
    # fill=np.nan must fall back to the segment's existing fill (or 0)
    # rather than raising.
    seg0 = GeoTensor(
        np.full((1, 2, 3), 1, dtype=np.int16),
        transform=Affine(1, 0, 0, 0, -1, 6),
        crs="EPSG:4326",
        fill_value_default=-1,
        attrs={"__geotoolz_segment_meta__": {"segment_index": 0, "n_segments": 3}},
    )
    seg2 = GeoTensor(
        np.full((1, 2, 3), 3, dtype=np.int16),
        transform=Affine(1, 0, 0, 0, -1, 2),
        crs="EPSG:4326",
        fill_value_default=-1,
        attrs={"__geotoolz_segment_meta__": {"segment_index": 2, "n_segments": 3}},
    )

    out = gz.geom.SegmentStitch(axis="scan")([seg2, seg0])

    arr = np.asarray(out)
    assert arr.dtype == np.int16
    # Missing middle segment is filled with the segments' existing fill (-1).
    assert (arr[..., 2:4, :] == -1).all()
    assert out.fill_value_default == -1


def test_segment_stitch_rejects_duplicate_segment_index() -> None:
    seg0 = GeoTensor(
        np.zeros((1, 2, 3), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 4),
        crs="EPSG:4326",
        attrs={"__geotoolz_segment_meta__": {"segment_index": 0, "n_segments": 2}},
    )
    dup = GeoTensor(
        np.zeros((1, 2, 3), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 4),
        crs="EPSG:4326",
        attrs={"__geotoolz_segment_meta__": {"segment_index": 0, "n_segments": 2}},
    )

    with pytest.raises(ValueError, match="Duplicate segment_index"):
        gz.geom.SegmentStitch()([seg0, dup])


def test_segment_stitch_handles_attrs_none_with_clear_error() -> None:
    bad = GeoTensor(
        np.zeros((1, 2, 3), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 4),
        crs="EPSG:4326",
        attrs=None,
    )
    with pytest.raises(ValueError, match="__geotoolz_segment_meta__"):
        gz.geom.SegmentStitch()([bad])


def test_bowtie_correction_preserves_float32_dtype_and_edge_pixels() -> None:
    values = np.tile(np.arange(7, dtype=np.float32), (5, 1))[None, ...]
    gt = _gt(values)

    out = gz.geom.BowtieCorrection(
        scan_angle_max_deg=60.0,
        pixels_per_scan=7,
        scans_per_granule=5,
        method="bilinear",
    )(gt)

    arr = np.asarray(out)
    # Bilinear resampling must not silently promote float32 -> float64.
    assert arr.dtype == np.float32
    # The trailing row/column lies inside the raster footprint and must not
    # be replaced with the fill value.
    assert arr[0, -1, 0] != gt.fill_value_default
    assert arr[0, 0, -1] != gt.fill_value_default


def test_antimeridian_split_rejects_mismatched_lon_shape() -> None:
    values = np.arange(1 * 2 * 4, dtype=np.float32).reshape(1, 2, 4)
    gt = GeoTensor(
        values,
        transform=Affine(1, 0, 170, 0, -1, 10),
        crs="EPSG:4326",
        fill_value_default=-9999,
        attrs={"lons": np.array([170.0, 175.0, -179.0])},  # wrong length
    )
    with pytest.raises(ValueError, match="lons"):
        gz.geom.AntimeridianSplit()(gt)


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


# ----------------------------------------------------------------------------
# georeader delegation regressions (#129)
# ----------------------------------------------------------------------------


def test_resample_non_square_pixels() -> None:
    """Each axis is scaled by its own resolution ratio (x: 1 -> 2, y: 2 -> 2).

    ``GeoTensor.resize(resolution_dst=...)`` scales the height by the x
    ratio, which gave ``(1, 5, 20)`` here.
    """
    gt = GeoTensor(
        np.ones((1, 10, 20), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -2, 0),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )

    out = gz.geom.Resample(resolution=(2, 2), anti_aliasing=False)(gt)

    assert out.shape == (1, 10, 10)
    assert out.res == pytest.approx((2.0, 2.0))
    assert out.transform == Affine(2, 0, 0, 0, -2, 0)
    assert out.bounds == pytest.approx(gt.bounds)
    # Resize with a non-uniform scale shares the per-axis transform fix.
    resized = gz.geom.Resize(shape=(5, 10), anti_aliasing=False)(gt)
    assert resized.transform == Affine(2, 0, 0, 0, -4, 0)
    assert resized.bounds == pytest.approx(gt.bounds)


def test_reproject_same_crs_applies_resolution() -> None:
    gt = GeoTensor(
        np.arange(10 * 20, dtype=np.float32).reshape(1, 10, 20),
        transform=Affine(10, 0, 500_000, 0, -10, 4_000_000),
        crs="EPSG:32629",
        fill_value_default=-9999,
    )

    out = gz.geom.Reproject(
        dst_crs="EPSG:32629", resolution=(20.0, 20.0), resampling="average"
    )(gt)

    assert out.shape == (1, 5, 10)
    assert out.res == pytest.approx((20.0, 20.0))
    assert out.bounds == pytest.approx(gt.bounds)
    assert str(out.crs) == str(gt.crs)
    # Without a resolution a same-CRS call is still the identity.
    same = gz.geom.Reproject(dst_crs="EPSG:32629")(gt)
    assert same.shape == gt.shape
    assert same.transform == gt.transform


def test_vectorize_band_axis_input() -> None:
    values = np.zeros((1, 5, 7), dtype=np.uint8)
    values[0, 1:4, 2:5] = 1
    mask = _gt(values)

    polygons = gz.geom.Vectorize(min_area=0.5)(mask)

    assert len(polygons) == 1
    assert polygons[0].equals(box(12, 16, 15, 19))
    with pytest.raises(ValueError, match="single-band"):
        gz.geom.Vectorize()(_gt(np.zeros((2, 5, 7), dtype=np.uint8)))


def test_tile_trailing_shapes_documented() -> None:
    """Trailing tiles are emitted at ``size``, padded with the fill sentinel."""
    gt = _gt()  # (1, 5, 7), fill -9999

    tiles = gz.geom.Tile(size=(3, 4))(gt)

    assert [tile.shape for tile in tiles] == [(1, 3, 4)] * 4
    bottom_right = np.asarray(tiles[-1])
    np.testing.assert_array_equal(bottom_right[0, :2, :3], np.asarray(gt)[0, 3:, 4:])
    assert (bottom_right[0, 2, :] == -9999).all()
    assert (bottom_right[0, :, 3] == -9999).all()
    # Stitch masks the sentinel and crops the padding: exact round trip.
    for blend in ("average", "feather", "first", "max"):
        stitched = gz.geom.Stitch(blend=blend, feather_width=1)(tiles)
        assert stitched.transform == gt.transform
        np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))
    # Overlapping tiles round-trip too.
    overlapping = gz.geom.SlidingWindow(size=(3, 3), overlap=1)(gt)
    assert {tile.shape for tile in overlapping} == {(1, 3, 3)}
    stitched = gz.geom.Stitch()(overlapping)
    np.testing.assert_allclose(np.asarray(stitched), np.asarray(gt))
    # Plain arrays and fill-less GeoTensors are zero-padded to ``size``.
    plain = gz.geom.Tile(size=(3, 4))(np.ones((1, 5, 7), dtype=np.float32))
    assert [tile.shape for tile in plain] == [(1, 3, 4)] * 4
    assert plain[-1][0, 2, :].tolist() == [0.0] * 4
    no_fill = GeoTensor(
        np.ones((1, 5, 7), dtype=np.float32),
        transform=gt.transform,
        crs=gt.crs,
        fill_value_default=None,
    )
    no_fill_tiles = gz.geom.Tile(size=(3, 4))(no_fill)
    assert [tile.shape for tile in no_fill_tiles] == [(1, 3, 4)] * 4
    assert no_fill_tiles[-1].transform == Affine(1, 0, 14, 0, -1, 17)
    np.testing.assert_array_equal(
        np.asarray(no_fill_tiles[-1]),
        plain[-1],
    )
    # ``include_incomplete=False`` keeps only full tiles.
    full_only = gz.geom.Tile(size=(3, 4), include_incomplete=False)(gt)
    assert [tile.shape for tile in full_only] == [(1, 3, 4)]


def test_pad_to_without_any_fill_falls_back_to_zero() -> None:
    gt = GeoTensor(
        np.ones((1, 5, 7), dtype=np.float32),
        transform=Affine(1, 0, 10, 0, -1, 20),
        crs="EPSG:4326",
        fill_value_default=None,
    )

    padded = gz.geom.PadTo(shape=(7, 9))(gt)

    assert padded.shape == (1, 7, 9)
    assert np.asarray(padded).sum() == 35
    assert np.asarray(padded)[0, 0, 0] == 0
    assert padded.transform == Affine(1, 0, 9, 0, -1, 21)
    # With a carrier fill, ``fill=None`` still uses it.
    padded_fill = gz.geom.PadTo(shape=(7, 9))(_gt())
    assert np.asarray(padded_fill)[0, 0, 0] == -9999


def test_rasterize_float_fill() -> None:
    gt = _gt()
    geometry = box(12, 16, 15, 19)

    burned = gz.geom.Rasterize(geometries=[geometry], fill=np.nan)(gt)

    values = np.asarray(burned)
    assert values.dtype == np.float32
    assert np.nansum(values) == 9
    assert np.isnan(values).sum() == values.size - 9
    # A negative fill cannot live in the default ``uint8`` either.
    negative = gz.geom.Rasterize(geometries=[geometry], fill=-1)(gt)
    assert np.asarray(negative).min() == -1
    # The default integer background keeps the compact ``uint8`` mask.
    assert np.asarray(gz.geom.Rasterize(geometries=[geometry])(gt)).dtype == np.uint8


def test_mosaic_builds_frame_without_spatial_mosaic_and_handles_no_fill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import warnings

    from geotoolz.geom._src import operators as geom_operators

    def _fail(*args, **kwargs):
        raise AssertionError("reducer methods must not run spatial_mosaic")

    monkeypatch.setattr(geom_operators.mosaic, "spatial_mosaic", _fail)

    def _pair(fill):
        left = GeoTensor(
            np.ones((1, 3, 4), dtype=np.float32),
            transform=Affine(1, 0, 10, 0, -1, 20),
            crs="EPSG:4326",
            fill_value_default=fill,
        )
        right = GeoTensor(
            np.full((1, 3, 4), 3, dtype=np.float32),
            transform=Affine(1, 0, 12, 0, -1, 18),
            crs="EPSG:4326",
            fill_value_default=fill,
        )
        return left, right

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = gz.geom.Mosaic(method="mean", resampling="nearest")(_pair(-9999))
    values = np.asarray(out)[0]
    assert out.shape == (1, 5, 6)
    assert out.transform == Affine(1, 0, 10, 0, -1, 20)
    assert values[2, 2] == 2  # overlap -> mean(1, 3)
    assert values[0, 5] == -9999  # covered by neither input
    assert out.fill_value_default == -9999

    for method in ("mean", "median", "max", "min", "first"):
        no_fill = gz.geom.Mosaic(method=method, resampling="nearest")(_pair(None))
        values = np.asarray(no_fill)[0]
        assert no_fill.shape == (1, 5, 6)
        assert np.isnan(values[0, 5])
        assert values[0, 0] == 1
        assert values[4, 5] == 3
    first = np.asarray(gz.geom.Mosaic(method="first")(_pair(None)))
    assert first[0, 2, 2] == 1  # first valid input wins on overlap


def test_stitch_rejects_mismatched_resolution() -> None:
    """A 2 m tile on a 1 m grid would be pasted at 1 m; raise instead."""
    fine = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, 2),
        crs="EPSG:4326",
    )
    coarse = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(2, 0, 2, 0, -2, 2),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="tile 1"):
        gz.geom.Stitch()([fine, coarse])
    with pytest.raises(ValueError, match="target_transform"):
        gz.geom.Stitch(target_transform=Affine(2, 0, 0, 0, -2, 2))([fine])


def test_stitch_rejects_south_up_tile() -> None:
    south_up = GeoTensor(
        np.ones((1, 2, 2), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, 1, 0),
        crs="EPSG:4326",
    )
    with pytest.raises(ValueError, match="north-up"):
        gz.geom.Stitch()([south_up])


def test_carrier_referencing_operators_flagged_forbid_in_yaml() -> None:
    """Operators that hold concrete GeoTensor / GeoDataFrame references
    must declare ``forbid_in_yaml = True`` so future YAML dumpers refuse
    to round-trip them silently."""
    like = _gt()
    poly_gdf = gpd.GeoDataFrame({"v": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
    assert gz.geom.ReprojectLike(like=like).forbid_in_yaml is True
    assert gz.geom.ResampleLike(like=like).forbid_in_yaml is True
    assert gz.geom.RasterizeLike(like=like, geometries=poly_gdf).forbid_in_yaml is True
    assert gz.geom.Rasterize(geometries=[box(0, 0, 1, 1)]).forbid_in_yaml is True
    assert (
        gz.geom.Georeference(
            glt=GeoTensor(
                np.zeros((2, 3, 3), dtype=np.int32),
                transform=Affine(1, 0, 0, 0, -1, 0),
                crs="EPSG:4326",
            )
        ).forbid_in_yaml
        is True
    )
    # YAML-safe ops stay False.
    assert gz.geom.Reproject(dst_crs="EPSG:4326").forbid_in_yaml is False
    assert gz.geom.Tile(size=(2, 2)).forbid_in_yaml is False


# ----------------------------------------------------------------------------
# Tier-C: hydra-zen builds() round-trip for YAML-safe operators
# ----------------------------------------------------------------------------


try:
    import hydra_zen as _hydra_zen  # type: ignore[import-not-found]

    _HAS_HYDRA_ZEN = True
except ImportError:
    _hydra_zen = None  # type: ignore[assignment]
    _HAS_HYDRA_ZEN = False


@pytest.mark.skipif(not _HAS_HYDRA_ZEN, reason="hydra-zen extra not installed")
@pytest.mark.parametrize(
    "operator",
    [
        gz.geom.Reproject(dst_crs="EPSG:4326", resolution=(10.0, 10.0)),
        gz.geom.Resize(shape=(256, 256)),
        gz.geom.Resample(resolution=(20.0, 20.0)),
        gz.geom.PadTo(shape=(512, 512), fill=0.0),
        gz.geom.CropTo(shape=(256, 256), anchor="upper_left"),
        gz.geom.CropToBounds(bounds=(0.0, 0.0, 1.0, 1.0), crs="EPSG:4326"),
        gz.geom.Tile(size=(128, 128), stride=(64, 64)),
        gz.geom.SlidingWindow(size=(128, 128), overlap=16),
        gz.geom.Stitch(blend="feather", feather_width=8),
        gz.geom.Mosaic(method="median", resampling="bilinear"),
        gz.geom.Vectorize(min_area=10.0, simplify_tolerance=0.5),
    ],
)
def test_yaml_safe_operators_roundtrip_through_hydra_zen(operator) -> None:
    cfg = _hydra_zen.builds(type(operator), **operator.get_config())
    restored = _hydra_zen.instantiate(cfg)
    assert type(restored) is type(operator)
    assert restored.get_config() == operator.get_config()


# ----------------------------------------------------------------------------
# Registration operators (PhaseAlign / OpticalFlow*)
# ----------------------------------------------------------------------------


def _registration_pair() -> tuple[GeoTensor, GeoTensor, int, int]:
    """Build a reference scene and an integer-pixel-shifted moving scene."""
    rng = np.random.default_rng(0)
    base = rng.uniform(0.0, 1.0, size=(48, 48)).astype(np.float32)
    yy, xx = np.ogrid[:48, :48]
    base += np.exp(-((yy - 24) ** 2 + (xx - 24) ** 2) / 32.0).astype(np.float32)
    reference = GeoTensor(
        base[None, ...],
        transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        crs="EPSG:32629",
        fill_value_default=0.0,
    )
    dy, dx = 3, -2
    moving_arr = np.roll(base, shift=(dy, dx), axis=(0, 1))
    moving = GeoTensor(
        moving_arr[None, ...],
        transform=Affine(1.0, 0.0, 100.0, 0.0, -1.0, 200.0),
        crs="EPSG:32629",
        fill_value_default=0.0,
    )
    return reference, moving, dy, dx


def test_phase_align_returns_shift_when_apply_false() -> None:
    reference, moving, dy, dx = _registration_pair()
    result = gz.geom.PhaseAlign(reference=reference, apply=False)(moving)
    assert isinstance(result, tuple)
    shift_y, shift_x, error = result
    # phase_cross_correlation returns the shift needed to align the moving
    # image back onto the reference, i.e. the negative of the applied roll.
    assert shift_y == pytest.approx(-dy, abs=0.5)
    assert shift_x == pytest.approx(-dx, abs=0.5)
    assert error >= 0.0


def test_phase_align_apply_keeps_transform_and_preserves_metadata() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    moving.attrs = {"sensor": "test"}
    aligned = gz.geom.PhaseAlign(reference=reference, apply=True)(moving)
    assert isinstance(aligned, GeoTensor)
    assert aligned.shape == moving.shape
    assert str(aligned.crs) == str(moving.crs)
    assert aligned.fill_value_default == moving.fill_value_default
    assert aligned.attrs == moving.attrs
    # The array shift carries the alignment; the grid is unchanged (#118).
    assert aligned.transform == moving.transform


def test_phase_align_world_coordinates_match_reference() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    aligned = gz.geom.PhaseAlign(reference=reference, apply=True)(moving)

    def peak_world(gt: GeoTensor) -> tuple[float, float]:
        row, col = np.unravel_index(np.argmax(np.asarray(gt)[0]), gt.shape[-2:])
        return gt.transform * (col + 0.5, row + 0.5)

    ref_x, ref_y = peak_world(reference)
    mov_x, mov_y = peak_world(moving)
    out_x, out_y = peak_world(aligned)
    # Sanity: the moving scene's peak is displaced before alignment.
    assert (mov_x, mov_y) != pytest.approx((ref_x, ref_y))
    assert (out_x, out_y) == pytest.approx((ref_x, ref_y))
    # Interior pixels (away from the edge-filled border) match exactly.
    np.testing.assert_allclose(
        np.asarray(aligned)[0, 4:-4, 4:-4],
        np.asarray(reference)[0, 4:-4, 4:-4],
        atol=1e-5,
    )


def test_phase_align_rejects_unsupported_ndim() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    four_d = np.asarray(moving)[None, ...]
    with pytest.raises(
        ValueError,
        match=r"PhaseAlign accepts 2-D \(H, W\) or 3-D \(C, H, W\) input; got a 4-D",
    ):
        gz.geom.PhaseAlign(reference=reference)(four_d)


def test_phase_align_rejects_mismatched_shapes() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    bigger = GeoTensor(
        np.zeros((1, 64, 64), dtype=np.float32),
        transform=moving.transform,
        crs=moving.crs,
        fill_value_default=0.0,
    )
    with pytest.raises(ValueError, match="identical spatial shape"):
        gz.geom.PhaseAlign(reference=reference)(bigger)


def test_optical_flow_tvl1_returns_displacement_field() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    flow = gz.geom.OpticalFlowTVL1(reference=reference)(moving)
    arr = np.asarray(flow)
    assert arr.shape == (2, *reference.shape[-2:])
    assert str(flow.crs) == str(moving.crs)


@pytest.mark.parametrize("name", ["OpticalFlowTVL1", "OpticalFlowILK"])
def test_optical_flow_declares_nan_fill(name: str) -> None:
    """A displacement is a new quantity: NaN nodata, not the scene fill (#146)."""
    reference, moving, _dy, _dx = _registration_pair()
    values = np.asarray(moving).copy()
    values[:, 5, 7] = 0.0  # nodata in the moving scene (fill 0.0)
    moving = GeoTensor(
        values, transform=moving.transform, crs=moving.crs, fill_value_default=0.0
    )
    flow = getattr(gz.geom, name)(reference=reference)(moving)
    assert np.isnan(flow.fill_value_default)
    arr = np.asarray(flow)
    assert np.isnan(arr[:, 5, 7]).all()
    assert np.isfinite(arr[:, 20, 20]).all()


def test_optical_flow_ilk_returns_displacement_field() -> None:
    reference, moving, _dy, _dx = _registration_pair()
    flow = gz.geom.OpticalFlowILK(reference=reference)(moving)
    arr = np.asarray(flow)
    assert arr.shape == (2, *reference.shape[-2:])
    assert str(flow.crs) == str(moving.crs)


# ----------------------------------------------------------------------------
# numpy-compat: plain ndarray in -> plain ndarray out (or clear TypeError)
# ----------------------------------------------------------------------------


_PLAIN_VALUES = np.arange(1 * 8 * 8, dtype=np.float32).reshape(1, 8, 8)
_PLAIN_REFERENCE = np.ascontiguousarray(_PLAIN_VALUES[:, ::-1, :])


@pytest.mark.parametrize(
    "make_op",
    [
        lambda: gz.geom.PadTo(shape=(10, 10), fill=0),
        lambda: gz.geom.CropTo(shape=(2, 3)),
        lambda: gz.geom.CropTo(shape=(2, 3), anchor="upper_left"),
        lambda: gz.geom.BowtieCorrection(
            scan_angle_max_deg=60.0,
            pixels_per_scan=8,
            scans_per_granule=8,
            method="nearest",
        ),
        lambda: gz.geom.OpticalFlowTVL1(reference=_gt(_PLAIN_REFERENCE.copy())),
        lambda: gz.geom.OpticalFlowILK(reference=_gt(_PLAIN_REFERENCE.copy())),
        lambda: gz.geom.PhaseAlign(reference=_gt(_PLAIN_REFERENCE.copy()), apply=True),
    ],
)
def test_pixel_space_operators_accept_plain_ndarray(make_op) -> None:
    """Plain array in -> plain array out, values equal to the GeoTensor path."""
    op = make_op()
    out_arr = op(_PLAIN_VALUES.copy())
    out_gt = op(_gt(_PLAIN_VALUES.copy()))
    assert type(out_arr) is np.ndarray
    assert isinstance(out_gt, GeoTensor)
    np.testing.assert_allclose(out_arr, np.asarray(out_gt))


def test_tile_accepts_plain_ndarray_and_zero_pads_edges() -> None:
    values = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    # Exact-fit tiling matches the GeoTensor path bit-for-bit.
    tiles_arr = gz.geom.Tile(size=(2, 2))(values.copy())
    tiles_gt = gz.geom.Tile(size=(2, 2))(_gt(values.copy()))
    assert len(tiles_arr) == len(tiles_gt) == 4
    for tile_arr, tile_gt in zip(tiles_arr, tiles_gt, strict=True):
        assert type(tile_arr) is np.ndarray
        np.testing.assert_array_equal(tile_arr, np.asarray(tile_gt))
    # SlidingWindow (Tile subclass) inherits the plain-array support.
    windows = gz.geom.SlidingWindow(size=(3, 3), overlap=1)(np.ones((4, 4)))
    assert all(type(window) is np.ndarray for window in windows)


def test_phase_align_plain_ndarray_returns_shift_tuple_when_apply_false() -> None:
    reference, moving, dy, dx = _registration_pair()
    result = gz.geom.PhaseAlign(reference=reference, apply=False)(
        np.asarray(moving).copy()
    )
    assert isinstance(result, tuple)
    assert result[0] == pytest.approx(-dy, abs=0.5)
    assert result[1] == pytest.approx(-dx, abs=0.5)


def test_georeference_accepts_plain_swath_array() -> None:
    # The GLT supplies the output geometry; the swath input itself is
    # sensor-space, so a plain ndarray is a legitimate carrier. The
    # output is always a georeferenced GeoTensor derived from the GLT.
    # Identity GLT: glt[0] holds source columns, glt[1] source rows.
    glt = GeoTensor(
        np.stack(
            [
                np.tile(np.arange(3, dtype=np.int32), (3, 1)),  # source cols
                np.repeat(np.arange(3, dtype=np.int32), 3).reshape(3, 3),  # rows
            ]
        ),
        transform=Affine(1, 0, 0, 0, -1, 3),
        crs="EPSG:4326",
        fill_value_default=-1,
    )
    swath = np.arange(9, dtype=np.float32).reshape(3, 3)
    out = gz.geom.Georeference(glt=glt)(swath)
    assert isinstance(out, GeoTensor)
    np.testing.assert_array_equal(np.asarray(out), swath)


@pytest.mark.parametrize(
    "make_op",
    [
        lambda: gz.geom.Reproject(dst_crs="EPSG:32629"),
        lambda: gz.geom.ReprojectLike(like=_gt()),
        lambda: gz.geom.ResampleLike(like=_gt()),
        lambda: gz.geom.Resize(shape=(2, 2)),
        lambda: gz.geom.Resample(resolution=(2.0, 2.0)),
        lambda: gz.geom.CropToBounds(bounds=(12, 16, 15, 19), crs="EPSG:4326"),
        lambda: gz.geom.Rasterize(geometries=[box(12, 16, 15, 19)]),
        lambda: gz.geom.Vectorize(),
        lambda: gz.geom.AntimeridianSplit(),
        lambda: gz.geom.GeostationaryParallaxCorrect(
            satellite_lon_deg=0.0, target_height_m=100.0
        ),
    ],
)
def test_geo_dependent_operators_reject_plain_ndarray(make_op) -> None:
    with pytest.raises(TypeError, match="GeoTensor"):
        make_op()(np.zeros((1, 4, 4), dtype=np.float32))


def test_geo_dependent_list_operators_reject_plain_ndarray() -> None:
    arr = np.ones((1, 4, 4), dtype=np.float32)
    with pytest.raises(TypeError, match="GeoTensor"):
        gz.geom.Stitch()([arr])
    with pytest.raises(TypeError, match="GeoTensor"):
        gz.geom.Mosaic()([arr, arr])
    with pytest.raises(TypeError, match="GeoTensor"):
        gz.geom.SegmentStitch()([arr])


def test_4d_time_stack() -> None:
    """Registration rejects a stack; grid ops keep it (#147)."""
    from _helpers import time_stack

    stack = time_stack((2, 3, 8, 8))
    reference = stack.isel({"time": 0})
    for cls in (gz.geom.OpticalFlowTVL1, gz.geom.OpticalFlowILK):
        with pytest.raises(ValueError, match=rf"{cls.__name__} accepts 2-D"):
            cls(reference=reference)(stack)
        with pytest.raises(ValueError, match=rf"{cls.__name__} reference accepts"):
            cls(reference=stack)(reference)
    # Registration reads the band along -3 (band 1, not frame 1).
    flow = gz.geom.OpticalFlowILK(reference=reference, band=2)(reference)
    assert flow.shape == (2, 8, 8)
    padded = gz.geom.PadTo(shape=(10, 10))(stack)
    assert padded.shape == (2, 3, 10, 10)
