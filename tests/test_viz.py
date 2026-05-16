"""Tests for `geotoolz.viz` display operators."""

from __future__ import annotations

import numpy as np
import rasterio
from georeader.geotensor import GeoTensor
from shapely.geometry import Point, Polygon

import geotoolz as gz
from geotoolz.viz import (
    AnnotatePoints,
    AnnotatePolygons,
    ApplyColormap,
    ApplyDiscreteColormap,
    FalseColor,
    GammaCorrect,
    Hillshade,
    Overlay,
    ShadedRelief,
    StretchToUint8,
    SWIRComposite,
    ToDisplayRange,
    TrueColor,
    composite,
    gamma_correct_display,
    hillshade,
    stretch_to_uint8,
)


def _toy_geotensor(values: np.ndarray, attrs: dict | None = None) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:32629",
        fill_value_default=0,
        attrs=attrs,
    )


def test_composite_selects_requested_bands() -> None:
    arr = np.arange(4 * 2 * 2).reshape(4, 2, 2)
    out = composite(arr, [2, 1, 0])
    np.testing.assert_array_equal(out, arr[[2, 1, 0]])


def test_true_color_produces_rgb_order_and_preserves_metadata() -> None:
    gt = _toy_geotensor(
        np.stack([np.full((2, 2), value) for value in [2, 3, 4]], axis=0),
        attrs={"bands": ["B02", "B03", "B04"]},
    )
    out = TrueColor(red="B04", green="B03", blue="B02")(gt)
    assert isinstance(out, GeoTensor)
    assert out.shape == (3, 2, 2)
    assert out.transform == gt.transform
    assert str(out.crs) == "EPSG:32629"
    np.testing.assert_array_equal(np.asarray(out)[:, 0, 0], [4, 3, 2])


def test_false_color_and_swir_composites() -> None:
    gt = _toy_geotensor(np.arange(5 * 2 * 2).reshape(5, 2, 2))
    arr = np.asarray(gt)
    np.testing.assert_array_equal(
        np.asarray(FalseColor(nir=3, red=2, green=1)(gt)), arr[[3, 2, 1]]
    )
    np.testing.assert_array_equal(
        np.asarray(SWIRComposite(swir2=4, nir=3, red=2)(gt)), arr[[4, 3, 2]]
    )


def test_stretch_to_uint8_lower0_upper100_matches_minmax_cast() -> None:
    arr = np.array([[[0.0, 0.5], [1.0, 2.0]]], dtype=np.float32)
    expected = (((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(np.uint8)
    np.testing.assert_array_equal(
        stretch_to_uint8(arr, lower=0.0, upper=100.0),
        expected,
    )
    out = StretchToUint8(lower=0.0, upper=100.0)(_toy_geotensor(arr))
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_to_display_range_alias() -> None:
    arr = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    expected = np.asarray(StretchToUint8(lower=0.0, upper=100.0)(_toy_geotensor(arr)))
    out = ToDisplayRange(lower=0.0, upper=100.0)(_toy_geotensor(arr))
    np.testing.assert_array_equal(np.asarray(out), expected)


def test_apply_colormap_outputs_rgba_and_nan_color() -> None:
    gt = _toy_geotensor(np.array([[0.0, 1.0], [np.nan, 0.5]], dtype=np.float32))
    out = ApplyColormap(name="viridis", vmin=0.0, vmax=1.0)(gt)
    arr = np.asarray(out)
    assert arr.shape == (4, 2, 2)
    assert arr.dtype == np.uint8
    np.testing.assert_array_equal(arr[:, 1, 0], [0, 0, 0, 0])
    assert out.transform == gt.transform


def test_apply_discrete_colormap_outputs_rgba() -> None:
    gt = _toy_geotensor(np.array([[1, 2], [0, 2]], dtype=np.uint8))
    out = ApplyDiscreteColormap(
        mapping={1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)}
    )(gt)
    arr = np.asarray(out)
    np.testing.assert_array_equal(arr[:, 0, 0], [255, 0, 0, 255])
    np.testing.assert_array_equal(arr[:, 1, 0], [0, 0, 0, 0])


def test_gamma_correct_identity() -> None:
    arr = np.linspace(0.0, 1.0, 5)
    np.testing.assert_allclose(gamma_correct_display(arr, gamma=1.0), arr)
    gt = _toy_geotensor(arr.reshape(1, 1, 5))
    np.testing.assert_allclose(np.asarray(GammaCorrect(gamma=1.0)(gt)), np.asarray(gt))


def test_hillshade_sun_overhead_and_flat_dem_are_constant() -> None:
    sloped = np.arange(16, dtype=np.float32).reshape(4, 4)
    overhead = hillshade(sloped, altitude_deg=90.0)
    assert np.unique(overhead).tolist() == [255]

    flat = _toy_geotensor(np.ones((4, 4), dtype=np.float32))
    out = Hillshade()(flat)
    assert np.unique(np.asarray(out)).size == 1
    assert out.transform == flat.transform


def test_shaded_relief_outputs_rgba() -> None:
    dem = _toy_geotensor(np.arange(16, dtype=np.float32).reshape(4, 4))
    out = ShadedRelief(colormap="terrain")(dem)
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8


def test_overlay_alpha_zero_returns_background_unchanged() -> None:
    bg = _toy_geotensor(np.full((3, 2, 2), 10, dtype=np.uint8))
    fg = _toy_geotensor(np.full((4, 2, 2), 255, dtype=np.uint8))
    out = Overlay(alpha=0.0)(bg, fg)
    np.testing.assert_array_equal(np.asarray(out), np.asarray(bg))


def test_overlay_alpha_blends_to_rgba() -> None:
    bg = _toy_geotensor(np.zeros((3, 2, 2), dtype=np.uint8))
    fg = _toy_geotensor(np.full((4, 2, 2), 255, dtype=np.uint8))
    out = Overlay(alpha=0.5)(bg, fg)
    assert out.shape == (4, 2, 2)
    np.testing.assert_array_equal(np.asarray(out)[:3], 127)


def test_annotate_polygons_rasterizes_polygon_outline() -> None:
    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    out = AnnotatePolygons(geometries=[polygon], color=(1.0, 0.0, 0.0, 1.0), width=1)(
        image
    )
    arr = np.asarray(out)
    assert arr.shape == (4, 4, 4)
    assert np.any(arr[0] == 255)
    assert np.all(arr[3, arr[0] == 255] == 255)


def test_annotate_points_draws_marker() -> None:
    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    out = AnnotatePoints(points=np.array([[1.5, 2.5]]), radius=0)(image)
    arr = np.asarray(out)
    np.testing.assert_array_equal(arr[:, 1, 1], [255, 255, 0, 255])


def test_annotate_points_accepts_geodataframe() -> None:
    import geopandas as gpd

    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    points = gpd.GeoDataFrame(geometry=[Point(1.5, 2.5)], crs="EPSG:32629")
    out = AnnotatePoints(points=points, radius=0)(image)
    np.testing.assert_array_equal(np.asarray(out)[:, 1, 1], [255, 255, 0, 255])


def test_viz_module_exported_from_top_level() -> None:
    assert gz.viz.TrueColor is TrueColor
    assert gz.TrueColor is TrueColor
