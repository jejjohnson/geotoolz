"""Tests for `geotoolz.viz` display operators."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import rasterio
from _helpers import fill_pixel_mask, toy_geotensor
from georeader.geotensor import GeoTensor
from shapely.geometry import Point, Polygon

import geotoolz as gz
from geotoolz.viz import (
    AnnotatePoints,
    AnnotatePolygons,
    ApplyColormap,
    ApplyDiscreteColormap,
    Composite,
    FalseColor,
    GammaCorrect,
    Hillshade,
    Overlay,
    ShadedRelief,
    StretchToUint8,
    SWIRComposite,
    TrueColor,
    blend_rgba,
    gamma_correct_display,
    hillshade,
    stretch_to_uint8,
)


def _toy_geotensor(
    values: np.ndarray, attrs: dict | None = None, fill_value_default: Any = -9999
) -> GeoTensor:
    # The fill must not collide with the data: 0.0 is a real value in most
    # fixtures here, and a pixel equal to ``fill_value_default`` is nodata.
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:32629",
        fill_value_default=fill_value_default,
        attrs=attrs,
    )


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
    expected = np.rint(((arr - arr.min()) / (arr.max() - arr.min())) * 255).astype(
        np.uint8
    )
    np.testing.assert_array_equal(
        stretch_to_uint8(arr, lower=0.0, upper=100.0),
        expected,
    )
    out = StretchToUint8(lower=0.0, upper=100.0)(_toy_geotensor(arr))
    assert out.dtype == np.uint8
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

    # A flat DEM has zero slope, so the shading is sin(altitude) everywhere:
    # 255 * sin(45 deg) = 180.3 -> 180 (GDAL / LightSource reference, #124).
    flat = _toy_geotensor(np.ones((4, 4), dtype=np.float32))
    out = Hillshade()(flat)
    assert np.unique(np.asarray(out)).tolist() == [180]
    assert out.transform == flat.transform
    low_sun = Hillshade(altitude_deg=30.0)(flat)
    assert np.unique(np.asarray(low_sun)).tolist() == [round(255 * 0.5)]


def _plane(east_rise: float, south_rise: float, n: int = 6) -> np.ndarray:
    """Tilted plane DEM: z rises by the given amounts per pixel east / south."""
    rows, cols = np.mgrid[0:n, 0:n].astype(np.float64)
    return east_rise * cols + south_rise * rows


def test_hillshade_lights_slopes_facing_the_sun() -> None:
    # Row 0 = north. A plane rising towards the SE faces NW (towards the
    # default az-315 sun): bright; the opposite plane is in shadow.
    nw_facing = _plane(1.0, 1.0)
    se_facing = _plane(-1.0, -1.0)
    assert hillshade(nw_facing)[2, 2] == 251
    assert hillshade(se_facing)[2, 2] == 0
    # N-facing (rises to the south) is lit by a NW sun, S-facing less so.
    assert hillshade(_plane(0.0, 1.0))[2, 2] > hillshade(_plane(0.0, -1.0))[2, 2]
    # E-facing is lit by an eastern sun and shadowed by a western one.
    e_facing = _plane(-1.0, 0.0)
    assert hillshade(e_facing, azimuth_deg=90)[2, 2] > 200
    assert hillshade(e_facing, azimuth_deg=270)[2, 2] == 0


@pytest.mark.parametrize("azimuth_deg", [0.0, 45.0, 90.0, 135.0, 200.0, 315.0])
@pytest.mark.parametrize("altitude_deg", [30.0, 45.0, 60.0])
@pytest.mark.parametrize(
    ("east_rise", "south_rise"),
    [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.0, 1.0), (2.0, -0.5)],
)
def test_hillshade_matches_matplotlib_lightsource(
    azimuth_deg: float, altitude_deg: float, east_rise: float, south_rise: float
) -> None:
    mcolors = pytest.importorskip("matplotlib.colors")
    dem = _plane(east_rise, south_rise)
    ours = hillshade(
        dem,
        x_resolution=2.0,
        y_resolution=3.0,
        azimuth_deg=azimuth_deg,
        altitude_deg=altitude_deg,
    )
    # Planar DEM -> constant intensity, so matplotlib's contrast stretch
    # is a no-op and the two are directly comparable (up to uint8 floor).
    expected = (
        mcolors.LightSource(azdeg=azimuth_deg, altdeg=altitude_deg).hillshade(
            dem, dx=2.0, dy=3.0
        )
        * 255.0
    )
    np.testing.assert_allclose(ours, expected, atol=1.0)


def test_hillshade_operator_uses_north_up_transform() -> None:
    # Hillshade inherits the fixed orientation (resolution from transform).
    nw_facing = _toy_geotensor(_plane(1.0, 1.0).astype(np.float32))
    out = Hillshade(azimuth_deg=315.0, altitude_deg=45.0)(nw_facing)
    assert np.asarray(out)[2, 2] == 251
    relief = ShadedRelief(colormap="gray")(nw_facing)
    shadow = ShadedRelief(colormap="gray")(
        _toy_geotensor(_plane(-1.0, -1.0).astype(np.float32))
    )
    assert np.asarray(relief)[2, 2, :3].sum() > np.asarray(shadow)[2, 2, :3].sum()


def test_shaded_relief_outputs_rgba() -> None:
    dem = _toy_geotensor(np.arange(16, dtype=np.float32).reshape(4, 4))
    out = ShadedRelief(colormap="terrain")(dem)
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8


def test_overlay_alpha_zero_preserves_background_rgb() -> None:
    """alpha=0 keeps background RGB intact while promoting to RGBA."""
    bg = _toy_geotensor(np.full((3, 2, 2), 10, dtype=np.uint8))
    fg = _toy_geotensor(np.full((4, 2, 2), 255, dtype=np.uint8))
    out = Overlay(alpha=0.0)(bg, fg)
    assert out.shape == (4, 2, 2)
    np.testing.assert_array_equal(np.asarray(out)[:3], np.asarray(bg))


def test_overlay_alpha_blends_to_rgba() -> None:
    bg = _toy_geotensor(np.zeros((3, 2, 2), dtype=np.uint8))
    fg = _toy_geotensor(np.full((4, 2, 2), 255, dtype=np.uint8))
    out = Overlay(alpha=0.5)(bg, fg)
    assert out.shape == (4, 2, 2)
    np.testing.assert_array_equal(np.asarray(out)[:3], 128)


def test_ensure_rgba_float_rgb_scales_to_bytes() -> None:
    rgb = np.full((3, 2, 2), 0.5)
    np.testing.assert_array_equal(
        gz.viz.ensure_rgba(rgb)[:, 0, 0], [128, 128, 128, 255]
    )
    gray = np.full((2, 2), 0.5, dtype=np.float32)
    out = gz.viz.ensure_rgba(gray)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, 0, 0], [128, 128, 128, 255])
    rgba = np.full((4, 2, 2), 0.5)
    np.testing.assert_array_equal(gz.viz.ensure_rgba(rgba)[:, 0, 0], [128] * 4)


def test_ensure_rgba_byte_range_inputs_are_not_rescaled() -> None:
    # Float values > 1 are already display-scaled; NaNs map to 0.
    rgb = np.array([[[0.5, 200.0]], [[1.0, 300.0]], [[np.nan, 2.0]]])
    out = gz.viz.ensure_rgba(rgb)
    np.testing.assert_array_equal(out[:, 0, 0], [0, 1, 0, 255])
    np.testing.assert_array_equal(out[:, 0, 1], [200, 255, 2, 255])
    ints = np.ones((3, 2, 2), dtype=np.int16)
    np.testing.assert_array_equal(gz.viz.ensure_rgba(ints)[:, 0, 0], [1, 1, 1, 255])
    u8 = np.full((2, 2), 7, dtype=np.uint8)
    np.testing.assert_array_equal(gz.viz.ensure_rgba(u8)[:, 0, 0], [7, 7, 7, 255])


def test_overlay_float_inputs_not_black() -> None:
    bg = _toy_geotensor(np.full((3, 2, 2), 0.5, dtype=np.float32))
    transparent_fg = _toy_geotensor(np.zeros((4, 2, 2), dtype=np.float32))
    out = np.asarray(Overlay(alpha=0.5)(bg, transparent_fg))
    np.testing.assert_array_equal(out[:, 0, 0], [128, 128, 128, 255])
    # Float grayscale foreground over float RGB background blends to grey.
    fg = _toy_geotensor(np.ones((2, 2), dtype=np.float32))
    out = np.asarray(Overlay(alpha=0.5)(bg, fg))
    assert np.all(out[:3] > 180)
    np.testing.assert_array_equal(out[3], 255)
    zero = np.asarray(Overlay(alpha=0.0)(bg, fg))
    np.testing.assert_array_equal(zero[:, 0, 0], [128, 128, 128, 255])


def test_annotate_float_inputs_not_black() -> None:
    image = _toy_geotensor(np.full((3, 8, 8), 0.5, dtype=np.float32))
    polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    poly = np.asarray(
        AnnotatePolygons(geometries=[polygon], color=(1.0, 0.0, 0.0, 1.0), width=1)(
            image
        )
    )
    points = np.asarray(AnnotatePoints(points=np.array([[1.5, 2.5]]), radius=0)(image))
    grey = np.array([128, 128, 128, 255])[:, None, None]
    for arr in (poly, points):
        # Unannotated pixels keep the float image's grey instead of black.
        assert np.all(arr == grey, axis=0).any()
        assert not np.all(arr[:3] == 0, axis=0).any()
    assert np.any(np.all(poly == np.array([255, 0, 0, 255])[:, None, None], axis=0))
    np.testing.assert_array_equal(points[:, 1, 1], [255, 255, 0, 255])


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


def test_annotate_polygons_width_zero_is_noop() -> None:
    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    out = AnnotatePolygons(geometries=[polygon], width=0)(image)
    np.testing.assert_array_equal(np.asarray(out), gz.viz.ensure_rgba(image))


def test_annotate_points_draws_marker() -> None:
    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    out = AnnotatePoints(points=np.array([[1.5, 2.5]]), radius=0)(image)
    arr = np.asarray(out)
    np.testing.assert_array_equal(arr[:, 1, 1], [255, 255, 0, 255])


def test_annotate_ops_draw_each_frame_of_a_time_stack() -> None:
    rng = np.random.default_rng(0)
    stack = _toy_geotensor(rng.integers(0, 200, (2, 3, 4, 4)).astype(np.uint8))
    polygon = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])
    for op in (
        AnnotatePoints(points=np.array([[1.5, 2.5]]), radius=0),
        AnnotatePolygons(geometries=[polygon], width=1),
    ):
        out = np.asarray(op(stack))
        assert out.shape == (2, 4, 4, 4)
        for t in range(2):
            frame = _toy_geotensor(np.asarray(stack)[t])
            np.testing.assert_array_equal(out[t], np.asarray(op(frame)))


def test_annotate_ops_name_themselves_on_a_non_display_input() -> None:
    image = _toy_geotensor(np.zeros((2, 4, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="AnnotatePoints: display arrays"):
        AnnotatePoints(points=np.array([[1.5, 2.5]]))(image)
    with pytest.raises(ValueError, match="AnnotatePolygons: display arrays"):
        AnnotatePolygons(geometries=[Polygon([(1, 1), (3, 1), (3, 3)])])(image)


def test_annotate_points_accepts_geodataframe() -> None:
    import geopandas as gpd

    image = _toy_geotensor(np.zeros((3, 4, 4), dtype=np.uint8))
    points = gpd.GeoDataFrame(geometry=[Point(1.5, 2.5)], crs="EPSG:32629")
    out = AnnotatePoints(points=points, radius=0)(image)
    np.testing.assert_array_equal(np.asarray(out)[:, 1, 1], [255, 255, 0, 255])


def test_viz_module_exported_from_top_level() -> None:
    assert gz.viz.TrueColor is TrueColor
    assert gz.TrueColor is TrueColor


def test_composite_operator_arbitrary_band_selection_by_index() -> None:
    """`Composite` is the generic band-selection op the named composites wrap."""
    gt = _toy_geotensor(np.arange(5 * 2 * 2).reshape(5, 2, 2))
    arr = np.asarray(gt)
    out = Composite(bands=[4, 0, 2])(gt)
    assert isinstance(out, GeoTensor)
    assert out.transform == gt.transform
    np.testing.assert_array_equal(np.asarray(out), arr[[4, 0, 2]])


def test_composite_operator_resolves_band_names() -> None:
    gt = _toy_geotensor(
        np.stack([np.full((2, 2), v) for v in [10, 20, 30, 40]], axis=0),
        attrs={"bands": ["B02", "B03", "B04", "B08"]},
    )
    out = Composite(bands=["B08", "B04", "B03"])(gt)
    np.testing.assert_array_equal(np.asarray(out)[:, 0, 0], [40, 30, 20])


def test_named_composites_subclass_generic_composite() -> None:
    """The named composites are thin shims so they share Composite's apply path."""
    assert issubclass(TrueColor, Composite)
    assert issubclass(FalseColor, Composite)
    assert issubclass(SWIRComposite, Composite)


def test_stretch_preserves_geometadata_no_mutation() -> None:
    """Stretching must not mutate the carrier's transform or CRS."""
    arr = np.linspace(0.0, 1.0, 4).reshape(1, 2, 2).astype(np.float32)
    gt = _toy_geotensor(arr)
    out = StretchToUint8()(gt)
    assert out.transform == gt.transform
    assert str(out.crs) == str(gt.crs)
    assert out.shape == gt.shape


def test_apply_discrete_colormap_get_config_is_json_safe() -> None:
    """`get_config()` emits ``[[class_id, rgba], ...]`` pairs that reload."""
    op = ApplyDiscreteColormap(
        mapping={1: (1.0, 0.0, 0.0, 1.0), 2: (0.0, 1.0, 0.0, 1.0)}
    )
    cfg = op.get_config()
    # Round-trip through json: pairs keep the int class IDs intact.
    restored = json.loads(json.dumps(cfg))
    assert restored == {
        "mapping": [[1, [1.0, 0.0, 0.0, 1.0]], [2, [0.0, 1.0, 0.0, 1.0]]]
    }
    assert ApplyDiscreteColormap(**restored).mapping == op.mapping


def test_apply_colormap_get_config_references_cmap_by_name() -> None:
    """The colormap is held by string name, not by a live Colormap object."""
    cfg = ApplyColormap(name="viridis", vmin=0.0, vmax=1.0).get_config()
    # JSON-safe round-trip — would fail on a live matplotlib Colormap.
    restored = json.loads(json.dumps(cfg))
    assert restored["name"] == "viridis"
    assert restored["nan_color"] == [0.0, 0.0, 0.0, 0.0]


def test_overlay_alpha_zero_returns_rgba_consistently() -> None:
    """alpha=0 must still produce 4-band RGBA so downstream shape is consistent."""
    bg = _toy_geotensor(np.full((3, 2, 2), 10, dtype=np.uint8))
    fg = _toy_geotensor(np.full((4, 2, 2), 255, dtype=np.uint8))
    out = Overlay(alpha=0.0)(bg, fg)
    assert out.shape == (4, 2, 2)
    # RGB channels untouched, alpha promoted to opaque.
    np.testing.assert_array_equal(np.asarray(out)[:3], 10)


def test_hydra_zen_roundtrip_viz_operators() -> None:
    """YAML-safe viz operators must round-trip through hydra-zen.builds."""
    hydra_zen = pytest.importorskip("hydra_zen")

    cases = [
        (TrueColor, {"red": 2, "green": 1, "blue": 0}),
        (FalseColor, {"nir": 3, "red": 2, "green": 1}),
        (SWIRComposite, {"swir2": 4, "nir": 3, "red": 2}),
        (Composite, {"bands": [2, 1, 0]}),
        (StretchToUint8, {"lower": 1.0, "upper": 99.0, "reduce_axes": None}),
        (GammaCorrect, {"gamma": 1.4}),
        (ApplyColormap, {"name": "viridis", "vmin": 0.0, "vmax": 1.0}),
        (Hillshade, {"azimuth_deg": 315.0, "altitude_deg": 45.0}),
        (Overlay, {"alpha": 0.5, "mode": "alpha"}),
    ]
    for cls, kwargs in cases:
        op = cls(**kwargs)
        cfg = hydra_zen.builds(cls, **op.get_config())
        restored = hydra_zen.instantiate(cfg)
        assert isinstance(restored, cls)


def test_annotate_operators_are_forbid_in_yaml() -> None:
    """Annotate ops hold runtime geometries — flag them as not YAML-safe."""
    assert AnnotatePolygons.forbid_in_yaml is True
    assert AnnotatePoints.forbid_in_yaml is True


def test_gamma_correct_display_normalises_uint8_inputs() -> None:
    """Integer inputs are normalised to [0, 1] before the gamma exponent.

    Without normalisation, ``128 ** 0.5 = ~11.3`` (uint8 round-trip 11);
    with normalisation, ``(128 / 255) ** 0.5 * 255 = ~180``.
    """
    arr = np.array([0, 128, 255], dtype=np.uint8)
    out = gamma_correct_display(arr, gamma=2.0)
    assert out.dtype == np.uint8
    assert out[0] == 0
    assert out[-1] == 255
    expected_mid = round(np.sqrt(128.0 / 255.0) * 255.0)
    assert abs(int(out[1]) - expected_mid) <= 1
    assert int(out[1]) > 150  # would be ~11 without normalisation


def test_gamma_correct_display_inplace_norm_opt_out() -> None:
    """``inplace_norm=False`` skips the normalisation step."""
    arr = np.array([0, 128, 255], dtype=np.uint8)
    out = gamma_correct_display(arr, gamma=2.0, inplace_norm=False)
    # Without normalisation: 128 ** 0.5 = 11.31, cast back through float.
    np.testing.assert_allclose(out, np.sqrt(np.maximum(arr.astype(float), 0.0)))


def test_gamma_correct_display_passes_through_floats() -> None:
    """Float inputs are assumed to already be in [0, 1]."""
    arr = np.array([0.0, 0.5, 1.0], dtype=np.float32)
    out = gamma_correct_display(arr, gamma=2.0)
    np.testing.assert_allclose(out, np.sqrt(arr), rtol=1e-6)


def test_blend_rgba_uses_source_over_alpha_composition() -> None:
    """Output alpha must accumulate via source-over, not ``max``.

    bg alpha = 0.5, fg alpha = 0.5 -> source-over gives
    0.5 + 0.5 * (1 - 0.5) = 0.75, whereas the old ``max`` formula
    would yield 0.5.
    """
    bg = np.full((4, 2, 2), 128, dtype=np.uint8)  # alpha ~ 0.502
    fg = np.full((4, 2, 2), 128, dtype=np.uint8)  # alpha ~ 0.502
    out = blend_rgba(bg, fg, alpha=1.0, mode="alpha")
    # alpha ~ 0.502; source-over -> 0.502 + 0.502 * (1 - 0.502) = 0.752
    # -> uint8 ~ 191. The old `max` formula would have yielded ~128.
    out_alpha = int(out[3, 0, 0])
    assert 188 <= out_alpha <= 194
    assert out_alpha > 150  # rules out the old `max(bg, fg)` result.


@pytest.mark.parametrize(
    ("op", "values"),
    [
        pytest.param(
            Composite(bands=[2, 0]),
            np.arange(12, dtype=np.float32).reshape(3, 2, 2),
            id="composite",
        ),
        pytest.param(
            StretchToUint8(lower=0.0, upper=100.0),
            np.linspace(0.0, 2.0, 12, dtype=np.float32).reshape(3, 2, 2),
            id="stretch-to-uint8",
        ),
        pytest.param(
            GammaCorrect(gamma=2.0),
            np.linspace(0.0, 1.0, 8, dtype=np.float32).reshape(2, 2, 2),
            id="gamma-correct",
        ),
        pytest.param(
            ApplyColormap(name="viridis", vmin=0.0, vmax=1.0),
            np.linspace(0.0, 1.0, 4, dtype=np.float32).reshape(2, 2),
            id="apply-colormap",
        ),
        pytest.param(
            ApplyDiscreteColormap(mapping={1: (1.0, 0.0, 0.0, 1.0)}),
            np.array([[0, 1], [1, 0]], dtype=np.uint8),
            id="apply-discrete-colormap",
        ),
        pytest.param(
            Hillshade(x_resolution=1.0, y_resolution=1.0),
            np.arange(16, dtype=np.float32).reshape(4, 4),
            id="hillshade-explicit-resolution",
        ),
    ],
)
def test_plain_ndarray_in_plain_ndarray_out(op, values) -> None:
    """Plain np.ndarray in -> plain np.ndarray out, matching the GeoTensor path."""
    out = op(values)
    assert type(out) is np.ndarray
    gt_out = op(_toy_geotensor(values))
    assert isinstance(gt_out, GeoTensor)
    np.testing.assert_array_equal(out, np.asarray(gt_out))


def test_overlay_plain_ndarrays_blend_without_metadata() -> None:
    """Two plain-array inputs skip the transform/CRS check and blend as arrays."""
    bg = np.zeros((3, 2, 2), dtype=np.uint8)
    fg = np.full((4, 2, 2), 255, dtype=np.uint8)
    out = Overlay(alpha=0.5)(bg, fg)
    assert type(out) is np.ndarray
    expected = Overlay(alpha=0.5)(_toy_geotensor(bg), _toy_geotensor(fg))
    np.testing.assert_array_equal(out, np.asarray(expected))


@pytest.mark.parametrize(
    ("op", "values"),
    [
        pytest.param(
            Hillshade(),
            np.arange(16, dtype=np.float32).reshape(4, 4),
            id="hillshade-no-resolution",
        ),
        pytest.param(
            ShadedRelief(),
            np.arange(16, dtype=np.float32).reshape(4, 4),
            id="shaded-relief",
        ),
        pytest.param(
            AnnotatePolygons(geometries=[Polygon([(1, 1), (3, 1), (3, 3)])]),
            np.zeros((3, 4, 4), dtype=np.uint8),
            id="annotate-polygons",
        ),
        pytest.param(
            AnnotatePoints(points=np.array([[1.5, 2.5]])),
            np.zeros((3, 4, 4), dtype=np.uint8),
            id="annotate-points",
        ),
    ],
)
def test_geo_dependent_ops_reject_plain_arrays(op, values) -> None:
    """Geo-dependent viz ops raise a clear TypeError on plain-array input."""
    with pytest.raises(TypeError, match="georeferenced GeoTensor"):
        op(values)


def test_composite_string_bands_require_attrs_metadata() -> None:
    """String band refs need a carrier with band names in attrs."""
    with pytest.raises(TypeError, match="band-name metadata"):
        Composite(bands=["B04"])(np.zeros((3, 2, 2), dtype=np.float32))


# ---------------------------------------------------------------------------
# Nodata (fill) pixels
# ---------------------------------------------------------------------------


def _with_fill(values: np.ndarray, fill: Any = -9999) -> GeoTensor:
    values = np.array(values, copy=True)
    values[..., fill_pixel_mask(values.shape)] = fill
    return _toy_geotensor(values, fill_value_default=fill)


def test_fill_pixels_are_excluded() -> None:
    """-9999 fill pixels neither drive stretches / colormaps nor leak."""
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.0, 1.0, size=(3, 6, 6)).astype(np.float32)
    fill = fill_pixel_mask(clean.shape)
    nan_clean = np.where(fill, np.nan, clean)

    # StretchToUint8: percentiles over valid pixels only; fill -> 0.
    out = StretchToUint8()(_with_fill(clean))
    assert out.fill_value_default == 0
    np.testing.assert_array_equal(np.asarray(out)[:, fill], 0)
    expected = stretch_to_uint8(nan_clean)
    np.testing.assert_array_equal(np.asarray(out)[:, ~fill], expected[:, ~fill])
    assert np.asarray(out)[:, ~fill].max() == 255  # -9999 did not squash the range

    # NaN fill is detected too.
    out_nan = StretchToUint8()(_with_fill(clean, fill=np.nan))
    np.testing.assert_array_equal(np.asarray(out_nan), np.asarray(out))

    # ApplyColormap: auto vmin / vmax over valid pixels; fill -> transparent 0.
    band = clean[0]
    rgba = np.asarray(ApplyColormap(name="viridis")(_with_fill(band)))
    np.testing.assert_array_equal(rgba[:, fill], 0)
    ref = ApplyColormap(
        name="viridis", vmin=float(band[~fill].min()), vmax=float(band[~fill].max())
    )(band)
    np.testing.assert_array_equal(rgba[:, ~fill], ref[:, ~fill])

    # ApplyDiscreteColormap: fill renders transparent even if mapped.
    labels = np.ones((6, 6), dtype=np.int16)
    mapping = {1: (1.0, 0.0, 0.0, 1.0), -9999: (0.0, 1.0, 0.0, 1.0)}
    cat = np.asarray(ApplyDiscreteColormap(mapping=mapping)(_with_fill(labels)))
    np.testing.assert_array_equal(cat[:, fill], 0)
    red = np.broadcast_to(np.array([[255], [0], [0], [255]]), (4, int((~fill).sum())))
    np.testing.assert_array_equal(cat[:, ~fill], red)

    # Hillshade: on a planar DEM the fill does not leak into any neighbour.
    yy, xx = np.mgrid[0:6, 0:6].astype(np.float64)
    dem = 3.0 * xx - 2.0 * yy
    shade = np.asarray(Hillshade()(_with_fill(dem)))
    np.testing.assert_array_equal(shade[fill], 0)
    np.testing.assert_array_equal(
        shade[~fill], np.asarray(Hillshade()(_toy_geotensor(dem)))[~fill]
    )

    # GammaCorrect is elementwise: fill values pass through untouched.
    gamma = np.asarray(GammaCorrect(gamma=2.0)(_with_fill(clean)))
    np.testing.assert_array_equal(gamma[:, fill], -9999)
    np.testing.assert_allclose(
        gamma[:, ~fill], np.asarray(GammaCorrect(gamma=2.0)(clean))[:, ~fill]
    )


def test_4d_time_stack() -> None:
    """Composites select along -3; colormaps render each frame (#147)."""
    from _helpers import time_stack

    stack = time_stack()
    rgb = TrueColor(red=2, green=1, blue=0)(stack)
    assert rgb.shape == (2, 3, 4, 4)
    np.testing.assert_array_equal(np.asarray(rgb), np.asarray(stack)[:, [2, 1, 0]])
    single = time_stack((2, 1, 4, 4))
    rgba = gz.viz.ApplyColormap(name="viridis")(single)
    assert rgba.shape == (2, 4, 4, 4)
    np.testing.assert_array_equal(
        np.asarray(rgba)[1],
        np.asarray(gz.viz.ApplyColormap(name="viridis")(single.isel({"time": 1}))),
    )


# --- #155: viz delegates to spectral / radiometry -------------------------


def test_composite_output_band_attrs() -> None:
    """Composites carry the *selected* bands' attrs, in output order."""
    attrs = {
        "band_names": ["B02", "B03", "B04", "B08"],
        "wavelengths": [490.0, 560.0, 665.0, 842.0],
        "sensor": "S2A",
    }
    gt = _toy_geotensor(np.arange(4 * 2 * 2, dtype=np.float32).reshape(4, 2, 2), attrs)
    out = TrueColor(red="B04", green="B03", blue="B02")(gt)
    assert isinstance(Composite(bands=[0]), gz.spectral.SelectBands)
    assert out.attrs is not gt.attrs
    assert out.attrs["band_names"] == ["B04", "B03", "B02"]
    np.testing.assert_array_equal(out.attrs["wavelengths"], [665.0, 560.0, 490.0])
    assert out.attrs["sensor"] == "S2A"
    assert gt.attrs["band_names"] == ["B02", "B03", "B04", "B08"]  # not mutated

    # Same band count, reordered: the names must follow the pixels too.
    rgb = _toy_geotensor(
        np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2),
        {"band_names": ["B02", "B03", "B04"]},
    )
    out = Composite(bands=[2, 1, 0])(rgb)
    assert out.attrs["band_names"] == ["B04", "B03", "B02"]
    np.testing.assert_array_equal(np.asarray(out), np.asarray(rgb)[[2, 1, 0]])
    assert out.fill_value_default == rgb.fill_value_default


@pytest.mark.parametrize("axis", [(-2, -1), None])
def test_stretch_to_uint8_equals_percentile_clip_pipeline(axis: Any) -> None:
    rng = np.random.default_rng(0)
    values = rng.normal(0.3, 0.1, (2, 3, 5, 5)).astype(np.float32)
    values[0, :, 1, 1] = -9999.0  # nodata pixel in frame 0
    values[1, 2, 0, 3] = np.nan
    gt = _toy_geotensor(values)
    out = StretchToUint8(lower=5.0, upper=95.0, reduce_axes=axis)(gt)
    stretched = gz.radiometry.PercentileClip(lower=5.0, upper=95.0, reduce_axes=axis)(
        gt
    )
    expected = np.rint(np.nan_to_num(np.asarray(stretched)) * 255.0).astype(np.uint8)
    assert out.dtype == np.uint8
    assert out.fill_value_default == 0
    np.testing.assert_array_equal(np.asarray(out), expected)
    np.testing.assert_array_equal(np.asarray(out)[0, :, 1, 1], 0)


def test_stretch_keeps_per_frame_nodata_on_time_stacks() -> None:
    """A pixel missing in one frame must stay valid in the others (4-D)."""
    rng = np.random.default_rng(1)
    values = rng.uniform(0.1, 0.9, (2, 3, 4, 4)).astype(np.float32)
    values[0, :, 2, 2] = -9999.0  # nodata in frame 0 only
    gt = _toy_geotensor(values)
    frame1 = StretchToUint8()(_toy_geotensor(values[1]))

    out = np.asarray(StretchToUint8()(gt))
    np.testing.assert_array_equal(out[0, :, 2, 2], 0)
    np.testing.assert_array_equal(out[1], np.asarray(frame1))
    assert (out[1, :, 2, 2] > 0).all()
    clipped = np.asarray(gz.radiometry.PercentileClip()(gt))
    assert np.isfinite(clipped[1, :, 2, 2]).all()


def test_composite_bands_is_the_selected_bands() -> None:
    """``Composite`` shares ``SelectBands``' single ``bands`` list."""
    op = Composite(bands=[0])
    op.bands.append(1)
    assert op.get_config()["bands"] == [0, 1]
    assert not hasattr(op, "indexes")
    op.bands = [2]
    arr = np.arange(3 * 2 * 2, dtype=np.float32).reshape(3, 2, 2)
    np.testing.assert_array_equal(op(arr), arr[[2]])


def test_stretch_to_uint8_rounds_instead_of_truncating() -> None:
    # 0.999 * 255 = 254.7 -> 255 (a bare cast truncated it to 254).
    arr = np.array([[[0.0, 0.999, 1.0]]], dtype=np.float32)
    out = stretch_to_uint8(arr, lower=0.0, upper=100.0)
    np.testing.assert_array_equal(out, [[[0, 255, 255]]])


def test_stretch_vocabulary_is_lower_upper_reduce_axes() -> None:
    for cls in (StretchToUint8, gz.radiometry.PercentileClip):
        cfg = cls(lower=1.0, upper=99.0, reduce_axes=None).get_config()
        assert {"lower", "upper", "reduce_axes"} <= set(cfg)
        assert "axis" not in cfg
    assert {"lower", "upper"} <= set(gz.normalize.HistogramStretch().get_config())
    with pytest.raises(TypeError):
        StretchToUint8(per_band=True)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        gz.radiometry.PercentileClip(p_min=2.0)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="upper > lower"):
        StretchToUint8(lower=50.0, upper=10.0)(np.zeros((1, 2, 2)))


def test_gamma_correct_clips_floats_to_unit_interval() -> None:
    out = gamma_correct_display(np.array([2.0, -1.0, 0.25, np.nan]), gamma=2.0)
    np.testing.assert_allclose(out, [1.0, 0.0, 0.5, np.nan])
    unit = np.linspace(0.0, 1.0, 7).reshape(1, 1, 7)
    np.testing.assert_allclose(
        np.asarray(GammaCorrect(gamma=1.7)(_toy_geotensor(unit))),
        np.asarray(gz.radiometry.Gamma(gamma=1.7)(_toy_geotensor(unit))),
    )
    with pytest.raises(ValueError, match="gamma > 0"):
        gamma_correct_display(np.ones(2), gamma=0.0)


def test_gamma_correct_display_rounds_integer_outputs() -> None:
    arr = np.arange(256, dtype=np.uint8)
    out = gamma_correct_display(arr, gamma=2.2)
    expected = np.rint((arr / 255.0) ** (1 / 2.2) * 255.0).astype(np.uint8)
    np.testing.assert_array_equal(out, expected)


def test_to_display_range_and_composite_primitive_are_removed() -> None:
    assert not hasattr(gz.viz, "ToDisplayRange")
    assert not hasattr(gz, "ToDisplayRange")
    assert not hasattr(gz.viz, "composite")
    assert not hasattr(gz.normalize, "percentile_clip")


def test_annotate_polygons_matches_direct_rasterize() -> None:
    """Delegating to georeader keeps the burned outline pixel-identical."""
    from rasterio.features import rasterize

    image = _toy_geotensor(np.zeros((3, 8, 8), dtype=np.uint8))
    polygons = [
        Polygon([(1, 1), (5, 1), (5, 3), (1, 3)]),
        Polygon([(2.5, 0.5), (6.5, 0.5), (6.5, 3.5)]),
    ]
    out = np.asarray(
        AnnotatePolygons(geometries=polygons, color=(0.0, 1.0, 0.0, 1.0), width=1)(
            image
        )
    )
    shapes = [(poly.boundary.buffer(0.5), 1) for poly in polygons]
    expected = rasterize(
        shapes, out_shape=(8, 8), transform=image.transform, all_touched=True
    ).astype(bool)
    assert expected.any()
    np.testing.assert_array_equal(out[1] == 255, expected)


# ---------------------------------------------------------------------------
# Non-default grids (``_helpers.TOY_GRIDS``)
# ---------------------------------------------------------------------------


def test_hillshade_reads_non_square_pixel_sizes_from_the_transform() -> None:
    """10 m x 20 m pixels: each axis gradient uses its own resolution."""
    dem = _plane(5.0, 2.0, n=8)
    out = Hillshade()(toy_geotensor(dem, grid="non_square"))
    expected = hillshade(dem, x_resolution=10.0, y_resolution=20.0)
    np.testing.assert_array_equal(np.asarray(out), expected)
    # Swapped or single-resolution handling gives a different picture.
    assert not np.array_equal(
        expected, hillshade(dem, x_resolution=20.0, y_resolution=10.0)
    )
    assert not np.array_equal(
        expected, hillshade(dem, x_resolution=10.0, y_resolution=10.0)
    )


def test_hillshade_rejects_a_geographic_crs() -> None:
    dem = toy_geotensor(_plane(5.0, 5.0, n=8), grid="geographic")
    with pytest.raises(ValueError, match=r"Hillshade.*projected"):
        Hillshade()(dem)
    with pytest.raises(ValueError, match=r"Hillshade.*projected"):
        ShadedRelief()(dem)
    # Explicit resolutions are the caller's statement of the units.
    out = Hillshade(x_resolution=10.0, y_resolution=10.0)(dem)
    np.testing.assert_array_equal(
        np.asarray(out),
        hillshade(np.asarray(dem), x_resolution=10.0, y_resolution=10.0),
    )


def test_hillshade_reads_ground_pixel_steps_on_a_rotated_grid() -> None:
    """The rotated 10 m grid has |a| = |e| = 8.66; the steps are still 10 m."""
    dem = _plane(5.0, 2.0, n=8)
    out = Hillshade()(toy_geotensor(dem, grid="rotated"))
    np.testing.assert_array_equal(
        np.asarray(out), hillshade(dem, x_resolution=10.0, y_resolution=10.0)
    )


def test_hillshade_rejects_a_sheared_grid() -> None:
    with pytest.raises(ValueError, match=r"Hillshade.*sheared"):
        Hillshade()(toy_geotensor(_plane(5.0, 2.0, n=8), grid="sheared"))
