"""Tests for `geotoolz.mask`."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from shapely.geometry import box

from geotoolz.mask import (
    AltitudeMask,
    ApplyMask,
    BBoxMask,
    BufferMask,
    CleanMask,
    CloseMask,
    CombineMasks,
    CountryMask,
    DilateMask,
    DistanceMask,
    ErodeMask,
    InvertMask,
    LandMask,
    OceanMask,
    OpenMask,
    PolygonMask,
    RemoveSmallHoles,
    RemoveSmallObjects,
    SlopeMask,
    combine_masks,
)
from geotoolz.mask._src import operators as mask_operators
from geotoolz.mask._src.array import (
    buffer_mask,
    close_mask,
    combine_masks as combine_masks_array,
    erode_mask,
    open_mask,
)


def _toy_geotensor(values: np.ndarray) -> GeoTensor:
    return GeoTensor(
        values=values,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:3857",
        fill_value_default=-9999,
    )


def test_polygon_mask_inside_and_outside_are_complements() -> None:
    gt = _toy_geotensor(np.zeros((1, 4, 4), dtype=np.float32))
    polygon = box(1.0, 1.0, 3.0, 3.0)

    inside = PolygonMask(geometry=polygon, inside=True)(gt)
    outside = PolygonMask(geometry=polygon, inside=False)(gt)

    np.testing.assert_array_equal(np.asarray(outside), ~np.asarray(inside))
    assert inside.dtype == bool
    assert inside.transform == gt.transform
    assert inside.crs == gt.crs


def test_bbox_mask_matches_expected_pixel_centers() -> None:
    gt = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))

    mask = BBoxMask(bounds=(1.0, 1.0, 3.0, 3.0))(gt)

    expected = np.array(
        [
            [False, False, False, False],
            [False, True, True, False],
            [False, True, True, False],
            [False, False, False, False],
        ]
    )
    np.testing.assert_array_equal(np.asarray(mask), expected)


def test_polygon_mask_accepts_geodataframe() -> None:
    gt = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))
    gdf = gpd.GeoDataFrame(geometry=[box(1.0, 1.0, 3.0, 3.0)], crs=gt.crs)

    mask = PolygonMask(geometry=gdf)(gt)

    assert mask.shape == gt.shape
    assert bool(np.asarray(mask)[1, 1])
    assert not bool(np.asarray(mask)[0, 0])
    assert PolygonMask(geometry=gdf).get_config()["geometry"]["type"] == "GeoDataFrame"


def test_distance_mask_inside_false_is_complement() -> None:
    gt = _toy_geotensor(np.zeros((5, 5), dtype=np.float32))
    polygon = box(2.0, 2.0, 3.0, 3.0)

    inside = DistanceMask(geometry=polygon, distance=1.0, inside=True)(gt)
    outside = DistanceMask(geometry=polygon, distance=1.0, inside=False)(gt)

    np.testing.assert_array_equal(np.asarray(outside), ~np.asarray(inside))


def test_buffer_mask_pixels_matches_euclidean_radius() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    out = BufferMask(radius=1, unit="pixels")(mask)

    expected = np.zeros((5, 5), dtype=bool)
    expected[2, 2] = True
    expected[1, 2] = True
    expected[2, 1] = True
    expected[2, 3] = True
    expected[3, 2] = True
    np.testing.assert_array_equal(out, expected)


def test_array_helpers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        combine_masks_array([], "or")
    with pytest.raises(ValueError, match="exactly one"):
        combine_masks_array([np.zeros((1, 1)), np.ones((1, 1))], "not")
    with pytest.raises(ValueError, match="unit"):
        buffer_mask(np.zeros((1, 1)), 1, unit="feet")
    with pytest.raises(ValueError, match="2D"):
        erode_mask(np.zeros((3, 3)), structure=np.ones((1, 1, 1)))


def test_dilate_mask_then_remove_small_objects() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    mask[0, 0] = True

    dilated = DilateMask(iterations=1)(mask)
    cleaned = RemoveSmallObjects(min_size=5)(dilated)

    assert bool(cleaned[3, 3])
    assert not bool(cleaned[0, 0])


def test_erode_open_close_operators_smoke() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[0, 0] = False

    eroded = ErodeMask(iterations=1)(mask)
    opened = OpenMask(iterations=1)(mask)
    closed = CloseMask(iterations=0)(mask)

    assert eroded.dtype == bool
    assert opened.dtype == bool
    np.testing.assert_array_equal(closed, close_mask(mask, iterations=0))
    assert open_mask(mask).dtype == bool


def test_remove_small_holes_fills_enclosed_hole_only() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False
    mask[0, 0] = False

    out = RemoveSmallHoles(area_threshold=1)(mask)

    assert bool(out[2, 2])
    assert not bool(out[0, 0])


def test_clean_mask_returns_boolean() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False

    out = CleanMask(max_hole_size=1, min_object_size=1, close_iter=0)(mask)

    assert out.dtype == bool
    assert bool(out[2, 2])


def test_combine_masks_or_is_commutative_and_operator_preserves_metadata() -> None:
    a = np.array([[True, False], [False, False]])
    b = np.array([[False, False], [True, False]])
    gt_a = _toy_geotensor(a)
    gt_b = _toy_geotensor(b)

    np.testing.assert_array_equal(
        combine_masks([a, b], "or"), combine_masks([b, a], "or")
    )
    out = CombineMasks(op="or")([gt_a, gt_b])

    assert isinstance(out, GeoTensor)
    assert out.transform == gt_a.transform
    np.testing.assert_array_equal(np.asarray(out), [[True, False], [True, False]])


def test_combine_masks_xor_and_not() -> None:
    a = np.array([[True, False], [False, False]])
    b = np.array([[True, True], [False, False]])

    xor = CombineMasks(op="xor")([a, b])
    not_a = CombineMasks(op="not")([a])

    np.testing.assert_array_equal(xor, [[False, True], [False, False]])
    np.testing.assert_array_equal(not_a, [[False, True], [True, True]])


def test_invert_mask_operator() -> None:
    mask = np.array([[True, False]])

    out = InvertMask()(mask)

    np.testing.assert_array_equal(out, [[False, True]])


def test_apply_mask_preserves_metadata_and_changes_only_masked_pixels() -> None:
    gt = _toy_geotensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    mask = np.array([[True, False], [False, True]])

    out = ApplyMask(mask=mask, fill=-1.0)(gt)

    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.shape == gt.shape
    arr = np.asarray(out)
    assert np.all(arr[:, 0, 0] == -1.0)
    assert np.all(arr[:, 1, 1] == -1.0)
    np.testing.assert_array_equal(arr[:, 0, 1], np.asarray(gt)[:, 0, 1])


def test_apply_mask_get_config_for_array_and_operator_masks() -> None:
    array_op = ApplyMask(mask=np.array([[True]]), fill=0.0)
    operator_op = ApplyMask(mask=BBoxMask(bounds=(0.0, 0.0, 1.0, 1.0)), fill=0.0)

    assert array_op.get_config()["mask"]["dtype"] == "bool"
    assert operator_op.get_config()["mask"]["class"] == "BBoxMask"


def test_altitude_mask_bounds() -> None:
    dem = _toy_geotensor(np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32))
    scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))

    out = AltitudeMask(dem=dem, min_elev=5.0, max_elev=20.0)(scene)

    np.testing.assert_array_equal(np.asarray(out), [[False, True], [True, False]])


def test_slope_mask_matches_flat_reference() -> None:
    dem = _toy_geotensor(np.ones((4, 4), dtype=np.float32) * 100.0)
    scene = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))

    out = SlopeMask(dem=dem, max_slope_deg=0.5)(scene)

    assert np.all(np.asarray(out))


def test_natural_earth_mask_constructors_use_cached_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    countries = gpd.GeoDataFrame(
        {"ISO_A3": ["GRL"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )

    def fake_read_file(source: str) -> gpd.GeoDataFrame:
        calls.append(source)
        return countries

    mask_operators._load_natural_earth.cache_clear()
    monkeypatch.setattr(mask_operators.gpd, "read_file", fake_read_file)
    source = "/tmp/natural-earth.gpkg"

    assert LandMask(source=source).get_config() == {"source": source}
    assert OceanMask(source=source).get_config() == {"source": source}
    country = CountryMask(iso_a3="GRL", source=source)
    assert country.get_config() == {
        "iso_a3": "GRL",
        "source": source,
    }
    scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
    country_mask = country(scene)
    assert country_mask.shape == scene.shape
    assert bool(np.asarray(country_mask)[1, 0])
    assert calls == [source, source, source]
    mask_operators._load_natural_earth.cache_clear()


def test_country_mask_rejects_unknown_iso(monkeypatch: pytest.MonkeyPatch) -> None:
    countries = gpd.GeoDataFrame(
        {"ISO_A3": ["GRL"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )
    monkeypatch.setattr(
        mask_operators, "_load_natural_earth", lambda _kind, _source: countries
    )

    with pytest.raises(ValueError, match="no countries"):
        CountryMask(iso_a3="USA")
