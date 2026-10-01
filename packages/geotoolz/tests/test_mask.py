"""Tests for `geotoolz.mask`."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from georeader.geotensor import GeoTensor
from pipekit import Operator
from shapely.geometry import MultiPolygon, box

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
    altitude_mask,
    apply_mask,
    combine_masks,
    distance_mask,
    invert_mask,
    slope_mask,
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


_BOX_INTERIOR = np.array(
    [
        [False, False, False, False],
        [False, True, True, False],
        [False, True, True, False],
        [False, False, False, False],
    ]
)


def test_polygon_mask_keep_inside_and_outside_are_complements() -> None:
    gt = _toy_geotensor(np.zeros((1, 4, 4), dtype=np.float32))
    polygon = box(1.0, 1.0, 3.0, 3.0)

    keep_inside = PolygonMask(geometry=polygon, keep="inside")(gt)
    keep_outside = PolygonMask(geometry=polygon, keep="outside")(gt)

    np.testing.assert_array_equal(np.asarray(keep_outside), ~np.asarray(keep_inside))
    # Drop polarity: keep="inside" masks (True) everything outside the polygon.
    np.testing.assert_array_equal(np.asarray(keep_inside), ~_BOX_INTERIOR)
    assert keep_inside.dtype == bool
    assert keep_inside.transform == gt.transform
    assert keep_inside.crs == gt.crs


def test_bbox_mask_default_drops_outside_the_box() -> None:
    gt = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))

    mask = BBoxMask(bounds=(1.0, 1.0, 3.0, 3.0))(gt)
    dropped_box = BBoxMask(bounds=(1.0, 1.0, 3.0, 3.0), keep="outside")(gt)

    np.testing.assert_array_equal(np.asarray(mask), ~_BOX_INTERIOR)
    np.testing.assert_array_equal(np.asarray(dropped_box), _BOX_INTERIOR)


def test_apply_mask_with_geometry_mask_keeps_aoi() -> None:
    """#154: ``ApplyMask(mask=BBoxMask(...))`` keeps the AOI by default."""
    values = np.arange(1, 17, dtype=np.float32).reshape(1, 4, 4)
    gt = _toy_geotensor(values)

    out = ApplyMask(mask=BBoxMask(bounds=(1.0, 1.0, 3.0, 3.0)), fill_value=0.0)(gt)

    arr = np.asarray(out)[0]
    np.testing.assert_array_equal(arr[_BOX_INTERIOR], values[0][_BOX_INTERIOR])
    assert np.all(arr[~_BOX_INTERIOR] == 0.0)


def test_geometry_masks_reject_unknown_keep() -> None:
    with pytest.raises(ValueError, match="'inside' or 'outside'"):
        BBoxMask(bounds=(0.0, 0.0, 1.0, 1.0), keep="both")  # type: ignore[arg-type]
    dem = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="'inside' or 'outside'"):
        AltitudeMask(dem=dem, min_elev=0.0, keep="nope")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="'inside' or 'outside'"):
        slope_mask(np.zeros((2, 2)), (1.0, 1.0), max_slope_deg=1.0, keep="x")  # type: ignore[arg-type]


def test_polygon_mask_accepts_geodataframe() -> None:
    gt = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))
    gdf = gpd.GeoDataFrame(geometry=[box(1.0, 1.0, 3.0, 3.0)], crs=gt.crs)

    mask = PolygonMask(geometry=gdf)(gt)

    assert mask.shape == gt.shape
    # Drop polarity: the polygon interior is kept (False).
    assert not bool(np.asarray(mask)[1, 1])
    assert bool(np.asarray(mask)[0, 0])
    assert PolygonMask(geometry=gdf).get_config()["geometry"]["type"] == "GeoDataFrame"


def test_distance_mask_keep_outside_is_complement() -> None:
    gt = _toy_geotensor(np.zeros((5, 5), dtype=np.float32))
    polygon = box(2.0, 2.0, 3.0, 3.0)

    keep_inside = DistanceMask(geometry=polygon, distance=1.0, keep="inside")(gt)
    keep_outside = DistanceMask(geometry=polygon, distance=1.0, keep="outside")(gt)

    np.testing.assert_array_equal(np.asarray(keep_outside), ~np.asarray(keep_inside))
    # The geometry pixel itself is within distance, so it is kept by default.
    assert not bool(np.asarray(keep_inside)[2, 2])
    assert bool(np.asarray(keep_inside)[0, 0])


@pytest.mark.parametrize("pixel_size", [(1.0, 1.0), (2.0, 0.5)])
@pytest.mark.parametrize("distance", [0.0, 1.0, 1.5, 3.0])
def test_distance_mask_is_buffer_mask_complement(distance, pixel_size) -> None:
    """``distance_mask`` is ``buffer_mask`` (+ the drop-polarity flip); the
    numerics equal the pre-refactor distance-transform threshold."""
    import scipy.ndimage as ndi

    geom = np.zeros((7, 7), dtype=bool)
    geom[3, 2:4] = True

    buffered = buffer_mask(geom, distance, unit="meters", pixel_size=pixel_size)
    reference = ndi.distance_transform_edt(~geom, sampling=pixel_size) <= distance

    np.testing.assert_array_equal(buffered, reference)
    np.testing.assert_array_equal(
        distance_mask(geom, distance, pixel_size=pixel_size), ~reference
    )
    np.testing.assert_array_equal(
        distance_mask(geom, distance, keep="outside", pixel_size=pixel_size),
        reference,
    )


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


def test_buffer_mask_meters_requires_geotensor_carrier() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[2, 2] = True

    with pytest.raises(ValueError, match="requires a GeoTensor"):
        BufferMask(radius=1.0, unit="meters")(mask)


def test_array_helpers_validate_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        combine_masks_array([], "or")
    with pytest.raises(ValueError, match="'or', 'and', 'xor'"):
        combine_masks_array([np.zeros((1, 1))], "not")
    with pytest.raises(ValueError, match="share one shape"):
        combine_masks_array([np.zeros((2, 2)), np.zeros((3, 2, 2))], "or")
    with pytest.raises(ValueError, match="unit"):
        buffer_mask(np.zeros((1, 1)), 1, unit="feet")
    with pytest.raises(ValueError, match="2D"):
        erode_mask(np.zeros((3, 3)), structure=np.ones((1, 1, 1)))


def test_dilate_mask_then_remove_small_objects() -> None:
    mask = np.zeros((7, 7), dtype=bool)
    mask[3, 3] = True
    mask[0, 0] = True

    dilated = DilateMask(iterations=1)(mask)
    cleaned = RemoveSmallObjects(min_area_px=5)(dilated)

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

    out = RemoveSmallHoles(max_hole_area_px=1)(mask)

    assert bool(out[2, 2])
    assert not bool(out[0, 0])


def test_clean_mask_returns_boolean() -> None:
    mask = np.ones((5, 5), dtype=bool)
    mask[2, 2] = False

    out = CleanMask(max_hole_area_px=1, min_area_px=1, close_iter=0)(mask)

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


def test_combine_masks_xor_and_invert() -> None:
    a = np.array([[True, False], [False, False]])
    b = np.array([[True, True], [False, False]])

    xor = CombineMasks(op="xor")([a, b])
    and_ = CombineMasks(op="AND")([a, b])
    not_a = InvertMask()(a)

    np.testing.assert_array_equal(xor, [[False, True], [False, False]])
    np.testing.assert_array_equal(and_, [[True, False], [False, False]])
    np.testing.assert_array_equal(not_a, [[False, True], [True, True]])
    np.testing.assert_array_equal(invert_mask(a), not_a)


@pytest.mark.parametrize(
    "op",
    [
        DilateMask(iterations=1),
        ErodeMask(iterations=1),
        BufferMask(radius=1.0, unit="pixels"),
        RemoveSmallObjects(min_area_px=2),
        RemoveSmallHoles(max_hole_area_px=1),
        CleanMask(min_area_px=1, max_hole_area_px=1, close_iter=1),
        InvertMask(),
    ],
    ids=lambda op: type(op).__name__,
)
def test_plain_ndarray_in_plain_ndarray_out(op) -> None:
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True
    mask[0, 5] = True

    out = op(mask)
    gt_out = op(_toy_geotensor(mask))

    assert type(out) is np.ndarray
    assert out.dtype == bool
    assert isinstance(gt_out, GeoTensor)
    np.testing.assert_array_equal(out, np.asarray(gt_out))


def test_invert_mask_operator() -> None:
    mask = np.array([[True, False]])

    out = InvertMask()(mask)

    np.testing.assert_array_equal(out, [[False, True]])


def test_apply_mask_preserves_metadata_and_changes_only_masked_pixels() -> None:
    gt = _toy_geotensor(np.arange(8, dtype=np.float32).reshape(2, 2, 2))
    mask = np.array([[True, False], [False, True]])

    out = ApplyMask(mask=mask, fill_value=-1.0)(gt)

    assert out.transform == gt.transform
    assert out.crs == gt.crs
    assert out.shape == gt.shape
    arr = np.asarray(out)
    assert np.all(arr[:, 0, 0] == -1.0)
    assert np.all(arr[:, 1, 1] == -1.0)
    np.testing.assert_array_equal(arr[:, 0, 1], np.asarray(gt)[:, 0, 1])


def test_apply_mask_get_config_for_array_and_operator_masks() -> None:
    array_op = ApplyMask(mask=np.array([[True]]), fill_value=0.0)
    operator_op = ApplyMask(mask=BBoxMask(bounds=(0.0, 0.0, 1.0, 1.0)), fill_value=0.0)

    array_cfg = array_op.get_config()
    operator_cfg = operator_op.get_config()
    assert array_cfg["mask"]["dtype"] == "bool"
    assert array_cfg["fill_value"] == 0.0
    assert operator_cfg["mask"]["class"] == "BBoxMask"


def test_apply_mask_explicit_fill_is_declared_and_marks_nodata() -> None:
    """Issue #146: ``fill_value=0.0`` wrote 0.0 but kept -9999, so
    ``validmask()`` reported every pixel valid."""
    values = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    values[:, 1, 0] = -9999  # an existing nodata pixel
    gt = _toy_geotensor(values)
    mask = np.array([[True, False], [False, False]])

    out = ApplyMask(mask=mask, fill_value=0.0)(gt)

    assert out.fill_value_default == 0.0
    arr = np.asarray(out)
    assert np.all(arr[:, 0, 0] == 0.0)
    # The input's own nodata is rewritten to the declared fill.
    assert np.all(arr[:, 1, 0] == 0.0)
    np.testing.assert_array_equal(
        np.asarray(out.validmask()).all(axis=0), [[False, True], [False, True]]
    )


def test_apply_mask_default_fill_is_the_carrier_fill() -> None:
    mask = np.array([[True, False], [False, False]])

    # A float carrier with a -9999 fill is masked with -9999.
    out = ApplyMask(mask=mask)(_toy_geotensor(np.ones((2, 2), dtype=np.float32)))
    assert out.fill_value_default == -9999
    assert np.asarray(out)[0, 0] == -9999

    # A uint16 carrier with a usable fill is not upcast to float64.
    dn = GeoTensor(
        np.full((2, 2), 100, dtype=np.uint16),
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:3857",
        fill_value_default=0,
    )
    out = ApplyMask(mask=mask)(dn)
    assert out.dtype == np.uint16
    assert out.fill_value_default == 0
    np.testing.assert_array_equal(np.asarray(out), [[0, 100], [100, 100]])


def test_apply_mask_default_fill_falls_back_to_nan() -> None:
    mask = np.array([[True, False]])
    gt = GeoTensor(
        np.ones((1, 2), dtype=np.float32),
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:3857",
        fill_value_default=None,
    )

    out = ApplyMask(mask=mask)(gt)

    assert np.isnan(out.fill_value_default)
    assert np.isnan(np.asarray(out)[0, 0])
    assert np.isnan(ApplyMask(mask=mask)(np.ones((1, 2)))[0, 0])
    assert ApplyMask(mask=mask).get_config()["fill_value"] is None


def test_apply_mask_broadcasts_2d_mask_against_3d_carrier() -> None:
    gt = _toy_geotensor(np.ones((3, 4, 4), dtype=np.float32))
    mask = np.zeros((4, 4), dtype=bool)
    mask[0, 0] = True

    out = ApplyMask(mask=mask, fill_value=0.0)(gt)
    arr = np.asarray(out)

    assert arr.shape == (3, 4, 4)
    assert np.all(arr[:, 0, 0] == 0.0)
    assert np.all(arr[:, 1:, 1:] == 1.0)


def test_polygon_mask_handles_multipolygon() -> None:
    gt = _toy_geotensor(np.zeros((5, 5), dtype=np.float32))
    multi = MultiPolygon([box(0.0, 0.0, 1.0, 1.0), box(3.0, 3.0, 4.0, 4.0)])

    mask = np.asarray(PolygonMask(geometry=multi, keep="outside")(gt))

    # Two disjoint True components, one per polygon in the MultiPolygon.
    assert bool(mask[3, 0])  # bottom-left polygon (row 3 = y in [0,1])
    assert bool(mask[0, 3])  # top-right polygon (row 0 = y in [3,4])
    assert not bool(mask[2, 2])  # gap in the middle


def test_polygon_mask_get_config_emits_geojson_dict() -> None:
    polygon = box(1.0, 1.0, 3.0, 3.0)

    cfg = PolygonMask(geometry=polygon, keep="outside").get_config()

    assert cfg["keep"] == "outside"
    assert "inside" not in cfg
    assert cfg["geometry"]["type"] == "Polygon"
    assert "coordinates" in cfg["geometry"]
    # JSON-safe: nested tuples coerced to lists.
    assert isinstance(cfg["geometry"]["coordinates"], list)


def test_morphology_operators_do_not_share_state_via_inheritance() -> None:
    # ErodeMask / CloseMask should subclass Operator directly, not their
    # dilate / open siblings — otherwise isinstance checks would lie.
    assert not isinstance(ErodeMask(), DilateMask)
    assert not isinstance(CloseMask(), OpenMask)
    assert not isinstance(
        SlopeMask(dem=_toy_geotensor(np.zeros((2, 2))), max_slope_deg=1.0), AltitudeMask
    )


def test_altitude_mask_bounds() -> None:
    dem = _toy_geotensor(np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32))
    scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))

    out = AltitudeMask(dem=dem, min_elev=5.0, max_elev=20.0)(scene)
    dropped_band = AltitudeMask(dem=dem, min_elev=5.0, max_elev=20.0, keep="outside")

    # Drop polarity: cells inside [5, 20] are kept (False).
    np.testing.assert_array_equal(np.asarray(out), [[True, False], [False, True]])
    np.testing.assert_array_equal(
        np.asarray(dropped_band(scene)), [[False, True], [True, False]]
    )
    np.testing.assert_array_equal(
        altitude_mask(np.asarray(dem), min_elev=5.0, max_elev=20.0), np.asarray(out)
    )


def test_slope_mask_matches_flat_reference() -> None:
    dem = _toy_geotensor(np.ones((4, 4), dtype=np.float32) * 100.0)
    scene = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))

    out = SlopeMask(dem=dem, max_slope_deg=0.5)(scene)
    steep_only = SlopeMask(dem=dem, max_slope_deg=0.5, keep="outside")(scene)

    # A flat DEM is inside the slope interval everywhere, so nothing drops.
    assert not np.any(np.asarray(out))
    assert np.all(np.asarray(steep_only))


def test_dem_masks_reject_mismatched_grid() -> None:
    dem_values = np.array([[0.0, 10.0], [20.0, 30.0]], dtype=np.float32)
    dem = GeoTensor(
        values=dem_values,
        transform=rasterio.Affine(2.0, 0.0, 100.0, 0.0, -2.0, 200.0),
        crs="EPSG:3857",
        fill_value_default=-9999,
    )
    scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))

    with pytest.raises(ValueError, match="DEM grid"):
        AltitudeMask(dem=dem, min_elev=5.0, max_elev=20.0)(scene)

    flat_dem = GeoTensor(
        values=np.ones((4, 4), dtype=np.float32) * 100.0,
        transform=rasterio.Affine(1.0, 0.0, 0.0, 0.0, -1.0, 4.0),
        crs="EPSG:4326",
        fill_value_default=-9999,
    )
    flat_scene = _toy_geotensor(np.zeros((4, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="DEM grid"):
        SlopeMask(dem=flat_dem, max_slope_deg=0.5)(flat_scene)


def test_natural_earth_mask_constructors_use_cached_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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
    source = str(tmp_path / "natural-earth.gpkg")

    try:
        assert LandMask(source=source).get_config() == {
            "source": source,
            "keep": "inside",
        }
        assert OceanMask(source=source, keep="outside").get_config() == {
            "source": source,
            "keep": "outside",
        }
        country = CountryMask(iso_a3="GRL", source=source)
        assert country.get_config() == {
            "iso_a3": "GRL",
            "source": source,
            "keep": "inside",
        }
        scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
        country_mask = country(scene)
        assert country_mask.shape == scene.shape
        # Drop polarity: pixels inside the country are kept (False).
        assert not bool(np.asarray(country_mask)[1, 0])
        dropped = CountryMask(iso_a3="GRL", source=source, keep="outside")(scene)
        np.testing.assert_array_equal(np.asarray(dropped), ~np.asarray(country_mask))
        # Land / Ocean never ran, so never loaded; the two CountryMasks share
        # one process-cached read of the countries layer.
        assert calls == [source]
        mask_operators.clear_natural_earth_cache()
        CountryMask(iso_a3="GRL", source=source)(scene)
        assert calls == [source, source]
    finally:
        mask_operators._load_natural_earth.cache_clear()


def test_natural_earth_not_loaded_in_ctor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Building / configuring / hydrating a Natural Earth mask does no I/O."""

    def boom(_kind: str, _source: str) -> gpd.GeoDataFrame:
        raise AssertionError("Natural Earth loaded eagerly")

    monkeypatch.setattr(mask_operators, "_load_natural_earth", boom)
    for op in (
        LandMask(),
        OceanMask(keep="outside"),
        CountryMask(iso_a3=("ESP", "PRT")),
    ):
        state = json.loads(json.dumps(op.state))
        rebuilt = Operator.from_state(state)
        assert type(rebuilt) is type(op)
        assert rebuilt.get_config() == op.get_config()
    scene = _toy_geotensor(np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(AssertionError, match="eagerly"):
        LandMask()(scene)


def test_country_mask_rejects_unknown_iso(monkeypatch: pytest.MonkeyPatch) -> None:
    countries = gpd.GeoDataFrame(
        {"ISO_A3": ["GRL"], "geometry": [box(0.0, 0.0, 1.0, 1.0)]},
        crs="EPSG:4326",
    )
    monkeypatch.setattr(
        mask_operators, "_load_natural_earth", lambda _kind, _source: countries
    )

    op = CountryMask(iso_a3="USA")  # codes are checked on first use
    with pytest.raises(ValueError, match="no countries"):
        op(_toy_geotensor(np.zeros((2, 2), dtype=np.float32)))


def test_4d_time_stack() -> None:
    """Geometry masks are one (H, W) grid mask for every frame (#147)."""
    from _helpers import frames, time_stack

    stack = time_stack()
    bounds = (500_000.0, 3_999_980.0, 500_020.0, 4_000_000.0)
    mask = BBoxMask(bounds=bounds)(stack)
    assert mask.shape == (4, 4)
    np.testing.assert_array_equal(
        np.asarray(mask), np.asarray(BBoxMask(bounds=bounds)(frames(stack)[0]))
    )
    # Morphology runs on every (H, W) plane of a stack.
    planes = np.zeros((2, 1, 5, 5), dtype=bool)
    planes[0, 0, 2, 2] = True
    dilated = DilateMask(iterations=1)(planes)
    assert dilated.shape == planes.shape
    assert dilated[0, 0].sum() == 9 and not dilated[1].any()


def test_apply_mask_primitive_fills_where_true_and_broadcasts() -> None:
    arr = np.array([[1.0, 2.0, 3.0]])
    np.testing.assert_array_equal(
        apply_mask(arr, np.array([[True, False, True]]), fill_value=-1.0),
        [[-1.0, 2.0, -1.0]],
    )
    cube = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])  # (2 bands, 1, 2)
    np.testing.assert_array_equal(
        apply_mask(cube, np.array([[True, False]]), fill_value=0.0),
        [[[0.0, 2.0]], [[0.0, 4.0]]],
    )


def test_apply_mask_preserves_float32_with_nan_fill() -> None:
    """Regression: fill_value=np.nan used to upcast float32 -> float64."""
    gt = _toy_geotensor(np.ones((2, 2, 2), dtype=np.float32))
    out = ApplyMask(mask=np.array([[True, False], [False, False]]), fill_value=np.nan)(
        gt
    )
    assert out.dtype == np.float32
    assert np.all(np.isnan(np.asarray(out)[:, 0, 0]))


def test_apply_mask_get_config_is_jsonable_and_has_no_invert() -> None:
    import json

    op = ApplyMask(mask=BBoxMask(bounds=(0.0, 0.0, 1.0, 1.0)), fill_value=float("nan"))
    decoded = json.loads(json.dumps(op.get_config()))
    assert decoded["mask"]["class"] == "BBoxMask"
    assert decoded["mask"]["config"]["keep"] == "inside"
    assert "invert" not in decoded
    assert ApplyMask.forbid_in_yaml is True
    with pytest.raises(TypeError):
        ApplyMask(mask=np.array([[True]]), invert=True)  # type: ignore[call-arg]
