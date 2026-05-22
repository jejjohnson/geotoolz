"""Tests for region registry and region-backed subsetting."""

from __future__ import annotations

import json

import numpy as np
import pytest
import regionmask
import xarray as xr
from shapely.geometry import MultiPolygon, Polygon

from xrtoolz.geo import (
    bbox_region,
    custom_region,
    load_region_file,
    polygon_from_geojson,
    region_from_dict,
    region_to_dict,
    resolve_region,
    subset_to_region,
)
from xrtoolz.geo.operators import SubsetToRegion
from xrtoolz.viz._src.projections import PRESETS, _resolve_projection


def _rectilinear_ds() -> xr.Dataset:
    lon = np.arange(-90.0, -35.0, 5.0)
    lat = np.arange(20.0, 55.0, 5.0)
    return xr.Dataset(
        {"ssh": (("lat", "lon"), np.ones((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    )


def test_subset_to_registered_region_trims_rectilinear_grid():
    out = subset_to_region(_rectilinear_ds(), "gulf_stream")

    assert float(out.lon.min()) > -80.0
    assert float(out.lon.max()) <= -50.0
    assert float(out.lat.min()) > 30.0
    assert float(out.lat.max()) <= 45.0


def test_subset_to_region_handles_scattered_points():
    ds = xr.Dataset(
        {"ssh": ("point", np.arange(4.0))},
        coords={
            "lon": ("point", [-75.0, -60.0, -10.0, 140.0]),
            "lat": ("point", [35.0, 40.0, 40.0, 35.0]),
        },
    )

    out = subset_to_region(ds, "gulf_stream")

    np.testing.assert_array_equal(out["ssh"].values, [0.0, 1.0])


def test_custom_region_antimeridian_subsets_both_sides():
    ds = xr.Dataset(
        {"ssh": (("lat", "lon"), np.ones((1, 4)))},
        coords={"lat": [0.0], "lon": [160.0, 175.0, -175.0, -160.0]},
    )
    region = custom_region(
        id="dateline",
        display_name="Dateline",
        lat_min=-10.0,
        lat_max=10.0,
        lon_min=170.0,
        lon_max=-170.0,
    )

    out = subset_to_region(ds, region)

    assert isinstance(region.regions.polygons[0], MultiPolygon)
    np.testing.assert_array_equal(out.lon.values, [175.0, -175.0])


def test_bbox_region_simple_bounds_is_single_polygon():
    region = bbox_region(
        id="box",
        name="Box",
        lat_min=-5.0,
        lat_max=5.0,
        lon_min=-10.0,
        lon_max=10.0,
    )

    assert isinstance(region.polygons[0], Polygon)


def test_subset_to_region_auto_wraps_negative_region_for_0_360_dataset():
    ds = xr.Dataset(
        {"ssh": (("lat", "lon"), np.ones((3, 5)))},
        coords={"lat": [32.0, 36.0, 40.0], "lon": [260.0, 280.0, 290.0, 300.0, 320.0]},
    )

    out = subset_to_region(ds, "gulf_stream")

    np.testing.assert_array_equal(out.lon.values, [290.0, 300.0])


def test_subset_to_region_validates_non_overlap():
    ds = xr.Dataset(
        {"ssh": (("lat", "lon"), np.ones((2, 2)))},
        coords={"lat": [-60.0, -50.0], "lon": [10.0, 20.0]},
    )

    with pytest.raises(ValueError, match="does not overlap"):
        subset_to_region(ds, "gulf_stream")
    out = subset_to_region(ds, "gulf_stream", validate=False)
    assert out.sizes["lat"] == 0
    assert out.sizes["lon"] == 0


def test_polygon_from_geojson_dict_and_path(tmp_path):
    geojson = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [
                    (-80.0, 30.0),
                    (-50.0, 30.0),
                    (-50.0, 45.0),
                    (-80.0, 45.0),
                    (-80.0, 30.0),
                ]
            ],
        },
        "properties": {},
    }
    path = tmp_path / "region.json"
    path.write_text(json.dumps(geojson))

    from_dict = polygon_from_geojson(geojson, name="poly")
    from_path = polygon_from_geojson(path, name="poly")

    assert isinstance(from_dict, regionmask.Regions)
    assert region_to_dict(from_dict) == region_to_dict(from_path)


def test_resolve_region_and_unknown_region_error():
    spec = resolve_region("gulf_stream")

    assert resolve_region(spec) is spec
    with pytest.raises(KeyError, match="atlantis"):
        resolve_region("atlantis")


def test_region_json_round_trip(tmp_path):
    region = custom_region(
        id="box",
        display_name="Box",
        lat_min=30.0,
        lat_max=45.0,
        lon_min=-80.0,
        lon_max=-50.0,
    )
    data = region_to_dict(region)
    path = tmp_path / "region.json"
    path.write_text(json.dumps(data))

    assert region_to_dict(region_from_dict(data)) == data
    assert region_to_dict(load_region_file(path)) == data


def test_subset_to_region_accepts_regionmask_regions():
    region = polygon_from_geojson(
        {
            "type": "Polygon",
            "coordinates": [
                [
                    (-80.0, 30.0),
                    (-50.0, 30.0),
                    (-50.0, 45.0),
                    (-80.0, 45.0),
                    (-80.0, 30.0),
                ]
            ],
        },
        name="poly",
    )

    out = subset_to_region(_rectilinear_ds(), region)

    assert out.sizes["lat"] > 0
    assert out.sizes["lon"] > 0


def test_subset_to_region_operator_config_round_trip():
    named = SubsetToRegion("gulf_stream")
    custom = SubsetToRegion(
        custom_region(
            id="box",
            display_name="Box",
            lat_min=30.0,
            lat_max=45.0,
            lon_min=-80.0,
            lon_max=-50.0,
        ),
        validate=False,
    )

    assert named(_rectilinear_ds()).sizes["lon"] > 0
    assert named.get_config()["region"] == "gulf_stream"
    cfg = custom.get_config()
    assert cfg["validate"] is False
    assert region_from_dict(cfg["region"]).id == "box"


def test_viz_projection_presets_derive_from_region_registry():
    assert "ibi" in PRESETS
    assert PRESETS["gulf_stream"]["extent"] == (-80.0, -50.0, 30.0, 45.0)
    assert PRESETS["global"]["extent"] is None
    assert _resolve_projection("gulf_stream") is not None


def test_subset_to_region_operator_dict_round_trip():
    """``SubsetToRegion(**op.get_config())`` must reconstruct a working
    operator for non-string regions — that's the leaf-operator replay
    contract used by ApplyToEach."""
    custom = SubsetToRegion(
        custom_region(
            id="box",
            display_name="Box",
            lat_min=30.0,
            lat_max=45.0,
            lon_min=-80.0,
            lon_max=-50.0,
        ),
        validate=False,
    )
    rebuilt = SubsetToRegion(**custom.get_config())
    out = rebuilt(_rectilinear_ds())
    assert out.sizes["lon"] > 0


def test_bbox_region_validates_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        bbox_region(
            id="", name="x", lat_min=0.0, lat_max=10.0, lon_min=0.0, lon_max=10.0
        )
    with pytest.raises(ValueError, match="lat_min"):
        bbox_region(
            id="r", name="r", lat_min=10.0, lat_max=10.0, lon_min=0.0, lon_max=10.0
        )


def test_polygon_from_geojson_rejects_top_level_array():
    with pytest.raises(ValueError, match="top-level"):
        polygon_from_geojson("[1, 2, 3]")
