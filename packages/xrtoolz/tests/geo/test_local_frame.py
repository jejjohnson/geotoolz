"""Tests for ``utm_crs_for`` / ``LocalFrame`` / ``assign_local_xy`` (#297)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.geo import (
    AssignLocalXY,
    LocalFrame,
    assign_local_xy,
    local_frame,
    lonlat_to_xy,
    utm_crs_for,
)


PERMIAN = (-102.5, 31.5)


# ============== utm_crs_for ===============================================


@pytest.mark.parametrize(
    ("lon", "lat", "expected"),
    [
        (-102.5, 31.5, "EPSG:32613"),  # Permian Basin, zone 13N
        (151.2, -33.9, "EPSG:32756"),  # Sydney, zone 56S
        (5.5, 60.0, "EPSG:32632"),  # Norway exception: zone 32 widened west
        (20.0, 78.0, "EPSG:32633"),  # Svalbard exception: zone 33 (not 34)
        (257.5, 31.5, "EPSG:32613"),  # 0-360 longitude is wrapped
        (0.0, 0.0, "EPSG:32631"),  # equator counts as north
    ],
)
def test_utm_crs_for_known_sites(lon, lat, expected):
    assert utm_crs_for(lon, lat) == expected


def test_utm_crs_for_zone_boundary_belongs_to_eastern_zone():
    # -102.0 is the meridian between zones 13 and 14: it lands in zone 14.
    assert utm_crs_for(-102.0, 31.5) == "EPSG:32614"
    assert utm_crs_for(-102.0 - 1e-6, 31.5) == "EPSG:32613"


def test_utm_crs_for_other_datum():
    assert utm_crs_for(*PERMIAN, datum="NAD83") == "EPSG:26913"


@pytest.mark.parametrize("lat", [84.5, -80.5, 90.0, np.nan])
def test_utm_crs_for_raises_outside_utm_latitudes(lat):
    # UTM spans 80°S–84°N: the southern edge is tighter than the northern.
    with pytest.raises(ValueError, match=r"-80\.0° <= lat <= 84\.0°"):
        utm_crs_for(0.0, lat)


@pytest.mark.parametrize("lat", [-80.0, 84.0])
def test_utm_crs_for_accepts_utm_latitude_edges(lat):
    assert utm_crs_for(0.0, lat) == ("EPSG:32631" if lat > 0 else "EPSG:32731")


# ============== LocalFrame ================================================


def test_local_frame_picks_utm_zone_and_origin_maps_to_zero():
    frame = local_frame(*PERMIAN)
    assert frame.crs == "EPSG:32613"
    x0, y0 = lonlat_to_xy(frame.crs, [PERMIAN[0]], [PERMIAN[1]])
    assert frame.origin_xy == (float(x0[0]), float(y0[0]))
    x, y = frame.to_xy(*PERMIAN)
    assert float(x) == 0.0
    assert float(y) == 0.0


def test_local_frame_explicit_crs():
    frame = local_frame(*PERMIAN, crs="EPSG:3857")
    assert frame.crs == "EPSG:3857"
    assert frame == LocalFrame("EPSG:3857", *PERMIAN)


@pytest.mark.parametrize(
    ("crs", "unit"),
    [
        ("EPSG:2263", "US survey foot"),  # projected, but feet
        ("EPSG:4326", "degree"),  # geographic
    ],
)
def test_local_frame_rejects_non_metre_crs(crs, unit):
    with pytest.raises(ValueError, match=unit):
        LocalFrame(crs, *PERMIAN)
    with pytest.raises(ValueError, match=unit):
        local_frame(*PERMIAN, crs=crs)


def test_local_frame_round_trip_within_one_millimetre():
    frame = local_frame(*PERMIAN)
    rng = np.random.default_rng(0)
    # 1000 points inside zone 13 (-108 ≤ lon < -102), across its latitudes.
    lon = rng.uniform(-108.0, -102.0, 1000)
    lat = rng.uniform(-80.0, 84.0, 1000)
    x, y = frame.to_xy(lon, lat)
    lon_back, lat_back = frame.to_lonlat(x, y)
    # 1 mm ≈ 9e-9° of latitude; longitude degrees are shorter still.
    np.testing.assert_allclose(lat_back, lat, atol=9e-9, rtol=0)
    np.testing.assert_allclose(lon_back, lon, atol=9e-9, rtol=0)
    x_back, y_back = frame.to_xy(lon_back, lat_back)
    np.testing.assert_allclose(x_back, x, atol=1e-3, rtol=0)
    np.testing.assert_allclose(y_back, y, atol=1e-3, rtol=0)


def test_local_frame_to_xy_is_equivariant_under_origin_shift():
    a = local_frame(*PERMIAN)
    b = local_frame(-103.0, 32.0, crs=a.crs)
    rng = np.random.default_rng(1)
    lon = rng.uniform(-105.0, -102.0, 50)
    lat = rng.uniform(30.0, 33.0, 50)
    xa, ya = a.to_xy(lon, lat)
    xb, yb = b.to_xy(lon, lat)
    dx = b.origin_xy[0] - a.origin_xy[0]
    dy = b.origin_xy[1] - a.origin_xy[1]
    np.testing.assert_allclose(xa - xb, dx, atol=1e-6)
    np.testing.assert_allclose(ya - yb, dy, atol=1e-6)


def test_local_frame_dict_round_trip_is_json_serialisable_and_hashable():
    frame = local_frame(*PERMIAN)
    payload = frame.to_dict()
    assert payload == {"crs": "EPSG:32613", "origin_lon": -102.5, "origin_lat": 31.5}
    assert json.loads(json.dumps(payload)) == payload
    rebuilt = LocalFrame.from_dict(json.loads(json.dumps(payload)))
    assert rebuilt == frame
    assert hash(rebuilt) == hash(frame)
    assert rebuilt.origin_xy == frame.origin_xy


# ============== assign_local_xy ===========================================


def _rectilinear() -> xr.Dataset:
    lon = np.linspace(-103.0, -102.0, 5)
    lat = np.linspace(31.0, 32.0, 4)
    return xr.Dataset(
        {"xch4": (("lat", "lon"), np.zeros((4, 5)))},
        coords={"lon": lon, "lat": lat},
    )


def _swath() -> xr.Dataset:
    lon2d, lat2d = np.meshgrid(
        np.linspace(-103.0, -102.0, 5), np.linspace(31.0, 32.0, 3)
    )
    lon2d = lon2d + 0.01 * np.arange(3)[:, None]  # skewed scan geometry
    return xr.Dataset(
        {"xch4": (("scanline", "ground_pixel"), np.zeros((3, 5)))},
        coords={
            "lon": (("scanline", "ground_pixel"), lon2d),
            "lat": (("scanline", "ground_pixel"), lat2d),
        },
    )


def test_assign_local_xy_rectilinear_gives_2d_coords_on_lat_lon():
    frame = local_frame(*PERMIAN)
    out = assign_local_xy(_rectilinear(), frame)
    assert out["x"].dims == ("lat", "lon")
    assert out["y"].dims == ("lat", "lon")
    assert out["x"].attrs["units"] == "m"
    assert out["y"].attrs["units"] == "m"
    # UTM easting is not constant along a meridian (grid convergence).
    assert np.ptp(out["x"].values[:, 0]) > 0.0
    xx, yy = frame.to_xy(*np.meshgrid(out["lon"].values, out["lat"].values))
    np.testing.assert_allclose(out["x"].values, xx)
    np.testing.assert_allclose(out["y"].values, yy)
    assert out.attrs["local_frame"] == frame.to_dict()
    assert json.loads(json.dumps(out.attrs["local_frame"])) == frame.to_dict()


def test_assign_local_xy_swath_gives_2d_coords_on_swath_dims():
    frame = local_frame(*PERMIAN)
    ds = _swath()
    out = assign_local_xy(ds, frame)
    assert out["x"].dims == ("scanline", "ground_pixel")
    assert out["y"].dims == ("scanline", "ground_pixel")
    xx, yy = frame.to_xy(ds["lon"].values, ds["lat"].values)
    np.testing.assert_allclose(out["x"].values, xx)
    np.testing.assert_allclose(out["y"].values, yy)
    assert out.attrs["local_frame"] == frame.to_dict()


def _track() -> xr.Dataset:
    n = 7
    return xr.Dataset(
        {"xch4": ("obs", np.zeros(n))},
        coords={
            "lon": ("obs", np.linspace(-103.0, -102.0, n)),
            "lat": ("obs", np.linspace(31.0, 32.0, n)),
        },
    )


def test_assign_local_xy_along_track_points_give_1d_coords_on_shared_dim():
    frame = local_frame(*PERMIAN)
    ds = _track()
    out = assign_local_xy(ds, frame)
    assert out["x"].dims == ("obs",)
    assert out["y"].dims == ("obs",)
    assert out["x"].attrs["units"] == "m"
    xx, yy = frame.to_xy(ds["lon"].values, ds["lat"].values)
    np.testing.assert_allclose(out["x"].values, xx)
    np.testing.assert_allclose(out["y"].values, yy)
    assert out.attrs["local_frame"] == frame.to_dict()


def test_assign_local_xy_custom_names_and_input_untouched():
    frame = local_frame(*PERMIAN)
    ds = _rectilinear().rename({"lon": "longitude", "lat": "latitude"})
    out = assign_local_xy(ds, frame, lon="longitude", lat="latitude", x="xe", y="yn")
    assert out["xe"].dims == ("latitude", "longitude")
    assert "yn" in out.coords
    assert "local_frame" not in ds.attrs


def test_assign_local_xy_rejects_mismatched_coord_dims():
    frame = local_frame(*PERMIAN)
    ds = _swath().assign_coords(lat=("scanline", np.linspace(31.0, 32.0, 3)))
    with pytest.raises(ValueError, match="both 1-D or both 2-D"):
        assign_local_xy(ds, frame)


# ============== AssignLocalXY operator ====================================


def test_assign_local_xy_operator_matches_function_and_config_round_trips():
    frame = local_frame(*PERMIAN)
    op = AssignLocalXY(frame, x="xe", y="yn")
    xr.testing.assert_identical(
        op(_rectilinear()), assign_local_xy(_rectilinear(), frame, x="xe", y="yn")
    )
    config = op.get_config()
    assert config == {
        "frame": frame.to_dict(),
        "lon": "lon",
        "lat": "lat",
        "x": "xe",
        "y": "yn",
    }
    assert json.loads(json.dumps(config)) == config
    rebuilt = AssignLocalXY(**json.loads(json.dumps(config)))
    assert rebuilt.frame == frame
    assert rebuilt.get_config() == config
    xr.testing.assert_identical(rebuilt(_rectilinear()), op(_rectilinear()))
    assert "AssignLocalXY" in repr(op)


def test_assign_local_xy_operator_maps_over_a_two_leaf_datatree():
    op = AssignLocalXY(local_frame(*PERMIAN).to_dict())
    tree = xr.DataTree.from_dict({"grid": _rectilinear(), "swath": _swath()})

    out = op(tree)

    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"grid", "swath"}
    assert out["grid"].dataset["x"].dims == ("lat", "lon")
    assert out["swath"].dataset["x"].dims == ("scanline", "ground_pixel")
    for leaf in ("grid", "swath"):
        assert out[leaf].dataset.attrs["local_frame"] == op.frame.to_dict()
