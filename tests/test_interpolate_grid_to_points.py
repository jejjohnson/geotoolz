"""Tests for grid-to-points interpolation helpers."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xr_toolz.core import Signature
from xr_toolz.interpolate import along_track, sample_at_points
from xr_toolz.interpolate.operators import AlongTrack, SampleAtPoints


def _surface() -> xr.DataArray:
    lat = np.linspace(-2.0, 2.0, 9)
    lon = np.linspace(10.0, 14.0, 11)
    values = lat[:, None] + 2.0 * lon[None, :]
    return xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})


def test_sample_at_points_identity_on_grid_nodes() -> None:
    da = _surface()
    lat2d, lon2d = np.meshgrid(da.lat.values, da.lon.values, indexing="ij")
    points = {"lat": lat2d.ravel(), "lon": lon2d.ravel()}

    for method in ("linear", "nearest"):
        out = sample_at_points(da, points, method=method)
        np.testing.assert_allclose(out.values, da.values.ravel())


def test_sample_at_points_cubic_matches_cubic_polynomial() -> None:
    lat = np.linspace(-1.0, 1.0, 8)
    lon = np.linspace(-2.0, 2.0, 9)
    values = lat[:, None] ** 3 + lon[None, :] ** 3
    da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    points = xr.Dataset(
        {
            "lat": ("points", [-0.75, -0.1, 0.55]),
            "lon": ("points", [-1.5, 0.25, 1.25]),
        }
    )

    out = sample_at_points(da, points, method="cubic")
    expected = points.lat.values**3 + points.lon.values**3
    np.testing.assert_allclose(out.values, expected, atol=1e-4)


def test_sample_at_points_out_of_grid_and_bounds_error() -> None:
    da = _surface()
    points = {"lat": [0.0, 99.0, np.nan], "lon": [11.0, 11.0, 11.0]}

    out = sample_at_points(da, points)
    assert np.isfinite(out.values[0])
    assert np.isnan(out.values[1])
    assert np.isnan(out.values[2])

    with pytest.raises(ValueError, match="out of bounds"):
        sample_at_points(da, {"lat": [99.0], "lon": [11.0]}, bounds_error=True)


def test_sample_at_points_handles_descending_coords() -> None:
    da = _surface()
    descending = da.isel(lat=slice(None, None, -1))
    points = {"lat": [-1.5, 0.0, 1.5], "lon": [10.5, 12.0, 13.5]}

    expected = sample_at_points(da, points)
    actual = sample_at_points(descending, points)
    xr.testing.assert_allclose(actual, expected)


def test_sample_at_points_broadcasts_leading_dims() -> None:
    base = _surface()
    time = np.arange(4)
    da = xr.concat([base + t for t in time], dim=xr.IndexVariable("time", time))
    points = {"lat": [-1.0, 0.0, 1.0], "lon": [10.0, 11.5, 13.0]}

    out = sample_at_points(da, points)
    assert out.dims == ("time", "points")
    for i in range(time.size):
        expected = sample_at_points(da.isel(time=i), points)
        xr.testing.assert_allclose(out.isel(time=i), expected)


def test_along_track_interpolates_datetime_axis() -> None:
    time = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    lat = np.array([0.0, 1.0, 2.0])
    values = np.arange(time.size)[:, None] + lat[None, :]
    da = xr.DataArray(values, dims=("time", "lat"), coords={"time": time, "lat": lat})
    track = xr.Dataset(
        {
            "time": (
                "points",
                np.array(["2020-01-01T12", "2020-01-02T12"], dtype="datetime64[ns]"),
            ),
            "lat": ("points", [1.0, 2.0]),
        }
    )

    out = along_track(da, track, coords=("time", "lat"))
    np.testing.assert_allclose(out.values, [1.5, 3.5])


def test_sample_at_points_preserves_dask_leading_chunks() -> None:
    pytest.importorskip("dask.array")
    base = _surface()
    da = xr.concat([base + t for t in range(4)], dim="time").chunk(
        {"time": 2, "lat": -1, "lon": -1}
    )
    points = {"lat": [-1.0, 0.0], "lon": [10.0, 11.5]}

    out = sample_at_points(da, points)
    assert out.chunks is not None
    expected = sample_at_points(da.compute(), points)
    xr.testing.assert_allclose(out.compute(), expected)


def test_sample_at_points_rejects_chunked_interp_dims() -> None:
    pytest.importorskip("dask.array")
    da = _surface().chunk({"lat": 3, "lon": -1})

    with pytest.raises(ValueError, match="interpolation dimensions"):
        sample_at_points(da, {"lat": [0.0], "lon": [11.0]})


def test_sample_at_points_operator_config_and_signature() -> None:
    da = _surface()
    points = xr.Dataset(
        {
            "lat": ("points", [-1.0, 1.0]),
            "lon": ("points", [10.5, 13.5]),
        },
        coords={"station": ("points", ["a", "b"])},
    )
    op = SampleAtPoints(points, method="linear")

    xr.testing.assert_allclose(op(da), sample_at_points(da, points))
    cfg = op.get_config()
    assert json.loads(json.dumps(cfg)) == cfg
    sig = op.compute_output_signature(
        Signature({"time": 3, "lat": 9, "lon": 11}, dtype="float64")
    )
    assert dict(sig.dims) == {"time": 3, "points": 2}


def test_along_track_operator_matches_function() -> None:
    time = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[ns]")
    lat = np.array([0.0, 1.0])
    da = xr.DataArray(
        np.array([[0.0, 1.0], [1.0, 2.0]]),
        dims=("time", "lat"),
        coords={"time": time, "lat": lat},
    )
    track = xr.Dataset({"time": ("points", time), "lat": ("points", lat)})
    op = AlongTrack(track, coords=("time", "lat"))

    xr.testing.assert_allclose(op(da), along_track(da, track, coords=("time", "lat")))
