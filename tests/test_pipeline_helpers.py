"""Tests for the small pipeline-helper operators promoted from V1.4
notebook prototypes: ``RenameVariables``, ``RemoveMean``, ``RegridLike``.

Each op gets a behaviour test plus a JSON round-trip on its config.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz import Sequential
from xrtoolz.geo import remove_mean, rename_variables
from xrtoolz.geo.operators import (
    RemoveMean,
    RenameVariables,
    SelectVariables,
    ValidateCoords,
)
from xrtoolz.interpolate import regrid_like
from xrtoolz.interpolate.operators import RegridLike


@pytest.fixture
def latlon_ds() -> xr.Dataset:
    rng = np.random.default_rng(0)
    return xr.Dataset(
        {"adt": (("latitude", "longitude"), rng.standard_normal((10, 12)))},
        coords={
            "latitude": np.linspace(30.0, 45.0, 10),
            "longitude": np.linspace(-70.0, -50.0, 12),
        },
    )


# ---------- rename_variables ----------------------------------------------


def test_rename_variables_renames_data_var(latlon_ds: xr.Dataset) -> None:
    out = rename_variables(latlon_ds, {"adt": "ssh"})
    assert "ssh" in out.data_vars and "adt" not in out.data_vars


def test_rename_variables_ignores_missing_keys(latlon_ds: xr.Dataset) -> None:
    out = rename_variables(latlon_ds, {"foo": "bar"})
    assert "foo" not in out.data_vars and "adt" in out.data_vars


def test_rename_variables_does_not_touch_coords(latlon_ds: xr.Dataset) -> None:
    # Even though "latitude" is in ds.variables, it must not be renamed
    # because it isn't a data_var.
    out = rename_variables(latlon_ds, {"latitude": "lat"})
    assert "latitude" in out.coords and "lat" not in out.coords


def test_rename_variables_op_round_trip() -> None:
    op = RenameVariables({"adt": "ssh"})
    cfg = json.loads(json.dumps(op.get_config()))
    assert cfg == {"mapping": {"adt": "ssh"}}


# ---------- remove_mean ---------------------------------------------------


def test_remove_mean_subtracts_spatial_mean() -> None:
    arr = np.arange(24, dtype=float).reshape(2, 3, 4)
    ds = xr.Dataset(
        {"x": (("time", "lat", "lon"), arr)},
        coords={"time": [0, 1], "lat": [0, 1, 2], "lon": [0, 1, 2, 3]},
    )
    out = remove_mean(ds, ("lat", "lon"))
    assert np.allclose(out["x"].mean(dim=["lat", "lon"]).values, 0.0, atol=1e-12)


def test_remove_mean_op_round_trip() -> None:
    op = RemoveMean(("lat", "lon"))
    cfg = json.loads(json.dumps(op.get_config()))
    assert cfg == {"dims": ["lat", "lon"]}


def test_remove_mean_accepts_single_dim_string() -> None:
    op = RemoveMean("lat")
    assert op.dims == ("lat",)


# ---------- regrid_like ---------------------------------------------------


def test_regrid_like_resamples_onto_target_grid(latlon_ds: xr.Dataset) -> None:
    target = xr.Dataset(
        coords={"lat": np.linspace(30.0, 45.0, 5), "lon": np.linspace(-70.0, -50.0, 6)}
    )
    src = latlon_ds.rename({"latitude": "lat", "longitude": "lon"})
    out = regrid_like(src, target)
    assert out.sizes["lat"] == 5
    assert out.sizes["lon"] == 6
    assert np.array_equal(out["lat"].values, target["lat"].values)


def test_regrid_like_3d_datetime_coords() -> None:
    time = np.array(
        ["2000-01-01", "2000-01-03", "2000-01-05", "2000-01-07"],
        dtype="datetime64[D]",
    )
    lat = np.linspace(-1.0, 1.0, 5)
    lon = np.linspace(10.0, 14.0, 6)
    time_days = (time - time[0]) / np.timedelta64(1, "D")
    vals = time_days[:, None, None] + 2.0 * lat[None, :, None] - lon[None, None, :]
    src = xr.Dataset(
        {"x": (("time", "lat", "lon"), vals)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    target_time = np.array(["2000-01-02", "2000-01-06"], dtype="datetime64[D]")
    target = xr.Dataset(
        coords={
            "time": target_time,
            "lat": np.linspace(-0.5, 0.5, 3),
            "lon": np.linspace(11.0, 13.0, 4),
        }
    )

    out = regrid_like(src, target, dims=("lat", "lon", "time"))
    target_days = (target_time - time[0]) / np.timedelta64(1, "D")
    expected = (
        target_days[:, None, None]
        + 2.0 * target["lat"].values[None, :, None]
        - target["lon"].values[None, None, :]
    )

    np.testing.assert_allclose(out["x"].values, expected, atol=1e-12)
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_regrid_like_op_in_pipeline(latlon_ds: xr.Dataset) -> None:
    target = xr.Dataset(
        coords={"lat": np.linspace(30.0, 45.0, 5), "lon": np.linspace(-70.0, -50.0, 6)}
    )
    pipe = Sequential(
        [
            SelectVariables(["adt"]),
            ValidateCoords(),
            RenameVariables({"adt": "ssh"}),
            RegridLike(target),
            RemoveMean(("lat", "lon")),
        ]
    )
    out = pipe(latlon_ds)
    assert dict(out.sizes) == {"lat": 5, "lon": 6}
    assert "ssh" in out.data_vars
    assert float(out["ssh"].mean(dim=["lat", "lon"])) == pytest.approx(0.0, abs=1e-12)


def test_regrid_like_raises_when_target_has_no_matching_coord(
    latlon_ds: xr.Dataset,
) -> None:
    target = xr.Dataset(coords={"depth": [0, 10, 20]})
    with pytest.raises(ValueError, match="target is missing requested dims"):
        regrid_like(latlon_ds.rename({"latitude": "lat", "longitude": "lon"}), target)


def test_regrid_like_raises_on_partial_target_coords(
    latlon_ds: xr.Dataset,
) -> None:
    # Target has lat but not lon → must error rather than silently
    # regridding only one axis.
    target = xr.Dataset(coords={"lat": np.linspace(30.0, 45.0, 5)})
    src = latlon_ds.rename({"latitude": "lat", "longitude": "lon"})
    with pytest.raises(ValueError, match=r"missing requested dims \['lon'\]"):
        regrid_like(src, target)


def test_regrid_like_raises_when_source_lacks_dim(latlon_ds: xr.Dataset) -> None:
    src = latlon_ds.rename({"latitude": "lat"})  # no "lon"
    target = xr.Dataset(
        coords={"lat": np.linspace(30.0, 45.0, 5), "lon": np.linspace(-70.0, -50.0, 6)}
    )
    with pytest.raises(ValueError, match=r"input is missing requested dims \['lon'\]"):
        regrid_like(src, target)


def test_regrid_like_op_round_trip() -> None:
    target = xr.Dataset(
        coords={"lat": np.linspace(0.0, 1.0, 4), "lon": np.linspace(0.0, 1.0, 5)}
    )
    op = RegridLike(target)
    cfg = json.loads(json.dumps(op.get_config()))
    assert cfg["target_shape"] == {"lat": 4, "lon": 5}
    assert cfg["dims"] == ["lat", "lon"]
    assert cfg["method"] == "linear"
