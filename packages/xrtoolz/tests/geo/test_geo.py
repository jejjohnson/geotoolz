"""Tests for Layer-0 primitives in :mod:`xrtoolz.geo`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrtoolz.geo import (
    add_climatology,
    bias,
    calculate_anomaly,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    check_dataset_coords,
    correlation,
    decode_cf_time,
    lat_90_to_180,
    lat_180_to_90,
    lon_180_to_360,
    lon_360_to_180,
    mae,
    mse,
    nrmse,
    remove_climatology,
    rename_coords,
    rmse,
    select_variables,
    subset_bbox,
    subset_time,
    subset_where,
    validate_latitude,
    validate_longitude,
    validate_time,
)


# ---------- fixtures ---------------------------------------------------------


@pytest.fixture
def ds_global() -> xr.Dataset:
    """A tiny 3-year daily global dataset."""
    time = pd.date_range("2000-01-01", "2002-12-31", freq="1D")
    lat = np.linspace(-60.0, 60.0, 7)
    lon = np.linspace(-170.0, 170.0, 9)
    rng = np.random.default_rng(42)
    data = rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.Dataset(
        {"ssh": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ---------- encoders ---------------------------------------------------------


def test_lon_360_to_180_wraps_eastern_values():
    np.testing.assert_allclose(
        lon_360_to_180(np.array([0.0, 90.0, 180.0, 200.0, 359.0])),
        np.array([0.0, 90.0, -180.0, -160.0, -1.0]),
    )


def test_lon_180_to_360_wraps_negative_values():
    np.testing.assert_allclose(
        lon_180_to_360(np.array([-180.0, -1.0, 0.0, 90.0, 179.0])),
        np.array([180.0, 359.0, 0.0, 90.0, 179.0]),
    )


def test_lon_round_trip_is_identity():
    lons = np.array([-170.0, -30.0, 0.0, 50.0, 179.0])
    np.testing.assert_allclose(lon_360_to_180(lon_180_to_360(lons)), lons)


def test_lat_180_to_90_wraps():
    np.testing.assert_allclose(
        lat_180_to_90(np.array([0.0, 45.0, 90.0, 100.0, 179.0])),
        np.array([0.0, 45.0, -90.0, -80.0, -1.0]),
    )


def test_lat_round_trip_is_identity():
    lats = np.array([-89.0, -30.0, 0.0, 30.0, 89.0])
    np.testing.assert_allclose(lat_180_to_90(lat_90_to_180(lats)), lats)


# ---------- validation -------------------------------------------------------


def test_validate_longitude_renames_and_wraps():
    ds = xr.Dataset(
        {"x": (("longitude",), np.array([0.0]))},
        coords={"longitude": np.array([200.0])},
    )
    out = validate_longitude(ds)
    assert "lon" in out.coords
    assert "longitude" not in out.coords
    np.testing.assert_allclose(out["lon"].values, np.array([-160.0]))
    assert out["lon"].attrs["units"] == "degrees_east"
    assert out["lon"].attrs["standard_name"] == "longitude"


def test_validate_longitude_preserves_existing_attrs():
    ds = xr.Dataset(
        coords={"lon": xr.DataArray([10.0], attrs={"provenance": "ERA5"})},
    )
    out = validate_longitude(ds)
    assert out["lon"].attrs["provenance"] == "ERA5"
    assert out["lon"].attrs["units"] == "degrees_east"


def test_validate_latitude_renames_and_wraps():
    ds = xr.Dataset(coords={"latitude": np.array([100.0])})
    out = validate_latitude(ds)
    assert "lat" in out.coords and "latitude" not in out.coords
    np.testing.assert_allclose(out["lat"].values, np.array([-80.0]))
    assert out["lat"].attrs["units"] == "degrees_north"


def test_validate_longitude_raises_when_missing():
    ds = xr.Dataset({"x": ("foo", [1.0])}, coords={"foo": [0.0]})
    with pytest.raises(KeyError, match="longitude"):
        validate_longitude(ds)


def test_rename_coords_ignores_missing_keys():
    ds = xr.Dataset(coords={"lon": [1.0]})
    out = rename_coords(ds, {"longitude": "lon", "time": "t"})
    assert list(out.coords) == ["lon"]


# ---------- decode_cf_time / validate_time / check_dataset_coords ------------


def test_decode_cf_time_with_units():
    ds = xr.Dataset({"ssh": ("time", [1.0, 2.0])}, coords={"time": [0, 1]})
    out = decode_cf_time(ds, units="days since 2000-01-01")
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_decode_cf_time_noop_when_already_datetime():
    time = pd.date_range("2000-01-01", periods=3)
    ds = xr.Dataset({"ssh": ("time", [1.0, 2.0, 3.0])}, coords={"time": time})
    out = decode_cf_time(ds)
    xr.testing.assert_identical(ds, out)


def test_decode_cf_time_no_units_no_op_for_integers():
    ds = xr.Dataset({"ssh": ("time", [1.0])}, coords={"time": [42]})
    out = decode_cf_time(ds)
    assert out["time"].values[0] == 42


def test_validate_time_coerces_string():
    ds = xr.Dataset({"ssh": ("time", [1.0])}, coords={"time": ["2000-01-01"]})
    out = validate_time(ds)
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_validate_time_noop_already_datetime():
    time = pd.date_range("2000-01-01", periods=3)
    ds = xr.Dataset({"ssh": ("time", [1.0, 2.0, 3.0])}, coords={"time": time})
    out = validate_time(ds)
    assert np.issubdtype(out["time"].dtype, np.datetime64)


def test_validate_time_does_not_mutate_input():
    time = pd.date_range("2000-01-01", periods=2)
    ds = xr.Dataset({"ssh": ("time", [1.0, 2.0])}, coords={"time": time})
    _ = validate_time(ds)
    xr.testing.assert_identical(
        ds["time"], xr.DataArray(time, dims="time", name="time")
    )


def test_check_dataset_coords_passes_valid(ds_global):
    from xrtoolz.geo import validate_latitude, validate_longitude

    ds = validate_longitude(validate_latitude(ds_global))
    check_dataset_coords(ds, require=("time", "lat", "lon"))


def test_check_dataset_coords_raises_on_missing():
    ds = xr.Dataset(coords={"lat": [0.0], "lon": [0.0]})
    with pytest.raises(AssertionError, match="time"):
        check_dataset_coords(ds)


def test_check_dataset_coords_custom_require(ds_global):
    check_dataset_coords(ds_global, require=("time",))


def test_check_dataset_coords_validate_false_skips_roundtrip(ds_global):
    check_dataset_coords(ds_global, require=("time", "lat", "lon"), validate=False)


def test_check_dataset_coords_sorted_missing_in_message():
    ds = xr.Dataset()
    with pytest.raises(AssertionError, match="lat") as exc_info:
        check_dataset_coords(ds, require=("time", "lat", "lon"))
    assert "lon" in str(exc_info.value)
    assert "time" in str(exc_info.value)


def test_check_dataset_coords_rejects_data_var_with_required_name():
    """A data variable named ``time`` does not satisfy the schema —
    ``.sel(time=...)`` and other coord-based ops still need the name to
    be a coordinate."""
    ds = xr.Dataset({"time": ("row", [0, 1, 2])})
    with pytest.raises(AssertionError, match="time"):
        check_dataset_coords(ds, require=("time",))


def test_validate_time_unit_seconds_since_epoch():
    """Numeric times need an explicit ``unit`` — without it,
    pandas would interpret 1.0 as one nanosecond past the epoch."""
    ds = xr.Dataset({"ssh": ("time", [1.0])}, coords={"time": [1.0]})
    out = validate_time(ds, unit="s")
    assert out["time"].values[0] == np.datetime64("1970-01-01T00:00:01")


def test_validate_time_preserves_attrs():
    time = pd.date_range("2000-01-01", periods=2)
    ds = xr.Dataset({"ssh": ("time", [1.0, 2.0])}, coords={"time": time})
    ds["time"].attrs["standard_name"] = "time"
    ds["time"].attrs["axis"] = "T"
    out = validate_time(ds)
    assert out["time"].attrs == {"standard_name": "time", "axis": "T"}


# ---------- subset -----------------------------------------------------------


def test_subset_bbox_restricts_to_box(ds_global):
    out = subset_bbox(ds_global, lon_bnds=(-30.0, 30.0), lat_bnds=(-20.0, 20.0))
    assert float(out.lon.min()) >= -30.0
    assert float(out.lon.max()) <= 30.0
    assert float(out.lat.min()) >= -20.0
    assert float(out.lat.max()) <= 20.0


def test_subset_time_slices_range(ds_global):
    out = subset_time(ds_global, "2001-06-01", "2001-08-31")
    assert pd.Timestamp(out.time.min().values) >= pd.Timestamp("2001-06-01")
    assert pd.Timestamp(out.time.max().values) <= pd.Timestamp("2001-08-31")


def test_subset_where_drops_outside_range():
    ds = xr.Dataset({"x": ("i", np.array([-5.0, 0.0, 5.0, 10.0]))})
    out = subset_where(ds, "x", min_val=-1.0, max_val=6.0)
    np.testing.assert_array_equal(out["x"].values, np.array([0.0, 5.0]))


def test_subset_where_can_preserve_shape_with_nan():
    ds = xr.Dataset({"x": ("i", np.array([-5.0, 0.0, 5.0, 10.0]))})
    out = subset_where(ds, "x", min_val=-1.0, max_val=6.0, drop=False)
    assert np.isnan(out["x"].values[0]) and np.isnan(out["x"].values[3])
    np.testing.assert_array_equal(out["x"].values[1:3], np.array([0.0, 5.0]))


def test_select_variables_accepts_single_string(ds_global):
    out = select_variables(ds_global, "ssh")
    assert list(out.data_vars) == ["ssh"]


def test_select_variables_accepts_sequence(ds_global):
    ds_global["sst"] = ds_global["ssh"] * 0.5
    out = select_variables(ds_global, ["ssh", "sst"])
    assert sorted(out.data_vars) == ["ssh", "sst"]


# ---------- detrend ----------------------------------------------------------


def test_climatology_dayofyear_shape(ds_global):
    clim = calculate_climatology(ds_global, freq="day")
    assert "dayofyear" in clim.dims
    assert "time" not in clim.dims
    assert clim.sizes["dayofyear"] <= 366


def test_climatology_month_shape(ds_global):
    clim = calculate_climatology(ds_global, freq="month")
    assert "month" in clim.dims
    assert clim.sizes["month"] == 12


def test_climatology_rejects_unknown_freq(ds_global):
    with pytest.raises(ValueError, match="freq must be one of"):
        calculate_climatology(ds_global, freq="decade")


def test_climatology_smoothed_reduces_variance(ds_global):
    clim = calculate_climatology(ds_global, freq="day")["ssh"]
    smoothed = calculate_climatology_smoothed(ds_global, window=60)["ssh"]
    # Reduce over spatial dims to get scalar variances along dayofyear.
    assert float(smoothed.var("dayofyear").mean()) < float(clim.var("dayofyear").mean())


def test_climatology_smoothed_rejects_odd_window(ds_global):
    with pytest.raises(ValueError, match="window"):
        calculate_climatology_smoothed(ds_global, window=61)


def test_remove_then_add_climatology_round_trips(ds_global):
    clim = calculate_climatology(ds_global, freq="day")
    anom = remove_climatology(ds_global, clim)
    recovered = add_climatology(anom, clim)
    # groupby arithmetic attaches a `dayofyear` helper coord; compare values.
    np.testing.assert_allclose(recovered["ssh"].values, ds_global["ssh"].values)


def test_calculate_anomaly_has_zero_climatology(ds_global):
    anom = calculate_anomaly(ds_global, freq="month")
    anom_clim = calculate_climatology(anom, freq="month")["ssh"]
    np.testing.assert_allclose(anom_clim.values, 0.0, atol=1e-10)


def test_calculate_climatology_season_returns_four_seasons(ds_global):
    clim = calculate_climatology_season(ds_global)
    assert "season" in clim.dims
    assert set(clim.season.values) == {"DJF", "MAM", "JJA", "SON"}


def test_remove_climatology_rejects_malformed_state(ds_global):
    # climatology without a recognized grouping dim
    bogus = xr.Dataset({"ssh": ("bogus", [1.0, 2.0])}, coords={"bogus": [0, 1]})
    with pytest.raises(ValueError, match="exactly one"):
        remove_climatology(ds_global, bogus)


# ---------- metrics ----------------------------------------------------------


def _pair(values_pred, values_ref):
    ds_pred = xr.Dataset({"x": ("i", np.asarray(values_pred, dtype=float))})
    ds_ref = xr.Dataset({"x": ("i", np.asarray(values_ref, dtype=float))})
    return ds_pred, ds_ref


def test_mse_and_rmse_known_values():
    p, r = _pair([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert float(mse(p["x"], r["x"], dim="i")) == pytest.approx(14.0 / 3.0)
    assert float(rmse(p["x"], r["x"], dim="i")) == pytest.approx((14.0 / 3.0) ** 0.5)


def test_mae_known_value():
    p, r = _pair([1.0, -2.0, 3.0], [0.0, 0.0, 0.0])
    assert float(mae(p["x"], r["x"], dim="i")) == pytest.approx(2.0)


def test_bias_signed():
    p, r = _pair([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
    assert float(bias(p["x"], r["x"], dim="i")) == pytest.approx(2.0)
    assert float(bias(r["x"], p["x"], dim="i")) == pytest.approx(-2.0)


def test_correlation_perfect_for_identical_series():
    p, r = _pair([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert float(correlation(p["x"], r["x"], dim="i")) == pytest.approx(1.0)


def test_correlation_negative_one_for_inverted_series():
    p, r = _pair([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0])
    assert float(correlation(p["x"], r["x"], dim="i")) == pytest.approx(-1.0)


def test_nrmse_is_one_for_perfect_prediction():
    p, r = _pair([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0])
    assert float(nrmse(p["x"], r["x"], dim="i")) == pytest.approx(1.0)


def test_metrics_preserve_non_reduced_dims():
    pred = xr.Dataset(
        {"x": (("t", "i"), np.ones((3, 4)))},
        coords={"t": [10, 20, 30], "i": [0, 1, 2, 3]},
    )
    ref = xr.Dataset(
        {"x": (("t", "i"), np.zeros((3, 4)))},
        coords={"t": [10, 20, 30], "i": [0, 1, 2, 3]},
    )
    out = rmse(pred["x"], ref["x"], dim="i")
    assert out.dims == ("t",)
    np.testing.assert_allclose(out.values, np.array([1.0, 1.0, 1.0]))
