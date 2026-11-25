"""Tests for the climatology/anomaly primitives in :mod:`xrops.anomaly`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrops.anomaly import (
    add_climatology,
    calculate_anomaly,
    calculate_climatology,
    calculate_climatology_season,
    calculate_climatology_smoothed,
    remove_climatology,
)
from xrops.interpolate import fillnan_climatology


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
    smoothed = calculate_climatology_smoothed(ds_global, window=61)["ssh"]
    # Reduce over spatial dims to get scalar variances along dayofyear.
    assert float(smoothed.var("dayofyear").mean()) < float(clim.var("dayofyear").mean())


@pytest.mark.parametrize("window", [60, 0, 367])
def test_climatology_smoothed_rejects_a_bad_window(ds_global, window):
    with pytest.raises(ValueError, match="window"):
        calculate_climatology_smoothed(ds_global, window=window)


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


# ---------- calendar dates (#774) ------------------------------------------------


def _first_of_march(start: str = "2001-01-01", end: str = "2008-12-31") -> xr.DataArray:
    time = pd.date_range(start, end, freq="1D")
    is_date = (time.month == 3) & (time.day == 1)
    return xr.DataArray(
        is_date.astype(float), dims="time", coords={"time": time}, name="first"
    )


def test_each_date_has_one_slot():
    clim = calculate_climatology(_first_of_march(), freq="day")
    np.testing.assert_array_equal(clim["dayofyear"], np.arange(1, 367))
    assert float(clim.sel(dayofyear=61)) == 1.0
    assert float(clim.drop_sel(dayofyear=61).max()) == 0.0


def test_the_anomaly_of_a_date_is_zero():
    np.testing.assert_array_equal(calculate_anomaly(_first_of_march(), freq="day"), 0.0)


def test_only_the_days_with_data_are_slots():
    clim = calculate_climatology(_first_of_march("2001-02-27", "2001-03-02"))
    np.testing.assert_array_equal(clim["dayofyear"], [58, 59, 61, 62])


def test_a_calendar_without_leap_days_keeps_its_day_of_the_year():
    pytest.importorskip("cftime")
    time = xr.date_range("2001-01-01", periods=730, calendar="noleap", use_cftime=True)
    x = xr.DataArray(np.arange(730.0), dims="time", coords={"time": time})
    clim = calculate_climatology(x, freq="day")
    np.testing.assert_array_equal(clim["dayofyear"], np.arange(1, 366))
    assert calculate_climatology_smoothed(x, window=3).sizes["dayofyear"] == 365


def test_gregorian_cftime_dates_are_the_same_slots():
    pytest.importorskip("cftime")
    x = _first_of_march()
    time = xr.date_range(
        "2001-01-01", "2008-12-31", calendar="standard", use_cftime=True
    )
    clim = calculate_climatology(x.assign_coords(time=time), freq="day")
    assert float(clim.sel(dayofyear=61)) == 1.0


# ---------- smoothing ------------------------------------------------------------


def test_smoothing_is_centred_and_wraps():
    time = pd.date_range("2001-01-01", "2001-12-31", freq="1D")
    x = xr.DataArray(
        (time.dayofyear == 1).astype(float), dims="time", coords={"time": time}
    )
    smooth = calculate_climatology_smoothed(x, window=5)
    # days 365 and 366 are 30 and 31 December; a year without a leap day has no 60
    hot = smooth["dayofyear"].values[smooth.values > 0]
    np.testing.assert_array_equal(hot, [1, 2, 3, 365, 366])
    np.testing.assert_allclose(smooth.sel(dayofyear=hot), 0.2)


def test_a_day_counts_as_often_as_it_was_sampled():
    time = pd.date_range("2001-01-01", "2004-12-31", freq="1D")
    leap = (time.month == 2) & (time.day == 29)
    x = xr.DataArray(leap.astype(float), dims="time", coords={"time": time})
    smooth = calculate_climatology_smoothed(x, window=3)
    # one 29 February among four 28 February and four 1 March
    assert float(smooth.sel(dayofyear=60)) == pytest.approx(1.0 / 9.0)


def test_smoothing_fills_an_empty_day_and_leaves_an_empty_window():
    time = pd.date_range("2001-01-01", "2001-03-31", freq="1D")
    x = xr.DataArray(np.ones(time.size), dims="time", coords={"time": time})
    smooth = calculate_climatology_smoothed(x, window=3)
    assert float(smooth.sel(dayofyear=60)) == 1.0
    assert np.isnan(float(smooth.sel(dayofyear=200)))


def test_smoothing_keeps_dims_names_attrs_and_laziness(ds_global):
    ds_global["ssh"].attrs["units"] = "m"
    expected = calculate_climatology_smoothed(ds_global, window=5)
    assert expected["ssh"].attrs == {"units": "m"}
    assert expected["ssh"].dims == calculate_climatology(ds_global)["ssh"].dims
    pytest.importorskip("dask.array")
    lazy = calculate_climatology_smoothed(ds_global.chunk({"time": 100}), window=5)
    assert lazy["ssh"].chunks is not None
    xr.testing.assert_allclose(lazy.compute(), expected)


def test_a_gap_is_filled_with_the_mean_of_its_date():
    first = _first_of_march()
    gappy = first.where(first["time"] != np.datetime64("2004-03-01"))
    filled = fillnan_climatology(gappy, group="dayofyear", residual="zero")
    xr.testing.assert_equal(filled, first)
