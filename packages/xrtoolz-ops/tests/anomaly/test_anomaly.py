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
