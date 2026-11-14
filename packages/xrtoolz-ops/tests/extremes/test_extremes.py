"""Tests for the extreme-value primitives in :mod:`xrops.extremes`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrops.extremes import (
    block_maxima,
    block_minima,
    pot_exceedances,
    pot_threshold,
    pp_counts,
    pp_stats,
)


@pytest.fixture
def ds_grid_daily() -> xr.Dataset:
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    lat = np.linspace(-40.0, 40.0, 9)
    lon = np.linspace(-60.0, 60.0, 13)
    rng = np.random.default_rng(0)
    # smooth-ish signal so PSDs and interpolation make sense
    tt = np.arange(len(time))
    base = np.sin(2 * np.pi * tt / 365.25)[:, None, None]
    data = base + 0.1 * rng.standard_normal((len(time), len(lat), len(lon)))
    return xr.Dataset(
        {"ssh": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )


# ============== extremes ===================================================


def test_block_maxima_shrinks_time(ds_grid_daily):
    out = block_maxima(ds_grid_daily["ssh"], block_size=30)
    assert out.sizes["time"] == ds_grid_daily.sizes["time"] // 30


def test_block_minima_less_than_maxima(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    max_ = block_maxima(da, block_size=30)
    min_ = block_minima(da, block_size=30)
    assert bool((min_ <= max_).all())


def test_pot_threshold_monotone_in_quantile(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    lo = float(pot_threshold(da, quantile=0.5).values)
    hi = float(pot_threshold(da, quantile=0.95).values)
    assert lo < hi


def test_pot_exceedances_masks_below_threshold(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    thr = float(pot_threshold(da, quantile=0.9).values)
    out = pot_exceedances(da, quantile=0.9)
    # remaining non-NaN values are all >= threshold
    finite = out.values[np.isfinite(out.values)]
    assert (finite >= thr).all()


def test_pp_counts_shape(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    counts = pp_counts(da, quantile=0.9, block_size=30)
    assert counts.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    assert (counts.values >= 0).all()


def test_pp_stats_mean_finite(ds_grid_daily):
    da = ds_grid_daily["ssh"].isel(lat=4, lon=6)
    stats = pp_stats(da, quantile=0.9, block_size=30, statistic=np.mean)
    # Some blocks will have no exceedance -> NaN. But at least one should be finite.
    assert np.any(np.isfinite(stats.values))


def test_pp_counts_supports_gridded_input(ds_grid_daily):
    """Regression: the threshold must be broadcast over non-time dims,
    not scalarized with ``.item()`` (which fails on gridded data)."""
    counts = pp_counts(ds_grid_daily["ssh"], quantile=0.9, block_size=30)
    # Output retains lat/lon as outer dims and coarsens time.
    assert set(counts.dims) >= {"lat", "lon"}
    assert counts.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    assert (counts.values >= 0).all()


def test_pp_stats_supports_gridded_input(ds_grid_daily):
    stats = pp_stats(
        ds_grid_daily["ssh"], quantile=0.9, block_size=30, statistic=np.mean
    )
    assert set(stats.dims) >= {"lat", "lon"}
    assert stats.sizes["time"] == ds_grid_daily.sizes["time"] // 30
    # At least one grid cell has exceedances across all blocks.
    assert np.any(np.isfinite(stats.values))
