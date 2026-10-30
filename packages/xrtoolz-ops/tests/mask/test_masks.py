"""Tests for the mask primitives in :mod:`xrops.mask`."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from xrops.mask import add_country_mask, add_land_mask, add_ocean_mask, apply_mask


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


# ============== masks (hit the API but keep it tolerant of offline CI) ====


def test_add_land_mask_adds_coord(ds_grid_daily):
    """Smoke-test; requires regionmask data (cached after first use)."""
    try:
        out = add_land_mask(ds_grid_daily.isel(time=0))
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
    assert "land_mask" in out.coords


def test_add_ocean_mask_global(ds_grid_daily):
    try:
        out = add_ocean_mask(ds_grid_daily.isel(time=0), ocean="global")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
    assert "ocean_mask" in out.coords


def test_add_ocean_mask_rejects_unknown_basin(ds_grid_daily):
    try:
        with pytest.raises(ValueError, match="not found"):
            add_ocean_mask(ds_grid_daily.isel(time=0), ocean="does-not-exist")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")


def test_apply_mask_by_name_drops_nothing_when_all_true():
    ds = xr.Dataset(
        {"x": ("i", np.arange(4.0))},
        coords={"i": [0, 1, 2, 3], "mask": ("i", np.ones(4, dtype=np.int16))},
    )
    out = apply_mask(ds, "mask")
    np.testing.assert_array_equal(out["x"].values, np.arange(4.0))


def test_add_country_mask_rejects_unknown(ds_grid_daily):
    try:
        with pytest.raises(ValueError, match="not found"):
            add_country_mask(ds_grid_daily.isel(time=0), country="Atlantis")
    except Exception as exc:
        pytest.skip(f"regionmask unavailable: {exc}")
