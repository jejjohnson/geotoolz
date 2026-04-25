"""Tests for :mod:`xr_toolz.transforms._src.dct`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.transforms import dct, dst
from xr_toolz.transforms._src.dct import idct, idst


@pytest.fixture
def da_2d() -> xr.DataArray:
    rng = np.random.default_rng(0)
    return xr.DataArray(
        rng.standard_normal((20, 8)),
        dims=("time", "x"),
        coords={"time": np.arange(20), "x": np.linspace(0.0, 1.0, 8)},
        name="signal",
    )


def test_dct_round_trip(da_2d):
    out = dct(da_2d, "x")
    assert out.name == "signal_dct"
    assert out.shape == da_2d.shape
    recon = idct(out, "x")
    np.testing.assert_allclose(recon.values, da_2d.values, atol=1e-10)


def test_dst_round_trip(da_2d):
    out = dst(da_2d, "x")
    assert out.name == "signal_dst"
    recon = idst(out, "x")
    np.testing.assert_allclose(recon.values, da_2d.values, atol=1e-10)


def test_dct_preserves_dim_names(da_2d):
    out = dct(da_2d, "x")
    assert out.dims == da_2d.dims


def test_dct_unknown_dim_raises(da_2d):
    with pytest.raises(ValueError, match="not present"):
        dct(da_2d, "nope")
