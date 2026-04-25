"""Tests for :mod:`xr_toolz.transforms._src.wavelet`.

Skipped wholesale when ``pywt`` is not importable — the wavelet
backend is an optional dependency advertised via the ``wavelets`` extra.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


pytest.importorskip("pywt")

from xr_toolz.transforms import cwt, dwt


@pytest.fixture
def da_1d() -> xr.DataArray:
    n = 256
    t = np.linspace(0.0, 1.0, n)
    sig = np.sin(2 * np.pi * 8.0 * t) + 0.5 * np.sin(2 * np.pi * 32.0 * t)
    return xr.DataArray(sig, dims=("time",), coords={"time": t}, name="sig")


def test_cwt_introduces_scale_axis(da_1d):
    scales = np.arange(1, 32)
    out = cwt(da_1d, dim="time", scales=scales)
    assert out.dims == ("scale", "time")
    assert out.sizes["scale"] == scales.size
    assert "frequency" in out.coords
    assert out.name == "sig_cwt"


def test_cwt_rejects_nonpositive_scales(da_1d):
    with pytest.raises(ValueError, match="strictly positive"):
        cwt(da_1d, dim="time", scales=[1.0, -1.0])


def test_dwt_returns_approx_and_detail(da_1d):
    coeffs = dwt(da_1d, dim="time", wavelet="db4", level=3)
    assert "approx" in coeffs
    assert {"detail_1", "detail_2", "detail_3"}.issubset(coeffs)
    # The transformed axis is renamed so per-level outputs can have
    # different lengths without colliding.
    for label, arr in coeffs.items():
        assert any(d.startswith("time_dwt_") for d in arr.dims)
        assert arr.name == f"sig_dwt_{label}"


def test_dwt_default_level_uses_max(da_1d):
    coeffs = dwt(da_1d, dim="time", wavelet="db4")
    # max_level for n=256, db4 (dec_len=8) = 5.
    assert "detail_5" in coeffs


def test_dwt_invalid_level_raises(da_1d):
    with pytest.raises(ValueError, match="level must be in"):
        dwt(da_1d, dim="time", wavelet="db4", level=99)
