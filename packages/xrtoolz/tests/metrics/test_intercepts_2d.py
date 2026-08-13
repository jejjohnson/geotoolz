"""Tests for :func:`xrtoolz.metrics.find_intercept_2D` (2-D resolved-scale boundary)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.metrics import find_intercept_2D


# find_intercept_2D extracts the contour via scikit-image ([image] extra).
pytest.importorskip("skimage", reason="requires the [image] extra")


def test_find_intercept_2D_recovers_analytic_boundary():
    n = 64
    fl = np.linspace(0.0, 1.0, n)
    ft = np.linspace(0.0, 1.0, n)
    FL, FT = np.meshgrid(fl, ft)
    # Linear monotone score: 1 - (fl + ft); score=0.5 along fl+ft=0.5
    score = 1.0 - (FL + FT)
    da = xr.DataArray(
        score,
        coords={"freq_time": ft, "freq_lon": fl},
        dims=("freq_time", "freq_lon"),
    )
    segments = find_intercept_2D(da, level=0.5)
    assert len(segments) >= 1
    seg = segments[0]
    assert "axis" in seg.dims
    sx = seg.sel(axis="freq_lon").values
    ty = seg.sel(axis="freq_time").values
    # Analytic boundary: sx + ty = 0.5
    np.testing.assert_allclose(sx + ty, 0.5, atol=2.0 / n)


def test_find_intercept_2D_rejects_non_2d():
    da = xr.DataArray([1.0, 2.0, 3.0], dims=("freq_lon",))
    with pytest.raises(ValueError, match="2-D"):
        find_intercept_2D(da)


def test_find_intercept_2D_missing_dim_raises():
    da = xr.DataArray(np.zeros((4, 4)), dims=("a", "b"))
    with pytest.raises(ValueError, match="must have dims"):
        find_intercept_2D(da)
