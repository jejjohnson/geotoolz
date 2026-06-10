"""Tests for the consolidated ``coord_spacing`` helper.

Pins the three policy combinations the Fourier and wavelet modules rely
on after their per-module ``_coord_spacing`` copies were unified.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrtoolz.utils._src.spacing import coord_spacing


def _da(coord: list[float] | None) -> xr.DataArray:
    n = 4 if coord is None else len(coord)
    if coord is None:
        return xr.DataArray(np.arange(n, dtype=float), dims="x")
    return xr.DataArray(np.arange(n, dtype=float), dims="x", coords={"x": coord})


def test_uniform_spacing():
    assert coord_spacing(_da([0.0, 2.0, 4.0, 6.0]), "x") == 2.0


def test_datetime_spacing_in_seconds():
    times = np.array(["2020-01-01", "2020-01-02", "2020-01-03"], dtype="datetime64[ns]")
    da = xr.DataArray([0.0, 1.0, 2.0], dims="t", coords={"t": times})
    assert coord_spacing(da, "t") == 86400.0  # one day in seconds


# ---- missing-coordinate policy (require_coord) -------------------------------


def test_missing_coord_raises_by_default():
    with pytest.raises(ValueError, match="no coordinate"):
        coord_spacing(_da(None), "x")


def test_missing_coord_falls_back_when_allowed():
    assert coord_spacing(_da(None), "x", require_coord=False, fallback=1.0) == 1.0


# ---- singleton policy (require_min_samples) ----------------------------------


def test_singleton_raises_by_default():
    da = xr.DataArray([1.0], dims="x", coords={"x": [0.0]})
    with pytest.raises(ValueError, match="at least two samples"):
        coord_spacing(da, "x")


def test_singleton_falls_back_when_allowed():
    da = xr.DataArray([1.0], dims="x", coords={"x": [0.0]})
    assert coord_spacing(da, "x", require_min_samples=False, fallback=1.0) == 1.0


# ---- uniformity policy (require_uniform) -------------------------------------


def test_non_uniform_raises_when_required():
    with pytest.raises(ValueError, match="uniformly spaced"):
        coord_spacing(_da([0.0, 1.0, 5.0]), "x")


def test_non_uniform_returns_median_when_tolerant():
    # |Δ| = [1, 4] -> median 2.5
    assert coord_spacing(_da([0.0, 1.0, 5.0]), "x", require_uniform=False) == 2.5
