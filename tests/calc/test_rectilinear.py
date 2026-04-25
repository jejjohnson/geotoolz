"""Tests for :mod:`xr_toolz.calc` on rectilinear (non-uniform) grids."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.calc import gradient, partial


def _stretched_grid(nx: int = 80) -> np.ndarray:
    """Monotonic non-uniform 1-D grid on [0, 1]."""
    s = np.linspace(0.0, 1.0, nx)
    return s + 0.1 * np.sin(np.pi * s)  # smoothly stretched, still monotonic


def _interior(arr: np.ndarray, n: int = 3) -> np.ndarray:
    return arr[(slice(n, -n),) * arr.ndim]


def test_partial_polynomial_on_stretched_grid():
    x = _stretched_grid()
    f = x**3
    da = xr.DataArray(f, dims=("x",), coords={"x": x}, name="f")
    dfdx = partial(da, "x", geometry="rectilinear", accuracy=3)
    np.testing.assert_allclose(_interior(dfdx.values), _interior(3 * x**2), atol=1e-3)


def test_partial_uniform_coord_matches_cartesian():
    x = np.linspace(0.0, 1.0, 40)
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    a = partial(da, "x", geometry="cartesian", accuracy=3)
    b = partial(da, "x", geometry="rectilinear", accuracy=3)
    np.testing.assert_allclose(a.values, b.values, atol=1e-12)


def test_partial_2d_rectilinear():
    x = _stretched_grid(60)
    y = _stretched_grid(40)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    f = grid_x**2 + grid_y**3
    da = xr.DataArray(f, dims=("x", "y"), coords={"x": x, "y": y}, name="f")
    dfdx = partial(da, "x", geometry="rectilinear", accuracy=3)
    dfdy = partial(da, "y", geometry="rectilinear", accuracy=3)
    np.testing.assert_allclose(_interior(dfdx.values), _interior(2 * grid_x), atol=1e-3)
    np.testing.assert_allclose(
        _interior(dfdy.values), _interior(3 * grid_y**2), atol=1e-2
    )


def test_gradient_rectilinear_returns_dataset():
    x = _stretched_grid(40)
    y = _stretched_grid(30)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    f = grid_x + grid_y
    da = xr.DataArray(f, dims=("x", "y"), coords={"x": x, "y": y}, name="f")
    grad = gradient(da, geometry="rectilinear", accuracy=3)
    assert set(grad.data_vars) == {"df_dx", "df_dy"}
    np.testing.assert_allclose(_interior(grad["df_dx"].values), 1.0, atol=1e-6)
    np.testing.assert_allclose(_interior(grad["df_dy"].values), 1.0, atol=1e-6)


def test_partial_unknown_dim_raises():
    x = _stretched_grid(20)
    da = xr.DataArray(x, dims=("x",), coords={"x": x})
    with pytest.raises(ValueError, match="not present"):
        partial(da, "z", geometry="rectilinear")


def test_partial_single_sample_raises():
    da = xr.DataArray([1.0], dims=("x",), coords={"x": [0.0]}, name="f")
    with pytest.raises(ValueError, match="at least 2"):
        partial(da, "x", geometry="rectilinear")
