"""Tests for :mod:`xr_toolz.calc` on uniform Cartesian grids."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.calc import gradient, partial


def _polynomial(nx: int = 64, ny: int = 32) -> xr.DataArray:
    """Return ``f(x, y) = x² + y³`` on a uniform grid."""
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 2.0, ny)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return xr.DataArray(
        grid_x**2 + grid_y**3,
        dims=("x", "y"),
        coords={"x": x, "y": y},
        name="f",
    )


def _interior(arr: np.ndarray, n: int = 2) -> np.ndarray:
    """Drop ``n`` cells from each end of every axis (skip boundary stencil)."""
    return arr[(slice(n, -n),) * arr.ndim]


def test_partial_x_polynomial_interior():
    da = _polynomial()
    grid_x, _ = np.meshgrid(da["x"].values, da["y"].values, indexing="ij")
    dfdx = partial(da, "x", geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(_interior(dfdx.values), _interior(2 * grid_x), atol=1e-9)


def test_partial_y_polynomial_interior():
    da = _polynomial()
    _, grid_y = np.meshgrid(da["x"].values, da["y"].values, indexing="ij")
    dfdy = partial(da, "y", geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(
        _interior(dfdy.values), _interior(3 * grid_y**2), atol=1e-3
    )


def test_partial_method_forward_matches_central_for_quadratic():
    """A 1st-order forward stencil is exact for a linear field."""
    x = np.linspace(0.0, 1.0, 16)
    da = xr.DataArray(2.0 * x + 1.0, dims=("x",), coords={"x": x}, name="f")
    dfdx = partial(da, "x", geometry="cartesian", method="forward")
    np.testing.assert_allclose(dfdx.values, 2.0, atol=1e-10)


def test_gradient_returns_one_dataarray_per_dim():
    da = _polynomial()
    grad = gradient(da, geometry="cartesian", accuracy=3)
    assert isinstance(grad, xr.Dataset)
    assert set(grad.data_vars) == {"df_dx", "df_dy"}


def test_gradient_values_match_analytic_interior():
    da = _polynomial()
    grid_x, grid_y = np.meshgrid(da["x"].values, da["y"].values, indexing="ij")
    grad = gradient(da, geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(
        _interior(grad["df_dx"].values), _interior(2 * grid_x), atol=1e-9
    )
    np.testing.assert_allclose(
        _interior(grad["df_dy"].values), _interior(3 * grid_y**2), atol=1e-3
    )


def test_gradient_subset_of_dims():
    da = _polynomial()
    grad = gradient(da, dims=("x",), geometry="cartesian", accuracy=3)
    assert set(grad.data_vars) == {"df_dx"}


def test_gradient_per_dim_accuracy_tuple():
    da = _polynomial()
    grad = gradient(da, geometry="cartesian", accuracy=(3, 5))
    assert set(grad.data_vars) == {"df_dx", "df_dy"}


def test_gradient_accuracy_tuple_mismatch_raises():
    da = _polynomial()
    with pytest.raises(ValueError, match="accuracy tuple"):
        gradient(da, dims=("x", "y"), geometry="cartesian", accuracy=(3,))


def test_partial_preserves_dims_and_coords():
    da = _polynomial()
    dfdx = partial(da, "x", geometry="cartesian")
    assert dfdx.dims == ("x", "y")
    np.testing.assert_array_equal(dfdx["x"].values, da["x"].values)
    np.testing.assert_array_equal(dfdx["y"].values, da["y"].values)


def test_partial_preserves_attrs():
    da = _polynomial().assign_attrs(units="m^2", long_name="phi")
    dfdx = partial(da, "x", geometry="cartesian")
    assert dfdx.attrs == {"units": "m^2", "long_name": "phi"}


def test_partial_default_name_for_anonymous_dataarray():
    x = np.linspace(0, 1, 16)
    da = xr.DataArray(np.zeros(16), dims=("x",), coords={"x": x})
    dfdx = partial(da, "x", geometry="cartesian")
    assert dfdx.name is None


def test_partial_unknown_dim_raises():
    da = _polynomial()
    with pytest.raises(ValueError, match="not present"):
        partial(da, "z", geometry="cartesian")


def test_partial_non_uniform_coord_raises():
    x = np.array([0.0, 0.1, 0.3, 0.4, 0.45])
    y = np.linspace(0.0, 1.0, 5)
    da = xr.DataArray(
        np.outer(x, y), dims=("x", "y"), coords={"x": x, "y": y}, name="f"
    )
    with pytest.raises(ValueError, match="not uniformly spaced"):
        partial(da, "x", geometry="cartesian")


def test_partial_single_sample_coord_raises():
    da = xr.DataArray([1.0], dims=("x",), coords={"x": [0.0]}, name="f")
    with pytest.raises(ValueError, match="at least 2"):
        partial(da, "x", geometry="cartesian")


def test_partial_zero_spacing_coord_raises():
    """A degenerate coord where every sample is identical must fail
    fast — otherwise we'd silently divide by zero in the FD kernel."""
    x = np.zeros(5)
    da = xr.DataArray(np.arange(5.0), dims=("x",), coords={"x": x}, name="f")
    with pytest.raises(ValueError, match="zero spacing"):
        partial(da, "x", geometry="cartesian")


def test_unknown_geometry_raises():
    da = _polynomial()
    with pytest.raises(ValueError, match="Unknown geometry"):
        partial(da, "x", geometry="not-a-geometry")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="Unknown geometry"):
        gradient(da, geometry="not-a-geometry")  # type: ignore[arg-type]
