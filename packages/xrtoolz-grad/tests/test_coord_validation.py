"""Tests for coordinate validation guards shared by the geometry backends."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import partial


def test_repeated_rectilinear_coord_raises_instead_of_returning_inf():
    """A duplicated coord value used to divide by zero and yield ``inf``."""
    x = np.array([0.0, 1.0, 1.0, 2.5])
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    with pytest.raises(ValueError, match="strictly monotonic"):
        partial(da, "x", geometry="rectilinear")


def test_non_monotonic_rectilinear_coord_raises():
    x = np.array([0.0, 1.0, 0.0])
    da = xr.DataArray(x**2, dims=("x",), coords={"x": x}, name="f")
    with pytest.raises(ValueError, match="strictly monotonic"):
        partial(da, "x", geometry="rectilinear")


def test_descending_non_uniform_coord_is_supported():
    """Strictly decreasing coords are valid — only reversals are rejected.

    Mirrors ``test_partial_polynomial_on_stretched_grid`` with the grid
    reversed, so the same smoothness/tolerance conventions apply.
    """
    s = np.linspace(0.0, 1.0, 80)
    x = (s + 0.1 * np.sin(np.pi * s))[::-1]
    da = xr.DataArray(x**3, dims=("x",), coords={"x": x}, name="f")
    dfdx = partial(da, "x", geometry="rectilinear", accuracy=3)
    np.testing.assert_allclose(dfdx.values[3:-3], (3 * x**2)[3:-3], atol=1e-3)


@pytest.mark.parametrize("geometry", ["cartesian", "rectilinear"])
def test_missing_coordinate_raises_actionable_error(geometry):
    da = xr.DataArray(np.arange(8.0), dims=("x",), name="f")
    with pytest.raises(ValueError, match="no coordinate"):
        partial(da, "x", geometry=geometry)


def _curvilinear_field() -> xr.DataArray:
    """A field whose lon/lat coords are 2-D, as on curvilinear model grids."""
    j = np.arange(6.0)
    i = np.arange(5.0)
    grid_i, grid_j = np.meshgrid(i, j, indexing="xy")
    return xr.DataArray(
        np.zeros((j.size, i.size)),
        dims=("y", "x"),
        coords={
            "lat": (("y", "x"), 10.0 + grid_j),
            "lon": (("y", "x"), 20.0 + grid_i),
        },
        name="f",
    )


def test_two_dimensional_lat_coord_raises_on_spherical():
    with pytest.raises(ValueError, match="1-D coordinates only"):
        partial(_curvilinear_field(), "lon", geometry="spherical")


def _lonlat_field() -> xr.DataArray:
    lon = np.linspace(-60.0, -20.0, 9)
    lat = np.linspace(10.0, 50.0, 9)
    return xr.DataArray(
        np.sin(np.deg2rad(lat))[:, None] * np.ones(lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="f",
    )


def test_lat_derivative_works_with_scalar_lon_coord():
    """A meridional profile keeps lon as a 0-D coord; ∂/∂lat ignores it."""
    da = _lonlat_field()
    profile = da.sel(lon=-40.0)
    assert profile["lon"].ndim == 0
    got = partial(profile, "lat", geometry="spherical")
    expected = partial(da, "lat", geometry="spherical").sel(lon=-40.0)
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


def test_lon_derivative_still_requires_1d_lat_coord():
    """The cos φ factor must broadcast, so ∂/∂lon does need a 1-D lat."""
    da = _lonlat_field().isel(lat=0)
    assert da["lat"].ndim == 0
    with pytest.raises(ValueError, match="1-D coordinates only"):
        partial(da, "lon", geometry="spherical")


def test_two_dimensional_coord_raises_on_cartesian():
    j = np.arange(6.0)
    i = np.arange(5.0)
    grid_i, _ = np.meshgrid(i, j, indexing="xy")
    da = xr.DataArray(
        np.zeros((j.size, i.size)),
        dims=("y", "x"),
        coords={"x": (("y", "x"), grid_i)},
        name="f",
    )
    with pytest.raises(ValueError, match="1-D coordinates only"):
        partial(da, "x", geometry="cartesian")


def test_single_sample_rectilinear_coord_still_raises():
    da = xr.DataArray([1.0], dims=("x",), coords={"x": [0.0]}, name="f")
    with pytest.raises(ValueError, match="need at least 2"):
        partial(da, "x", geometry="rectilinear")


def test_wide_integer_coordinate_is_uniform():
    """A range past int64 with representable steps must still difference.

    ``[-9e18, …, 6e18]`` steps uniformly by ``5e18``; anchoring on the
    first sample would overflow and make the axis look non-uniform.
    """
    coord = np.array([-9e18, -4e18, 1e18, 6e18], dtype=np.int64)
    da = xr.DataArray(
        np.arange(4, dtype=float), dims=("i",), coords={"i": coord}, name="f"
    )
    got = partial(da, "i")
    np.testing.assert_allclose(got.values, 1.0 / 5e18, rtol=1e-9)


def test_large_origin_integer_coordinate_keeps_its_step():
    """``10**18 + k`` falls inside one float64 ulp if cast directly."""
    coord = 10**18 + np.arange(8)
    da = xr.DataArray(
        (np.arange(8, dtype=float) ** 2), dims=("i",), coords={"i": coord}, name="f"
    )
    np.testing.assert_allclose(
        partial(da, "i").values[1:-1], (2.0 * np.arange(8))[1:-1], atol=1e-9
    )
