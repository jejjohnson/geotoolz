"""Tests for :mod:`xr_toolz.calc` on spherical (lon/lat) geometries."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.calc import EARTH_RADIUS, gradient, partial


def _ssh_grid(nlon: int = 73, nlat: int = 37) -> xr.DataArray:
    """Smooth field on a uniform 5° lon/lat grid covering most of the globe.

    ``F(λ, φ) = sin(2λ) cos(φ)`` — analytic gradients are
    ``∂F/∂x = 2 cos(2λ) cos(φ) / (R cos φ) = 2 cos(2λ) / R`` and
    ``∂F/∂y = -sin(2λ) sin(φ) / R``.
    """
    lon_deg = np.linspace(-180.0, 180.0, nlon)
    lat_deg = np.linspace(-80.0, 80.0, nlat)  # avoid poles where cos(φ)→0
    lam, phi = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg), indexing="ij")
    f = np.sin(2.0 * lam) * np.cos(phi)
    return xr.DataArray(
        f,
        dims=("lon", "lat"),
        coords={"lon": lon_deg, "lat": lat_deg},
        name="f",
    )


def _interior_2d(arr: np.ndarray, n: int = 3) -> np.ndarray:
    return arr[n:-n, n:-n]


def test_spherical_partial_lon_returns_metric_x_derivative():
    da = _ssh_grid()
    lam = np.deg2rad(da["lon"].values)
    phi = np.deg2rad(da["lat"].values)
    expected = (
        2.0 * np.cos(2.0 * lam)[:, None] * np.ones_like(phi)[None, :] / EARTH_RADIUS
    )
    dfdx = partial(da, "lon", geometry="spherical", accuracy=4)
    np.testing.assert_allclose(
        _interior_2d(dfdx.values), _interior_2d(expected), atol=5e-11
    )


def test_spherical_partial_lat_returns_metric_y_derivative():
    da = _ssh_grid()
    lam = np.deg2rad(da["lon"].values)
    phi = np.deg2rad(da["lat"].values)
    expected = -np.sin(2.0 * lam)[:, None] * np.sin(phi)[None, :] / EARTH_RADIUS
    dfdy = partial(da, "lat", geometry="spherical", accuracy=4)
    np.testing.assert_allclose(
        _interior_2d(dfdy.values), _interior_2d(expected), atol=5e-11
    )


def test_spherical_gradient_returns_dx_and_dy():
    da = _ssh_grid()
    grad = gradient(da, geometry="spherical", accuracy=3)
    assert isinstance(grad, xr.Dataset)
    assert set(grad.data_vars) == {"df_dx", "df_dy"}


def test_spherical_partial_name_uses_dx_suffix_for_lon():
    da = _ssh_grid()
    out = partial(da, "lon", geometry="spherical")
    assert out.name == "df_dx"


def test_spherical_partial_name_uses_dy_suffix_for_lat():
    da = _ssh_grid()
    out = partial(da, "lat", geometry="spherical")
    assert out.name == "df_dy"


def test_spherical_partial_custom_coord_names():
    da = _ssh_grid().rename({"lon": "longitude", "lat": "latitude"})
    out = partial(
        da, "longitude", geometry="spherical", lon="longitude", lat="latitude"
    )
    assert out.name == "df_dx"


def test_spherical_partial_invalid_dim_raises():
    da = _ssh_grid()
    with pytest.raises(ValueError, match="must be the lon coord"):
        partial(da, "x", geometry="spherical")


def test_spherical_partial_missing_lon_coord_raises():
    da = _ssh_grid().drop_vars("lon")
    with pytest.raises(ValueError, match="'lon' not present"):
        partial(da, "lat", geometry="spherical")


def test_spherical_partial_non_uniform_lat_raises():
    nonuniform = np.array([-60.0, -50.0, -30.0, 0.0, 30.0, 50.0, 60.0])
    lon = np.linspace(-180.0, 180.0, 24)
    lam, phi = np.meshgrid(np.deg2rad(lon), np.deg2rad(nonuniform), indexing="ij")
    da = xr.DataArray(
        np.cos(phi) * np.sin(lam),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": nonuniform},
        name="f",
    )
    with pytest.raises(ValueError, match="not uniformly spaced"):
        partial(da, "lat", geometry="spherical")


def test_spherical_gradient_subset_dims_only_dx():
    da = _ssh_grid()
    grad = gradient(da, dims=("lon",), geometry="spherical", accuracy=3)
    assert set(grad.data_vars) == {"df_dx"}


def test_spherical_gradient_rejects_non_lonlat_dim():
    da = _ssh_grid()
    with pytest.raises(ValueError, match="for geometry='spherical'"):
        gradient(da, dims=("lon", "z"), geometry="spherical")


def test_unknown_geometry_raises():
    da = _ssh_grid()
    with pytest.raises(ValueError, match="Unknown geometry"):
        partial(da, "lon", geometry="bogus")  # type: ignore[arg-type]


def test_spherical_partial_radius_overrides_constant():
    """Halving R should double ∂F/∂x."""
    da = _ssh_grid()
    a = partial(da, "lon", geometry="spherical", accuracy=3)
    b = partial(da, "lon", geometry="spherical", accuracy=3, radius=EARTH_RADIUS / 2)
    np.testing.assert_allclose(b.values, 2.0 * a.values, atol=1e-15)
