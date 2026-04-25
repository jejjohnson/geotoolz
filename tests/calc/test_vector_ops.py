"""Tests for divergence, curl, laplacian across all three geometries."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xr_toolz.calc import EARTH_RADIUS, curl, divergence, laplacian


# ---- helpers -------------------------------------------------------------


def _interior_2d(arr: np.ndarray, n: int = 3) -> np.ndarray:
    return arr[n:-n, n:-n]


def _cartesian_uv(nx: int = 64, ny: int = 48) -> xr.Dataset:
    """Polynomial vector field with known div / curl.

    ``u(x, y) = x² + y³``,  ``v(x, y) = x³ y``.

    ``∇·F = 2x + x³``,  ``∂v/∂x − ∂u/∂y = 3 x² y − 3 y²``.
    """
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    u = grid_x**2 + grid_y**3
    v = grid_x**3 * grid_y
    return xr.Dataset(
        {
            "u": (("x", "y"), u),
            "v": (("x", "y"), v),
        },
        coords={"x": x, "y": y},
    )


def _spherical_uv(nlon: int = 73, nlat: int = 37) -> xr.Dataset:
    lon_deg = np.linspace(-180.0, 180.0, nlon)
    lat_deg = np.linspace(-80.0, 80.0, nlat)
    lam, phi = np.meshgrid(np.deg2rad(lon_deg), np.deg2rad(lat_deg), indexing="ij")
    u = np.sin(2.0 * lam) * np.cos(phi)
    v = np.cos(lam) * np.sin(phi)
    return xr.Dataset(
        {
            "u": (("lon", "lat"), u),
            "v": (("lon", "lat"), v),
        },
        coords={"lon": lon_deg, "lat": lat_deg},
    )


# ---- divergence ---------------------------------------------------------


def test_cartesian_divergence_matches_analytic():
    ds = _cartesian_uv()
    grid_x, _ = np.meshgrid(ds["x"].values, ds["y"].values, indexing="ij")
    expected = 2 * grid_x + grid_x**3  # ∂u/∂x + ∂v/∂y
    div = divergence(ds, ("u", "v"), dims=("x", "y"), geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(
        _interior_2d(div.values), _interior_2d(expected), atol=1e-3
    )


def test_curl_of_gradient_is_zero_cartesian():
    """``∇×∇φ ≡ 0`` for a smooth scalar (analytic gradient field)."""
    ds = _cartesian_uv()
    grid_x, grid_y = np.meshgrid(ds["x"].values, ds["y"].values, indexing="ij")
    grad = xr.Dataset(
        {
            "gx": (("x", "y"), 3 * grid_x**2 + grid_y),
            "gy": (("x", "y"), 3 * grid_y**2 + grid_x),
        },
        coords={"x": ds["x"], "y": ds["y"]},
    )
    z = curl(grad, ("gx", "gy"), dims=("x", "y"), geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(_interior_2d(z.values), 0.0, atol=1e-2)


def test_divergence_components_dims_mismatch_raises():
    ds = _cartesian_uv()
    with pytest.raises(ValueError, match="must have the same length"):
        divergence(ds, ("u",), dims=("x", "y"), geometry="cartesian")


def test_curl_2d_requires_two_components_two_dims():
    ds = _cartesian_uv()
    with pytest.raises(ValueError, match="two components and two dims"):
        curl(ds, ("u", "v", "w"), dims=("x", "y"), geometry="cartesian")  # type: ignore[arg-type]


# ---- spherical curvature corrections ------------------------------------


def test_spherical_divergence_includes_curvature_term():
    """The ``-(v tan φ)/R`` correction must be applied."""
    ds = _spherical_uv()
    div_with = divergence(
        ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", accuracy=3
    )
    # Reproduce the un-corrected ``∂u/∂x + ∂v/∂y`` by dropping v's
    # curvature contribution.
    from xr_toolz.calc import partial

    du_dx = partial(ds["u"], "lon", geometry="spherical", accuracy=3)
    dv_dy = partial(ds["v"], "lat", geometry="spherical", accuracy=3)
    plain_sum = du_dx + dv_dy

    diff = (div_with - plain_sum).values
    phi = np.deg2rad(ds["lat"].values)
    expected_curvature = -(ds["v"].values * np.tan(phi)[None, :]) / EARTH_RADIUS
    np.testing.assert_allclose(diff, expected_curvature, atol=1e-15)


def test_spherical_curl_includes_curvature_term():
    ds = _spherical_uv()
    z = curl(ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", accuracy=3)
    from xr_toolz.calc import partial

    dvdx = partial(ds["v"], "lon", geometry="spherical", accuracy=3)
    dudy = partial(ds["u"], "lat", geometry="spherical", accuracy=3)
    plain = dvdx - dudy
    diff = (z - plain).values
    phi = np.deg2rad(ds["lat"].values)
    expected_curvature = (ds["u"].values * np.tan(phi)[None, :]) / EARTH_RADIUS
    np.testing.assert_allclose(diff, expected_curvature, atol=1e-15)


def test_spherical_divergence_3d_components_raises():
    ds = _spherical_uv().assign(w=lambda d: d["u"] * 0.0)
    with pytest.raises(ValueError, match="2-D"):
        divergence(
            ds,
            ("u", "v", "w"),
            dims=("lon", "lat", "lat"),
            geometry="spherical",
        )


# ---- laplacian ----------------------------------------------------------


def test_cartesian_laplacian_polynomial():
    nx, ny = 80, 80
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    f = grid_x**4 + grid_y**3
    da = xr.DataArray(f, dims=("x", "y"), coords={"x": x, "y": y})
    expected = 12 * grid_x**2 + 6 * grid_y
    lap = laplacian(da, geometry="cartesian", accuracy=3)
    np.testing.assert_allclose(
        _interior_2d(lap.values), _interior_2d(expected), atol=1e-1
    )


def test_spherical_laplacian_of_constant_is_zero():
    nlon, nlat = 73, 37
    lon = np.linspace(-180.0, 180.0, nlon)
    lat = np.linspace(-80.0, 80.0, nlat)
    da = xr.DataArray(
        np.full((nlon, nlat), 3.14, dtype=np.float64),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
    )
    lap = laplacian(da, geometry="spherical", accuracy=3)
    np.testing.assert_allclose(lap.values, 0.0, atol=1e-20)


def test_spherical_laplacian_of_cos_phi_matches_analytic():
    """``Δ cos φ = −cos(2φ) / (R² cos φ)`` (spherical Laplace–Beltrami)."""
    nlon, nlat = 73, 73
    lon = np.linspace(-180.0, 180.0, nlon)
    lat = np.linspace(-70.0, 70.0, nlat)
    _, phi = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat), indexing="ij")
    da = xr.DataArray(
        np.cos(phi),
        dims=("lon", "lat"),
        coords={"lon": lon, "lat": lat},
        name="f",
    )
    lap = laplacian(da, geometry="spherical", accuracy=4)
    expected = -np.cos(2.0 * phi) / (EARTH_RADIUS**2 * np.cos(phi))
    np.testing.assert_allclose(
        _interior_2d(lap.values), _interior_2d(expected), atol=1e-15
    )
