"""Tests for per-dim ``accuracy`` tuples in divergence / curl / laplacian."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import curl, divergence, gradient, laplacian, partial


def _vector_field(n: int = 24) -> xr.Dataset:
    """Return a smooth non-polynomial 2-D vector field on a uniform grid."""
    x = np.linspace(0.0, 2.0, n)
    y = np.linspace(0.0, 3.0, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    coords = {"x": x, "y": y}
    return xr.Dataset(
        {
            "u": xr.DataArray(np.sin(grid_x) * grid_y, dims=("x", "y"), coords=coords),
            "v": xr.DataArray(grid_x * np.cos(grid_y), dims=("x", "y"), coords=coords),
        }
    )


def _scalar_field(n: int = 24) -> xr.DataArray:
    x = np.linspace(0.0, 2.0, n)
    y = np.linspace(0.0, 3.0, n)
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    return xr.DataArray(
        np.sin(grid_x) * np.cos(grid_y),
        dims=("x", "y"),
        coords={"x": x, "y": y},
        name="f",
    )


def test_divergence_per_dim_accuracy_matches_explicit_partials():
    ds = _vector_field()
    got = divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=(1, 3))
    expected = partial(ds["u"], "x", accuracy=1) + partial(ds["v"], "y", accuracy=3)
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


def test_divergence_per_dim_accuracy_differs_from_scalar():
    """Guard against the tuple being silently collapsed to its first entry."""
    ds = _vector_field()
    mixed = divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=(1, 3))
    uniform = divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=1)
    assert not np.allclose(mixed.values, uniform.values)


def test_curl_per_dim_accuracy_matches_explicit_partials():
    ds = _vector_field()
    got = curl(ds, ("u", "v"), dims=("x", "y"), accuracy=(1, 3))
    expected = partial(ds["v"], "x", accuracy=1) - partial(ds["u"], "y", accuracy=3)
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


def test_laplacian_cartesian_per_dim_accuracy_matches_explicit_partials():
    f = _scalar_field()
    got = laplacian(f, dims=("x", "y"), accuracy=(1, 3))
    expected = partial(f, "x", order=2, accuracy=1) + partial(
        f, "y", order=2, accuracy=3
    )
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


def test_laplacian_spherical_per_dim_accuracy_matches_composition():
    """The non-cartesian path forwards the tuple through gradient+divergence."""
    lon = np.linspace(0.0, 20.0, 16)
    lat = np.linspace(10.0, 30.0, 16)
    grid_lon, grid_lat = np.meshgrid(lon, lat, indexing="xy")
    f = xr.DataArray(
        np.sin(np.deg2rad(grid_lon)) * np.cos(np.deg2rad(grid_lat)),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
        name="f",
    )
    got = laplacian(f, geometry="spherical", accuracy=(1, 3))
    grad = gradient(f, dims=("lon", "lat"), geometry="spherical", accuracy=(1, 3))
    expected = divergence(
        grad,
        tuple(grad.data_vars),
        dims=("lon", "lat"),
        geometry="spherical",
        accuracy=(1, 3),
    )
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3)])
def test_divergence_accuracy_tuple_mismatch_raises(bad):
    ds = _vector_field()
    with pytest.raises(ValueError, match="accuracy tuple length"):
        divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=bad)


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3)])
def test_curl_accuracy_tuple_mismatch_raises(bad):
    ds = _vector_field()
    with pytest.raises(ValueError, match="accuracy tuple length"):
        curl(ds, ("u", "v"), dims=("x", "y"), accuracy=bad)


@pytest.mark.parametrize("bad", [(1,), (1, 2, 3)])
def test_laplacian_accuracy_tuple_mismatch_raises(bad):
    with pytest.raises(ValueError, match="accuracy tuple length"):
        laplacian(_scalar_field(), dims=("x", "y"), accuracy=bad)


def test_scalar_accuracy_equals_uniform_tuple():
    """Scalar broadcasting is unchanged behaviour."""
    ds = _vector_field()
    scalar = divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=3)
    as_tuple = divergence(ds, ("u", "v"), dims=("x", "y"), accuracy=(3, 3))
    np.testing.assert_allclose(scalar.values, as_tuple.values, atol=0.0)
