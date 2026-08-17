"""Tests for per-dim geometry in :func:`xrgrad.divergence`."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from xrgrad import divergence, partial


def _ocean_box(nlon: int = 12, nlat: int = 10) -> xr.Dataset:
    """A 3-D field on spherical lon/lat plus a stretched depth axis."""
    lon = np.linspace(-40.0, -10.0, nlon)
    lat = np.linspace(10.0, 40.0, nlat)
    depth = np.array([0.0, 5.0, 15.0, 35.0, 80.0, 150.0])
    rng = np.random.default_rng(0)
    shape = (depth.size, lat.size, lon.size)
    dims = ("depth", "lat", "lon")
    coords = {"depth": depth, "lat": lat, "lon": lon}
    return xr.Dataset(
        {
            name: xr.DataArray(rng.normal(size=shape), dims=dims, coords=coords)
            for name in ("u", "v", "w")
        }
    )


def test_mixed_geometry_matches_two_step_composition():
    """The pattern this replaces: spherical 2-D + separate vertical partial."""
    ds = _ocean_box()
    mixed = divergence(
        ds,
        ("u", "v", "w"),
        dims=("lon", "lat", "depth"),
        geometry=("spherical", "spherical", "rectilinear"),
    )
    composed = divergence(
        ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical"
    ) + partial(ds["w"], "depth", geometry="rectilinear")
    # Not bit-identical: the curvature term is now summed last, so the
    # difference is pure float associativity.
    np.testing.assert_allclose(mixed.values, composed.values, rtol=1e-12, atol=1e-15)


def test_mixed_geometry_includes_curvature_term():
    """The -(v tan φ)/R metric term survives the per-dim dispatch."""
    ds = _ocean_box()
    const = ds.copy()
    const["u"] = xr.zeros_like(ds["u"])
    const["v"] = xr.ones_like(ds["v"])
    const["w"] = xr.zeros_like(ds["w"])
    got = divergence(
        const,
        ("u", "v", "w"),
        dims=("lon", "lat", "depth"),
        geometry=("spherical", "spherical", "rectilinear"),
    )
    phi = np.deg2rad(ds["lat"].values)
    expected = np.broadcast_to((-np.tan(phi) / 6_371_000.0)[None, :, None], got.shape)
    np.testing.assert_allclose(got.values, expected, atol=1e-18)


def test_scalar_geometry_is_unchanged():
    ds = _ocean_box()
    scalar = divergence(ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical")
    sequence = divergence(
        ds, ("u", "v"), dims=("lon", "lat"), geometry=("spherical", "spherical")
    )
    np.testing.assert_allclose(scalar.values, sequence.values, atol=0.0)


def test_all_cartesian_three_dims():
    ds = _ocean_box()
    got = divergence(
        ds,
        ("u", "v", "w"),
        dims=("lon", "lat", "depth"),
        geometry=("cartesian", "cartesian", "rectilinear"),
    )
    expected = (
        partial(ds["u"], "lon")
        + partial(ds["v"], "lat")
        + partial(ds["w"], "depth", geometry="rectilinear")
    )
    np.testing.assert_allclose(got.values, expected.values, atol=0.0)


def test_geometry_sequence_length_mismatch_raises():
    ds = _ocean_box()
    with pytest.raises(ValueError, match="geometry sequence length"):
        divergence(
            ds,
            ("u", "v", "w"),
            dims=("lon", "lat", "depth"),
            geometry=("spherical", "spherical"),
        )


def test_unknown_geometry_in_sequence_raises():
    ds = _ocean_box()
    with pytest.raises(ValueError, match="Unknown geometry"):
        divergence(
            ds,
            ("u", "v"),
            dims=("lon", "lat"),
            geometry=("spherical", "conformal"),
        )


def test_lone_spherical_axis_raises():
    """The metric couples lon and lat; neither is spherical on its own."""
    ds = _ocean_box()
    with pytest.raises(ValueError, match="must be exactly the"):
        divergence(
            ds,
            ("u", "v", "w"),
            dims=("lon", "lat", "depth"),
            geometry=("spherical", "rectilinear", "rectilinear"),
        )


def test_all_spherical_tuple_points_at_the_mixed_form():
    """Marking the vertical axis spherical too is the mistake this guides."""
    ds = _ocean_box()
    with pytest.raises(ValueError, match="per-dim geometry sequence"):
        divergence(
            ds,
            ("u", "v", "w"),
            dims=("lon", "lat", "depth"),
            geometry=("spherical", "spherical", "spherical"),
        )


def test_all_spherical_three_dims_keeps_the_2d_message():
    """The pre-existing all-spherical error still names the 2-D constraint."""
    ds = _ocean_box()
    with pytest.raises(ValueError, match="2-D"):
        divergence(
            ds, ("u", "v", "w"), dims=("lon", "lat", "lat"), geometry="spherical"
        )


def test_spherical_kwargs_do_not_reach_cartesian_backend():
    """``radius`` accompanies a mixed geometry without breaking the vertical."""
    ds = _ocean_box()
    got = divergence(
        ds,
        ("u", "v", "w"),
        dims=("lon", "lat", "depth"),
        geometry=("spherical", "spherical", "rectilinear"),
        radius=6_000_000.0,
    )
    assert np.isfinite(got.values).all()


def test_unknown_kwarg_still_raises():
    """Per-geometry filtering must not swallow a typo."""
    ds = _ocean_box()
    with pytest.raises(ValueError, match="Unexpected keyword"):
        divergence(
            ds,
            ("u", "v"),
            dims=("lon", "lat"),
            geometry="spherical",
            radius_typo=6e6,
        )


def test_custom_lon_lat_names_route_the_curvature_term():
    ds = _ocean_box().rename({"lon": "longitude", "lat": "latitude"})
    got = divergence(
        ds,
        ("u", "v", "w"),
        dims=("longitude", "latitude", "depth"),
        geometry=("spherical", "spherical", "rectilinear"),
        lon="longitude",
        lat="latitude",
    )
    assert got.dims == ds["u"].dims
