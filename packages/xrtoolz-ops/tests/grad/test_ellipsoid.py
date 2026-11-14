"""The spherical metric: the sphere by default, an ellipsoid's on request (#364)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from geoarray import GeoArray
from geoops.geom import Gradient2D

from xrops import grad


TRANSFORM = (0.25, 0.0, -10.0, 0.0, -0.25, 60.0)


def _field(nan: bool = False) -> tuple[xr.DataArray, GeoArray]:
    h, w = 12, 16
    lon = -10.0 + 0.125 + 0.25 * np.arange(w)
    lat = 60.0 - 0.125 - 0.25 * np.arange(h)
    rng = np.random.default_rng(0)
    values = np.sin(np.deg2rad(lon))[None, :] * np.cos(np.deg2rad(lat))[:, None]
    values = values + 0.01 * rng.normal(size=(h, w))
    if nan:
        values[4:6, 5:8] = np.nan
    da = xr.DataArray(values, dims=("lat", "lon"), coords={"lat": lat, "lon": lon})
    return da, GeoArray(values, TRANSFORM, "EPSG:4326", dims=("y", "x"))


@pytest.mark.parametrize("nan_policy", ["propagate", "adaptive"])
def test_wgs84_matches_geoops_gradient_on_a_geographic_crs(nan_policy) -> None:
    da, ga = _field(nan=nan_policy == "adaptive")
    want = np.asarray(Gradient2D(nan_policy=nan_policy)(ga))
    kw = dict(geometry="spherical", ellipsoid="WGS84", nan_policy=nan_policy)
    d_lat = grad.partial(da, "lat", **kw)
    d_lon = grad.partial(da, "lon", **kw)
    np.testing.assert_allclose(d_lat.values, want[0], rtol=1e-10, equal_nan=True)
    np.testing.assert_allclose(d_lon.values, want[1], rtol=1e-10, equal_nan=True)


def test_the_sphere_is_the_default_and_differs_from_the_ellipsoid_slightly() -> None:
    da, _ = _field()
    sphere = grad.partial(da, "lon", geometry="spherical")
    wgs84 = grad.partial(da, "lon", geometry="spherical", ellipsoid="WGS84")
    ratio = float(np.nanmax(np.abs(sphere / wgs84 - 1.0)))
    assert 0.0 < ratio < 0.01
    named = da.rename("f")
    g = grad.gradient(
        named, dims=("lon", "lat"), geometry="spherical", ellipsoid="WGS84"
    )
    xr.testing.assert_allclose(g["df_dx"], wgs84.rename("df_dx"))


@pytest.mark.parametrize("fn", ["divergence", "curl"])
def test_vector_operators_refuse_an_ellipsoid(fn) -> None:
    da, _ = _field()
    ds = xr.Dataset({"u": da, "v": da})
    with pytest.raises(NotImplementedError, match="ellipsoid"):
        getattr(grad, fn)(
            ds, ("u", "v"), dims=("lon", "lat"), geometry="spherical", ellipsoid="WGS84"
        )
