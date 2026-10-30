"""xrops.grad on geoops's kernels equals xrgrad on finitediffx (#364).

xrgrad is the reference here only; it is deleted once this parity holds.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


xrgrad = pytest.importorskip("xrgrad")
from xrops import grad


def _field(geometry: str, *, masked: bool) -> xr.Dataset:
    rng = np.random.default_rng(0)
    if geometry == "spherical":
        lon = np.arange(0.0, 360.0, 10.0)
        lat = np.linspace(-60.0, 60.0, 13)
        dims, coords = ("lat", "lon"), {"lat": lat, "lon": lon}
    elif geometry == "rectilinear":
        x = np.cumsum(rng.uniform(0.5, 2.0, 15))
        y = np.cumsum(rng.uniform(0.5, 2.0, 12))
        dims, coords = ("y", "x"), {"y": y, "x": x}
    else:
        dims, coords = (
            ("y", "x"),
            {"y": np.arange(12.0) * 3.0, "x": np.arange(15.0) * 2.0},
        )
    shape = tuple(len(coords[d]) for d in dims)
    u, v = rng.normal(size=shape), rng.normal(size=shape)
    if masked:
        u[3:5, 4:7] = np.nan
        v[3:5, 4:7] = np.nan
        u[0, 0] = v[-1, -1] = np.nan
    return xr.Dataset({"u": (dims, u), "v": (dims, v)}, coords=coords)


GEOMETRIES = ["cartesian", "rectilinear", "spherical"]
POLICIES = ["propagate", "adaptive"]


def _xy(geometry: str) -> tuple[str, str]:
    return ("lon", "lat") if geometry == "spherical" else ("x", "y")


def _close(
    a: xr.DataArray | xr.Dataset, b: xr.DataArray | xr.Dataset, rtol: float = 1e-12
) -> None:
    xr.testing.assert_allclose(a, b, rtol=rtol, atol=1e-12)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("nan_policy", POLICIES)
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("accuracy", [1, 2, 3])
def test_partial_and_gradient(geometry, nan_policy, masked, accuracy) -> None:
    ds = _field(geometry, masked=masked)
    kw = dict(geometry=geometry, nan_policy=nan_policy, accuracy=accuracy)
    # the non-uniform stencil coefficients of order 3 are solved in another
    # order than finitediffx's: the two differ by rounding (1e-10 relative)
    rtol = 1e-9 if geometry == "rectilinear" else 1e-12
    for dim in _xy(geometry):
        a, b = grad.partial(ds["u"], dim, **kw), xrgrad.partial(ds["u"], dim, **kw)
        _close(a, b, rtol)
    dims = _xy(geometry)
    a, b = (
        grad.gradient(ds["u"], dims=dims, **kw),
        xrgrad.gradient(ds["u"], dims=dims, **kw),
    )
    _close(a, b, rtol)


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("nan_policy", POLICIES)
@pytest.mark.parametrize("masked", [False, True])
def test_vector_operators(geometry, nan_policy, masked) -> None:
    ds = _field(geometry, masked=masked)
    dims = _xy(geometry)
    kw = dict(geometry=geometry, nan_policy=nan_policy, accuracy=2)
    _close(
        grad.divergence(ds, ("u", "v"), dims=dims, **kw),
        xrgrad.divergence(ds, ("u", "v"), dims=dims, **kw),
    )
    _close(
        grad.curl(ds, ("u", "v"), dims=dims, **kw),
        xrgrad.curl(ds, ("u", "v"), dims=dims, **kw),
    )
    _close(
        grad.laplacian(ds["u"], dims=dims, **kw),
        xrgrad.laplacian(ds["u"], dims=dims, **kw),
    )


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("method", ["central", "forward", "backward"])
def test_cartesian_orders_methods_and_periodic(order, method) -> None:
    ds = _field("cartesian", masked=False)
    for periodic in (False, True):
        kw = dict(order=order, method=method, periodic=periodic, accuracy=2)
        _close(grad.partial(ds["u"], "x", **kw), xrgrad.partial(ds["u"], "x", **kw))


def test_periodic_longitude_and_float32() -> None:
    ds = _field("spherical", masked=True).astype("float32")
    kw = dict(geometry="spherical", periodic=True, nan_policy="adaptive")
    a = grad.partial(ds["u"], "lon", **kw)
    b = xrgrad.partial(ds["u"], "lon", **kw)
    assert a.dtype == b.dtype
    xr.testing.assert_allclose(a, b, rtol=1e-5)


def test_grid_metrics_and_constants() -> None:
    ds = _field("spherical", masked=False)
    for ours, theirs in zip(
        grad.grid_metrics_from_coords(ds),
        xrgrad.grid_metrics_from_coords(ds),
        strict=True,
    ):
        xr.testing.assert_identical(ours, theirs)
    assert (grad.EARTH_RADIUS, grad.GRAVITY, grad.OMEGA) == (
        xrgrad.EARTH_RADIUS,
        xrgrad.GRAVITY,
        xrgrad.OMEGA,
    )
