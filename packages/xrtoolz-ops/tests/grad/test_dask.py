"""Dask input: lazy stencils with chunk halos, equal to the in-memory result (#270)."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr


dask = pytest.importorskip("dask")
import dask.array as dsa

from xrops import grad


def _no_compute(*args, **kwargs):
    raise AssertionError("the graph was computed while being built")


def _field(geometry: str, *, masked: bool) -> xr.Dataset:
    rng = np.random.default_rng(0)
    if geometry == "spherical":
        coords = {
            "lat": np.linspace(-60.0, 60.0, 17),
            "lon": np.arange(0.0, 360.0, 10.0),
        }
        dims = ("lat", "lon")
    elif geometry == "rectilinear":
        coords = {
            "y": np.cumsum(rng.uniform(0.5, 2.0, 17)),
            "x": np.cumsum(rng.uniform(0.5, 2.0, 23)),
        }
        dims = ("y", "x")
    else:
        coords = {"y": np.arange(17.0) * 3.0, "x": np.arange(23.0) * 2.0}
        dims = ("y", "x")
    shape = tuple(len(coords[d]) for d in dims)
    u, v = rng.normal(size=shape), rng.normal(size=shape)
    if masked:
        for a in (u, v):
            a[5:8, 6:9] = np.nan
            a[0, 3] = a[-1, -2] = np.nan
    return xr.Dataset({"u": (dims, u), "v": (dims, v)}, coords=coords)


def _xy(geometry: str) -> tuple[str, str]:
    return ("lon", "lat") if geometry == "spherical" else ("x", "y")


def _lazy(ds: xr.Dataset, chunk: int = 5) -> xr.Dataset:
    return ds.chunk({d: chunk for d in ds.dims})


def _same(lazy: xr.DataArray, eager: xr.DataArray) -> None:
    assert isinstance(lazy.data, dsa.Array)
    with dask.config.set(scheduler="sync"):
        xr.testing.assert_allclose(lazy.compute(), eager, rtol=0, atol=0)


@pytest.mark.parametrize("geometry", ["cartesian", "rectilinear", "spherical"])
@pytest.mark.parametrize("nan_policy", ["propagate", "adaptive"])
@pytest.mark.parametrize("accuracy", [1, 2, 3, 4])
def test_partial_on_chunks_equals_the_whole_array(
    geometry, nan_policy, accuracy
) -> None:
    ds = _field(geometry, masked=nan_policy == "adaptive")
    kw = dict(geometry=geometry, nan_policy=nan_policy, accuracy=accuracy)
    lazy = _lazy(ds)
    for dim in _xy(geometry):
        with dask.config.set(scheduler=_no_compute):
            out = grad.partial(lazy["u"], dim, **kw)
        _same(out, grad.partial(ds["u"], dim, **kw))


@pytest.mark.parametrize("nan_policy", ["propagate", "adaptive"])
def test_periodic_longitude_wraps_across_chunks(nan_policy) -> None:
    ds = _field("spherical", masked=True)
    kw = dict(geometry="spherical", periodic=True, nan_policy=nan_policy, accuracy=3)
    _same(grad.partial(_lazy(ds)["u"], "lon", **kw), grad.partial(ds["u"], "lon", **kw))


def test_chunks_narrower_than_the_halo_are_merged() -> None:
    ds = _field("cartesian", masked=False)
    out = grad.partial(_lazy(ds, chunk=2)["u"], "x", accuracy=4, order=2)
    _same(out, grad.partial(ds["u"], "x", accuracy=4, order=2))


@pytest.mark.parametrize("geometry", ["cartesian", "spherical"])
def test_vector_operators_stay_lazy(geometry) -> None:
    ds = _field(geometry, masked=True)
    lazy = _lazy(ds)
    dims = _xy(geometry)
    kw = dict(geometry=geometry, nan_policy="adaptive", accuracy=2)
    with dask.config.set(scheduler=_no_compute):
        div = grad.divergence(lazy, ("u", "v"), dims=dims, **kw)
        curl = grad.curl(lazy, ("u", "v"), dims=dims, **kw)
        lap = grad.laplacian(lazy["u"], dims=dims, **kw)
        g = grad.Gradient(dims=dims[::-1], geometry=geometry)(lazy["u"])
    _same(div, grad.divergence(ds, ("u", "v"), dims=dims, **kw))
    _same(curl, grad.curl(ds, ("u", "v"), dims=dims, **kw))
    _same(lap, grad.laplacian(ds["u"], dims=dims, **kw))
    _same(g, grad.Gradient(dims=dims[::-1], geometry=geometry)(ds["u"]))


def test_periodic_cartesian_stays_lazy() -> None:
    ds = _field("cartesian", masked=False)
    with dask.config.set(scheduler=_no_compute):
        out = grad.partial(_lazy(ds)["u"], "x", periodic=True, accuracy=2)
    _same(out, grad.partial(ds["u"], "x", periodic=True, accuracy=2))


@pytest.mark.parametrize("geometry", ["cartesian", "rectilinear"])
def test_an_axis_shorter_than_the_halo_keeps_the_adaptive_descent(geometry) -> None:
    ds = _field(geometry, masked=False).isel(x=slice(0, 2))
    kw = dict(geometry=geometry, nan_policy="adaptive", accuracy=4)
    lazy = ds.chunk({"y": 5, "x": 1})
    _same(grad.partial(lazy["u"], "x", **kw), grad.partial(ds["u"], "x", **kw))


@pytest.mark.parametrize("nan_policy", ["propagate", "adaptive"])
def test_the_lazy_dtype_is_the_computed_one(nan_policy) -> None:
    ds = _field("rectilinear", masked=True).astype("float32")
    lazy = grad.partial(
        _lazy(ds)["u"], "x", geometry="rectilinear", nan_policy=nan_policy
    )
    eager = grad.partial(ds["u"], "x", geometry="rectilinear", nan_policy=nan_policy)
    assert lazy.dtype == lazy.compute().dtype == eager.dtype
