"""The index operators on every carrier (#533).

Each index runs one element-wise kernel through pipekit-array's
``KernelOp``, so the same operator must give the same values on

- a NumPy `GeoArray` (the reference);
- a JAX `GeoArray`, eagerly and under ``jax.jit``, tracing once across
  transforms (GA-G2);
- its ``to_xarray()``, eagerly and dask-backed, computing nothing until
  ``.compute()`` (GH1);
- a plain array.
"""

from __future__ import annotations

import numpy as np
import pytest
from _helpers import toy_geoarray
from geoarray import GeoArray

import geoops.indices as gi


jax = pytest.importorskip("jax")
xr = pytest.importorskip("xarray")
pytest.importorskip("dask")

DESCRIPTIONS = ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")

INDEX_OPS = [
    pytest.param(cls, id=cls.__name__)
    for cls in (
        gi.NormalizedDifference,
        gi.NDVI,
        gi.NDWI,
        gi.NDBI,
        gi.NBR,
        gi.SAVI,
        gi.EVI,
        gi.EVI2,
        gi.ARVI,
        gi.GCI,
        gi.kNDVI,
        gi.MNDWI,
        gi.NDMI,
        gi.NDSI,
        gi.NBR2,
        gi.BAIS2,
        gi.BSI,
        gi.IronOxide,
        gi.ClayMinerals,
        gi.CIRI,
    )
]


def _make(cls: type) -> object:
    if cls is gi.NormalizedDifference:
        return cls(a=6, b=2)
    return cls()


@pytest.fixture(autouse=True)
def _x64():
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture
def scene() -> GeoArray:
    """A 10-band Sentinel-2 stack, 6 x 5, with two fill pixels."""
    rng = np.random.default_rng(3)
    values = rng.uniform(0.05, 0.6, size=(10, 6, 5)).astype(np.float32)
    return toy_geoarray(values, with_fill_pixels=True).with_attrs(
        descriptions=DESCRIPTIONS
    )


@pytest.mark.parametrize("cls", INDEX_OPS)
def test_index_agrees_on_every_carrier(cls: type, scene: GeoArray) -> None:
    op = _make(cls)
    want = op(scene)
    assert isinstance(want, GeoArray)
    assert want.dims == ("y", "x")
    assert np.isnan(want.fill_value)
    assert np.isnan(np.asarray(want)[0, 0])  # a fill pixel of the input

    on_jax = op(scene.to_jax())
    jitted = jax.jit(lambda g: op(g))(scene.to_jax())
    da = scene.to_xarray()
    on_xarray = op(da)
    on_dask = op(da.chunk({"y": 3}))
    plain = op(np.asarray(scene))

    assert on_jax.backend == "jax"
    assert isinstance(on_xarray, xr.DataArray)
    assert on_xarray.dims == ("y", "x")
    assert on_dask.chunks is not None
    for got in (on_jax, jitted, on_xarray, on_dask.compute()):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-6, equal_nan=True
        )
    # a plain array declares no fill, so only its fill pixels differ
    valid = np.isfinite(np.asarray(want))
    assert isinstance(plain, np.ndarray)
    np.testing.assert_allclose(plain[valid], np.asarray(want)[valid], rtol=1e-6)


def test_jit_traces_once_across_transforms(scene: GeoArray) -> None:
    """GA-G2: a new transform of the same shape does not retrace."""
    traces = []

    @jax.jit
    def step(g: GeoArray) -> GeoArray:
        traces.append(1)
        return gi.NDVI(nir=6, red=2)(g)

    a = scene.to_jax()
    shifted = scene.replace(transform=np.asarray(scene.transform) + 5.0).to_jax()
    step(a)
    out = step(shifted)
    assert len(traces) == 1
    np.testing.assert_array_equal(out.transform, shifted.transform)


def test_dask_cube_is_lazy_until_compute(scene: GeoArray) -> None:
    """NDVI on a dask (time, band, y, x) cube builds a graph and equals each slice."""
    frames = [scene, scene.map(lambda d: d * 0.5)]
    cube = xr.concat([f.to_xarray() for f in frames], dim="time").chunk({"time": 1})
    calls = []

    def counting(nir, red, *, eps):
        calls.append(1)
        return gi.ndvi(nir, red, eps=eps)

    from geoops._src.kernels import band_kernel

    lazy = band_kernel(counting, cube, (6, 2), dim="band", name="NDVI", eps=1e-10)
    assert lazy.dims == ("time", "y", "x")
    blocks = len(calls)
    assert hasattr(lazy.data, "dask")
    out = lazy.compute()
    assert len(calls) > blocks
    for t, frame in enumerate(frames):
        np.testing.assert_allclose(
            out.isel(time=t).values,
            np.asarray(gi.NDVI(nir=6, red=2)(frame)),
            rtol=1e-6,
            equal_nan=True,
        )


def test_band_names_resolve_against_a_dataarray_coordinate(scene: GeoArray) -> None:
    da = scene.to_xarray().assign_coords(band=list(DESCRIPTIONS))
    got = gi.NDVI(nir="B08", red="B04")(da)
    want = gi.NDVI(nir=6, red=2)(scene)
    np.testing.assert_allclose(got.values, np.asarray(want), equal_nan=True)
    assert np.isnan(got.attrs["fill_value"])


def test_time_stack_keeps_a_singleton_band_dim(scene: GeoArray) -> None:
    stack = GeoArray(
        np.stack([np.asarray(scene)] * 3),
        scene.transform,
        scene.crs,
        fill_value=scene.fill_value,
        dims=("time", "band", "y", "x"),
    )
    out = gi.NDVI(nir=6, red=2)(stack)
    assert out.dims == ("time", "band", "y", "x")
    assert out.shape == (3, 1, 6, 5)
    jitted = jax.jit(lambda g: gi.NDVI(nir=6, red=2)(g))(stack.to_jax())
    np.testing.assert_allclose(np.asarray(jitted), np.asarray(out), equal_nan=True)


def test_dnbr_on_every_carrier(scene: GeoArray) -> None:
    pre = gi.NBR(nir=6, swir2=9)(scene)
    post = pre.map(lambda d: d * 0.5)
    want = gi.dNBR()(pre, post)
    on_jax = jax.jit(lambda a, b: gi.dNBR()(a, b))(pre.to_jax(), post.to_jax())
    on_dask = gi.dNBR()(
        pre.to_xarray().chunk({"y": 3}), post.to_xarray().chunk({"y": 3})
    )
    for got in (on_jax, on_dask.compute()):
        np.testing.assert_allclose(
            np.asarray(got), np.asarray(want), rtol=1e-6, equal_nan=True
        )
