"""to_pixel and the adapter-level map_over (#522; GA-G15). interp_at's tests
moved to geoops with it (GA-B6, #631)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import DimArray

import geoarray
from geoarray import GeoArray, to_pixel
from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"


def _patches(n: int = 64, xp: Any = np) -> GeoArray:
    rng = np.random.default_rng(0)
    transforms = np.stack(
        [np.array(T) + np.array([0, 0, 640.0 * k, 0, 0, -320.0 * k]) for k in range(n)]
    )
    data = rng.random((n, 2, 16, 16))
    return GeoArray(
        xp.asarray(data),
        transforms,
        CRS,
        dims="patch band y x",
        coords=(SpectralAxis(("B11", "B12")),),
    )


def _score(p: GeoArray) -> GeoArray:
    # per-patch: b12 − b11 scaled by the patch's own x origin (reads the transform)
    diff = pa.combine(lambda a, b: a - b, pa.sel(p, band="B12"), pa.sel(p, band="B11"))
    return diff.with_data(diff.data * (p.transform[2] / 1e6))


# -- map_over -------------------------------------------------------------------------


def test_ga_g15_patches_in_one_trace(x64: ModuleType) -> None:
    jax = x64
    traces = 0

    def run(patches: GeoArray) -> GeoArray:
        nonlocal traces
        traces += 1
        return geoarray.map_over(patches, _score, "patch")

    f = jax.jit(run)
    patches = _patches().to_jax()
    out = f(patches)
    f(patches.shift_origin((1000.0, 0.0)))
    assert traces == 1
    assert out.dims == ("patch", "y", "x") and out.batch_shape == (64,)
    np.testing.assert_array_equal(
        np.asarray(out.transform), np.asarray(patches.transform)
    )
    # each patch as if run alone, eagerly
    host = _patches()
    for k in (0, 17, 63):
        alone = _score(host.isel(patch=k))
        np.testing.assert_allclose(np.asarray(out.data[k]), alone.data, rtol=1e-12)


def test_map_over_numpy_matches_jax(x64: ModuleType) -> None:
    on_numpy = geoarray.map_over(_patches(8), _score, "patch")
    on_jax = geoarray.map_over(_patches(8).to_jax(), _score, "patch")
    assert on_numpy.dims == on_jax.dims == ("patch", "y", "x")
    np.testing.assert_allclose(np.asarray(on_jax.data), on_numpy.data, rtol=1e-12)
    np.testing.assert_array_equal(np.asarray(on_jax.transform), on_numpy.transform)


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_map_over_time_keeps_the_dates(backend: str, x64: ModuleType) -> None:
    dates = TimeAxis(["2026-06-01", "2026-06-11", "2026-06-21"])
    series = GeoArray(
        np.random.default_rng(0).random((3, 2, 4, 4)),
        T,
        CRS,
        dims="time band y x",
        coords=(dates, SpectralAxis(("a", "b"))),
    )
    if backend == "jax":
        series = series.to_jax()
    out = geoarray.map_over(series, lambda img: pa.reduce(img, "sum", "band"), "time")
    assert out.dims == ("time", "y", "x") and out.batch_shape == ()
    assert out.labels("time") == dates.labels
    np.testing.assert_allclose(np.asarray(out.data), np.asarray(series.data).sum(1))


@pytest.mark.parametrize("backend", ["numpy", "jax"])
def test_map_over_to_dimarrays(backend: str, x64: ModuleType) -> None:
    patches = _patches(4)
    if backend == "jax":
        patches = patches.to_jax()
    out = geoarray.map_over(
        patches, lambda p: pa.reduce(p, "mean", ("y", "x")), "patch"
    )
    assert isinstance(out, DimArray) and out.dims == ("patch", "band")
    assert out.labels == {"band": ("B11", "B12")}


def test_map_over_refuses_spatial_dims() -> None:
    with pytest.raises(ValueError, match="non-spatial"):
        geoarray.map_over(_patches(2), lambda p: p, "x")


# -- to_pixel -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "transform",
    [T, (20.0, 3.0, 600000.0, -2.0, -20.0, 3550000.0)],
    ids=["north-up", "rotated"],
)
def test_to_pixel_inverts_pixel_centres(transform: tuple[float, ...]) -> None:
    grid = PixelGrid((5, 7), CRS)
    x, y = grid.centres(transform)
    xy = np.stack([x.ravel(), y.ravel()], axis=-1)
    rc = to_pixel(np.asarray(transform), xy)
    rows, cols = np.meshgrid(np.arange(5.0), np.arange(7.0), indexing="ij")
    np.testing.assert_allclose(rc[:, 0], rows.ravel(), atol=1e-9)
    np.testing.assert_allclose(rc[:, 1], cols.ravel(), atol=1e-9)


def test_map_over_a_non_batch_dim_keeps_batch_dims_first(x64: ModuleType) -> None:
    transforms = np.array([T, (20.0, 0.0, 600100.0, 0.0, -20.0, 3550000.0), T])
    data = np.arange(3 * 2 * 2 * 2, dtype=float).reshape(3, 2, 2, 2)
    g = GeoArray(data, transforms, CRS, dims="patch band y x")
    for carrier in (g, g.to_jax()):
        out = geoarray.map_over(carrier, lambda item: item, "band")
        assert out.dims == ("patch", "band", "y", "x")
        assert out.batch_dims == ("patch",)
        np.testing.assert_array_equal(np.asarray(out.data), data)
        np.testing.assert_array_equal(np.asarray(out.transform), transforms)


def test_map_over_on_jax_refuses_items_on_different_grids(x64: ModuleType) -> None:
    g = GeoArray(np.zeros((2, 3, 3)), T, CRS, dims="time y x").to_jax()

    def shift(item: GeoArray) -> GeoArray:
        return item.shift_origin(x64.numpy.stack([item.data[0, 0] + 1.0, 0.0]))

    g = g.with_data(x64.numpy.stack([x64.numpy.zeros((3, 3)), x64.numpy.ones((3, 3))]))
    with pytest.raises(Exception, match="different grids"):
        x64.block_until_ready(geoarray.map_over(g, shift, "time").data)


def test_map_over_band_keeps_per_band_attrs(x64: ModuleType) -> None:
    g = GeoArray(
        np.zeros((2, 3, 3)),
        T,
        CRS,
        dims="band y x",
        attrs={"band_names": ("red", "nir"), "sensor": "S2A"},
    )
    for carrier in (g, g.to_jax()):
        out = geoarray.map_over(carrier, lambda item: item, "band")
        assert out.attrs["band_names"] == ("red", "nir")
        assert out.attrs["sensor"] == "S2A"
