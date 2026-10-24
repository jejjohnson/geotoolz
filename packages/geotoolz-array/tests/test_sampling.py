"""to_pixel, interp_at and the adapter-level map_over (#522; GA-G15, GA-G16)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit_array import DimArray

import geoarray
from geoarray import GeoArray, GeoPoints, interp_at, pad_points, to_pixel
from geoarray.coords import PixelGrid, SpectralAxis, TimeAxis, VariableAxis
from geoarray.types import GeoTimeSeries, matches


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


# -- interp_at ------------------------------------------------------------------------


def _ramp() -> GeoArray:
    # v = 10·row + col, so bilinear sampling is exact everywhere inside
    rows, cols = np.meshgrid(np.arange(6.0), np.arange(5.0), indexing="ij")
    return GeoArray(10 * rows + cols, T, CRS, dims="y x")


def _at(g: GeoArray, rc: list[tuple[float, float]], **kw: Any) -> GeoPoints:
    a, _, c, _, e, f = T
    xy = np.array([[c + a * (col + 0.5), f + e * (row + 0.5)] for row, col in rc])
    return GeoPoints(np.zeros(len(rc)), xy=xy, dims="obs", crs=CRS, **kw)


def test_bilinear_is_exact_on_a_ramp_and_at_the_edges() -> None:
    g = _ramp()
    pts = _at(g, [(0, 0), (2.5, 1.25), (5, 4), (5, 0.5), (0.3, 4)])
    out = interp_at(g, pts)
    np.testing.assert_allclose(out.data, [0.0, 26.25, 54.0, 50.5, 7.0], atol=1e-9)
    assert out.valid.all()


def test_outside_gives_nan_and_invalid() -> None:
    out = interp_at(_ramp(), _at(_ramp(), [(-3, 0), (2, 9), (1, 1)]))
    assert np.isnan(out.data[:2]).all() and out.data[2] == 11.0
    assert out.valid.tolist() == [False, False, True]


def test_nearest() -> None:
    out = interp_at(_ramp(), _at(_ramp(), [(2.4, 1.6), (5.2, 3.9)]), method="nearest")
    assert out.data.tolist() == [22.0, 54.0]


def test_padded_points_give_zero() -> None:
    pts = pad_points(_at(_ramp(), [(1, 1), (2, 2)]), 4)
    out = interp_at(_ramp(), pts)
    assert out.data.tolist() == [11.0, 22.0, 0.0, 0.0]
    assert out.valid.tolist() == [True, True, False, False]


def test_sampling_a_stack_at_stations_is_a_time_series() -> None:
    met = VariableAxis(("t2m", "psfc"), ("K", "hPa"))
    hours = TimeAxis(
        np.datetime64("2026-06-01", "ns") + np.arange(3) * np.timedelta64(1, "h")
    )
    stack = GeoArray(
        np.random.default_rng(0).random((3, 2, 6, 5)),
        T,
        CRS,
        dims="time variable y x",
        coords=(hours, met),
    )
    stations = _at(stack, [(1, 1), (4, 3)])
    series = interp_at(stack, stations)
    assert series.dims == ("time", "variable", "obs")
    assert matches(series, GeoTimeSeries)
    np.testing.assert_allclose(series.data[:, :, 1], stack.data[:, :, 4, 3])


def test_each_sample_reads_its_own_frame() -> None:
    frames = GeoArray(
        np.arange(3.0)[:, None, None] * np.ones((3, 6, 5)), T, CRS, dims="time y x"
    )
    pts = _at(frames, [(1, 1), (2, 2), (3, 3)])
    out = interp_at(frames, pts, frame=np.array([2, 0, 1]))
    assert out.dims == ("obs",) and out.data.tolist() == [2.0, 0.0, 1.0]


def test_sample_needs_one_crs() -> None:
    pts = GeoPoints(np.zeros(1), xy=np.zeros((1, 2)), dims="obs")
    with pytest.raises(ValueError, match="transform the points"):
        interp_at(_ramp(), pts)


def test_ga_g16_cost_unchanged_by_padding(x64: ModuleType) -> None:
    """J and ∇ₓJ do not depend on the pad length, and are finite with NaN padding."""
    jax = x64
    jnp = x64.numpy
    truth = _ramp()
    obs_rc = [(0.5, 0.5), (2.2, 3.1), (4.9, 1.0), (3.0, 3.0)]
    clean = interp_at(truth, _at(truth, obs_rc))

    def cost(ssh: GeoArray, obs: GeoPoints, sigma: float) -> Any:
        predicted = interp_at(ssh, obs)
        # d = y − H(x), zero on padded slots
        d = jnp.where(obs.valid, obs.data - predicted.data, 0.0)
        # J = ½ dᵀ R⁻¹ d, R = σ² I
        return 0.5 * (d @ d) / sigma**2

    state = truth.with_data(truth.data * 0.9).to_jax()
    results = []
    for n in (8, 32):
        padded = pad_points(clean.with_data(clean.data), n)
        nan_xy = np.where(
            np.asarray(padded.valid)[:, None], np.asarray(padded.xy), np.nan
        )
        padded = GeoPoints(
            np.where(np.asarray(padded.valid), np.asarray(padded.data), np.nan),
            xy=nan_xy,
            valid=padded.valid,
            dims="obs",
            crs=CRS,
            fill_value=np.nan,
        ).to_jax()
        f = jax.jit(
            jax.value_and_grad(
                lambda data, p=padded: cost(state.with_data(data), p, 0.5)
            )
        )
        j, g = f(state.data)
        results.append((float(j), np.asarray(g)))
    assert np.isfinite(results[0][0]) and np.isfinite(results[0][1]).all()
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-14)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-14, atol=0)
