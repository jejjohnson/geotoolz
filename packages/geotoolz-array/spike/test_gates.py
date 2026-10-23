"""One test per spike gate (#516): GA-G2 and GA-G15 to GA-G21.

Run from packages/geotoolz-array: ``uv run pytest spike -o addopts= -q``.
Each test's docstring is the gate's pass condition from geoarray.md.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import xarray as xr
from conftest import CRS, T0
from proto.adapters import BackendError, register_xarray, unregister_adapter
from proto.carrier import GeoArray
from proto.dimarray import DimArray
from proto.kernelop import KernelOp, KernelSpec
from proto.kernels import (
    background_stats,
    gradient,
    matched_filter_score,
    normalized_difference,
    spacing_of,
)
from proto.sample import sample
from proto.shared import combine, isel, map_over, reduce, stack
from proto.xr import from_xarray, to_xarray

import geoarray
from geoarray._src.errors import GridMismatchError


ND = KernelOp(normalized_difference, "(),()->()", core={}, eps=1e-10)
GRAD = KernelOp(
    gradient,
    "(y,x)->(y,x),(y,x)",
    core={"y": "y", "x": "x"},
    geometry={"spacing": spacing_of},
)
STATS = KernelOp(
    background_stats,
    KernelSpec("(r,c)->(c),(c,d)", vectorize=True, sizes=(("d", "c"),)),
    core={"r": "y", "c": "band", "d": "band2"},
)


def scene(
    rng: np.random.Generator, shape: tuple[int, int, int] = (4, 32, 24), t=T0
) -> GeoArray:
    """A (C, H, W) float64 NumPy carrier with band labels."""
    data = rng.random(shape)
    bands = tuple(f"B{i}" for i in range(shape[0]))
    return GeoArray(data, t, ("band", "y", "x"), CRS, labels=(("band", bands),))


def random_transform(rng: np.random.Generator) -> np.ndarray:
    """A north-up 20 m transform with a random origin."""
    c = 600000.0 + rng.uniform(-1e5, 1e5)
    f = 3550000.0 + rng.uniform(-1e5, 1e5)
    return np.array([20.0, 0.0, c, 0.0, -20.0, f])


# -- GA-G2 ---------------------------------------------------------------------


def test_g2_one_trace_many_scenes(rng: np.random.Generator) -> None:
    """One trace across 100 random transforms of the same shape, dims, CRS and fill;
    a new CRS or shape adds exactly one trace."""
    traces = 0

    def step(g: GeoArray) -> GeoArray:
        nonlocal traces
        traces += 1
        m = ND(isel(g, "band", 3), isel(g, "band", 2))
        gy, gx = GRAD(m)  # spacing derived from the traced transform leaf
        return combine(lambda a, b: jnp.hypot(a, b), gy, gx)

    f = jax.jit(step)
    base = scene(rng).to_jax()
    for _ in range(100):
        g = GeoArray(
            base.data,
            jnp.asarray(random_transform(rng)),
            base.dims,
            base.crs,
            labels=base.labels,
        )
        f(g).data.block_until_ready()
    assert traces == 1
    other_crs = GeoArray(
        base.data, base.transform, base.dims, "EPSG:32614", labels=base.labels
    )
    f(other_crs)
    assert traces == 2
    f(scene(rng, (4, 16, 24)).to_jax())
    assert traces == 3


# -- GA-G15 --------------------------------------------------------------------


def _patches(g: GeoArray, size: int) -> GeoArray:
    """Non-overlapping size × size patches, each with its own transform."""
    _, H, W = g.shape
    t = np.asarray(g.transform)
    data, rows = [], []
    for i in range(0, H, size):
        for j in range(0, W, size):
            data.append(np.asarray(g.data)[:, i : i + size, j : j + size])
            # origin moves by (j, i) pixels: c' = c + a·j + b·i, f' = f + d·j + e·i
            rows.append(
                [
                    t[0],
                    t[1],
                    t[2] + t[0] * j + t[1] * i,
                    t[3],
                    t[4],
                    t[5] + t[3] * j + t[4] * i,
                ]
            )
    return GeoArray(
        np.stack(data), np.array(rows), ("patch", *g.dims), g.crs, labels=g.labels
    )


def _stitch(p: GeoArray, t0: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Place each patch back by its own transform's origin."""
    out = np.full(shape, np.nan)
    data, rows = np.asarray(p.data), np.asarray(p.transform)
    for k in range(data.shape[0]):
        j = round((rows[k, 2] - t0[2]) / t0[0])
        i = round((rows[k, 5] - t0[5]) / t0[4])
        h, w = data.shape[-2:]
        out[i : i + h, j : j + w] = data[k]
    return out


def test_g15_patches_one_trace_and_stitch(rng: np.random.Generator) -> None:
    """64 patches with different transforms run in one trace via map_over, and the
    stitch matches today's carrier within float tolerance."""
    big = scene(rng, (4, 128, 128))
    traces = 0

    def per_patch(g: GeoArray) -> GeoArray:
        return ND(isel(g, "band", 3), isel(g, "band", 2))

    def run(p: GeoArray) -> GeoArray:
        nonlocal traces
        traces += 1
        return map_over(per_patch, p, "patch")

    f = jax.jit(run)
    patches = _patches(big, 16).to_jax()
    assert patches.transform.shape == (64, 6)
    out = f(patches)
    # a second scene, other origins: still one trace
    shifted = GeoArray(
        big.data, random_transform(rng), big.dims, big.crs, labels=big.labels
    )
    f(_patches(shifted, 16).to_jax())
    assert traces == 1
    assert out.dims == ("patch", "y", "x")
    np.testing.assert_array_equal(
        np.asarray(out.transform), np.asarray(patches.transform)
    )
    # today's carrier (geotoolz-array 0.2.0) on the whole scene
    today = geoarray.GeoArray(np.asarray(big.data), T0, CRS)
    ref = geoarray.combine(
        lambda a, b: normalized_difference(a, b, eps=1e-10), today[3], today[2]
    )
    stitched = _stitch(out.to_numpy(), np.array(T0), (128, 128))
    np.testing.assert_allclose(stitched, np.asarray(ref.data), rtol=1e-12, atol=0)


# -- GA-G16 --------------------------------------------------------------------


def _obs(rng: np.random.Generator, n_real: int, n_pad: int, g: GeoArray):
    """n_real points inside the grid, padded to n_pad with NaN coordinates."""
    t = np.asarray(g.transform)
    H, W = g.shape[-2:]
    cols = rng.uniform(1, W - 2, n_real)
    rows_ = rng.uniform(1, H - 2, n_real)
    xy = np.stack([t[2] + t[0] * cols, t[5] + t[4] * rows_], axis=1)
    y = rng.normal(size=n_real)
    pad = n_pad - n_real
    xy = np.concatenate([xy, np.full((pad, 2), np.nan)])
    y = np.concatenate([y, np.full(pad, np.nan)])
    valid = np.arange(n_pad) < n_real
    return jnp.asarray(xy), jnp.asarray(y), jnp.asarray(valid)


def test_g16_da_gradient_with_padding(rng: np.random.Generator) -> None:
    """jax.grad(cost) w.r.t. the state works with observations padded to a fixed N;
    J and ∇ₓJ are unchanged by the pad length and finite with NaN-filled padding."""
    background = GeoArray(jnp.zeros((24, 32)), jnp.asarray(T0), ("y", "x"), CRS)
    sigma = 0.1
    state_rng = np.random.default_rng(1)
    x0 = jnp.asarray(state_rng.normal(size=(24, 32)))

    def cost(x: jax.Array, xy: jax.Array, y: jax.Array, valid: jax.Array) -> jax.Array:
        # J(x) = ½ ∑ₙ mₙ (yₙ − H(x)ₙ)² / σ²
        g = background.with_data(x)
        hx = sample(g, xy, valid)
        r = jnp.where(valid, y - hx, 0.0)
        return 0.5 * jnp.sum(r**2) / sigma**2

    obs_rng = np.random.default_rng(2)
    xy64, y64, v64 = _obs(obs_rng, 40, 64, background)
    xy128 = jnp.concatenate([xy64, jnp.full((64, 2), jnp.nan)])
    y128 = jnp.concatenate([y64, jnp.full(64, jnp.nan)])
    v128 = jnp.concatenate([v64, jnp.zeros(64, bool)])
    vg = jax.jit(jax.value_and_grad(cost))
    j1, g1 = vg(x0, xy64, y64, v64)
    j2, g2 = vg(x0, xy128, y128, v128)
    assert np.isfinite(j1) and bool(jnp.all(jnp.isfinite(g1)))
    assert bool(jnp.all(jnp.isfinite(g2)))
    np.testing.assert_allclose(j1, j2, rtol=1e-14)
    np.testing.assert_allclose(g1, g2, rtol=1e-14, atol=1e-14)
    # the gradient is non-zero exactly where observations touch the grid
    assert int(jnp.sum(g1 != 0)) > 0


# -- GA-G17 --------------------------------------------------------------------


def test_g17_no_jax_on_io(tmp_path: Path, rng: np.random.Generator) -> None:
    """geocatalog, geocloud and geoproducts import and read with JAX not installed.

    JAX is hidden with an import hook that raises ImportError, as a missing
    package does. (Setting sys.modules["jax"] = None is not the same: geoarray's
    ``jax_loaded`` checks the key, so it would see JAX as loaded.)

    The file is written by the prototype's to_xarray (geostack#77), so this also
    checks that geocloud's reader takes the decided encoding (GF2).
    """
    nc = tmp_path / "scene.nc"
    g = scene(rng, (1, 3, 4))
    to_xarray(isel(g, "band", 0)).rename("t2m").to_netcdf(nc)
    code = textwrap.dedent(
        f"""
        import importlib.abc, sys

        class NoJax(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "jax" or name.startswith(("jax.", "jaxlib")):
                    raise ImportError(f"No module named {{name!r}}")
                return None

        sys.meta_path.insert(0, NoJax())
        import geocatalog, geocloud, geoproducts
        from geocloud.hdf import read_netcdf
        g = read_netcdf({str(nc)!r}, "t2m")
        assert g.shape == (3, 4), g.shape
        assert tuple(g.transform.tolist()) == {tuple(T0)!r}, g.transform
        assert g.crs.to_epsg() == 32613
        assert not any(m == "jax" or m.startswith("jax.") for m in sys.modules)
        print("ok")
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("ok")


# -- GA-G18 --------------------------------------------------------------------


def _three(g: GeoArray) -> dict[str, object]:
    return {
        "numpy": g,
        "jax": g.to_jax(),
        "xarray-dask": to_xarray(g).chunk({"x": 8}),
    }


def _values(x: object) -> np.ndarray:
    if isinstance(x, xr.DataArray):
        assert not isinstance(x.data, np.ndarray), "the xarray path computed eagerly"
        return np.asarray(x.compute().data)
    return np.asarray(x.data)


@pytest.mark.parametrize(
    "fn", ["combine", "stack", "reduce", "map_over", "isel", "kernelop"]
)
def test_g18_shared_functions_agree(rng: np.random.Generator, fn: str) -> None:
    """Each shared function gives identical results on NumPy, JAX and dask-backed xarray
    (reductions and the matched filter to 1e-12, everything else exactly)."""
    g = scene(rng)
    t = np.array([0.1, -0.2, 0.3, 0.05])
    score = KernelOp(
        matched_filter_score,
        KernelSpec("(r,c),(c),(c,d)->(r)", vectorize=True),
        core={"r": "y", "c": "band", "d": "band2"},
        t=t,
    )
    results = {}
    for name, x in _three(g).items():
        if fn == "combine":
            r = combine(lambda a, b: a * 2.0 - b, x, x)
        elif fn == "stack":
            r = stack([x, x], dim="time")
        elif fn == "reduce":
            r = reduce(x, "mean", "band")
        elif fn == "map_over":
            # map over bands: each band item is a (y, x) map
            r = map_over(lambda b: combine(lambda a: a * a, b), x, "band")
        elif fn == "isel":
            r = isel(x, "band", 2)
        else:
            mu, sig = STATS(x)
            r = score(x, mu, sig)
        results[name] = (r, _values(r))
    ref_obj, ref = results["numpy"]
    for name, (obj, vals) in results.items():
        dims = tuple(obj.dims)
        assert dims == tuple(ref_obj.dims), (name, dims)
        if fn in ("reduce", "kernelop"):
            np.testing.assert_allclose(vals, ref, rtol=1e-12, atol=1e-14, err_msg=name)
        else:
            np.testing.assert_array_equal(vals, ref, err_msg=name)


# -- GA-G19 --------------------------------------------------------------------


def _offset(g: GeoArray, dx: float) -> GeoArray:
    t = np.asarray(g.transform).copy()
    t[2] += dx
    return GeoArray(np.asarray(g.data), t, g.dims, g.crs, labels=g.labels)


@pytest.mark.parametrize("backend", ["numpy", "jax-eager", "jax-jit", "xarray-dask"])
@pytest.mark.parametrize("op", ["combine", "stack", "kernelop"])
def test_g19_no_silent_alignment(
    rng: np.random.Generator, backend: str, op: str
) -> None:
    """combine, stack and a KernelOp raise on a 1e-7 m offset on every backend."""
    a = scene(rng)
    b = _offset(a, 1e-7)
    assert np.asarray(b.transform)[2] != np.asarray(a.transform)[2]

    def call(x: object, y: object) -> object:
        if op == "combine":
            return combine(lambda p, q: p + q, x, y)
        if op == "stack":
            return stack([x, y], dim="time")
        return ND(x, y)

    if backend == "numpy":
        with pytest.raises(GridMismatchError):
            call(a, b)
    elif backend == "jax-eager":
        with pytest.raises(GridMismatchError):
            call(a.to_jax(), b.to_jax())
    elif backend == "jax-jit":
        # under jit the transforms are tracers; the check runs on device
        f = jax.jit(lambda x, y: call(x, y).data)
        with pytest.raises(Exception, match="GridMismatchError"):
            jax.block_until_ready(f(a.to_jax(), b.to_jax()))
        # and passes when the grids are equal
        jax.block_until_ready(f(a.to_jax(), a.to_jax()))
    else:
        xa, xb = to_xarray(a).chunk({"x": 8}), to_xarray(b).chunk({"x": 8})
        with pytest.raises(GridMismatchError):
            call(xa, xb)


def test_g19_dataarray_without_adapter_names_the_extra(
    rng: np.random.Generator,
) -> None:
    """GA-Q5 (geostack#90): a kernel operator given a DataArray with no xarray adapter
    raises and names pipekit-array[xarray]."""
    da = to_xarray(scene(rng))
    unregister_adapter(xr.DataArray)
    try:
        with pytest.raises(BackendError, match=r"pipekit-array\[xarray\]"):
            ND(da, da)
    finally:
        register_xarray()


def test_g19_mixed_backends_refuse(rng: np.random.Generator) -> None:
    """Mixed NumPy and JAX carriers refuse and name to_jax() (GA-C1, 21 Oct)."""
    g = scene(rng)
    with pytest.raises(BackendError, match="to_jax"):
        ND(g, g.to_jax())


# -- GA-G20 --------------------------------------------------------------------


def test_g20_round_trip_random_windows(rng: np.random.Generator) -> None:
    """from_xarray(to_xarray(g)) returns the same transform, dims and labels for
    random windows, bit for bit."""
    g = scene(rng, (4, 64, 64))
    t = np.asarray(g.transform)
    for _ in range(200):
        i, j = rng.integers(0, 48, 2)
        h, w = rng.integers(1, 16, 2)
        tw = np.array([t[0], t[1], t[2] + t[0] * j, t[3], t[4], t[5] + t[4] * i])
        win = GeoArray(
            np.asarray(g.data)[:, i : i + h, j : j + w],
            tw,
            g.dims,
            g.crs,
            labels=g.labels,
        )
        back = from_xarray(to_xarray(win))
        np.testing.assert_array_equal(np.asarray(back.transform), tw)
        assert back.dims == win.dims
        assert back.labels == win.labels
        assert back.crs == win.crs
        np.testing.assert_array_equal(back.data, win.data)


def test_g20_round_trip_rotated_grid(rng: np.random.Generator) -> None:
    """A rotated grid is written with GeoTransform only and comes back exactly."""
    th = np.deg2rad(12.5)
    tr = np.array(
        [
            20 * np.cos(th),
            -20 * np.sin(th),
            600000.0,
            20 * np.sin(th),
            -20 * np.cos(th),
            3550000.0,
        ]
    )
    g = GeoArray(rng.random((3, 10, 7)), tr, ("band", "y", "x"), CRS)
    da = to_xarray(g)
    assert "x" not in da.coords and "y" not in da.coords
    back = from_xarray(da)
    np.testing.assert_array_equal(np.asarray(back.transform), tr)


def test_g20_regression_463_centres_only() -> None:
    """geotoolz#463: a CF grid with only x/y centres (no GeoTransform) gets the edge
    origin x[0] − dx/2, y[0] − dy/2, not the identity."""
    x = 500_000 + 10 * (np.arange(4) + 0.5)
    y = 4_200_000 - 10 * (np.arange(3) + 0.5)
    da = xr.DataArray(np.zeros((3, 4)), dims=("y", "x"), coords={"x": x, "y": y})
    back = from_xarray(da)
    np.testing.assert_array_equal(
        np.asarray(back.transform), [10.0, 0.0, 500_000.0, 0.0, -10.0, 4_200_000.0]
    )


def test_g20_regression_464_edges_are_not_centres(rng: np.random.Generator) -> None:
    """geotoolz#464: x/y that are pixel edges while GeoTransform says centres differ by
    half a pixel; the reader raises instead of shifting the grid."""
    g = scene(rng, (1, 3, 4))
    da = to_xarray(g)
    edges = da.coords["x"].values - 10.0  # half a 20 m pixel
    with pytest.raises(ValueError, match="disagree"):
        from_xarray(da.assign_coords(x=edges))


def test_g20_rioxarray_reads_the_encoding(rng: np.random.Generator) -> None:
    """GF2: rioxarray reads the hand-written spatial_ref: same transform and CRS."""
    rioxarray = pytest.importorskip("rioxarray")
    assert rioxarray
    g = scene(rng)
    da = to_xarray(g)
    t = np.asarray(g.transform)
    got = da.rio.transform()
    np.testing.assert_allclose(
        [got.a, got.b, got.c, got.d, got.e, got.f], t, rtol=0, atol=1e-9
    )
    assert da.rio.crs.to_epsg() == 32613


# -- GA-G21 --------------------------------------------------------------------

#: Budget set before measuring (#516): the prototype's eager dispatch may cost
#: at most today's carrier plus 25 µs per op per patch.
BUDGET_US_PER_OP = 25.0


def test_g21_patch_overhead(rng: np.random.Generator) -> None:
    """10k patches through a 5-op chain cost no more than today's carrier plus the
    budget; reported, with the batched (one call for all patches) cost beside it."""
    n, ops = 10_000, 5
    data = rng.random((n, 2, 16, 16))
    t = np.array(T0)

    def today_chain(d: np.ndarray) -> float:
        g = geoarray.GeoArray(d, T0, CRS)
        m = geoarray.combine(
            lambda a, b: normalized_difference(a, b, eps=1e-10), g[1], g[0]
        )
        m = m.map(lambda a: a * 2.0)
        m = m.map(lambda a: np.clip(a, -1.0, 1.0))
        gy, gx = (
            m.with_data(v) for v in gradient(m.data, spacing=np.array([t[4], t[0]]))
        )
        m = geoarray.combine(np.hypot, gy, gx)
        return float(np.mean(m.data))

    scale = KernelOp(lambda a: a * 2.0, "()->()", core={})
    # written on array methods, so the same lambdas run on NumPy and JAX
    clip = KernelOp(lambda a: a.clip(-1.0, 1.0), "()->()", core={})
    hyp = KernelOp(lambda a, b: (a * a + b * b) ** 0.5, "(),()->()", core={})

    def proto_chain(d: np.ndarray) -> float:
        g = GeoArray(d, t, ("band", "y", "x"), CRS)
        m = ND(isel(g, "band", 1), isel(g, "band", 0))
        m = clip(scale(m))
        gy, gx = GRAD(m)
        m = hyp(gy, gx)
        return float(np.mean(m.data))

    def bench(f) -> float:
        f(data[0])
        t0 = time.perf_counter()
        for k in range(n):
            f(data[k])
        return (time.perf_counter() - t0) / n * 1e6

    today_us, proto_us = bench(today_chain), bench(proto_chain)

    # batched: all patches as one carrier with a (n, 6) transform, through
    # map_over under jit (one vmap); geometry= needs one transform per item (F4)
    def chain_one(g: GeoArray) -> GeoArray:
        m = ND(isel(g, "band", 1), isel(g, "band", 0))
        m = clip(scale(m))
        gy, gx = GRAD(m)
        return hyp(gy, gx)

    rows = np.broadcast_to(t, (n, 6)).copy()
    rows[:, 2] += 320.0 * np.arange(n)  # every patch its own origin
    batched = GeoArray(data, rows, ("patch", "band", "y", "x"), CRS).to_jax()
    run = jax.jit(lambda p: map_over(chain_one, p, "patch").data.mean())
    run(batched).block_until_ready()  # compile
    t1 = time.perf_counter()
    run(batched).block_until_ready()
    batched_us = (time.perf_counter() - t1) / n * 1e6
    with pytest.raises(ValueError, match="map_over"):
        GRAD(GeoArray(data[:, 0], rows, ("patch", "y", "x"), CRS))
    print(
        f"\nGA-G21 per patch: today {today_us:.1f} µs, prototype {proto_us:.1f} µs, "
        f"batched {batched_us:.2f} µs; budget today + {BUDGET_US_PER_OP * ops:.0f} µs"
    )
    assert proto_us <= today_us + BUDGET_US_PER_OP * ops


# -- PK-K1 / PK-K2 ---------------------------------------------------------------


def test_pk_k2_dask_stays_lazy(rng: np.random.Generator) -> None:
    """A KernelOp on a dask-backed cube builds a graph and computes nothing."""
    import dask

    calls = 0

    def counting(a, b, eps):
        nonlocal calls
        calls += 1
        return normalized_difference(a, b, eps)

    op = KernelOp(counting, "(),()->()", core={}, eps=1e-10)
    da = to_xarray(scene(rng)).chunk({"x": 8})
    with dask.config.set(scheduler="synchronous"):
        out = op(da.isel(band=3), da.isel(band=2))
        meta_calls = calls
        out.compute()
    assert meta_calls <= 1  # dask may call once on a 0-size meta array
    assert calls > meta_calls


def test_dimarray_outputs_have_no_georeference(rng: np.random.Generator) -> None:
    """Per-column statistics consume y: a DimArray['band x'], not a GeoArray (PK-P7)."""
    mu, sig = STATS(scene(rng))
    assert isinstance(mu, DimArray) and mu.dims == ("band", "x")
    assert isinstance(sig, DimArray) and sig.dims == ("band", "x", "band2")
