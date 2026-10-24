"""The pytree with stored dims, and the gates of #519 (GA-G1, G2, G4, G10)."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from types import ModuleType
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest

import geoarray
from geoarray import GeoArray, PixelWindow, X64Error
from geoarray.coords import SpectralAxis, TimeAxis


TRANSFORM = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"
BANDS = SpectralAxis(("B4", "B8"), (665.0, 842.0))


def _dates(start: str, n: int = 3) -> TimeAxis:
    return TimeAxis(np.datetime64(start, "D") + np.arange(n) * np.timedelta64(10, "D"))


def _series(xp: Any = np, start: str = "2026-06-01", **kw: Any) -> GeoArray:
    data = np.random.default_rng(0).random((3, 2, 6, 5))
    return GeoArray(
        xp.asarray(data),
        TRANSFORM,
        CRS,
        np.nan,
        dims="time band y x",
        coords=(_dates(start), BANDS),
        **kw,
    )


def test_ga_g1_round_trip(x64: ModuleType) -> None:
    jax = x64
    g = _series().to_jax()
    leaves, treedef = jax.tree_util.tree_flatten(g)
    assert len(leaves) == 3  # data, transform, time stamps
    assert str(leaves[2].dtype) == "int64"
    assert jax.tree_util.tree_unflatten(treedef, leaves).identical(g)
    out = jax.jit(lambda c: c)(g)
    assert out.identical(g)
    np.testing.assert_array_equal(np.asarray(out.transform), np.asarray(g.transform))
    paths = [
        jax.tree_util.keystr(p) for p, _ in jax.tree_util.tree_flatten_with_path(g)[0]
    ]
    assert paths == [".data", ".transform", ".time_stamps"]


def test_stamps_are_labels_on_the_host_and_a_leaf_on_jax(x64: ModuleType) -> None:
    g = _series()
    assert g.coord("time").stamps.dtype == np.dtype("datetime64[ns]")
    assert str(g.sel(time=g.labels("time")[1]).dims) == "('band', 'y', 'x')"
    j = g.to_jax()
    assert type(j.coord("time").stamps).__module__.startswith("jax")
    assert j.labels("time") == g.labels("time")
    back = j.to_numpy()
    assert back.coord("time").stamps.dtype == np.dtype("datetime64[ns]")
    assert back.identical(g)


def test_ga_g2_one_trace_across_scenes(x64: ModuleType) -> None:
    """100 random transforms, and two stamp sets of one size: one trace."""
    jax = x64
    traces = 0

    def step(c: GeoArray) -> Any:
        nonlocal traces
        traces += 1
        return pa.reduce(c, "mean", "time").data.sum()

    f = jax.jit(step)
    base = _series().to_jax()
    rng = np.random.default_rng(1)
    for _ in range(100):
        dx, dy = rng.uniform(-1e5, 1e5, 2)
        f(base.shift_origin((dx, dy)))
    f(_series(start="2027-01-01").to_jax())
    assert traces == 1
    f(base.replace(crs="EPSG:32614"))
    assert traces == 2
    f(base[..., :4, :4])
    assert traces == 3


def test_stamps_change_inside_jit(x64: ModuleType) -> None:
    jax = x64

    @jax.jit
    def first_stamp(c: GeoArray) -> Any:
        assert c.labels("time") is None  # traced: no labels inside jit
        return c.coord("time").stamps[0]

    a = first_stamp(_series().to_jax())
    b = first_stamp(_series(start="2030-01-01").to_jax())
    assert int(b) > int(a)
    assert np.datetime64(int(a), "ns") == np.datetime64("2026-06-01", "ns")


def test_ga_g4_to_jax_needs_x64(no_x64: ModuleType) -> None:
    with pytest.raises(X64Error, match="jax_enable_x64"):
        _series().to_jax()


def test_ga_g4_transform_is_float64(x64: ModuleType) -> None:
    assert str(_series().to_jax().transform.dtype) == "float64"


def _functions() -> list[tuple[str, Any]]:
    window = PixelWindow(col_off=-1, row_off=2, width=4, height=5)
    return [
        ("isel", lambda g: g.isel(time=[2, 0], x=slice(1, 4))),
        ("sel", lambda g: g.sel(band="B8")),
        ("reduce", lambda g: geoarray.reduce(g, "mean", "time")),
        ("combine", lambda g: geoarray.combine(lambda a, b: a - 2 * b, g, g)),
        ("stack", lambda g: geoarray.stack([g, g], "run")),
        (
            "concat",
            lambda g: geoarray.concat([g.sel(band=["B8"]), g.sel(band=["B4"])], "band"),
        ),
        ("crop", lambda g: geoarray.crop(g, (1, 1), (3, 3))),
        ("pad", lambda g: geoarray.pad(g, 1)),
        ("read_window", lambda g: geoarray.read_window(g, window)),
        (
            "valid_mask",
            lambda g: g.with_data(
                g.valid_mask(keep_time=True), dims="time y x", fill_value=False
            ),
        ),
    ]


@pytest.mark.parametrize("case", _functions(), ids=lambda c: c[0])
def test_ga_g10_numpy_and_jax_agree(case: tuple[str, Any], x64: ModuleType) -> None:
    _, fn = case
    on_numpy = fn(_series())
    on_jax = fn(_series().to_jax())
    assert on_jax.backend == "jax"
    assert on_jax.dims == on_numpy.dims
    back = on_jax.to_numpy()
    assert back.coords == on_numpy.coords
    # reductions may round differently on XLA: values to round-off, the rest exact
    np.testing.assert_allclose(back.data, on_numpy.data, rtol=1e-12)
    assert back.replace(data=on_numpy.data).identical(on_numpy)


def test_grad_reaches_data_and_transform(x64: ModuleType) -> None:
    jax = x64
    g = _series().to_jax()

    def loss(data: Any, dx: Any) -> Any:
        c = g.with_data(data).shift_origin((dx, 0.0))
        # J = Σ x · c′, so ∂J/∂x = c′ and ∂J/∂Δx = Σ x
        return (c.data * c.transform[2]).sum()

    d_data, d_dx = jax.grad(loss, argnums=(0, 1))(g.data, 0.0)
    np.testing.assert_allclose(np.asarray(d_data), TRANSFORM[2])
    np.testing.assert_allclose(float(d_dx), float(np.asarray(g.data).sum()))


def test_to_jax_without_jax_names_the_extra() -> None:
    code = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in ("jax", "jaxlib"):
                    raise ImportError(name)

        sys.meta_path.insert(0, Block())
        import geoarray, numpy as np
        g = geoarray.GeoArray(np.zeros((2, 2)), (1, 0, 0, 0, -1, 0), None, dims="y x")
        try:
            g.to_jax()
        except ImportError as exc:
            print(exc)
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0, out.stderr[-2000:]
    assert "geotoolz-array[jax]" in out.stdout
