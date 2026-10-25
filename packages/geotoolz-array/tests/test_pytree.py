"""`GeoArray` as a JAX pytree: gates GA1 (round trip), GA2 (no retrace on a new
transform) and GA4 (the x64 guard)."""

from __future__ import annotations

import math
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from conftest import CRS, TRANSFORM

from geoarray import GeoArray, GeoArrayTraceError, X64Error


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _scene(transform: tuple[float, ...] = TRANSFORM, **kwargs: object) -> GeoArray:
    rng = np.random.default_rng(1)
    data = rng.random((2, 6, 6)).astype(np.float32)
    return GeoArray(data, transform, kwargs.pop("crs", CRS), **kwargs)  # type: ignore[arg-type]


class TestRoundTrip:
    """GA1: unflatten(flatten(g)) is g, and jit(identity) is bit for bit."""

    def test_flatten_unflatten(self, x64: ModuleType) -> None:
        g = _scene(fill_value=np.nan, attrs={"band_names": ["a", "b"]})
        leaves, treedef = jax.tree_util.tree_flatten(g)
        assert len(leaves) == 2
        assert leaves[0] is g.data
        assert leaves[1] is g.transform
        assert jax.tree_util.tree_unflatten(treedef, leaves).identical(g)

    def test_key_paths_name_the_fields(self, x64: ModuleType) -> None:
        paths = [
            jax.tree_util.keystr(path)
            for path, _ in jax.tree_util.tree_flatten_with_path(_scene())[0]
        ]
        assert paths == [".data", ".transform"]

    def test_jit_identity_is_bit_for_bit(self, x64: ModuleType) -> None:
        g = _scene(fill_value=np.nan, attrs={"band_names": ["a", "b"]})
        out = jax.jit(lambda carrier: carrier)(g)
        assert out.backend == "jax"
        assert out.transform.dtype == jnp.float64
        assert out.identical(g)
        np.testing.assert_array_equal(np.asarray(out.transform), g.transform)

    def test_unflatten_accepts_placeholders(self, x64: ModuleType) -> None:
        shapes = jax.eval_shape(lambda c: c, _scene())
        assert shapes.data.shape == (2, 6, 6)
        assert shapes.transform.shape == (6,)
        assert jax.tree_util.tree_map(lambda _: None, _scene()).data is None

    def test_jax_carrier_round_trips_through_numpy(self, x64: ModuleType) -> None:
        g = _scene(fill_value=0.0).to_jax()
        back = g.to_numpy()
        assert back.backend == "numpy"
        assert back.identical(g)
        assert g.to_jax() is g


class TestNoRetrace:
    """GA2: new transforms reuse the compiled function; static fields do not."""

    def test_transform_changes_never_retrace(self, x64: ModuleType) -> None:
        traces = 0

        def f(g: GeoArray) -> jax.Array:
            nonlocal traces
            traces += 1
            return g.data.sum() + g.transform[2]

        jitted = jax.jit(f)
        rng = np.random.default_rng(2)
        for _ in range(100):
            origin = rng.uniform(-1e6, 1e6, size=2)
            jitted(_scene((20.0, 0.0, origin[0], 0.0, -20.0, origin[1])))
        assert traces == 1

        jitted(_scene(crs="EPSG:32614"))
        assert traces == 2
        rng_data = rng.random((2, 6, 4)).astype(np.float32)
        jitted(GeoArray(rng_data, TRANSFORM, CRS))
        assert traces == 3

    def test_nan_fills_from_different_sources_share_a_compile(
        self, x64: ModuleType
    ) -> None:
        traces = 0

        def f(g: GeoArray) -> jax.Array:
            nonlocal traces
            traces += 1
            return g.data.sum()

        jitted = jax.jit(f)
        jitted(_scene(fill_value=float("nan"), attrs={"scale": float("nan")}))
        jitted(_scene(fill_value=np.nan, attrs={"scale": np.float32("nan")}))
        jitted(_scene(fill_value=np.float64("nan"), attrs={"scale": math.nan}))
        assert traces == 1

    def test_new_attrs_retrace(self, x64: ModuleType) -> None:
        traces = 0

        def f(g: GeoArray) -> jax.Array:
            nonlocal traces
            traces += 1
            return g.data.sum()

        jitted = jax.jit(f)
        jitted(_scene(attrs={"band_names": ["a", "b"]}))
        jitted(_scene(attrs={"band_names": ["a", "b"]}))
        jitted(_scene(attrs={"band_names": ["c", "d"]}))
        assert traces == 2


class TestTransforms:
    def test_grad_reaches_the_transform(self, x64: ModuleType) -> None:
        g = _scene().to_jax()

        def loss(carrier: GeoArray) -> jax.Array:
            return carrier.transform[2] * carrier.data.sum()

        grads = jax.grad(loss)(g)
        assert isinstance(grads, GeoArray)
        assert float(grads.transform[2]) == pytest.approx(float(g.data.sum()))
        assert float(grads.transform[0]) == 0.0

    def test_vmap_over_stacked_carriers(self, x64: ModuleType) -> None:
        g = _scene()
        stacked = GeoArray(
            jnp.stack([g.data, g.data * 2]),
            jnp.stack([g.transform, g.transform + 10.0]),
            CRS,
        )
        origins = jax.vmap(lambda c: c.transform[2])(stacked)
        np.testing.assert_allclose(np.asarray(origins), [600000.0, 600010.0])

    def test_rewrap_inside_jit(self, x64: ModuleType) -> None:
        out = jax.jit(lambda c: c.with_data(c.data * 2).with_attrs(scaled=True))(
            _scene()
        )
        assert out.attrs["scaled"] is True
        np.testing.assert_allclose(np.asarray(out.data), _scene().data * 2)

    def test_host_reads_raise_a_clear_error_under_trace(self, x64: ModuleType) -> None:
        with pytest.raises(GeoArrayTraceError, match="grid needs a concrete transform"):
            jax.jit(lambda c: c.grid.res[0] * c.data)(_scene())
        with pytest.raises(GeoArrayTraceError, match="to_numpy needs concrete data"):
            jax.jit(lambda c: c.to_numpy().data)(_scene())
        with pytest.raises(GeoArrayTraceError, match="NumPy data cannot take a traced"):
            jax.jit(lambda t: GeoArray(np.zeros((2, 2)), t, CRS).transform)(
                jnp.asarray(TRANSFORM)
            )

    def test_repr_under_trace(self, x64: ModuleType) -> None:
        seen: list[str] = []
        jax.jit(lambda c: seen.append(repr(c)) or c.data)(_scene())
        assert "traced" in seen[0]
        assert seen[0].startswith("GeoArray(jax float32 (2, 6, 6)")


class TestX64Guard:
    """GA4: no silent float32 transform."""

    def test_to_jax_needs_x64(self, no_x64: ModuleType) -> None:
        with pytest.raises(X64Error, match="jax_enable_x64"):
            _scene().to_jax()

    def test_jax_data_needs_x64(self, no_x64: ModuleType) -> None:
        with pytest.raises(X64Error, match="jax_enable_x64"):
            GeoArray(jnp.zeros((2, 2)), TRANSFORM, CRS)

    def test_float32_jax_transform_is_rejected(self, x64: ModuleType) -> None:
        with pytest.raises(X64Error, match="must be float64"):
            GeoArray(jnp.zeros((2, 2)), jnp.asarray(TRANSFORM, dtype=jnp.float32), CRS)
        with pytest.raises(ValueError, match=r"shape \(6,\)"):
            GeoArray(jnp.zeros((2, 2)), jnp.zeros(5), CRS)

    def test_with_x64_the_transform_is_float64(self, x64: ModuleType) -> None:
        g = GeoArray(jnp.zeros((2, 2), dtype=jnp.float32), TRANSFORM, CRS)
        assert g.transform.dtype == jnp.float64
        assert g.dtype == jnp.float32
        assert (
            GeoArray(jnp.zeros((2, 2)), np.array(TRANSFORM), CRS).transform.dtype
            == jnp.float64
        )
        assert g.xp is jnp
        assert g.backend == "jax"
        assert g.grid.transform == TRANSFORM

    def test_jax_transform_with_numpy_data_is_pulled_to_the_host(
        self, x64: ModuleType
    ) -> None:
        g = GeoArray(np.zeros((2, 2)), jnp.asarray(TRANSFORM), CRS)
        assert isinstance(g.transform, np.ndarray)
        assert tuple(g.transform) == TRANSFORM


# -- review fixes (Codex, #547) -------------------------------------------------------


def _dated(xp: Any = np) -> GeoArray:
    from geoarray.coords import TimeAxis

    return GeoArray(
        xp.zeros((2, 2, 2)),
        (20.0, 0.0, 0.0, 0.0, -20.0, 0.0),
        32613,
        dims="time y x",
        coords=(
            TimeAxis(np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]")),
        ),
    )


def test_selected_stamps_stay_read_only() -> None:
    picked = _dated().coord("time").isel([0])
    with pytest.raises(ValueError, match="read-only"):
        picked.stamps[0] = np.datetime64("2030-01-01", "ns")


def test_concat_along_time_under_jit(x64: ModuleType) -> None:
    import geoarray

    g = _dated().to_jax()
    out = x64.jit(lambda c: geoarray.concat([c, c], "time"))(g)
    assert out.coord("time").size == 4


def test_to_jax_moves_stamps_of_a_carrier_built_on_jax(x64: ModuleType) -> None:
    g = _dated(x64.numpy)
    assert g.backend == "jax"
    moved = g.to_jax()
    assert type(moved.coord("time").stamps).__module__.startswith("jax")


def test_importing_geoarray_does_not_import_jax_and_jax_registers_later() -> None:
    """The pytrees are registered when JAX is imported (pipekit.on_import, #528)."""
    import subprocess
    import sys

    code = (
        "import sys, numpy as np, geoarray\n"
        "assert 'jax' not in sys.modules, 'geoarray imported jax'\n"
        "import jax\n"
        "jax.config.update('jax_enable_x64', True)\n"
        "t = (20.0, 0.0, 0.0, 0.0, -20.0, 0.0)\n"
        "g = geoarray.GeoArray(np.zeros((2, 3, 3)), t, 32613, dims='band y x')\n"
        "lat, lon = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing='ij')\n"
        "s = geoarray.GeoSwath(np.zeros((3, 3)), lat, lon, dims='row col')\n"
        "p = geoarray.GeoPoints(np.zeros(2), xy=np.zeros((2, 2)), dims='obs')\n"
        "for c in (g, s, p):\n"
        "    assert len(jax.tree_util.tree_leaves(c)) >= 2, type(c).__name__\n"
        "print(jax.jit(lambda c: c)(g.to_jax()).dims)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "('band', 'y', 'x')"
