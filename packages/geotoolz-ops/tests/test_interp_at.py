"""geoops.geom.interp_at and InterpAt: H(x), a grid read at points.

Moved from geoarray with the function (GA-B6, #631); GA-G16 is the
data-assimilation gate: a masked cost and its gradient are unchanged by
padding the observations.
"""

from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from geoarray import GeoArray, GeoPoints, pad_points
from geoarray.coords import TimeAxis, VariableAxis
from geoarray.types import GeoTimeSeries, matches

import geoops as gz
from geoops.geom import interp_at


T = (20.0, 0.0, 600000.0, 0.0, -20.0, 3550000.0)
CRS = "EPSG:32613"


@pytest.fixture
def x64() -> Iterator[ModuleType]:
    """JAX with ``jax_enable_x64`` on for one test, restored afterwards."""
    jax = pytest.importorskip("jax")
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield jax
    jax.config.update("jax_enable_x64", previous)


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


# -- review fixes (Codex, #550) -------------------------------------------------------


def test_non_finite_positions_are_invalid() -> None:
    # x = 0 is inside this grid, so a NaN read "at the origin" would look valid
    g = GeoArray(np.arange(16.0).reshape(4, 4), (1, 0, 0, 0, -1, 4), None, dims="y x")
    xy = np.array([[np.nan, 1.5], [0.5, 3.5]])
    out = interp_at(g, GeoPoints(np.zeros(2), xy=xy, dims="obs", crs=None))
    assert out.valid.tolist() == [False, True]
    assert np.isnan(out.data[0])


@pytest.mark.parametrize("method", ["nearest", "bilinear"])
def test_nodata_pixels_are_never_observations(method: str) -> None:
    data = np.array(_ramp().data)
    data[2, 2] = -9999.0
    g = GeoArray(data, T, CRS, fill_value=-9999.0, dims="y x")
    # (2, 2) reads the nodata pixel itself; (2.4, 1.0) blends it in with
    # weight 0 along columns but not rows; (4, 4) is far away
    out = interp_at(g, _at(g, [(2, 2), (4, 4)]), method=method)
    assert out.valid.tolist() == [False, True]
    assert np.isnan(out.data[0])
    assert out.data[1] == 44.0


def test_a_zero_weight_nodata_neighbour_is_ignored() -> None:
    data = np.array(_ramp().data)
    data[3, 3] = np.nan  # the (1, 1) corner of the cell at (2, 2)
    g = GeoArray(data, T, CRS, fill_value=np.nan, dims="y x")
    out = interp_at(g, _at(g, [(2.0, 2.5)]))  # w_r = 0: row 3 has no weight
    assert out.valid.tolist() == [True]
    assert out.data[0] == pytest.approx(22.5)


def test_jax_points_read_a_numpy_grid(x64: ModuleType) -> None:
    g = _ramp()
    pts = _at(g, [(1.5, 1.5)]).to_jax()
    with pytest.warns(DeprecationWarning, match="interp_at"):
        out = interp_at(g, pts)
    assert type(out.data).__module__.startswith("jax")

    def value(xy: Any) -> Any:
        moved = GeoPoints(pts.data, xy=xy, dims="obs", crs=CRS)
        return interp_at(g, moved).data.sum()

    grad = x64.jit(x64.grad(value))(pts.xy)
    # v = 10·row + col; x moves col by 1/20, y moves row by −1/20
    np.testing.assert_allclose(np.asarray(grad), [[1 / 20, -10 / 20]])


# -- InterpAt: the operator ------------------------------------------------------------


@pytest.mark.parametrize("method", ["bilinear", "nearest"])
def test_operator_matches_the_function(method: str) -> None:
    g = _ramp()
    pts = _at(g, [(1.5, 1.5), (2.2, 3.1), (-3, 0)])
    got = gz.InterpAt(method=method)(g, pts)
    want = interp_at(g, pts, method)
    np.testing.assert_array_equal(got.data, want.data)
    assert got.valid.tolist() == want.valid.tolist()


def test_operator_reads_its_own_frame() -> None:
    frames = GeoArray(
        np.stack([_ramp().data + k for k in range(3)]), T, CRS, dims="time y x"
    )
    pts = _at(frames, [(1, 1), (1, 1), (1, 1)])
    out = gz.InterpAt()(frames, pts, np.array([2, 0, 1]))
    assert out.data.tolist() == pytest.approx([13.0, 11.0, 12.0])


def test_operator_runs_in_a_graph() -> None:
    from pipekit import Graph, Input

    g = _ramp()
    pts = _at(g, [(2.2, 3.1)])
    grid_in, points_in = Input("grid"), Input("points")
    graph = Graph(
        inputs={"grid": grid_in, "points": points_in},
        outputs={"h": gz.InterpAt()(grid_in, points_in)},
    )
    got = graph(grid=g, points=pts)["h"]
    np.testing.assert_array_equal(got.data, interp_at(g, pts).data)


def test_operator_config_round_trips() -> None:
    from pipekit import Operator

    op = gz.InterpAt(method="nearest", along="level")
    assert op.get_config() == {"method": "nearest", "along": "level"}
    again = Operator.from_state(op.state)
    assert isinstance(again, gz.InterpAt) and again.get_config() == op.get_config()


def test_operator_refuses_an_unknown_method() -> None:
    with pytest.raises(ValueError, match=r"InterpAt\.method"):
        gz.InterpAt(method="cubic")  # ty: ignore[invalid-argument-type]


def test_operator_on_jax_matches_numpy_under_jit(x64: ModuleType) -> None:
    g = _ramp()
    pts = _at(g, [(0.5, 0.5), (2.2, 3.1), (4.9, 1.0)])
    want = gz.InterpAt()(g, pts)
    op = gz.InterpAt()
    got = x64.jit(lambda grid, p: op(grid, p).data)(g.to_jax(), pts.to_jax())
    np.testing.assert_allclose(np.asarray(got), want.data, rtol=1e-14)


def test_operator_refuses_inputs_that_are_not_a_grid_and_points() -> None:
    g = _ramp()
    pts = _at(g, [(1.5, 1.5)])
    with pytest.raises(TypeError, match="InterpAt"):
        gz.InterpAt()(np.asarray(g.data), pts)
    with pytest.raises(TypeError, match="GeoPoints"):
        gz.InterpAt()(g, np.asarray(pts.xy))


def test_operator_refuses_a_geotensor_as_either_input() -> None:
    pytest.importorskip("georeader")
    from geoarray.interop.georeader import to_georeader

    g = _ramp()
    pts = _at(g, [(1.5, 1.5)])
    with pytest.raises(TypeError, match="from_georeader"):
        gz.InterpAt()(to_georeader(g), pts)
    with pytest.raises(TypeError, match="from_georeader"):
        gz.InterpAt()(g, to_georeader(g))


# -- against scipy (#635) ---------------------------------------------------------
#
# scipy.ndimage.map_coordinates(order=1, mode="nearest") is bilinear
# interpolation with the edge value extended beyond the grid: the same maths
# as sample_bilinear everywhere, and as interp_at inside the grid's extent
# (interp_at is NaN and invalid outside it). order=0 is nearest, away from
# exact half-pixel ties, where the two round differently.


def _field(shape: tuple[int, ...] = (7, 9), seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=shape)


def _positions(h: int, w: int, n: int = 200, pad: float = 2.0, seed: int = 1):
    rng = np.random.default_rng(seed)
    return rng.uniform(-pad, h - 1 + pad, n), rng.uniform(-pad, w - 1 + pad, n)


@pytest.mark.parametrize("shape", [(7, 9), (3, 7, 9), (1, 9), (7, 1)])
def test_sample_bilinear_matches_scipy(shape: tuple[int, ...]) -> None:
    from scipy.ndimage import map_coordinates

    data = _field(shape)
    rows, cols = _positions(*shape[-2:])
    got = gz.geom.sample_bilinear(data, rows, cols)
    want = np.stack(
        [
            map_coordinates(band, [rows, cols], order=1, mode="nearest")
            for band in data.reshape(-1, *shape[-2:])
        ]
    ).reshape(*shape[:-2], rows.size)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_sample_nearest_matches_scipy() -> None:
    from scipy.ndimage import map_coordinates

    data = _field()
    rows, cols = _positions(7, 9)
    got = gz.geom.sample_nearest(data, rows, cols)
    want = map_coordinates(data, [rows, cols], order=0, mode="nearest")
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("method", ["bilinear", "nearest"])
def test_interp_at_matches_scipy_inside_and_is_nan_outside(method: str) -> None:
    from scipy.ndimage import map_coordinates

    data = _field((6, 5))
    g = GeoArray(data, T, CRS, dims="y x")
    rows, cols = _positions(6, 5, pad=1.0)
    out = interp_at(g, _at(g, list(zip(rows, cols, strict=True))), method)
    inside = (rows >= -0.5) & (rows <= 5.5) & (cols >= -0.5) & (cols <= 4.5)
    order = 1 if method == "bilinear" else 0
    want = map_coordinates(data, [rows, cols], order=order, mode="nearest")
    # positions travel to UTM metres (|x| ≈ 6e5, 20 m pixels) and back through
    # to_pixel: δr ≈ ε·|x|/res ≈ 2.2e-16 · 6e5 / 20 ≈ 7e-12 px, and the field's
    # slope is O(1) per pixel, so values agree to ~1e-11; 1e-9 bounds it
    np.testing.assert_allclose(out.data[inside], want[inside], rtol=0, atol=1e-9)
    assert np.isnan(out.data[~inside]).all()
    assert out.valid.tolist() == inside.tolist()


def test_stencil_weights_are_a_partition_of_unity() -> None:
    rows, cols = _positions(7, 9)
    ri, ci, w = gz.geom.bilinear_stencil(rows, cols, 7, 9)
    assert ri.shape == ci.shape == w.shape == (rows.size, 4)
    assert (w >= 0).all()
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-14)
    assert ri.min() >= 0 and ri.max() <= 6 and ci.min() >= 0 and ci.max() <= 8


def test_stencil_gives_the_adjoint() -> None:
    """Hᵀ scatters with the stencil's weights: ⟨Hx, y⟩ = ⟨x, Hᵀy⟩."""
    x = _field((7, 9))
    rows, cols = _positions(7, 9, n=50)
    y = np.random.default_rng(2).normal(size=rows.size)
    ri, ci, w = gz.geom.bilinear_stencil(rows, cols, 7, 9)
    # Hᵀy: each yₙ added to its four pixels with weight wₙₖ
    adjoint = np.zeros_like(x)
    np.add.at(adjoint, (ri, ci), w * y[:, None])
    lhs = gz.geom.sample_bilinear(x, rows, cols) @ y
    assert lhs == pytest.approx(float(np.sum(x * adjoint)), rel=1e-12)


def test_jax_sampling_matches_jax_scipy(x64: ModuleType) -> None:
    from jax.scipy.ndimage import map_coordinates

    jnp = x64.numpy
    data = jnp.asarray(_field())
    rows, cols = (jnp.asarray(a) for a in _positions(7, 9))
    got = x64.jit(gz.geom.sample_bilinear)(data, rows, cols)
    want = map_coordinates(data, [rows, cols], order=1, mode="nearest")
    np.testing.assert_allclose(np.asarray(got), np.asarray(want), rtol=1e-12)
    near = x64.jit(gz.geom.sample_nearest)(data, rows, cols)
    want0 = map_coordinates(data, [rows, cols], order=0, mode="nearest")
    np.testing.assert_array_equal(np.asarray(near), np.asarray(want0))
