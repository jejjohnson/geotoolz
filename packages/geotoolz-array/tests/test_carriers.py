"""`GeoSwath`, `GeoPoints` and their adapters (#520)."""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np
import pipekit_array as pa
import pytest
from pipekit import get_adapter
from pipekit_array import DimArray, KernelOp

import geoarray
from geoarray import (
    GeoPoints,
    GeoPointsAdapter,
    GeoSwath,
    GeoSwathAdapter,
    GridMismatchError,
    pad_points,
)
from geoarray.coords import (
    CurvilinearGrid,
    SpectralAxis,
    TimeAxis,
    VariableAxis,
    Vertical,
)


EMIT = SpectralAxis(("b1", "b2", "b3"), (450.0, 550.0, 650.0))
DEPTH = Vertical(kind="depth", units="m", positive="down")
ARGO = VariableAxis(("temp", "psal"), ("degC", "1"))


def _swath(xp: Any = np) -> GeoSwath:
    lat, lon = np.meshgrid(
        np.linspace(31.0, 32.0, 6), np.linspace(-104.0, -103.0, 5), indexing="ij"
    )
    data = np.random.default_rng(0).random((3, 6, 5))
    return GeoSwath(
        xp.asarray(data),
        lat,
        lon,
        dims="band row col",
        coords=(EMIT,),
        fill_value=np.nan,
    )


def _profile(n: int = 4, seed: int = 0, xp: Any = np) -> GeoPoints:
    rng = np.random.default_rng(seed)
    return GeoPoints(
        xp.asarray(rng.random((2, n))),
        xy=rng.uniform(-30, 30, (n, 2)),
        z=np.linspace(5.0, 200.0, n),
        vertical=DEPTH,
        time=np.datetime64("2026-06-01T00", "ns")
        + np.arange(n) * np.timedelta64(1, "h"),
        dims="variable obs",
        coords=(ARGO,),
        fill_value=np.nan,
    )


# -- GeoSwath -------------------------------------------------------------------------


def test_swath_isel_slices_lat_lon_with_the_data() -> None:
    s = _swath()
    part = s.isel(row=slice(1, 4), band=[2, 0])
    assert part.dims == ("band", "row", "col")
    assert part.lat.shape == (3, 5) and part.lon.shape == (3, 5)
    np.testing.assert_array_equal(part.lat, s.lat[1:4])
    assert part.labels("band") == ("b3", "b1")
    assert s.sel(band="b2").dims == ("row", "col")
    with pytest.raises(IndexError, match="slice"):
        s.isel(row=2)


def test_swath_validation() -> None:
    with pytest.raises(ValueError, match="row, col"):
        GeoSwath(
            np.zeros((2, 4, 5)), np.zeros((4, 4)), np.zeros((4, 4)), dims="band row col"
        )
    with pytest.raises(ValueError, match="end in"):
        GeoSwath(np.zeros((4, 5)), np.zeros((4, 5)), np.zeros((4, 5)), dims="y x")
    with pytest.raises(TypeError, match="dims="):
        GeoSwath(np.zeros((4, 5)), np.zeros((4, 5)), np.zeros((4, 5)))


def test_per_granule_geolocation_is_a_batch_dim() -> None:
    lat = np.zeros((2, 4, 5)) + np.arange(2)[:, None, None]
    s = GeoSwath(np.zeros((2, 3, 4, 5)), lat, lat, dims="time band row col")
    assert s.batch_dims == ("time",)
    one = s.isel(time=1)
    assert one.lat.shape == (4, 5) and float(one.lat[0, 0]) == 1.0


def test_curvilinear_grid_swath() -> None:
    lat, lon = np.meshgrid(np.arange(4.0), np.arange(5.0), indexing="ij")
    grid = CurvilinearGrid(lat, lon, "EPSG:4326")
    sst = GeoSwath.from_grid(np.zeros((12, 4, 5)), grid, dims="time row col")
    assert sst.crs == grid.crs and sst.lat.shape == (4, 5)


def test_swath_adapter_and_reductions() -> None:
    s = _swath()
    assert isinstance(get_adapter(s), GeoSwathAdapter)
    total = pa.reduce(s, "sum", "band")
    assert isinstance(total, GeoSwath) and total.dims == ("row", "col")
    per_column = KernelOp(lambda x: x.mean(-1), "(r)->()", core={"r": "row"})(s)
    assert isinstance(per_column, DimArray)
    assert per_column.dims == ("band", "col")
    profile = pa.isel(s, row=2)
    assert isinstance(profile, DimArray) and profile.dims == ("band", "col")


def test_swath_geolocation_must_match() -> None:
    s = _swath()
    moved = GeoSwath(s.data, s.lat + 1e-9, s.lon, dims=s.dims, coords=s.coords)
    with pytest.raises(GridMismatchError, match="lat"):
        geoarray.combine(np.add, s, moved)


def test_swath_round_trips(x64: ModuleType) -> None:
    jax = x64
    s = _swath().to_jax()
    assert s.backend == "jax" and str(s.lat.dtype) == "float64"
    leaves, treedef = jax.tree_util.tree_flatten(s)
    assert len(leaves) == 3
    assert jax.tree_util.tree_unflatten(treedef, leaves).identical(s)
    assert jax.jit(lambda c: c)(s).identical(s)
    assert s.to_numpy().identical(_swath())


# -- GeoPoints ------------------------------------------------------------------------


def test_points_isel_slices_every_leaf() -> None:
    p = _profile(6)
    part = p.isel(obs=[4, 1])
    np.testing.assert_array_equal(part.xy, p.xy[[4, 1]])
    np.testing.assert_array_equal(part.z, p.z[[4, 1]])
    np.testing.assert_array_equal(part.time, p.time[[4, 1]])
    np.testing.assert_array_equal(part.data, p.data[:, [4, 1]])
    assert p.isel(obs=slice(2, None)).size == 4
    assert p.sel(variable="psal").dims == ("obs",)
    with pytest.raises(IndexError, match=r"obs=\[k\]"):
        p.isel(obs=0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"xy": np.zeros((3, 3))}, "xy is"),
        ({"z": np.zeros(4)}, "vertical="),
        ({"vertical": DEPTH}, "no z"),
        ({"time": np.zeros(3, "datetime64[ns]")}, "time is"),
        ({"valid": np.ones(5, bool)}, "valid is"),
    ],
)
def test_points_validation(kwargs: dict, match: str) -> None:
    fields = {"xy": np.zeros((4, 2)), **kwargs}
    with pytest.raises(ValueError, match=match):
        GeoPoints(np.zeros(4), dims="obs", **fields)


def test_time_dim_and_time_leaf_together_raise() -> None:
    hours = TimeAxis(
        np.datetime64("2026-06-01", "ns") + np.arange(3) * np.timedelta64(1, "h")
    )
    stations = GeoPoints(
        np.zeros((3, 2, 4)),
        xy=np.zeros((4, 2)),
        dims="time variable obs",
        coords=(hours, ARGO),
    )
    assert stations.dims == ("time", "variable", "obs")
    with pytest.raises(ValueError, match="per-point time"):
        GeoPoints(
            np.zeros((3, 4)),
            xy=np.zeros((4, 2)),
            time=np.zeros(4, "datetime64[ns]"),
            dims="time obs",
        )


def test_pad_points() -> None:
    p = _profile(3)
    padded = pad_points(p, 5)
    assert padded.size == 5
    assert padded.valid.tolist() == [True, True, True, False, False]
    assert np.isfinite(padded.xy).all()
    assert np.isnan(padded.data[:, 3:]).all()
    np.testing.assert_array_equal(padded.isel(obs=slice(0, 3)).xy, p.xy)
    assert padded.valid_mask().tolist() == [True, True, True, False, False]
    with pytest.raises(ValueError, match="smaller"):
        pad_points(p, 2)
    ints = GeoPoints(np.ones(2, int), xy=np.zeros((2, 2)), dims="obs")
    with pytest.raises(ValueError, match="fill_value"):
        pad_points(ints, 4)


def test_points_adapter() -> None:
    p = _profile()
    assert isinstance(get_adapter(p), GeoPointsAdapter)
    mean = pa.reduce(p, "mean", "variable")
    assert isinstance(mean, GeoPoints) and mean.dims == ("obs",)
    np.testing.assert_array_equal(mean.z, p.z)
    assert mean.vertical == DEPTH
    profile_mean = pa.reduce(p, "mean", "obs")
    assert isinstance(profile_mean, DimArray) and profile_mean.dims == ("variable",)
    other = _profile(seed=1)
    with pytest.raises(GridMismatchError, match="xy"):
        geoarray.combine(np.subtract, p, other.with_data(p.data))


def test_same_n_new_positions_one_trace(x64: ModuleType) -> None:
    jax = x64
    traces = 0

    def cost(points: GeoPoints) -> Any:
        nonlocal traces
        traces += 1
        # a cost that reads the leaves: Σ x² over valid samples
        return (points.xy[:, 0] ** 2 * points.valid).sum()

    f = jax.jit(cost)
    for seed in range(3):
        f(pad_points(_profile(4, seed), 8).to_jax())
    assert traces == 1


def test_points_round_trip(x64: ModuleType) -> None:
    jax = x64
    p = pad_points(_profile(), 6).to_jax()
    assert str(p.time.dtype) == "int64"
    leaves, treedef = jax.tree_util.tree_flatten(p)
    assert len(leaves) == 5  # data, xy, z, time, valid
    back = jax.tree_util.tree_unflatten(treedef, leaves)
    assert back.identical(p)
    host = p.to_numpy()
    assert host.time.dtype == np.dtype("datetime64[ns]")
    assert host.identical(pad_points(_profile(), 6))
    assert "leaves=xy, z, time, valid" in repr(p)


def test_padding_leaves_a_masked_cost_unchanged(x64: ModuleType) -> None:
    """GA-G16 at the carrier level: J and ∇J do not depend on the pad length."""
    jax = x64
    jnp = x64.numpy
    p = _profile(5)

    def cost(values: Any, points: GeoPoints) -> Any:
        resid = jnp.where(points.valid, values - points.data[0], 0.0)
        return 0.5 * (resid**2).sum()

    results = []
    for n in (8, 16):
        padded = pad_points(p, n).to_jax()
        x = jnp.where(padded.valid, 1.0, jnp.nan)
        x = jnp.nan_to_num(x, nan=0.0)
        j, g = jax.value_and_grad(cost)(x, padded)
        results.append((float(j), np.asarray(g)[:5]))
    assert results[0][0] == pytest.approx(results[1][0], rel=1e-14)
    np.testing.assert_allclose(results[0][1], results[1][1], rtol=1e-14)
    assert np.isfinite(results[0][1]).all()


# -- review fixes (Codex, #548) -------------------------------------------------------


def test_swath_band_selection_moves_per_band_attrs() -> None:
    s = _swath().with_data(_swath().data, attrs={"band_names": ("a", "b", "c")})
    picked = s.isel(band=[2, 0])
    assert picked.attrs["band_names"] == ("c", "a")
    reduced = pa.reduce(s, "mean", "band")
    assert "band_names" not in reduced.attrs


def test_points_with_another_vertical_do_not_combine() -> None:
    depth = _profile()
    height = GeoPoints(
        depth.data,
        xy=depth.xy,
        z=depth.z,
        vertical=Vertical(kind="height", units="m", positive="up"),
        time=depth.time,
        dims=depth.dims,
        coords=depth.coords,
        fill_value=np.nan,
    )
    with pytest.raises(GridMismatchError, match="vertical"):
        geoarray.combine(np.subtract, depth, height)


def test_points_with_times_combine_across_backends(x64: ModuleType) -> None:
    """datetime64 times (NumPy) and int64 ns (JAX) are the same geolocation.

    Through the adapter's check: ``combine`` itself refuses mixed backends
    (pipekit's BackendError, decision GA-C1), but the check must not.
    """
    p = _profile()
    GeoPointsAdapter().check_compatible(p, p.to_jax())
    j = p.to_jax()
    out = geoarray.combine(x64.numpy.add, j, j)
    np.testing.assert_allclose(np.asarray(out.data), 2 * np.asarray(p.data))


def test_sel_rules_are_shared_with_geoarray() -> None:
    s = _swath()
    assert s.sel(band="b2").dims == ("row", "col")
    with pytest.raises(geoarray.LabelError, match="is a range"):
        s.sel(band=slice("b1", "b2"))
    with pytest.raises(geoarray.LabelError, match="not a label"):
        s.sel(band="b9")


def test_time_axis_stamps_are_leaves_on_swath_and_points(x64: ModuleType) -> None:
    stamps = TimeAxis(np.array(["2026-01-01", "2026-01-02"], dtype="datetime64[ns]"))
    lat, lon = np.meshgrid(np.arange(2.0), np.arange(3.0), indexing="ij")
    s = GeoSwath(
        np.zeros((2, 2, 3)), lat, lon, dims="time row col", coords=(stamps,)
    ).to_jax()
    later = s.with_data(s.data)
    object.__setattr__(
        later,
        "coords",
        (TimeAxis(np.array(["2027-01-01", "2027-01-02"], dtype="datetime64[ns]")),),
    )
    traces = []

    @x64.jit
    def first_stamp(c: GeoSwath) -> Any:
        traces.append(1)
        return c.coord("time").stamps[0]

    assert int(first_stamp(s)) == int(np.datetime64("2026-01-01", "ns").view(np.int64))
    first_stamp(later.to_jax())
    assert len(traces) == 1
    assert s.to_numpy().coord("time").stamps.dtype.kind == "M"
