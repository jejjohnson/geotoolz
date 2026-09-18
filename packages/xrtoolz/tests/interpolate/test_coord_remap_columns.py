"""Tests for per-column ``remap_axis`` (``source_coords`` / ``extrapolate``)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import xarray as xr

from xrtoolz.interpolate import remap_axis
from xrtoolz.interpolate.operators import RemapAxis, ToHeight
from xrtoolz.transforms._src import _coord_remap_kernels as ia


# ---------------------------------------------------------------------------
# Kernel — remap_axis_columns
# ---------------------------------------------------------------------------


@pytest.fixture
def field_1d_levels() -> tuple[np.ndarray, np.ndarray]:
    """(values, 1-D source levels) with the remap axis in the middle."""
    rng = np.random.default_rng(0)
    src = np.linspace(0.0, 10.0, 11)
    values = rng.standard_normal((3, src.size, 4))
    return values, src


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_matches_1d_kernel_exactly(field_1d_levels, method) -> None:
    """Invariant (i): a broadcast 1-D coordinate reproduces the old kernel."""
    values, src = field_1d_levels
    tgt = np.array([-1.0, 0.0, 0.3, 4.999, 5.0, 7.25, 10.0, 11.0, np.nan])
    expected = ia.remap_axis(
        values, axis=1, source_coords=src, target_coords=tgt, method=method
    )
    out = ia.remap_axis_columns(
        values,
        axis=1,
        source_coords=np.broadcast_to(src[None, :, None], values.shape),
        target_coords=tgt,
        method=method,
    )
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_matches_1d_kernel_descending(field_1d_levels, method) -> None:
    values, src = field_1d_levels
    values, src = values[:, ::-1], src[::-1]
    tgt = np.array([-0.5, 2.0, 5.0, 9.75, 10.5])
    expected = ia.remap_axis(
        values, axis=1, source_coords=src, target_coords=tgt, method=method
    )
    out = ia.remap_axis_columns(
        values,
        axis=1,
        source_coords=np.broadcast_to(src[None, :, None], values.shape),
        target_coords=tgt,
        method=method,
    )
    np.testing.assert_array_equal(out, expected)


def test_columns_kernel_recovers_linear_field_per_column() -> None:
    """Invariant (ii): a field linear in z is exact for any in-range target,
    even when columns have different level spacing (different brackets)."""
    n_lev, n_col = 6, 5
    rng = np.random.default_rng(1)
    # Each column: strictly increasing levels with its own random spacing.
    z = np.cumsum(rng.uniform(0.5, 2.0, size=(n_col, n_lev)), axis=1)
    z -= z[:, :1]  # every column starts at 0 but ends at a different height
    a = rng.standard_normal(n_col)
    b = rng.standard_normal(n_col)
    f = a[:, None] * z + b[:, None]
    tgt = np.array([0.0, 0.7, 1.9, 2.5])  # within every column's range
    out = ia.remap_axis_columns(f, axis=1, source_coords=z, target_coords=tgt)
    np.testing.assert_allclose(out, a[:, None] * tgt[None, :] + b[:, None], rtol=1e-12)
    # Different spacing means different bracketing: the level index bracketing
    # the same target differs between columns.
    hi = (z <= 1.9).sum(axis=1)
    assert len(set(hi.tolist())) > 1


def test_columns_kernel_nearest_extrapolation_holds_end_values() -> None:
    """Invariant (iii): out-of-range targets equal the column's end values."""
    z = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
    f = np.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    tgt = np.array([0.0, 12.0, 33.0, 100.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nearest"
    )
    # Column 0 range [10, 30]: 0 -> f0, 12 -> interp, 33 -> f[-1], 100 -> f[-1].
    np.testing.assert_allclose(out[0], [1.0, 1.2, 3.0, 3.0])
    # Column 1 range [15, 35]: 0 & 12 -> f0, 33 -> interp, 100 -> f[-1].
    np.testing.assert_allclose(out[1], [-1.0, -1.0, -2.8, -3.0])


def test_columns_kernel_nan_extrapolation_uses_each_columns_own_range() -> None:
    z = np.array([[10.0, 20.0, 30.0], [15.0, 25.0, 35.0]])
    f = np.ones_like(z)
    tgt = np.array([12.0, 33.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nan"
    )
    # 12 is inside column 0 but below column 1; 33 the other way round.
    assert out[0, 0] == 1.0 and np.isnan(out[1, 0])
    assert np.isnan(out[0, 1]) and out[1, 1] == 1.0


@pytest.mark.parametrize("method", ["linear", "nearest"])
def test_columns_kernel_nearest_method_extrapolation_modes(method) -> None:
    z = np.array([[0.0, 1.0, 2.0]])
    f = np.array([[10.0, 20.0, 30.0]])
    tgt = np.array([-5.0, 0.4, 1.6, 7.0])
    out_nan = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, method=method
    )
    out_near = ia.remap_axis_columns(
        f,
        axis=1,
        source_coords=z,
        target_coords=tgt,
        method=method,
        extrapolate="nearest",
    )
    assert np.isnan(out_nan[0, 0]) and np.isnan(out_nan[0, -1])
    assert out_near[0, 0] == 10.0 and out_near[0, -1] == 30.0
    np.testing.assert_array_equal(out_nan[0, 1:3], out_near[0, 1:3])


def test_columns_kernel_descending_and_mixed_columns() -> None:
    """Invariant (iv): a descending column equals its ascending flip, and
    ascending / descending columns may be mixed in one call."""
    rng = np.random.default_rng(2)
    z_asc = np.sort(rng.uniform(0.0, 10.0, size=(4, 7)), axis=1)
    f_asc = rng.standard_normal((4, 7))
    tgt = np.array([-1.0, 1.0, 4.5, 8.0, 12.0])
    ref = ia.remap_axis_columns(
        f_asc, axis=1, source_coords=z_asc, target_coords=tgt, extrapolate="nearest"
    )
    # All descending.
    out_desc = ia.remap_axis_columns(
        f_asc[:, ::-1],
        axis=1,
        source_coords=z_asc[:, ::-1],
        target_coords=tgt,
        extrapolate="nearest",
    )
    np.testing.assert_array_equal(out_desc, ref)
    # Mixed: flip odd columns only.
    z_mix = z_asc.copy()
    f_mix = f_asc.copy()
    z_mix[1::2] = z_asc[1::2, ::-1]
    f_mix[1::2] = f_asc[1::2, ::-1]
    out_mix = ia.remap_axis_columns(
        f_mix, axis=1, source_coords=z_mix, target_coords=tgt, extrapolate="nearest"
    )
    np.testing.assert_array_equal(out_mix, ref)


def test_columns_kernel_non_monotone_column_raises() -> None:
    z = np.array([[0.0, 1.0, 2.0], [0.0, 2.0, 1.0]])
    f = np.ones_like(z)
    with pytest.raises(ValueError, match="monotone"):
        ia.remap_axis_columns(f, axis=1, source_coords=z, target_coords=np.array([1.0]))


def test_columns_kernel_nan_target_and_bad_args() -> None:
    z = np.array([[0.0, 1.0, 2.0]])
    f = np.array([[1.0, 2.0, 3.0]])
    out = ia.remap_axis_columns(
        f,
        axis=1,
        source_coords=z,
        target_coords=np.array([np.nan, 1.0]),
        extrapolate="nearest",
    )
    assert np.isnan(out[0, 0]) and out[0, 1] == 2.0
    with pytest.raises(ValueError, match="shape"):
        ia.remap_axis_columns(
            f, axis=1, source_coords=z[:, :2], target_coords=np.array([1.0])
        )
    with pytest.raises(ValueError, match="extrapolate"):
        ia.remap_axis_columns(
            f,
            axis=1,
            source_coords=z,
            target_coords=np.array([1.0]),
            extrapolate="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="method"):
        ia.remap_axis_columns(
            f, axis=1, source_coords=z, target_coords=np.array([1.0]), method="bogus"
        )


def test_columns_kernel_preserves_complex_dtype() -> None:
    z = np.linspace(0.0, 1.0, 11)[None, :]
    f = z + 1j * (2.0 * z)
    tgt = np.array([0.25, 0.5, 2.0])
    out = ia.remap_axis_columns(
        f, axis=1, source_coords=z, target_coords=tgt, extrapolate="nearest"
    )
    assert np.iscomplexobj(out)
    np.testing.assert_allclose(out[0], [0.25 + 0.5j, 0.5 + 1.0j, 1.0 + 2.0j])


# ---------------------------------------------------------------------------
# Layer 0 — remap_axis wrapper
# ---------------------------------------------------------------------------


@pytest.fixture
def ds_columns() -> xr.Dataset:
    """Synthetic ``(time, level, y, x)`` model-level dataset with ``z_agl``.

    Levels sit 20 m apart; column ``(y=0, x=0)`` is shifted up by 15 m so
    its range differs from every other column. ``t`` is linear in height.
    """
    level = np.arange(5)
    z = 20.0 * (level + 1)
    z_agl = np.broadcast_to(z[None, :, None, None], (2, 5, 3, 4)).copy()
    z_agl[:, :, 0, 0] += 15.0
    t = 300.0 - 0.01 * z_agl
    return xr.Dataset(
        {
            "t": (("time", "level", "y", "x"), t),
            "psfc": (("time", "y", "x"), np.full((2, 3, 4), 1000.0)),
        },
        coords={
            "time": np.arange(2),
            "level": level,
            "z_agl": (("time", "level", "y", "x"), z_agl),
            "lat": (("y", "x"), np.arange(12, dtype=float).reshape(3, 4)),
        },
    )


def test_remap_axis_source_coords_by_name_replaces_dim(ds_columns) -> None:
    # 30 m is inside every regular column (20..100 m) but below the shifted
    # column (0, 0), whose levels start at 35 m.
    tgt = np.array([30.0, 50.0, 90.0])
    out = remap_axis(
        ds_columns["t"],
        source_dim="level",
        target_coords=tgt,
        target_name="height",
        source_coords="z_agl",
        extrapolate="nearest",
    )
    assert out.dims == ("time", "height", "y", "x")
    np.testing.assert_array_equal(out["height"].values, tgt)
    assert "level" not in out.coords and "z_agl" not in out.coords
    assert "lat" in out.coords  # coords not on the source dim survive
    # In-range targets are linear in z; 30 m in column (0, 0) holds its f_0.
    expected = np.broadcast_to([299.7, 299.5, 299.1], (2, 3))
    np.testing.assert_allclose(out.isel(y=1, x=1).values, expected)
    expected_shifted = np.broadcast_to([299.65, 299.5, 299.1], (2, 3))
    np.testing.assert_allclose(out.isel(y=0, x=0).values, expected_shifted)
    # "nan" leaves that column's out-of-range target as NaN, others intact.
    out_nan = remap_axis(
        ds_columns["t"],
        source_dim="level",
        target_coords=tgt,
        target_name="height",
        source_coords="z_agl",
    )
    assert np.isnan(out_nan.isel(y=0, x=0, height=0).values).all()
    assert np.isfinite(out_nan.isel(y=1, x=1).values).all()


def test_remap_axis_source_coords_as_dataarray_and_1d_extrapolate() -> None:
    z = np.linspace(0.0, 100.0, 11)
    da = xr.DataArray(
        2.0 * z[:, None] + np.zeros((11, 3)),
        dims=("depth", "x"),
        coords={"depth": z, "x": np.arange(3)},
        name="T",
        attrs={"units": "K"},
    )
    tgt = np.array([-10.0, 50.0, 150.0])
    # 1-D coordinate + nearest extrapolation: the per-column path with a
    # broadcast coordinate.
    out = remap_axis(da, source_dim="depth", target_coords=tgt, extrapolate="nearest")
    assert out.dims == ("depth", "x") and out.name == "T" and out.attrs == da.attrs
    np.testing.assert_allclose(out.values[:, 0], [0.0, 100.0, 200.0])
    # Explicit DataArray source coordinate that varies along x.
    z2 = xr.DataArray(
        z[:, None] + 10.0 * np.arange(3)[None, :], dims=("depth", "x"), name="z2"
    )
    out2 = remap_axis(
        da,
        source_dim="depth",
        target_coords=np.array([50.0]),
        source_coords=z2,
        extrapolate="nearest",
    )
    # Column x=k has z2 = z + 10k, so T(z2=50) = 2 * (50 - 10k).
    np.testing.assert_allclose(out2.values[0], [100.0, 80.0, 60.0])


def test_remap_axis_source_coords_validation(ds_columns) -> None:
    da = ds_columns["t"]
    with pytest.raises(ValueError, match="not a coordinate"):
        remap_axis(da, source_dim="level", target_coords=[1.0], source_coords="nope")
    with pytest.raises(ValueError, match="must carry source_dim"):
        remap_axis(da, source_dim="level", target_coords=[1.0], source_coords=da["lat"])
    with pytest.raises(ValueError, match="extrapolate"):
        remap_axis(
            da,
            source_dim="level",
            target_coords=[1.0],
            extrapolate="bogus",  # type: ignore[arg-type]
        )
    bad = xr.DataArray(np.ones((5, 2)), dims=("level", "other"))
    with pytest.raises(ValueError, match="not on the DataArray"):
        remap_axis(da, source_dim="level", target_coords=[1.0], source_coords=bad)
    # A non-monotone column surfaces the kernel error.
    z_bad = da["z_agl"].copy()
    z_bad[0, 1, 0, 0], z_bad[0, 2, 0, 0] = z_bad[0, 2, 0, 0], z_bad[0, 1, 0, 0]
    with pytest.raises(ValueError, match="monotone"):
        remap_axis(da, source_dim="level", target_coords=[50.0], source_coords=z_bad)


@pytest.mark.dask
def test_remap_axis_source_coords_dask_stays_lazy(ds_columns) -> None:
    pytest.importorskip("dask.array")
    tgt = np.array([10.0, 50.0, 90.0])
    kwargs = dict(
        source_dim="level",
        target_coords=tgt,
        target_name="height",
        source_coords="z_agl",
        extrapolate="nearest",
    )
    eager = remap_axis(ds_columns["t"], **kwargs)
    lazy = remap_axis(ds_columns["t"].chunk({"y": 1}), **kwargs)
    assert lazy.chunks is not None
    assert lazy.dims == eager.dims
    np.testing.assert_array_equal(lazy.compute().values, eager.values)


# ---------------------------------------------------------------------------
# Layer 1 — operators
# ---------------------------------------------------------------------------


def test_to_height_operator_per_column(ds_columns) -> None:
    tgt = np.array([30.0, 50.0, 90.0])
    op = ToHeight(tgt, source_coords="z_agl", extrapolate="nearest")
    out = op(ds_columns)
    assert "level" not in out.dims and out.sizes["height"] == 3
    assert out["t"].dims == ("time", "height", "y", "x")
    xr.testing.assert_identical(out["psfc"], ds_columns["psfc"])
    assert "z_agl" not in out.coords and "lat" in out.coords
    np.testing.assert_allclose(
        out["t"].isel(y=2, x=3).values, np.broadcast_to([299.7, 299.5, 299.1], (2, 3))
    )
    np.testing.assert_allclose(
        out["t"].isel(y=0, x=0).values, np.broadcast_to([299.65, 299.5, 299.1], (2, 3))
    )
    np.testing.assert_array_equal(out["height"].values, tgt)


def test_to_height_operator_get_config_round_trips(ds_columns) -> None:
    op = ToHeight(
        np.array([10.0, 50.0, 90.0]), source_coords="z_agl", extrapolate="nearest"
    )
    cfg = op.get_config()
    assert cfg == {
        "source_axis": "level",
        "target_axis": [10.0, 50.0, 90.0],
        "target_name": "height",
        "method": "linear",
        "source_coords": "z_agl",
        "extrapolate": "nearest",
    }
    assert json.loads(json.dumps(cfg)) == cfg
    rebuilt = RemapAxis(
        cfg["source_axis"],
        cfg["target_axis"],
        target_name=cfg["target_name"],
        method=cfg["method"],
        source_coords=cfg["source_coords"],
        extrapolate=cfg["extrapolate"],
    )
    assert rebuilt.get_config() == cfg
    xr.testing.assert_identical(rebuilt(ds_columns), op(ds_columns))


def test_remap_axis_operator_default_config_unchanged() -> None:
    """Defaults keep the pre-existing config shape (no new keys)."""
    cfg = RemapAxis("depth", [0.0, 1.0]).get_config()
    assert "source_coords" not in cfg and "extrapolate" not in cfg


def test_to_height_operator_datatree(ds_columns) -> None:
    tree = xr.DataTree.from_dict({"a": ds_columns, "b": ds_columns * 2.0})
    out = ToHeight(np.array([50.0]), source_coords="z_agl")(tree)
    assert isinstance(out, xr.DataTree)
    assert set(out.children) == {"a", "b"}
    np.testing.assert_allclose(
        out["b"].dataset["t"].values, 2.0 * out["a"].dataset["t"].values
    )
